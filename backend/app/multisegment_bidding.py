"""Independent monotone demand curves on a finite price lattice.

Integer segment widths and price-level activation are optimized jointly across
24 hours, with the same paired paths and absolute daily CVaR budget as v2.
The customer template is a serialization contract, never a trading rule source.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .independent_bidding import finite, weighted_cvar
from .shandong_settlement import contract_book, contract_scenario_costs, EVIDENCE, EXCLUSIONS

VERSION = 'independent-multisegment-settlement-budget-v2'
EXPORTABLE_VERSIONS = {VERSION, 'independent-multisegment-daily-budget-v1'}
OFFICIAL_RULE_EVIDENCE = dict(
    title='山东电力市场规则（试行）（2026年4月修订版）',
    url='https://sdb.nea.gov.cn/dtyw/tzgg/202605/P020260508675725961073.pdf',
    sha256='0745e827bcc147780b368a76f593306feeacb04ac066ff3de48f53ecb4c252f2',
    document_status='OFFICIAL_REVISION', published_at='2026-05-08', effective_from='2026-04-30',
    clauses=['7.2.12','7.2.13','7.2.15','7.3.13','7.3.14','3.2.11','3.2.13','3.2.14'],
    pdf_pages=[114,115,143,42,43], verified_at='2026-09-19',
    max_segments=5, min_segment_mw=1., declaration_deadline='D-1 15:00 Asia/Shanghai',
    endpoint_basis='MAXIMUM_DEMAND_NOT_RESIDUAL_EXPOSURE',
    price_limits_status='SEPARATE_EFFECTIVE_PARAMETER_NOTICE_REQUIRED',
    newer_amendments_status='NOT_EXHAUSTIVELY_VERIFIED',
)
RULE_READINESS = dict(
    status='REAL_PARTIAL',
    verified_constraints=[
        'RETAILER_DAY_AHEAD_DECLARATION_HOUR_24',
        'MAX_FIVE_SEGMENTS_PER_HOUR',
        'FIRST_SEGMENT_STARTS_AT_ZERO',
        'CONTIGUOUS_SEGMENTS',
        'MINIMUM_SEGMENT_WIDTH_1_MW',
        'BUY_PRICE_MONOTONE_NONINCREASING',
        'CURVE_ENDPOINT_EQUALS_MAXIMUM_DEMAND',
        'MAXIMUM_DEMAND_NOT_ABOVE_REGISTERED_CAPACITY',
        'DECLARATION_DEADLINE_D_MINUS_1_1500_ASIA_SHANGHAI',
        'USER_SETTLEMENT_RT_FULL_ENERGY_PLUS_DA_AND_CONTRACT_DIFFERENCES',
    ],
    pending_parameters=[
        'EFFECTIVE_BID_PRICE_FLOOR',
        'EFFECTIVE_BID_PRICE_CEILING',
        'PRICE_TICK',
        'POWER_TICK',
        'EQUAL_PRICE_CLEARING_RULE',
        'SECONDARY_PRICE_CAP_STATUS',
        'NODE_MAPPING_AND_LOAD_ALLOCATION',
        'TERMINAL_IMPORT_SCHEMA',
    ],
    time_basis=dict(
        retailer_declaration='HOUR_24',
        reliability_and_system_operation='QUARTER_HOUR_96',
        current_strategy_input='HOUR_24',
        quarter_hour_upgrade='BLOCKED_UNTIL_PREMARKET_96_POINT_INPUTS_ARE_AVAILABLE',
    ),
)
HEADERS = ['时间段', '序号', '类型', '起始功率(spower)', '结束功率(epower)', '费用(cost)']
DEFAULT_RULES = dict(
    rule_version='sd-customer-template-research-v1', market_code='SD',
    max_segments=5, interval_hours=1., price_tick=.01, power_tick_mw=.01,
    min_segment_mw=.01, price_floor=None, price_ceiling=None,
    price_tick_origin=0., type_code='用电', time_label='HOUR_END',
    equal_price_fill_fraction=0., effective_from=None, effective_to=None,
    policy_citation=None, confirmed_by=None,
)


def decimal_units(value, step, rounding=ROUND_FLOOR):
    return int((Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=rounding))


def validate_rules(overrides, business_date):
    if overrides is not None and (not isinstance(overrides, dict) or set(overrides)-set(DEFAULT_RULES)):
        raise ValueError('Unknown market rule fields')
    rules = {**DEFAULT_RULES, **(overrides or {})}
    if rules['market_code'] != 'SD':
        raise ValueError('Only SD retail demand curves supported')
    if type(rules['max_segments']) is not int or not 1 <= rules['max_segments'] <= 5:
        raise ValueError('Article 7.2.13 permits at most 5 active segments')
    if rules['interval_hours'] != 1. or type(rules['interval_hours']) is bool:
        raise ValueError('Only 24 one-hour periods supported; do not silently convert 96 periods')
    for key in ('price_tick', 'power_tick_mw', 'min_segment_mw'):
        if not finite(rules[key]) or rules[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not finite(rules['price_tick_origin']):
        raise ValueError('Finite price tick origin required')
    if rules['type_code'] != '用电' or rules['time_label'] != 'HOUR_END':
        raise ValueError('Only observed customer type/time labels supported')
    if not finite(rules['equal_price_fill_fraction']) or not 0 <= rules['equal_price_fill_fraction'] <= 1:
        raise ValueError('Equal-price fill must be in [0,1]')
    for key in ('price_floor', 'price_ceiling'):
        if rules[key] is not None and not finite(rules[key]):
            raise ValueError('Finite price limits or null required')
    floor, ceiling = rules['price_floor'], rules['price_ceiling']
    if floor is not None and ceiling is not None and floor > ceiling:
        raise ValueError('Reversed price limits')
    for key in ('rule_version',):
        if not isinstance(rules[key], str) or not rules[key].strip():
            raise ValueError('Nonempty rule version required')
    for key in ('policy_citation', 'confirmed_by'):
        if rules[key] is not None and (not isinstance(rules[key], str) or not rules[key].strip()):
            raise ValueError(f'Nonempty {key} or null required')
    for key in ('effective_from', 'effective_to'):
        if rules[key] is not None:
            date.fromisoformat(rules[key])
    if rules['effective_from'] and rules['effective_to'] and rules['effective_from'] > rules['effective_to']:
        raise ValueError('Reversed rule effective dates')
    if rules['effective_from'] and business_date < rules['effective_from']:
        raise ValueError('Rule not effective on target date')
    if rules['effective_to'] and business_date > rules['effective_to']:
        raise ValueError('Rule expired on target date')
    return rules


def price_lattice(row, banks, rules, require_full_curve=False):
    # Price choices use only forecasts and pre-target scenario paths.
    rt, da = row['forecast_rt'], row['forecast_da']
    values = [rt-50, rt-20, rt, rt+50, rt+100, max(da, rt)+200]
    if rules['price_floor'] is not None: values.append(rules['price_floor'])
    for bank in banks.values():
        prices = bank['clearing_da'][:, row['period']-1]
        order = np.argsort(prices)
        cumulative = np.cumsum(bank['weights'][order])
        for q in (.1, .5, .9):
            values.append(float(prices[order[min(np.searchsorted(cumulative, q), len(order)-1)]]))
    origin, step = Decimal(str(rules['price_tick_origin'])), Decimal(str(rules['price_tick']))
    lower = None if rules['price_floor'] is None else int(((Decimal(str(rules['price_floor']))-origin)/step).to_integral_value(rounding=ROUND_CEILING))
    upper = None if rules['price_ceiling'] is None else int(((Decimal(str(rules['price_ceiling']))-origin)/step).to_integral_value(rounding=ROUND_FLOOR))
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('No price tick inside configured limits')
    ticks = set()
    for value in values:
        tick = int(((Decimal(str(value))-origin)/step).to_integral_value(rounding=ROUND_HALF_UP))
        if lower is not None: tick = max(tick, lower)
        if upper is not None: tick = min(tick, upper)
        ticks.add(tick)
    # Under the current price-taking uniform-clearing assumption, prices with
    # identical fills in every bank have identical objective/constraint columns.
    # Keep the lowest such bid to remove MILP symmetry; no scenario is discarded.
    unique = {}
    for price in sorted(float(origin+t*step) for t in ticks):
        signature = []
        for bank in banks.values():
            delta = price-bank['clearing_da'][:,row['period']-1]
            signature.extend(np.where(np.abs(delta)<=1e-9, rules['equal_price_fill_fraction'], (delta>0).astype(float)).tolist())
        if not any(signature) and not require_full_curve: continue
        unique.setdefault(tuple(signature),price)
    return sorted(unique.values(), reverse=True)


def optimize_multisegment_bids(*, business_date, records, scenario_banks,
        exposure_basis='UNIT_RESEARCH', rule_profile=None, risk_aversion=.3,
        risk_budget_tightening=.5, absolute_cost_budget_yuan=None,
        fixed_procurement_cost_yuan=None, time_limit_seconds=60.,
        quantity_basis='RESIDUAL_RESEARCH', registered_capacity_mw=None,
        contract_settlement=None, budget_scope='SPOT_COMPONENTS', clearing_context=None):
    cutoff = date.fromisoformat(business_date)-timedelta(days=2)
    rules = validate_rules(rule_profile, business_date)
    if quantity_basis not in ('RESIDUAL_RESEARCH','TOTAL_DEMAND_CURVE'):
        raise ValueError('Explicit quantity basis required')
    require_full_curve = quantity_basis == 'TOTAL_DEMAND_CURVE'
    if budget_scope not in ('SPOT_COMPONENTS', 'ENERGY_WITH_CONTRACT_DIFFERENCE'):
        raise ValueError('Unknown budget scope')
    if not require_full_curve and (contract_settlement is not None or budget_scope != 'SPOT_COMPONENTS'):
        raise ValueError('Contract settlement requires total-load costing, not residual exposure')
    if require_full_curve and fixed_procurement_cost_yuan is not None:
        raise ValueError('Use hourly contract difference settlement; a fixed procurement cost cannot replace it')
    contracts = contract_book(contract_settlement, business_date)
    if require_full_curve:
        if business_date < '2026-04-30': raise ValueError('Verified 2026-04 rule cannot be applied to earlier dates')
        if not finite(registered_capacity_mw) or registered_capacity_mw <= 0:
            raise ValueError('Positive registered capacity (MW) required for total-demand curves')
        if exposure_basis != 'PROVIDED_RESEARCH': raise ValueError('Provide 24-hour maximum demand; unit exposure is not customer demand')
        if rule_profile and 'min_segment_mw' in rule_profile and rule_profile['min_segment_mw'] < 1:
            raise ValueError('Article 7.2.13 requires minimum segment width 1 MW')
        rules['min_segment_mw'] = max(1., rules['min_segment_mw'])
    elif registered_capacity_mw is not None:
        raise ValueError('Registered capacity belongs to total-demand mode, not residual research')
    for value, label in [(risk_aversion, 'risk aversion'), (risk_budget_tightening, 'budget tightening')]:
        if value is None and label == 'budget tightening': continue
        if not finite(value) or not 0 <= value <= 1: raise ValueError(f'Invalid {label}')
    for value in (absolute_cost_budget_yuan, fixed_procurement_cost_yuan):
        if value is not None and not finite(value): raise ValueError('Finite cost/budget required')
    if absolute_cost_budget_yuan is not None and risk_budget_tightening is not None:
        raise ValueError('Choose explicit budget or tightening, not both')
    if not finite(time_limit_seconds) or not 0 < time_limit_seconds <= 120:
        raise ValueError('Solver time limit must be in (0,120]')
    if exposure_basis not in ('UNIT_RESEARCH', 'PROVIDED_RESEARCH'):
        raise ValueError('Explicit exposure basis required')
    if not isinstance(records, list) or len(records) != 24 or any(not isinstance(r, dict) for r in records):
        raise ValueError('Exactly 24 input records required')
    allowed_fields={'period','forecast_da','forecast_rt','exposure_mwh'} | ({'load_forecast_mwh'} if require_full_curve else set())
    if any(set(r)-allowed_fields for r in records):
        raise ValueError('Actual outcomes and customer bids are not accepted')
    if any(type(r.get('period')) is not int for r in records) or {r['period'] for r in records} != set(range(1,25)):
        raise ValueError('Unique periods 1-24 required')
    rows = sorted(records, key=lambda r:r['period'])
    budget_enabled = risk_budget_tightening is not None or absolute_cost_budget_yuan is not None
    base = dict(version=VERSION, business_date=business_date, source_cutoff=cutoff.isoformat(),
        market_code='SD', trading_subject='retail', granularity='HOUR_24',
        execution_allowed=False, formal_action='HOLD', formal_gate='BLOCKED',
        formal_blockers=['OFFICIAL_PARAMETERS_NOT_FULLY_VERIFIED', 'EXPOSURE_SETTLEMENT_AND_REVIEW_NOT_CERTIFIED'],
        customer_bids_used=False, exposure_basis=exposure_basis, rule_profile=rules,
        quantity_basis=quantity_basis, registered_capacity_mw=registered_capacity_mw,
        official_rule_evidence=OFFICIAL_RULE_EVIDENCE, rule_readiness=RULE_READINESS,
        quantity_interpretation='用户提供的每小时最大用电需求；曲线终点固定等于该需求' if require_full_curve else '剩余敞口方法研究；不是第7.2.13条规定的总需求申报口径',
        rule_evidence_status='OFFICIAL_STRUCTURE_VERIFIED_OPERATIONAL_PARAMETERS_PARTIAL',
        rule_override_status='USER_PROVIDED_PENDING_VERIFICATION' if rules['policy_citation'] else 'RESEARCH_DEFAULTS_PENDING_CONFIRMATION',
        template=dict(sheet='数据', headers=HEADERS, slots_per_hour=5, time_label='01:00..24:00',
            evidence='Customer July workbook: 24 hours x 5 slots; not proof of market rules'),
        risk_aversion=risk_aversion, confidence=.95, objective_basis='DAILY_RELATIVE_COST_VS_FULL_RT',
        fixed_procurement_cost_yuan=fixed_procurement_cost_yuan,
        absolute_cost_scope=('GROSS_SPOT_COST_EXCLUDING_CONTRACT_DIFFERENCE_SETTLEMENT' if fixed_procurement_cost_yuan is None else 'GROSS_SPOT_PLUS_PROVIDED_FIXED_COST') if require_full_curve else ('RESIDUAL_EXPOSURE_ONLY' if fixed_procurement_cost_yuan is None else 'RESIDUAL_PLUS_PROVIDED_LOCKED_COST'),
        fee_basis='EXCLUDED_PENDING_ACTUAL_RULES',
        optimization_scope='GLOBAL_MILP_ON_FORECAST_DERIVED_FINITE_PRICE_LATTICE',
        risk_budget=dict(enabled=budget_enabled, tightening=risk_budget_tightening,
            requested_budget_yuan=absolute_cost_budget_yuan, confidence=.95, unit='CNY_PER_DAY', scope=budget_scope),
        settlement_audit=dict(evidence=EVIDENCE, contract_status=contracts['status'],
            contract_data_complete=contracts['complete'], contract_source=contracts.get('source'),
            contract_data_version=contracts.get('data_version'), exclusions=EXCLUSIONS,
            full_settlement_bill=False, price_basis_required='SD_USER_UNIFIED_SETTLEMENT_POINT',
            formula='sum(load*RT + cleared_DA*(DA-RT) + contract_MWh*(contract_price-reference_price))',
            budget_scope=budget_scope, missing=contracts['missing'],
            load_uncertainty='FIXED_LOAD_FORECAST_NOT_STOCHASTIC_LOAD_PATHS'),
        clearing_audit=dict(basis='UNIFIED_PRICE_PROXY_RESEARCH',
            approximation=True, actual_sced_reproduced=False,
            missing=['NODE_LOAD_ALLOCATION', 'PAIRED_NODE_CLEARING_PRICE_SCENARIOS'],
            note='统一结算点价格不是节点出清价格；默认仅用其近似判断成交。',
            quote_limits_distinct_from_clearing_limits=True,
            secondary_price_cap_status='EFFECTIVE_NOTICE_AND_TRIGGER_DATA_MISSING'),
        display_title='多段报价人工复核申报草稿 · 不可自动提交')
    def blocked(reason, **extra):
        return {**base, 'status':'BLOCKED', 'reason':reason, 'records':[], **extra}
    if budget_scope == 'ENERGY_WITH_CONTRACT_DIFFERENCE':
        if not contracts['complete']:
            return blocked('CONTRACT_DATA_REQUIRED_FOR_ENERGY_BUDGET')
        base['absolute_cost_scope'] = 'ENERGY_WITH_CONTRACT_DIFFERENCE_EXCLUDING_FEES_AND_MONTH_END'
    if any(not finite(r.get(k)) for r in rows for k in ('forecast_da','forecast_rt','exposure_mwh')):
        return blocked('MISSING_FORECAST_OR_EXPOSURE')
    if any(r['exposure_mwh'] < 0 for r in rows): raise ValueError('Negative exposure requires separate review')
    if require_full_curve and any(not finite(r.get('load_forecast_mwh')) or not 0<=r['load_forecast_mwh']<=r['exposure_mwh'] for r in rows):
        raise ValueError('Provide finite nonnegative load forecasts not exceeding maximum demand; do not substitute contract residuals')
    if require_full_curve and any(r['exposure_mwh']/rules['interval_hours'] > registered_capacity_mw+1e-9 for r in rows):
        raise ValueError('Maximum demand cannot exceed registered capacity')
    if exposure_basis == 'UNIT_RESEARCH' and any(r['exposure_mwh'] != 1 for r in rows):
        raise ValueError('Unit research means 1 MWh per hour')
    exposure = np.array([r['load_forecast_mwh'] if require_full_curve else r['exposure_mwh'] for r in rows])
    total = float(exposure.sum())
    if total <= 0: return blocked('NO_POSITIVE_EXPOSURE')
    if not isinstance(scenario_banks, dict) or not scenario_banks: return blocked('SCENARIOS_MISSING')
    banks = {}
    for name, paths in scenario_banks.items():
        if not isinstance(paths, list): raise ValueError('Scenario paths list required')
        seen = set()
        for p in paths:
            if not isinstance(p, dict): raise ValueError('Scenario object required')
            stamp = p.get('source_date')
            if date.fromisoformat(stamp) > cutoff or stamp in seen: raise ValueError('Unique D-2 paths required')
            seen.add(stamp)
            if not finite(p.get('weight')) or p['weight'] <= 0: raise ValueError('Positive scenario weight required')
            for key in ('day_ahead','real_time'):
                if not isinstance(p.get(key), list) or len(p[key]) != 24 or any(not finite(v) for v in p[key]):
                    raise ValueError('24 paired finite prices required')
        if len(paths) < 10: return blocked('INSUFFICIENT_HISTORY', bank=name)
        w = np.array([p['weight'] for p in paths]); w = w/w.sum()
        ess = float(1/(w@w))
        if ess < 8: return blocked('INSUFFICIENT_EFFECTIVE_HISTORY', bank=name)
        banks[name] = dict(weights=w, ess=ess, source_dates=[p['source_date'] for p in paths],
            da=np.array([p['day_ahead'] for p in paths]), rt=np.array([p['real_time'] for p in paths]),
            clearing_da=np.array([p['day_ahead'] for p in paths]))
    if clearing_context is not None:
        fields={'mode','node_id','source','data_version','business_date','scenario_banks'}
        if not require_full_curve or not isinstance(clearing_context,dict) or set(clearing_context)!=fields:
            raise ValueError('Single-node clearing context requires complete source fields in total-demand mode')
        if clearing_context['mode']!='SINGLE_NODE_SCENARIOS' or clearing_context['business_date']!=business_date:
            raise ValueError('Single-node clearing scenarios must match delivery date')
        for key in ('node_id','source','data_version'):
            if not isinstance(clearing_context[key],str) or not clearing_context[key].strip():
                raise ValueError(f'Clearing {key} required')
        node_banks=clearing_context['scenario_banks']
        if not isinstance(node_banks,dict) or set(node_banks)!=set(banks):
            raise ValueError('Node price scenarios must match all settlement scenario banks')
        for name,b in banks.items():
            paths=node_banks[name]
            if not isinstance(paths,list) or any(not isinstance(p,dict) or set(p)!={'source_date','day_ahead_node'} for p in paths):
                raise ValueError('Node paths require source date and hourly node prices only')
            lookup={p['source_date']:p['day_ahead_node'] for p in paths}
            if len(lookup)!=len(paths) or set(lookup)!=set(b['source_dates']):
                raise ValueError('Node path source dates must match paired scenario dates exactly')
            for prices in lookup.values():
                if not isinstance(prices,list) or len(prices)!=24 or any(not finite(v) for v in prices):
                    raise ValueError('24 finite node prices per paired path required')
            b['clearing_da']=np.array([lookup[d] for d in b['source_dates']])
        base['clearing_audit'].update(basis='PROVIDED_SINGLE_NODE_PRICE_TAKING_SCENARIOS',missing=[],
            node_id=clearing_context['node_id'],source=clearing_context['source'],data_version=clearing_context['data_version'],
            note='仅适用于该单节点的需求与容量。以节点价格模拟成交，统一结算价计算电费；仍不是SCUC/SCED出清。')
    quantum = rules['power_tick_mw'] * rules['interval_hours']
    caps = [decimal_units(r['exposure_mwh'], quantum) for r in rows]
    if require_full_curve and any(abs(r['exposure_mwh']-float(Decimal(cap)*Decimal(str(quantum))))>1e-8 for r,cap in zip(rows,caps)):
        raise ValueError('Total-demand endpoint must align with configured power tick; do not silently round demand')
    if max(caps) > 10**7: raise ValueError('Power tick too fine for provided exposure')
    minimum = max(1, decimal_units(rules['min_segment_mw'], rules['power_tick_mw'], ROUND_CEILING))
    if require_full_curve and any(0 < cap < minimum for cap in caps):
        return blocked('DEMAND_BELOW_MINIMUM_SEGMENT_REQUIRES_RULE_REVIEW')
    levels, hours = [], []
    for row, cap in zip(rows, caps):
        ids = []
        for price in price_lattice(row, banks, rules, require_full_curve):
            ids.append(len(levels)); levels.append(dict(period=row['period'],price=price,cap=cap))
        hours.append(ids)
    m = len(levels)
    if not m:
        # Retain one zero-fill column to represent a valid no-bid solution.
        levels=[dict(period=1,price=rows[0]['forecast_rt'],cap=0)]
        hours[0]=[0];m=1
    for b in banks.values():
        da = np.array([b['da'][:,v['period']-1] for v in levels]).T
        rt = np.array([b['rt'][:,v['period']-1] for v in levels]).T
        node_da = np.array([b['clearing_da'][:,v['period']-1] for v in levels]).T
        delta = np.array([v['price'] for v in levels])-node_da
        fill = np.where(np.abs(delta) <= 1e-9, rules['equal_price_fill_fraction'], (delta>0).astype(float))
        b['accepted'] = fill*quantum
        b['loss'] = b['accepted']*(da-rt)/total
        b['contract_cost'] = contract_scenario_costs(contracts, b['da'], b['rt'])
        b['spot_baseline'] = b['rt']@exposure
        b['rt_cost'] = (b['spot_baseline']+(fixed_procurement_cost_yuan or 0.))/total
        if budget_scope == 'ENERGY_WITH_CONTRACT_DIFFERENCE':
            b['rt_cost'] += b['contract_cost']/total
    extra = sum(1+len(b['weights']) for b in banks.values())
    z = 2*m+extra*(2 if budget_enabled else 1)
    absolute_z = z+1 if budget_enabled else None
    size = z+1+int(budget_enabled)
    low = np.zeros(size); high = np.full(size, np.inf)
    high[:m] = [v['cap'] for v in levels]; high[m:2*m] = 1; low[z] = -np.inf
    if budget_enabled: low[absolute_z] = -np.inf
    A, lb, ub = [], [], []
    def constraint(entries, lower=-np.inf, upper=0.):
        a = np.zeros(size)
        for ix, val in entries.items(): a[ix] = val
        A.append(a); lb.append(lower); ub.append(upper)
    for j, level in enumerate(levels):
        constraint({j:1, m+j:-level['cap']})
        constraint({j:-1, m+j:minimum})
    for ids, cap in zip(hours, caps):
        constraint({j:1 for j in ids}, lower=cap if require_full_curve else -np.inf, upper=cap)
        constraint({m+j:1 for j in ids}, upper=rules['max_segments'])
    cursor = 2*m
    for b in banks.values():
        n = len(b['weights']); eta = cursor; slack = cursor+1; cursor += n+1; low[eta] = -np.inf
        for s in range(n):
            a = {j:v for j,v in enumerate(b['loss'][s])}; a.update({eta:-1, slack+s:-1}); constraint(a)
        a = {j:v for j,v in enumerate((1-risk_aversion)*(b['weights']@b['loss']))}
        a.update({eta:risk_aversion, z:-1, **{slack+s:risk_aversion*w/.05 for s,w in enumerate(b['weights'])}})
        constraint(a)
    if budget_enabled:
        for b in banks.values():
            n = len(b['weights']); eta = cursor; slack = cursor+1; cursor += n+1; low[eta] = -np.inf
            for s in range(n):
                a = {j:v for j,v in enumerate(b['loss'][s])}; a.update({eta:-1, slack+s:-1})
                constraint(a, upper=-b['rt_cost'][s])
            constraint({eta:1, absolute_z:-1, **{slack+s:w/.05 for s,w in enumerate(b['weights'])}})
    A, lb, ub = np.array(A), np.array(lb), np.array(ub)
    integrality = np.zeros(size); integrality[:2*m] = 1
    options = dict(time_limit=time_limit_seconds, mip_rel_gap=1e-6, threads=1)
    def solve(objective, bounds_high, relaxed=False):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected')
            return milp(objective, integrality=np.zeros(size) if relaxed else integrality, bounds=Bounds(low,bounds_high),
                constraints=LinearConstraint(A,lb,ub), options=options)
    rt_tail = max(weighted_cvar(b['rt_cost'], b['weights']) for b in banks.values())*total
    budget, minimum_tail, relaxation_bound = absolute_cost_budget_yuan, None, None
    if risk_budget_tightening is not None:
        objective = np.zeros(size); objective[absolute_z] = 1
        # Fixed total-demand endpoints make the integer minimum-tail frontier
        # expensive. A continuous relaxation supplies a certified LOWER bound;
        # interpolating toward it tightens (never weakens) the research budget.
        # It is not called attainable. The final curve remains fully integer and
        # must meet the actual cap; infeasibility still blocks rather than relaxes.
        result = solve(objective, high, relaxed=require_full_curve)
        if result.status != 0: return blocked('RISK_BUDGET_FEASIBILITY_NOT_PROVEN', solver_status=int(result.status))
        frontier_value = float(result.fun*total)
        if require_full_curve: relaxation_bound = frontier_value
        else: minimum_tail = frontier_value
        budget = rt_tail-risk_budget_tightening*max(0,rt_tail-frontier_value)
    if budget_enabled:
        high[absolute_z] = budget/total+1e-8
        base['risk_budget'].update(limit_yuan=budget, full_rt_cvar95_yuan=rt_tail,
            minimum_attainable_cvar95_yuan=minimum_tail,
            relaxation_lower_bound_cvar95_yuan=relaxation_bound,
            source='EXPLICIT_BUSINESS_CAP' if risk_budget_tightening is None else 'RESEARCH_RELAXATION_LOWER_BOUND' if require_full_curve else 'RESEARCH_SCENARIO_FRONTIER',
            guarantee='SCENARIO_CONSTRAINT_ONLY_NOT_REALIZED_COST_GUARANTEE')
    objective = np.zeros(size); objective[z] = 1
    primary = solve(objective, high)
    if primary.status != 0:
        return blocked('ABSOLUTE_COST_BUDGET_INFEASIBLE' if primary.status == 2 and budget_enabled else 'OPTIMIZATION_NOT_PROVEN', solver_status=int(primary.status))
    # Tie-break only; it cannot sacrifice the primary risk-cost objective.
    tied_high = high.copy(); tied_high[z] = primary.fun+1e-7
    tie = np.zeros(size); tie[m:2*m] = 1
    tie[:m] = quantum/(2*max(total,sum(r['exposure_mwh'] for r in rows)))
    secondary = solve(tie, tied_high)
    chosen = secondary if secondary.status == 0 else primary
    ticks = np.rint(chosen.x[:m]).astype(int)
    if np.max(np.abs(ticks-chosen.x[:m])) > 1e-5: return blocked('INVALID_INTEGER_SOLUTION')
    diagnostics = {}
    for name,b in banks.items():
        loss = b['loss']@ticks
        mean = float(b['weights']@loss); tail = weighted_cvar(loss,b['weights'])
        costs = (b['rt_cost']+loss)*total
        spot_costs = b['spot_baseline']+loss*total
        energy_costs = None if b['contract_cost'] is None else spot_costs+b['contract_cost']
        diagnostics[name] = dict(mean_relative_cost=mean, cvar95_relative_cost=tail,
            risk_score=(1-risk_aversion)*mean+risk_aversion*tail,
            mean_absolute_cost_yuan=float(b['weights']@costs), cvar95_absolute_cost_yuan=weighted_cvar(costs,b['weights']),
            daily_absolute_costs_yuan=costs.tolist(), scenario_weights=b['weights'].tolist(),
            source_dates=b['source_dates'], effective_sample_size=b['ess'],
            daily_spot_component_costs_yuan=spot_costs.tolist(),
            daily_contract_difference_costs_yuan=None if b['contract_cost'] is None else b['contract_cost'].tolist(),
            daily_energy_component_costs_yuan=None if energy_costs is None else energy_costs.tolist(),
            cvar95_energy_components_yuan=None if energy_costs is None else weighted_cvar(energy_costs,b['weights']))
    score = max(b['risk_score'] for b in diagnostics.values())
    if abs(score-primary.fun) > 2e-5: return blocked('OBJECTIVE_RECONCILIATION_FAILED')
    if budget_enabled:
        used = max(b['cvar95_absolute_cost_yuan'] for b in diagnostics.values())
        if used > budget+max(1e-5,total*2e-5): return blocked('RISK_BUDGET_RECONCILIATION_FAILED')
        base['risk_budget'].update(used_cvar95_yuan=used, slack_yuan=budget-used, passed=True)
    output = []
    for row, ids, cap in zip(rows,hours,caps):
        segments, cumulative = [], 0
        for j in ids:  # Descending price, so positive widths form a monotone buy curve.
            if ticks[j] == 0: continue
            start = float(Decimal(cumulative)*Decimal(str(rules['power_tick_mw'])))
            cumulative += int(ticks[j])
            end = float(Decimal(cumulative)*Decimal(str(rules['power_tick_mw'])))
            segments.append(dict(sequence=len(segments)+1, start_power_mw=start, end_power_mw=end,
                quantity_mwh=float(Decimal(int(ticks[j]))*Decimal(str(quantum))),
                bid_price_yuan_per_mwh=levels[j]['price']))
        if cumulative > cap or (require_full_curve and cumulative != cap) or len(segments) > rules['max_segments'] or any(ticks[j] and ticks[j]<minimum for j in ids):
            return blocked('CURVE_CONSTRAINT_RECONCILIATION_FAILED')
        quantity = float(Decimal(cumulative)*Decimal(str(quantum)))
        flags = []
        if any(any(abs(s['bid_price_yuan_per_mwh']-limit)<1e-9 for limit in
                   (rules['price_floor'],rules['price_ceiling']) if limit is not None) for s in segments):
            flags.append('LIMIT_PRICE_JUSTIFICATION_REVIEW')
        if any(np.max(b['accepted'][:,ids]@ticks[ids])>exposure[row['period']-1]+1e-8 for b in banks.values()):
            flags.append('SCENARIO_DA_ABOVE_LOAD_REVIEW')
        output.append(dict(**row, maximum_demand_mw=row['exposure_mwh']/rules['interval_hours'] if require_full_curve else None,
            segments=segments, segment_count=len(segments),
            max_bid_quantity_mwh=quantity, unsubmitted_exposure_mwh=row['exposure_mwh']-quantity,
            rounding_residual_mwh=row['exposure_mwh']-float(Decimal(cap)*Decimal(str(quantum))),
            expected_accepted_by_bank={name:float(b['weights']@(b['accepted'][:,ids]@ticks[ids])) for name,b in banks.items()},
            expected_realtime_balance_by_bank={name:float(exposure[row['period']-1]-b['weights']@(b['accepted'][:,ids]@ticks[ids])) for name,b in banks.items()},
            risk_flags=flags,
            price_basis='FORECAST_DERIVED_CANDIDATES_WITH_PAIRED_DAILY_SCENARIOS',
            research_action='BID_RESEARCH' if segments else 'RESERVE_REALTIME', formal_action='HOLD', execution_allowed=False))
    return {**base, 'status':'RESEARCH_ONLY', 'records':output, 'total_exposure_mwh':total,
        'scenario_diagnostics':diagnostics, 'selected_score':score,
        'cost_load_basis':'PROVIDED_LOAD_FORECAST' if require_full_curve else 'RESIDUAL_EXPOSURE',
        'cost_reference_energy_mwh':total,
        'market_shape_checks':{
            'max_five_segments':'PASS', 'continuous_monotone_curve':'PASS',
            'min_one_mw_segment':'PASS' if all(s['end_power_mw']-s['start_power_mw']>=1-1e-9 for r in output for s in r['segments']) else 'FAIL_RESEARCH_ONLY',
            'maximum_demand_endpoint':'PASS' if require_full_curve else 'UNCONFIRMED_RESIDUAL_IS_NOT_TOTAL_DEMAND',
            'registered_capacity':'PASS' if require_full_curve else 'MISSING',
            'price_limits':'USER_PROVIDED_PENDING_EFFECTIVE_NOTICE' if rules['price_floor'] is not None and rules['price_ceiling'] is not None else 'MISSING',
            'price_and_power_ticks':'RESEARCH_SETTINGS_PENDING_CONFIRMATION',
            'settlement':'CONTRACT_DIFFERENCE_INCLUDED_FEES_MONTHLY_RECOVERY_EXCLUDED' if budget_scope=='ENERGY_WITH_CONTRACT_DIFFERENCE' else 'PARTIAL_SPOT_BUDGET_NOT_TOTAL_PROCUREMENT_COST'},
        'solver_status':int(primary.status), 'tie_solver_status':int(secondary.status),
        'price_level_count':m, 'primary_mip_gap':float(primary.mip_gap),
        'warning':'人工复核申报草稿；不可自动提交；申报量不等于成交量；研究规则未经正式核验。'}


def customer_template_rows(result):
    """Exactly 120 rows; unused slots stay blank, never guessed zero bids."""
    if result.get('status') != 'RESEARCH_ONLY' or result.get('version') not in EXPORTABLE_VERSIONS:
        raise ValueError('Only successful multisegment research can be exported')
    rows = result['records']
    if len(rows) != 24 or [r['period'] for r in rows] != list(range(1,25)):
        raise ValueError('Complete ordered day required')
    output = []
    for row in rows:
        for slot in range(1,6):
            s = next((s for s in row['segments'] if s['sequence']==slot), None)
            output.append([f"{row['period']:02d}:00", slot, '用电',
                None if s is None else s['start_power_mw'], None if s is None else s['end_power_mw'],
                None if s is None else s['bid_price_yuan_per_mwh']])
    return output
