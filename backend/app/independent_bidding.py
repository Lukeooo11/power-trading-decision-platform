"""Independent hourly bids with daily joint CVaR, strictly offline research.

No original customer bids or target actual prices are decision inputs. Each bid
is a single demand segment whose scenario acceptance depends on DA price.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import isfinite
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

VERSION = 'independent-joint-bid-v1'
BUDGET_VERSION = 'independent-joint-bid-absolute-budget-v2'
QUANTITY_FRACTIONS = (.25, .5, .75, 1.)
PRICE_DISCOUNTS = (0., 20., 50.)


def finite(value):
    return type(value) in (int, float) and isfinite(value)


def weighted_cvar(values, weights, confidence=.95):
    """Finite-distribution expected shortfall with partial boundary probability."""
    remaining=1-confidence; total=0.
    for value,weight in sorted(zip(values,weights),reverse=True):
        take=min(remaining,weight);total+=take*value;remaining-=take
        if remaining<=1e-12:break
    return float(total/(1-confidence))


def accepted_quantity(quantity, price, market_price, tie_fraction=0.):
    if not finite(quantity) or quantity<0 or not finite(market_price):
        raise ValueError('Finite nonnegative quantity and finite market price required')
    if not finite(tie_fraction) or not 0<=tie_fraction<=1:
        raise ValueError('Tie fraction must be in [0,1]')
    if quantity==0:return 0.
    if not finite(price):raise ValueError('Positive quantity requires finite bid price')
    diff=price-market_price
    return quantity*(tie_fraction if abs(diff)<=1e-9 else float(diff>0))


def optimize_independent_bids(*, business_date, records, scenario_banks,
                              exposure_basis='UNIT_RESEARCH', risk_aversion=.3,
                              time_limit_seconds=15., absolute_cost_budget_yuan=None,
                              risk_budget_tightening=None, include_hedge_candidates=False,
                              fixed_procurement_cost_yuan=None, signal_min_advantage_yuan_mwh=None,
                              period_fraction_bounds=None):
    """Optimize standalone bids subject to an optional absolute daily CVaR cap.

    A business cap is in yuan/day, NOT yuan/MWh or relative cost versus RT.
    Research tightening interpolates from full-RT worst-bank tail toward the
    minimum attainable worst-bank tail in the same candidate set. No realized
    target-day prices are used to construct this budget. Unknown locked costs
    are excluded explicitly, not represented as known zero procurement costs.
    """
    cutoff=date.fromisoformat(business_date)-timedelta(days=2)
    if signal_min_advantage_yuan_mwh is not None and (not finite(signal_min_advantage_yuan_mwh) or signal_min_advantage_yuan_mwh<0):
        raise ValueError('Finite nonnegative cost advantage buffer required')
    if absolute_cost_budget_yuan is not None and not finite(absolute_cost_budget_yuan):
        raise ValueError('Finite absolute daily budget required')
    if risk_budget_tightening is not None and (not finite(risk_budget_tightening) or not 0<=risk_budget_tightening<=1):
        raise ValueError('Risk budget tightening must be in [0,1]')
    if absolute_cost_budget_yuan is not None and risk_budget_tightening is not None:
        raise ValueError('Choose a business budget or a research tightening, not both')
    if type(include_hedge_candidates) is not bool:
        raise ValueError('Explicit boolean hedge candidate setting required')
    if fixed_procurement_cost_yuan is not None and not finite(fixed_procurement_cost_yuan):
        raise ValueError('Finite locked procurement cost required')
    budget_enabled=absolute_cost_budget_yuan is not None or risk_budget_tightening is not None
    fixed_cost=0. if fixed_procurement_cost_yuan is None else fixed_procurement_cost_yuan
    if not finite(risk_aversion) or not 0<=risk_aversion<=1:raise ValueError('Invalid risk aversion')
    if not finite(time_limit_seconds) or time_limit_seconds<=0:raise ValueError('Invalid solver time limit')
    if exposure_basis not in {'UNIT_RESEARCH','PROVIDED_RESEARCH'}:raise ValueError('Explicit research exposure basis required')
    if not isinstance(records,list) or len(records)!=24 or any(not isinstance(r,dict) for r in records):
        raise ValueError('Exactly 24 records required')
    fields={'period','forecast_da','forecast_rt','exposure_mwh'}
    if any(set(r)-fields for r in records):raise ValueError('Unexpected inputs: actuals and customer bids are not accepted')
    if any(type(r.get('period')) is not int for r in records) or {r['period'] for r in records}!=set(range(1,25)):
        raise ValueError('Unique periods 1-24 required')
    rows=sorted(records,key=lambda r:r['period'])
    allowed_fractions={0.,*QUANTITY_FRACTIONS}
    if period_fraction_bounds is None:
        fraction_bounds={period:{'min_fraction':0.,'max_fraction':1.,'mode':'UNCONSTRAINED','reason':None}
                         for period in range(1,25)}
    else:
        if not isinstance(period_fraction_bounds,dict) or any(type(period) is not int for period in period_fraction_bounds):
            raise ValueError('Period fraction bounds must be an integer-keyed object')
        if set(period_fraction_bounds)-set(range(1,25)):
            raise ValueError('Period fraction bounds must use periods 1-24')
        fraction_bounds={}
        for period in range(1,25):
            item=period_fraction_bounds.get(period,{'min_fraction':0.,'max_fraction':1.,'mode':'UNCONSTRAINED','reason':None})
            if not isinstance(item,dict) or set(item)-{'min_fraction','max_fraction','mode','reason'}:
                raise ValueError('Unexpected period fraction bound fields')
            lower=item.get('min_fraction',0.);upper=item.get('max_fraction',1.)
            if lower not in allowed_fractions or upper not in allowed_fractions or lower>upper:
                raise ValueError('Fraction bounds must use available candidate fractions in increasing order')
            fraction_bounds[period]={'min_fraction':float(lower),'max_fraction':float(upper),
                'mode':item.get('mode','UNCONSTRAINED'),'reason':item.get('reason')}
    base=dict(version=BUDGET_VERSION if budget_enabled or include_hedge_candidates else VERSION,business_date=business_date,source_cutoff=cutoff.isoformat(),
              execution_allowed=False,formal_action='HOLD',formal_gate='BLOCKED',
              formal_blockers=['EXPOSURE_AND_SETTLEMENT_NOT_CERTIFIED','MARKET_BID_RULES_UNCONFIRMED'],
              exposure_basis=exposure_basis,customer_bids_used=False,objective_basis='DAILY_RELATIVE_COST_VS_FULL_RT',
              risk_aversion=risk_aversion,confidence=.95,price_discounts=list(PRICE_DISCOUNTS),
              quantity_fractions=list(QUANTITY_FRACTIONS),fee_basis='EXCLUDED_PENDING_ACTUAL_RULES',
              include_hedge_candidates=include_hedge_candidates,
              period_fraction_bounds=fraction_bounds,
              fixed_procurement_cost_yuan=fixed_procurement_cost_yuan,
              absolute_cost_scope='RESIDUAL_EXPOSURE_ONLY' if fixed_procurement_cost_yuan is None else 'RESIDUAL_PLUS_PROVIDED_LOCKED_COST',
              settlement_rule_notice='历史剩余敞口研究对照；不是山东总需求申报曲线，也不是含中长期差价、月度回收和费用的完整结算成本。正式形状研究请使用多段总需求模式。',
              equal_price_fill_assumption=.5,
              risk_budget={'enabled':budget_enabled,'confidence':.95,'unit':'CNY_PER_DAY',
                           'requested_budget_yuan':absolute_cost_budget_yuan,'tightening':risk_budget_tightening})
    base['signal_gate']={'enabled':signal_min_advantage_yuan_mwh is not None,
                         'buffer_yuan_mwh':signal_min_advantage_yuan_mwh,
                         'uncertainty_method':'ONE_WEIGHTED_STANDARD_ERROR_HEURISTIC_NOT_CONFIDENCE_GUARANTEE',
                         'weak_signal_hedge_fraction_limit':.25}
    def blocked(reason,**details):
        return {**base,'status':'BLOCKED','reason':reason,'records':[],**details}
    if any(not finite(r.get(k)) for r in rows for k in ('forecast_da','forecast_rt','exposure_mwh')):
        return blocked('MISSING_FORECAST_OR_EXPOSURE')
    if any(r['exposure_mwh']<0 for r in rows):raise ValueError('Overcoverage requires separate review; do not pass negative exposure')
    if exposure_basis=='UNIT_RESEARCH' and any(r['exposure_mwh']!=1 for r in rows):
        raise ValueError('UNIT_RESEARCH means exactly 1 MWh per hour')
    total_exposure=sum(r['exposure_mwh'] for r in rows)
    if total_exposure<=0:return blocked('NO_POSITIVE_EXPOSURE')
    if not isinstance(scenario_banks,dict) or not scenario_banks:return blocked('SCENARIOS_MISSING')
    banks={}
    for name,paths in scenario_banks.items():
        if not isinstance(paths,list):raise ValueError('Expected scenario paths list')
        seen=set()
        for p in paths:
            if not isinstance(p,dict):raise ValueError('Scenario must be an object')
            stamp=p.get('source_date')
            try:parsed=date.fromisoformat(stamp)
            except (TypeError,ValueError):raise ValueError('Invalid source date') from None
            if parsed>cutoff or stamp in seen:raise ValueError('Unique D-2 historical sources required')
            seen.add(stamp)
            if not finite(p.get('weight')) or p['weight']<=0:raise ValueError('Positive finite weight required')
            for k in ['day_ahead','real_time']:
                if not isinstance(p.get(k),list) or len(p[k])!=24 or any(not finite(v) for v in p[k]):
                    raise ValueError('24 finite paired prices required')
        weights=np.array([p['weight'] for p in paths],float)
        if len(paths)<10:return blocked('INSUFFICIENT_HISTORY',bank=name)
        weights/=sum(weights);ess=float(1/(weights@weights))
        if ess<8:return blocked('INSUFFICIENT_EFFECTIVE_HISTORY',bank=name)
        banks[name]=dict(weights=weights,da=np.array([p['day_ahead'] for p in paths]),
                         rt=np.array([p['real_time'] for p in paths]),ess=ess,
                         source_dates=[p['source_date'] for p in paths])
    candidates=[];hour_ids=[]
    for row in rows:
        ids=[]
        bound=fraction_bounds[row['period']]
        prices=[(row['forecast_rt']-d,d,'RT_MINUS_DISCOUNT') for d in PRICE_DISCOUNTS]
        if include_hedge_candidates:
            prices += [(row['forecast_rt']+p,-p,'RT_PLUS_HEDGE_PREMIUM') for p in (50.,100.)]
            prices += [(max(row['forecast_da'],row['forecast_rt'])+200.,None,'MAX_DA_RT_PLUS_200')]
        choices=[(0.,None,None,'NO_DA_BID'),*[(f,p,d,b) for f in QUANTITY_FRACTIONS for p,d,b in prices]]
        for fraction,price,discount,basis in choices:
            if row['exposure_mwh']==0 and fraction:continue
            if not bound['min_fraction']<=fraction<=bound['max_fraction']:continue
            ids.append(len(candidates))
            candidates.append(dict(period=row['period'],fraction=fraction,
                                   quantity_mwh=row['exposure_mwh']*fraction,
                                   price=price,price_basis=basis,
                                   price_discount=discount if fraction else None))
        hour_ids.append(ids)
    m=len(candidates)
    for bank in banks.values():
        bank['accepted']=np.array([[accepted_quantity(c['quantity_mwh'],c['price'],float(price[c['period']-1]),.5)
                                   for c in candidates] for price in bank['da']])
        delta=np.array([bank['da'][:,c['period']-1]-bank['rt'][:,c['period']-1] for c in candidates]).T
        bank['loss']=bank['accepted']*delta/total_exposure
        bank['rt_cost']=(bank['rt']@np.array([r['exposure_mwh'] for r in rows])+fixed_cost)/total_exposure
    allowed=np.ones(m,dtype=bool)
    candidate_signals=[]
    rt_worst=max(weighted_cvar(b['rt_cost'],b['weights']) for b in banks.values())
    for j,c in enumerate(candidates):
        signal={'status':'DISABLED','economic_lower_advantage_yuan_mwh':None}
        if signal_min_advantage_yuan_mwh is not None and c['fraction']:
            lower=[]
            for bank in banks.values():
                w=bank['weights'];saving=-bank['loss'][:,j]*total_exposure
                mean=float(w@saving);std=float(np.sqrt(w@((saving-mean)**2)))
                expected_quantity=float(w@bank['accepted'][:,j])
                lower.append((mean-std/np.sqrt(bank['ess']))/expected_quantity if expected_quantity>1e-9 else -1e12)
            advantage=min(lower)
            hedge_tail=max(weighted_cvar(b['rt_cost']+b['loss'][:,j],b['weights']) for b in banks.values())
            economic=advantage>=signal_min_advantage_yuan_mwh
            hedge=c['fraction']<=.25 and hedge_tail<rt_worst-1e-8
            allowed[j]=economic or hedge
            signal={'status':'ECONOMIC_ADVANTAGE' if economic else 'LIMITED_TAIL_HEDGE' if hedge else 'INSUFFICIENT_COST_ADVANTAGE',
                    'economic_lower_advantage_yuan_mwh':advantage,
                    'standalone_tail_reduction_yuan':float((rt_worst-hedge_tail)*total_exposure)}
        elif not c['fraction']:signal['status']='NO_DA_BID'
        candidate_signals.append(signal)
    base['signal_gate'].update(rejected_candidates=int((~allowed).sum()),
        hedge_candidates=sum(s['status']=='LIMITED_TAIL_HEDGE' for s in candidate_signals))
    # Binary candidate choices, one eta + tail slacks per bank, and worst-bank epigraph z.
    extra=sum(1+len(b['weights']) for b in banks.values())
    size=m+extra*(2 if budget_enabled else 1)+1+int(budget_enabled)
    z=m+extra*(2 if budget_enabled else 1);absolute_z=z+1 if budget_enabled else None
    low=np.zeros(size);high=np.full(size,np.inf);high[:m]=1;low[z]=-np.inf
    if budget_enabled:low[absolute_z]=-np.inf
    A=[];lb=[];ub=[];cursor=m
    for ids in hour_ids:
        row=np.zeros(size);row[ids]=1;A.append(row);lb.append(1);ub.append(1)
    for bank in banks.values():
        n=len(bank['weights']);eta=cursor;slack=cursor+1;cursor+=1+n;low[eta]=-np.inf
        for s in range(n):
            row=np.zeros(size);row[:m]=bank['loss'][s];row[eta]=-1;row[slack+s]=-1
            A.append(row);lb.append(-np.inf);ub.append(0)
        row=np.zeros(size);row[:m]=(1-risk_aversion)*(bank['weights']@bank['loss'])
        row[eta]=risk_aversion;row[slack:slack+n]=risk_aversion*bank['weights']/.05;row[z]=-1
        A.append(row);lb.append(-np.inf);ub.append(0)
    if budget_enabled:
        for bank in banks.values():
            n=len(bank['weights']);eta=cursor;slack=cursor+1;cursor+=1+n;low[eta]=-np.inf
            for s in range(n):
                row=np.zeros(size);row[:m]=bank['loss'][s];row[eta]=-1;row[slack+s]=-1
                A.append(row);lb.append(-np.inf);ub.append(-bank['rt_cost'][s])
            row=np.zeros(size);row[eta]=1;row[slack:slack+n]=bank['weights']/.05;row[absolute_z]=-1
            A.append(row);lb.append(-np.inf);ub.append(0)
    A=np.array(A);lb=np.array(lb);ub=np.array(ub)
    objective=np.zeros(size);objective[z]=1
    integral=np.zeros(size);integral[:m]=1
    options={'time_limit':time_limit_seconds,'mip_rel_gap':1e-9,'threads':1}
    rt_tail=max(weighted_cvar(b['rt_cost'],b['weights']) for b in banks.values())*total_exposure
    budget=absolute_cost_budget_yuan;minimum_tail=None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected')
        if risk_budget_tightening is not None:
            min_objective=np.zeros(size);min_objective[absolute_z]=1
            minimum=milp(min_objective,integrality=integral,bounds=Bounds(low,high),
                         constraints=LinearConstraint(A,lb,ub),options=options)
            if minimum.status!=0:
                return blocked('RISK_BUDGET_FEASIBILITY_NOT_PROVEN',solver_status=int(minimum.status))
            minimum_tail=float(minimum.fun*total_exposure)
            budget=rt_tail-risk_budget_tightening*max(0.,rt_tail-minimum_tail)
        if budget_enabled:
            base['risk_budget'].update(limit_yuan=float(budget),full_rt_cvar95_yuan=rt_tail,
                minimum_attainable_cvar95_yuan=minimum_tail,
                source='EXPLICIT_BUSINESS_CAP' if risk_budget_tightening is None else 'RESEARCH_SCENARIO_FRONTIER',
                guarantee='SCENARIO_CONSTRAINT_ONLY_NOT_REALIZED_COST_GUARANTEE')
            high[absolute_z]=budget/total_exposure+1e-8
        # Determine the budget on the original space, then filter weak signals.
        # Recomputing the budget after filtering would silently weaken protection.
        high[:m]=allowed.astype(float)
        first=milp(objective,integrality=integral,bounds=Bounds(low,high),
                   constraints=LinearConstraint(A,lb,ub),options=options)
        if first.status!=0:
            return blocked('ABSOLUTE_COST_BUDGET_INFEASIBLE' if budget_enabled and first.status==2 else 'OPTIMIZATION_NOT_PROVEN',solver_status=int(first.status))
        tie=np.zeros(size);tie[:m]=[c['quantity_mwh']/total_exposure for c in candidates]
        high2=high.copy();high2[z]=first.fun+1e-7
        second=milp(tie,integrality=integral,bounds=Bounds(low,high2),
                    constraints=LinearConstraint(A,lb,ub),options=options)
    chosen=second if second.status==0 else first
    selected=[]
    for ids in hour_ids:
        selected.append(max(ids,key=lambda j:chosen.x[j]))
    if any(abs(chosen.x[j]-1)>1e-5 for j in selected):return blocked('INVALID_SOLVER_ASSIGNMENT')
    diagnostics={}
    for name,bank in banks.items():
        loss=bank['loss'][:,selected].sum(axis=1);mean=float(bank['weights']@loss)
        tail=weighted_cvar(loss,bank['weights']);score=(1-risk_aversion)*mean+risk_aversion*tail
        absolute_cost=(bank['rt_cost']+loss)*total_exposure
        absolute_tail=weighted_cvar(absolute_cost,bank['weights'])
        diagnostics[name]=dict(mean_relative_cost=mean,cvar95_relative_cost=tail,risk_score=score,
                               scenario_count=len(loss),effective_sample_size=bank['ess'],
                               source_dates=bank['source_dates'],scenario_weights=bank['weights'].tolist(),
                               daily_relative_losses=loss.tolist(),
                               daily_absolute_costs_yuan=absolute_cost.tolist(),
                               mean_absolute_cost_yuan=float(bank['weights']@absolute_cost),
                               cvar95_absolute_cost_yuan=absolute_tail,
                               budget_slack_yuan=None if not budget_enabled else float(budget-absolute_tail),
                               expected_accepted_da_mwh=float(bank['weights']@bank['accepted'][:,selected].sum(axis=1)))
    score=max(b['risk_score'] for b in diagnostics.values())
    all_rt_feasible=not budget_enabled or rt_tail<=budget+1e-6
    if abs(score-first.fun)>2e-5 or (all_rt_feasible and score>2e-5):return blocked('OBJECTIVE_RECONCILIATION_FAILED')
    if budget_enabled:
        used=max(b['cvar95_absolute_cost_yuan'] for b in diagnostics.values())
        if used>budget+max(1e-5,total_exposure*2e-5):return blocked('RISK_BUDGET_RECONCILIATION_FAILED')
        base['risk_budget'].update(used_cvar95_yuan=used,slack_yuan=float(budget-used),passed=True)
    output=[]
    for row,j in zip(rows,selected):
        c=candidates[j]
        output.append(dict(period=row['period'],exposure_mwh=row['exposure_mwh'],forecast_da=row['forecast_da'],
                           forecast_rt=row['forecast_rt'],submitted_fraction=c['fraction'],
                           max_bid_quantity_mwh=c['quantity_mwh'],bid_price_yuan_per_mwh=c['price'],
                           price_basis=c['price_basis'],
                           event_fraction_bound=fraction_bounds[row['period']],
                           signal_gate=candidate_signals[j],
                           price_discount=c['price_discount'],research_action='BID_RESEARCH' if c['fraction'] else 'RESERVE_REALTIME',
                           formal_action='HOLD',execution_allowed=False,
                           expected_accepted_by_bank={name:float(b['weights']@b['accepted'][:,j]) for name,b in banks.items()}))
    return {**base,'status':'RESEARCH_ONLY','records':output,'total_exposure_mwh':total_exposure,
            'selected_score':score,'scenario_diagnostics':diagnostics,
            'solver_status':int(first.status),'tie_solver_status':int(second.status),
            'candidate_count':m,'all_rt_feasible':all_rt_feasible,
            'warning':'Submitted quantity is not cleared quantity. Research prices do not certify market-valid bids.'}
