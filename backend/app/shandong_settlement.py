"""SD ordinary retailer energy components, not a complete settlement bill.

2026-04 rules 14.6.3--5: load*RT + DA_cleared*(DA-RT)
                         + sum(contract_energy*(contract_price-reference)).
Contract quantities never reduce the maximum-demand declaration endpoint.
"""
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite

import numpy as np

EVIDENCE = {
    'url': 'https://sdb.nea.gov.cn/dtyw/tzgg/202605/P020260508675725961073.pdf',
    'clauses': ['14.2.13', '14.3.2', '14.3.3', '14.6.1', '14.6.3', '14.6.4', '14.6.5', '14.10.4'],
    'pdf_pages': [240, 241, 250, 251, 252, 253, 269, 270],
    'effective_from': '2026-04-30',
}
EXCLUSIONS = ['MONTH_END_ADJUSTMENT', 'MONTHLY_CONTRACT_DEVIATION_RECOVERY',
              'OPERATION_AND_ANCILLARY_COSTS', 'OTHER_FEES', 'RETAIL_REVENUE']


def number(value):
    return type(value) in (int, float) and isfinite(value)


def contract_book(book, business_date):
    """An empty hourly contract list is explicit zero; absent data is unknown."""
    if book is None:
        return {'status': 'UNKNOWN', 'complete': False, 'records': [],
                'missing': ['24_HOUR_SIGNED_CONTRACTS_AND_REFERENCE_PRICES']}
    if not isinstance(book, dict) or set(book)-{'status', 'business_date', 'source', 'data_version',
            'confirmed_by', 'negative_quantity_confirmed', 'records'}:
        raise ValueError('Invalid contract settlement fields')
    if book.get('business_date') != business_date:
        raise ValueError('Contract settlement date must match delivery date')
    if business_date < EVIDENCE['effective_from']:
        raise ValueError('Verified April settlement rules cannot certify earlier dates')
    for key in ('source', 'data_version', 'confirmed_by'):
        if not isinstance(book.get(key), str) or not book[key].strip():
            raise ValueError(f'Contract {key} required')
    if book.get('status') not in ('CONFIRMED_NONE', 'PROVIDED'):
        raise ValueError('Provide confirmed contracts, confirmed no contracts, or null for unknown')
    if 'negative_quantity_confirmed' in book and type(book['negative_quantity_confirmed']) is not bool:
        raise ValueError('Explicit boolean negative quantity confirmation required')
    rr = book.get('records')
    if book['status'] == 'CONFIRMED_NONE':
        if rr not in (None, []):
            raise ValueError('Confirmed no contracts cannot carry contract records')
        return {**book, 'complete': True, 'records': [], 'missing': []}
    if not isinstance(rr, list) or len(rr) != 24 or any(not isinstance(r, dict) for r in rr):
        raise ValueError('Contract book requires 24 explicit hourly records')
    if any(type(r.get('period')) is not int for r in rr) or {r['period'] for r in rr} != set(range(1,25)):
        raise ValueError('Unique contract periods 1-24 required')
    for row in rr:
        if set(row) != {'period', 'contracts'} or not isinstance(row['contracts'], list):
            raise ValueError('Each hour requires an explicit contracts list; [] means confirmed none')
        for c in row['contracts']:
            if not isinstance(c, dict) or set(c) != {'quantity_mwh', 'price_yuan_per_mwh', 'reference'}:
                raise ValueError('Contract quantity, price and explicit settlement reference required')
            if not number(c['quantity_mwh']) or not number(c['price_yuan_per_mwh']):
                raise ValueError('Unknown contract quantities/prices cannot be replaced by zero')
            if c['reference'] not in ('DA_UNIFIED', 'RT_UNIFIED'):
                raise ValueError('Non-unified contract references need separate paired node-price paths; no silent substitution')
            if c['quantity_mwh'] < 0 and book.get('negative_quantity_confirmed') is not True:
                raise ValueError('Negative contract quantity requires explicit adjustment/offset confirmation')
    return {**book, 'complete': True, 'records': sorted(rr,key=lambda r:r['period']), 'missing': []}


def contract_scenario_costs(book, da, rt):
    """Return one contract-difference cost per scenario; None means unknown."""
    da, rt = np.asarray(da, dtype=float), np.asarray(rt, dtype=float)
    if da.ndim != 2 or da.shape != rt.shape or da.shape[1] != 24 or not np.isfinite(da).all() or not np.isfinite(rt).all():
        raise ValueError('Aligned finite 24-hour DA/RT scenario arrays required')
    if not book['complete']:
        return None
    costs = np.zeros(len(da))
    for row in book['records']:
        h = row['period']-1
        for c in row['contracts']:
            reference = da[:,h] if c['reference'] == 'DA_UNIFIED' else rt[:,h]
            costs += c['quantity_mwh']*(c['price_yuan_per_mwh']-reference)
    return costs


def monthly_contract_recovery(payload):
    """Use externally published monthly recovery prices, not the client's price.

    Article 14.10.4 uses ALL generation-side contracts to form recovery prices.
    Those aggregate inputs cannot be reconstructed from this retailer's book.
    """
    fields = {'month', 'settled_load_mwh', 'net_contract_mwh', 'upper_ratio', 'lower_ratio',
              'excess_recovery_price', 'shortfall_recovery_price', 'parameter_source', 'parameter_version'}
    if not isinstance(payload, dict) or set(payload)-fields:
        raise ValueError('Invalid monthly recovery fields')
    month = payload.get('month')
    if not isinstance(month, str) or len(month) != 7:
        raise ValueError('Month YYYY-MM required')
    date.fromisoformat(month+'-01')
    if month < '2026-05':
        raise ValueError('Monthly application before May requires historical rule version')
    numeric = fields-{'month', 'parameter_source', 'parameter_version'}
    missing = [k for k in sorted(numeric) if payload.get(k) is None]
    for k in numeric:
        if payload.get(k) is not None and (not number(payload[k]) or (k != 'net_contract_mwh' and payload[k] < 0)):
            raise ValueError(f'Invalid {k}')
    for k in ('parameter_source', 'parameter_version'):
        if not isinstance(payload.get(k), str) or not payload[k].strip():
            missing.append(k)
    base = dict(month=month, execution_allowed=False, scope='MONTHLY_USER_CONTRACT_RECOVERY_ONLY',
                evidence=EVIDENCE, inputs=payload, missing=missing)
    if missing:
        return {**base, 'status': 'BLOCKED', 'total_recovery_yuan': None}
    if payload['upper_ratio'] < payload['lower_ratio']:
        raise ValueError('Reversed contract coverage limits')
    d = lambda k: Decimal(str(payload[k]))
    excess = max(Decimal(0), d('net_contract_mwh')-d('settled_load_mwh')*d('upper_ratio'))
    short = max(Decimal(0), d('settled_load_mwh')*d('lower_ratio')-d('net_contract_mwh'))
    # Published recovery prices are yuan/MWh, rounded to 3 dp per 14.10.4.
    ep = d('excess_recovery_price').quantize(Decimal('.001'), rounding=ROUND_HALF_UP)
    sp = d('shortfall_recovery_price').quantize(Decimal('.001'), rounding=ROUND_HALF_UP)
    amount = (excess*ep+short*sp).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    return {**base, 'status': 'RESEARCH_ONLY', 'excess_mwh': float(excess), 'shortfall_mwh': float(short),
            'excess_price_used': float(ep), 'shortfall_price_used': float(sp),
            'total_recovery_yuan': float(amount), 'not_hourly_da_load_deviation_penalty': True}
