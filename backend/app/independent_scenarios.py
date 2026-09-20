"""Paired historical residual paths for independent bids, with a D-2 boundary."""
from datetime import date, datetime, timedelta, timezone
from math import isfinite

DA='day_ahead_price_yuan_per_mwh'
RT='real_time_price_yuan_per_mwh'


def audit_premarket_snapshot(day):
    """Reject contradictory timing/price identity, disclose missing provenance.

    File generated_at is often a retrospective replay timestamp. It must NOT
    be presented as proof of a forecast issued before historical bidding.
    """
    delivery = date.fromisoformat(day['market_date'])
    audit = day.get('audit') or {}
    if day.get('forecast_scenario') != 'pre_market':
        raise ValueError('Premarket snapshots required; post-clearing RT cannot drive DA bids')
    cutoff = delivery-timedelta(days=2)
    train_end = audit.get('train_end')
    if train_end is not None and date.fromisoformat(train_end) > cutoff:
        raise ValueError('Forecast training cutoff exceeds D-2')
    # The official deadline is D-1 15:00 China time, for the SAME delivery day.
    deadline = datetime.combine(delivery-timedelta(days=1), datetime.min.time(),
        timezone(timedelta(hours=8))).replace(hour=15)
    issued = audit.get('forecast_issued_at')
    if issued is not None:
        stamp = datetime.fromisoformat(issued.replace('Z','+00:00'))
        if stamp.tzinfo is None or stamp > deadline:
            raise ValueError('Timezone-qualified pre-deadline forecast issuance required')
    price_basis = audit.get('settlement_price_basis')
    if price_basis not in (None, 'SD_USER_UNIFIED_SETTLEMENT_POINT'):
        raise ValueError('Retail strategy requires SD user unified settlement prices, not generator node prices')
    return dict(train_end=train_end, train_cutoff_status='CHECKED' if train_end else 'MISSING',
        forecast_issued_at=issued, declaration_deadline=deadline.isoformat(),
        issue_evidence='DECLARED_TIMESTAMP_NOT_INDEPENDENTLY_VERIFIED' if issued else 'MISSING_REPLAY_IS_NOT_ISSUANCE',
        settlement_price_basis=price_basis, price_basis_status='DECLARED' if price_basis else 'UNCONFIRMED',
        effective_rule_from='2026-04-30', formal_availability_certified=False)


def build_independent_inputs(history, business_date, exposure_by_period=None):
    """Never consume target-day realized prices or customer declarations.

    Missing exposure defaults to an explicitly labelled unit research, not a
    customer volume estimate. No post-clearing RT forecast enters the paths.
    """
    cutoff=(date.fromisoformat(business_date)-timedelta(days=2)).isoformat()
    lookup={r['market_date']:r for r in history}
    if len(lookup)!=len(history):raise ValueError('Duplicate forecast date')
    target=lookup.get(business_date)
    if target is None:raise ValueError('Target forecast snapshot missing')

    def rows(day):
        audit_premarket_snapshot(day)
        result=sorted(day['periods'],key=lambda r:r['period'])
        if [r['period'] for r in result]!=list(range(1,25)):
            raise ValueError('Complete forecast day required')
        if day.get('forecast_scenario')!='pre_market':raise ValueError('Premarket snapshots required')
        if any(not isfinite(float(r[k]['p50'])) for r in result for k in (DA,RT)):
            raise ValueError('Finite DA/RT centers required')
        return result

    target_rows=rows(target)
    if exposure_by_period is not None and set(exposure_by_period)!=set(range(1,25)):
        raise ValueError('Explicit 24-period exposure required')
    records=[dict(period=r['period'],forecast_da=r[DA]['p50'],forecast_rt=r[RT]['p50'],
                  exposure_mwh=1. if exposure_by_period is None else exposure_by_period[r['period']]) for r in target_rows]
    sources=sorted((d for d in lookup if d<=cutoff),reverse=True)[:30]
    weights=[.95**i for i in range(len(sources))];total=sum(weights)
    banks={'historical':[],'transport':[]}
    for stamp,weight in zip(sources,weights):
        past=rows(lookup[stamp])
        if any(not isfinite(float(r['actual_'+k])) for r in past for k in (DA,RT)):
            raise ValueError('Historical outcomes must be finite')
        da=[t[DA]['p50']+p['actual_'+DA]-p[DA]['p50'] for t,p in zip(target_rows,past)]
        rt=[t[RT]['p50']+p['actual_'+RT]-p[RT]['p50'] for t,p in zip(target_rows,past)]
        historical=[d+p['actual_'+RT]-p['actual_'+DA] for d,p in zip(da,past)]
        for key,real_time in [('historical',historical),('transport',rt)]:
            banks[key].append(dict(source_date=stamp,weight=weight/total,day_ahead=da,real_time=real_time))
    return dict(records=records,scenario_banks=banks,
                exposure_basis='UNIT_RESEARCH' if exposure_by_period is None else 'PROVIDED_RESEARCH')
