"""Offline challenger: paired whole-day residual paths and discrete daily CVaR.

Not wired into live routes. No execution, synthetic quantile fallback or target
actuals. Prices for all 24 hours in a scenario share one historical source day.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import isfinite

from .bidding_strategy import _num, _weighted_cvar

VERSION = "research-paired-day-path-cvar-v1"
DA = "day_ahead_price_yuan_per_mwh"
RT = "real_time_price_yuan_per_mwh"


def _period_map(rows):
    if len(rows) != 24:
        raise ValueError("Exactly 24 periods required")
    result = {}
    for row in rows:
        p = row.get("period")
        if type(p) is not int or not 1 <= p <= 24 or p in result:
            raise ValueError("Periods must be unique integers 1..24")
        result[p] = row
    return result


def _center(row, key):
    q = row.get(key) or {}
    values = [_num(q.get(k)) for k in ("p10", "p50", "p90")]
    if None in values or values != sorted(values):
        raise ValueError("Finite ordered forecast quantiles required")
    return values[1]


def build_day_paths(*, forecast_doc, target_date, target_forecasts,
                    samples=30, decay=.95, outcome_lag_days=2):
    """D-2 is an availability assumption, not proof of publication timestamps."""
    if type(samples) is not int or samples < 1 or type(outcome_lag_days) is not int or outcome_lag_days < 2:
        raise ValueError("Positive sample count and at least D-2 lag required")
    if _num(decay) is None or not 0 < decay <= 1:
        raise ValueError("Invalid decay")
    target = _period_map(target_forecasts)
    centers = {p: (_center(target[p], DA), _center(target[p], RT)) for p in target}
    cutoff = (date.fromisoformat(target_date) - timedelta(days=outcome_lag_days)).isoformat()
    usable, seen, excluded = [], set(), []
    for day in forecast_doc.get("results", []):
        stamp = day.get("market_date", "")
        try:
            date.fromisoformat(stamp)
        except (ValueError, TypeError):
            continue
        if stamp > cutoff:
            continue
        if stamp in seen:
            raise ValueError("Duplicate historical date")
        seen.add(stamp)
        if (day.get("forecast_scenario") or day.get("scenario") or "pre_market") not in {"pre_market", "PRE_MARKET"}:
            excluded.append({"date": stamp, "reason": "NOT_PRE_MARKET"})
            continue
        try:
            rows = _period_map(day.get("periods", []))
            da, rt = [], []
            for p in range(1, 25):
                r = rows[p]
                ad, ar = _num(r.get("actual_" + DA)), _num(r.get("actual_" + RT))
                if ad is None or ar is None:
                    raise ValueError("Incomplete historical outcome")
                da.append(centers[p][0] + ad - _center(r, DA))
                rt.append(centers[p][1] + ar - _center(r, RT))
            usable.append({"source_date": stamp, "day_ahead": da, "real_time": rt})
        except ValueError as exc:
            excluded.append({"date": stamp, "reason": str(exc)})
    usable = sorted(usable, key=lambda s: s["source_date"], reverse=True)[:samples]
    total = sum(decay ** i for i in range(len(usable)))
    for i, s in enumerate(usable):
        s["weight"] = decay ** i / total
    ess = 1 / sum(s["weight"] ** 2 for s in usable) if usable else 0
    return {"paths": usable, "source_cutoff": cutoff, "sample_count": len(usable),
            "effective_sample_size": ess, "excluded_days": excluded,
            "execution_allowed": False, "availability": "D-2_ASSUMED_NOT_TIMESTAMP_VERIFIED"}


def evaluate_allocation(paths, exposure, ratios, confidence=.95, risk_aversion=.3):
    total_weight = sum(s["weight"] for s in paths)
    losses = [(sum(e * (r * da + (1-r) * rt) for e, r, da, rt in
                        zip(exposure, ratios, s["day_ahead"], s["real_time"])),
               s["weight"] / total_weight) for s in paths]
    mean = sum(cost * w for cost, w in losses)
    tail = _weighted_cvar(losses, confidence)
    remaining, contributors = 1-confidence, []
    for (cost, w), s in sorted(zip(losses, paths), key=lambda pair: pair[0][0], reverse=True):
        take = min(w, remaining)
        if take > 1e-12:
            contributors.append({"source_date": s["source_date"], "cost": cost, "tail_mass": take})
        remaining -= take
        if remaining < 1e-12:
            break
    return {"expected_daily_cost": mean, "daily_cost_cvar": tail,
            "objective": (1-risk_aversion)*mean + risk_aversion*tail,
            "tail_contributors": contributors}


def optimize_day_paths(paths, exposure, *, confidence=.95, risk_aversion=.3):
    """MILP: r_h=k_h/20, k_h in 1..19; CVaR of total daily cost.

    Same 5%-95% grid as incumbent. Secondary solve prefers proximity to 50%
    within 1e-7 yuan/MWh of the optimum. Solver failures produce no allocation.
    """
    if len(exposure) != 24 or any(_num(e) is None or e < 0 for e in exposure):
        raise ValueError("24 finite nonnegative exposures required")
    if _num(confidence) is None or not 0 < confidence < 1 or _num(risk_aversion) is None or not 0 <= risk_aversion <= 1:
        raise ValueError("Invalid risk parameters")
    seen = set()
    for s in paths:
        if s["source_date"] in seen:
            raise ValueError("Duplicate path source date")
        seen.add(s["source_date"])
        if _num(s.get("weight")) is None or s["weight"] <= 0:
            raise ValueError("Invalid scenario weight")
        for key in ("day_ahead", "real_time"):
            if len(s[key]) != 24 or any(_num(v) is None for v in s[key]):
                raise ValueError("24 finite prices required in every path")
    total_w = sum(s["weight"] for s in paths)
    ess = total_w**2 / sum(s["weight"]**2 for s in paths) if paths else 0
    base_result = {"version": VERSION, "execution_allowed": False, "ratios": None,
                   "sample_count": len(paths), "effective_sample_size": ess}
    if len(paths) < 10 or ess < 8:
        return {**base_result, "status": "BLOCKED", "reason": "INSUFFICIENT_DAY_PATHS"}
    if sum(exposure) == 0:
        return {**base_result, "status": "RESEARCH_ONLY", "ratios": [0.5]*24,
                "allocation": evaluate_allocation(paths, exposure, [.5]*24, confidence, risk_aversion)}
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    w = np.array([s["weight"] / total_w for s in paths])
    da = np.array([s["day_ahead"] for s in paths])
    rt = np.array([s["real_time"] for s in paths])
    e = np.array(exposure) / sum(exposure)
    b = (rt * e).sum(axis=1)
    delta = (da - rt) * e * .05
    n = len(paths)
    # 24 integer grid indices, VaR t, n tail slacks, 24 tie-break distances.
    size = 49 + n
    c = np.zeros(size)
    c[:24] = (1-risk_aversion) * (w @ delta)
    c[24] = risk_aversion
    c[25:25+n] = risk_aversion*w/(1-confidence)
    matrix = np.zeros((n+48, size))
    matrix[:n, :24] = delta
    matrix[:n, 24] = -1
    matrix[:n, 25:25+n] = -np.eye(n)
    upper = np.r_[-b, np.full(24, 10), np.full(24, -10)]
    for h in range(24):
        matrix[n+h, h], matrix[n+h, 25+n+h] = 1, -1
        matrix[n+24+h, h], matrix[n+24+h, 25+n+h] = -1, -1
    lower_bounds = np.r_[np.ones(24), -np.inf, np.zeros(n+24)]
    upper_bounds = np.r_[np.full(24, 19), np.inf, np.full(n+24, np.inf)]
    bounds = Bounds(lower_bounds, upper_bounds)
    integrality = np.r_[np.ones(24), np.zeros(n+25)]
    options = {"time_limit": 20.0, "mip_rel_gap": 1e-10}
    constraints = LinearConstraint(matrix, np.full(len(upper), -np.inf), upper)
    solved = milp(c, integrality=integrality, bounds=bounds, constraints=constraints, options=options)
    if solved.status != 0:
        return {**base_result, "status": "BLOCKED", "reason": "SOLVER_NOT_OPTIMAL", "solver_message": solved.message}
    tie_c = np.r_[np.zeros(25+n), np.ones(24)]
    second = milp(tie_c, integrality=integrality, bounds=bounds,
                  constraints=LinearConstraint(np.vstack([matrix, c]), np.full(len(upper)+1, -np.inf),
                                               np.r_[upper, solved.fun+1e-7]), options=options)
    chosen = second if second.status == 0 else solved
    ratios = [round(round(float(k))*.05, 6) for k in chosen.x[:24]]
    evaluation = evaluate_allocation(paths, exposure, ratios, confidence, risk_aversion)
    optimal = (solved.fun + (1-risk_aversion)*float(w@b)) * sum(exposure)
    if not isfinite(evaluation["objective"]) or abs(evaluation["objective"]-optimal) > max(.01, sum(exposure)*2e-7):
        raise ArithmeticError("Solver and independent CVaR calculation disagree")
    return {**base_result, "status": "RESEARCH_ONLY", "ratios": ratios,
            "allocation": evaluation, "confidence": confidence, "risk_aversion": risk_aversion,
            "solver": {"method": "scipy.optimize.milp/HiGHS", "mip_gap": float(solved.mip_gap),
                       "tie_break_succeeded": second.status == 0, "primary_objective_yuan": optimal}}


