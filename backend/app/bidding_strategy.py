"""Deterministic research bidding strategy for the Shandong retail portfolio.

The strategy deliberately separates decision inputs from realised prices.  A
historical forecast row is used only through its forecast quantiles; realised
DA/RT prices are attached afterwards for backtest scoring.
"""
from __future__ import annotations

from typing import Any

import math


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _spread_probability(da: dict[str, Any], rt: dict[str, Any]) -> tuple[float, float]:
    """Approximate P(RT > DA) from the two forecast intervals.

    P10/P90 are treated as an 80% interval, so sigma ~= width / 2.563.
    This is the normal-approximation form of the Newsvendor critical-fractile
    decision and is only used for research when both intervals are present.
    """
    da50, rt50 = _num(da.get("p50")), _num(rt.get("p50"))
    da10, da90 = _num(da.get("p10")), _num(da.get("p90"))
    rt10, rt90 = _num(rt.get("p10")), _num(rt.get("p90"))
    if None in (da50, rt50, da10, da90, rt10, rt90):
        return 0.5, 0.0
    sigma_da = max(1e-6, (da90 - da10) / 2.563103)
    sigma_rt = max(1e-6, (rt90 - rt10) / 2.563103)
    sigma_spread = math.sqrt(sigma_da * sigma_da + sigma_rt * sigma_rt)
    return _normal_cdf((rt50 - da50) / sigma_spread), sigma_spread


def _scenario_grid(da: dict[str, Any], rt: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a transparent nine-point price scenario grid from P10/P50/P90.

    The quantile points are used as a research approximation until calibrated
    historical residual scenarios are available.  Band weights (10%, 80%,
    10%) are deliberately explicit and sum to one for each market.
    """
    levels = (("P10", 0.10), ("P50", 0.80), ("P90", 0.10))
    scenarios: list[dict[str, Any]] = []
    for da_name, weight_da in levels:
        da_value = _num(da.get(da_name.lower()))
        for rt_name, weight_rt in levels:
            rt_value = _num(rt.get(rt_name.lower()))
            if da_value is None or rt_value is None:
                continue
            scenarios.append({"name": f"DA_{da_name}_RT_{rt_name}",
                              "day_ahead_price": da_value,
                              "real_time_price": rt_value,
                              "weight": weight_da * weight_rt})
    return scenarios


def build_historical_price_scenarios(*, forecast_doc: dict[str, Any], target_date: str,
                                     period: int, target_da: dict[str, Any],
                                     target_rt: dict[str, Any], lookback_days: int = 30) -> list[dict[str, Any]]:
    """Translate pre-target DA/RT forecast residuals to target price scenarios.

    Only rows strictly before ``target_date`` are used.  The paired residuals
    preserve the historical dependence between DA and RT errors and avoid
    using target-day realised prices during a decision.
    """
    target_da50, target_rt50 = _num(target_da.get("p50")), _num(target_rt.get("p50"))
    if target_da50 is None or target_rt50 is None:
        return []
    history = []
    for day in forecast_doc.get("results", []):
        date = str(day.get("market_date", ""))
        if not date or date >= target_date:
            continue
        row = next((item for item in day.get("periods", []) if int(item.get("period", -1)) == period), None)
        if not row:
            continue
        da = row.get("day_ahead_price_yuan_per_mwh") or {}
        rt = row.get("real_time_price_yuan_per_mwh") or {}
        da50 = _num(da.get("p50"))
        rt50 = _num(rt.get("p50"))
        actual_da = _num(row.get("actual_day_ahead_price_yuan_per_mwh"))
        actual_rt = _num(row.get("actual_real_time_price_yuan_per_mwh"))
        if None in (da50, rt50, actual_da, actual_rt):
            continue
        history.append((date, target_da50 + actual_da - da50,
                        target_rt50 + actual_rt - rt50))
    history = sorted(history, key=lambda item: item[0])[-max(1, lookback_days):]
    if not history:
        return []
    # Recency weighting is deterministic and normalized later by the optimizer.
    scenarios = []
    for index, (date, da_value, rt_value) in enumerate(reversed(history)):
        scenarios.append({"name": f"HIST_{date}", "day_ahead_price": da_value,
                          "real_time_price": rt_value, "weight": 0.95 ** index})
    return scenarios


def _weighted_cvar(losses: list[tuple[float, float]], confidence: float = 0.95) -> float | None:
    """Return a discrete, upper-tail weighted CVaR for cost losses."""
    if not losses:
        return None
    tail_mass = max(1e-9, 1.0 - confidence)
    remaining = tail_mass
    weighted = 0.0
    for loss, weight in sorted(losses, key=lambda item: item[0], reverse=True):
        take = min(weight, remaining)
        weighted += loss * take
        remaining -= take
        if remaining <= 1e-9:
            break
    return weighted / tail_mass


def optimize_lock_ratio(da: dict[str, Any], rt: dict[str, Any], *,
                        risk_aversion: float = 0.5, confidence: float = 0.95,
                        minimum_ratio: float = 0.0, maximum_ratio: float = 1.0,
                        step: float = 0.05,
                        historical_scenarios: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Choose a DA lock ratio by minimizing expected cost plus CVaR.

    This is the first scenario-based implementation inspired by the paper.
    It is intentionally deterministic and does not use realised prices.
    """
    scenarios = list(historical_scenarios or []) or _scenario_grid(da, rt)
    if not scenarios:
        return {"ratio": None, "expected_cost": None, "cvar_cost": None,
                "objective": None, "scenario_count": 0, "confidence": confidence,
                "risk_aversion": risk_aversion}
    total_weight = sum(float(x["weight"]) for x in scenarios)
    for scenario in scenarios:
        scenario["weight"] = float(scenario["weight"]) / total_weight
    candidates = []
    count = int(round((maximum_ratio - minimum_ratio) / step))
    for index in range(count + 1):
        ratio = round(minimum_ratio + index * step, 6)
        losses = [(ratio * x["day_ahead_price"] + (1.0 - ratio) * x["real_time_price"], x["weight"])
                  for x in scenarios]
        expected = sum(loss * weight for loss, weight in losses)
        cvar = _weighted_cvar(losses, confidence)
        objective = (1.0 - risk_aversion) * expected + risk_aversion * (cvar or expected)
        candidates.append({"ratio": ratio, "expected_cost": round(expected, 6),
                           "cvar_cost": round(cvar or expected, 6), "objective": round(objective, 6)})
    selected = min(candidates, key=lambda item: (item["objective"], abs(item["ratio"] - 0.5)))
    return {"ratio": selected["ratio"], "expected_cost": selected["expected_cost"],
            "cvar_cost": selected["cvar_cost"], "objective": selected["objective"],
            "scenario_count": len(scenarios), "confidence": confidence,
            "risk_aversion": risk_aversion, "candidate_count": len(candidates),
            "scenario_method": ("historical pre-target paired DA/RT residuals with recency weights"
                                 if historical_scenarios else "P10/P50/P90 quantile grid with 10%/80%/10% band weights"),
            "selected_candidate": selected, "candidate_evaluations": candidates}


def lock_ratio(day_ahead_p50: float, realtime_p50: float, *, interval_width: float,
               realtime_p10: float | None = None, risk_aversion: float = 0.5,
               negative_risk: bool = False, high_risk: bool = False) -> tuple[float, list[str]]:
    """Return the share of residual exposure locked day-ahead.

    For a buyer, RT > DA means day-ahead locking is economically favourable.
    The thresholds are conservative research defaults, not market rules.
    """
    rt_minus_da = realtime_p50 - day_ahead_p50
    downside = max(0.0, day_ahead_p50 - realtime_p10) if realtime_p10 is not None else interval_width * 0.5
    # A conservative edge subtracts half of the lower-tail loss. This is a
    # compact CVaR-style penalty using the available P10/P50/P90 contract.
    risk_adjusted_edge = rt_minus_da - risk_aversion * downside
    if risk_adjusted_edge >= 30:
        ratio, reason = 0.90, "RT_EXPECTED_MUCH_HIGHER"
    elif risk_adjusted_edge >= 10:
        ratio, reason = 0.70, "RT_EXPECTED_HIGHER"
    elif risk_adjusted_edge <= -30:
        ratio, reason = 0.20, "RT_EXPECTED_MUCH_LOWER"
    elif risk_adjusted_edge <= -10:
        ratio, reason = 0.40, "RT_EXPECTED_LOWER"
    else:
        ratio, reason = 0.55, "SPREAD_UNCERTAIN"
    reasons = [reason]
    if risk_adjusted_edge != rt_minus_da:
        reasons.append("CVAR_TAIL_PENALTY")
    if interval_width >= 250:
        ratio = min(ratio, 0.50)
        reasons.append("WIDE_RT_INTERVAL")
    if negative_risk:
        ratio = min(ratio, 0.35)
        reasons.append("NEGATIVE_PRICE_RISK")
    if high_risk:
        ratio = min(ratio, 0.60)
        reasons.append("HIGH_PRICE_RISK")
    return round(max(0.0, min(1.0, ratio)), 4), reasons


def generate_strategy(*, load_mwh: float | None, medium_position_mwh: float | None,
                      cleared_energy_mwh: float | None,
                      forecast: dict[str, Any], actual: dict[str, Any] | None = None,
                      position_quality_flags: list[str] | None = None,
                      risk_aversion: float = 0.5,
                      period: int | None = None,
                      supply_context: dict[str, Any] | None = None,
                      historical_scenarios: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Generate one period's research-only DA/RT allocation."""
    da = forecast.get("day_ahead_price_yuan_per_mwh") or {}
    rt = forecast.get("real_time_price_yuan_per_mwh") or {}
    da50, rt50 = _num(da.get("p50")), _num(rt.get("p50"))
    p10, p90 = _num(rt.get("p10")), _num(rt.get("p90"))
    missing: list[str] = []
    for name, value in (("load_forecast_mwh", load_mwh), ("medium_position_mwh", medium_position_mwh),
                        ("cleared_energy_mwh", cleared_energy_mwh), ("day_ahead_p50", da50),
                        ("real_time_p50", rt50), ("real_time_quantiles", p10 if p10 is not None and p90 is not None else None)):
        if value is None:
            missing.append(name)
    gross = load_mwh - medium_position_mwh - cleared_energy_mwh if not missing[:3] else None
    remaining = max(0.0, gross) if gross is not None else None
    over_covered = gross is not None and gross < 0
    if da50 is None or rt50 is None or p10 is None or p90 is None or remaining is None or over_covered:
        return {
            "action": "HOLD", "day_ahead_quantity_mwh": 0.0,
            "real_time_reserved_mwh": remaining, "day_ahead_price_yuan_per_mwh": None,
            "suggested_price_lower_yuan_per_mwh": None, "suggested_price_upper_yuan_per_mwh": None,
            "remaining_exposure_mwh": remaining, "expected_saving_yuan": None,
            "expected_cost_advantage_yuan": None,
            "risk_adjusted_edge_yuan_per_mwh": None,
            "gate_status": "BLOCKED", "risk_flags": (["OVER_COVERED_REVIEW"] if over_covered else []),
            "reason": ";".join(missing) or "OVER_COVERED_REVIEW",
            "actual": actual or {}, "execution_allowed": False,
        }
    negative_risk = bool((forecast.get("negative_price_risk") or {}).get("level") in {"HIGH", "MEDIUM"})
    high_risk = bool((forecast.get("high_price_risk") or {}).get("level") in {"HIGH", "MEDIUM"})
    residual_demand = _num((supply_context or {}).get("residualDemandProxyMw"))
    renewable_oversupply = residual_demand is not None and residual_demand <= 0
    spread_probability, spread_sigma = _spread_probability(da, rt)
    base_risk_aversion = max(0.0, min(1.0, risk_aversion))
    effective_risk_aversion = base_risk_aversion
    risk_regime = "NORMAL"
    regime_signals: list[str] = []
    if renewable_oversupply and rt50 <= da50:
        effective_risk_aversion = min(effective_risk_aversion, 0.25)
        risk_regime = "MIDDAY_RENEWABLE_OVERSUPPLY"
        regime_signals.append("LOWER_RISK_AVERSION_FOR_REALTIME_FLEXIBILITY")
    da90 = _num(da.get("p90"))
    rt90 = _num(rt.get("p90"))
    if period is not None and 17 <= period <= 21 and da90 is not None and rt90 is not None and rt90 - da90 >= 100:
        effective_risk_aversion = max(effective_risk_aversion, 0.65)
        risk_regime = "EVENING_PEAK_TAIL_RISK"
        regime_signals.append("HIGHER_RISK_AVERSION_FOR_RT_SPIKE")
    if high_risk:
        # An absolute P90>500 flag is retained for review, but it does not by
        # itself imply that RT is riskier than DA.  Relative DA/RT scenarios
        # and the explicit evening trigger determine the risk parameter.
        regime_signals.append("HIGH_PRICE_REVIEW_ONLY")
    maximum_ratio = 0.95
    if renewable_oversupply and rt50 <= da50:
        maximum_ratio = 0.25
    optimization = optimize_lock_ratio(da, rt, risk_aversion=effective_risk_aversion, confidence=0.95,
                                       minimum_ratio=0.05, maximum_ratio=maximum_ratio, step=0.05,
                                       historical_scenarios=historical_scenarios)
    if optimization.get("ratio") is not None:
        ratio = float(optimization["ratio"])
    reasons = ["SCENARIO_CVAR_OPTIMIZATION", "NEWSVENDOR_UNDER_OVERAGE_BALANCE"]
    if negative_risk:
        reasons.append("NEGATIVE_PRICE_RISK")
    if high_risk:
        reasons.append("HIGH_PRICE_RISK")
    if renewable_oversupply and rt50 <= da50:
        reasons.append("MIDDAY_RENEWABLE_OVERSUPPLY_GUARD")
    da_qty = round(remaining * ratio, 6)
    rt_qty = round(remaining - da_qty, 6)
    # Positive means locking this volume day-ahead is cheaper than buying it
    # at the predicted real-time price. It is a cost advantage, not a profit
    # or a guaranteed saving.
    expected_cost_advantage = round((rt50 - da50) * da_qty, 4)
    actual_cost_advantage = None
    if actual and _num(actual.get("dayAheadPriceYuanMwh")) is not None and _num(actual.get("realtimePriceYuanMwh")) is not None:
        actual_cost_advantage = round((float(actual["realtimePriceYuanMwh"]) - float(actual["dayAheadPriceYuanMwh"])) * da_qty, 4)
    risk_adjusted_edge = (rt50 - da50) - effective_risk_aversion * max(0.0, da50 - p10)
    return {
        "action": "BUY_SPLIT", "day_ahead_quantity_mwh": da_qty,
        "real_time_reserved_mwh": rt_qty, "remaining_exposure_mwh": round(remaining, 6),
        "lock_ratio": ratio,
        "spread_probability_rt_gt_da": round(spread_probability, 6),
        "spread_sigma_yuan_per_mwh": round(spread_sigma, 4),
        "optimization": optimization,
        "risk_regime": risk_regime,
        "base_risk_aversion": base_risk_aversion,
        "effective_risk_aversion": effective_risk_aversion,
        "regime_signals": regime_signals,
        "renewable_oversupply_signal": renewable_oversupply,
        "supply_context": supply_context or {},
        "day_ahead_price_yuan_per_mwh": round(da50, 4),
        "day_ahead_price_p10_yuan_per_mwh": _num(da.get("p10")),
        "day_ahead_price_p90_yuan_per_mwh": _num(da.get("p90")),
        "suggested_price_lower_yuan_per_mwh": _num(da.get("p10")),
        "suggested_price_upper_yuan_per_mwh": _num(da.get("p90")),
        "suggested_price_yuan_per_mwh": round(da50, 4),
        "real_time_price_p50_yuan_per_mwh": round(rt50, 4),
        "real_time_price_p10_yuan_per_mwh": p10,
        "real_time_price_p90_yuan_per_mwh": p90,
        "expected_cost_advantage_yuan": expected_cost_advantage,
        "actual_cost_advantage_yuan": actual_cost_advantage,
        # Compatibility aliases for existing consumers; new UI should use the
        # explicit cost-advantage names above.
        "expected_saving_yuan": expected_cost_advantage,
        "actual_saving_yuan": actual_cost_advantage,
        "risk_adjusted_edge_yuan_per_mwh": round(risk_adjusted_edge, 4),
        "gate_status": "RESEARCH_ONLY", "risk_flags": position_quality_flags or [],
        "reason": ";".join(reasons), "execution_allowed": False,
        "quantity_basis": "剩余敞口 × 场景化CVaR优化选定的日前锁定比例",
        "price_basis": "日前预测P50作为研究报价中心，P10-P90作为报价区间",
    }
