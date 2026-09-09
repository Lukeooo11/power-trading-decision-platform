"""Advanced, deterministic research strategies for the Shandong DA/RT split.

This module deliberately keeps every extra idea visible in the result.  It
does not create a live order, does not use target-day realised values and does
not use random Monte-Carlo samples.  The advanced challenger is composed of:

* a calibrated quantile view from pre-target residuals;
* a fixed-weight mixture of similar-day, recent-error and extreme-tail scenes;
* a direct RT-DA residual distribution;
* a Newsvendor-style under/over-purchase regret term;
* confidence shrinkage toward a 50/50 split; and
* an optional daily joint-CVaR adjustment.

The fixed configuration is intentionally modest because the available sample
is short and the strategy remains research-only.
"""
from __future__ import annotations

import math
from copy import deepcopy
from datetime import date
from typing import Any

from .bidding_strategy import _num, _weighted_cvar, build_historical_price_scenarios, generate_strategy
from .similar_day_scenarios import build_similar_day_scenarios

ADVANCED_KEY = "advanced_cvar_v05"
JOINT_KEY = "joint_cvar_v06"
ADVANCED_VERSION = "da-rt-split-hybrid-calibrated-newsvendor-v0.5"
JOINT_VERSION = "da-rt-split-joint-hybrid-cvar-v0.6"

CONFIG: dict[str, Any] = {
    "mixture_weights": {"similar_day": 0.55, "recent_error": 0.25, "extreme_tail": 0.20},
    "recent_lookback_days": 30,
    "calibration_lookback_days": 45,
    "outcome_lag_days": 2,
    "minimum_residual_samples": 10,
    "newsvendor_weight": 0.20,
    "confidence_sample_target": 20.0,
    "confidence_shrink_floor": 0.15,
    "candidate_min_ratio": 0.05,
    "candidate_max_ratio": 0.95,
    "candidate_step": 0.05,
    "joint_delta_grid": [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20],
}


def _valid_quantiles(row: dict[str, Any], key: str) -> tuple[float, float, float] | None:
    q = row.get(key) or {}
    values = tuple(_num(q.get(name)) for name in ("p10", "p50", "p90"))
    return values if None not in values and values == tuple(sorted(values)) else None


def _pre_target_rows(forecast_doc: dict[str, Any], target_date: str, period: int,
                     lookback_days: int, *, require_actual: bool = True) -> list[dict[str, Any]]:
    target = date.fromisoformat(target_date)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for day in forecast_doc.get("results", []):
        stamp = str(day.get("market_date", ""))
        try:
            age = (target - date.fromisoformat(stamp)).days
        except (TypeError, ValueError):
            continue
        if age < CONFIG["outcome_lag_days"] or age > lookback_days or stamp in seen:
            continue
        if (day.get("forecast_scenario") or day.get("scenario") or "pre_market") not in {"pre_market", "PRE_MARKET"}:
            continue
        row = next((item for item in day.get("periods", []) if item.get("period") == period), None)
        if not isinstance(row, dict):
            continue
        da = _valid_quantiles(row, "day_ahead_price_yuan_per_mwh")
        rt = _valid_quantiles(row, "real_time_price_yuan_per_mwh")
        actual_da = _num(row.get("actual_day_ahead_price_yuan_per_mwh"))
        actual_rt = _num(row.get("actual_real_time_price_yuan_per_mwh"))
        if da is None or rt is None or (require_actual and (actual_da is None or actual_rt is None)):
            continue
        seen.add(stamp)
        rows.append({"date": stamp, "age_days": age, "row": row, "da": da, "rt": rt,
                     "actual_da": actual_da, "actual_rt": actual_rt,
                     "da_error": (actual_da - da[1]) if actual_da is not None else None,
                     "rt_error": (actual_rt - rt[1]) if actual_rt is not None else None,
                     "spread_error": ((actual_rt - actual_da) - (rt[1] - da[1]))
                     if actual_da is not None and actual_rt is not None else None})
    return sorted(rows, key=lambda item: item["date"], reverse=True)


def _weighted_quantile(values: list[tuple[float, float]], probability: float) -> float | None:
    usable = [(float(value), max(0.0, float(weight))) for value, weight in values
              if _num(value) is not None and _num(weight) is not None and weight > 0]
    if not usable:
        return None
    total = sum(weight for _, weight in usable)
    threshold = total * max(0.0, min(1.0, probability))
    cumulative = 0.0
    for value, weight in sorted(usable):
        cumulative += weight
        if cumulative >= threshold - 1e-12:
            return value
    return sorted(usable)[-1][0]


def _normalise_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for scene in scenarios:
        da = _num(scene.get("day_ahead_price")); rt = _num(scene.get("real_time_price"))
        weight = _num(scene.get("weight"))
        if da is None or rt is None or weight is None or weight <= 0:
            continue
        clean.append({**scene, "day_ahead_price": da, "real_time_price": rt, "weight": weight})
    total = sum(item["weight"] for item in clean)
    if total <= 0:
        return []
    for item in clean:
        item["weight"] = item["weight"] / total
    return clean


def _scaled_recent_scenarios(*, forecast_doc: dict[str, Any], target_date: str, period: int,
                             target_forecast: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    da = _valid_quantiles(target_forecast, "day_ahead_price_yuan_per_mwh")
    rt = _valid_quantiles(target_forecast, "real_time_price_yuan_per_mwh")
    rows = _pre_target_rows(forecast_doc, target_date, period, CONFIG["recent_lookback_days"])
    if da is None or rt is None or not rows:
        return [], {"status": "BLOCKED", "sample_count": 0, "source": "recent_paired_residuals"}
    scenes: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        decay = 0.95 ** index
        scenes.append({"name": f"RECENT_{item['date']}", "source": "recent_error",
                       "source_date": item["date"], "day_ahead_price": da[1] + item["da_error"],
                       "real_time_price": rt[1] + item["rt_error"], "weight": decay})
    return _normalise_scenarios(scenes), {"status": "READY", "sample_count": len(scenes),
                                          "source": "recent_paired_residuals", "lookback_days": CONFIG["recent_lookback_days"]}


def _extreme_scenarios(*, forecast_doc: dict[str, Any], target_date: str, period: int,
                       target_forecast: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    da = _valid_quantiles(target_forecast, "day_ahead_price_yuan_per_mwh")
    rt = _valid_quantiles(target_forecast, "real_time_price_yuan_per_mwh")
    rows = _pre_target_rows(forecast_doc, target_date, period, CONFIG["calibration_lookback_days"])
    if da is None or rt is None or not rows:
        return [], {"status": "BLOCKED", "sample_count": 0, "source": "extreme_tail_residuals"}
    # Four deterministic tails cover the known Shandong failure modes without
    # inspecting target-day outcomes: RT premium, RT discount, high RT level,
    # and negative/low RT level.
    selectors = [
        ("RT_PREMIUM", lambda item: (item["actual_rt"] - item["actual_da"])),
        ("RT_DISCOUNT", lambda item: -(item["actual_rt"] - item["actual_da"])),
        ("RT_HIGH", lambda item: item["actual_rt"]),
        ("RT_LOW", lambda item: -item["actual_rt"]),
    ]
    selected: dict[str, dict[str, Any]] = {}
    for label, key in selectors:
        pool = [item for item in rows if item["actual_rt"] is not None and item["actual_da"] is not None]
        if pool:
            selected[label] = max(pool, key=key)
    scenes: list[dict[str, Any]] = []
    for label, item in selected.items():
        scenes.append({"name": f"EXTREME_{label}_{item['date']}", "source": "extreme_tail",
                       "source_date": item["date"], "day_ahead_price": da[1] + item["da_error"],
                       "real_time_price": rt[1] + item["rt_error"], "weight": 1.0})
    return _normalise_scenarios(scenes), {"status": "READY" if scenes else "BLOCKED",
                                          "sample_count": len(scenes), "source": "extreme_tail_residuals",
                                          "selector_labels": list(selected)}


def calibrate_forecast_quantiles(*, forecast_doc: dict[str, Any], target_date: str, period: int,
                                 target_forecast: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate target P10/P50/P90 by empirical pre-target residual quantiles."""
    calibrated = deepcopy(target_forecast)
    rows = _pre_target_rows(forecast_doc, target_date, period, CONFIG["calibration_lookback_days"])
    audit: dict[str, Any] = {"version": "residual-quantile-calibration-v0.1", "lookback_days": CONFIG["calibration_lookback_days"],
                             "sample_count": len(rows), "status": "BLOCKED", "markets": {}}
    if len(rows) < CONFIG["minimum_residual_samples"]:
        audit["reason"] = "CALIBRATION_SAMPLE_INSUFFICIENT"
        return calibrated, audit
    for market, key in (("day_ahead", "day_ahead_price_yuan_per_mwh"), ("real_time", "real_time_price_yuan_per_mwh")):
        q = _valid_quantiles(target_forecast, key)
        if q is None:
            audit["reason"] = "TARGET_QUANTILES_MISSING"
            return calibrated, audit
        error_key = "da_error" if market == "day_ahead" else "rt_error"
        errors = [(item[error_key], 0.95 ** index) for index, item in enumerate(rows)
                  if item[error_key] is not None]
        if len(errors) < CONFIG["minimum_residual_samples"]:
            audit["reason"] = f"{market.upper()}_CALIBRATION_SAMPLE_INSUFFICIENT"
            return calibrated, audit
        p10_error = _weighted_quantile(errors, .10); p50_error = _weighted_quantile(errors, .50); p90_error = _weighted_quantile(errors, .90)
        values = [q[1] + error for error in (p10_error, p50_error, p90_error)]
        values = sorted(values)
        calibrated[key] = {"p10": round(values[0], 6), "p50": round(values[1], 6), "p90": round(values[2], 6)}
        nominal_coverage = sum(item["actual_da"] >= item["da"][0] and item["actual_da"] <= item["da"][2] for item in rows) if market == "day_ahead" else sum(item["actual_rt"] >= item["rt"][0] and item["actual_rt"] <= item["rt"][2] for item in rows)
        audit["markets"][market] = {"residual_p10": p10_error, "residual_p50": p50_error, "residual_p90": p90_error,
                                     "nominal_interval_coverage": nominal_coverage / len(rows),
                                     "nominal_target_coverage": .80,
                                     "nominal_coverage_error": nominal_coverage / len(rows) - .80,
                                     "calibrated": calibrated[key]}
    audit["status"] = "READY"
    audit["method"] = "weighted empirical residual quantiles; target-day realised values excluded"
    return calibrated, audit


def predict_spread_from_residuals(*, forecast_doc: dict[str, Any], target_date: str, period: int,
                                  target_forecast: dict[str, Any]) -> dict[str, Any]:
    da = _valid_quantiles(target_forecast, "day_ahead_price_yuan_per_mwh")
    rt = _valid_quantiles(target_forecast, "real_time_price_yuan_per_mwh")
    rows = _pre_target_rows(forecast_doc, target_date, period, CONFIG["calibration_lookback_days"])
    errors = [(item["spread_error"], 0.95 ** index) for index, item in enumerate(rows) if item["spread_error"] is not None]
    if da is None or rt is None or len(errors) < CONFIG["minimum_residual_samples"]:
        return {"status": "BLOCKED", "sample_count": len(errors), "reason": "SPREAD_RESIDUAL_SAMPLE_INSUFFICIENT"}
    spread = rt[1] - da[1]
    residual_q = {name: _weighted_quantile(errors, prob) for name, prob in (("p10", .10), ("p50", .50), ("p90", .90))}
    values = {name: round(spread + value, 6) for name, value in residual_q.items()}
    # Direct spread probability is empirical and paired; it does not assume
    # independent normal DA and RT distributions.
    probability = sum(weight for error, weight in errors if spread + error > 0) / sum(weight for _, weight in errors)
    return {"status": "READY", "sample_count": len(errors), "spread_forecast_yuan_per_mwh": values,
            "probability_rt_gt_da": round(probability, 6), "residual_quantiles": residual_q,
            "method": "direct paired RT-DA residual quantiles"}


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / total if total else 0.0


def optimise_advanced_ratio(scenarios: list[dict[str, Any]], *, risk_aversion: float,
                           newsvendor_weight: float = 0.20) -> dict[str, Any]:
    scenarios = _normalise_scenarios(scenarios)
    if not scenarios:
        return {"ratio": None, "status": "BLOCKED"}
    candidates: list[dict[str, Any]] = []
    for index in range(int(round((CONFIG["candidate_max_ratio"] - CONFIG["candidate_min_ratio"]) / CONFIG["candidate_step"])) + 1):
        ratio = round(CONFIG["candidate_min_ratio"] + index * CONFIG["candidate_step"], 6)
        costs = [(ratio * x["day_ahead_price"] + (1 - ratio) * x["real_time_price"], x["weight"]) for x in scenarios]
        expected = _weighted_mean(costs)
        cvar_value = _weighted_cvar(costs, .95)
        cvar = expected if cvar_value is None else cvar_value
        regret = _weighted_mean([(ratio * max(x["day_ahead_price"] - x["real_time_price"], 0.0)
                                 + (1 - ratio) * max(x["real_time_price"] - x["day_ahead_price"], 0.0), x["weight"])
                                for x in scenarios])
        cvar_objective = (1 - risk_aversion) * expected + risk_aversion * cvar
        objective = (1 - newsvendor_weight) * cvar_objective + newsvendor_weight * regret
        candidates.append({"ratio": ratio, "expected_cost": round(expected, 6), "cvar_cost": round(cvar, 6),
                           "newsvendor_regret": round(regret, 6), "objective": round(objective, 6)})
    selected = min(candidates, key=lambda item: (item["objective"], abs(item["ratio"] - .5)))
    under = _weighted_mean([(max(x["real_time_price"] - x["day_ahead_price"], 0), x["weight"]) for x in scenarios])
    over = _weighted_mean([(max(x["day_ahead_price"] - x["real_time_price"], 0), x["weight"]) for x in scenarios])
    critical = under / (under + over) if under + over > 1e-9 else .5
    return {"status": "READY", "ratio": selected["ratio"], "selected_candidate": selected,
            "candidate_evaluations": candidates, "expected_cost": selected["expected_cost"],
            "cvar_cost": selected["cvar_cost"], "objective": selected["objective"],
            "newsvendor": {"under_purchase_cost_yuan_per_mwh": round(under, 6),
                           "over_purchase_cost_yuan_per_mwh": round(over, 6),
                           "critical_fractile": round(critical, 6),
                           "ratio_minimising_expected_regret": min(candidates, key=lambda x: x["newsvendor_regret"])["ratio"]},
            "scenario_count": len(scenarios), "confidence": .95, "risk_aversion": risk_aversion}


def _confidence(*, probability: float, effective_sample_size: float, sample_target: float) -> float:
    directional = min(1.0, abs(probability - .5) * 2.0)
    sample = min(1.0, max(0.0, effective_sample_size) / sample_target)
    return max(CONFIG["confidence_shrink_floor"], directional * sample)


def build_hybrid_scenarios(*, forecast_doc: dict[str, Any], target_date: str, period: int,
                           target_forecast: dict[str, Any], weather_doc: dict[str, Any] | None,
                           target_model_version: str | None) -> dict[str, Any]:
    similar = build_similar_day_scenarios(forecast_doc=forecast_doc, target_date=target_date,
        period=period, target_forecast=target_forecast, weather_doc=weather_doc,
        target_model_version=target_model_version)
    recent, recent_audit = _scaled_recent_scenarios(forecast_doc=forecast_doc, target_date=target_date,
        period=period, target_forecast=target_forecast)
    extreme, extreme_audit = _extreme_scenarios(forecast_doc=forecast_doc, target_date=target_date,
        period=period, target_forecast=target_forecast)
    components = []
    weights = CONFIG["mixture_weights"]
    if similar.get("status") == "READY":
        components.extend([{**scene, "weight": scene["weight"] * weights["similar_day"]} for scene in similar.get("scenarios", [])])
    if recent:
        components.extend([{**scene, "weight": scene["weight"] * weights["recent_error"]} for scene in recent])
    if extreme:
        components.extend([{**scene, "weight": scene["weight"] * weights["extreme_tail"]} for scene in extreme])
    scenarios = _normalise_scenarios(components)
    effective = 1.0 / sum(item["weight"] ** 2 for item in scenarios) if scenarios else 0.0
    blockers = []
    if len(scenarios) < CONFIG["minimum_residual_samples"]:
        blockers.append("HYBRID_SCENARIO_SAMPLE_INSUFFICIENT")
    if similar.get("status") != "READY":
        blockers.extend(similar.get("blockers", []))
    return {"version": ADVANCED_VERSION, "status": "READY" if scenarios and not blockers else "BLOCKED",
            "blockers": list(dict.fromkeys(blockers)), "scenarios": scenarios, "scenario_count": len(scenarios),
            "effective_sample_size": round(effective, 6), "components": {"similar_day": {"status": similar.get("status"), "sample_count": len(similar.get("scenarios", []))},
                "recent_error": recent_audit, "extreme_tail": extreme_audit}, "mixture_weights": dict(weights),
            "similar_day_audit": similar}


def generate_advanced_strategy(*, forecast_doc: dict[str, Any], target_date: str, period: int,
                               weather_doc: dict[str, Any] | None = None,
                               target_model_version: str | None = None,
                               supply_context: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
    original_forecast = kwargs.get("forecast") or {}
    calibrated, calibration = calibrate_forecast_quantiles(forecast_doc=forecast_doc, target_date=target_date,
        period=period, target_forecast=original_forecast)
    spread_prediction = predict_spread_from_residuals(forecast_doc=forecast_doc, target_date=target_date,
        period=period, target_forecast=calibrated)
    audit = build_hybrid_scenarios(forecast_doc=forecast_doc, target_date=target_date, period=period,
        target_forecast=calibrated, weather_doc=weather_doc, target_model_version=target_model_version)
    base_kwargs = {**kwargs, "forecast": calibrated, "supply_context": supply_context}
    if audit["status"] != "READY":
        result = generate_strategy(**{**base_kwargs, "forecast": {}})
        result["scenario_audit"] = audit; result["quantile_calibration"] = calibration; result["spread_prediction"] = spread_prediction
        result["strategy_version"] = ADVANCED_VERSION
        result["reason"] = ";".join([x for x in [result.get("reason", ""), *audit["blockers"]] if x])
        result["risk_flags"] = list(dict.fromkeys([*result.get("risk_flags", []), *audit["blockers"], calibration.get("reason", "")]))
        return result
    optimisation = optimise_advanced_ratio(audit["scenarios"], risk_aversion=float(kwargs.get("risk_aversion", .3)),
                                           newsvendor_weight=CONFIG["newsvendor_weight"])
    result = generate_strategy(**base_kwargs, historical_scenarios=audit["scenarios"])
    if result.get("action") != "BUY_SPLIT" or optimisation.get("ratio") is None:
        result["scenario_audit"] = audit; result["quantile_calibration"] = calibration; result["spread_prediction"] = spread_prediction
        result["strategy_version"] = ADVANCED_VERSION
        return result
    probability = sum(item["weight"] for item in audit["scenarios"] if item["real_time_price"] > item["day_ahead_price"])
    ess = audit["effective_sample_size"]
    confidence = _confidence(probability=probability, effective_sample_size=ess, sample_target=CONFIG["confidence_sample_target"])
    raw_ratio = float(optimisation["ratio"])
    ratio = .5 + confidence * (raw_ratio - .5)
    ratio = max(CONFIG["candidate_min_ratio"], min(CONFIG["candidate_max_ratio"], ratio))
    negative = bool((calibrated.get("negative_price_risk") or {}).get("level") in {"HIGH", "MEDIUM"})
    rt50 = _num((calibrated.get("real_time_price_yuan_per_mwh") or {}).get("p50"))
    da50 = _num((calibrated.get("day_ahead_price_yuan_per_mwh") or {}).get("p50"))
    regime_signals = []
    ratio_floor = CONFIG["candidate_min_ratio"]
    ratio_ceiling = CONFIG["candidate_max_ratio"]
    renewable = _num((supply_context or {}).get("residualDemandProxyMw"))
    if renewable is not None and renewable <= 0 and rt50 is not None and da50 is not None and rt50 <= da50:
        ratio_ceiling = min(ratio_ceiling, .25); regime_signals.append("MIDDAY_RENEWABLE_OVERSUPPLY_GUARD")
    spread_p50 = _num((spread_prediction.get("spread_forecast_yuan_per_mwh") or {}).get("p50"))
    if 9 <= period <= 16 and negative and spread_p50 is not None and spread_p50 < 0:
        ratio_ceiling = min(ratio_ceiling, .25); regime_signals.append("MIDDAY_NEGATIVE_PRICE_GUARD")
    rt90 = _num((calibrated.get("real_time_price_yuan_per_mwh") or {}).get("p90")); da90 = _num((calibrated.get("day_ahead_price_yuan_per_mwh") or {}).get("p90"))
    spread_p90 = _num((spread_prediction.get("spread_forecast_yuan_per_mwh") or {}).get("p90"))
    if period in range(17, 22) and ((rt90 is not None and da90 is not None and rt90 - da90 >= 100)
                                    or (spread_p90 is not None and spread_p90 >= 100)):
        ratio_floor = max(ratio_floor, .65); regime_signals.append("EVENING_PEAK_TAIL_GUARD")
    ratio = max(ratio_floor, min(ratio_ceiling, ratio))
    remaining = _num(result.get("remaining_exposure_mwh")) or 0.0
    result["lock_ratio_raw"] = round(raw_ratio, 6)
    result["confidence"] = round(confidence, 6)
    result["confidence_method"] = "directional paired spread signal × effective sample size; ratio shrunk toward 50%"
    result["lock_ratio"] = round(ratio, 6)
    result["lock_ratio_floor"] = ratio_floor
    result["lock_ratio_ceiling"] = ratio_ceiling
    result["day_ahead_quantity_mwh"] = round(remaining * ratio, 6)
    result["real_time_reserved_mwh"] = round(remaining - result["day_ahead_quantity_mwh"], 6)
    result["expected_cost_advantage_yuan"] = round(((_num((calibrated.get("real_time_price_yuan_per_mwh") or {}).get("p50")) or 0) - (_num((calibrated.get("day_ahead_price_yuan_per_mwh") or {}).get("p50")) or 0)) * result["day_ahead_quantity_mwh"], 4)
    result["expected_saving_yuan"] = result["expected_cost_advantage_yuan"]
    actual_da = _num((kwargs.get("actual") or {}).get("dayAheadPriceYuanMwh"))
    actual_rt = _num((kwargs.get("actual") or {}).get("realtimePriceYuanMwh"))
    if actual_da is not None and actual_rt is not None:
        result["backtest_actual_prices"] = {"day_ahead_price_yuan_per_mwh": actual_da,
                                             "real_time_price_yuan_per_mwh": actual_rt}
        result["actual_cost_advantage_yuan"] = round((actual_rt - actual_da) * result["day_ahead_quantity_mwh"], 4)
        result["actual_saving_yuan"] = result["actual_cost_advantage_yuan"]
    result["optimization"] = {**optimisation, "scenario_probability_rt_gt_da": round(probability, 6),
                               "scenario_method": "55%相似日 + 25%近期误差 + 20%极端尾部；均为目标日前历史残差",
                               "raw_ratio": round(raw_ratio, 6), "confidence_adjusted_ratio": round(ratio, 6)}
    result["scenario_audit"] = audit; result["quantile_calibration"] = calibration; result["spread_prediction"] = spread_prediction
    result["risk_regime"] = "ADVANCED_HYBRID"
    result["regime_signals"] = list(dict.fromkeys([*result.get("regime_signals", []), *regime_signals]))
    result["risk_flags"] = list(dict.fromkeys([*result.get("risk_flags", []), "QUANTILE_CALIBRATED", "DIRECT_SPREAD_RESIDUAL", "NEWSVENDOR_REGRET", "CONFIDENCE_SHRINKAGE", *regime_signals]))
    result["strategy_version"] = ADVANCED_VERSION
    result["reason"] = ";".join([result.get("reason", ""), "HYBRID_SCENARIO_CVAR", "DIRECT_SPREAD_PREDICTION", "NEWSVENDOR_BALANCE", "CONFIDENCE_SHRINKAGE"]).strip(";")
    return result


def _joint_cvar(costs: list[tuple[float, float]], confidence: float = .95) -> float:
    value = _weighted_cvar(costs, confidence)
    return 0.0 if value is None else value


def apply_joint_daily_cvar(records: list[dict[str, Any]], *, risk_aversion: float) -> dict[str, Any]:
    """Apply a transparent daily ratio shift using shared ranked price paths."""
    usable = [row for row in records if row.get("strategy", {}).get("action") == "BUY_SPLIT"
              and row["strategy"].get("scenario_audit", {}).get("scenarios")]
    if len(records) != 24 or len(usable) != 24 or {row.get("period") for row in records} != set(range(1, 25)):
        return {"status": "BLOCKED", "reason": "JOINT_CVAR_REQUIRES_24_READY_PERIODS"}
    path_count = max(4, min(24, max(len(row["strategy"]["scenario_audit"]["scenarios"]) for row in usable)))
    paths = []
    for index in range(path_count):
        probability = (index + .5) / path_count
        scene_rows = []
        for row in usable:
            scenes = row["strategy"]["scenario_audit"]["scenarios"]
            base_ratio = float(row["strategy"].get("lock_ratio", .5))
            ordered = sorted(scenes, key=lambda scene: base_ratio * scene["day_ahead_price"] + (1 - base_ratio) * scene["real_time_price"])
            cumulative = 0.0
            scene = ordered[-1]
            for candidate in ordered:
                cumulative += candidate["weight"]
                if cumulative >= probability - 1e-12:
                    scene = candidate
                    break
            scene_rows.append(scene)
        paths.append((scene_rows, 1.0 / path_count))
    evaluations = []
    for delta in CONFIG["joint_delta_grid"]:
        costs = []
        for scenes, weight in paths:
            day_cost = 0.0
            for row, scene in zip(usable, scenes):
                base = float(row["strategy"].get("lock_ratio", .5))
                floor = float(row["strategy"].get("lock_ratio_floor", CONFIG["candidate_min_ratio"]))
                ceiling = float(row["strategy"].get("lock_ratio_ceiling", CONFIG["candidate_max_ratio"]))
                ratio = max(floor, min(ceiling, base + delta))
                exposure = _num(row["strategy"].get("remaining_exposure_mwh")) or 0.0
                day_cost += exposure * (ratio * scene["day_ahead_price"] + (1 - ratio) * scene["real_time_price"])
            costs.append((day_cost, weight))
        expected = _weighted_mean(costs); cvar = _joint_cvar(costs)
        objective = (1 - risk_aversion) * expected + risk_aversion * cvar
        evaluations.append({"delta": delta, "expected_cost_yuan": round(expected, 6), "cvar_cost_yuan": round(cvar, 6), "objective_yuan": round(objective, 6)})
    selected = min(evaluations, key=lambda x: (x["objective_yuan"], abs(x["delta"])))
    for row in usable:
        strategy = row["strategy"]; remaining = _num(strategy.get("remaining_exposure_mwh")) or 0.0
        floor = float(strategy.get("lock_ratio_floor", CONFIG["candidate_min_ratio"]))
        ceiling = float(strategy.get("lock_ratio_ceiling", CONFIG["candidate_max_ratio"]))
        ratio = max(floor, min(ceiling, float(strategy.get("lock_ratio", .5)) + selected["delta"]))
        strategy["lock_ratio_before_joint_cvar"] = strategy.get("lock_ratio")
        strategy["lock_ratio"] = round(ratio, 6)
        strategy["day_ahead_quantity_mwh"] = round(remaining * ratio, 6)
        strategy["real_time_reserved_mwh"] = round(remaining - strategy["day_ahead_quantity_mwh"], 6)
        predicted_da = _num(strategy.get("day_ahead_price_yuan_per_mwh"))
        predicted_rt = _num(strategy.get("real_time_price_p50_yuan_per_mwh"))
        if predicted_da is not None and predicted_rt is not None:
            strategy["expected_cost_advantage_yuan"] = round((predicted_rt - predicted_da) * strategy["day_ahead_quantity_mwh"], 4)
            strategy["expected_saving_yuan"] = strategy["expected_cost_advantage_yuan"]
        actual_da = _num((strategy.get("backtest_actual_prices") or {}).get("day_ahead_price_yuan_per_mwh"))
        actual_rt = _num((strategy.get("backtest_actual_prices") or {}).get("real_time_price_yuan_per_mwh"))
        if actual_da is not None and actual_rt is not None:
            strategy["actual_cost_advantage_yuan"] = round((actual_rt - actual_da) * strategy["day_ahead_quantity_mwh"], 4)
            strategy["actual_saving_yuan"] = strategy["actual_cost_advantage_yuan"]
        strategy["risk_flags"] = list(dict.fromkeys([*strategy.get("risk_flags", []), "JOINT_24H_CVAR"])); strategy["reason"] += ";JOINT_24H_CVAR"
    return {"status": "READY", "version": JOINT_VERSION, "path_count": path_count, "selected_delta": selected["delta"],
            "expected_cost_yuan": selected["expected_cost_yuan"], "cvar_cost_yuan": selected["cvar_cost_yuan"],
            "objective_yuan": selected["objective_yuan"], "candidate_evaluations": evaluations,
            "path_method": "按各小时基础组合成本的加权分位数进行同秩耦合，构造24条等权24小时价格路径；这是保守相关性假设，不冒充真实联合分布"}
