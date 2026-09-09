"""Forecast-conditioned, paired residual scenarios; research only.

Ranking never uses the target's realised price/load/supply. Parameters below
are fixed research settings, not fitted on the reported evaluation window.
"""
from __future__ import annotations

import math
from datetime import date
from statistics import median
from typing import Any

from .bidding_strategy import _num, generate_strategy

STRATEGY_KEY = "similar_day_cvar_v04"
VERSION = "da-rt-split-similar-day-cvar-v0.4"
CONFIG = {
    "lookback_calendar_days": 90, "outcome_lag_days": 2,
    "max_neighbors": 20, "min_neighbors": 10,
    "min_effective_samples": 8.0, "max_distance": 2.0,
    "max_newest_age_days": 14, "recency_half_life_days": 30,
    "interval_scale_floor_yuan_mwh": 10.0,
    "median_anchor": False,
}
PRICE_FEATURES = ("da_p50", "rt_p50", "rt_minus_da", "da_lower_width",
                  "da_upper_width", "rt_lower_width", "rt_upper_width")
PRICE_WEIGHTS = (.20, .20, .30, .075, .075, .075, .075)
WEATHER_FEATURES = ("temperature2mC", "windSpeed100mMs", "shortwaveRadiationGhiWm2", "cloudCoverPct")
WEATHER_FLOORS = (2.0, 1.0, 100.0, 10.0)


def _quantiles(row, key):
    values = [_num((row.get(key) or {}).get(q)) for q in ("p10", "p50", "p90")]
    return values if None not in values and values == sorted(values) else None


def _prices(row):
    da = _quantiles(row, "day_ahead_price_yuan_per_mwh")
    rt = _quantiles(row, "real_time_price_yuan_per_mwh")
    if da is None or rt is None:
        return None
    return [da[1], rt[1], rt[1] - da[1], da[1] - da[0], da[2] - da[1], rt[1] - rt[0], rt[2] - rt[1]]


def _weighted_quantile(values, probability):
    cumulative = 0.0
    for value, weight in sorted(values):
        cumulative += weight
        if cumulative >= probability - 1e-12:
            return value
    return max(value for value, _ in values)


def build_similar_day_scenarios(*, forecast_doc: dict[str, Any], target_date: str,
                                period: int, target_forecast: dict[str, Any],
                                weather_doc: dict[str, Any] | None = None,
                                target_model_version: str | None = None) -> dict[str, Any]:
    target = date.fromisoformat(target_date)
    feature = _prices(target_forecast)
    history_version = (forecast_doc.get("model") or {}).get("version")
    audit: dict[str, Any] = {
        "version": VERSION, "status": "BLOCKED", "config": dict(CONFIG),
        "selection_basis": "PRE_MARKET_PRICE_FORECAST", "blockers": [],
        "warnings": ["FORECAST_PUBLICATION_TIMES_NOT_FULLY_AUDITED", "TAIL_SAMPLE_LIMITED"],
        "candidate_count": 0, "selected_count": 0, "effective_sample_size": 0.0,
        "historical_model_version": history_version, "target_model_version": target_model_version,
        "target_date": target_date, "period": period, "scenarios": [],
        "weight_method": "exp(-distance^2/2) * 2^(-age_days/30), normalized",
        "price_transform": "target P50 + paired interval-scaled historical residual (conditional bias retained)",
        "weather_source": None,
    }
    if feature is None:
        audit["blockers"].append("SIMILAR_DAY_TARGET_QUANTILES_MISSING")
        return audit
    if not history_version or not target_model_version or history_version != target_model_version:
        audit["blockers"].append("SIMILAR_DAY_MODEL_VERSION_UNVERIFIED")
        return audit
    weather = weather_doc or {}
    confirmed = (weather.get("knownBeforeDeclaration") is True
                 and (weather.get("availabilityConfirmation") or {}).get("status") == "CONFIRMED"
                 and weather.get("sourceType") == "hourly-weather-forecast")
    weather_by = {(r.get("marketDate"), r.get("period")): r for r in weather.get("rows", [])} if confirmed else {}
    def weather_values(day):
        row = weather_by.get((day, period), {})
        values = [_num(row.get(key)) for key in WEATHER_FEATURES]
        return values if None not in values else None
    target_weather = weather_values(target_date)
    if target_weather is not None:
        audit["selection_basis"] = "PRE_MARKET_PRICE_AND_BUSINESS_CONFIRMED_WEATHER_FORECAST"
        audit["weather_source"] = {"version": weather.get("dataVersion"),
            "availability_basis": "BUSINESS_CONFIRMED_PRE_DECLARATION",
            "precise_issue_time_available": weather.get("forecastIssueTimeAvailable") is True}
    else:
        audit["warnings"].append("SIMILAR_DAY_WEATHER_UNAVAILABLE_PRICE_ONLY")
    candidates = []
    seen_dates = set()
    for day in forecast_doc.get("results", []):
        try:
            stamp = date.fromisoformat(day.get("market_date", ""))
        except (ValueError, TypeError):
            continue
        age = (target - stamp).days
        if not CONFIG["outcome_lag_days"] <= age <= CONFIG["lookback_calendar_days"]:
            continue
        if stamp in seen_dates:
            audit["blockers"].append("SIMILAR_DAY_DUPLICATE_HISTORY_DATE")
            return audit
        seen_dates.add(stamp)
        if (day.get("forecast_scenario") or day.get("scenario") or "pre_market") not in {"pre_market", "PRE_MARKET"}:
            continue
        if (day.get("model") or {}).get("version", history_version) != target_model_version:
            continue
        row = next((r for r in day.get("periods", []) if r.get("period") == period), {})
        values = _prices(row)
        actual_da = _num(row.get("actual_day_ahead_price_yuan_per_mwh"))
        actual_rt = _num(row.get("actual_real_time_price_yuan_per_mwh"))
        day_weather = weather_values(stamp.isoformat())
        if values is None or actual_da is None or actual_rt is None or (target_weather is not None and day_weather is None):
            continue
        candidates.append({"date": stamp.isoformat(), "age": age, "features": values,
            "weather": day_weather, "row": row, "actual_da": actual_da, "actual_rt": actual_rt,
            "weekend_mismatch": int((stamp.weekday() >= 5) != (target.weekday() >= 5))})
    audit["candidate_count"] = len(candidates)
    if not candidates:
        audit["blockers"].append("SIMILAR_DAY_HISTORY_INSUFFICIENT")
        return audit
    def robust_scale(values, floor):
        center = median(values)
        return max(floor, 1.4826 * median(abs(v - center) for v in values))
    scales = [robust_scale([r["features"][j] for r in candidates], 25.0) for j in range(len(feature))]
    weather_scales = [robust_scale([r["weather"][j] for r in candidates], floor) for j, floor in enumerate(WEATHER_FLOORS)] if target_weather is not None else []
    price_share = .70 if target_weather is not None else .95
    audit["feature_weights"] = {key: price_share * weight for key, weight in zip(PRICE_FEATURES, PRICE_WEIGHTS)}
    audit["feature_weights"]["weekend_mismatch"] = .05
    audit["feature_scales"] = dict(zip(PRICE_FEATURES, scales))
    if target_weather is not None:
        audit["feature_weights"].update({key: .25 / len(WEATHER_FEATURES) for key in WEATHER_FEATURES})
        audit["feature_scales"].update(dict(zip(WEATHER_FEATURES, weather_scales)))
    audit["target_features"] = dict(zip(PRICE_FEATURES, feature))
    if target_weather is not None:
        audit["target_features"].update(dict(zip(WEATHER_FEATURES, target_weather)))
    for item in candidates:
        distance2 = price_share * sum(w * ((x - y) / scale) ** 2 for x, y, scale, w in zip(feature, item["features"], scales, PRICE_WEIGHTS))
        distance2 += .05 * item["weekend_mismatch"]
        if target_weather is not None:
            distance2 += .25 * sum(((x - y) / scale) ** 2 for x, y, scale in zip(target_weather, item["weather"], weather_scales)) / len(WEATHER_FEATURES)
        item["distance"] = math.sqrt(distance2)
    selected = sorted((r for r in candidates if r["distance"] <= CONFIG["max_distance"]), key=lambda r: (r["distance"], r["age"]))[:CONFIG["max_neighbors"]]
    audit["selected_count"] = len(selected)
    if not selected:
        audit["blockers"].append("SIMILAR_DAY_OUT_OF_DISTRIBUTION")
        return audit
    weights = [math.exp(-r["distance"] ** 2 / 2) * 2 ** (-r["age"] / CONFIG["recency_half_life_days"]) for r in selected]
    total = sum(weights)
    weights = [w / total for w in weights]
    effective = 1 / sum(w * w for w in weights)
    audit.update(effective_sample_size=round(effective, 6), newest_age_days=min(r["age"] for r in selected),
                 mean_distance=sum(w * r["distance"] for w, r in zip(weights, selected)))
    if len(selected) < CONFIG["min_neighbors"]:
        audit["blockers"].append("SIMILAR_DAY_HISTORY_INSUFFICIENT")
    if effective < CONFIG["min_effective_samples"]:
        audit["blockers"].append("SIMILAR_DAY_EFFECTIVE_SAMPLE_INSUFFICIENT")
    if audit["newest_age_days"] > CONFIG["max_newest_age_days"]:
        audit["blockers"].append("SIMILAR_DAY_HISTORY_STALE")
    scaled = {"da": [], "rt": []}
    for item, weight in zip(selected, weights):
        for market, key in (("da", "day_ahead_price_yuan_per_mwh"), ("rt", "real_time_price_yuan_per_mwh")):
            old = _quantiles(item["row"], key)
            new = _quantiles(target_forecast, key)
            error = item[f"actual_{market}"] - old[1]
            side = 0 if error < 0 else 2
            old_width = max(CONFIG["interval_scale_floor_yuan_mwh"], abs(old[side] - old[1]))
            new_width = max(CONFIG["interval_scale_floor_yuan_mwh"], abs(new[side] - new[1]))
            item[f"raw_{market}_error"] = error
            item[f"scaled_{market}_error"] = error / old_width * new_width
            scaled[market].append((item[f"scaled_{market}_error"], weight))
    medians = {market: _weighted_quantile(values, .5) for market, values in scaled.items()}
    # A calibrated target P50 cannot be assumed: empirical conditional bias is
    # retained. Forced median anchoring remains an offline ablation only.
    shifts = {market: 0.0 for market in medians}
    audit["conditional_error_median_yuan_mwh"] = medians
    audit["median_adjustment_yuan_mwh"] = shifts
    for item, weight in zip(selected, weights):
        audit["scenarios"].append({"name": f"SIMILAR_{item['date']}", "source_date": item["date"],
            "source_period": period, "distance": item["distance"], "age_days": item["age"], "weight": weight,
            "day_ahead_price": feature[0] + item["scaled_da_error"] - shifts["da"],
            "real_time_price": feature[1] + item["scaled_rt_error"] - shifts["rt"],
            "paired_raw_errors": {"da": item["raw_da_error"], "rt": item["raw_rt_error"]},
            "paired_scaled_errors": {"da": item["scaled_da_error"], "rt": item["scaled_rt_error"]},
            "selection_features": dict(zip(PRICE_FEATURES, item["features"])),
            "weather_features": dict(zip(WEATHER_FEATURES, item["weather"])) if target_weather is not None else None})
    if not audit["blockers"]:
        audit["status"] = "READY"
    return audit


def generate_similar_day_strategy(*, scenario_audit: dict[str, Any], **kwargs) -> dict[str, Any]:
    """No grid fallback: insufficient conditional scenarios must remain HOLD."""
    if scenario_audit["status"] != "READY":
        # Reuse the electricity checks without allowing an invented scenario.
        result = generate_strategy(**{**kwargs, "forecast": {}})
        genuine_input_gaps = [key for key in ("load_forecast_mwh", "medium_position_mwh", "cleared_energy_mwh", "OVER_COVERED_REVIEW") if key in result.get("reason", "").split(";")]
        result["reason"] = ";".join([*genuine_input_gaps, *scenario_audit["blockers"]])
        result["scenario_audit"] = scenario_audit
        result["risk_flags"] = list(dict.fromkeys([*result.get("risk_flags", []), *scenario_audit["warnings"], *scenario_audit["blockers"]]))
        return result
    result = generate_strategy(**kwargs, historical_scenarios=scenario_audit["scenarios"])
    result["scenario_audit"] = scenario_audit
    result["risk_flags"] = list(dict.fromkeys([*result.get("risk_flags", []), *scenario_audit["warnings"]]))
    optimization = result.get("optimization")
    if optimization:
        optimization["scenario_method"] = "target forecast plus similar-day paired interval-scaled residuals, conditional bias retained"
        scenes = scenario_audit["scenarios"]
        optimization["scenario_probability_rt_gt_da"] = sum(s["weight"] for s in scenes if s["real_time_price"] > s["day_ahead_price"])
        optimization["scenario_quantiles"] = {market: {q: _weighted_quantile([(s[key], s["weight"]) for s in scenes], prob) for q, prob in (("p10", .1), ("p50", .5), ("p90", .9))} for market, key in (("day_ahead", "day_ahead_price"), ("real_time", "real_time_price"))}
        # Sensitivity to the discrete candidate grid; never silently overrides r*.
        near = [c["ratio"] for c in optimization["candidate_evaluations"] if c["objective"] - optimization["objective"] <= 1.0]
        optimization["near_optimal_ratio_range"] = [min(near), max(near)]
        optimization["near_optimal_tolerance_yuan_mwh"] = 1.0
        result["reason"] += ";SIMILAR_DAY_FORECAST_CONDITIONED"
    return result
