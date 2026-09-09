from __future__ import annotations

import csv
import io
import json
import math
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any


DEFAULT_STRATEGY_VERSION = "historical_cvar_v02"
DEFAULT_RISK_AVERSION = 0.3
SUPPORTED_STRATEGY_VERSIONS = {
    DEFAULT_STRATEGY_VERSION,
    "regime_cvar_v03",
    "legacy_spread_v01",
}


class InputValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def strict_period(value: Any) -> int | None:
    if type(value) is int:
        return value if 1 <= value <= 24 else None
    if isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        number = int(value.strip())
        return number if 1 <= number <= 24 else None
    return None


def ensure_finite_output(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise InputValidationError(["计算结果超出有限数值范围，未生成可审核草稿"])
    if isinstance(value, dict):
        for item in value.values():
            ensure_finite_output(item)
    elif isinstance(value, list):
        for item in value:
            ensure_finite_output(item)


def parse_input(payload: bytes, content_type: str = "") -> list[dict[str, Any]]:
    text = payload.decode("utf-8-sig")
    if "json" in content_type or text.lstrip().startswith(("[", "{")):
        value = json.loads(text)
        rows = value.get("records", value) if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise ValueError("输入必须是记录数组")
        return rows
    return list(csv.DictReader(io.StringIO(text)))


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_rows(rows: list[dict[str, Any]], business_date: str, market_code: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(rows, list):
        return ["records 必须是记录数组"]
    try:
        if date.fromisoformat(business_date).isoformat() != business_date:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("business_date 必须为 YYYY-MM-DD 日期")
    if market_code != "SD":
        errors.append("首版仅支持 SD 市场")
    periods = []
    versions: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"第 {index + 1} 行必须是对象")
            continue
        period = strict_period(row.get("period"))
        if period is None:
            errors.append(f"第 {index + 1} 行 period 必须为 1-24 整数")
        else:
            periods.append(period)
        if row.get("business_date") != business_date:
            errors.append(f"period {period} 日期缺失或不一致")
        if row.get("market_code") != market_code:
            errors.append(f"period {period} 市场缺失或不一致")
        version = row.get("data_version")
        if not isinstance(version, str) or not version.strip():
            errors.append(f"period {period} data_version 必填")
        else:
            versions.add(version.strip())
        # The existing *_mwh columns declare MWh; an explicit unit cannot override it.
        for unit_field in ("unit", "energy_unit"):
            if unit_field in row and row[unit_field] != "MWh":
                errors.append(f"period {period} {unit_field} 必须为 MWh")
        for field in (
            "load_forecast_mwh",
            "medium_position_mwh",
            "cleared_energy_mwh",
            "adjustable_min_mwh",
            "adjustable_max_mwh",
        ):
            if row.get(field) not in (None, ""):
                try:
                    value = _num(row.get(field))
                    if value is None:
                        raise ValueError
                    if value < 0 and field != "medium_position_mwh":
                        errors.append(f"period {period} {field} 不得为负数")
                except (TypeError, ValueError):
                    errors.append(f"period {period} {field} 必须是有限数字")
        try:
            lower = _num(row.get("adjustable_min_mwh"))
            upper = _num(row.get("adjustable_max_mwh"))
            if lower is not None and upper is not None and lower > upper:
                errors.append(f"period {period} 调整下限不能大于上限")
        except (TypeError, ValueError):
            pass
    if len(versions) > 1:
        errors.append("同一批 records 的 data_version 必须一致")
    for row in rows:
        if not isinstance(row, dict):
            continue
        values = [_num(row.get(key)) for key in ("load_forecast_mwh", "medium_position_mwh", "cleared_energy_mwh")]
        if all(value is not None for value in values) and not math.isfinite(values[0] - values[1] - values[2]):
            errors.append(f"period {row.get('period')} 敞口计算超出有限数值范围")
    if len(rows) != 24:
        errors.append(f"必须提供24条记录，当前 {len(rows)} 条")
    if len(set(periods)) != len(periods):
        errors.append("period 不得重复")
    if set(periods) != set(range(1, 25)):
        errors.append("period 必须覆盖1-24")
    return errors


def validate_forecasts(forecasts: Any, business_date: str) -> list[str]:
    if forecasts is None:
        return []
    if not isinstance(forecasts, list):
        return ["forecasts 必须是记录数组"]
    errors: list[str] = []
    periods: set[int] = set()
    for item in forecasts:
        if not isinstance(item, dict):
            errors.append("forecasts 每条记录必须是对象")
            continue
        period = strict_period(item.get("period"))
        if period is None or period in periods:
            errors.append("forecasts period 必须为 1-24 且不可重复")
        else:
            periods.add(period)
        if "business_date" in item and item["business_date"] != business_date:
            errors.append(f"forecast period {period} 日期不一致")
        if "market_code" in item and item["market_code"] != "SD":
            errors.append(f"forecast period {period} 市场不一致")
    return errors


def _nested_quantiles(forecast: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        value = forecast.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _flat_quantile(forecast: dict[str, Any], prefixes: tuple[str, ...], level: str) -> float | None:
    for prefix in prefixes:
        value = _num(forecast.get(f"{prefix}_{level}"))
        if value is not None:
            return value
    return None


def _forecast_quantiles(forecast: dict[str, Any], market: str) -> dict[str, float | None]:
    if market == "day_ahead":
        nested_names = (
            "day_ahead_price_yuan_per_mwh",
            "day_ahead_price",
            "day_ahead",
            "da",
        )
        prefixes = ("day_ahead_price", "day_ahead", "da_price", "da")
    else:
        nested_names = (
            "real_time_price_yuan_per_mwh",
            "real_time_price",
            "realtime_price",
            "real_time",
            "realtime",
            "rt",
        )
        prefixes = (
            "real_time_price",
            "realtime_price",
            "real_time",
            "realtime",
            "rt_price",
            "rt",
        )
    nested = _nested_quantiles(forecast, nested_names)
    result = {
        level: _num(nested.get(level))
        if nested
        else _flat_quantile(forecast, prefixes, level)
        for level in ("p10", "p50", "p90")
    }
    # The original draft contract exposed one price interval. Keep it as a
    # day-ahead-only alias; treating it as both markets would invent an RT view.
    if market == "day_ahead" and not nested:
        for level in ("p10", "p50", "p90"):
            if result[level] is None:
                result[level] = _num(forecast.get(f"price_{level}"))
    return result


def _valid_quantiles(values: dict[str, float | None]) -> bool:
    p10, p50, p90 = values["p10"], values["p50"], values["p90"]
    return None not in (p10, p50, p90) and p10 <= p50 <= p90


def _rules_status(rules: dict[str, Any]) -> tuple[bool, float | None, float | None]:
    floor = _num(rules.get("price_floor"))
    ceiling = _num(rules.get("price_ceiling"))
    version = str(rules.get("rule_version") or "").strip()
    policy_citation = str(rules.get("policy_citation") or "").strip()
    status = str(rules.get("status") or rules.get("confirmation_status") or "").upper()
    explicitly_unconfirmed = rules.get("confirmed") is not True or status in {
        "UNCONFIRMED",
        "RULES_UNCONFIRMED",
        "DRAFT",
    }
    confirmed = (
        bool(version)
        and bool(policy_citation)
        and floor is not None
        and ceiling is not None
        and floor <= ceiling
    )
    return confirmed and not explicitly_unconfirmed, floor, ceiling


def _quantile_scenarios(
    day_ahead: dict[str, float | None], real_time: dict[str, float | None]
) -> list[dict[str, Any]]:
    levels = (("p10", 0.10), ("p50", 0.80), ("p90", 0.10))
    return [
        {
            "name": f"DA_{da_level.upper()}_RT_{rt_level.upper()}",
            "day_ahead_price": day_ahead[da_level],
            "real_time_price": real_time[rt_level],
            "weight": da_weight * rt_weight,
        }
        for da_level, da_weight in levels
        for rt_level, rt_weight in levels
    ]


def _supplied_scenarios(
    forecast: dict[str, Any], business_date: str
) -> tuple[list[dict[str, Any]], int]:
    raw = (
        forecast.get("historical_scenarios")
        or forecast.get("price_scenarios")
        or forecast.get("scenarios")
        or []
    )
    if not isinstance(raw, list):
        return [], 0
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for index, scenario in enumerate(raw):
        if not isinstance(scenario, dict):
            rejected += 1
            continue
        # A target-day or future-dated scenario is excluded so realised future
        # data cannot silently leak into a decision.
        source_date = str(
            scenario.get("scenario_date")
            or scenario.get("observed_date")
            or scenario.get("market_date")
            or ""
        )
        try:
            valid_date = date.fromisoformat(source_date).isoformat() == source_date
        except (TypeError, ValueError):
            valid_date = False
        if not valid_date or source_date >= business_date:
            rejected += 1
            continue
        day_ahead_price = _num(
            scenario.get("day_ahead_price", scenario.get("day_ahead_price_yuan_per_mwh"))
        )
        real_time_price = _num(
            scenario.get("real_time_price", scenario.get("real_time_price_yuan_per_mwh"))
        )
        weight = _num(scenario.get("weight"))
        if day_ahead_price is None or real_time_price is None or weight is None or weight <= 0:
            rejected += 1
            continue
        accepted.append(
            {
                "name": str(scenario.get("name") or f"SNAPSHOT_{index + 1}"),
                "day_ahead_price": day_ahead_price,
                "real_time_price": real_time_price,
                "weight": weight,
                "scenario_date": source_date or None,
            }
        )
    return accepted, rejected


def _weighted_cvar(losses: list[tuple[float, float]], confidence: float = 0.95) -> float:
    tail_mass = max(1e-9, 1.0 - confidence)
    remaining = tail_mass
    weighted_loss = 0.0
    for loss, weight in sorted(losses, key=lambda item: item[0], reverse=True):
        take = min(weight, remaining)
        weighted_loss += loss * take
        remaining -= take
        if remaining <= 1e-12:
            break
    return weighted_loss / tail_mass


def evaluate_allocation(
    scenarios: list[dict[str, Any]], ratio: float, risk_aversion: float,
    confidence: float = 0.95,
) -> dict[str, float]:
    # Scale first to avoid overflow when individually finite weights are very large.
    scale = max(float(item["weight"]) for item in scenarios)
    total = sum(float(item["weight"]) / scale for item in scenarios)
    losses = [
        (ratio * float(item["day_ahead_price"]) + (1 - ratio) * float(item["real_time_price"]),
         (float(item["weight"]) / scale) / total)
        for item in scenarios
    ]
    expected = sum(loss * weight for loss, weight in losses)
    cvar = _weighted_cvar(losses, confidence)
    return {
        "expected_cost_yuan_per_mwh": round(expected, 6),
        "cvar_cost_yuan_per_mwh": round(cvar, 6),
        "objective_yuan_per_mwh": round((1 - risk_aversion) * expected + risk_aversion * cvar, 6),
    }


def _optimize_lock_ratio(
    scenarios: list[dict[str, Any]],
    risk_aversion: float,
    confidence: float = 0.95,
    minimum_ratio: float = 0.05,
    maximum_ratio: float = 0.95,
) -> dict[str, Any]:
    candidates: list[dict[str, float]] = []
    # The bounded grid matches the platform research backtest and prevents an
    # unaudited all-in recommendation.
    candidate_count = int(round((maximum_ratio - minimum_ratio) / 0.05)) + 1
    for index in range(candidate_count):
        ratio = round(minimum_ratio + index * 0.05, 6)
        candidates.append(
            {
                "ratio": ratio,
                **evaluate_allocation(scenarios, ratio, risk_aversion, confidence),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (item["objective_yuan_per_mwh"], abs(item["ratio"] - 0.5)),
    )
    return {
        "lock_ratio": selected["ratio"],
        "expected_cost_yuan_per_mwh": selected["expected_cost_yuan_per_mwh"],
        "cvar_cost_yuan_per_mwh": selected["cvar_cost_yuan_per_mwh"],
        "objective_yuan_per_mwh": selected["objective_yuan_per_mwh"],
        "confidence": confidence,
        "candidate_evaluations": candidates,
    }


def _legacy_lock_ratio(
    day_ahead: dict[str, float | None],
    real_time: dict[str, float | None],
    risk_aversion: float,
) -> tuple[float, list[str]]:
    da_p50 = float(day_ahead["p50"])
    rt_p10 = float(real_time["p10"])
    rt_p50 = float(real_time["p50"])
    rt_p90 = float(real_time["p90"])
    raw_edge = rt_p50 - da_p50
    downside = max(0.0, da_p50 - rt_p10)
    risk_adjusted_edge = raw_edge - risk_aversion * downside
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
    reasons = [reason, "LEGACY_TRANSPARENT_SPREAD_THRESHOLDS"]
    if rt_p90 - rt_p10 >= 250:
        ratio = min(ratio, 0.50)
        reasons.append("WIDE_RT_INTERVAL")
    return ratio, reasons


def build_draft(
    rows: list[dict[str, Any]],
    forecasts: list[dict[str, Any]] | None,
    rules: dict[str, Any] | None,
    *,
    business_date: str,
    market_code: str = "SD",
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    scenario_source: str | None = None,
    scenario_version: str | None = None,
) -> dict[str, Any]:
    errors = validate_rows(rows, business_date, market_code)
    errors.extend(validate_forecasts(forecasts, business_date))
    if rules is not None and not isinstance(rules, dict):
        errors.append("rules 必须是对象")
    if errors:
        raise InputValidationError(errors)
    if strategy_version not in SUPPORTED_STRATEGY_VERSIONS:
        raise ValueError(f"不支持的策略版本：{strategy_version}")
    if isinstance(risk_aversion, bool) or not 0 <= float(risk_aversion) <= 1:
        raise ValueError("risk_aversion 必须在 0 到 1 之间")
    risk_aversion = float(risk_aversion)
    forecast_by = {
        strict_period(item["period"]): item
        for item in (forecasts or [])
    }
    rules = rules or {}
    formal_rules, price_floor, price_ceiling = _rules_status(rules)
    rows_by = {
        strict_period(item["period"]): item
        for item in rows
    }
    result: list[dict[str, Any]] = []
    for period in range(1, 25):
        row = rows_by.get(period, {})
        forecast = forecast_by.get(period, {})
        load, medium, cleared = (
            _num(row.get(key))
            for key in ("load_forecast_mwh", "medium_position_mwh", "cleared_energy_mwh")
        )
        day_ahead = _forecast_quantiles(forecast, "day_ahead")
        real_time = _forecast_quantiles(forecast, "real_time")
        missing: list[str] = []
        if load is None:
            missing.append("load_forecast_mwh")
        if medium is None:
            missing.append("medium_position_mwh")
        if cleared is None:
            missing.append("cleared_energy_mwh")
        if not _valid_quantiles(day_ahead):
            missing.append("day_ahead_price_quantiles")
        if not _valid_quantiles(real_time):
            missing.append("real_time_price_quantiles")
        if not isinstance(forecast.get("forecast_version"), str) or not forecast["forecast_version"].strip():
            missing.append("forecast_version")
        if not formal_rules:
            missing.append("confirmed_rules")

        gross = (
            load - medium - cleared
            if load is not None and medium is not None and cleared is not None
            else None
        )
        remaining = max(0.0, gross) if gross is not None else None
        risk_flags: list[str] = []
        if medium is not None and medium < 0:
            risk_flags.append("NEGATIVE_POSITION_REVIEW")
        if gross is not None and gross < 0:
            risk_flags.append("OVER_COVERED_REVIEW")

        suggested_quantity = 0.0
        suggested_price_lower = None
        suggested_price_upper = None
        action = "HOLD"
        gate_status = "BLOCKED" if missing or "OVER_COVERED_REVIEW" in risk_flags else "PASSED"
        optimization: dict[str, Any] | None = None
        scenario_method = None
        record_scenario_source = forecast.get("scenario_source") or scenario_source
        record_scenario_version = (
            forecast.get("scenario_version")
            or forecast.get("scenario_data_version")
            or scenario_version
        )
        scenario_count = 0
        rejected_scenario_count = 0
        strategy_reasons: list[str] = []
        effective_risk_aversion = risk_aversion
        risk_regime = "NORMAL"
        effective_strategy_version = strategy_version
        formal_strategy_blockers: list[str] = []

        if gate_status == "PASSED":
            scenarios, rejected_scenario_count = _supplied_scenarios(forecast, business_date)
            if not scenarios:
                scenarios = _quantile_scenarios(day_ahead, real_time)
                scenario_method = "p10_p50_p90_grid_fallback"
                if strategy_version != "legacy_spread_v01":
                    risk_flags.append("HISTORICAL_SCENARIOS_UNAVAILABLE")
                if strategy_version == DEFAULT_STRATEGY_VERSION:
                    effective_strategy_version = "quantile_cvar_v02_fallback"
                    formal_strategy_blockers.append("HISTORICAL_SCENARIOS_REQUIRED")
            else:
                scenario_method = "uploaded_pre_target_scenario_snapshot"
            if rejected_scenario_count:
                risk_flags.append("FUTURE_OR_INVALID_SCENARIOS_EXCLUDED")
            scenario_count = len(scenarios)

            if strategy_version == DEFAULT_STRATEGY_VERSION and scenarios:
                if scenario_method == "uploaded_pre_target_scenario_snapshot":
                    if not record_scenario_source:
                        formal_strategy_blockers.append("SCENARIO_SOURCE_REQUIRED")
                    if not record_scenario_version:
                        formal_strategy_blockers.append("SCENARIO_VERSION_REQUIRED")
            elif strategy_version != DEFAULT_STRATEGY_VERSION:
                formal_strategy_blockers.append("CHALLENGER_RESEARCH_ONLY")

            if strategy_version == "legacy_spread_v01":
                ratio, strategy_reasons = _legacy_lock_ratio(
                    day_ahead, real_time, risk_aversion
                )
                audit_optimization = _optimize_lock_ratio(scenarios, risk_aversion)
                selected = next(
                    item
                    for item in audit_optimization["candidate_evaluations"]
                    if item["ratio"] == ratio
                )
                optimization = {
                    **audit_optimization,
                    "lock_ratio": ratio,
                    "expected_cost_yuan_per_mwh": selected[
                        "expected_cost_yuan_per_mwh"
                    ],
                    "cvar_cost_yuan_per_mwh": selected["cvar_cost_yuan_per_mwh"],
                    "objective_yuan_per_mwh": selected["objective_yuan_per_mwh"],
                    "selection_method": "transparent_spread_thresholds",
                }
                scenario_method = "not_used_for_legacy_selection"
            elif strategy_version == "regime_cvar_v03":
                supply_context = forecast.get("supply_context") or {}
                residual_demand = _num(
                    supply_context.get(
                        "residualDemandProxyMw",
                        supply_context.get("residual_demand_proxy_mw"),
                    )
                )
                maximum_ratio = 0.95
                if residual_demand is None:
                    risk_flags.append("REGIME_CONTEXT_UNAVAILABLE")
                    strategy_reasons.append("REGIME_FALLBACK_TO_BASE_CVAR")
                elif residual_demand <= 0 and float(real_time["p50"]) <= float(
                    day_ahead["p50"]
                ):
                    effective_risk_aversion = min(risk_aversion, 0.25)
                    maximum_ratio = 0.25
                    risk_regime = "MIDDAY_RENEWABLE_OVERSUPPLY"
                    strategy_reasons.append("KEEP_REALTIME_FLEXIBILITY")
                if (
                    17 <= period <= 21
                    and float(real_time["p90"]) - float(day_ahead["p90"]) >= 100
                ):
                    effective_risk_aversion = max(effective_risk_aversion, 0.65)
                    risk_regime = "EVENING_PEAK_TAIL_RISK"
                    strategy_reasons.append("CONTROL_REALTIME_SPIKE_TAIL")
                optimization = _optimize_lock_ratio(
                    scenarios,
                    effective_risk_aversion,
                    maximum_ratio=maximum_ratio,
                )
                strategy_reasons.append("REGIME_CVAR_OPTIMIZATION")
            else:
                optimization = _optimize_lock_ratio(scenarios, risk_aversion)
                strategy_reasons.append("HISTORICAL_SCENARIO_CVAR_OPTIMIZATION")

            minimum = _num(row.get("adjustable_min_mwh"))
            maximum = _num(row.get("adjustable_max_mwh"))
            minimum = 0.0 if minimum is None else minimum
            maximum = remaining if maximum is None else min(maximum, remaining)
            if minimum < 0 or maximum is None or maximum < 0 or minimum > maximum:
                missing.append("adjustable_quantity_bounds")
                gate_status = "BLOCKED"
            elif remaining == 0:
                action = "HOLD"
                gate_status = "PASSED"
            else:
                target_quantity = remaining * float(optimization["lock_ratio"])
                suggested_quantity = min(maximum, max(minimum, round(target_quantity, 6)))
                suggested_price_lower = max(float(day_ahead["p10"]), price_floor)
                suggested_price_upper = min(float(day_ahead["p90"]), price_ceiling)
                if suggested_price_lower > suggested_price_upper:
                    gate_status = "BLOCKED"
                    missing.append("day_ahead_price_range_after_rule_limits")
                    suggested_quantity = 0.0
                    suggested_price_lower = suggested_price_upper = None
                    risk_flags.append("PRICE_RANGE_INVALID")
                else:
                    action = "BUY_DRAFT" if suggested_quantity > 0 else "HOLD"

        if optimization and gate_status == "PASSED" and remaining:
            applied_ratio = suggested_quantity / remaining
            optimization = {
                **optimization,
                "applied_lock_ratio": applied_ratio,
                **evaluate_allocation(scenarios, applied_ratio, effective_risk_aversion),
            }

        reason = "; ".join(missing)
        if not reason:
            if gross is not None and gross < 0:
                reason = "超覆盖待人工核验"
            elif remaining == 0:
                reason = "无剩余敞口，无需申报"
            else:
                reason = "剩余敞口按日前/实时场景CVaR最优锁定比例分配"
        result.append(
            {
                "period": period,
                "load_forecast_mwh": load,
                "medium_position_mwh": medium,
                "cleared_energy_mwh": cleared,
                "remaining_exposure_mwh": remaining,
                # Compatibility aliases: these represent the day-ahead interval.
                "price_p10": day_ahead["p10"],
                "price_p50": day_ahead["p50"],
                "price_p90": day_ahead["p90"],
                "day_ahead_price_p10": day_ahead["p10"],
                "day_ahead_price_p50": day_ahead["p50"],
                "day_ahead_price_p90": day_ahead["p90"],
                "real_time_price_p10": real_time["p10"],
                "real_time_price_p50": real_time["p50"],
                "real_time_price_p90": real_time["p90"],
                "suggested_quantity_mwh": suggested_quantity,
                "real_time_reserved_mwh": (
                    round(remaining - suggested_quantity, 6)
                    if remaining is not None
                    else None
                ),
                "suggested_price_lower": suggested_price_lower,
                "suggested_price_upper": suggested_price_upper,
                "action": action,
                "gate_status": gate_status,
                "risk_flags": risk_flags,
                "reason": reason,
                "strategy_version": strategy_version,
                "effective_strategy_version": effective_strategy_version,
                "risk_aversion": risk_aversion,
                "effective_risk_aversion": effective_risk_aversion,
                "risk_regime": risk_regime,
                "strategy_reasons": strategy_reasons,
                "strategy_gate_status": "RESEARCH_ONLY",
                "formal_strategy_ready": not formal_strategy_blockers,
                "formal_strategy_blockers": formal_strategy_blockers,
                "lock_ratio": (
                    round(suggested_quantity / remaining, 6)
                    if remaining
                    else 0.0 if remaining == 0 else None
                ),
                "unconstrained_lock_ratio": (
                    optimization.get("lock_ratio") if optimization else None
                ),
                "expected_cost_yuan_per_mwh": (
                    optimization.get("expected_cost_yuan_per_mwh")
                    if optimization
                    else None
                ),
                "cvar_cost_yuan_per_mwh": (
                    optimization.get("cvar_cost_yuan_per_mwh")
                    if optimization
                    else None
                ),
                "objective_yuan_per_mwh": (
                    optimization.get("objective_yuan_per_mwh")
                    if optimization
                    else None
                ),
                "cvar_confidence": optimization.get("confidence") if optimization else None,
                "candidate_evaluations": (
                    optimization.get("candidate_evaluations") if optimization else []
                ),
                "optimization": optimization,
                "scenario_method": scenario_method,
                "scenario_fallback": scenario_method == "p10_p50_p90_grid_fallback",
                "scenario_source": record_scenario_source,
                "scenario_version": record_scenario_version,
                "scenario_count": scenario_count,
                "rejected_scenario_count": rejected_scenario_count,
                "data_version": row.get("data_version"),
                "forecast_version": forecast.get("forecast_version"),
                "rule_version": rules.get("rule_version"),
                "execution_allowed": False,
            }
        )

    strategy_formal_ready = all(item["formal_strategy_ready"] for item in result)
    formal_gate = "PASSED" if formal_rules and strategy_formal_ready and not any(
        item["gate_status"] == "BLOCKED" for item in result
    ) else "BLOCKED"
    effective_versions = sorted({item["effective_strategy_version"] for item in result})
    formal_strategy_blockers = sorted(
        {
            blocker
            for item in result
            for blocker in item["formal_strategy_blockers"]
        }
    )
    # Keep research allocations separate from gated, reviewable draft quantities.
    for item in result:
        item["research_quantity_mwh"] = item["suggested_quantity_mwh"]
        item["research_real_time_reserved_mwh"] = item["real_time_reserved_mwh"]
        item["research_lock_ratio"] = item["lock_ratio"]
        item["research_price_lower"] = item["suggested_price_lower"]
        item["research_price_upper"] = item["suggested_price_upper"]
        item["research_optimization"] = deepcopy(item["optimization"])
        if formal_gate == "BLOCKED":
            item["gate_status"] = "BLOCKED"
            item["action"] = "HOLD"
            item["suggested_quantity_mwh"] = 0.0
            item["suggested_price_lower"] = item["suggested_price_upper"] = None
            item["real_time_reserved_mwh"] = item["remaining_exposure_mwh"]
            item["lock_ratio"] = 0.0 if item["remaining_exposure_mwh"] is not None else None
            for key in ("expected_cost_yuan_per_mwh", "cvar_cost_yuan_per_mwh", "objective_yuan_per_mwh"):
                item[key] = None
            item["optimization"] = None
            item["risk_flags"].append("BATCH_GATE_BLOCKED")
            item["reason"] = "; ".join(["整批草稿门禁阻断", *item["formal_strategy_blockers"], item["reason"]])
    actionable = [item for item in result if item["action"] == "BUY_DRAFT"]
    draft = {
        "business_date": business_date,
        "market_code": market_code,
        "strategy_version": strategy_version,
        "effective_strategy_version": (
            effective_versions[0] if len(effective_versions) == 1 else "MIXED"
        ),
        "risk_aversion": risk_aversion,
        "scenario_source": scenario_source,
        "scenario_version": scenario_version,
        "strategy_status": "研究策略 / 人工复核草稿 / 不可自动提交",
        "strategy_gate_status": "RESEARCH_ONLY",
        "formal_strategy_ready": strategy_formal_ready,
        "formal_strategy_blockers": formal_strategy_blockers,
        "records": result,
        "summary": {
            "research_day_ahead_quantity_mwh": round(sum(item["research_quantity_mwh"] for item in result), 6),
            "suggested_day_ahead_quantity_mwh": round(
                sum(float(item["suggested_quantity_mwh"]) for item in actionable), 6
            ),
            "reserved_real_time_quantity_mwh": round(
                sum(float(item["real_time_reserved_mwh"] or 0) for item in result), 6
            ),
            "actionable_period_count": len(actionable),
            "blocked_period_count": sum(
                item["gate_status"] == "BLOCKED" for item in result
            ),
        },
        "execution_allowed": False,
        "formal_gate": formal_gate,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    ensure_finite_output(draft)
    return draft


def apply_draft_modifications(
    draft: dict[str, Any], snapshot: dict[str, Any], changes: list[dict[str, Any]],
    *, reason: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modified = deepcopy(draft)
    records = {row["period"]: row for row in modified["records"]}
    inputs = {strict_period(row["period"]): row for row in snapshot["records"]}
    forecasts = {strict_period(row["period"]): row for row in snapshot["forecasts"]}
    rules = snapshot["rules"]
    audit: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for change in changes:
        period = strict_period(change.get("period"))
        field = change.get("field")
        value = _num(change.get("value"))
        if period not in records or field not in {"suggested_quantity_mwh", "suggested_price_lower", "suggested_price_upper"} or value is None:
            raise ValueError("修改必须指定有效时段、量价字段和有限数值")
        if (period, field) in seen:
            raise ValueError("同一次修改不能重复指定同一时段和字段")
        seen.add((period, field))
        record = records[period]
        if modified.get("formal_gate") != "PASSED" or record["gate_status"] != "PASSED":
            raise ValueError("门禁阻断的草稿不得通过人工修改绕过校验")
        if field == "suggested_quantity_mwh":
            lower = _num(inputs[period].get("adjustable_min_mwh"))
            upper = _num(inputs[period].get("adjustable_max_mwh"))
            lower = 0.0 if lower is None else lower
            remaining = record["remaining_exposure_mwh"]
            upper = remaining if upper is None else min(upper, remaining)
            if not lower <= value <= upper:
                raise ValueError(f"period {period} 申报量超出原始调整上下限或剩余敞口")
        elif not float(rules["price_floor"]) <= value <= float(rules["price_ceiling"]):
            raise ValueError(f"period {period} 报价超出任务锁定的规则限价")
        audit.append({"period": period, "field": field, "old_value": record.get(field), "new_value": value})
        record[field] = value

    for period in {period for period, _ in seen}:
        record = records[period]
        if record["suggested_price_lower"] > record["suggested_price_upper"]:
            raise ValueError(f"period {period} 修改后报价下限不得高于上限")
        remaining, quantity = record["remaining_exposure_mwh"], record["suggested_quantity_mwh"]
        ratio = quantity / remaining if remaining else 0.0
        record["real_time_reserved_mwh"] = round(remaining - quantity, 6)
        record["lock_ratio"] = round(ratio, 6)
        record["action"] = "BUY_DRAFT" if quantity > 0 else "HOLD"
        scenarios, _ = _supplied_scenarios(forecasts[period], draft["business_date"])
        if not scenarios:
            raise ValueError("原始历史场景不可用，不能修改为可审核草稿")
        metrics = evaluate_allocation(scenarios, ratio, record["effective_risk_aversion"])
        record.update(metrics)
        record["optimization"] = {
            **(record["optimization"] or {}), **metrics,
            "applied_lock_ratio": ratio, "selection_method": "manual_review_adjustment",
        }
        record["reason"] = f"人工调整后重新计算分配及场景成本；原因：{reason}"
        record["strategy_reasons"] = ["MANUAL_REVIEW_ADJUSTMENT"]
    modified["summary"].update({
        "suggested_day_ahead_quantity_mwh": round(sum(row["suggested_quantity_mwh"] for row in records.values()), 6),
        "reserved_real_time_quantity_mwh": round(sum(row["real_time_reserved_mwh"] or 0 for row in records.values()), 6),
        "actionable_period_count": sum(row["action"] == "BUY_DRAFT" for row in records.values()),
    })
    ensure_finite_output(modified)
    return modified, audit
