"""Month-generic unit-exposure allocation research for the public API."""
from __future__ import annotations

import hashlib
import json

from .bidding_strategy import optimize_lock_ratio
from .day_path_cvar import build_day_paths, optimize_day_paths


VERSION = "sd-monthly-quantile-research-v1"


def _metric(periods, forecast_key, actual_key):
    errors = []
    for row in periods:
        forecast = (row.get(forecast_key) or {}).get("p50")
        actual = row.get(actual_key)
        if isinstance(forecast, (int, float)) and isinstance(actual, (int, float)):
            errors.append(float(forecast) - float(actual))
    if not errors:
        return {"mae": None, "rmse": None, "bias": None}
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": (sum(error * error for error in errors) / len(errors)) ** .5,
        "bias": sum(errors) / len(errors),
    }


def _day_metrics(periods):
    return {
        "da": _metric(periods, "day_ahead_price_yuan_per_mwh", "actual_day_ahead_price_yuan_per_mwh"),
        "rt": _metric(periods, "real_time_price_yuan_per_mwh", "actual_real_time_price_yuan_per_mwh"),
        "post_da": _metric(periods, "real_time_post_da_price_yuan_per_mwh", "actual_real_time_price_yuan_per_mwh"),
    }


def _scope(history):
    dates = sorted(day.get("market_date") for day in history.get("results", []) if day.get("market_date"))
    return dates, ({"start": dates[0], "end": dates[-1]} if dates else None)


def price_allocation(history, business_date):
    dates, research_scope = _scope(history)
    target = next((day for day in history.get("results", []) if day.get("market_date") == business_date), None)
    if not target:
        return {
            "market_date": business_date,
            "status": "BLOCKED",
            "reason": "FORECAST_SNAPSHOT_MISSING",
            "execution_allowed": False,
            "formal_gate": "BLOCKED",
            "model_version": VERSION,
            "research_scope": research_scope,
            "sample_count": 0,
            "required_data": [
                "目标日期24小时日前价格预测（P10/P50/P90）",
                "目标日期24小时盘前实时价格预测（P10/P50/P90）",
                "目标日前至少10个完整历史日的预测快照与实际日前/实时价格",
                "预测快照发布时间及D-2可用性证明",
                "生成实际申报MWh还需目标日期24小时客户负荷预测",
                "生成实际申报MWh还需24小时中长期持仓与已成交电量",
                "生成正式多段草稿还需有效山东申报规则、限价与步长",
            ],
            "periods": [],
        }

    scenarios = build_day_paths(
        forecast_doc=history,
        target_date=business_date,
        target_forecasts=target.get("periods", []),
    )
    paths = scenarios["paths"]
    joint = optimize_day_paths(paths, [1.0] * 24)
    if joint["status"] == "BLOCKED":
        return {
            "market_date": business_date,
            "status": "BLOCKED",
            "reason": joint["reason"],
            "execution_allowed": False,
            "formal_gate": "BLOCKED",
            "model_version": VERSION,
            "research_scope": research_scope,
            "source_cutoff": scenarios["source_cutoff"],
            "sample_count": len(paths),
            "required_data": [
                "目标日前至少10个完整历史日的预测快照与实际日前/实时价格",
                "每个历史日必须包含24个时段和有序P10/P50/P90",
                "预测快照发布时间及D-2可用性证明",
                "生成实际申报MWh还需目标日期24小时客户负荷、持仓和已成交量",
            ],
            "periods": [],
        }

    hourly = []
    for hour in range(24):
        optimized = optimize_lock_ratio(
            {}, {}, risk_aversion=.3, confidence=.95,
            minimum_ratio=.05, maximum_ratio=.95,
            historical_scenarios=[{
                "name": path["source_date"],
                "weight": path["weight"],
                "day_ahead_price": path["day_ahead"][hour],
                "real_time_price": path["real_time"][hour],
            } for path in paths],
        )
        hourly.append(optimized["ratio"])

    periods = []
    for hour, point in enumerate(target["periods"]):
        periods.append({
            "period": point["period"],
            "da_p50": point["day_ahead_price_yuan_per_mwh"]["p50"],
            "rt_p50": point["real_time_price_yuan_per_mwh"]["p50"],
            "rt_post_da_p50": (point.get("real_time_post_da_price_yuan_per_mwh") or {}).get("p50"),
            "actual_da": point.get("actual_day_ahead_price_yuan_per_mwh"),
            "actual_rt": point.get("actual_real_time_price_yuan_per_mwh"),
            "hourly_cvar_ratio": hourly[hour],
            "joint_cvar_ratio": joint["ratios"][hour],
            "suggested_quantity_mwh": None,
            "formal_action": "HOLD",
            "execution_allowed": False,
        })

    actual_costs = {}
    if all(isinstance(point.get("actual_day_ahead_price_yuan_per_mwh"), (int, float)) and
           isinstance(point.get("actual_real_time_price_yuan_per_mwh"), (int, float))
           for point in target["periods"]):
        for key, ratios in (("hourly", hourly), ("joint", joint["ratios"]),
                            ("full_da", [1.0] * 24), ("full_rt", [0.0] * 24)):
            actual_costs[key] = sum(
                ratio * point["actual_day_ahead_price_yuan_per_mwh"] +
                (1 - ratio) * point["actual_real_time_price_yuan_per_mwh"]
                for ratio, point in zip(ratios, target["periods"])
            ) / 24

    result = {
        "market_date": business_date,
        "status": "UNIT_EXPOSURE_RESEARCH_ONLY",
        "model_version": VERSION,
        "risk_aversion": .30,
        "confidence": .95,
        "execution_allowed": False,
        "formal_gate": "BLOCKED",
        "quantity_basis": "每小时相同单位敞口，仅用于价格策略对比；不是客户实际申报量。",
        "price_basis": "P50为预测中心，不是成交报价；缺申报规则与成交约束。",
        "research_scope": research_scope,
        "missing_data": ["对应日期24点客户负荷/预测", "对应日期中长期持仓", "已成交电量"],
        "periods": periods,
        "day_metrics": target.get("day_metrics") or _day_metrics(target["periods"]),
        "actual_unit_cost_yuan_per_mwh": actual_costs,
        "source_cutoff": scenarios["source_cutoff"],
        "source_dates": [path["source_date"] for path in paths],
        "sample_count": len(paths),
        "effective_sample_size": scenarios["effective_sample_size"],
        "joint_scenario_evaluation": joint["allocation"],
        "forecast_audit": target.get("audit") or {},
    }
    result["snapshot_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    return result
