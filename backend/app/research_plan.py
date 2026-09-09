"""Date-based research using the platform's existing deterministic strategy."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone, timedelta
from typing import Any, Callable

from .bidding_strategy import build_historical_price_scenarios, generate_strategy
from .similar_day_scenarios import (STRATEGY_KEY, VERSION, build_similar_day_scenarios,
                                    generate_similar_day_strategy)
from .advanced_strategies import (ADVANCED_KEY, ADVANCED_VERSION, JOINT_KEY, JOINT_VERSION,
                                  apply_joint_daily_cvar, generate_advanced_strategy)


def number(value: Any, *, signed: bool = True) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return float(value) if signed or value >= 0 else None


def build_research_plan(
    business_date: str, read_asset: Callable[[str], Any], *,
    cached_forecasts: list[dict[str, Any]] | None = None,
    input_batches: list[dict[str, Any]] | None = None,
    risk_aversion: float = 0.3, strategy_version: str = "historical_cvar_v02",
    today: str | None = None,
) -> dict[str, Any]:
    date.fromisoformat(business_date)
    if strategy_version not in {"historical_cvar_v02", "regime_cvar_v03", STRATEGY_KEY, ADVANCED_KEY, JOINT_KEY}:
        raise ValueError("Unsupported research strategy version")
    today = today or datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    if date.fromisoformat(business_date).isoformat() != business_date:
        raise ValueError("business_date must use YYYY-MM-DD")
    historical = business_date < today
    mode = "historical_research" if historical else "target_day_research"
    history = read_asset("price_forecast_history_colleague_2026h1.json") or {}
    history_day = next((day for day in history.get("results", []) if day.get("market_date") == business_date), None)
    if history_day:
        history_day = {"model": history.get("model"), "data_snapshot": history.get("data_snapshot"), **history_day}
    candidate_assets = [
        ("price_forecast_history_colleague_2026h1.json", history_day),
        *[("model-runs-v1", value) for value in (cached_forecasts or [])],
        (f"price_forecast_result_colleague_{business_date}.json", read_asset(f"price_forecast_result_colleague_{business_date}.json")),
        (f"price_forecast_result_{business_date}.json", read_asset(f"price_forecast_result_{business_date}.json")),
    ]
    forecast_source, forecast = None, {}
    for source, candidate in candidate_assets:
        if not isinstance(candidate, dict) or candidate.get("market_date") != business_date:
            continue
        scenario = candidate.get("forecast_scenario") or candidate.get("scenario") or "pre_market"
        if scenario not in {"pre_market", "PRE_MARKET"}:
            continue
        rows = candidate.get("periods", [])
        if isinstance(rows, list) and len(rows) == 24 and all(isinstance(row, dict) and type(row.get("period")) is int for row in rows) and {row["period"] for row in rows} == set(range(1, 25)):
            forecast_source, forecast = source, candidate
            break
    forecasts = {row["period"]: row for row in forecast.get("periods", [])}
    weather = (read_asset("weather_hourly_gfs_20260501_20260701.json") or {}) if strategy_version in {STRATEGY_KEY, ADVANCED_KEY, JOINT_KEY} else {}
    actual_load_rows = read_asset("portfolio_load_hourly_2026h1.json") or []
    loads = {int(row["time"].split(":")[0]): row for row in actual_load_rows if historical and row.get("date") == business_date}
    position_doc = read_asset("medium_long_term_positions_2026h1.json") or {}
    daily_position = next((row for row in position_doc.get("daily", []) if row.get("date") == business_date), {})
    price_rows = read_asset("spot_prices_2026h1.json") or []
    actual_prices = {int(row["time"].split(":")[0]): row for row in price_rows if historical and row.get("date") == business_date}
    supply_doc = read_asset("market_supply_hourly_2026h1.json") or {}
    supply = {row["period"]: row for row in supply_doc.get("rows", []) if historical and row.get("marketDate") == business_date}
    sources = []
    if forecast:
        sources.append({"domain": "price_forecast", "source": forecast_source,
                        "version": (forecast.get("model") or {}).get("version") or forecast.get("data_version"),
                        "generated_at": forecast.get("generated_at") or (forecast.get("data_snapshot") or {}).get("created_at"),
                        "basis": "PRE_MARKET_FORECAST"})
    explicit: dict[str, dict[int, dict[str, Any]]] = {}
    for batch in input_batches or []:
        domain = batch.get("domain")
        if domain not in {"load_forecast", "medium_long_term_positions", "clearing_results"} or domain in explicit:
            continue
        rows = (batch.get("payload") or {}).get("records", [])
        selected = [row for row in rows if isinstance(row, dict) and (row.get("business_date") or row.get("market_date")) == business_date]
        if not selected:
            continue
        periods = [row.get("period") for row in selected]
        if len(selected) != 24 or any(type(p) is not int for p in periods) or set(periods) != set(range(1, 25)) or any(row.get("unit", "MWh") != "MWh" or row.get("market_code", "SD") != "SD" for row in selected):
            continue
        explicit[domain] = {row["period"]: row for row in selected}
        sources.append({"domain": domain, "source": batch.get("source_system"), "version": batch.get("data_version"),
                        "generated_at": batch.get("updated_at"), "basis": "PLATFORM_INPUT_BATCH"})
    load_total = sum(row["totalMwh"] for row in loads.values()) if len(loads) == 24 and all(number(row.get("totalMwh"), signed=False) is not None for row in loads.values()) else None
    net_position = number(daily_position.get("netPositionMwh"))
    assumptions = ["研究量价不是交易终端申报文件；不自动提交", "预测与原始数据的精确事前可得时间尚未完整核验，不标注为严格事前回测"]
    if loads and "load_forecast" not in explicit:
        assumptions.append("历史组合实际负荷仅作研究输入，不冒充当时负荷预测")
        sources.append({"domain": "load", "source": "portfolio_load_hourly_2026h1.json", "basis": "HISTORICAL_ACTUAL"})
    allocate_daily = historical and "medium_long_term_positions" not in explicit and load_total and isinstance(net_position, (int, float))
    if allocate_daily:
        assumptions.append("日级净持仓按历史实际负荷比例分摊，仅为研究假设")
        sources.append({"domain": "positions", "source": "medium_long_term_positions_2026h1.json", "basis": "DAILY_ALLOCATION_ASSUMPTION"})
    if historical and "clearing_results" not in explicit:
        assumptions.append("额外已成交电量按 0 测算，仅为历史研究假设，不等于已核实无成交")
    records = []
    for period in range(1, 25):
        load_row = explicit.get("load_forecast", {}).get(period)
        load = number(load_row.get("load_forecast_mwh") if load_row is not None else loads.get(period, {}).get("totalMwh"), signed=False)
        position_row = explicit.get("medium_long_term_positions", {}).get(period)
        position = number(position_row.get("medium_position_mwh") if position_row is not None else (net_position * load / load_total if allocate_daily and load is not None else None))
        cleared_row = explicit.get("clearing_results", {}).get(period)
        cleared = number(cleared_row.get("cleared_energy_mwh") if cleared_row is not None else (0.0 if historical else None), signed=False)
        point = forecasts.get(period, {})
        # Never forward realised target-day prices as a prediction or scenario.
        decision_forecast = {key: point[key] for key in ("period", "day_ahead_price_yuan_per_mwh", "real_time_price_yuan_per_mwh", "negative_price_risk", "high_price_risk") if key in point}
        for key in ("day_ahead_price_yuan_per_mwh", "real_time_price_yuan_per_mwh"):
            quantiles = decision_forecast.get(key) or {}
            values = [number(quantiles.get(level)) for level in ("p10", "p50", "p90")]
            if None in values or values != sorted(values):
                decision_forecast[key] = {}
        scenarios = build_historical_price_scenarios(forecast_doc=history, target_date=business_date, period=period,
            target_da=decision_forecast.get("day_ahead_price_yuan_per_mwh") or {}, target_rt=decision_forecast.get("real_time_price_yuan_per_mwh") or {})
        flags = ["DAILY_POSITION_ALLOCATION_RESEARCH_ASSUMPTION"] if allocate_daily else []
        if position is not None and position < 0:
            flags.append("NEGATIVE_POSITION_REVIEW")
        strategy_args = dict(load_mwh=load, medium_position_mwh=position, cleared_energy_mwh=cleared,
            forecast=decision_forecast, actual=actual_prices.get(period), risk_aversion=risk_aversion,
            period=period if strategy_version in {"regime_cvar_v03", ADVANCED_KEY, JOINT_KEY} else None,
            supply_context=supply.get(period) if strategy_version in {"regime_cvar_v03", ADVANCED_KEY, JOINT_KEY} else None,
            position_quality_flags=flags)
        if strategy_version == STRATEGY_KEY:
            audit = build_similar_day_scenarios(forecast_doc=history, target_date=business_date,
                period=period, target_forecast=decision_forecast, weather_doc=weather,
                target_model_version=(forecast.get("model") or {}).get("version"))
            strategy = generate_similar_day_strategy(scenario_audit=audit, **strategy_args)
        elif strategy_version in {ADVANCED_KEY, JOINT_KEY}:
            advanced_args = {key: value for key, value in strategy_args.items()
                             if key not in {"period", "supply_context"}}
            strategy = generate_advanced_strategy(
                forecast_doc=history, target_date=business_date, period=period,
                weather_doc=weather,
                target_model_version=(forecast.get("model") or history.get("model") or {}).get("version"),
                supply_context=supply.get(period), **advanced_args)
        else:
            strategy = generate_strategy(**strategy_args, historical_scenarios=scenarios)
        records.append({"period": period, "load_mwh": load, "medium_position_allocated_mwh": position,
                        "cleared_energy_mwh": cleared, "load_basis": "FORECAST" if load_row is not None else "HISTORICAL_ACTUAL" if load is not None else "MISSING",
                        "position_basis": "HOURLY_INPUT" if position_row is not None else "DAILY_ALLOCATION_ASSUMPTION" if position is not None else "MISSING",
                        "forecast": decision_forecast, "strategy": strategy})
    active = [row["strategy"] for row in records if row["strategy"]["action"] == "BUY_SPLIT"]
    complete = len(active) == 24
    def total(field):
        values = [row["strategy"].get(field) for row in records]
        return round(sum(values), 6) if all(value is not None for value in values) else None
    gaps = sorted({item for row in records if row["strategy"]["gate_status"] == "BLOCKED" for item in row["strategy"]["reason"].split(";") if item})
    if not forecast:
        gaps.append("TARGET_DATE_PRE_MARKET_FORECAST_MISSING")
    joint_result = None
    if strategy_version == JOINT_KEY:
        joint_result = apply_joint_daily_cvar(records, risk_aversion=risk_aversion)
        if joint_result.get("status") == "READY":
            active = [row["strategy"] for row in records if row["strategy"]["action"] == "BUY_SPLIT"]
            complete = len(active) == 24
    result = {"market_code": "SD", "business_date": business_date, "mode": mode,
              "research_status": "READY" if complete else "BLOCKED", "strategy_version": "da-rt-split-regime-cvar-v0.3" if strategy_version == "regime_cvar_v03" else "da-rt-split-historical-cvar-v0.2",
              "risk_aversion": risk_aversion, "forecast_version": (forecast.get("model") or {}).get("version") or forecast.get("data_version"),
              "position_data_version": "sd-contract-position-2026h1-v1" if allocate_daily else None,
              "assumptions": assumptions, "missing_data": gaps, "sources": sources, "records": records,
              "summary": {"period_count": 24, "actionable_periods": len(active), "quote_periods": len(active),
                  "day_ahead_quantity_mwh": total("day_ahead_quantity_mwh") if complete else None,
                  "reserved_realtime_mwh": total("real_time_reserved_mwh") if complete else None,
                  "expected_cost_advantage_yuan": total("expected_cost_advantage_yuan"),
                  "expected_saving_yuan": total("expected_cost_advantage_yuan"),
                  "actual_cost_advantage_yuan": total("actual_cost_advantage_yuan"),
                  "full_day_ahead_actual_advantage_yuan": round(sum(row["strategy"]["actual_cost_advantage_yuan"] / row["strategy"]["lock_ratio"] for row in records), 4) if complete and all(row["strategy"].get("actual_cost_advantage_yuan") is not None and row["strategy"].get("lock_ratio") for row in records) else None},
              "method": {"name": "分时状态CVaR日前锁定-实时保留" if strategy_version == "regime_cvar_v03" else "历史误差场景CVaR日前锁定-实时保留", "buyer_rule": "复用平台策略引擎，在5%-95%候选比例中比较期望采购成本与95%尾部成本", "note": "风险按小时计算；价格区间为预测参考，不是确认的申报曲线"},
              "top_attention_periods": [{"period": row["period"], "lock_ratio": row["strategy"].get("lock_ratio"), "reason": row["strategy"].get("reason"), "risk_adjusted_edge_yuan_per_mwh": row["strategy"].get("risk_adjusted_edge_yuan_per_mwh")} for row in sorted(records, key=lambda row: abs(row["strategy"].get("risk_adjusted_edge_yuan_per_mwh") or 0), reverse=True)[:5]],
              "formal_gate": "BLOCKED", "execution_allowed": False}
    if strategy_version == STRATEGY_KEY:
        result["strategy_version"] = VERSION
        result["method"]["name"] = "预测与相似日误差CVaR（挑战者）"
        result["assumptions"].extend([
            "相似日采用目标日价格预测与申报前业务确认的天气预测；目标日实际价格、负荷和供需不参与相似度筛选",
            "误差仅取D-2及更早日期；结算实际发布时间仍待逐条核验",
            "相似度阈值、样本门槛与特征权重为预设研究参数；强制P50对齐仅作离线消融，不用于当前策略",
            "场景以目标预测为基础并保留相似日偏差修正，场景分位数可能与原预测不同，权重尚非已校准真实概率",
        ])
        result["sources"].append({"domain": "scenario_weather", "source": "weather_hourly_gfs_20260501_20260701.json",
            "version": weather.get("dataVersion"), "basis": "BUSINESS_CONFIRMED_PRE_DECLARATION_IF_AVAILABLE"})
    if strategy_version in {ADVANCED_KEY, JOINT_KEY}:
        result["strategy_version"] = JOINT_VERSION if strategy_version == JOINT_KEY else ADVANCED_VERSION
        result["method"] = {"name": "混合场景 + 分位数校准 + 直接价差 + Newsvendor + 置信度收缩" if strategy_version == ADVANCED_KEY else "24小时联合CVaR混合策略",
            "buyer_rule": "先用目标日前历史误差构造场景，再在5%-95%候选日前比例中比较期望成本、95%尾部成本与多购/欠购损失",
            "note": "所有场景来自目标日前历史预测误差；结果为研究方案，不是正式申报文件"}
        result["assumptions"].extend([
            "混合场景权重固定为相似日55%、近期误差25%、极端尾部20%，参数预先锁定，不按测试结果反复调参",
            "分位数校准使用目标日前加权经验残差，不使用目标日实际价格；直接价差预测使用配对RT-DA残差",
            "置信度由价差方向信号与有效样本量共同决定，比例向50%收缩；样本不足或模型版本不一致时保持HOLD",
            "Newsvendor项按场景中的日前多购和实时欠购单位损失计入目标；不解释为发生概率",
        ])
        result["sources"].append({"domain": "advanced_scenarios", "source": "price_forecast_history_colleague_2026h1.json",
            "version": history.get("model", {}).get("version"), "basis": "PRE_TARGET_PAIRED_RESIDUALS"})
        if joint_result is not None:
            result["joint_cvar"] = joint_result
            if joint_result.get("status") != "READY":
                result["missing_data"].append(joint_result.get("reason", "JOINT_CVAR_BLOCKED"))
    result["source_sha256"] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result
