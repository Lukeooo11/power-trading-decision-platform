"""Comparable evaluation of all research strategy variants."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .bidding_strategy import (
    _num,
    _spread_probability,
    build_historical_price_scenarios,
    generate_strategy,
    lock_ratio,
)
from .similar_day_scenarios import (STRATEGY_KEY, CONFIG, build_similar_day_scenarios,
                                    generate_similar_day_strategy)
from .advanced_strategies import ADVANCED_KEY, ADVANCED_VERSION, generate_advanced_strategy


def _actual_prices(row: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        _num(row.get("actual_day_ahead_price_yuan_per_mwh", row.get("dayAheadPriceYuanMwh"))),
        _num(row.get("actual_real_time_price_yuan_per_mwh", row.get("realtimePriceYuanMwh"))),
    )


def _legacy_lock_ratio(forecast: dict[str, Any]) -> float | None:
    """Reproduce the first spread-probability version for a fair comparison."""
    da = forecast.get("day_ahead_price_yuan_per_mwh") or {}
    rt = forecast.get("real_time_price_yuan_per_mwh") or {}
    da50, rt50 = _num(da.get("p50")), _num(rt.get("p50"))
    p10, p90 = _num(rt.get("p10")), _num(rt.get("p90"))
    if None in (da50, rt50, p10, p90):
        return None
    negative = bool((forecast.get("negative_price_risk") or {}).get("level") in {"HIGH", "MEDIUM"})
    high = bool((forecast.get("high_price_risk") or {}).get("level") in {"HIGH", "MEDIUM"})
    ratio, _ = lock_ratio(da50, rt50, interval_width=p90 - p10,
                          realtime_p10=p10, negative_risk=negative, high_risk=high)
    probability, _ = _spread_probability(da, rt)
    probability_ratio = max(0.05, min(0.95, probability))
    if probability < 0.5:
        probability_ratio *= 0.85
    return round(max(0.05, min(ratio, probability_ratio)), 4)


def _cost(remaining: float, ratio: float, da: float, rt: float) -> float:
    return remaining * (ratio * da + (1.0 - ratio) * rt)


def _metric(rows: list[dict[str, Any]], name: str, baseline: str = "full_day_ahead") -> dict[str, Any]:
    total = sum(float(row["costs"][name]) for row in rows)
    base = sum(float(row["costs"][baseline]) for row in rows)
    by_date: dict[str, float] = defaultdict(float)
    by_base: dict[str, float] = defaultdict(float)
    ratios = []
    for row in rows:
        by_date[row["date"]] += float(row["costs"][baseline]) - float(row["costs"][name])
        by_base[row["date"]] += float(row["costs"][baseline])
        if row.get("ratios", {}).get(name) is not None:
            ratios.append(float(row["ratios"][name]))
    daily = list(by_date.values())
    losses = [value for value in daily if value < 0]
    return {
        "total_cost_yuan": round(total, 4),
        "vs_full_day_ahead_saving_yuan": round(base - total, 4),
        "mean_daily_saving_yuan": round(sum(daily) / len(daily), 4) if daily else None,
        "worst_day_vs_full_day_ahead_yuan": round(min(daily), 4) if daily else None,
        "loss_days": len(losses),
        "average_lock_ratio": round(sum(ratios) / len(ratios), 6) if ratios else None,
        "period_count": len(rows),
        "day_count": len({row["date"] for row in rows}),
    }


def compare_strategy_versions(*, forecast_doc: dict[str, Any], load_rows: list[dict[str, Any]],
                              position_doc: dict[str, Any], price_rows: list[dict[str, Any]],
                              supply_rows: list[dict[str, Any]] | None = None,
                              weather_doc: dict[str, Any] | None = None,
                              start: str | None = None, end: str | None = None,
                              risk_aversion: float = 0.5) -> dict[str, Any]:
    forecasts = {str(x.get("market_date")): x for x in forecast_doc.get("results", [])}
    loads: dict[str, dict[int, float]] = defaultdict(dict)
    for item in load_rows:
        value = _num(item.get("totalMwh"))
        if value is not None:
            loads[str(item.get("date"))][int(str(item.get("time", "0"))[:2])] = value
    positions = {str(x.get("date")): _num(x.get("netPositionMwh"))
                 for x in position_doc.get("daily", [])}
    prices: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in price_rows:
        prices[str(item.get("date"))][int(str(item.get("time", "0"))[:2])] = item
    supply: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in supply_rows or []:
        supply[str(item.get("marketDate"))][int(item.get("period", 0))] = item

    all_dates = sorted(set(loads) & set(positions) & set(prices))
    all_dates = [d for d in all_dates if (not start or d >= start) and (not end or d <= end)]
    forecast_dates = [d for d in all_dates if d in forecasts]
    policy_names = (
        "full_day_ahead", "full_realtime", "fixed_half",
        "legacy_spread_v01", "quantile_cvar_v02", "historical_cvar_v02", "regime_cvar_v03",
    )

    similar_diagnostics = []

    def collect(dates: list[str], include_forecast_policies: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for date in dates:
            day_loads = loads[date]
            total_load = sum(day_loads.values())
            position = positions[date]
            if total_load <= 0 or position is None:
                continue
            forecast_day = forecasts.get(date, {})
            forecast_by = {int(x["period"]): x for x in forecast_day.get("periods", [])}
            for period in range(1, 25):
                load = day_loads.get(period)
                actual = prices[date].get(period, {})
                da, rt = _actual_prices(actual)
                if load is None or da is None or rt is None:
                    continue
                remaining = max(0.0, float(load) - float(position) * float(load) / total_load)
                ratios: dict[str, float] = {
                    "full_day_ahead": 1.0, "full_realtime": 0.0, "fixed_half": 0.5,
                }
                if include_forecast_policies:
                    forecast = forecast_by.get(period, {})
                    legacy = _legacy_lock_ratio(forecast)
                    if legacy is not None:
                        ratios["legacy_spread_v01"] = legacy
                        historical = build_historical_price_scenarios(
                            forecast_doc=forecast_doc, target_date=date, period=period,
                            target_da=(forecast.get("day_ahead_price_yuan_per_mwh") or {}),
                            target_rt=(forecast.get("real_time_price_yuan_per_mwh") or {}),
                        )
                        q = generate_strategy(load_mwh=load, medium_position_mwh=float(position) * float(load) / total_load,
                                              cleared_energy_mwh=0.0, forecast=forecast,
                                              risk_aversion=risk_aversion, historical_scenarios=None)
                        h = generate_strategy(load_mwh=load, medium_position_mwh=float(position) * float(load) / total_load,
                                              cleared_energy_mwh=0.0, forecast=forecast,
                                              risk_aversion=risk_aversion, historical_scenarios=historical)
                        regime = generate_strategy(load_mwh=load, medium_position_mwh=float(position) * float(load) / total_load,
                                                  cleared_energy_mwh=0.0, forecast=forecast,
                                                  risk_aversion=risk_aversion, period=period,
                                                  supply_context=supply[date].get(period),
                                                  historical_scenarios=historical)
                        advanced = generate_advanced_strategy(
                            forecast_doc=forecast_doc, target_date=date, period=period,
                            weather_doc=weather_doc, target_model_version=(forecast_day.get("model") or forecast_doc.get("model") or {}).get("version"),
                            supply_context=supply[date].get(period),
                            load_mwh=load, medium_position_mwh=float(position) * float(load) / total_load,
                            cleared_energy_mwh=0.0, forecast=forecast, actual=actual,
                            risk_aversion=risk_aversion)
                        ratios["quantile_cvar_v02"] = float(q.get("lock_ratio") or 0.05)
                        ratios["historical_cvar_v02"] = float(h.get("lock_ratio") or 0.05)
                        ratios["regime_cvar_v03"] = float(regime.get("lock_ratio") or 0.05)
                        if advanced.get("lock_ratio") is not None and advanced.get("action") == "BUY_SPLIT":
                            ratios[ADVANCED_KEY] = float(advanced["lock_ratio"])
                        audit = build_similar_day_scenarios(forecast_doc=forecast_doc, target_date=date,
                            period=period, target_forecast=forecast, weather_doc=weather_doc,
                            target_model_version=(forecast_day.get("model") or forecast_doc.get("model") or {}).get("version"))
                        similar = generate_similar_day_strategy(scenario_audit=audit,
                            load_mwh=load, medium_position_mwh=float(position) * float(load) / total_load,
                            cleared_energy_mwh=0.0, forecast=forecast, risk_aversion=risk_aversion)
                        similar_diagnostics.append({"date": date, "period": period,
                            "status": similar["gate_status"], "blockers": audit["blockers"],
                            "selected_count": audit["selected_count"], "effective_sample_size": audit["effective_sample_size"],
                            "selection_basis": audit["selection_basis"]})
                        if similar.get("lock_ratio") is not None and similar["action"] == "BUY_SPLIT":
                            ratios[STRATEGY_KEY] = similar["lock_ratio"]
                costs = {name: round(_cost(remaining, ratio, da, rt), 4)
                         for name, ratio in ratios.items()}
                rows.append({"date": date, "period": period, "costs": costs, "ratios": ratios})
        return rows

    benchmark_rows = collect(all_dates, include_forecast_policies=False)
    forecast_rows = collect(forecast_dates, include_forecast_policies=True)
    benchmark_metrics = {name: _metric(benchmark_rows, name) for name in ("full_day_ahead", "full_realtime", "fixed_half")}
    forecast_metrics = {name: _metric(forecast_rows, name) for name in policy_names}
    # Keep incumbent coverage unchanged; do not score blocked hours as free power.
    eligible_dates = [d for d in forecast_dates if
        len([r for r in forecast_rows if r["date"] == d and STRATEGY_KEY in r["costs"] and ADVANCED_KEY in r["costs"]]) == 24]
    matched_rows = [r for r in forecast_rows if r["date"] in eligible_dates]
    similar_names = (*policy_names, STRATEGY_KEY, ADVANCED_KEY)

    def grouped_metrics(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        months = sorted({row["date"][:7] for row in rows})
        return {month: {name: _metric([row for row in rows if row["date"].startswith(month)], name)
                        for name in names} for month in months}

    return {
        "mode": "strategy_version_comparison",
        "strategy_version": "comparison-2026h1-v1",
        "date_start": all_dates[0] if all_dates else None,
        "date_end": all_dates[-1] if all_dates else None,
        "benchmark_coverage": {"date_start": all_dates[0] if all_dates else None,
                                "date_end": all_dates[-1] if all_dates else None,
                                "day_count": len({row["date"] for row in benchmark_rows}),
                                "period_count": len(benchmark_rows)},
        "forecast_strategy_coverage": {"date_start": forecast_dates[0] if forecast_dates else None,
                                        "date_end": forecast_dates[-1] if forecast_dates else None,
                                        "day_count": len({row["date"] for row in forecast_rows}),
                                        "period_count": len(forecast_rows)},
        "benchmarks_full_available_window": benchmark_metrics,
        "strategies_forecast_comparable_window": forecast_metrics,
        "monthly_benchmarks_full_available_window": grouped_metrics(
            benchmark_rows, ("full_day_ahead", "full_realtime", "fixed_half")),
        "monthly_strategies_forecast_comparable_window": grouped_metrics(
            forecast_rows, policy_names),
        "similar_day_evaluation": {
            "config": dict(CONFIG), "risk_aversion": risk_aversion,
            "status": "RESEARCH_COMPARISON" if eligible_dates else "INSUFFICIENT_COMPLETE_DAYS",
            "date_start": eligible_dates[0] if eligible_dates else None,
            "date_end": eligible_dates[-1] if eligible_dates else None,
            "eligible_dates": eligible_dates, "day_count": len(eligible_dates), "period_count": len(matched_rows),
            "excluded_dates": [d for d in forecast_dates if d not in eligible_dates],
            "strategies": {name: _metric(matched_rows, name) for name in similar_names} if matched_rows else {},
            "monthly": grouped_metrics(matched_rows, similar_names),
            "period_quality": similar_diagnostics,
            "daily": [{"date": d, "saving_vs_full_day_ahead": {name: round(sum(r["costs"]["full_day_ahead"] - r["costs"][name] for r in matched_rows if r["date"] == d), 4) for name in similar_names}} for d in eligible_dates],
            "limitations": ["仅比较新版24小时全部通过的共同日期；缺样本日不计为0成本或0收益",
                "已有历史预测的发布时间和训练泄漏仍待完整审计，不称严格样本外实盘收益",
                "相似度与门槛是预设研究参数；对比结果不得用于同窗反复择优调参"]},
        "daily_forecast_comparison": [
            {"date": date, "saving_vs_full_day_ahead": {
                name: round(sum(float(row["costs"]["full_day_ahead"]) - float(row["costs"][name])
                                for row in forecast_rows if row["date"] == date), 4)
                for name in policy_names
            }} for date in forecast_dates
        ],
        "assumptions": ["cleared_energy_mwh=0", "日级持仓按分时负荷比例分摊",
                        "基准策略使用半年可用数据，预测策略仅在47天预测覆盖窗口比较",
                        "实际价格仅用于回测评分，不参与策略生成", "研究结果不构成交易指令"],
    }
