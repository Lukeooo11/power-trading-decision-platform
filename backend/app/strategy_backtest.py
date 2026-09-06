"""Leakage-safe historical evaluation for the DA/RT split strategy."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from .bidding_strategy import build_historical_price_scenarios, generate_strategy


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _cost(da_qty: float, rt_qty: float, actual: dict[str, Any]) -> float | None:
    da = _number(actual.get("actual_day_ahead_price_yuan_per_mwh", actual.get("dayAheadPriceYuanMwh")))
    rt = _number(actual.get("actual_real_time_price_yuan_per_mwh", actual.get("realtimePriceYuanMwh")))
    return None if da is None or rt is None else da_qty * da + rt_qty * rt


def run_strategy_backtest(*, forecast_doc: dict[str, Any], load_rows: list[dict[str, Any]],
                          position_doc: dict[str, Any], price_rows: list[dict[str, Any]],
                          start: str | None = None, end: str | None = None,
                          risk_aversion: float = 0.3,
                          strategy_version: str = "historical_cvar_v02",
                          supply_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compare deterministic allocation policies over available historical days."""
    forecasts = {str(x.get("market_date")): x for x in forecast_doc.get("results", [])}
    loads: dict[str, dict[int, float]] = defaultdict(dict)
    for row in load_rows:
        value = _number(row.get("totalMwh"))
        if value is not None:
            loads[str(row.get("date"))][int(str(row.get("time", "0"))[:2])] = value
    prices: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in price_rows:
        prices[str(row.get("date"))][int(str(row.get("time", "0"))[:2])] = row
    supply: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in supply_rows or []:
        supply[str(row.get("marketDate"))][int(row.get("period", 0))] = row
    positions = {str(x.get("date")): _number(x.get("netPositionMwh"))
                 for x in position_doc.get("daily", [])}
    dates = sorted(set(forecasts) & set(loads) & set(prices))
    dates = [d for d in dates if (not start or d >= start) and (not end or d <= end)]
    policy_names = ("full_day_ahead", "full_realtime", "fixed_half", "spread_strategy")
    totals = {name: 0.0 for name in policy_names}
    daily: list[dict[str, Any]] = []
    for date in dates:
        day_forecast = forecasts[date]
        forecast_by = {int(x["period"]): x for x in day_forecast.get("periods", [])}
        day_loads = loads[date]
        total_load = sum(day_loads.values())
        position = positions.get(date)
        rows = []
        for period in range(1, 25):
            load = day_loads.get(period)
            actual = prices[date].get(period, {})
            if load is None or total_load <= 0:
                continue
            allocated = position * load / total_load if position is not None else None
            forecast = forecast_by.get(period, {})
            strategy = generate_strategy(load_mwh=load, medium_position_mwh=allocated,
                                         cleared_energy_mwh=0.0, forecast=forecast, actual=actual,
                                         risk_aversion=risk_aversion,
                                         period=period if strategy_version == "regime_cvar_v03" else None,
                                         supply_context=(supply[date].get(period)
                                                         if strategy_version == "regime_cvar_v03" else None),
                                         historical_scenarios=build_historical_price_scenarios(
                                             forecast_doc=forecast_doc, target_date=date,
                                             period=period,
                                             target_da=(forecast.get("day_ahead_price_yuan_per_mwh") or {}),
                                             target_rt=(forecast.get("real_time_price_yuan_per_mwh") or {}),
                                         ))
            remaining = _number(strategy.get("remaining_exposure_mwh")) or 0.0
            da_actual = _number(actual.get("actual_day_ahead_price_yuan_per_mwh", actual.get("dayAheadPriceYuanMwh")))
            rt_actual = _number(actual.get("actual_real_time_price_yuan_per_mwh", actual.get("realtimePriceYuanMwh")))
            if da_actual is None or rt_actual is None:
                continue
            quantities = {
                "full_day_ahead": (remaining, 0.0),
                "full_realtime": (0.0, remaining),
                "fixed_half": (remaining * 0.5, remaining * 0.5),
                "spread_strategy": (_number(strategy.get("day_ahead_quantity_mwh")) or 0.0,
                                    _number(strategy.get("real_time_reserved_mwh")) or 0.0),
            }
            costs = {name: round(_cost(*qty, actual) or 0.0, 4) for name, qty in quantities.items()}
            for name, value in costs.items():
                totals[name] += value
            rows.append({"period": period, "remaining_exposure_mwh": round(remaining, 6),
                         "actual_day_ahead_price": da_actual, "actual_real_time_price": rt_actual,
                         "strategy_lock_ratio": strategy.get("lock_ratio"), "costs": costs})
        if rows:
            daily_costs = {name: round(sum(x["costs"][name] for x in rows), 4) for name in policy_names}
            daily.append({"date": date, "costs": daily_costs,
                          "strategy_vs_full_da_yuan": round(daily_costs["full_day_ahead"] - daily_costs["spread_strategy"], 4),
                          "periods": rows})
    baseline = totals["full_day_ahead"]
    savings = {name: round(baseline - value, 4) for name, value in totals.items()}
    losses = [x["strategy_vs_full_da_yuan"] for x in daily]
    negative = sorted((x for x in losses if x < 0), key=lambda x: x)
    version_label = ("da-rt-split-regime-cvar-v0.3" if strategy_version == "regime_cvar_v03"
                     else "da-rt-split-historical-cvar-v0.2")
    return {"mode": "historical_strategy_backtest", "strategy_version": version_label,
            "risk_aversion": risk_aversion,
            "date_start": dates[0] if dates else None, "date_end": dates[-1] if dates else None,
            "day_count": len(daily), "policies": {name: {"total_cost_yuan": round(totals[name], 4),
            "vs_full_day_ahead_saving_yuan": savings[name]} for name in policy_names},
            "risk": {"strategy_mean_daily_saving_yuan": round(sum(losses) / len(losses), 4) if losses else None,
                     "strategy_worst_day_vs_full_da_yuan": round(min(losses), 4) if losses else None,
                     "strategy_loss_days": len(negative), "strategy_loss_p95_yuan": round(negative[max(0, int(len(negative) * 0.05) - 1)], 4) if negative else 0.0},
            "daily": daily, "assumptions": ["cleared_energy_mwh=0", "日级持仓按分时负荷比例分摊", "实际价格仅用于回测评分", "研究结果不构成申报指令"]}
