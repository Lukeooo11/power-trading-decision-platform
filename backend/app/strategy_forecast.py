"""Deterministic selector for the deployed Shandong forecast snapshots."""
from copy import deepcopy


CONDITIONAL_VERSION = "sd-target-supply-conditional-v1"
PROFILE_FILES = {
    "online_challenger": "price_forecast_history_online_challenger.json",
    "target_supply": "price_forecast_history_target_supply.json",
}
TARGET_SUPPLY_FIELDS = (
    "directDispatchLoadMw",
    "interconnectorMw",
    "windMw",
    "solarMw",
    "localPlantMw",
    "captiveUnitMw",
    "nonMarketNuclearMw",
)


def _complete_forecast_dates(supply):
    """Complete target-day forecast-type rows may activate the enhanced model.

    来源判定交由 supply_source_policy：FORECAST / PROXY_FORECAST /
    SEASONAL_PROXY_FORECAST 均为事前可得的预测类来源；ACTUAL 默认排除。
    """
    from .supply_source_policy import source_rank
    grouped = {}
    for row in (supply or {}).get("rows", []):
        if source_rank(row.get("sourceType")) is None:
            continue
        grouped.setdefault(row.get("marketDate"), []).append(row)
    complete = set()
    for stamp, rows in grouped.items():
        periods = {row.get("period") for row in rows}
        if periods == set(range(1, 25)) and len(rows) == 24 and all(
            row.get(field) is not None for row in rows for field in TARGET_SUPPLY_FIELDS
        ):
            complete.add(stamp)
    return complete


def compose_conditional_target_supply(target, fallback, supply):
    if not target or not fallback:
        raise ValueError("Conditional forecast requires target and fallback snapshots")
    target_days = {day["market_date"]: day for day in target.get("results", [])}
    fallback_days = {day["market_date"]: day for day in fallback.get("results", [])}
    if len(target_days) != len(target.get("results", [])) or len(fallback_days) != len(fallback.get("results", [])):
        raise ValueError("Duplicate forecast date")
    eligible = _complete_forecast_dates(supply)
    results = []
    selected = 0
    for stamp, old in sorted(fallback_days.items()):
        use_target = stamp in eligible and stamp in target_days
        day = deepcopy(target_days[stamp] if use_target else old)
        reason = None
        if not use_target:
            reason = (
                "TARGET_SUPPLY_MODEL_SNAPSHOT_MISSING"
                if stamp in eligible
                else "TARGET_SUPPLY_FORECAST_MISSING_OR_INCOMPLETE"
            )
        day["audit"] = deepcopy(day.get("audit") or {})
        day["audit"]["forecast_selection"] = {
            "profile": "target_supply_conditional",
            "selected_model_version": day.get("model", {}).get("version"),
            "target_supply_model_used": use_target,
            "fallback_used": not use_target,
            "fallback_reason": reason,
            "target_supply_source_required": "FORECAST",
            "target_supply_24h_complete": stamp in eligible,
            "target_actual_supply_used": False,
        }
        day["model"] = {"id": "price-forecast", "version": CONDITIONAL_VERSION}
        day["run_id"] = f"{CONDITIONAL_VERSION}-{stamp}"
        day["execution_allowed"] = False
        results.append(day)
        selected += int(use_target)
    return {
        "schema_version": "forecast-research-history-v1",
        "model": {"id": "price-forecast", "version": CONDITIONAL_VERSION},
        "generated_at": target.get("generated_at"),
        "results": results,
        "execution_allowed": False,
        "selection_summary": {
            "target_supply_days": selected,
            "fallback_days": len(results) - selected,
            "target_supply_source_required": "FORECAST",
            "fallback_model_version": fallback.get("model", {}).get("version"),
            "target_model_version": target.get("model", {}).get("version"),
        },
    }


def resolve_strategy_forecast(read_asset):
    return compose_conditional_target_supply(
        read_asset(PROFILE_FILES["target_supply"]),
        read_asset(PROFILE_FILES["online_challenger"]),
        read_asset("market_supply_hourly_2026h1.json"),
    )
