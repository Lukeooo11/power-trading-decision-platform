from app.day_path_cvar import build_day_paths
from app.information_boundary import (
    PRICE_SETTLEMENT_LAG_DAYS,
    PRICE_SETTLEMENT_LAG_HOURS,
    price_history_cutoff,
)


def _period(period: int, da: float, rt: float, *, actual: bool = False) -> dict:
    row = {
        "period": period,
        "day_ahead_price_yuan_per_mwh": {"p10": da - 10, "p50": da, "p90": da + 10},
        "real_time_price_yuan_per_mwh": {"p10": rt - 10, "p50": rt, "p90": rt + 10},
    }
    if actual:
        row["actual_day_ahead_price_yuan_per_mwh"] = da + 2
        row["actual_real_time_price_yuan_per_mwh"] = rt - 3
    return row


def test_price_settlement_boundary_is_previous_day():
    assert PRICE_SETTLEMENT_LAG_DAYS == 1
    assert PRICE_SETTLEMENT_LAG_HOURS == 24
    assert price_history_cutoff("2026-09-02").isoformat() == "2026-09-01"


def test_day_path_scenarios_accept_previous_day_settlement_prices():
    target = [_period(period, 300.0, 320.0) for period in range(1, 25)]
    prior = [_period(period, 280.0, 310.0, actual=True) for period in range(1, 25)]
    result = build_day_paths(
        forecast_doc={
            "results": [{
                "market_date": "2026-09-01",
                "forecast_scenario": "pre_market",
                "periods": prior,
            }]
        },
        target_date="2026-09-02",
        target_forecasts=target,
        samples=30,
    )
    assert result["source_cutoff"] == "2026-09-01"
    assert result["sample_count"] == 1
    assert result["paths"][0]["source_date"] == "2026-09-01"
    assert result["availability"] == "D-1_SETTLEMENT_AVAILABLE_BEFORE_TRADE"
