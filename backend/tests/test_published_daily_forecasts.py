"""Checks for the read-only, precomputed daily research forecasts."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from app.d1_shandong_price_forecast import MODEL_VERSION as V6_VERSION
from app.v7_shandong_price_forecast import MODEL_VERSION as V7_VERSION

PUBLIC = Path(__file__).resolve().parents[2] / "data" / "forecast-daily"


def test_all_dates_and_periods_are_published_without_trading_actions():
    index = json.loads((PUBLIC / "index.json").read_text(encoding="utf-8"))
    assert index["day_count"] == 273
    assert index["result_count"] == 365
    assert index["period_count"] == 8760
    assert index["dates"][0]["market_date"] == "2026-01-01"
    assert index["dates"][-1]["market_date"] == "2026-09-30"
    assert index["dates"][0]["evaluation_mode"] == "IN_SAMPLE_DIAGNOSTIC"
    assert len({row["market_date"] for row in index["dates"]}) == 273
    assert index["execution_allowed"] is False
    total = 0
    for month in range(1, 10):
        label = f"2026-{month:02d}"
        document = json.loads((PUBLIC / f"sd-{label}.json").read_text(encoding="utf-8"))
        assert document["month"] == label
        assert document["execution_allowed"] is False
        for result in document["results"]:
            total += 1
            assert result["market_date"].startswith(label)
            assert result["model"]["version"] in ([V6_VERSION] if month <= 6 else [V6_VERSION, V7_VERSION])
            assert result["run_id"] is None
            assert result["strategy_ready"] is False
            assert result["execution_allowed"] is False
            assert [row["period"] for row in result["periods"]] == list(range(1, 25))
            for row in result["periods"]:
                assert row["strategy_suggestion"]["action"] == "HOLD"
                assert row["strategy_suggestion"]["volume_mwh"] == 0
                for key in ("day_ahead_price_yuan_per_mwh", "real_time_price_yuan_per_mwh"):
                    values = row[key]
                    assert values["p10"] <= values["p50"] <= values["p90"]
    assert total == 365


def test_missing_realtime_actual_remains_unknown_and_static_assets_are_served():
    september = json.loads((PUBLIC / "sd-2026-09.json").read_text(encoding="utf-8"))
    last = september["results"][-1]
    assert last["market_date"] == "2026-09-30"
    assert last["backtest"]["real_time_mae_yuan_per_mwh"] is None
    assert "V7_NEGATIVE_PRICE_TAIL_UNDERCOVERED_RESEARCH_ONLY" in last["warnings"]
    with TestClient(main.app) as client:
        index = client.get("/data/forecast-daily/index.json")
        month = client.get("/data/forecast-daily/sd-2026-09.json")
    assert index.status_code == month.status_code == 200
    assert index.json()["day_count"] == 273
    assert index.json()["result_count"] == 365
    assert len(month.json()["results"]) == 60
