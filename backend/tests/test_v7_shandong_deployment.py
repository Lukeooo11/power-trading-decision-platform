from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app import main
from app.d1_shandong_price_forecast import MODEL_VERSION as V6_VERSION
from app.price_forecast_model import run_price_forecast
from app.v7_shandong_price_forecast import MODEL_VERSION as V7_VERSION


PRIVATE = Path(__file__).resolve().parents[2] / "private-data" / "shandong-2026h1"


def test_v7_replay_has_24_points_and_matches_walk_forward_asset():
    result = run_price_forecast(PRIVATE, "2026-09-19", model_version=V7_VERSION)
    assert result["model_version"] == V7_VERSION
    assert result["summary"]["evaluation_mode"] == "MONTHLY_WALK_FORWARD_RESEARCH"
    assert result["summary"]["forecast_selection"]["target_actual_supply_used"] is False
    assert [point["period"] for point in result["forecast"]] == list(range(1, 25))
    asset = Path(__file__).resolve().parents[1] / "model_assets" / V7_VERSION / "v7_walk_forward_predictions.csv.gz"
    replay = pd.read_csv(asset, compression="gzip").query("date == '2026-09-19'").sort_values("period")
    for point, row in zip(result["forecast"], replay.itertuples()):
        assert point["da_p50"] == round(row.da_q50, 3)
        assert point["rt_p50"] == round(row.rt_regime, 3)
        assert point["da_p10"] <= point["da_p50"] <= point["da_p90"]
        assert point["rt_p10"] <= point["rt_p50"] <= point["rt_p90"]


def test_v7_keeps_missing_realtime_actual_unknown():
    result = run_price_forecast(PRIVATE, "2026-09-30", model_version=V7_VERSION)
    assert len(result["forecast"]) == 24
    assert result["summary"]["rt_metrics"]["mae"] is None
    assert result["summary"]["interval_status"] == "RESEARCH_ONLY_NEGATIVE_TAIL_NOT_CALIBRATED"


def test_v7_is_default_and_v6_remains_selectable_through_api():
    with tempfile.TemporaryDirectory() as directory, patch.object(main, "DB_PATH", Path(directory) / "platform.db"):
        with TestClient(main.app) as client:
            for version, expected in ((V7_VERSION, V7_VERSION), (V6_VERSION, V6_VERSION)):
                response = client.post("/api/v1/models/price-forecast/runs", json={
                    "request_id": f"compare-{version}", "market_code": "SD", "market_date": "2026-09-19",
                    "model_id": "price-forecast", "model_version": version,
                    "data_version": "sd-16city-gfs24-actual-supply-20260101-20260930-d1-enhanced-v6",
                    "input_summary": {"available_domains": ["prices", "weather"]},
                })
                assert response.status_code == 202, response.text
                run = response.json()
                assert run["status"] == "SUCCEEDED", run
                assert run["model_version"] == expected
                payload = client.get(run["result_url"]).json()["result"]
                assert payload["model"]["version"] == expected
                assert len(payload["periods"]) == 24
                assert all(row["strategy_suggestion"]["action"] == "HOLD" for row in payload["periods"])
                assert all(row["strategy_suggestion"]["volume_mwh"] == 0 for row in payload["periods"])
                if expected == V7_VERSION:
                    assert "V7_NEGATIVE_PRICE_TAIL_UNDERCOVERED_RESEARCH_ONLY" in payload["warnings"]


def test_v7_rejects_dates_outside_packaged_replay():
    try:
        run_price_forecast(PRIVATE, "2026-06-30", model_version=V7_VERSION)
    except RuntimeError as error:
        assert "only from 2026-07-01 to 2026-09-30" in str(error)
    else:
        raise AssertionError("V7 should not silently serve a V6 date")
