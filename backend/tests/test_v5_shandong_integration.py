from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import main
from app.v5_shandong_price_forecast import MODEL_VERSION, run_v5_shandong_forecast
from app.price_forecast_model import run_price_forecast

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "private-data" / "shandong-2026h1"

def test_v5_emits_24_ordered_quantile_periods():
    result = run_v5_shandong_forecast(target_date="2026-07-01", private_data_dir=PRIVATE)
    assert result["model_version"] == MODEL_VERSION
    assert len(result["forecast"]) == 24
    assert [row["period"] for row in result["forecast"]] == list(range(1, 25))
    for row in result["forecast"]:
        assert row["da_p10"] <= row["da_p50"] <= row["da_p90"]
        assert row["rt_p10"] <= row["rt_p50"] <= row["rt_p90"]
        assert 0 <= row["negative_risk_probability"] <= 1
        assert 0 <= row["high_price_risk_probability"] <= 1

def test_price_forecast_entry_selects_v5():
    result = run_price_forecast(PRIVATE, "2026-07-01")
    assert result["model_version"] == MODEL_VERSION
    assert len(result["forecast"]) == 24


def test_v5_raw_contract_keeps_platform_hold_controls():
    from app.main import ModelRunCreateRequestV1, build_price_forecast_v1_result
    raw = run_price_forecast(PRIVATE, "2026-07-01")
    request = ModelRunCreateRequestV1(
        request_id="test-v5-contract", market_code="SD", market_date="2026-07-01",
        model_id="price-forecast", model_version="sd-target-supply-conditional-v1",
        data_version=raw["data_version"], parameters={},
        input_summary={"available_domains": ["prices", "weather"]}, timeout_seconds=120,
    )
    result = build_price_forecast_v1_result(request, "run-test", raw)
    assert result.model.version == MODEL_VERSION
    assert raw["model_version"] == MODEL_VERSION
    assert result.strategy_ready is False
    assert all(period.strategy_suggestion.action == "HOLD" for period in result.periods)
    assert "GFS_FIXED_LEAD24_IS_NOT_A_UNIFIED_DECLARATION_CUTOFF_SNAPSHOT" in result.warnings


def test_v5_uses_explicit_evaluation_modes_across_extended_range():
    expected = {
        "2026-01-01": "IN_SAMPLE_DIAGNOSTIC",
        "2026-02-01": "EXPANDING_WINDOW_OOF",
        "2026-06-30": "EXPANDING_WINDOW_OOF",
        "2026-07-01": "FROZEN_HOLDOUT",
    }
    for target_date, mode in expected.items():
        result = run_v5_shandong_forecast(target_date=target_date, private_data_dir=PRIVATE)
        assert len(result["forecast"]) == 24
        assert result["summary"]["evaluation_mode"] == mode
        assert result["summary"]["window_start"] == target_date
        assert result["summary"]["window_end"] == target_date


def test_v5_rejects_dates_outside_extended_range():
    for target_date in ("2025-12-31", "2026-08-01"):
        try:
            run_v5_shandong_forecast(target_date=target_date, private_data_dir=PRIVATE)
        except ValueError as error:
            assert "validated range" in str(error)
        else:
            raise AssertionError("V5 must reject dates outside the controlled replay range")


def test_forecast_api_reports_v5_instead_of_silent_fallback():
    with tempfile.TemporaryDirectory() as directory, patch.object(main, "DB_PATH", Path(directory) / "platform.db"):
        with TestClient(main.app) as client:
            response = client.post(
                "/api/v1/models/price-forecast/runs",
                json={
                    "request_id": "v5-online-contract-test",
                    "market_code": "SD",
                    "market_date": "2026-07-01",
                    "model_id": "price-forecast",
                    "model_version": "sd-target-supply-conditional-v1",
                    "data_version": "legacy-ui-alias",
                    "input_summary": {"available_domains": ["prices", "weather"]},
                },
            )
            assert response.status_code == 202, response.text
            run = response.json()
            assert run["status"] == "SUCCEEDED"
            assert run["model_version"] == MODEL_VERSION
            result = client.get(run["result_url"]).json()["result"]
            assert result["model"]["version"] == MODEL_VERSION
            assert len(result["periods"]) == 24
            assert all(row["strategy_suggestion"]["action"] == "HOLD" for row in result["periods"])


def test_forecast_api_keeps_prediction_when_actual_realtime_is_missing():
    with tempfile.TemporaryDirectory() as directory, patch.object(main, "DB_PATH", Path(directory) / "platform.db"):
        with TestClient(main.app) as client:
            response = client.post(
                "/api/v1/models/price-forecast/runs",
                json={
                    "request_id": "v5-missing-actual-test",
                    "market_code": "SD",
                    "market_date": "2026-02-12",
                    "model_id": "price-forecast",
                    "model_version": "sd-target-supply-conditional-v1",
                    "data_version": "legacy-ui-alias",
                    "input_summary": {"available_domains": ["prices", "weather"]},
                },
            )
            run = response.json()
            assert run["status"] == "SUCCEEDED", run
            result = client.get(run["result_url"]).json()["result"]
            assert len(result["periods"]) == 24
            assert result["backtest"]["real_time_mae_yuan_per_mwh"] is None
            assert "SPREAD_DIRECTION_ACCURACY_UNAVAILABLE" in result["warnings"]


def test_public_platform_serves_price_data_required_by_date_selector():
    with TestClient(main.app) as client:
        response = client.get("/data/spot-prices.json")
        assert response.status_code == 200
        rows = response.json()
        assert rows[0]["date"] == "2026-01-01"
        assert rows[-1]["date"] == "2026-07-31"
