"""Build versioned, read-only daily price forecasts for the public desk.

V6 is retained for every date January-September. V7's monthly walk-forward
research replay is additionally published for July-September. This never
creates trading instructions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.d1_shandong_price_forecast import MODEL_VERSION as V6_VERSION  # noqa: E402
from app.main import (  # noqa: E402
    ModelRunCreateRequestV1,
    build_price_forecast_v1_result,
)
from app.price_forecast_model import run_price_forecast  # noqa: E402
from app.v7_shandong_price_forecast import MODEL_VERSION as V7_VERSION  # noqa: E402

START = "2026-01-01"
END = "2026-09-30"
DATA_VERSION = "sd-16city-gfs24-actual-supply-20260101-20260930-d1-enhanced-v6"
PRIVATE = ROOT / "private-data" / "shandong-2026h1"
DESTINATION = ROOT / "data" / "forecast-daily"
ASSET = ROOT / "backend" / "model_assets" / V7_VERSION / "v7_walk_forward_predictions.csv.gz"


def build() -> dict:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    published_at = datetime.now(timezone.utc).isoformat()
    dates = pd.date_range(START, END, freq="D")
    index_rows: list[dict] = []
    month_rows: list[dict] = []
    current_month = ""

    def save_month(month: str, rows: list[dict]) -> None:
        if not rows:
            return
        target = DESTINATION / f"sd-{month}.json"
        document = {
            "schema_version": "sd-daily-forecast-v1",
            "market_code": "SD",
            "month": month,
            "generated_at": published_at,
            "results": rows,
            "execution_allowed": False,
        }
        target.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        days = len({row["market_date"] for row in rows})
        print(f"BUILT {month}: {days} days, {len(rows)} model results, {len(rows) * 24} periods, {target.stat().st_size} bytes", flush=True)

    for target in dates:
        day = target.date().isoformat()
        month = day[:7]
        if current_month and current_month != month:
            save_month(current_month, month_rows)
            month_rows = []
        current_month = month
        versions = [V6_VERSION] + ([V7_VERSION] if day >= "2026-07-01" else [])
        model_metrics: dict[str, dict] = {}
        for version in versions:
            raw = run_price_forecast(PRIVATE, day, model_version=version)
            request = ModelRunCreateRequestV1(
                request_id=f"published-{version}-{day}",
                market_code="SD", market_date=day, model_id="price-forecast",
                model_version=version, data_version=DATA_VERSION,
                input_summary={"available_domains": ["prices", "weather"]},
            )
            result = build_price_forecast_v1_result(request, f"published-{version}-{day}", raw).model_dump(mode="json")
            periods = result["periods"]
            assert result["model"]["version"] == version
            assert [row["period"] for row in periods] == list(range(1, 25))
            assert not result["strategy_ready"]
            assert all(row["strategy_suggestion"]["action"] == "HOLD"
                       and row["strategy_suggestion"]["volume_mwh"] == 0 for row in periods)
            # Static snapshots are not persisted model runs.
            result["run_id"] = None
            result["forecast_scenario"] = "pre_market_research_replay"
            result["audit"] = {
                "evaluation_mode": raw["summary"].get("evaluation_mode"),
                "training_end": raw["summary"].get("training_end"),
                "target_actual_supply_used": False,
                "price_actuals_used_only_for_scoring": True,
            }
            result["execution_allowed"] = False
            month_rows.append(result)
            model_metrics[version] = {
                "evaluation_mode": result["audit"]["evaluation_mode"],
                "day_ahead_mae_yuan_per_mwh": result["backtest"].get("mae_yuan_per_mwh"),
                "real_time_mae_yuan_per_mwh": result["backtest"].get("real_time_mae_yuan_per_mwh"),
            }
        default_version = versions[-1]
        index_rows.append({
            "market_date": day, "month": month,
            "model_version": default_version, "available_versions": versions,
            "evaluation_mode": model_metrics[default_version]["evaluation_mode"],
            "models": model_metrics,
        })
    save_month(current_month, month_rows)
    assert len(index_rows) == len(dates) == 273
    assert len({row["market_date"] for row in index_rows}) == 273
    result_count = sum(len(row["available_versions"]) for row in index_rows)
    assert result_count == 365
    manifest = {
        "schema_version": "sd-daily-forecast-index-v1",
        "market_code": "SD",
        "start_date": START, "end_date": END,
        "generated_at": published_at,
        "data_version": DATA_VERSION,
        "v7_replay_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "dates": index_rows,
        "day_count": len(index_rows), "result_count": result_count,
        "period_count": result_count * 24,
        "note": "V6 is available January-September; V7 is additionally available July-September. January V6 is in-sample diagnostic. V7 July-September is not an independent untouched holdout. Missing actuals remain null. No automatic trading.",
        "execution_allowed": False,
    }
    (DESTINATION / "index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"COMPLETE {len(index_rows)} days, {result_count} model results, {result_count * 24} periods", flush=True)
    return manifest


if __name__ == "__main__":
    build()
