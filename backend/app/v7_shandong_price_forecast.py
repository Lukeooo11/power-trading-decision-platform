"""Versioned July-September V7 walk-forward price forecast replay.

The asset preserves the monthly prediction-time models used in the V7
comparison. October is intentionally unavailable until its GFS snapshot and
prior settled inputs are built and independently checked.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .d1_shandong_price_forecast import _load as load_v6, run_d1_shandong_forecast
from .v5_shandong_price_forecast import _interval_coverage, _point_metrics

MODEL_VERSION = "sd-gfs24-d1-qra-regime-v7-research"
DATA_VERSION = "sd-16city-gfs24-actual-supply-20260101-20260930-d1-enhanced-v6"
SUPPORTED_START = "2026-07-01"
SUPPORTED_END = "2026-09-30"


@lru_cache(maxsize=1)
def _asset() -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = Path(__file__).resolve().parents[1] / "model_assets" / MODEL_VERSION
    replay = pd.read_csv(directory / "v7_walk_forward_predictions.csv.gz", compression="gzip")
    card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
    if replay.duplicated(["date", "period"]).any():
        raise ValueError("V7 asset has duplicate date/period")
    return replay, card


def run_v7_shandong_forecast(
    *, target_date: str, private_data_dir: Path,
    declaration_cutoff=None, high_price_threshold: float = 500.0,
) -> dict[str, Any]:
    if not SUPPORTED_START <= target_date <= SUPPORTED_END:
        raise ValueError(f"V7 research model validated range is {SUPPORTED_START} to {SUPPORTED_END}")
    replay, card = _asset()
    match = replay[replay["date"] == target_date].sort_values("period")
    if len(match) != 24 or match["period"].tolist() != list(range(1, 25)):
        raise ValueError(f"V7 requires exactly 24 period predictions for {target_date}")
    base = run_d1_shandong_forecast(
        target_date=target_date, private_data_dir=private_data_dir,
        declaration_cutoff=declaration_cutoff, high_price_threshold=high_price_threshold,
    )
    _, frame, _, _, _ = load_v6()
    actual = frame[frame["日期"] == pd.Timestamp(target_date)].sort_values("小时")
    if len(actual) != 24:
        raise ValueError(f"V7 actual-price join failed for {target_date}")
    if not np.array_equal(actual["小时"].to_numpy(int), match["period"].to_numpy(int)):
        raise ValueError("V7 period alignment failed")
    da_point = match["da_q50"].to_numpy(float)
    rt_point = match["rt_regime"].to_numpy(float)
    da_lo = match["da_q10"].to_numpy(float)
    da_hi = match["da_q90"].to_numpy(float)
    rt_lo = match["rt_q10"].to_numpy(float)
    rt_hi = match["rt_q90"].to_numpy(float)
    forecast = []
    for index, old in enumerate(base["forecast"]):
        da = (min(da_lo[index], da_point[index]), da_point[index], max(da_hi[index], da_point[index]))
        rt = (min(rt_lo[index], rt_point[index]), rt_point[index], max(rt_hi[index], rt_point[index]))
        width = max(da[2] - da[0], rt[2] - rt[0])
        forecast.append({
            **old,
            "da_p10": round(float(da[0]), 3), "da_p50": round(float(da[1]), 3),
            "da_p90": round(float(da[2]), 3),
            "rt_p10": round(float(rt[0]), 3), "rt_p50": round(float(rt[1]), 3),
            "rt_p90": round(float(rt[2]), 3),
            "negative_risk_probability": round(float(match.iloc[index]["p_negative"]), 6),
            "high_price_risk_probability": round(float(match.iloc[index]["p_spike"]), 6),
            "confidence": round(float(np.clip(.8 - width / 1400, .45, .70)), 3),
        })
    da_actual = pd.to_numeric(actual["日前价格"], errors="coerce")
    rt_actual = pd.to_numeric(actual["实时价格"], errors="coerce")
    spread_actual = pd.to_numeric(actual["实时价差"], errors="coerce")
    summary = {**base["summary"]}
    summary.update({
        "da_selected": "Regularized QRA P50",
        "rt_selected": "LightGBM three-regime probability-weighted price",
        "spread_selected": "DA minus RT P50",
        "model_routing": {"day_ahead": "QRA", "real_time": "LightGBM three-regime mixture",
                          "negative": "OOF-calibrated LightGBM probability",
                          "spike": "OOF-calibrated LightGBM probability"},
        "da_metrics": _point_metrics(da_actual, da_point, high_price_threshold),
        "rt_metrics": _point_metrics(rt_actual, rt_point, high_price_threshold),
        "spread_mae": _point_metrics(spread_actual, da_point - rt_point, high_price_threshold)["mae"],
        "spread_direction_accuracy": (
            float(np.mean(np.sign((da_point - rt_point)[spread_actual.notna().to_numpy()]) ==
                          np.sign(spread_actual.dropna().to_numpy(float))))
            if spread_actual.notna().any() else None
        ),
        "da_interval_coverage": _interval_coverage(da_actual, da_lo, da_hi),
        "rt_interval_coverage": _interval_coverage(rt_actual, rt_lo, rt_hi),
        "evaluation_method": "July-September monthly walk-forward V7 research replay; October remains untouched",
        "evaluation_mode": "MONTHLY_WALK_FORWARD_RESEARCH",
        "training_end": str(match.iloc[0]["training_end"]),
        "event_probability_calibration": "PRIOR_OOF_LOGIT",
        "forecast_selection": {"selected_model_version": MODEL_VERSION, "fallback_used": False,
                               "target_actual_supply_used": False,
                               "evaluation_mode": "MONTHLY_WALK_FORWARD_RESEARCH",
                               "training_end": str(match.iloc[0]["training_end"])},
        "model_card_comparison": card["metrics"],
        "interval_status": "RESEARCH_ONLY_NEGATIVE_TAIL_NOT_CALIBRATED",
        "extreme_event_warning": "Negative-price point error increased; risk probability and interval require manual review",
        "spread_output_policy": "DA P50 minus RT P50; no independent spread forecast in the V7 output",
    })
    return {**base, "model_version": MODEL_VERSION, "data_version": DATA_VERSION,
            "forecast": forecast, "summary": summary}
