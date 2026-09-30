"""D-1 settlement and measured-supply-aware mixed forecast asset for Shandong.

This module is deliberately separate from the v2 loader so the previous
safe-lag48plus model remains available for rollback and comparison.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v5_shandong_price_forecast import (
    _bounds,
    _interval_coverage,
    _point_metrics,
    _probability,
)
from .d1_feature_engineering import build_d1_features

MODEL_VERSION = "sd-gfs24-d1-price-actual-supply-mixed-v6"
DATA_VERSION = "sd-16city-gfs24-actual-supply-20260101-20260930-d1-enhanced-v6"
SUPPORTED_START = "2026-01-01"
SUPPORTED_END = "2026-09-30"
OOF_START = "2026-02-01"
OOF_END = "2026-09-30"
MODEL_FILES = {"da": "day_ahead.pkl", "rt": "real_time.pkl", "spread": "spread.pkl", "negative": "negative.pkl", "spike": "spike.pkl"}
PRICE_COLUMNS = ("日前价格", "实时价格", "实时价差")
EVENT_COLUMNS = ("负价事件", "尖峰事件")
SUPPLY_D1_COLUMNS = ("直调负荷", "风电", "光伏", "新能源出力", "简单净负荷")


def _features(source: pd.DataFrame) -> pd.DataFrame:
    return build_d1_features(source)


@lru_cache(maxsize=1)
def _load():
    model_dir = Path(__file__).resolve().parents[1] / "model_assets" / MODEL_VERSION
    feature_path = model_dir / "shandong_feature_store_20260101_20260930.csv.gz"
    required = [model_dir / filename for filename in MODEL_FILES.values()] + [
        model_dir / "model_metadata.json",
        model_dir / "residual_calibration.json",
        model_dir / "v5_expanding_oof_predictions.csv.gz",
        model_dir / "extended_backtest_metrics.json",
        feature_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("D-1 mixed model assets missing: " + ", ".join(missing))
    source = pd.read_csv(feature_path, compression="gzip", low_memory=False)
    source["日期"] = pd.to_datetime(source["date"]).dt.normalize()
    source["小时"] = source["period"].astype(int)
    source["有效时间"] = pd.to_datetime(source["validTime"])
    source["预报生成参考时间"] = pd.to_datetime(source["forecastIssueTime"])
    source["预测提前量_小时"] = pd.to_numeric(source["leadHours"])
    frame = _features(source)
    models = {}
    for key, filename in MODEL_FILES.items():
        with (model_dir / filename).open("rb") as handle:
            models[key] = pickle.load(handle)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_dir / "residual_calibration.json").read_text(encoding="utf-8"))
    return source, frame, models, metadata, calibration


@lru_cache(maxsize=1)
def _load_extended_backtest():
    model_dir = Path(__file__).resolve().parents[1] / "model_assets" / MODEL_VERSION
    predictions = pd.read_csv(model_dir / "v5_expanding_oof_predictions.csv.gz", compression="gzip")
    metrics = json.loads((model_dir / "extended_backtest_metrics.json").read_text(encoding="utf-8"))
    return predictions, metrics


def run_d1_shandong_forecast(
    *,
    target_date: str,
    private_data_dir: Path,
    declaration_cutoff: datetime | None = None,
    high_price_threshold: float = 500.0,
) -> dict[str, Any]:
    del private_data_dir
    if not SUPPORTED_START <= target_date <= SUPPORTED_END:
        raise ValueError(f"D-1 mixed model validated range is {SUPPORTED_START} to {SUPPORTED_END}")
    source, frame, models, metadata, calibration = _load()
    target = pd.Timestamp(target_date).normalize()
    raw_target = source[source["日期"] == target].sort_values("小时")
    if len(raw_target) != 24 or set(raw_target["小时"]) != set(range(1, 25)):
        raise ValueError(f"D-1 model requires exactly 24 rows for {target_date}")
    if not raw_target["预测提前量_小时"].eq(24).all():
        raise ValueError("weather lead is not exactly 24 hours")
    if not ((raw_target["有效时间"] - raw_target["预报生成参考时间"]).dt.total_seconds() / 3600).eq(24).all():
        raise ValueError("weather issue/valid time audit failed")
    cutoff_status, cutoff_iso = "UNVERIFIED_NO_CUTOFF_SUPPLIED", None
    if declaration_cutoff is not None:
        cutoff = pd.Timestamp(declaration_cutoff)
        issue = raw_target["预报生成参考时间"]
        if cutoff.tzinfo is None and issue.dt.tz is not None:
            cutoff = cutoff.tz_localize(issue.dt.tz)
        if cutoff.tzinfo is not None and issue.dt.tz is None:
            cutoff = cutoff.tz_localize(None)
        if (issue > cutoff).any():
            raise ValueError("GFS issue time later than declaration cutoff")
        cutoff_status, cutoff_iso = "VERIFIED", cutoff.isoformat()
    rows = frame[frame["日期"] == target].sort_values("小时")
    for key, package in models.items():
        absent = [column for column in package["features"] if column not in rows]
        if absent:
            raise KeyError(f"{key} missing features: {absent[:5]}")

    evaluation_mode = "EXPANDING_WINDOW_OOF"
    training_end = metadata.get("trainingCutoffByTarget", {}).get("rt", "2026-09-28")
    event_probability_calibration = "OOF_CALIBRATED"
    if OOF_START <= target_date <= OOF_END:
        extended, _ = _load_extended_backtest()
        replay = extended[extended["date"] == target_date].sort_values("period")
        if len(replay) != 24:
            raise ValueError(f"D-1 OOF replay requires exactly 24 rows for {target_date}")
        da = np.clip(replay["da_p50"].to_numpy(float), -100, 1300)
        rt = np.clip(replay["rt_p50"].to_numpy(float), -100, 1300)
        spread = replay["spread_p50"].to_numpy(float)
        negative = np.clip(replay["negative_probability"].to_numpy(float), 0, 1)
        spike = np.clip(replay["spike_probability"].to_numpy(float), 0, 1)
        evaluation_mode = "EXPANDING_WINDOW_OOF"
        training_end = str(replay.iloc[0]["training_end"])
        event_probability_calibration = "RAW_UNCALIBRATED_OOF"
    else:
        da = np.clip(models["da"]["model"].predict(rows[models["da"]["features"]]), -100, 1300)
        rt = np.clip(models["rt"]["model"].predict(rows[models["rt"]["features"]]), -100, 1300)
        spread = models["spread"]["model"].predict(rows[models["spread"]["features"]])
        negative = np.clip(_probability(models["negative"], rows), 0, 1)
        spike = np.clip(_probability(models["spike"], rows), 0, 1)
        if target_date < OOF_START:
            evaluation_mode = "IN_SAMPLE_DIAGNOSTIC"
            training_end = metadata.get("trainingCutoffByTarget", {}).get("rt", "2026-09-28")

    periods = rows["小时"].to_numpy(int)
    da10, da90, _ = _bounds(da, periods, "da", calibration)
    rt10, rt90, _ = _bounds(rt, periods, "rt", calibration)
    forecast = []
    for index, period in enumerate(periods):
        dq = sorted([float(np.clip(da10[index], -100, 1300)), float(da[index]), float(np.clip(da90[index], -100, 1300))])
        rq = sorted([float(np.clip(rt10[index], -100, 1300)), float(rt[index]), float(np.clip(rt90[index], -100, 1300))])
        width = max(dq[2] - dq[0], rq[2] - rq[0])
        forecast.append({"period": int(period), "datetime": rows.iloc[index]["有效时间"].isoformat(), "da_p10": round(dq[0], 3), "da_p50": round(dq[1], 3), "da_p90": round(dq[2], 3), "rt_p10": round(rq[0], 3), "rt_p50": round(rq[1], 3), "rt_p90": round(rq[2], 3), "negative_risk_probability": round(float(negative[index]), 6), "high_price_risk_probability": round(float(spike[index]), 6), "confidence": round(float(np.clip(.86 - width / 1600, .5, .82)), 3)})
    da_metrics = _point_metrics(rows["日前价格"], da, high_price_threshold)
    rt_metrics = _point_metrics(rows["实时价格"], rt, high_price_threshold)
    spread_metrics = _point_metrics(rows["实时价差"], spread, high_price_threshold)
    actual_spread = pd.to_numeric(rows["实时价差"], errors="coerce").to_numpy(float)
    valid_spread = np.isfinite(actual_spread) & np.isfinite(spread)
    spread_direction_accuracy = float(np.mean(np.sign(spread[valid_spread]) == np.sign(actual_spread[valid_spread]))) if valid_spread.any() else None
    return {
        "model_version": MODEL_VERSION,
        "data_version": DATA_VERSION,
        "forecast": forecast,
        "summary": {
            "da_selected": metadata.get("selected", {}).get("da", {}).get("algorithm", "XGBoost"),
            "rt_selected": metadata.get("selected", {}).get("rt", {}).get("algorithm", "LightGBM"),
            "spread_selected": metadata.get("selected", {}).get("spread", {}).get("algorithm", "LightGBM"),
            "model_routing": {
                "day_ahead": metadata.get("selected", {}).get("da", {}).get("algorithm", "XGBoost"),
                "real_time": metadata.get("selected", {}).get("rt", {}).get("algorithm", "LightGBM"),
                "spread": metadata.get("selected", {}).get("spread", {}).get("algorithm", "LightGBM"),
                "negative": metadata.get("selected", {}).get("negative", {}).get("algorithm", "LightGBM"),
                "spike": metadata.get("selected", {}).get("spike", {}).get("algorithm", "LightGBM"),
            },
            "da_metrics": da_metrics,
            "rt_metrics": rt_metrics,
            "window_start": target_date,
            "window_end": target_date,
            "sample_count": int(pd.to_numeric(rows["日前价格"], errors="coerce").notna().sum()),
            "forecast_start": forecast[0]["datetime"],
            "forecast_end": forecast[-1]["datetime"],
            "evaluation_method": "monthly expanding-window OOF through September; robust feature selection across July-September; October is the next untouched holdout",
            "evaluation_mode": evaluation_mode,
            "training_end": training_end,
            "event_probability_calibration": event_probability_calibration,
            "feature_set": "16-city GFS24 + D-1 settlement/load/wind/solar + seven-day same-hour statistics + D-1 daily shape and weather-change features; event heads retain baseline features when stronger",
            "spread_output_policy": "independent LightGBM spread is retained for audit and strategy scoring; public contract displays DA P50 minus RT P50 for consistency",
            "weather_data_version": DATA_VERSION,
            "weather_known_before_declaration": cutoff_status == "VERIFIED",
            "weather_used_in_da_final": True,
            "weather_used_in_rt_final": True,
            "supply_data_version": "actual-supply-20260101-20260928-safe-lag24plus",
            "supply_used_in_da_final": True,
            "supply_used_in_rt_final": True,
            "supply_issue_time_available": False,
            "supply_backtest_leakage_safe": evaluation_mode != "IN_SAMPLE_DIAGNOSTIC",
            "supply_usage_boundary": "D-1实际负荷、风电、光伏及衍生出力已加入lag24；目标日实际出力未使用",
            "price_information_boundary": "D-1日前和D-1实时结算价格在目标交易前可用",
            "d1_actual_supply_used": True,
            "forecast_supply_pending": True,
            "spread_direction_accuracy": spread_direction_accuracy,
            "spread_mae": spread_metrics["mae"],
            "da_interval_coverage": _interval_coverage(rows["日前价格"], da10, da90),
            "rt_interval_coverage": _interval_coverage(rows["实时价格"], rt10, rt90),
            "declaration_cutoff_audit": {"status": cutoff_status, "declaration_cutoff": cutoff_iso, "issue_time_max": raw_target["预报生成参考时间"].max().isoformat(), "lead_hours": 24},
            "forecast_selection": {"selected_model_version": MODEL_VERSION, "fallback_used": False, "target_actual_supply_used": False, "evaluation_mode": evaluation_mode, "training_end": training_end},
            "model_card_september_metrics": metadata.get("septemberMetrics"),
            "model_card_september_events": metadata.get("septemberEvents"),
        },
    }
