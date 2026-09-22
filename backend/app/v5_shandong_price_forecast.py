from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
import pickle
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

MODEL_VERSION = "sd-gfs24-spatial-lgbm-v1"
DATA_VERSION = "sd-16city-gfs-fixed-lead24-20260101-20260831-v1"
SUPPORTED_START = "2026-07-01"
SUPPORTED_END = "2026-07-31"
MODEL_FILES = {"da": "day_ahead.pkl", "rt": "real_time.pkl", "spread": "spread.pkl", "negative": "negative.pkl", "spike": "spike.pkl"}
HISTORY_COLUMNS = ["日前价格", "实时价格", "实时价差", "直调负荷", "联络线", "风电", "光伏", "地方电厂", "自备机组", "非市场化核电总加", "新能源出力", "简单净负荷"]
LAGS = [48, 72, 96, 120, 144, 168, 192]


def _features(source: pd.DataFrame) -> pd.DataFrame:
    d = source.sort_values(["日期", "小时"]).reset_index(drop=True).copy()
    d["月份"] = d["日期"].dt.month
    d["星期"] = d["日期"].dt.weekday
    d["是否周末"] = (d["星期"] >= 5).astype(int)
    d["年内日"] = d["日期"].dt.dayofyear
    d["小时_sin"] = np.sin(2*np.pi*d["小时"]/24)
    d["小时_cos"] = np.cos(2*np.pi*d["小时"]/24)
    d["星期_sin"] = np.sin(2*np.pi*d["星期"]/7)
    d["星期_cos"] = np.cos(2*np.pi*d["星期"]/7)
    d["是否供暖季"] = d["月份"].isin([11, 12, 1, 2, 3]).astype(int)
    d["是否迎峰度夏"] = d["月份"].isin([6, 7, 8]).astype(int)
    d["是否午间"] = d["小时"].between(10, 15).astype(int)
    d["是否晚高峰"] = d["小时"].between(17, 21).astype(int)
    for column in HISTORY_COLUMNS:
        lagged = []
        values = pd.to_numeric(d[column], errors="coerce")
        for lag in LAGS:
            name = f"{column}_safe_lag{lag}"
            d[name] = values.shift(lag)
            lagged.append(name)
        d[f"{column}_safe_7日同小时均值"] = d[lagged].mean(axis=1)
        d[f"{column}_safe_7日同小时标准差"] = d[lagged].std(axis=1)
        d[f"{column}_safe_短期变化"] = d[lagged[0]] - d[lagged[1]]
    spike = float(d.loc[d["日期"] <= pd.Timestamp("2026-06-30"), "实时价格"].quantile(.95))
    d["负价事件"] = (d["实时价格"] < 0).astype(float)
    d.loc[d["实时价格"].isna(), "负价事件"] = np.nan
    d["尖峰事件"] = (d["实时价格"] >= spike).astype(float)
    d.loc[d["实时价格"].isna(), "尖峰事件"] = np.nan
    for event in ["负价事件", "尖峰事件"]:
        lagged = []
        for lag in LAGS:
            name = f"{event}_safe_lag{lag}"
            d[name] = d[event].shift(lag)
            lagged.append(name)
        d[f"{event}_safe_7日发生率"] = d[lagged].mean(axis=1)
    wind = [c for c in d.columns if "风向" in c and "safe_" not in c and pd.api.types.is_numeric_dtype(d[c])]
    for column in wind:
        radians = np.deg2rad(d[column])
        d[column+"_sin"] = np.sin(radians)
        d[column+"_cos"] = np.cos(radians)
    for column in ["2米气温_C_均值", "100米风速_mps_均值", "短波太阳辐射_Wm2_均值", "总云量_pct_均值", "降水_mm_均值"]:
        group = d.groupby("日期")[column]
        d[column+"_日均"] = group.transform("mean")
        d[column+"_日最大"] = group.transform("max")
        d[column+"_日最小"] = group.transform("min")
        d[column+"_日标准差"] = group.transform("std")
    d["短波太阳辐射_Wm2_日总和"] = d.groupby("日期")["短波太阳辐射_Wm2_均值"].transform("sum")
    d["日温差"] = d["2米气温_C_均值_日最大"] - d["2米气温_C_均值_日最小"]
    d["GFS光伏压力"] = d["短波太阳辐射_Wm2_均值"]*(1-d["总云量_pct_均值"]/100)
    d["GFS风电势能"] = d["100米风速三次方_均值"]
    d["GFS午间光伏压力"] = d["GFS光伏压力"]*d["是否午间"]
    d["GFS高温高湿"] = ((d["2米气温_C_均值"] >= 28) & (d["2米相对湿度_pct_均值"] >= 70)).astype(int)
    return d


@lru_cache(maxsize=1)
def _load():
    # Local layout: <repo>/backend/app + <repo>/backend/model_assets.
    # Render layout: /app/app + /app/model_assets.
    model_dir = Path(__file__).resolve().parents[1] / "model_assets" / MODEL_VERSION
    feature_path = model_dir / "v5_gfs24_feature_store.csv.gz"
    required = [model_dir / v for v in MODEL_FILES.values()] + [model_dir/"model_metadata.json", model_dir/"residual_calibration.json", feature_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("V5 assets missing: " + ", ".join(missing))
    source = pd.read_csv(feature_path, compression="gzip", low_memory=False)
    source["日期"] = pd.to_datetime(source["date"]).dt.normalize()
    source["小时"] = source["period"].astype(int)
    source["有效时间"] = pd.to_datetime(source["validTime"])
    source["预报生成参考时间"] = pd.to_datetime(source["forecastIssueTime"])
    source["预测提前量_小时"] = pd.to_numeric(source["leadHours"])
    models = {}
    for key, filename in MODEL_FILES.items():
        with (model_dir/filename).open("rb") as handle:
            models[key] = pickle.load(handle)
    metadata = json.loads((model_dir/"model_metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_dir/"residual_calibration.json").read_text(encoding="utf-8"))
    return source, _features(source), models, metadata, calibration


def _probability(package, rows):
    raw = package["model"].predict_proba(rows[package["features"]])[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return package["calibrator"].predict_proba(pd.DataFrame({"p": raw}))[:, 1]


def _bounds(point, periods, key, calibration):
    cfg = calibration["targets"][key]
    low, high = [], []
    for value, period in zip(point, periods):
        item = cfg["byPeriod"].get(str(int(period)), cfg)
        low.append(value + item["q10"])
        high.append(value + item["q90"])
    return np.asarray(low), np.asarray(high), float(cfg["coverage"])


def run_v5_shandong_forecast(*, target_date: str, private_data_dir: Path, declaration_cutoff: datetime | None = None, high_price_threshold: float = 500.0) -> dict[str, Any]:
    del private_data_dir  # Signature retained for compatibility with the platform adapter.
    if not SUPPORTED_START <= target_date <= SUPPORTED_END:
        raise ValueError(f"V5 validated range is {SUPPORTED_START} to {SUPPORTED_END}")
    source, frame, models, meta, calibration = _load()
    target = pd.Timestamp(target_date).normalize()
    raw_target = source[source["日期"] == target].sort_values("小时")
    if len(raw_target) != 24 or set(raw_target["小时"]) != set(range(1, 25)):
        raise ValueError(f"V5 requires exactly 24 rows for {target_date}")
    if not raw_target["预测提前量_小时"].eq(24).all():
        raise ValueError("weather lead is not exactly 24 hours")
    if not ((raw_target["有效时间"]-raw_target["预报生成参考时间"]).dt.total_seconds()/3600).eq(24).all():
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
        absent = [c for c in package["features"] if c not in rows]
        if absent:
            raise KeyError(f"{key} missing features: {absent[:5]}")
    da = np.clip(models["da"]["model"].predict(rows[models["da"]["features"]]), -100, 1300)
    rt = np.clip(models["rt"]["model"].predict(rows[models["rt"]["features"]]), -100, 1300)
    spread = models["spread"]["model"].predict(rows[models["spread"]["features"]])
    negative = np.clip(_probability(models["negative"], rows), 0, 1)
    spike = np.clip(_probability(models["spike"], rows), 0, 1)
    periods = rows["小时"].to_numpy(int)
    da10, da90, da_cov = _bounds(da, periods, "da", calibration)
    rt10, rt90, rt_cov = _bounds(rt, periods, "rt", calibration)
    forecast = []
    for i, period in enumerate(periods):
        dq = sorted([float(np.clip(da10[i], -100, 1300)), float(da[i]), float(np.clip(da90[i], -100, 1300))])
        rq = sorted([float(np.clip(rt10[i], -100, 1300)), float(rt[i]), float(np.clip(rt90[i], -100, 1300))])
        width = max(dq[2]-dq[0], rq[2]-rq[0])
        forecast.append({"period": int(period), "datetime": rows.iloc[i]["有效时间"].isoformat(), "da_p10": round(dq[0],3), "da_p50": round(dq[1],3), "da_p90": round(dq[2],3), "rt_p10": round(rq[0],3), "rt_p50": round(rq[1],3), "rt_p90": round(rq[2],3), "negative_risk_probability": round(float(negative[i]),6), "high_price_risk_probability": round(float(spike[i]),6), "confidence": round(float(np.clip(.86-width/1600,.5,.82)),3)})
    metrics = meta["julyMetrics"]
    return {"model_version": MODEL_VERSION, "data_version": DATA_VERSION, "forecast": forecast, "summary": {"da_selected": "V5 GFS完整+日内形态 LightGBM", "rt_selected": "V5 GFS完整+日内形态 LightGBM", "da_metrics": metrics["dayAhead"], "rt_metrics": metrics["realTime"], "window_start": "2026-07-01", "window_end": "2026-07-31", "sample_count": 744, "forecast_start": forecast[0]["datetime"], "forecast_end": forecast[-1]["datetime"], "evaluation_method": "expanding-window OOF selection plus frozen July holdout", "feature_set": "safe-lag48plus + 16-city GFS24 spatial distribution and daily shape", "weather_data_version": DATA_VERSION, "weather_known_before_declaration": cutoff_status == "VERIFIED", "weather_used_in_da_final": True, "weather_used_in_rt_final": True, "supply_data_version": "embedded-v5-safe-lag-history", "supply_used_in_da_final": True, "supply_used_in_rt_final": True, "supply_issue_time_available": False, "supply_backtest_leakage_safe": True, "supply_usage_boundary": "lag48 or older only; target-date actual supply excluded", "spread_direction_accuracy": metrics["spreadDirectionAccuracy"], "spread_mae": metrics["spreadMae"], "da_interval_coverage": da_cov, "rt_interval_coverage": rt_cov, "consistency_constraint": "platform spread = DA p50 - RT p50; V5 direct spread retained in audit", "post_day_ahead_realtime": {"status": "UNCHANGED_PLATFORM_PATH"}, "declaration_cutoff_audit": {"status": cutoff_status, "declaration_cutoff": cutoff_iso, "issue_time_max": raw_target["预报生成参考时间"].max().isoformat(), "lead_hours": 24}, "forecast_selection": {"selected_model_version": MODEL_VERSION, "fallback_used": False, "target_actual_supply_used": False}, "v5_direct_spread_p50": [round(float(v),3) for v in spread]}}
