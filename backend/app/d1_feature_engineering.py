"""Shared, declaration-time-safe feature engineering for the D-1 model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .v5_shandong_price_forecast import _features as legacy_features


PRICE_COLUMNS = ("日前价格", "实时价格", "实时价差")
EVENT_COLUMNS = ("负价事件", "尖峰事件")
SUPPLY_D1_COLUMNS = ("直调负荷", "风电", "光伏", "新能源出力", "简单净负荷")
D1_COLUMNS = (*PRICE_COLUMNS, *EVENT_COLUMNS, *SUPPLY_D1_COLUMNS)
NEW_FEATURE_COLUMNS: list[str] = []


def build_d1_features(source: pd.DataFrame) -> pd.DataFrame:
    frame = legacy_features(source)
    # Use an interpretable fixed price event threshold instead of a threshold
    # computed with future labels.
    frame["负价事件"] = np.where(frame["实时价格"].isna(), np.nan, (frame["实时价格"] < 0).astype(float))
    frame["尖峰事件"] = np.where(frame["实时价格"].isna(), np.nan, (frame["实时价格"] >= 500).astype(float))
    new: list[str] = []
    for event in EVENT_COLUMNS:
        for lag in (48, 72, 96, 120, 144, 168, 192):
            name = f"{event}_safe_lag{lag}"
            frame[name] = pd.to_numeric(frame[event], errors="coerce").shift(lag)
    for column in D1_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        lag24 = f"{column}_safe_lag24"
        frame[lag24] = values.shift(24)
        new.append(lag24)
        same_hour = []
        for lag in (24, 48, 72, 96, 120, 144, 168):
            name = f"{column}_safe_lag{lag}"
            if name not in frame:
                frame[name] = values.shift(lag)
            same_hour.append(name)
        mean_name = f"{column}_d1_7d_same_hour_mean"
        std_name = f"{column}_d1_7d_same_hour_std"
        short_name = f"{column}_d1_change_vs_d2"
        week_name = f"{column}_d1_change_vs_d7"
        dev_name = f"{column}_d1_deviation_from_7d"
        frame[mean_name] = frame[same_hour].mean(axis=1)
        frame[std_name] = frame[same_hour].std(axis=1)
        frame[short_name] = frame[lag24] - frame[f"{column}_safe_lag48"]
        frame[week_name] = frame[lag24] - frame[f"{column}_safe_lag168"]
        frame[dev_name] = frame[lag24] - frame[mean_name]
        new.extend([mean_name, std_name, short_name, week_name, dev_name])
        grouped = frame.groupby("日期")[lag24]
        for suffix, series in (("mean", grouped.transform("mean")), ("std", grouped.transform("std")), ("min", grouped.transform("min")), ("max", grouped.transform("max"))):
            name = f"{column}_d1_daily_{suffix}"
            frame[name] = series
            new.append(name)
    for column in ("2米气温_C_均值", "100米风速_mps_均值", "短波太阳辐射_Wm2_均值", "总云量_pct_均值", "降水_mm_均值"):
        if column in frame:
            name = f"{column}_target_vs_d1_change"
            frame[name] = pd.to_numeric(frame[column], errors="coerce") - pd.to_numeric(frame[column], errors="coerce").shift(24)
            new.append(name)
    frame["month_sin"] = np.sin(2 * np.pi * frame["月份"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["月份"] / 12)
    new.extend(["month_sin", "month_cos"])
    globals()["NEW_FEATURE_COLUMNS"] = list(dict.fromkeys(new))
    return frame.copy()


def enhanced_features(base_features: list[str], frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys([*base_features, *[c for c in NEW_FEATURE_COLUMNS if c in frame]]))
