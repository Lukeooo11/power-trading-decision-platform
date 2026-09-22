"""河南价格预测适配器。

本模块复用山东模型的核心思想（价格滞后、日历、天气、供需和新能源特征），
但不复用山东参数或山东数据。河南分别输出：

* ``day_ahead``：日前价格，标签截止到 D-1；
* ``real_time_pre_da``：盘前实时价格，预测跨度为 D+2；若在当日结束前生成，
  最新完整实时标签相对目标日截止到 D-3，不使用目标日日前出清价；
* ``real_time_post_da``：日前出清后实时价格，允许使用目标日已出清日前价。

天气和河南电网预测表缺少发布时间，因此当前结果明确标记为研究性回测。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional runtime dependency
    XGBRegressor = None  # type: ignore[assignment,misc]


MODES = {
    "day_ahead": {"target": "da", "cutoff_days": 1, "use_same_day_da": False, "target_lags": [1, 2, 3, 7, 14]},
    "real_time_pre_da": {"target": "rt", "cutoff_days": 3, "use_same_day_da": False, "target_lags": [3, 4, 7, 14]},
    "real_time_post_da": {"target": "rt", "cutoff_days": 1, "use_same_day_da": True, "target_lags": [2, 3, 7, 14]},
}
PRE_DA_L1_WEIGHT = 0.6
PRE_DA_ANCHOR_WEIGHTS = {
    "night": {"periods": range(1, 8), "weight": 0.40},
    "morning": {"periods": range(8, 12), "weight": 0.30},
    "solar_core": {"periods": range(12, 18), "weight": 0.00},
    "evening": {"periods": range(18, 25), "weight": 0.25},
}
HENAN_PRE_DA_MODEL_ID = "henan-real-time-pre-da-da-anchor-v3"
RELIABILITY_PROFILE_PATH = Path(__file__).resolve().parent / "model_cards" / "henan_pre_da_period_reliability_v3.json"
AUXILIARY_MODES = {
    "day_ahead_d2": {"target": "da", "cutoff_days": 2, "use_same_day_da": False, "target_lags": [2, 3, 4, 7, 14]},
    "spread_pre_da": {"target": "spread", "cutoff_days": 3, "use_same_day_da": False, "target_lags": [3, 4, 7, 14]},
}
MODE_CONFIGS = {**MODES, **AUXILIARY_MODES}
WEATHER_FEATURES = ["temperature2mC", "apparentTemperatureC", "precipitationMm", "windSpeed10mMs", "relativeHumidityPct"]
POWER_FEATURES = ["loadForecastMw", "interconnectorMw", "totalOutputMw", "nonSpotOutputMw", "renewableMw", "hydroMw", "pumpedStorageMw", "windForecastMw", "solarForecastMw"]


def _metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae_yuan_per_mwh": float(np.mean(np.abs(error))),
        "rmse_yuan_per_mwh": float(np.sqrt(np.mean(error**2))),
        "bias_yuan_per_mwh": float(np.mean(error)),
        "sample_count": int(len(error)),
    }


def _point_accuracy_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Return price-safe accuracy measures for one fixed delivery period.

    Percentage error is intentionally excluded because Henan prices can be zero
    or negative. Tolerance hit rates stay interpretable across those regimes.
    """
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    error = predicted_values - actual_values
    absolute_error = np.abs(error)
    result: dict[str, float | int | None] = {
        **_metric(actual_values, predicted_values),
        "mean_actual_yuan_per_mwh": float(np.mean(actual_values)),
        "mean_predicted_yuan_per_mwh": float(np.mean(predicted_values)),
        "within_20_yuan_accuracy": float(np.mean(absolute_error <= 20.0)),
        "within_50_yuan_accuracy": float(np.mean(absolute_error <= 50.0)),
        "within_100_yuan_accuracy": float(np.mean(absolute_error <= 100.0)),
        "p10_p90_coverage": None,
    }
    if lower is not None and upper is not None:
        lower_values = np.asarray(lower, dtype=float)
        upper_values = np.asarray(upper, dtype=float)
        result["p10_p90_coverage"] = float(
            np.mean((actual_values >= lower_values) & (actual_values <= upper_values))
        )
    return result


def _as_frame(data_dir: Path, include_future: bool = False) -> pd.DataFrame:
    prices = pd.DataFrame(json.loads((data_dir / "prices_hourly.json").read_text(encoding="utf-8")))
    prices = prices.rename(columns={"marketDate": "date", "dayAheadPriceYuanMwh": "da", "realTimePriceYuanMwh": "rt"})
    prices["date"] = pd.to_datetime(prices["date"])
    prices["period"] = prices["period"].astype(int)
    for column in ["da", "rt"]:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices["spread"] = prices["rt"] - prices["da"]

    def read(name: str) -> pd.DataFrame:
        frame = pd.DataFrame(json.loads((data_dir / name).read_text(encoding="utf-8")))
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame.pop("marketDate"))
        frame["period"] = frame["period"].astype(int)
        return frame

    power = read("power_forecast_hourly.json")
    renewable = read("renewable_forecast_hourly.json")
    weather = read("weather_hourly_province.json")
    for frame in (power, renewable, weather):
        if not frame.empty:
            frame.drop(columns=[column for column in ["cityCount"] if column in frame], inplace=True)
    if include_future:
        key_frames = [item[["date", "period"]] for item in (prices, power, renewable, weather) if not item.empty]
        skeleton = pd.concat(key_frames, ignore_index=True).drop_duplicates().sort_values(["date", "period"])
        merged = skeleton.merge(prices, on=["date", "period"], how="left")
    else:
        merged = prices.copy()
    merged = merged.merge(power, on=["date", "period"], how="left", suffixes=("", "_power"))
    merged = merged.merge(renewable, on=["date", "period"], how="left", suffixes=("", "_renewable"))
    merged = merged.merge(weather, on=["date", "period"], how="left", suffixes=("", "_weather"))
    merged["netLoadForecastMw"] = merged["loadForecastMw"] - merged["renewableMw"]
    merged["date_only"] = merged["date"].dt.date.astype(str)
    return merged.sort_values(["date", "period"]).reset_index(drop=True)


def load_henan_frame(data_dir: Path) -> pd.DataFrame:
    """Load the normalized private snapshot and return a deterministic frame."""
    frame = _as_frame(data_dir)
    if frame.empty:
        raise ValueError("河南数据快照为空")
    return frame


def _load_period_reliability() -> dict[str, Any] | None:
    if not RELIABILITY_PROFILE_PATH.exists():
        return None
    profile = json.loads(RELIABILITY_PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("modelId") != HENAN_PRE_DA_MODEL_ID:
        return None
    periods = profile.get("periods") or []
    if len(periods) != 24 or {int(item["period"]) for item in periods} != set(range(1, 25)):
        return None
    return profile


def _safe_value(indexed: pd.DataFrame, date: pd.Timestamp, period: int, column: str) -> float | None:
    result = indexed.loc[(indexed["date"] == date) & (indexed["period"] == period), column]
    if result.empty or pd.isna(result.iloc[0]):
        return None
    return float(result.iloc[0])


def build_feature_frame(
    frame: pd.DataFrame,
    mode: str,
    feature_set: str = "base",
) -> tuple[pd.DataFrame, list[str]]:
    """Create leakage-safe rows for a mode using only fixed-date lags."""
    if mode not in MODE_CONFIGS:
        raise ValueError(f"unsupported mode: {mode}")
    if feature_set not in {"base", "enhanced"}:
        raise ValueError(f"unsupported feature set: {feature_set}")
    config = MODE_CONFIGS[mode]
    indexed = frame.set_index(["date", "period"], drop=False)

    def value_at(date: pd.Timestamp, period: int, column: str) -> float | None:
        try:
            value = indexed.at[(date, period), column]
        except (KeyError, ValueError):
            return None
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return None if pd.isna(value) else float(value)

    def values_at(date: pd.Timestamp, period: int, column: str, lags: list[int]) -> list[float]:
        values = [value_at(date - pd.Timedelta(days=lag), period, column) for lag in lags]
        return [value for value in values if value is not None]

    target_lookup_column = f"{config['target']}_feature" if f"{config['target']}_feature" in frame else config["target"]
    da_lookup_column = "da_feature" if "da_feature" in frame else "da"
    rt_lookup_column = "rt_feature" if "rt_feature" in frame else "rt"
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        date = pd.Timestamp(row.date)
        period = int(row.period)
        values: dict[str, Any] = {"date": date, "period": period, "target": getattr(row, config["target"])}
        hour = period - 1
        values.update(
            {
                "hour": hour,
                "dow": float(date.dayofweek),
                "month": float(date.month),
                "dayofyear": float(date.dayofyear),
                "hour_sin": math.sin(2 * math.pi * hour / 24),
                "hour_cos": math.cos(2 * math.pi * hour / 24),
                "dow_sin": math.sin(2 * math.pi * date.dayofweek / 7),
                "dow_cos": math.cos(2 * math.pi * date.dayofweek / 7),
                "is_weekend": float(date.dayofweek >= 5),
            }
        )
        target_lags = list(config["target_lags"])
        for lag in target_lags:
            value = value_at(date - pd.Timedelta(days=lag), period, target_lookup_column)
            values[f"{config['target']}_lag_{lag}d"] = value
        if config["target"] == "rt":
            for lag in [2, 3, 7, 14]:
                value = value_at(date - pd.Timedelta(days=lag), period, da_lookup_column)
                values[f"da_lag_{lag}d"] = value
        if config["use_same_day_da"]:
            values["same_day_da_cleared"] = value_at(date, period, da_lookup_column)
        else:
            values["same_day_da_cleared"] = 0.0
        for name in POWER_FEATURES + WEATHER_FEATURES + ["netLoadForecastMw"]:
            value = getattr(row, name, None)
            values[name] = None if value is None or pd.isna(value) else float(value)
        if feature_set == "enhanced":
            first_lag = int(config["cutoff_days"])
            for window in (3, 7, 14):
                history = values_at(date, period, target_lookup_column, list(range(first_lag, first_lag + window)))
                values[f"{config['target']}_same_hour_mean_{window}d"] = float(np.mean(history)) if history else None
                if window >= 7:
                    values[f"{config['target']}_same_hour_std_{window}d"] = float(np.std(history)) if history else None
            if config["target"] == "rt":
                da_history = values_at(date, period, da_lookup_column, list(range(first_lag, first_lag + 7)))
                values["da_same_hour_mean_7d"] = float(np.mean(da_history)) if da_history else None
                values["da_same_hour_std_7d"] = float(np.std(da_history)) if da_history else None
                for lag in dict.fromkeys(config["target_lags"]):
                    rt_lag = value_at(date - pd.Timedelta(days=lag), period, rt_lookup_column)
                    da_lag = value_at(date - pd.Timedelta(days=lag), period, da_lookup_column)
                    values[f"spread_lag_{lag}d"] = None if rt_lag is None or da_lag is None else rt_lag - da_lag
            load = values.get("loadForecastMw")
            renewable = values.get("renewableMw")
            wind = values.get("windForecastMw")
            solar = values.get("solarForecastMw")
            values["renewable_load_ratio"] = None if load in {None, 0.0} or renewable is None else renewable / load
            values["wind_solar_load_ratio"] = None if load in {None, 0.0} or wind is None or solar is None else (wind + solar) / load
            values["is_morning_peak"] = float(7 <= hour <= 10)
            values["is_evening_peak"] = float(17 <= hour <= 21)
            values["is_solar_window"] = float(9 <= hour <= 16)
            for column in ("loadForecastMw", "renewableMw", "windForecastMw", "solarForecastMw", "netLoadForecastMw"):
                current = values.get(column)
                previous = value_at(date, period - 1, column) if period > 1 else current
                values[f"{column}_ramp"] = None if current is None or previous is None else current - previous
        rows.append(values)
    result = pd.DataFrame(rows)
    excluded = {"date", "period", "target"}
    feature_columns = [column for column in result.columns if column not in excluded]
    return result, feature_columns


def _model(kind: str) -> Any:
    if kind in {"xgboost", "xgboost_l1"} and XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=8.0,
            objective="reg:absoluteerror" if kind == "xgboost_l1" else "reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=17,
            verbosity=0,
        )
    return make_pipeline(StandardScaler(), Ridge(alpha=20.0))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], kind: str) -> np.ndarray:
    model = _model(kind)
    x_train = train[columns].apply(pd.to_numeric, errors="coerce")
    x_test = test[columns].apply(pd.to_numeric, errors="coerce")
    medians = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(medians).fillna(0.0)
    x_test = x_test.fillna(medians).fillna(0.0)
    model.fit(x_train, train["target"].astype(float))
    return np.asarray(model.predict(x_test), dtype=float)


def _complete(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return rows.loc[rows["target"].notna() & rows[columns].notna().sum(axis=1).ge(max(1, len(columns) - 2))].copy()


@dataclass
class ModeForecast:
    mode: str
    target_date: str
    rows: list[dict[str, Any]]
    metrics: dict[str, Any]
    selected_model: str
    training_cutoff_date: str
    validation_window: dict[str, str | None]


def _pre_da_anchor_weight(period: int) -> tuple[str, float]:
    for segment, config in PRE_DA_ANCHOR_WEIGHTS.items():
        if period in config["periods"]:
            return segment, float(config["weight"])
    raise ValueError(f"unsupported delivery period: {period}")


def _apply_pre_da_da_anchor(
    realtime: ModeForecast,
    day_ahead_d2: ModeForecast,
) -> ModeForecast:
    """Anchor pre-DA RT point forecasts to a separately forecast D+2 DA level."""
    anchor_by_period = {int(item["period"]): item for item in day_ahead_d2.rows}
    rows: list[dict[str, Any]] = []
    for item in realtime.rows:
        period = int(item["period"])
        segment, weight = _pre_da_anchor_weight(period)
        base_p50 = float(item["p50"])
        anchor_p50 = float(anchor_by_period[period]["p50"])
        p50 = (1.0 - weight) * base_p50 + weight * anchor_p50
        shift = p50 - base_p50
        rows.append(
            {
                **item,
                "p10": round(float(np.clip(float(item["p10"]) + shift, -100.0, 2000.0)), 3),
                "p50": round(float(np.clip(p50, -100.0, 2000.0)), 3),
                "p90": round(float(np.clip(float(item["p90"]) + shift, -100.0, 2000.0)), 3),
                "baseP50": round(base_p50, 3),
                "dayAheadD2AnchorP50": round(anchor_p50, 3),
                "anchorSegment": segment,
                "anchorWeight": weight,
            }
        )
    return ModeForecast(
        mode=realtime.mode,
        target_date=realtime.target_date,
        rows=rows,
        metrics={
            **realtime.metrics,
            "point_optimization": {
                "method": "segmented_D_plus_2_day_ahead_anchor",
                "weights": {
                    name: float(config["weight"])
                    for name, config in PRE_DA_ANCHOR_WEIGHTS.items()
                },
                "interval_policy": "preserve direct RT interval width and shift with anchored P50",
            },
        },
        selected_model=f"{realtime.selected_model}+segmented_da_anchor",
        training_cutoff_date=realtime.training_cutoff_date,
        validation_window=realtime.validation_window,
    )


def forecast_mode(
    feature_frame: pd.DataFrame,
    columns: list[str],
    target_date: str,
    mode: str,
    point_calibration: str = "none",
) -> ModeForecast:
    if mode not in MODE_CONFIGS:
        raise ValueError(f"unsupported mode: {mode}")
    if point_calibration not in {"none", "median", "pre_da_median"}:
        raise ValueError(f"unsupported point calibration: {point_calibration}")
    target = pd.Timestamp(target_date)
    cutoff = target - pd.Timedelta(days=int(MODE_CONFIGS[mode]["cutoff_days"]))
    eligible = _complete(feature_frame, columns)
    train = eligible.loc[eligible["date"] <= cutoff]
    target_rows = feature_frame.loc[feature_frame["date"] == target].sort_values("period").copy()
    if len(target_rows) != 24:
        raise ValueError(f"{mode} target {target_date} has {len(target_rows)} feature rows; 24 required")
    if len(train) < 24 * 21:
        raise ValueError(f"{mode} needs at least 21 historical days, got {len(train)} rows")
    validation_start = cutoff - pd.Timedelta(days=20)
    validation = train.loc[train["date"] >= validation_start]
    fit_train = train.loc[train["date"] < validation_start]
    if len(validation) < 24 * 7:
        validation = train.tail(24 * 14)
        fit_train = train.iloc[:-len(validation)]
    candidates: dict[str, dict[str, Any]] = {}
    for kind in (["xgboost", "ridge"] if XGBRegressor is not None else ["ridge"]):
        validation_pred = _fit_predict(fit_train, validation, columns, kind)
        candidates[kind] = {"validation": validation_pred, "metrics": _metric(validation["target"].to_numpy(float), validation_pred)}
    ranked = sorted(candidates, key=lambda name: candidates[name]["metrics"]["mae_yuan_per_mwh"])
    selected = ranked[0]
    blend = None
    if len(ranked) > 1:
        best = candidates[ranked[0]]["metrics"]["mae_yuan_per_mwh"]
        second = candidates[ranked[1]]["metrics"]["mae_yuan_per_mwh"]
        if second <= best * 1.06:
            blend = (0.7, ranked[0], 0.3, ranked[1])
            blend_pred = 0.7 * candidates[ranked[0]]["validation"] + 0.3 * candidates[ranked[1]]["validation"]
            blend_metrics = _metric(validation["target"].to_numpy(float), blend_pred)
            if blend_metrics["mae_yuan_per_mwh"] < best:
                candidates["xgb_ridge_blend"] = {"validation": blend_pred, "metrics": blend_metrics}
                selected = "xgb_ridge_blend"
    base_selected = selected
    if mode == "real_time_pre_da" and XGBRegressor is not None:
        l1_validation = _fit_predict(fit_train, validation, columns, "xgboost_l1")
        candidates["xgboost_l1"] = {
            "validation": l1_validation,
            "metrics": _metric(validation["target"].to_numpy(float), l1_validation),
        }
        fixed_blend_validation = (
            (1.0 - PRE_DA_L1_WEIGHT) * candidates[base_selected]["validation"]
            + PRE_DA_L1_WEIGHT * l1_validation
        )
        candidates["pre_da_l1_blend"] = {
            "validation": fixed_blend_validation,
            "metrics": _metric(validation["target"].to_numpy(float), fixed_blend_validation),
        }
        selected = "pre_da_l1_blend"
    if selected == "pre_da_l1_blend":
        if base_selected == "xgb_ridge_blend" and blend is not None:
            first, first_name, second, second_name = blend
            base_prediction = (
                first * _fit_predict(train, target_rows, columns, first_name)
                + second * _fit_predict(train, target_rows, columns, second_name)
            )
        else:
            base_prediction = _fit_predict(train, target_rows, columns, base_selected)
        l1_prediction = _fit_predict(train, target_rows, columns, "xgboost_l1")
        prediction = (
            (1.0 - PRE_DA_L1_WEIGHT) * base_prediction
            + PRE_DA_L1_WEIGHT * l1_prediction
        )
    elif selected == "xgb_ridge_blend" and blend is not None:
        first, first_name, second, second_name = blend
        pred_a = _fit_predict(train, target_rows, columns, first_name)
        pred_b = _fit_predict(train, target_rows, columns, second_name)
        prediction = first * pred_a + second * pred_b
    else:
        prediction = _fit_predict(train, target_rows, columns, selected)
    selected_validation = candidates[selected]["validation"]
    raw_residual = validation["target"].to_numpy(float) - selected_validation
    calibration_applied = point_calibration == "median" or (
        point_calibration == "pre_da_median" and mode == "real_time_pre_da"
    )
    point_adjustment = float(np.median(raw_residual)) if calibration_applied else 0.0
    prediction = prediction + point_adjustment
    residual = raw_residual - point_adjustment
    if len(residual) < 20:
        residual = train["target"].to_numpy(float)[-24 * 14:] - train["target"].to_numpy(float)[-24 * 14:].mean()
    q10, q90 = np.quantile(residual, [0.1, 0.9])
    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(target_rows.iterrows()):
        p50 = float(np.clip(prediction[index], -100.0, 2000.0))
        rows.append({"period": int(row["period"]), "p10": round(float(np.clip(p50 + q10, -100, 2000)), 3), "p50": round(p50, 3), "p90": round(float(np.clip(p50 + q90, -100, 2000)), 3)})
    return ModeForecast(
        mode=mode,
        target_date=target_date,
        rows=rows,
        metrics={
            "selected": candidates[selected]["metrics"],
            "candidates": {name: item["metrics"] for name, item in candidates.items()},
            "point_calibration": point_calibration,
            "point_calibration_applied": calibration_applied,
            "point_adjustment_yuan_per_mwh": point_adjustment,
            "pre_da_l1_weight": PRE_DA_L1_WEIGHT if mode == "real_time_pre_da" else None,
            "calibrated_validation": _metric(
                validation["target"].to_numpy(float), selected_validation + point_adjustment
            ),
            "residual_q10": float(q10),
            "residual_q90": float(q90),
        },
        selected_model=selected,
        training_cutoff_date=cutoff.date().isoformat(),
        validation_window={"start": validation["date"].min().date().isoformat(), "end": validation["date"].max().date().isoformat()},
    )


def run_henan_forecast(
    data_dir: Path,
    target_date: str,
    backtest_start: str | None = None,
    backtest_end: str | None = None,
    feature_set: str = "base",
    point_calibration: str = "none",
) -> dict[str, Any]:
    frame = load_henan_frame(data_dir)
    modes: dict[str, tuple[pd.DataFrame, list[str]]] = {
        mode: build_feature_frame(frame, mode, feature_set=feature_set) for mode in MODES
    }
    mode_outputs: dict[str, ModeForecast] = {}
    for mode, (features, columns) in modes.items():
        mode_outputs[mode] = forecast_mode(
            features, columns, target_date, mode, point_calibration=point_calibration
        )
    anchor_features, anchor_columns = build_feature_frame(frame, "day_ahead_d2", feature_set=feature_set)
    day_ahead_d2_anchor = forecast_mode(
        anchor_features,
        anchor_columns,
        target_date,
        "day_ahead_d2",
        point_calibration=point_calibration,
    )
    mode_outputs["real_time_pre_da"] = _apply_pre_da_da_anchor(
        mode_outputs["real_time_pre_da"],
        day_ahead_d2_anchor,
    )
    by_period: dict[int, dict[str, Any]] = {}
    reliability_profile = _load_period_reliability()
    reliability_by_period = {
        int(item["period"]): item for item in (reliability_profile or {}).get("periods", [])
    }
    target_actuals = frame.loc[frame["date"] == pd.Timestamp(target_date)].set_index("period")
    for period in range(1, 25):
        row = {"marketDate": target_date, "period": period}
        for mode, result in mode_outputs.items():
            row[mode] = next(item for item in result.rows if item["period"] == period)
        da = row["day_ahead"]["p50"]
        rt_post = row["real_time_post_da"]["p50"]
        row["spread_rt_minus_da"] = round(rt_post - da, 3)
        if period in target_actuals.index:
            actual = target_actuals.loc[period]
            if pd.notna(actual["da"]):
                row["actual_day_ahead_price_yuan_per_mwh"] = round(float(actual["da"]), 3)
            if pd.notna(actual["rt"]):
                row["actual_real_time_price_yuan_per_mwh"] = round(float(actual["rt"]), 3)
        if period in reliability_by_period:
            row["real_time_pre_da"]["historicalReliability"] = reliability_by_period[period]
        by_period[period] = row
    output: dict[str, Any] = {
        "schemaVersion": "henan-forecast-v1",
        "marketCode": "HA",
        "marketName": "河南",
        "targetDate": target_date,
        "dataVersion": "henan-spot-weather-power-2026-v1",
        "modelVersion": HENAN_PRE_DA_MODEL_ID,
        "forecastStatus": "RESEARCH_BACKTEST_ONLY",
        "warnings": [
            "Historical weather is not a frozen forecast snapshot.",
            "Power forecast publication timestamps are unavailable.",
            "Per-period reliability is historical tolerance performance, not a future correctness probability.",
            "Do not use this output as a live declaration without a reviewed issue-time data feed.",
        ],
        "forecast": list(by_period.values()),
        "models": {
            **{
                mode: {
                    "selected": result.selected_model,
                    "metrics": result.metrics,
                    "trainingCutoffDate": result.training_cutoff_date,
                    "validationWindow": result.validation_window,
                }
                for mode, result in mode_outputs.items()
            },
            "day_ahead_d2_anchor": {
                "selected": day_ahead_d2_anchor.selected_model,
                "metrics": day_ahead_d2_anchor.metrics,
                "trainingCutoffDate": day_ahead_d2_anchor.training_cutoff_date,
                "validationWindow": day_ahead_d2_anchor.validation_window,
            },
        },
        "architecture": {
            "primary": "XGBoost with Ridge challenger and validation-gated blend",
            "real_time_pre_da": (
                "strict D-3 direct RT blend with a segmented D+2 day-ahead forecast anchor"
            ),
            "real_time_pre_da_weight_protocol": (
                "The internal 60% L1 blend remains the direct RT base. Segment anchor weights "
                "were selected before an independent 2026-08-08/2026-09-06 holdout."
            ),
            "real_time_pre_da_point_optimization": {
                "method": "segmented D+2 day-ahead forecast anchor",
                "weights": {
                    name: float(config["weight"])
                    for name, config in PRE_DA_ANCHOR_WEIGHTS.items()
                },
                "selection_window": "2026-06-01/2026-08-07",
                "independent_holdout": "2026-08-08/2026-09-06",
                "target_day_cleared_day_ahead_used": False,
            },
            "features": "calendar + same-period price lags + target-day power forecast + province-average weather",
            "feature_set": feature_set,
            "point_calibration": point_calibration,
            "interval": "rolling validation residual P10/P90",
            "post_da_identity": "real-time post-DA is directly modelled with cleared DA price as an available feature",
            "label_cutoffs": {mode: f"D-{config['cutoff_days']}" for mode, config in MODES.items()},
            "no_random_split": True,
        },
    }
    if reliability_profile is not None:
        output["periodReliability"] = reliability_profile
    if backtest_start and backtest_end:
        output["backtest"] = run_henan_walk_forward(
            data_dir,
            backtest_start,
            backtest_end,
            feature_set=feature_set,
            point_calibration=point_calibration,
        )
    return output


def run_henan_walk_forward(
    data_dir: Path,
    start: str,
    end: str,
    feature_set: str = "base",
    point_calibration: str = "none",
) -> dict[str, Any]:
    frame = load_henan_frame(data_dir)
    feature_tables = {
        mode: build_feature_frame(frame, mode, feature_set=feature_set) for mode in MODES
    }
    anchor_features, anchor_columns = build_feature_frame(
        frame, "day_ahead_d2", feature_set=feature_set
    )
    dates = [date for date in pd.date_range(start, end, freq="D") if int((frame["date"] == date).sum()) == 24]
    result_rows: list[dict[str, Any]] = []
    for date in dates:
        try:
            outputs = {
                mode: forecast_mode(
                    features,
                    columns,
                    date.date().isoformat(),
                    mode,
                    point_calibration=point_calibration,
                )
                for mode, (features, columns) in feature_tables.items()
            }
            day_ahead_d2_anchor = forecast_mode(
                anchor_features,
                anchor_columns,
                date.date().isoformat(),
                "day_ahead_d2",
                point_calibration=point_calibration,
            )
            outputs["real_time_pre_da"] = _apply_pre_da_da_anchor(
                outputs["real_time_pre_da"],
                day_ahead_d2_anchor,
            )
        except ValueError:
            continue
        actual = frame.loc[frame["date"] == date].sort_values("period")
        for index, (_, actual_row) in enumerate(actual.iterrows()):
            row: dict[str, Any] = {"marketDate": date.date().isoformat(), "period": int(actual_row["period"]), "da_actual": float(actual_row["da"]), "rt_actual": float(actual_row["rt"])}
            for mode, output in outputs.items():
                row[f"{mode}_pred"] = output.rows[index]["p50"]
                row[f"{mode}_p10"] = output.rows[index]["p10"]
                row[f"{mode}_p90"] = output.rows[index]["p90"]
            row["spread_actual"] = row["rt_actual"] - row["da_actual"]
            row["spread_pre_da_pred"] = row["real_time_pre_da_pred"] - row["day_ahead_pred"]
            row["spread_post_da_pred"] = row["real_time_post_da_pred"] - row["day_ahead_pred"]
            result_rows.append(row)
    result = pd.DataFrame(result_rows)
    scores: dict[str, Any] = {"windowStart": start, "windowEnd": end, "dayCount": int(result["marketDate"].nunique()) if not result.empty else 0, "hourCount": len(result)}
    if not result.empty:
        for mode, actual_column in [("day_ahead", "da_actual"), ("real_time_pre_da", "rt_actual"), ("real_time_post_da", "rt_actual")]:
            scores[mode] = _metric(result[actual_column].to_numpy(float), result[f"{mode}_pred"].to_numpy(float))
        scores["spread_post_da"] = _metric(result["spread_actual"].to_numpy(float), result["spread_post_da_pred"].to_numpy(float))
        scores["real_time_pre_da_direction_accuracy"] = float(np.mean((result["real_time_pre_da_pred"] >= result["day_ahead_pred"]) == (result["rt_actual"] >= result["da_actual"])))
        scores["real_time_post_da_direction_accuracy"] = float(np.mean((result["real_time_post_da_pred"] >= result["day_ahead_pred"]) == (result["rt_actual"] >= result["da_actual"])))
    period_scores: dict[str, list[dict[str, Any]]] = {
        "day_ahead": [],
        "real_time_pre_da": [],
        "real_time_post_da": [],
        "spread_pre_da": [],
        "spread_post_da": [],
    }
    if not result.empty:
        mode_columns = {
            "day_ahead": ("da_actual", "day_ahead_pred"),
            "real_time_pre_da": ("rt_actual", "real_time_pre_da_pred"),
            "real_time_post_da": ("rt_actual", "real_time_post_da_pred"),
            "spread_pre_da": ("spread_actual", "spread_pre_da_pred"),
            "spread_post_da": ("spread_actual", "spread_post_da_pred"),
        }
        for period, group in result.groupby("period", sort=True):
            actual_direction = group["rt_actual"].to_numpy(float) >= group["da_actual"].to_numpy(float)
            for name, (actual_column, predicted_column) in mode_columns.items():
                lower = group[f"{name}_p10"].to_numpy(float) if f"{name}_p10" in group else None
                upper = group[f"{name}_p90"].to_numpy(float) if f"{name}_p90" in group else None
                item: dict[str, Any] = {
                    "period": int(period),
                    **_point_accuracy_metrics(
                        group[actual_column].to_numpy(float),
                        group[predicted_column].to_numpy(float),
                        lower,
                        upper,
                    ),
                }
                if name in {"real_time_pre_da", "real_time_post_da"}:
                    predicted_direction = (
                        group[predicted_column].to_numpy(float)
                        >= group["day_ahead_pred"].to_numpy(float)
                    )
                    item["spread_direction_accuracy"] = float(
                        np.mean(predicted_direction == actual_direction)
                    )
                elif name.startswith("spread_"):
                    item["spread_direction_accuracy"] = float(
                        np.mean((group[predicted_column].to_numpy(float) >= 0) == actual_direction)
                    )
                else:
                    item["spread_direction_accuracy"] = None
                period_scores[name].append(item)
    return {"scores": scores, "periodScores": period_scores, "rows": result.to_dict(orient="records")}


def _prepare_horizon_frame(data_dir: Path, start: pd.Timestamp, days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _as_frame(data_dir, include_future=True)
    frame["da_feature"] = frame["da"]
    frame["rt_feature"] = frame["rt"]
    source_counts: dict[str, dict[str, int]] = {}
    for offset in range(days):
        date = start + pd.Timedelta(days=offset)
        day_mask = frame["date"] == date
        direct_rows = frame.loc[day_mask, POWER_FEATURES].notna().all(axis=1)
        direct_count = int(direct_rows.sum())
        proxy_cells = 0
        proxy_lags: dict[int, int] = {}
        for index in frame.index[day_mask]:
            period = int(frame.at[index, "period"])
            for column in POWER_FEATURES + ["netLoadForecastMw"]:
                if column not in frame or pd.notna(frame.at[index, column]):
                    continue
                for lag_days in [7, 14, 21, 28]:
                    source_date = date - pd.Timedelta(days=lag_days)
                    proxy = _safe_value(frame, source_date, period, column)
                    if proxy is not None:
                        frame.at[index, column] = proxy
                        proxy_cells += 1
                        proxy_lags[lag_days] = proxy_lags.get(lag_days, 0) + 1
                        break
        source_counts[date.date().isoformat()] = {
            "direct_complete_periods": direct_count,
            "proxy_cells": proxy_cells,
            "proxy_lag_cells": proxy_lags,
            "weather_complete_periods": int(frame.loc[day_mask, WEATHER_FEATURES].notna().all(axis=1).sum()),
        }
    return frame, source_counts


def _inflate_interval(rows: list[dict[str, Any]], horizon_day: int) -> list[dict[str, Any]]:
    factor = math.sqrt(1.0 + 0.25 * max(0, horizon_day - 1))
    output: list[dict[str, Any]] = []
    for row in rows:
        p50 = float(row["p50"])
        lower_width = p50 - float(row["p10"])
        upper_width = float(row["p90"]) - p50
        output.append(
            {
                **row,
                "p10": round(float(np.clip(p50 - lower_width * factor, -100, 2000)), 3),
                "p90": round(float(np.clip(p50 + upper_width * factor, -100, 2000)), 3),
                "horizonIntervalFactor": round(factor, 4),
            }
        )
    return output


def run_henan_seven_day_forecast(data_dir: Path, start_date: str = "2026-09-08", days: int = 7) -> dict[str, Any]:
    """Generate a recursive DA and pre-DA RT research forecast.

    Future days use target-day weather from the supplied seven-day weather
    package. Missing target-day power forecasts are explicitly filled from the
    D-7 same-period forecast, and the provenance is returned with every day.
    """
    if days < 1 or days > 7:
        raise ValueError("days must be between 1 and 7")
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=days - 1)
    frame, source_counts = _prepare_horizon_frame(data_dir, start, days)
    available_dates = set(frame["date"].dt.date.astype(str))
    missing_dates = [date.date().isoformat() for date in pd.date_range(start, end, freq="D") if date.date().isoformat() not in available_dates]
    if missing_dates:
        raise ValueError(f"future feature dates are unavailable: {missing_dates}")

    # Seed any missing RT label dates between the final observed RT day and the
    # requested start. They are used only as recursive lag features, never as
    # training labels or backtest truth.
    last_rt = frame.loc[frame["rt"].notna(), "date"].max()
    for seed_date in pd.date_range(last_rt + pd.Timedelta(days=1), start - pd.Timedelta(days=1), freq="D"):
        features, columns = build_feature_frame(frame, "real_time_pre_da")
        seed = forecast_mode(features, columns, seed_date.date().isoformat(), "real_time_pre_da")
        for item in seed.rows:
            mask = (frame["date"] == seed_date) & (frame["period"] == item["period"])
            frame.loc[mask, "rt_feature"] = item["p50"]

    forecast_days: list[dict[str, Any]] = []
    for horizon_day, date in enumerate(pd.date_range(start, end, freq="D"), 1):
        date_text = date.date().isoformat()
        da_features, da_columns = build_feature_frame(frame, "day_ahead")
        da = forecast_mode(da_features, da_columns, date_text, "day_ahead")
        da.rows = _inflate_interval(da.rows, horizon_day)
        for item in da.rows:
            mask = (frame["date"] == date) & (frame["period"] == item["period"])
            frame.loc[mask, "da_feature"] = item["p50"]

        rt_features, rt_columns = build_feature_frame(frame, "real_time_pre_da")
        rt = forecast_mode(rt_features, rt_columns, date_text, "real_time_pre_da")
        rt = _apply_pre_da_da_anchor(rt, da)
        rt.rows = _inflate_interval(rt.rows, horizon_day)
        for item in rt.rows:
            mask = (frame["date"] == date) & (frame["period"] == item["period"])
            frame.loc[mask, "rt_feature"] = item["p50"]

        rows: list[dict[str, Any]] = []
        for period in range(1, 25):
            da_row = next(item for item in da.rows if item["period"] == period)
            rt_row = next(item for item in rt.rows if item["period"] == period)
            rows.append(
                {
                    "period": period,
                    "dayAhead": da_row,
                    "realTimePreDa": rt_row,
                    "spreadRtMinusDaP50": round(float(rt_row["p50"] - da_row["p50"]), 3),
                    "realTimePostDa": None,
                    "realTimePostDaStatus": "WAITING_FOR_CLEARED_DAY_AHEAD_PRICE",
                }
            )
        forecast_days.append(
            {
                "marketDate": date_text,
                "horizonDay": horizon_day,
                "featureAvailability": source_counts[date_text],
                "models": {"dayAhead": da.selected_model, "realTimePreDa": rt.selected_model},
                "rows": rows,
            }
        )
    return {
        "schemaVersion": "henan-seven-day-price-forecast-v1",
        "marketCode": "HA",
        "marketName": "河南",
        "startDate": start.date().isoformat(),
        "endDate": end.date().isoformat(),
        "dayCount": days,
        "pointCount": days * 24,
        "dataVersion": "henan-spot-weather-power-2026-v1",
        "forecastStatus": "RESEARCH_FORECAST_ONLY",
        "executionAllowed": False,
        "forecast": forecast_days,
        "method": {
            "dayAhead": "recursive XGBoost/Ridge validation-selected forecast",
            "realTime": "recursive strict pre-DA forecast with segmented predicted-DA anchor; no target-day cleared DA price",
            "powerProxy": "D-7 same-period power forecast where target-day power is unavailable",
            "uncertainty": "validation residual P10/P90 enlarged by sqrt(1 + 0.25 * (horizon day - 1))",
        },
        "warnings": [
            "Weather issue timestamps are unavailable; the seven-day weather values are treated as supplied forecast features.",
            "Power forecasts after 2026-09-08 are unavailable and use an explicit D-7 proxy.",
            "Recursive forecasts use prior predicted prices as lag features after observed labels end.",
            "Post-DA real-time forecasts are withheld until each day's cleared DA price is available.",
            "No bidding quantity, order, or automatic execution is generated.",
        ],
    }
