from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUPPLY_DATA_VERSION = "sd-system-output-hourly-2026h1-v1"
SUPPLY_COLUMNS = [
    "directDispatchLoadMw",
    "interconnectorMw",
    "windMw",
    "solarMw",
    "localPlantMw",
    "captiveUnitMw",
    "testUnitMw",
    "nonMarketNuclearMw",
    "renewableMw",
    "fixedOutputProxyMw",
    "residualDemandProxyMw",
]
WEATHER_COLUMNS = [
    "temperature2mC",
    "windSpeed10mMs",
    "windSpeed100mMs",
    "shortwaveRadiationGhiWm2",
    "cloudCoverPct",
    "precipitationMm",
    "relativeHumidityPct",
]


def _timestamp(date: str, period: int) -> pd.Timestamp:
    return pd.Timestamp(date) + pd.Timedelta(hours=period - 1)


def _load_prices(private_data_dir: Path) -> pd.DataFrame:
    rows = json.loads((private_data_dir / "spot_prices_2026h1.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(rows)
    frame["datetime"] = [_timestamp(row["date"], int(str(row["time"])[:2])) for row in rows]
    frame["da"] = pd.to_numeric(frame["dayAheadPriceYuanMwh"], errors="coerce")
    frame["rt"] = pd.to_numeric(frame["realtimePriceYuanMwh"], errors="coerce")
    frame = frame.set_index("datetime").sort_index()[["da", "rt"]]
    frame["spread"] = frame["da"] - frame["rt"]
    return frame


def _load_weather(private_data_dir: Path) -> dict[pd.Timestamp, dict[str, float]]:
    asset = json.loads((private_data_dir / "weather_hourly_gfs_20260501_20260701.json").read_text(encoding="utf-8"))
    if not asset.get("knownBeforeDeclaration") or not asset.get("backtestLeakageSafe"):
        raise ValueError("weather snapshot is not confirmed as available before declaration")
    lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for row in asset.get("rows", []):
        values = {name: float(row[name]) for name in WEATHER_COLUMNS}
        if all(np.isfinite(value) for value in values.values()):
            lookup[_timestamp(row["marketDate"], int(row["period"]))] = values
    return lookup


def _load_supply(private_data_dir: Path) -> tuple[dict[tuple[str, int], dict[str, float]], dict[str, Any]]:
    asset = json.loads((private_data_dir / "market_supply_hourly_2026h1.json").read_text(encoding="utf-8"))
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    for row in asset.get("rows", []):
        if row.get("sourceType") != "FORECAST":
            continue
        values = {name: row.get(name) for name in SUPPLY_COLUMNS}
        if all(value is not None and np.isfinite(float(value)) for value in values.values()):
            lookup[(row["marketDate"], int(row["period"]))] = {
                name: float(value) for name, value in values.items()
            }
    if not lookup:
        raise ValueError("market supply forecast snapshot has no complete eligible rows")
    return lookup, asset


def _add_calendar(features: dict[str, float], ts: pd.Timestamp) -> None:
    hour, dow, doy = ts.hour, ts.dayofweek, ts.dayofyear
    features.update(
        {
            "hour": float(hour),
            "dow": float(dow),
            "month": float(ts.month),
            "weekend": float(dow >= 5),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
            "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        }
    )
    features.update({f"hour_{value}": float(hour == value) for value in range(24)})
    features.update({f"dow_{value}": float(dow == value) for value in range(7)})


def _add_weather(features: dict[str, float], values: dict[str, float] | None) -> bool:
    if values is None:
        return False
    features.update({f"weather_{name}": value for name, value in values.items()})
    temperature = values["temperature2mC"]
    radiation = values["shortwaveRadiationGhiWm2"]
    cloud = values["cloudCoverPct"]
    features.update(
        {
            "weather_temperature_sq": temperature**2,
            "weather_heat_degree": max(0.0, temperature - 25.0),
            "weather_cool_degree": max(0.0, 10.0 - temperature),
            "weather_wind100_sq": values["windSpeed100mMs"] ** 2,
            "weather_clear_radiation": radiation * max(0.0, 1.0 - cloud / 100.0),
            "weather_precip_log": float(np.log1p(max(0.0, values["precipitationMm"]))),
        }
    )
    return True


def _value_at(series: pd.Series, ts: pd.Timestamp) -> float | None:
    value = series.get(ts, np.nan)
    return float(value) if pd.notna(value) else None


def _add_series_lags(
    features: dict[str, float],
    series: pd.Series,
    name: str,
    ts: pd.Timestamp,
    lags: list[int],
) -> bool:
    for lag in lags:
        value = _value_at(series, ts - pd.Timedelta(hours=lag))
        if value is None:
            return False
        features[f"{name}_lag_{lag}h"] = value
    return True


def _add_window_stats(
    features: dict[str, float],
    series: pd.Series,
    name: str,
    end: pd.Timestamp,
) -> bool:
    for hours in [24, 168]:
        window = series.loc[end - pd.Timedelta(hours=hours - 1) : end]
        if len(window) < hours or window.isna().any():
            return False
        features[f"{name}_mean_{hours}h"] = float(window.mean())
        features[f"{name}_std_{hours}h"] = float(window.std(ddof=0))
        features[f"{name}_min_{hours}h"] = float(window.min())
        features[f"{name}_max_{hours}h"] = float(window.max())
    return True


def _add_supply_lags(
    features: dict[str, float],
    supply: dict[tuple[str, int], dict[str, float]],
    ts: pd.Timestamp,
) -> bool:
    period = ts.hour + 1
    for days in [1, 7]:
        date = (ts.normalize() - pd.Timedelta(days=days)).date().isoformat()
        values = supply.get((date, period))
        if values is None:
            return False
        features.update({f"supply_{name}_lag_{days}d": value for name, value in values.items()})
    for name in ["residualDemandProxyMw", "renewableMw", "directDispatchLoadMw"]:
        features[f"supply_{name}_delta_1d_7d"] = (
            features[f"supply_{name}_lag_1d"] - features[f"supply_{name}_lag_7d"]
        )
    return True


def _build_frame(
    prices: pd.DataFrame,
    weather: dict[pd.Timestamp, dict[str, float]],
    supply: dict[tuple[str, int], dict[str, float]],
    target: str,
    dates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Callers commonly append the target date to the historical date list;
    # normalize here so each market day is materialized exactly once.
    dates = list(dict.fromkeys(str(date) for date in dates))
    own_series = prices[target].ffill()
    cross_name = "rt" if target == "da" else "da"
    cross_series = prices[cross_name].ffill()
    spread_series = prices["spread"].ffill()
    own_lags = [24, 48, 72, 168, 336] if target == "da" else [48, 72, 96, 168, 336]
    cross_lags = [48, 72, 168, 336] if target == "da" else [24, 48, 72, 168]
    own_end_offset = 24 if target == "da" else 48
    cross_end_offset = 48 if target == "da" else 24
    for date in dates:
        for period in range(1, 25):
            ts = _timestamp(date, period)
            features: dict[str, Any] = {"marketDate": date, "period": float(period)}
            _add_calendar(features, ts)
            if not _add_series_lags(features, own_series, target, ts, own_lags):
                continue
            if not _add_series_lags(features, cross_series, cross_name, ts, cross_lags):
                continue
            if not _add_series_lags(features, spread_series, "spread", ts, [48, 168, 336]):
                continue
            if not _add_window_stats(features, own_series, f"{target}_history", ts - pd.Timedelta(hours=own_end_offset)):
                continue
            if not _add_window_stats(features, cross_series, f"{cross_name}_history", ts - pd.Timedelta(hours=cross_end_offset)):
                continue
            if not _add_weather(features, weather.get(ts)) or not _add_supply_lags(features, supply, ts):
                continue
            label = prices.at[ts, target] if ts in prices.index else np.nan
            features["target"] = float(label) if pd.notna(label) else np.nan
            rows.append(features)
    return pd.DataFrame(rows)


def _make_model(model_kind: str):
    if model_kind == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ValueError("xgboost_not_installed") from error
        return XGBRegressor(
            n_estimators=320,
            max_depth=6,
            learning_rate=0.035,
            subsample=0.82,
            colsample_bytree=0.82,
            min_child_weight=8,
            reg_alpha=0.1,
            reg_lambda=8.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=17,
        )
    if model_kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            max_features=0.75,
            n_jobs=-1,
            random_state=17,
        )
    if model_kind == "histgb":
        return HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=18,
            l2_regularization=12.0,
            early_stopping=False,
            random_state=17,
        )
    if model_kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=35.0))
    raise ValueError(f"unsupported supply model: {model_kind}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    high_actual = actual > 500
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "negative_accuracy": float(np.mean((actual < 0) == (predicted < 0))),
        "high_price_recall": float(np.sum(high_actual & (predicted > 500)) / max(1, np.sum(high_actual))),
    }


def run_supply_candidate(
    private_data_dir: Path,
    target_date: str,
    target: str,
    model_kind: str,
    label_cutoff_days: int,
) -> dict[str, Any]:
    prices = _load_prices(private_data_dir)
    weather = _load_weather(private_data_dir)
    supply, supply_asset = _load_supply(private_data_dir)
    historical_dates = sorted({value.date().isoformat() for value in prices.index})
    frame = _build_frame(prices, weather, supply, target, historical_dates + [target_date])
    if frame.empty:
        raise ValueError("no complete weather and lagged supply feature rows")
    feature_columns = [column for column in frame.columns if column not in {"target", "marketDate"}]
    history_end = min(prices.index.max().normalize(), pd.Timestamp(target_date) - pd.Timedelta(days=1))
    validation_start = max(frame["marketDate"].min(), (history_end - pd.Timedelta(days=29)).date().isoformat())
    validation_dates = [
        value.date().isoformat()
        for value in pd.date_range(validation_start, history_end.date().isoformat(), freq="D")
    ]
    predictions: list[float] = []
    actuals: list[float] = []
    keys: list[str] = []
    periods: list[int] = []
    for date in validation_dates:
        cutoff = (pd.Timestamp(date) - pd.Timedelta(days=label_cutoff_days)).date().isoformat()
        train = frame[(frame["marketDate"] <= cutoff) & frame["target"].notna()]
        validation = frame[(frame["marketDate"] == date) & frame["target"].notna()].sort_values("period")
        if len(train) < 100 or len(validation) != 24:
            continue
        model = _make_model(model_kind)
        model.fit(train[feature_columns], train["target"])
        predicted = np.clip(model.predict(validation[feature_columns]), -100.0, 1300.0)
        predictions.extend(predicted.tolist())
        actuals.extend(validation["target"].to_numpy(float).tolist())
        periods.extend(validation["period"].astype(int).tolist())
        keys.extend([f"{date}|{int(period)}" for period in validation["period"]])
    if not predictions:
        raise ValueError("supply candidate rolling validation produced no complete days")

    final_cutoff = (pd.Timestamp(target_date) - pd.Timedelta(days=label_cutoff_days)).date().isoformat()
    train = frame[(frame["marketDate"] <= final_cutoff) & frame["target"].notna()]
    target_rows = frame[frame["marketDate"] == target_date].sort_values("period")
    if len(target_rows) != 24:
        raise ValueError(f"target date {target_date} has {len(target_rows)} complete supply-lag feature rows")
    model = _make_model(model_kind)
    model.fit(train[feature_columns], train["target"])
    forecast = np.clip(model.predict(target_rows[feature_columns]), -100.0, 1300.0)
    actual_array = np.asarray(actuals)
    prediction_array = np.asarray(predictions)
    return {
        "forecast": forecast,
        "validation_actual": actual_array,
        "validation_forecast": prediction_array,
        "validation_keys": keys,
        "validation_periods": np.asarray(periods),
        "metrics": _metrics(actual_array, prediction_array),
        "validation_start": keys[0].split("|")[0],
        "validation_end": keys[-1].split("|")[0],
        "sample_count": len(keys),
        "training_rows": len(train),
        "model_kind": model_kind,
        "supply_data_version": supply_asset.get("dataVersion", SUPPLY_DATA_VERSION),
        "supply_issue_time_available": supply_asset.get("sourceIssueTimeAvailable", False),
        "supply_backtest_leakage_safe": supply_asset.get("backtestLeakageSafe", False),
    }
