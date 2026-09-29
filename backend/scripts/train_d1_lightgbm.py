"""Train the mixed D-1 price model package for Shandong.

The source feature store already contains the leakage-safe GFS and supply
features used by v2.  This script adds only 24-hour lag features for prices
and price-event heads.  Supply/output lags remain 48 hours or older.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.v5_shandong_price_forecast import (  # noqa: E402
    HISTORY_COLUMNS,
    _features as legacy_features,
    _load as load_legacy,
)

MODEL_VERSION = "sd-gfs24-d1-price-actual-supply-mixed-v5"
DATA_VERSION = "sd-16city-gfs24-actual-supply-20260101-20260831-d1-mixed-v5"
ASSET_DIR = BACKEND / "model_assets" / MODEL_VERSION
TRAIN_END = pd.Timestamp("2026-06-30")
PRICE_COLUMNS = ("日前价格", "实时价格", "实时价差")
EVENT_COLUMNS = ("负价事件", "尖峰事件")
SUPPLY_D1_COLUMNS = ("直调负荷", "风电", "光伏", "新能源出力", "简单净负荷")
FOLDS = [
    ("2026-02", "2026-01-31", "2026-02-01", "2026-02-28"),
    ("2026-03", "2026-02-28", "2026-03-01", "2026-03-31"),
    ("2026-04", "2026-03-31", "2026-04-01", "2026-04-30"),
    ("2026-05", "2026-04-30", "2026-05-01", "2026-05-31"),
    ("2026-06", "2026-05-31", "2026-06-01", "2026-06-30"),
]


def _build_d1_features(source: pd.DataFrame) -> pd.DataFrame:
    frame = legacy_features(source)
    # D-1 settlement prices and yesterday's measured load/renewable output are
    # confirmed available before the target declaration. Target-day actuals
    # are never used; forecast output will be added separately when supplied.
    for column in (*PRICE_COLUMNS, *EVENT_COLUMNS, *SUPPLY_D1_COLUMNS):
        frame[f"{column}_safe_lag24"] = pd.to_numeric(frame[column], errors="coerce").shift(24)
    return frame.copy()


def _feature_names(package: dict, frame: pd.DataFrame) -> list[str]:
    names = list(package["features"])
    names.extend(
        f"{column}_safe_lag24"
        for column in (*PRICE_COLUMNS, *EVENT_COLUMNS, *SUPPLY_D1_COLUMNS)
        if f"{column}_safe_lag24" in frame.columns
    )
    return list(dict.fromkeys(names))


def _lightgbm_regressor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=450,
                    learning_rate=0.035,
                    num_leaves=24,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_lambda=5,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def _xgboost_regressor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                XGBRegressor(
                    n_estimators=450,
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
                ),
            ),
        ]
    )


def _regressor(key: str) -> tuple[Pipeline, str]:
    if key == "da":
        return _xgboost_regressor(), "XGBoost"
    return _lightgbm_regressor(), "LightGBM"


def _classifier() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMClassifier(
                    class_weight="balanced",
                    n_estimators=400,
                    learning_rate=0.035,
                    num_leaves=20,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_lambda=5,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def _metric(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    return {
        "mae": float(mean_absolute_error(a, p)),
        "rmse": float(mean_squared_error(a, p) ** 0.5),
        "bias": float(np.mean(p - a)),
        "sampleCount": int(len(a)),
    }


def _probability(model: Pipeline, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(rows[features])
    classes = list(model.named_steps["model"].classes_)
    return raw[:, classes.index(1)] if 1 in classes else np.zeros(len(rows))


def _calibrate(raw: np.ndarray, actual: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1.0, random_state=42)
    if len(np.unique(actual.astype(int))) >= 2:
        calibrator.fit(np.asarray(raw).reshape(-1, 1), actual.astype(int))
    else:
        calibrator.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    return calibrator


def _f_beta_threshold(actual: np.ndarray, probability: np.ndarray, beta: float) -> float:
    best_score, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.02, 0.8, 79):
        predicted = probability >= threshold
        tp = float(np.sum(predicted & (actual == 1)))
        fp = float(np.sum(predicted & (actual == 0)))
        fn = float(np.sum(~predicted & (actual == 1)))
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        b2 = beta * beta
        score = (1 + b2) * precision * recall / max(1e-12, b2 * precision + recall)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    source, _, old_models, _, _ = load_legacy()
    frame = _build_d1_features(source)
    frame["日期"] = pd.to_datetime(frame["日期"]).dt.normalize()

    packages: dict[str, dict] = {}
    target_names = {"da": "日前价格", "rt": "实时价格", "spread": "实时价差"}
    for key, target in target_names.items():
        packages[key] = {"target": target, "features": _feature_names(old_models[key], frame)}
    for key, target in (("negative", "负价事件"), ("spike", "尖峰事件")):
        packages[key] = {"target": target, "features": _feature_names(old_models[key], frame)}

    records: list[pd.DataFrame] = []
    event_raw: dict[str, list[np.ndarray]] = {"negative": [], "spike": []}
    event_actual: dict[str, list[np.ndarray]] = {"negative": [], "spike": []}
    monthly: dict[str, dict] = {}
    for month, train_end, valid_start, valid_end in FOLDS:
        train = frame[frame["日期"] <= pd.Timestamp(train_end)]
        valid = frame[frame["日期"].between(pd.Timestamp(valid_start), pd.Timestamp(valid_end))].copy()
        out = pd.DataFrame(
            {
                "date": valid["日期"].dt.strftime("%Y-%m-%d").to_numpy(),
                "period": valid["小时"].astype(int).to_numpy(),
                "evaluation_mode": "EXPANDING_WINDOW_OOF",
                "training_end": train_end,
            }
        )
        month_metrics: dict[str, dict] = {}
        for key, target in target_names.items():
            package = packages[key]
            fit = train[train[target].notna()]
            estimator, _ = _regressor(key)
            estimator.fit(fit[package["features"]], fit[target])
            prediction = estimator.predict(valid[package["features"]])
            out[f"{key}_p50"] = prediction
            observed = valid[target].notna().to_numpy()
            month_metrics[key] = _metric(valid.loc[valid[target].notna(), target], prediction[observed])
        for key, target in (("negative", "负价事件"), ("spike", "尖峰事件")):
            package = packages[key]
            fit = train[train[target].notna()]
            estimator = _classifier()
            estimator.fit(fit[package["features"]], fit[target].astype(int))
            raw = _probability(estimator, valid, package["features"])
            out[f"{key}_probability"] = raw
            observed = valid[target].notna().to_numpy()
            event_raw[key].append(raw[observed])
            event_actual[key].append(valid.loc[valid[target].notna(), target].to_numpy(float))
        out["actual_da"] = valid["日前价格"].to_numpy()
        out["actual_rt"] = valid["实时价格"].to_numpy()
        out["actual_spread"] = valid["实时价差"].to_numpy()
        records.append(out)
        monthly[month] = {"trainingEnd": train_end, "sampleCount": len(valid), "dayAhead": month_metrics["da"], "realTime": month_metrics["rt"], "spread": month_metrics["spread"]}

    oof = pd.concat(records, ignore_index=True)
    oof.to_csv(ASSET_DIR / "v5_expanding_oof_predictions.csv.gz", index=False, compression="gzip", encoding="utf-8")

    # Fit production models using only information available by 2026-06-30.
    train = frame[frame["日期"] <= TRAIN_END]
    for key, target in target_names.items():
        package = packages[key]
        fit = train[train[target].notna()]
        model, algorithm = _regressor(key)
        model.fit(fit[package["features"]], fit[target])
        with (ASSET_DIR / {"da": "day_ahead.pkl", "rt": "real_time.pkl", "spread": "spread.pkl"}[key]).open("wb") as handle:
            pickle.dump({"model": model, "features": package["features"], "feature_set": "GFS完整+日内形态+D-1结算价+D-1实际负荷风光", "algorithm": algorithm}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for key, target, filename in (("negative", "负价事件", "negative.pkl"), ("spike", "尖峰事件", "spike.pkl")):
        package = packages[key]
        fit = train[train[target].notna()]
        model = _classifier()
        model.fit(fit[package["features"]], fit[target].astype(int))
        raw = np.concatenate(event_raw[key])
        actual = np.concatenate(event_actual[key]).astype(int)
        calibrator = _calibrate(raw, actual)
        calibrated = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        threshold_f1 = _f_beta_threshold(actual, calibrated, 1.0)
        threshold_f2 = _f_beta_threshold(actual, calibrated, 2.0)
        with (ASSET_DIR / filename).open("wb") as handle:
            pickle.dump({"model": model, "calibrator": calibrator, "features": package["features"], "feature_set": "GFS完整+日内形态+D-1结算价+D-1实际负荷风光", "algorithm": "LightGBM", "threshold_F1": threshold_f1, "threshold_F2": threshold_f2}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Residual intervals use only OOF predictions from April-June.
    calibration_targets: dict[str, dict] = {}
    calibration_records = oof[oof["date"] >= "2026-04-01"]
    for key, actual_column in (("da", "actual_da"), ("rt", "actual_rt")):
        all_residuals = calibration_records[f"{key}_p50"].to_numpy(float) - calibration_records[actual_column].to_numpy(float)
        valid = np.isfinite(all_residuals)
        residuals = all_residuals[valid]
        by_period = {}
        for period in range(1, 25):
            part = calibration_records.loc[calibration_records["period"] == period, f"{key}_p50"].to_numpy(float) - calibration_records.loc[calibration_records["period"] == period, actual_column].to_numpy(float)
            part = part[np.isfinite(part)]
            by_period[str(period)] = {"q10": float(np.quantile(part, .10)), "q90": float(np.quantile(part, .90))}
        calibration_targets[key] = {"sampleCount": int(len(residuals)), "q10": float(np.quantile(residuals, .10)), "q90": float(np.quantile(residuals, .90)), "coverage": float(np.mean((residuals >= np.quantile(residuals, .10)) & (residuals <= np.quantile(residuals, .90)))), "byPeriod": by_period}
    _write_json(ASSET_DIR / "residual_calibration.json", {"schemaVersion": "v8-residual-calibration-d1-actual-supply-v1", "method": "expanding-window-OOF-2026-04-to-06", "targets": calibration_targets})

    july = frame[frame["日期"].between(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31"))]
    july_metrics = {}
    for key, target in target_names.items():
        package = packages[key]
        model_file = {"da": "day_ahead.pkl", "rt": "real_time.pkl", "spread": "spread.pkl"}[key]
        model = pickle.load((ASSET_DIR / model_file).open("rb"))["model"]
        pred = model.predict(july[package["features"]])
        july_metrics[key] = _metric(july[target], pred)
    july_event = {}
    for key, target, filename in (("negative", "负价事件", "negative.pkl"), ("spike", "尖峰事件", "spike.pkl")):
        package = packages[key]
        artifact = pickle.load((ASSET_DIR / filename).open("rb"))
        raw = _probability(artifact["model"], july, package["features"])
        probability = artifact["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
        actual = july[target].to_numpy(float)
        july_event[key] = {"sampleCount": int(len(actual)), "prAuc": None, "thresholdF1": artifact["threshold_F1"], "thresholdF2": artifact["threshold_F2"], "positiveCount": int(np.sum(actual > 0.5))}

    metrics_payload = {"schemaVersion": "v9-d1-mixed-model-backtest-v1", "modelVersion": MODEL_VERSION, "method": "monthly expanding-window OOF plus frozen July holdout", "windowStart": "2026-02-01", "windowEnd": "2026-07-31", "monthly": monthly, "julyMetrics": july_metrics, "julyEvents": july_event, "priceInformationBoundary": "D-1日前和D-1实时结算价以及D-1实际负荷/风光出力可用；目标日实际数据未使用", "forecastSupplyPending": True, "modelSelection": {"da": "XGBoost", "rt": "LightGBM", "spread": "LightGBM", "negative": "LightGBM", "spike": "LightGBM"}}
    _write_json(ASSET_DIR / "extended_backtest_metrics.json", metrics_payload)

    shutil.copy2(BACKEND / "model_assets" / "sd-gfs24-actual-supply-lgbm-v2" / "v6_gfs24_actual_supply_feature_store.csv.gz", ASSET_DIR / "v6_gfs24_actual_supply_feature_store.csv.gz")
    metadata = {"modelVersion": MODEL_VERSION, "weatherDataVersion": DATA_VERSION, "trainingCutoff": "2026-06-30", "featurePolicy": "D-1 settlement prices/events plus D-1 measured load and renewable output; target-day actual supply excluded; forecast output pending", "selected": {"da": {"featureSet": "GFS完整+日内形态+D-1结算价+D-1实际负荷风光", "algorithm": "XGBoost"}, "rt": {"featureSet": "GFS完整+日内形态+D-1结算价+D-1实际负荷风光", "algorithm": "LightGBM"}, "spread": {"featureSet": "GFS核心+空间分布+D-1结算价+D-1实际负荷风光", "algorithm": "LightGBM"}, "negative": {"featureSet": "GFS完整+日内形态+D-1结算价+D-1实际负荷风光", "algorithm": "LightGBM"}, "spike": {"featureSet": "GFS完整+日内形态+D-1结算价+D-1实际负荷风光", "algorithm": "LightGBM"}}, "julyMetrics": july_metrics, "evaluationModes": {"2026-02_to_2026-06": "EXPANDING_WINDOW_OOF", "2026-07": "FROZEN_HOLDOUT"}, "availabilityControl": {"priceSettlementLagDays": 1, "supplyMinimumLagHours": 24, "d1ActualLoadWindSolarUsed": True, "targetDateActualSupplyUsed": False, "forecastSupplyPending": True}, "sha256": {}}
    for path in ASSET_DIR.iterdir():
        if path.is_file() and path.name not in {"model_metadata.json"}:
            metadata["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(ASSET_DIR / "model_metadata.json", metadata)
    print(json.dumps({"modelVersion": MODEL_VERSION, "assetDir": str(ASSET_DIR), "julyMetrics": july_metrics, "monthly": monthly}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
