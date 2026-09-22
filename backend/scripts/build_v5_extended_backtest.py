from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.v5_shandong_price_forecast import MODEL_VERSION, _load


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "model_assets" / MODEL_VERSION
PREDICTION_PATH = ASSET_DIR / "v5_expanding_oof_predictions.csv.gz"
METRICS_PATH = ASSET_DIR / "extended_backtest_metrics.json"
FOLDS = [
    ("2026-02", "2026-01-31", "2026-02-01", "2026-02-28"),
    ("2026-03", "2026-02-28", "2026-03-01", "2026-03-31"),
    ("2026-04", "2026-03-31", "2026-04-01", "2026-04-30"),
    ("2026-05", "2026-04-30", "2026-05-01", "2026-05-31"),
    ("2026-06", "2026-05-31", "2026-06-01", "2026-06-30"),
]


def _regression_metrics(actual, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, float)
    predicted_values = np.asarray(predicted, float)
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        "bias": float(np.mean(predicted_values - actual_values)),
    }


def _event_probability(package: dict, train: pd.DataFrame, validation: pd.DataFrame, target: str) -> np.ndarray:
    estimator = clone(package["model"])
    estimator.fit(train[package["features"]], train[target].astype(int))
    probability = estimator.predict_proba(validation[package["features"]])
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(len(validation), dtype=float)
    return probability[:, classes.index(1)]


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    source, frame, models, metadata, _ = _load()
    records: list[pd.DataFrame] = []
    monthly: dict[str, dict] = {}

    for month, train_end, validation_start, validation_end in FOLDS:
        train_mask = frame["日期"].le(pd.Timestamp(train_end))
        validation_mask = frame["日期"].between(pd.Timestamp(validation_start), pd.Timestamp(validation_end))
        validation = frame.loc[validation_mask].copy()
        output = pd.DataFrame(
            {
                "date": validation["日期"].dt.strftime("%Y-%m-%d").to_numpy(),
                "period": validation["小时"].astype(int).to_numpy(),
                "evaluation_mode": "EXPANDING_WINDOW_OOF",
                "training_end": train_end,
            }
        )
        month_metrics: dict[str, dict] = {}
        for key, target in (("da", "日前价格"), ("rt", "实时价格"), ("spread", "实时价差")):
            package = models[key]
            target_train = frame.loc[train_mask & frame[target].notna()]
            estimator = clone(package["model"])
            estimator.fit(target_train[package["features"]], target_train[target])
            predicted = estimator.predict(validation[package["features"]])
            output[f"{key}_p50"] = predicted
            observed = validation[target].notna().to_numpy()
            month_metrics[key] = {
                **_regression_metrics(validation.loc[validation[target].notna(), target], predicted[observed]),
                "sampleCount": int(observed.sum()),
            }

        for key, target in (("negative", "负价事件"), ("spike", "尖峰事件")):
            package = models[key]
            target_train = frame.loc[train_mask & frame[target].notna()]
            probability = _event_probability(package, target_train, validation, target)
            output[f"{key}_probability"] = probability

        output["actual_da"] = validation["日前价格"].to_numpy()
        output["actual_rt"] = validation["实时价格"].to_numpy()
        output["actual_spread"] = validation["实时价差"].to_numpy()
        records.append(output)
        monthly[month] = {
            "evaluationMode": "EXPANDING_WINDOW_OOF",
            "trainingEnd": train_end,
            "sampleCount": int(len(validation)),
            "dayAhead": month_metrics["da"],
            "realTime": month_metrics["rt"],
            "spread": month_metrics["spread"],
        }

    predictions = pd.concat(records, ignore_index=True)
    predictions.to_csv(PREDICTION_PATH, index=False, compression="gzip", encoding="utf-8")

    january = frame[frame["日期"].between(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31"))]
    july = frame[frame["日期"].between(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31"))]
    january_metrics: dict[str, dict] = {}
    july_predictions: dict[str, np.ndarray] = {}
    combined_metrics: dict[str, dict] = {}
    for key, target, actual_column in (
        ("da", "日前价格", "actual_da"),
        ("rt", "实时价格", "actual_rt"),
        ("spread", "实时价差", "actual_spread"),
    ):
        package = models[key]
        january_predicted = package["model"].predict(january[package["features"]])
        january_valid = january[target].notna().to_numpy()
        january_metrics[key] = {
            **_regression_metrics(january.loc[january[target].notna(), target], january_predicted[january_valid]),
            "sampleCount": int(january_valid.sum()),
        }
        july_predicted = package["model"].predict(july[package["features"]])
        july_predictions[key] = july_predicted
        historical_actual = pd.to_numeric(predictions[actual_column], errors="coerce").to_numpy(float)
        historical_predicted = predictions[f"{key}_p50"].to_numpy(float)
        july_actual = pd.to_numeric(july[target], errors="coerce").to_numpy(float)
        all_actual = np.concatenate([historical_actual, july_actual])
        all_predicted = np.concatenate([historical_predicted, july_predicted])
        valid = np.isfinite(all_actual) & np.isfinite(all_predicted)
        combined_metrics[key] = {
            **_regression_metrics(all_actual[valid], all_predicted[valid]),
            "sampleCount": int(valid.sum()),
        }

    payload = {
        "modelVersion": MODEL_VERSION,
        "method": "monthly expanding-window out-of-fold replay",
        "supportedOofStart": "2026-02-01",
        "supportedOofEnd": "2026-06-30",
        "januaryMode": "IN_SAMPLE_DIAGNOSTIC_NO_PRIOR_TRAINING_WINDOW",
        "julyMode": "FROZEN_HOLDOUT",
        "januaryInSampleDiagnostic": january_metrics,
        "monthly": monthly,
        "julyMetrics": metadata["julyMetrics"],
        "combinedOutOfSampleFebruaryToJuly": combined_metrics,
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = ASSET_DIR / "model_metadata.json"
    metadata["controlledReplayRange"] = {"start": "2026-01-01", "end": "2026-07-31"}
    metadata["evaluationModes"] = {
        "2026-01": "IN_SAMPLE_DIAGNOSTIC",
        "2026-02_to_2026-06": "EXPANDING_WINDOW_OOF",
        "2026-07": "FROZEN_HOLDOUT",
    }
    metadata["sha256"][PREDICTION_PATH.name] = hashlib.sha256(PREDICTION_PATH.read_bytes()).hexdigest()
    metadata["sha256"][METRICS_PATH.name] = hashlib.sha256(METRICS_PATH.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"predictions={PREDICTION_PATH} rows={len(predictions)}")


if __name__ == "__main__":
    main()
