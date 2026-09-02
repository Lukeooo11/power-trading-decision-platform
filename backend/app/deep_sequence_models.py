"""Optional GRU and residual dilated TCN challengers for Shandong prices.

The platform's required runtime intentionally stays sklearn-only.  This module
is therefore an opt-in challenger: when TensorFlow is unavailable, callers get
a structured skip reason instead of a hard failure.  Training uses only the
price history available at the declaration cutoff and a fixed 168-hour context.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SequenceCandidate:
    name: str
    forecast: np.ndarray
    metrics: dict[str, float]
    training_samples: int
    validation_samples: int


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "sample_count": int(actual.size),
    }


def _calendar_channels(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy(dtype=float)
    dow = index.dayofweek.to_numpy(dtype=float)
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
    )


def _make_sequences(
    frame: pd.DataFrame,
    column: str,
    cutoff: pd.Timestamp,
    context_hours: int = 168,
    horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    series = frame.set_index("datetime")[column].astype(float).sort_index()
    series = series[~series.index.duplicated(keep="last")].asfreq("h").interpolate(limit_direction="both")
    values = series.to_numpy(dtype=float)
    index = series.index
    if len(values) < context_hours + horizon + 14:
        return None
    center = float(np.median(values))
    scale = float(np.std(values)) or 1.0
    normalized = (values - center) / scale
    calendar = _calendar_channels(index)
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    holdout_x: list[np.ndarray] = []
    holdout_y: list[np.ndarray] = []
    cutoff = pd.Timestamp(cutoff)
    for end in range(context_hours, len(values) - horizon + 1, 24):
        target_start = index[end]
        target_end = index[end + horizon - 1]
        x = np.column_stack([normalized[end - context_hours : end], calendar[end - context_hours : end]])
        y = normalized[end : end + horizon]
        if target_end <= cutoff:
            train_x.append(x)
            train_y.append(y)
        elif target_start > cutoff and len(holdout_x) < 30:
            holdout_x.append(x)
            holdout_y.append(y)
    if len(train_x) < 14:
        return None
    return (
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.float32),
        np.asarray(holdout_x, dtype=np.float32),
        np.asarray(holdout_y, dtype=np.float32),
    ), center, scale


def _build_model(kind: str, input_shape: tuple[int, int], horizon: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape)
    if kind == "gru":
        x = tf.keras.layers.GRU(32, dropout=0.05)(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
    elif kind == "tcn":
        x = tf.keras.layers.Conv1D(32, 3, padding="causal", dilation_rate=1, activation="swish")(inputs)
        for dilation in (2, 4, 8, 16):
            residual = x
            y = tf.keras.layers.Conv1D(32, 3, padding="causal", dilation_rate=dilation, activation="swish")(x)
            y = tf.keras.layers.Dropout(0.05)(y)
            y = tf.keras.layers.Conv1D(32, 1, padding="same")(y)
            x = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([residual, y]))
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(f"unknown sequence model: {kind}")
    outputs = tf.keras.layers.Dense(horizon)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss="mae")
    return model


def run_sequence_challengers(
    frame: pd.DataFrame,
    column: str,
    train_end: pd.Timestamp,
    future_times: list[pd.Timestamp],
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Train GRU/TCN once for a target date and return auditable candidates.

    Set ``POWER_TRADING_ENABLE_DEEP_MODELS=1`` or pass ``enabled=True`` to
    activate.  The default is disabled to keep the platform's minimal runtime
    deterministic and fast.
    """
    if enabled is None:
        enabled = os.getenv("POWER_TRADING_ENABLE_DEEP_MODELS", "0") == "1"
    if not enabled:
        return {"status": "disabled", "reason": "opt_in_required", "candidates": {}}
    if importlib.util.find_spec("tensorflow") is None:
        return {"status": "skipped", "reason": "tensorflow_not_installed", "candidates": {}}
    if len(future_times) != 24:
        return {"status": "skipped", "reason": "horizon_must_be_24", "candidates": {}}
    prepared = _make_sequences(frame, column, pd.Timestamp(train_end))
    if prepared is None:
        return {"status": "skipped", "reason": "insufficient_sequence_history", "candidates": {}}
    (train_x, train_y, holdout_x, holdout_y), center, scale = prepared
    import tensorflow as tf

    tf.keras.utils.set_random_seed(17)
    results: dict[str, Any] = {}
    for kind in ("gru", "tcn"):
        try:
            model = _build_model(kind, train_x.shape[1:], train_y.shape[1])
            model.fit(
                train_x,
                train_y,
                epochs=6,
                batch_size=min(16, len(train_x)),
                validation_split=0.15 if len(train_x) >= 20 else 0.0,
                verbose=0,
            )
            series = frame.set_index("datetime")[column].astype(float).sort_index()
            series = series[~series.index.duplicated(keep="last")].asfreq("h").interpolate(limit_direction="both")
            values = series.to_numpy(dtype=float)
            index = series.index
            history = values[index <= pd.Timestamp(train_end)]
            if len(history) < 168:
                continue
            history_index = index[index <= pd.Timestamp(train_end)]
            cal = _calendar_channels(history_index)[-168:]
            x = np.column_stack([(history[-168:] - center) / scale, cal]).astype(np.float32)[None, ...]
            forecast = model.predict(x, verbose=0)[0] * scale + center
            metrics = {"mae": None, "rmse": None, "bias": None, "sample_count": 0}
            if len(holdout_x):
                holdout_pred = model.predict(holdout_x, verbose=0) * scale + center
                holdout_actual = holdout_y * scale + center
                metrics = _metrics(holdout_actual.reshape(-1), holdout_pred.reshape(-1))
            results[kind.upper()] = {
                "forecast": np.clip(forecast, -100.0, 1300.0),
                "metrics": metrics,
                "training_samples": int(len(train_x)),
                "validation_samples": int(len(holdout_x)),
            }
        except Exception as error:  # pragma: no cover - backend optional dependency
            results[kind.upper()] = {"error": str(error)}
    return {"status": "available" if results else "failed", "reason": None, "candidates": results}
