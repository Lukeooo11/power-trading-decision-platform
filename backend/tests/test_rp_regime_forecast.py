from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.evaluate_rp_regime_forecast import (
    _chronological_split,
    _conformalize_quantiles,
)


def _hourly_history() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", "2026-02-28", freq="h")
    return pd.DataFrame(
        {
            "日期": dates.normalize(),
            "实时价格": np.linspace(100, 300, len(dates)),
        }
    )


def test_chronological_split_never_uses_test_month() -> None:
    frame = _hourly_history()
    train, calibration, calibration_start = _chronological_split(
        frame, pd.Timestamp("2026-02-01"), "2026-02"
    )

    assert train["日期"].max() < calibration_start
    assert calibration["日期"].min() == calibration_start
    assert calibration["日期"].max() < pd.Timestamp("2026-02-01")


def test_conformal_interval_is_ordered_and_uses_calibration_only() -> None:
    calibration_actual = np.array([5.0, 20.0, 35.0, 50.0, 65.0])
    calibration_quantiles = np.array(
        [
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
        ]
    )
    test_quantiles = np.array([[30.0, 20.0, 10.0], [15.0, 25.0, 35.0]])

    lower, median, upper, adjustment = _conformalize_quantiles(
        calibration_actual, calibration_quantiles, test_quantiles
    )

    assert adjustment >= 0
    assert np.all(lower <= median)
    assert np.all(median <= upper)
    assert median.tolist() == [20.0, 25.0]


def test_conformal_interval_rejects_invalid_target_coverage() -> None:
    with pytest.raises(ValueError, match="target coverage"):
        _conformalize_quantiles(
            np.array([1.0]),
            np.array([[0.0, 1.0, 2.0]]),
            np.array([[0.0, 1.0, 2.0]]),
            target_coverage=1.0,
        )
