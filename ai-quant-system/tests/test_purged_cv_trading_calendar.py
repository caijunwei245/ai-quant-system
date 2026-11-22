import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import patch

from layers import screening
from layers.screening import ScreeningLayer


class _DeterministicClassifier:
    """Small deterministic stand-in: this test targets date purging, not model quality."""

    def __init__(self, **kwargs):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probabilities = np.full(len(X), 0.5)
        return np.column_stack((1.0 - probabilities, probabilities))


def test_purged_kfold_uses_exchange_trading_day_ranks(tmp_path):
    # 120 actual exchange sessions; the fold boundary is session 40.
    trading_calendar = pd.bdate_range(start="2026-01-05", periods=120)
    sample_dates = list(trading_calendar)
    rng = np.random.RandomState(42)
    X = rng.randn(len(sample_dates), 5).astype("float32")
    y = np.array([i % 2 for i in range(len(sample_dates))], dtype=int)

    layer = ScreeningLayer(config={})
    layer.model_dir = tmp_path

    # Keep the test independent of an optional LightGBM installation.
    fake_lightgbm = SimpleNamespace(LGBMClassifier=_DeterministicClassifier)
    with patch.object(screening, "lgb", fake_lightgbm, create=True):
        result = layer._purged_kfold_cv(
            X,
            y,
            sample_dates,
            n_splits=3,
            embargo_days=1,
            forward_days=2,
            trading_calendar=trading_calendar,
        )

    assert isinstance(result, dict)
    assert result["n_folds"] == 2

    # The second fold starts at trading session 40. Training samples initially
    # cover ranks 0..39; purge condition rank + 2 >= 40 - 1 removes 37..39.
    expected_val_start = trading_calendar[40].strftime("%Y-%m-%d")
    target_fold = next(
        (fold for fold in result["folds"] if fold["val_start"] == expected_val_start),
        None,
    )
    assert target_fold is not None
    assert target_fold["n_train"] == 37
    assert target_fold["n_val"] == 40

    # Third fold starts at session 80: ranks 77..79 are purged, leaving 77.
    third_fold_start = trading_calendar[80].strftime("%Y-%m-%d")
    third_fold = next(
        (fold for fold in result["folds"] if fold["val_start"] == third_fold_start),
        None,
    )
    assert third_fold is not None
    assert third_fold["n_train"] == 77
