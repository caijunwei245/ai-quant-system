import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import patch

from layers import screening
from layers.screening import ScreeningLayer


class _DeterministicRanker:
    """Deterministic test double; this test targets fold indices and group sizes."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X, y, group):
        assert sum(group) == len(X)
        assert len(group) > 0
        self.group = list(group)
        return self

    def predict(self, X):
        return np.arange(len(X), dtype=float)


def test_purged_walk_forward_uses_exchange_trading_day_ranks(tmp_path):
    calendar = pd.bdate_range(start="2026-01-05", periods=120)
    dates = list(calendar)
    rng = np.random.RandomState(42)
    X = rng.randn(120, 5).astype("float32")
    y = np.array([i % 5 for i in range(120)], dtype=int)
    raw_returns = np.array([i / 10000 for i in range(120)], dtype=float)

    layer = ScreeningLayer(config={"lightgbm": {
        "min_cv_train_samples": 20,
        "min_cv_validation_samples": 5,
    }})
    layer.model_dir = tmp_path
    fake_lightgbm = SimpleNamespace(LGBMRanker=_DeterministicRanker)
    with patch.object(screening, "lgb", fake_lightgbm, create=True):
        result = layer._purged_kfold_cv(
            X, y, dates, raw_returns, calendar,
            n_splits=3, embargo_days=1, forward_days=2,
        )

    assert result["n_folds"] == 2
    second = next(f for f in result["folds"]
                  if f["val_start"] == calendar[40].strftime("%Y-%m-%d"))
    assert second["n_train"] == 37
    assert second["n_val"] == 40
    assert second["train_groups"] == [1] * 37
    third = next(f for f in result["folds"]
                 if f["val_start"] == calendar[80].strftime("%Y-%m-%d"))
    assert third["n_train"] == 77
    assert third["n_val"] == 40
