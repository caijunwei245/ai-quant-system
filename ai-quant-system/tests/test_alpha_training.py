import numpy as np
import pandas as pd
import pytest

from data.alpha_training import (build_rank_dataset, purged_walk_forward_splits, rank_metrics, split_development_oos)


def _panel_inputs(n_dates=12, n_stocks=6):
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    features, returns = {}, {}
    for stock in range(n_stocks):
        code = f"{stock:06d}"
        features[code] = pd.DataFrame(
            {"momentum": np.arange(n_dates) + stock,
             "volatility": np.full(n_dates, stock / 10)},
            index=dates,
        )
        returns[code] = pd.Series(
            [(stock - 2.5) * 0.01 + day * 0.0001 for day in range(n_dates)],
            index=dates,
        )
    return list(features), features, returns, dates


def test_rank_dataset_preserves_daily_groups_and_ordinal_labels():
    stocks, features, returns, dates = _panel_inputs()
    X, y, raw, codes, names, sample_dates, groups = build_rank_dataset(
        stocks, features, returns, min_group_size=5
    )
    assert X.shape == (len(dates) * len(stocks), 2)
    assert groups == [len(stocks)] * len(dates)
    assert sum(groups) == len(y) == len(raw) == len(codes)
    assert set(np.unique(y)).issubset({0, 1, 2, 3, 4})
    first_day = np.array(sample_dates[:len(stocks)])
    assert len(set(pd.to_datetime(first_day))) == 1
    # Within each date, higher forward returns must receive higher relevance.
    first_relevance = y[:len(stocks)]
    assert np.all(np.diff(first_relevance) >= 0)


def test_rank_dataset_drops_incomplete_cross_sections():
    stocks, features, returns, dates = _panel_inputs(n_dates=8, n_stocks=6)
    returns[stocks[-1]].iloc[3] = np.nan
    _, _, _, _, _, sample_dates, groups = build_rank_dataset(
        stocks, features, returns, min_group_size=6
    )
    assert pd.Timestamp(dates[3]) not in set(pd.to_datetime(sample_dates))
    assert len(groups) == 7


def test_purged_walk_forward_uses_calendar_sessions_and_embargo():
    calendar = pd.bdate_range("2024-01-02", periods=120)
    dates = list(calendar)
    splits = list(purged_walk_forward_splits(
        dates, calendar, n_splits=3, forward_days=2, embargo_days=1,
        min_train_samples=20, min_validation_samples=5,
    ))
    assert len(splits) == 2
    _, train, val, start, _ = splits[0]
    assert start == calendar[40].strftime("%Y-%m-%d")
    assert len(train) == 37
    assert len(val) == 40
    assert max(train) < min(val)


def test_purged_walk_forward_rejects_unknown_sample_dates():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    dates = list(calendar[:20]) + [pd.Timestamp("2024-03-01")]
    with pytest.raises(ValueError, match="missing from trading calendar"):
        list(purged_walk_forward_splits(dates, calendar, n_splits=2))


def test_purged_walk_forward_rejects_unsorted_or_duplicate_calendar():
    dates = list(pd.bdate_range("2024-01-02", periods=20))
    bad_calendar = list(pd.bdate_range("2024-01-02", periods=20))
    bad_calendar[5] = bad_calendar[4]
    with pytest.raises(ValueError, match="unique and sorted"):
        list(purged_walk_forward_splits(dates, bad_calendar, n_splits=2))


def test_rank_metrics_calculates_daily_rankic_and_top_bottom_spread():
    dates = list(pd.bdate_range("2024-01-02", periods=2).repeat(10))
    returns = np.tile(np.arange(10, dtype=float) / 100, 2)
    scores = returns.copy()
    metrics = rank_metrics(dates, returns, scores, n_quantiles=5)
    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert metrics["top_bottom_return"] == pytest.approx(0.08)
    assert metrics["n_dates"] == 2



def test_independent_oos_is_latest_date_block_and_purges_overlap():
    calendar = pd.bdate_range("2024-01-02", periods=100)
    dates = list(calendar)
    dev, oos, oos_start = split_development_oos(
        dates, calendar, oos_fraction=0.2, forward_days=5, embargo_days=2
    )
    assert oos_start == calendar[80]
    assert len(oos) == 20
    assert max(dev) < min(oos)
    # Last development observation must finish its forward label before OOS embargo.
    dev_ranks = calendar.get_indexer(pd.DatetimeIndex(dates)[dev])
    assert np.all(dev_ranks + 5 < (80 - 2))


def test_independent_oos_rejects_unknown_dates_and_invalid_fraction():
    calendar = pd.bdate_range("2024-01-02", periods=50)
    with pytest.raises(ValueError, match="missing from trading calendar"):
        split_development_oos(
            list(calendar[:20]) + [pd.Timestamp("2024-04-01")], calendar
        )
    with pytest.raises(ValueError, match="oos_fraction"):
        split_development_oos(list(calendar), calendar, oos_fraction=0.05)
