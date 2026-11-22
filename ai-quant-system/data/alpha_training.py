"""Leakage-resistant cross-sectional ranking dataset and time splits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_rank_dataset(valid_stocks, all_features, all_returns, min_group_size=5):
    """Build date-sorted cross-sections with ordinal relevance labels.

    Returns X, relevance, raw_returns, stock_codes, feature_names, sample_dates,
    group_sizes. Every date is an independent LightGBM ranking group.
    """
    feature_names = sorted({
        col
        for code in valid_stocks
        for col in getattr(all_features.get(code), "columns", [])
    })
    if len(feature_names) < 1:
        raise ValueError("no feature columns available")

    rows = []
    for code in valid_stocks:
        features = all_features.get(code)
        returns = all_returns.get(code)
        if features is None or returns is None:
            continue
        if not features.index.is_unique or not returns.index.is_unique:
            raise ValueError(f"duplicate dates for stock {code}")
        common = features.index.intersection(returns.dropna().index)
        for date in common:
            row = features.loc[date]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"duplicate feature row for {code} on {date}")
            values = pd.to_numeric(row.reindex(feature_names), errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).any():
                continue
            raw_return = float(returns.loc[date])
            if not np.isfinite(raw_return):
                continue
            rows.append((pd.Timestamp(date).normalize(), str(code),
                         values.to_numpy(dtype=float), raw_return))

    if not rows:
        raise ValueError("no valid cross-sectional training rows")
    frame = pd.DataFrame(rows, columns=["date", "code", "features", "return"])
    frame = frame.sort_values(["date", "code"], kind="mergesort")
    kept = []
    for date, group in frame.groupby("date", sort=True):
        if len(group) >= int(min_group_size):
            kept.append(group.assign(
                relevance=pd.qcut(
                    group["return"].rank(method="first"),
                    q=min(5, len(group)),
                    labels=False,
                    duplicates="drop",
                ).astype(int)
            ))
    if not kept:
        raise ValueError("no dates meet minimum cross-sectional group size")

    frame = pd.concat(kept, ignore_index=True)
    X = np.vstack(frame["features"].map(
        lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    ).to_list()).astype(np.float32)
    groups = [int(len(g)) for _, g in frame.groupby("date", sort=True)]
    return (
        X,
        frame["relevance"].to_numpy(dtype=np.int32),
        frame["return"].to_numpy(dtype=float),
        frame["code"].tolist(),
        feature_names,
        pd.to_datetime(frame["date"]).tolist(),
        groups,
    )


def purged_walk_forward_splits(sample_dates, trading_calendar, n_splits=5,
                               forward_days=5, embargo_days=2,
                               min_train_samples=20, min_validation_samples=5):
    """Yield expanding-window train/validation indices; unknown dates fail closed."""
    dates = pd.DatetimeIndex(pd.to_datetime(sample_dates)).normalize()
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    if dates.isna().any() or calendar.isna().any():
        raise ValueError("sample dates and trading calendar must be valid")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("trading calendar must be unique and sorted")
    if dates.empty or calendar.empty:
        raise ValueError("sample dates and trading calendar must not be empty")
    ranks = calendar.get_indexer(dates)
    if (ranks < 0).any():
        unknown = sorted({str(d.date()) for d, rank in zip(dates, ranks) if rank < 0})
        raise ValueError("sample dates missing from trading calendar: " + ", ".join(unknown))
    if int(forward_days) < 0 or int(embargo_days) < 0 or int(n_splits) < 2:
        raise ValueError("forward_days/embargo_days must be non-negative and n_splits >= 2")

    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    groups = [g for g in np.array_split(unique_dates, min(int(n_splits), len(unique_dates))) if len(g)]
    for fold, val_dates in enumerate(groups):
        start = val_dates[0]
        start_rank = int(calendar.get_loc(start))
        val_mask = dates.isin(val_dates)
        val_idx = np.flatnonzero(val_mask)
        prior_idx = np.flatnonzero(dates < start)
        cutoff = start_rank - int(embargo_days)
        train_idx = prior_idx[
            (ranks[prior_idx] + int(forward_days)) < cutoff
        ]
        if len(train_idx) >= min_train_samples and len(val_idx) >= min_validation_samples:
            yield fold, train_idx, val_idx, str(start.date()), str(val_dates[-1].date())


def rank_metrics(sample_dates, raw_returns, scores, n_quantiles=5):
    """Compute mean daily RankIC and mean top-minus-bottom quantile return."""
    dates = pd.DatetimeIndex(pd.to_datetime(sample_dates)).normalize()
    returns = np.asarray(raw_returns, dtype=float)
    scores = np.asarray(scores, dtype=float)
    daily_ic, spreads = [], []
    for date in dates.unique():
        mask = dates == date
        if mask.sum() < 5:
            continue
        r = pd.Series(returns[mask])
        s = pd.Series(scores[mask])
        ic = s.corr(r, method="spearman")
        if pd.notna(ic):
            daily_ic.append(float(ic))
        ranks = s.rank(method="first", pct=True)
        top = returns[mask][ranks.to_numpy() > (1 - 1 / n_quantiles)]
        bottom = returns[mask][ranks.to_numpy() <= (1 / n_quantiles)]
        if len(top) and len(bottom):
            spreads.append(float(np.mean(top) - np.mean(bottom)))
    return {
        "rank_ic": float(np.mean(daily_ic)) if daily_ic else None,
        "rank_ic_std": float(np.std(daily_ic)) if daily_ic else None,
        "top_bottom_return": float(np.mean(spreads)) if spreads else None,
        "n_dates": len(daily_ic),
    }
