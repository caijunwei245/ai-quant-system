#!/usr/bin/env python
"""Fail-closed integrity and coverage audit for the full CSI300 historical panel."""
import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {"code", "date", "open", "high", "low", "close", "volume"}


def audit_panel(path, min_rows=1_000_000, min_stocks=200, min_dates=1_000,
                min_start="2011-01-01", min_end="2025-01-01"):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Historical panel missing or empty: {path}")
    # pandas reads the complete gzip stream; truncated/corrupt gzip files fail here.
    df = pd.read_csv(path, compression="gzip" if path.suffix == ".gz" else None)
    df.columns = [str(c).strip().lstrip("\ufeff").strip() for c in df.columns]
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["code"] = df["code"].astype(str).str.strip().str.lower().str.replace(r"\D", "", regex=True).str[-6:]
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_dates = int(df["date"].isna().sum())
    invalid_codes = int((df["code"].str.len() != 6).sum())
    null_required = int(df[["code", "date"] + numeric].isna().any(axis=1).sum())
    duplicates = int(df.duplicated(["code", "date"]).sum())
    zero_volume = df["volume"] == 0
    positive_volume = df["volume"] > 0
    nonpositive_price = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    inconsistent_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    negative_volume = df["volume"] < 0
    # A zero-volume row is a non-trading/suspension observation and is excluded
    # from factor construction. Invalid prices/OHLC on a traded row remain fatal.
    bad_traded_mask = positive_volume & (nonpositive_price | inconsistent_ohlc)
    bad_traded_rows = int(bad_traded_mask.sum())
    ohlc_issues = {
        "nonpositive_price_rows_with_volume": int((positive_volume & nonpositive_price).sum()),
        "high_below_low_rows_with_volume": int((positive_volume & (df["high"] < df["low"])).sum()),
        "high_below_open_rows_with_volume": int((positive_volume & (df["high"] < df["open"])).sum()),
        "high_below_close_rows_with_volume": int((positive_volume & (df["high"] < df["close"])).sum()),
        "low_above_open_rows_with_volume": int((positive_volume & (df["low"] > df["open"])).sum()),
        "low_above_close_rows_with_volume": int((positive_volume & (df["low"] > df["close"])).sum()),
    }
    issue_examples = df.loc[
        bad_traded_mask, ["code", "date", "open", "high", "low", "close", "volume"]
    ].head(20).astype({"date": str}).to_dict(orient="records")
    bad_by_code = (
        df.loc[bad_traded_mask].groupby("code").size()
        .sort_values(ascending=False).head(20).astype(int).to_dict()
    )
    bad_by_year = (
        df.loc[bad_traded_mask].assign(year=df.loc[bad_traded_mask, "date"].dt.year)
        .groupby("year").size().astype(int).to_dict()
    )
    price_quantiles = {
        col: {str(q): float(df.loc[positive_volume, col].quantile(q))
              for q in [0.0, 0.01, 0.5, 0.99, 1.0]}
        for col in ["open", "high", "low", "close"]
    }
    sample_600000 = (
        df.loc[df["code"] == "600000", ["date", "open", "high", "low", "close", "volume"]]
        .iloc[::max(1, int((df["code"] == "600000").sum() / 12))]
        .head(12).astype({"date": str}).to_dict(orient="records")
    )
    bad_ohlc = int((negative_volume | (positive_volume & (nonpositive_price | inconsistent_ohlc))).sum())
    all_zero_ohlc = int((df[["open", "high", "low", "close"]] == 0).all(axis=1).sum())
    date_min = df["date"].min()
    date_max = df["date"].max()
    stocks = int(df["code"].nunique())
    dates = int(df["date"].nunique())
    rows_per_date = df.groupby("date").size()
    rows_per_stock = df.groupby("code").size()
    audit = {
        "path": str(path),
        "file_bytes": int(path.stat().st_size),
        "rows": int(len(df)),
        "stocks": stocks,
        "trading_dates": dates,
        "date_min": str(date_min.date()) if pd.notna(date_min) else None,
        "date_max": str(date_max.date()) if pd.notna(date_max) else None,
        "invalid_dates": invalid_dates,
        "invalid_stock_codes": invalid_codes,
        "rows_with_null_required_fields": null_required,
        "duplicate_code_date_rows": duplicates,
        "zero_volume_rows": int(zero_volume.sum()),
        "all_zero_ohlc_rows": all_zero_ohlc,
        "bad_traded_ohlc_rows": bad_traded_rows,
        "ohlc_issue_breakdown": ohlc_issues,
        "bad_traded_row_examples": issue_examples,
        "bad_traded_rows_by_stock": bad_by_code,
        "bad_traded_rows_by_year": bad_by_year,
        "traded_price_quantiles": price_quantiles,
        "sample_rows_600000": sample_600000,
        "negative_volume_rows": int(negative_volume.sum()),
        "invalid_ohlc_or_volume_rows": bad_ohlc,
        "rows_per_date": {
            "min": int(rows_per_date.min()) if len(rows_per_date) else 0,
            "median": float(rows_per_date.median()) if len(rows_per_date) else 0,
            "max": int(rows_per_date.max()) if len(rows_per_date) else 0,
        },
        "rows_per_stock": {
            "min": int(rows_per_stock.min()) if len(rows_per_stock) else 0,
            "median": float(rows_per_stock.median()) if len(rows_per_stock) else 0,
            "max": int(rows_per_stock.max()) if len(rows_per_stock) else 0,
        },
    }
    failures = []
    if len(df) < min_rows: failures.append(f"rows {len(df)} < {min_rows}")
    if stocks < min_stocks: failures.append(f"stocks {stocks} < {min_stocks}")
    if dates < min_dates: failures.append(f"trading dates {dates} < {min_dates}")
    if pd.isna(date_min) or date_min > pd.Timestamp(min_start): failures.append(f"date_min {date_min} later than {min_start}")
    if pd.isna(date_max) or date_max < pd.Timestamp(min_end): failures.append(f"date_max {date_max} earlier than {min_end}")
    if invalid_dates or invalid_codes or null_required or duplicates or bad_ohlc:
        failures.append(
            f"integrity violations: invalid_dates={invalid_dates}, invalid_codes={invalid_codes}, "
            f"null_required={null_required}, duplicate_code_date={duplicates}, bad_traded_ohlc={bad_traded_rows}, negative_volume={int(negative_volume.sum())}"
        )
    audit["status"] = "failed" if failures else "passed"
    audit["failures"] = failures
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", default="reports/alpha-training/data_audit.json")
    parser.add_argument("--min-rows", type=int, default=1_000_000)
    parser.add_argument("--min-stocks", type=int, default=200)
    parser.add_argument("--min-dates", type=int, default=1_000)
    parser.add_argument("--min-start", default="2011-01-01")
    parser.add_argument("--min-end", default="2025-01-01")
    args = parser.parse_args()
    audit = audit_panel(args.panel, args.min_rows, args.min_stocks, args.min_dates,
                        args.min_start, args.min_end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit("Historical panel validation failed: " + "; ".join(audit["failures"]))


if __name__ == "__main__":
    main()
