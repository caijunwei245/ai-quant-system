#!/usr/bin/env python
"""Rebuild CSI300 OHLC from independent Sina qfq bars; preserve panel trading fields."""
import argparse
import json
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]


def _symbol(code):
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def _load_sina_qfq(code, start, end):
    fetcher = getattr(ak, "stock_zh_a_hist_sina", None) or getattr(ak, "stock_zh_a_daily", None)
    if fetcher is None:
        raise RuntimeError("AKShare Sina daily history endpoint is unavailable")
    raw = fetcher(symbol=_symbol(code), start_date=start, end_date=end, adjust="qfq")
    if raw is None or raw.empty:
        raise RuntimeError(f"Sina qfq returned no data for {code}")
    raw = raw.reset_index() if isinstance(raw.index, pd.DatetimeIndex) else raw.copy()
    if "date" not in raw.columns and "日期" in raw.columns:
        raw = raw.rename(columns={"日期": "date"})
    required = {"date", *PRICE_COLS, "volume", "amount"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Sina schema for {code} missing {sorted(missing)}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    for col in PRICE_COLS + ["volume", "amount"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    if raw["date"].duplicated().any():
        raise ValueError(f"Sina returned duplicate dates for {code}")
    return raw[["date"] + PRICE_COLS + ["volume", "amount"]]


def repair_panel(panel_path, output_path, report_path, sleep_seconds=0.5):
    panel_path, output_path, report_path = map(Path, (panel_path, output_path, report_path))
    if not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    df = pd.read_csv(panel_path, compression="gzip" if panel_path.suffix == ".gz" else None)
    df.columns = [str(c).strip().lstrip("\ufeff").strip() for c in df.columns]
    required = {"code", "date", *PRICE_COLS, "volume", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")
    df["code"] = df["code"].astype(str).str.extract(r"(\d{6})")[0]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["code"].isna().any() or df["date"].isna().any():
        raise ValueError("Panel contains invalid stock codes or dates")
    if df.duplicated(["code", "date"]).any():
        raise ValueError("Panel contains duplicate code-date rows")
    for col in PRICE_COLS + ["volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["code", "date", "volume", "amount"]].isna().any().any():
        raise ValueError("Panel has null code/date/volume/amount values")

    repaired = df.copy()
    repaired["price_source"] = "akshare_sina_qfq"
    repaired["price_adjustment"] = "qfq"
    stock_reports = {}
    failures = []
    start = df["date"].min().strftime("%Y%m%d")
    end = df["date"].max().strftime("%Y%m%d")

    for idx, code in enumerate(sorted(df["code"].unique()), start=1):
        local_idx = repaired.index[repaired["code"] == code]
        local = repaired.loc[local_idx].sort_values("date").copy()
        local["_row_id"] = local.index
        try:
            sina = _load_sina_qfq(code, start, end)
            merged = local[["date", "volume", "amount"]].merge(
                sina, on="date", how="left", validate="one_to_one",
                suffixes=("_panel", "_sina"), indicator=True
            )
            missing_mask = merged["_merge"] == "left_only"
            missing_positive_volume = missing_mask & (merged["volume_panel"] > 0)
            if missing_positive_volume.any():
                raise ValueError(
                    f"{int(missing_positive_volume.sum())} traded panel dates missing from Sina"
                )

            matched = merged["_merge"] == "both"
            traded = merged["volume_panel"] > 0
            if not (matched | ~traded).all():
                raise ValueError("Some traded rows do not have independent Sina prices")

            # Sina reports volume in shares; this panel records lots for these symbols.
            # Confirm scale from positive-volume rows, but preserve original panel volume.
            active = matched & traded
            if active.sum() == 0:
                raise ValueError("No overlapping traded rows for volume/amount reconciliation")
            volume_ratio = merged.loc[active, "volume_sina"] / merged.loc[active, "volume_panel"]
            median_volume_ratio = float(volume_ratio.median())
            if min(abs(median_volume_ratio - 100.0), abs(median_volume_ratio - 1.0)) > 0.02:
                raise ValueError(f"Unexpected Sina/panel volume unit ratio: {median_volume_ratio}")

            amount_active = active & (merged["amount_panel"] > 0)
            amount_rel = (
                (merged.loc[amount_active, "amount_sina"] - merged.loc[amount_active, "amount_panel"]).abs()
                / merged.loc[amount_active, "amount_panel"].abs().clip(lower=1)
            )
            amount_p99 = float(amount_rel.quantile(0.99)) if len(amount_rel) else None
            if amount_p99 is None or amount_p99 > 0.001:
                raise ValueError(f"Amount reconciliation p99 relative error too large: {amount_p99}")

            price_frame = merged[["date"] + PRICE_COLS].copy()
            price_frame.columns = ["date"] + PRICE_COLS
            local = local.drop(columns=PRICE_COLS + ["price_source", "price_adjustment"]).merge(
                price_frame, on="date", how="left", validate="one_to_one"
            ).sort_values("date")
            # A missing Sina date is allowed only when the original panel has zero volume
            # (suspension/no-trade). Carry forward the last available adjusted close,
            # never a future value, and mark the imputed row explicitly.
            local["price_source"] = "akshare_sina_qfq"
            local["price_adjustment"] = "qfq"
            missing_prices = local["close"].isna()
            zero_volume_missing = missing_prices & (local["volume"] == 0)
            if (missing_prices & ~zero_volume_missing).any():
                raise ValueError("Missing Sina prices on a traded date")
            previous_close = local["close"].ffill()
            if (zero_volume_missing & previous_close.isna()).any():
                raise ValueError("Cannot forward-fill initial zero-volume rows without prior price")
            local.loc[zero_volume_missing, PRICE_COLS] = np.repeat(
                previous_close.loc[zero_volume_missing].to_numpy()[:, None], 4, axis=1
            )
            local.loc[zero_volume_missing, "price_source"] = "akshare_sina_qfq_forward_fill_no_trade"
            local["price_adjustment"] = "qfq"

            bad_price = (
                local[PRICE_COLS].isna().any(axis=1)
                | (local[PRICE_COLS] <= 0).any(axis=1)
                | (local["high"] < local[["open", "close", "low"]].max(axis=1))
                | (local["low"] > local[["open", "close", "high"]].min(axis=1))
            )
            if bad_price.any():
                raise ValueError(f"Rebuilt OHLC invalid rows: {int(bad_price.sum())}")

            # Replace OHLC/provenance by original row IDs; preserve all trading fields.
            row_ids = local["_row_id"].to_numpy(dtype=int)
            repaired.loc[row_ids, PRICE_COLS] = local[PRICE_COLS].to_numpy()
            repaired.loc[row_ids, "price_source"] = local["price_source"].to_numpy()
            repaired.loc[row_ids, "price_adjustment"] = "qfq"

            stock_reports[code] = {
                "panel_rows": int(len(local)),
                "sina_rows": int(len(sina)),
                "matched_dates": int(matched.sum()),
                "forward_filled_zero_volume_rows": int(zero_volume_missing.sum()),
                "median_sina_volume_to_panel_volume_ratio": median_volume_ratio,
                "amount_relative_error_p99": amount_p99,
                "bad_rebuilt_price_rows": int(bad_price.sum()),
            }
            print(f"[{idx}/{df['code'].nunique()}] repaired {code}", flush=True)
        except Exception as exc:
            failures.append({"code": code, "error": repr(exc)})
            print(f"[{idx}/{df['code'].nunique()}] FAILED {code}: {exc}", flush=True)
        time.sleep(sleep_seconds)

    report = {
        "status": "failed" if failures else "passed",
        "source": "AKShare stock_zh_a_hist_sina",
        "adjustment": "qfq",
        "panel_rows": int(len(df)),
        "stocks": int(df["code"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "source_volume_unit": "shares",
        "panel_volume_unit": "preserved from original panel; expected lots when ratio=100",
        "price_fields_replaced": PRICE_COLS,
        "non_price_fields_preserved": [c for c in df.columns if c not in PRICE_COLS],
        "stocks_repaired": len(stock_reports),
        "failures": failures,
        "per_stock": stock_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Refusing to publish incomplete repair: {len(failures)} stock failures")
    if len(stock_reports) != df["code"].nunique():
        raise RuntimeError("Not all panel stocks were repaired")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired = repaired.sort_values(["code", "date"])
    if output_path.suffix == ".gz":
        repaired.to_csv(output_path, index=False, compression="gzip")
    else:
        repaired.to_csv(output_path, index=False)
    print(json.dumps({
        "status": "passed", "output": str(output_path),
        "rows": int(len(repaired)), "stocks": int(repaired["code"].nunique()),
        "report": str(report_path)
    }, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()
    repair_panel(args.panel, args.output, args.report, args.sleep_seconds)


if __name__ == "__main__":
    main()
