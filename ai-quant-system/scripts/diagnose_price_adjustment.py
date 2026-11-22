#!/usr/bin/env python
"""Compare suspect historical panel bars with independent AKShare adjustment modes."""
import argparse
import json
import time
from pathlib import Path

import akshare as ak
import pandas as pd

CODES = ["600000", "601919", "600188", "600039", "600519", "000651", "000858", "600809"]
START, END = "20100101", "20260807"
SUSPECT_DATES = ["2010-01-04", "2012-07-12", "2012-07-23", "2012-10-10",
                 "2013-06-03", "2014-03-04", "2015-07-14", "2022-06-06", "2025-03-11"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--output", default="reports/alpha-training/adjustment_diagnosis.json")
    args = p.parse_args()
    panel = pd.read_csv(args.panel, compression="gzip")
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel["code"] = panel["code"].astype(str).str.extract(r"(\d{6})")[0]
    result = {"provider": "AKShare stock_zh_a_hist (Eastmoney interface)",
              "range": [START, END], "codes": {}, "notes": [
                  "Comparison only; no source values are written into the historical panel.",
                  "Adjustment mode is recorded explicitly for each independent response."
              ]}
    for code in CODES:
        source = {}
        for adjust in ["", "qfq", "hfq"]:
            try:
                fetcher = getattr(ak, "stock_zh_a_hist_em", None) or getattr(ak, "stock_zh_a_hist", None)
                if fetcher is None:
                    raise RuntimeError("AKShare has no stock_zh_a_hist_em/stock_zh_a_hist endpoint")
                raw = fetcher(symbol=code, period="daily",
                    start_date=START, end_date=END, adjust=adjust)
                if raw is None or raw.empty:
                    source[adjust or "none"] = {"status": "empty"}
                    continue
                raw = raw.rename(columns={"日期":"date","开盘":"open","收盘":"close",
                    "最高":"high","最低":"low","成交量":"volume"})
                raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
                keep = raw[raw["date"].isin(SUSPECT_DATES)]
                local = panel[(panel["code"] == code) & panel["date"].isin(SUSPECT_DATES)]
                joined = keep.merge(local[["date","open","high","low","close","volume"]],
                                    on="date", how="left", suffixes=("_vendor","_panel"))
                source[adjust or "none"] = {
                    "status": "ok", "rows": int(len(raw)),
                    "date_min": raw["date"].min(), "date_max": raw["date"].max(),
                    "samples": joined.to_dict(orient="records")
                }
                time.sleep(0.6)
            except Exception as exc:
                source[adjust or "none"] = {"status":"error","error":repr(exc)}
        result["codes"][code] = source
    # Independent Sina endpoint probe for one representative stock. This
    # avoids repeatedly querying Eastmoney, which rate-limits bulk diagnostics.
    try:
        sina = ak.stock_zh_a_daily(
            symbol="sh600000", start_date=START, end_date=END, adjust="qfq"
        )
        sina = sina.reset_index() if isinstance(sina.index, pd.DatetimeIndex) else sina.copy()
        if "date" not in sina.columns and "日期" in sina.columns:
            sina = sina.rename(columns={"日期": "date"})
        if "date" in sina.columns:
            sina["date"] = pd.to_datetime(sina["date"]).dt.strftime("%Y-%m-%d")
            selected = sina[sina["date"].isin(SUSPECT_DATES)]
            result["sina_600000_qfq"] = {
                "status": "ok", "rows": int(len(sina)),
                "date_min": sina["date"].min(), "date_max": sina["date"].max(),
                "samples": selected.to_dict(orient="records"),
            }
        else:
            result["sina_600000_qfq"] = {
                "status": "unexpected_schema", "columns": list(sina.columns)
            }
    except Exception as exc:
        result["sina_600000_qfq"] = {"status": "error", "error": repr(exc)}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
