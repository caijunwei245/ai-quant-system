#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the historical-panel LambdaRank model; never substitute sample data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger

from layers.screening import ScreeningLayer
from data.panel_loader import load_panel, iter_stock_frames
from utils.config_loader import load_config


def build_dataset_from_panel(panel_path, factor_processor, forward_days, panel_df=None):
    """Build per-stock features/forward-return labels from the explicitly supplied panel."""
    df = panel_df if panel_df is not None else load_panel(panel_path)
    logger.info(
        "Panel loaded: {} rows, {} stocks, {} ~ {}".format(
            len(df), df["code"].nunique(),
            df["date"].min().date(), df["date"].max().date(),
        )
    )
    all_features, all_labels, valid_stocks = {}, {}, []
    skipped = 0
    for code, sdf in iter_stock_frames(df):
        try:
            factors = factor_processor.compute_all_factors(sdf)
            if factors.empty:
                skipped += 1
                continue
            labels = factor_processor.generate_labels(sdf, forward_days=forward_days)
            common = factors.index.intersection(labels.index)
            if len(common) < 60:
                skipped += 1
                continue
            all_features[code] = factors.loc[common]
            all_labels[code] = labels.loc[common]
            valid_stocks.append(code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping stock {} during feature construction: {}", code, exc)
            skipped += 1
    logger.info("Built features/labels for {} stocks (skipped {}).", len(valid_stocks), skipped)
    return valid_stocks, all_features, all_labels


def _write_report(report_dir, report):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "alpha_training_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    cv = report["training"]["cv"]
    oos = report["training"]["oos"]
    lines = [
        "# Historical CSI300 Alpha Training Report", "",
        f"- Dataset: `{report['dataset']['path']}`",
        f"- Rows: {report['dataset']['rows']:,}",
        f"- Stocks / trading dates: {report['dataset']['stocks']:,} / {report['dataset']['trading_dates']:,}",
        f"- Date range: {report['dataset']['date_min']} to {report['dataset']['date_max']}",
        f"- Duplicate code-date rows: {report['dataset']['duplicate_code_date_rows']:,}",
        f"- Valid feature stocks: {report['training']['valid_feature_stocks']:,}",
        f"- Rank samples: {report['training']['samples']:,}",
        f"- CV mean RankIC / Top-Bottom: {cv.get('mean_rank_ic')} / {cv.get('mean_top_bottom_return')}",
        f"- OOS RankIC / Top-Bottom: {oos.get('rank_ic')} / {oos.get('top_bottom_return')}",
        f"- OOS dates: {report['training']['oos_coverage']['date_start']} to {report['training']['oos_coverage']['date_end']} "
        f"({report['training']['oos_coverage']['trading_dates']:,} dates)",
        f"- OOS samples / unique stocks: {report['training']['oos_coverage']['samples']:,} / "
        f"{report['training']['oos_coverage']['unique_stocks']:,}", "",
        "## Purged Walk-Forward CV folds", "",
        "| Fold | Validation range | Train samples | Validation samples | RankIC | Top-Bottom return |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for fold in cv.get("folds", []):
        lines.append(
            f"| {fold['fold']} | {fold['val_start']} – {fold['val_end']} | {fold['n_train']:,} | "
            f"{fold['n_val']:,} | {fold.get('rank_ic')} | {fold.get('top_bottom_return')} |"
        )
    lines += ["", "## OOS daily coverage", "",
              "| Metric | Value |", "|---|---:|"]
    for key, value in report["training"]["oos_coverage"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Interpretation", "",
              "This report describes historical-panel model validation only. A positive OOS RankIC or spread is not proof of deployable alpha; "
              "survivorship, point-in-time fundamentals, corporate actions, adjustment methodology and transaction-cost robustness remain separate reviews.", ""]
    (report_dir / "alpha_training_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Historical Alpha report written to {}", report_dir)


def train_from_panel(panel_path, config, args):
    panel_path = Path(panel_path)
    if not panel_path.is_file():
        raise FileNotFoundError(f"Explicit historical panel does not exist: {panel_path}")

    # Load exactly once and derive the exchange-session calendar from the panel itself.
    panel_df = load_panel(panel_path)
    trading_calendar = pd.DatetimeIndex(sorted(pd.to_datetime(panel_df["date"]).dt.normalize().unique()))
    if len(trading_calendar) < 1000 or panel_df["code"].nunique() < 200:
        raise ValueError(
            f"Refusing non-historical/sample panel: rows={len(panel_df)}, "
            f"stocks={panel_df['code'].nunique()}, dates={len(trading_calendar)}"
        )

    lgb_config = dict(config.get("screening", {}).get("lightgbm", config.get("lightgbm", {})))
    lgb_config["min_oos_rank_ic"] = float(args.min_oos_rank_ic)
    lgb_config["lookback_days"] = 10 ** 9 if args.lookback_days <= 0 else int(args.lookback_days)
    forward_days = int(lgb_config.get("forward_days", 5))
    layer = ScreeningLayer({"lightgbm": lgb_config, "data": config.get("data", {})})
    # The panel's observed market sessions are the authoritative calendar for this offline run.
    layer.data_fetcher.fetch_trading_calendar = lambda start, end: trading_calendar[
        (trading_calendar >= pd.Timestamp(start)) & (trading_calendar <= pd.Timestamp(end))
    ]

    valid_stocks, all_features, all_labels = build_dataset_from_panel(
        panel_path, layer.factor_processor, forward_days, panel_df=panel_df
    )
    if len(valid_stocks) < 100:
        raise RuntimeError(f"Too few valid feature stocks ({len(valid_stocks)}); aborting full-panel training.")

    layer._train_model(valid_stocks, all_features, all_labels)
    if not layer.is_trained:
        raise RuntimeError("Ranker training did not reach the trained state.")
    layer.save_model()

    meta_path = layer.model_dir / "screening_meta.json"
    if not meta_path.is_file():
        raise RuntimeError("Training completed without screening_meta.json; refusing success.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cv, oos = meta.get("cv", {}), meta.get("oos", {})
    if len(cv.get("folds", [])) < 2 or oos.get("rank_ic") is None:
        raise RuntimeError("Training report is missing required CV folds or independent OOS metrics.")

    dates = pd.to_datetime(panel_df["date"]).dt.normalize()
    rows_by_date = dates.value_counts()
    report = {
        "status": "success",
        "dataset": {
            "path": str(panel_path), "rows": int(len(panel_df)),
            "stocks": int(panel_df["code"].nunique()),
            "trading_dates": int(dates.nunique()),
            "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
            "duplicate_code_date_rows": int(panel_df.duplicated(["code", "date"]).sum()),
            "rows_per_date": {
                "min": int(rows_by_date.min()), "median": float(rows_by_date.median()),
                "max": int(rows_by_date.max()),
            },
        },
        "training": {
            "valid_feature_stocks": int(len(valid_stocks)),
            "samples": int(meta.get("samples", 0)),
            "cv": cv, "oos": oos,
            "oos_coverage": meta.get("oos_coverage", {}),
            "oos_start": meta.get("oos_start"), "oos_end": meta.get("oos_end"),
        },
    }
    _write_report(args.report_dir, report)
    print(json.dumps({"status": "success", "report_dir": args.report_dir,
                      "cv_folds": cv.get("n_folds"), "oos_rank_ic": oos.get("rank_ic")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Train historical CSI300 Screening LambdaRank model")
    parser.add_argument("--panel", required=True, help="Explicit full historical CSV/CSV.GZ panel; no sample fallback.")
    parser.add_argument("--min-oos-rank-ic", type=float, default=0.0)
    parser.add_argument("--lookback-days", type=int, default=0, help="0 = use all available history.")
    parser.add_argument("--report-dir", default="reports/alpha-training")
    args = parser.parse_args()
    config = load_config()
    train_from_panel(args.panel, config, args)


if __name__ == "__main__":
    main()
