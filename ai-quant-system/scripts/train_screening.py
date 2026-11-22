#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the Screening (LightGBM) model of the AI quant system.

Two modes:

  * Panel mode (default): load a prepared CSI300 daily panel
    file (data/csi300_daily_2010_2026.csv[.gz]) via ``data.panel_loader``,
    compute factors + N-day forward-return labels per stock, build a
    cross-sectional training set, run Purged-K-Fold CV with embargo and train
    a final LightGBM model. Artifacts are saved under ``../models``.

  * Live mode (``--live``): use ``DataFetcher`` (requires network / akshare)
    and the full ``ScreeningLayer.run_screening`` pipeline.

Usage:
  # offline, from the prepared panel (default path)
  python scripts/train_screening.py
  python scripts/train_screening.py --panel data/csi300_daily_2010_2026.csv.gz

  # live (needs network / data source configured in DataFetcher)
  python scripts/train_screening.py --live --date 2026-07-21 --universe hs300
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from layers.screening import ScreeningLayer
from data.panel_loader import load_panel, iter_stock_frames
from utils.config_loader import load_config


def build_dataset_from_panel(panel_path, factor_processor, forward_days):
    """Build (valid_stocks, all_features, all_labels) from a panel file.

    Each stock contributes a (date-indexed) factor frame and a forward-return
    label series. Only rows where both factor and label are available are kept.
    """
    df = load_panel(panel_path)
    logger.info(
        "Panel loaded: {} rows, {} stocks, {} ~ {}".format(
            len(df), df["code"].nunique(),
            df["date"].min().date(), df["date"].max().date(),
        )
    )

    all_features = {}
    all_labels = {}
    valid_stocks = []
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
        except Exception as e:  # noqa: BLE001
            logger.debug("skip {}: {}".format(code, e))
            skipped += 1
            continue

    logger.info(
        "Built features/labels for {} stocks (skipped {}).".format(
            len(valid_stocks), skipped
        )
    )
    return valid_stocks, all_features, all_labels


def train_from_panel(panel_path, config, args):
    lgb_config = dict(config.get("lightgbm", {}))
    # The final untouched OOS period is a hard deployment gate.
    lgb_config["min_oos_rank_ic"] = float(args.min_oos_rank_ic)
    # Lookback window for the cross-sectional training set. 0 = use all history
    # (recommended for the panel; the repo default is 30 trading days).
    lookback = int(args.lookback_days)
    lgb_config["lookback_days"] = 10 ** 9 if lookback <= 0 else lookback
    forward_days = int(lgb_config.get("forward_days", 5))

    layer = ScreeningLayer({"lightgbm": lgb_config, "data": config.get("data", {})})

    valid_stocks, all_features, all_labels = build_dataset_from_panel(
        panel_path, layer.factor_processor, forward_days
    )
    if len(valid_stocks) < 10:
        logger.error("Too few valid stocks ({}); aborting.".format(len(valid_stocks)))
        return

    layer._train_model(valid_stocks, all_features, all_labels)
    layer.save_model()

    meta_path = layer.model_dir / "screening_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cv = meta.get("cv", {})
        oos = meta.get("oos", {})
        logger.info(
            "Trained Ranker saved -> {} | CV RankIC={}, OOS RankIC={}, samples={}".format(
                layer.model_dir, cv.get("mean_rank_ic"), oos.get("rank_ic"), meta.get("samples")
            )
        )
    else:
        logger.warning("Training did not produce screening_meta.json (CV gate).")


def train_live(config, args):
    screening_config = dict(config.get("screening", {}))
    screening_config["data"] = config.get("data", {})
    lgb_config = dict(screening_config.get("lightgbm", {}))
    lgb_config["min_oos_rank_ic"] = float(args.min_oos_rank_ic)
    screening_config["lightgbm"] = lgb_config
    layer = ScreeningLayer(screening_config)
    if layer.load_model() and not args.force:
        print("Existing screening model loaded. Use --force to retrain.")
        return
    res = layer.run_screening(
        date=args.date, universe=args.universe, top_k=args.top_k
    )
    layer.save_model()
    print("Live training finished.")
    print("  valid_stocks: {}".format(res.get("valid_stocks", 0)))
    print("  candidate_stocks: {}".format(len(res.get("candidate_stocks", []))))


def resolve_panel_path(default_path):
    """Return the panel path to use, falling back to the committed sample."""
    p = Path(default_path)
    if p.exists():
        return p
    sample = Path("data/csi300_sample.csv")
    if sample.exists():
        logger.warning(
            "Full panel '{}' not found; using sample '{}' for a smoke run.".format(
                p, sample
            )
        )
        return sample
    return p  # let the caller report the missing file


def main():
    parser = argparse.ArgumentParser(description="Train Screening (LightGBM) model")
    parser.add_argument(
        "--panel",
        default="data/csi300_daily_2010_2026.csv.gz",
        help="Path to CSI300 daily panel (csv or csv.gz).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live DataFetcher instead of the prepared panel.",
    )
    parser.add_argument("--date", help="Reference date (live mode)", default=None)
    parser.add_argument("--universe", help="Stock universe (live mode)", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--min-oos-rank-ic",
        type=float,
        default=0.0,
        help="Minimum independent OOS mean daily RankIC required to accept the Ranker.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=0,
        help="Cross-sectional lookback window (0 = use all available history).",
    )
    parser.add_argument("--force", action="store_true", help="Force retrain.")
    args = parser.parse_args()

    config = load_config()

    if args.live:
        train_live(config, args)
        return

    panel_path = resolve_panel_path(args.panel)
    if not panel_path.exists():
        logger.error(
            "Panel file not found: {}. Download it (see data/README.md) or use --live.".format(
                panel_path
            )
        )
        sys.exit(1)

    train_from_panel(panel_path, config, args)


if __name__ == "__main__":
    main()
