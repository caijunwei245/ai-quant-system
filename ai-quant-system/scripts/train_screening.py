#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train / update Screening (LightGBM) model wrapper script
Usage:
  python scripts/train_screening.py --date 2026-07-21 --universe hs300

This script will try to load an existing model. If missing, it runs the screening pipeline which
internally collects factors/labels and trains a LightGBM model (if lightgbm is installed).
After training it saves model artifacts under ../models.
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.screening import ScreeningLayer
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Train Screening (LightGBM) model")
    parser.add_argument("--date", help="Reference date for training data (YYYY-MM-DD)", default=None)
    parser.add_argument("--universe", help="Stock universe", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")

    args = parser.parse_args()

    config = load_config()
    screening = ScreeningLayer(config.get("screening", {}))

    # try to load existing model
    loaded = screening.load_model()
    if loaded and not args.force:
        print("Existing screening model loaded. Use --force to retrain.")
        return

    # run screening which will gather data and call internal training when possible
    print("Collecting data and training screening model (this may take a while)...")
    res = screening.run_screening(date=args.date, universe=args.universe, top_k=args.top_k)

    # Save the trained model (if applicable)
    screening.save_model()

    print("Training finished. Summary:")
    print(f"  Valid stocks processed: {res.get('valid_stocks', 0)}")
    print(f"  Candidate stocks selected: {len(res.get('candidate_stocks', []))}")
    print(f"  Feature importance keys: {len(res.get('factor_importance', {}) )}")


if __name__ == "__main__":
    main()
