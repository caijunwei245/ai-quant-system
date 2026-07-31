#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train PPO model wrapper script
Usage:
  python scripts/train_ppo.py --episodes 50

This script uses ExecutionLayer.train which builds a PortfolioEnv internally and
trains a stable-baselines3 PPO model. It saves the model under ../models/ppo_model.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.fetcher import DataFetcher
from layers.execution import ExecutionLayer
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Train PPO model for execution layer")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes (each episode ~= 2048 steps)")
    parser.add_argument("--universe", default=None, help="Stock universe for training (e.g. hs300)")
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    cfg = load_config()
    exec_cfg = cfg.get("execution", {})

    # build a minimal stock_pool using DataFetcher
    fetcher = DataFetcher()
    market = args.universe or cfg.get("screening", {}).get("universe", "hs300")
    stock_list = fetcher.fetch_stock_list(market)

    # for market_data we can use index-level timeseries as a simple proxy
    index_df = fetcher.fetch_market_index("000300.SH")

    # prepare ExecutionLayer and call train
    exec_layer = ExecutionLayer(exec_cfg)

    # trim stock_list to a manageable size to speed up training in demo (e.g., top 50)
    stock_pool = stock_list[: min(len(stock_list), exec_cfg.get("max_train_universe", 50))]

    print(f"Starting PPO training | episodes={args.episodes} | universe={market} | n_stocks={len(stock_pool)}")
    result = exec_layer.train(stock_pool=stock_pool, market_data=index_df, n_episodes=args.episodes, verbose=args.verbose)

    print("Training result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("is_trained"):
        saved = exec_layer.save_model()
        print("Model saved:", saved)
    else:
        print("Training skipped or failed. See logs.")


if __name__ == "__main__":
    main()
