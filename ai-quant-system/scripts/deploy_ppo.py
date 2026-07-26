#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple deployment wrapper for PPO model
Usage:
  # Output positions for a small verified_stocks JSON file
  python scripts/deploy_ppo.py --input example_verified.json

This script demonstrates loading the saved PPO model and using it to produce
position suggestions for a given list of verified stocks. It's a minimal, local
wrapper intended for integration into a serving layer (Flask/FastAPI) later.
"""
import argparse
import json
from pathlib import Path

from layers.execution import ExecutionLayer


def load_verified(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def predict_positions(exec_layer: ExecutionLayer, verified_stocks):
    # Ensure model loaded
    loaded = exec_layer.load_model()
    if not loaded:
        raise RuntimeError("PPO model not found or failed to load. Train model first.")

    # Ask execution layer to optimize positions using PPO model
    res = exec_layer.optimize_positions(verified_stocks=verified_stocks, market_state=None)
    return res


def main():
    parser = argparse.ArgumentParser(description="Deploy PPO model to predict positions")
    parser.add_argument("--input", help="Path to JSON file with verified_stocks list", required=True)
    parser.add_argument("--model-dir", help="Optional model dir (ignored, using configured ../models)", default=None)
    args = parser.parse_args()

    verified_stocks = load_verified(Path(args.input))

    exec_layer = ExecutionLayer({})

    try:
        res = predict_positions(exec_layer, verified_stocks)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Error during prediction:", e)


if __name__ == "__main__":
    main()
