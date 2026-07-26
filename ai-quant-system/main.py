#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI量化交易系统 - 主入口
2026年实战组合方案

使用方式:
    python main.py --mode backtest --start 2024-01-01 --end 2025-12-31
    python main.py --mode live
    python main.py --mode screening --date 2026-07-21
    python main.py --demo
"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import QuantPipeline
from utils.logger import setup_logger, get_logger
from utils.config_loader import load_config


def demo_mode():
    """演示模式 - 展示系统架构和设计理念"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           AI量化交易系统 v2026 - 实战组合方案               ║
║                                                              ║
║         AI量化的核心竞争力:                                  ║
║         "把不同模型放在对的位置上"                            ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐  ║
║  │  第1层: 初筛层 (Screening Layer)                     │  ║
║  │  ──────────────────────────────────────────────────  │  ║
║  │  LightGBM + 全市场因子选股                           │  ║
║  │  负责: "挖Alpha" - 从全市场选出候选股票池            │  ║
║  │  功能: 因子计算 → LightGBM预测 → 排序选股            │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                          ↓                                  ║
║  ┌──────────────────────────────────────────────────────┐  ║
║  │  第2层: 精研层 (Research Layer)                      │  ║
║  │  ──────────────────────────────────────────────────  │  ║
║  │  TradingAgents 多智能体系统                          │  ║
║  │  负责: "读信息" - 研报/新闻逻辑验真                  │  ║
║  │  Agent1: 情感分析 → 新闻情感评分                     │  ║
║  │  Agent2: 风险识别 → 排除风险标的                     │  ║
║  │  Agent3: 新闻分析 → 时效性评估                       │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                          ↓                                  ║
║  ┌──────────────────────────────────────────────────────┐  ║
║  │  第3层: 执行层 (Execution Layer)                     │  ║
║  │  ──────────────────────────────────────────────────  │  ║
║  │  PPO 强化学习动态仓位管理                            │  ║
║  │  负责: "调仓位" - 根据市场状态动态调整仓位           │  ║
║  │  功能: PortfolioEnv环境 → PPO训练 → 仓位优化         │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                          ↓                                  ║
║  ┌──────────────────────────────────────────────────────┐  ║
║  │  第4层: 风控层 (Risk Layer)                          │  ║
║  │  ──────────────────────────────────────────────────  │  ║
║  │  DoubleEnsemble 低过拟合风控                         │  ║
║  │  负责: "安全垫" - 多层风控保障                       │  ║
║  │  个股风控: 止损/移动止损                             │  ║
║  │  组合风控: VaR/波动率/集中度                         │  ║
║  │  市场风控: 熔断/波动率目标                           │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  核心设计原则:                                               ║
║  • 树模型(LightGBM)负责"挖Alpha"                           ║
║  • 强化学习(PPO)负责"调仓位"                               ║
║  • 多智能体(TradingAgents)负责"读信息"                     ║
║  • 各司其职、层层过滤                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def run_screening(args):
    """运行初筛"""
    from layers.screening import ScreeningLayer
    from utils.config_loader import load_config

    config = load_config()
    screening = ScreeningLayer(config.get("screening", {}))

    result = screening.run_screening(
        date=args.date,
        universe=args.universe,
        top_k=args.top_k,
    )

    print(f"\n初筛结果 ({result['date']}):")
    print(f"  有效股票: {result['valid_stocks']}")
    print(f"  失败股票: {result['failed_stocks']}")
    print(f"  选出股票: {len(result['candidate_stocks'])}")

    if result['candidate_stocks']:
        print(f"\n  Top10:")
        for i, stock in enumerate(result['candidate_stocks'][:10]):
            print(f"    {i+1}. {stock['code']} - 得分: {stock['score']:.4f}")

    return result


def run_backtest(args):
    """运行回测"""
    pipeline = QuantPipeline()

    result = pipeline.run_backtest(
        start_date=args.start,
        end_date=args.end,
    )

    metrics = result['metrics']
    print(f"\n{'='*60}")
    print(f"回测结果 ({result['period']['start']} ~ {result['period']['end']})")
    print(f"{'='*60}")
    print(f"  初始资金:     {metrics.get('initial_capital', 1000000):,.0f}")
    print(f"  最终净值:     {metrics.get('final_value', 0):,.0f}")
    print(f"  总收益率:     {metrics.get('total_return', 0):.2%}")
    print(f"  年化收益:     {metrics.get('annualized_return', 0):.2%}")
    print(f"  夏普比率:     {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  最大回撤:     {metrics.get('max_drawdown', 0):.2%}")
    print(f"  胜率:         {metrics.get('win_rate', 0):.2%}")
    print(f"  盈亏比:       {metrics.get('profit_loss_ratio', 0):.3f}")
    print(f"  Calmar比率:   {metrics.get('calmar_ratio', 0):.3f}")
    print(f"  Sortino比率:  {metrics.get('sortino_ratio', 0):.3f}")
    print(f"  交易天数:     {metrics.get('n_trading_days', 0)}")
    print(f"  交易次数:     {metrics.get('n_trades', 0)}")
    print(f"{'='*60}")

    return result


def run_live(args):
    """运行实盘"""
    pipeline = QuantPipeline()

    print("⚠️  实盘交易模式")
    print("   请确认已配置真实数据源和交易接口")

    result = pipeline.run_live(
        stock_pool=args.universe,
        max_days=args.max_days,
    )

    print(f"\n实盘结果:")
    print(f"  交易天数: {result['n_days']}")
    print(f"  最终净值: {result['final_value']:,.0f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AI量化交易系统 - 2026实战组合方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --demo                                    # 演示模式
  python main.py --mode screening --date 2026-07-21       # 运行初筛
  python main.py --mode backtest --start 2024-01-01       # 运行回测
  python main.py --mode live --max-days 5                 # 运行实盘
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["demo", "screening", "backtest", "live"],
        default="demo",
        help="运行模式: demo(演示) / screening(初筛) / backtest(回测) / live(实盘)",
    )
    parser.add_argument("--date", help="交易日期 (YYYY-MM-DD)")
    parser.add_argument("--start", help="回测起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--universe", default="hs300", help="股票池: hs300 / zz500 / gem")
    parser.add_argument("--top-k", type=int, default=50, help="选股数量")
    parser.add_argument("--max-days", type=int, default=10, help="最大交易日数")
    parser.add_argument("--config", help="配置文件路径")

    args = parser.parse_args()

    # 初始化日志
    setup_logger(log_dir="../logs", log_level="INFO")
    log = get_logger("main")

    if args.mode == "demo":
        demo_mode()

    elif args.mode == "screening":
        log.info("运行初筛模式")
        run_screening(args)

    elif args.mode == "backtest":
        log.info("运行回测模式")
        run_backtest(args)

    elif args.mode == "live":
        log.info("运行实盘模式")
        run_live(args)


if __name__ == "__main__":
    main()
