"""
AI量化交易系统 - 主调度器 (Pipeline)
串联四层架构: 初筛 -> 精研 -> 执行 -> 风控
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger

from backtest.engine import BacktestEngine
from utils.config_loader import load_config
from layers.screening import ScreeningLayer
from layers.research import ResearchLayer
from layers.execution import ExecutionLayer
from layers.risk import RiskLayer
from utils.logger import setup_logger, get_logger
from utils.data_helper import (
    annualized_return, sharpe_ratio, max_drawdown, calmar_ratio, sortino_ratio, win_rate, profit_loss_ratio
)


class QuantPipeline:
    """
    量化交易主调度器
    串联四层架构，管理整个交易流程
    """

    def __init__(self, config_path: str = None):
        """
        初始化管道

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path)

        # 初始化日志
        system_config = self.config.get("system", {})
        setup_logger(
            log_dir=system_config.get("log_dir", "../logs"),
            log_level=system_config.get("log_level", "INFO"),
        )
        self.log = get_logger("pipeline")

        # 初始化四层
        self.screening = ScreeningLayer(self.config.get("screening", {}))
        self.research = ResearchLayer(self.config.get("research", {}))
        self.execution = ExecutionLayer(self.config.get("execution", {}))
        self.risk = RiskLayer(self.config.get("risk", {}))

        # 回测配置
        self.backtest_config = self.config.get("backtest", {})

        # 交易历史
        self.trade_history = []
        self.performance_history = []

        # 组合状态
        self.portfolio = {
            "value": self.backtest_config.get("initial_capital", 1000000),
            "positions": {},
            "cash": self.backtest_config.get("initial_capital", 1000000),
            "entry_date": {},
        }

    def run_daily(
        self,
        date: str = None,
        mode: str = "backtest"
    ) -> Dict:
        """
        执行单日交易流程

        Args:
            date: 交易日期
            mode: 模式 "backtest" / "live"

        Returns:
            当日交易结果
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self.log.info(f"\n{'='*70}")
        self.log.info(f"AI量化交易系统 | 日期: {date} | 模式: {mode}")
        self.log.info(f"{'='*70}")

        result = {
            "date": date,
            "mode": mode,
            "layers": {},
            "portfolio": {},
            "risk_level": "normal",
        }

        # ===== 第1层: 初筛 =====
        screening_config = self.config.get("screening", {})
        if screening_config.get("enabled", True):
            try:
                screening_result = self.screening.run_screening(date=date)
                result["layers"]["screening"] = screening_result
                self.log.info(f"初筛完成: {screening_result.get('candidate_count', 0)} -> {screening_result.get('verified_count', 0)}")
            except Exception as e:
                self.log.error(f"初筛层失败: {e}")
                screening_result = {"error": str(e), "candidate_stocks": []}
                result["layers"]["screening"] = screening_result
        else:
            screening_result = {"candidate_stocks": []}
            result["layers"]["screening"] = {"enabled": False}

        # ===== 第2层: 精研 =====
        research_config = self.config.get("research", {})
        if research_config.get("enabled", True) and screening_result.get("candidate_stocks"):
            try:
                research_result = self.research.run_research(
                    candidate_stocks=screening_result.get("candidate_stocks", []),
                    date=date,
                )
                result["layers"]["research"] = research_result
                self.log.info(f"精研完成: {research_result.get('verified_count', 0)}只通过")
            except Exception as e:
                self.log.error(f"精研层失败: {e}")
                research_result = {"verified_stocks": [], "error": str(e)}
                result["layers"]["research"] = research_result
        else:
            research_result = {"verified_stocks": []}
            result["layers"]["research"] = {"enabled": False}

        # ===== 第3层: 执行 =====
        execution_config = self.config.get("execution", {})
        if execution_config.get("enabled", True) and research_result.get("verified_stocks"):
            try:
                execution_result = self.execution.optimize_positions(
                    verified_stocks=research_result.get("verified_stocks", []),
                    market_state=self._get_market_state(date),
                    date=date,
                )
                result["layers"]["execution"] = execution_result
                self.log.info(f"仓位优化: {execution_result.get('n_positions', 0)}只 | 总权重: {execution_result.get('total_weight', 0):.2%}")
            except Exception as e:
                self.log.error(f"执行层失败: {e}")
                execution_result = {"positions": {}, "error": str(e)}
                result["layers"]["execution"] = execution_result
        else:
            execution_result = {"positions": {}}
            result["layers"]["execution"] = {"enabled": False}

        # ===== 第4层: 风控 =====
        risk_config = self.config.get("risk", {})
        if risk_config.get("enabled", True):
            try:
                daily_returns = self._compute_daily_returns(date)
                risk_result = self.risk.check_risk(
                    positions=execution_result.get("positions", {}),
                    portfolio_value=self.portfolio["value"],
                    daily_returns=daily_returns,
                    market_data=self._get_market_state(date),
                )
                result["layers"]["risk"] = risk_result
                self.log.info(f"风控检查: {risk_result.get('risk_level', 'normal')}")
            except Exception as e:
                self.log.error(f"风控层失败: {e}")
                risk_result = {"risk_level": "normal", "triggers": []}
                result["layers"]["risk"] = risk_result
        else:
            risk_result = {"risk_level": "normal"}
            result["layers"]["risk"] = {"enabled": False}

        # ===== 执行交易 =====
        if risk_result.get("risk_level") != "risk_off":
            trades = self._execute_trades(execution_result.get("positions", {}), date)
            result["trades"] = trades
        else:
            self.log.warning("风控熔断，执行全部平仓")
            trades = self._emergency_close(date)
            result["trades"] = trades
            result["risk_level"] = "risk_off"

        # ===== 更新组合状态 =====
        self._update_portfolio(date)
        result["portfolio"] = {
            "value": self.portfolio["value"],
            "cash": self.portfolio["cash"],
            "n_positions": len(self.portfolio["positions"]),
        }

        # ===== 记录绩效 =====
        perf = self._calculate_performance(date)
        self.performance_history.append(perf)
        result["performance"] = perf

        self.log.info(f"当日完成 | 组合净值: {self.portfolio['value']:,.0f} | 持仓数: {len(self.portfolio['positions'])}")

        return result

    def _get_backtest_target_weights(self, signal_date: str, stock_pool: str) -> dict:
        """Generate weights after signal_date close; they are filled next session."""
        screening_cfg = self.config.get("screening", {})
        if not screening_cfg.get("enabled", True):
            return {}

        screened = self.screening.run_screening(date=signal_date, universe=stock_pool)
        candidates = screened.get("candidate_stocks", [])
        research_cfg = self.config.get("research", {})
        if research_cfg.get("enabled", True) and candidates:
            researched = self.research.run_research(candidate_stocks=candidates, date=signal_date)
            verified = researched.get("verified_stocks", [])
        else:
            verified = candidates

        execution_cfg = self.config.get("execution", {})
        if execution_cfg.get("enabled", True) and verified:
            optimized = self.execution.optimize_positions(
                verified_stocks=verified,
                market_state=self._get_market_state(signal_date),
                date=signal_date,
            )
            weights = optimized.get("positions", {})
        else:
            weights = {}

        weights = {str(code): float(weight) for code, weight in weights.items()
                   if float(weight) > 0}
        total = sum(weights.values())
        if total > 1.0:
            weights = {code: weight / total for code, weight in weights.items()}
        return weights

    def _load_backtest_price_frame(self, code: str, start_date: str, end_date: str,
                                   cache: dict) -> pd.DataFrame:
        """Load unadjusted OHLCV bars once per symbol for execution and valuation."""
        if code in cache:
            return cache[code]
        fetcher = self.screening.data_fetcher
        frame = fetcher.fetch_stock_data(
            code, start_date=start_date, end_date=end_date, adjust=""
        )
        if frame is None or frame.empty:
            cache[code] = pd.DataFrame()
            return cache[code]
        frame = frame.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).set_index("date")
        else:
            frame.index = pd.to_datetime(frame.index, errors="coerce")
            frame = frame.loc[~frame.index.isna()]
        frame.index = frame.index.normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            missing = sorted(required - set(frame.columns))
            raise ValueError(f"{code} OHLCV data missing columns: {missing}")
        if "pre_close" not in frame.columns:
            frame["pre_close"] = pd.to_numeric(frame["close"], errors="coerce").shift(1)
        cache[code] = frame
        return frame

    def run_backtest(
        self,
        start_date: str = None,
        end_date: str = None,
        stock_pool: str = "all_a_share"
    ) -> Dict:
        """Run a daily-bar backtest with next-session-open fills and close marking."""
        if start_date is None:
            start_date = self.backtest_config.get("start_date", "2024-01-01")
        if end_date is None:
            end_date = self.backtest_config.get("end_date", "2025-12-31")

        self.log.info(f"开始回测 | {start_date} ~ {end_date}")
        engine_config = dict(self.backtest_config)
        engine = BacktestEngine(engine_config)
        trading_days = self._generate_trading_days(start_date, end_date)
        if len(trading_days) < 2:
            raise ValueError("回测至少需要两个真实交易日，才能使用前一日信号、下一日开盘成交")

        # Include warm-up bars for previous-close/limit checks.
        warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        price_cache = {}
        target_weights = {}
        results = []
        self.trade_history = []
        self.performance_history = []

        for index, date in enumerate(trading_days):
            trade_date = pd.Timestamp(date)
            active_codes = set(engine.positions) | set(target_weights)
            bars = {}
            for code in sorted(active_codes):
                frame = self._load_backtest_price_frame(code, warmup_start, end_date, price_cache)
                if not frame.empty and trade_date in frame.index:
                    bars[code] = frame.loc[trade_date]

            before = len(engine.trades)
            snapshot = engine.step(date, target_weights, bars)
            day_trades = engine.trades[before:]
            daily = {
                "date": date,
                "trades": day_trades,
                "portfolio": {
                    "value": snapshot["nav"],
                    "cash": snapshot["cash"],
                    "market_value": snapshot["market_value"],
                    "n_positions": snapshot["n_positions"],
                    "gross_exposure": snapshot["gross_exposure"],
                },
                "positions": snapshot["positions"],
                "stale_marks": snapshot["stale_marks"],
            }
            results.append(daily)
            self.performance_history.append({
                "date": date, "value": snapshot["nav"], "cash": snapshot["cash"],
                "daily_return": (snapshot["nav"] / results[-2]["portfolio"]["value"] - 1)
                                if len(results) > 1 and results[-2]["portfolio"]["value"] else 0.0,
            })

            # Today's close is now known. Only now generate the target for the
            # next exchange session, preventing same-close look-ahead fills.
            if index < len(trading_days) - 1:
                target_weights = self._get_backtest_target_weights(date, stock_pool)

        self.trade_history = engine.trades
        final_snapshot = engine.history[-1]
        self.portfolio = {
            "value": final_snapshot["nav"],
            "cash": final_snapshot["cash"],
            "positions": {code: data["shares"] for code, data in final_snapshot["positions"].items()},
            "entry_date": {},
        }

        values = [item["portfolio"]["value"] for item in results]
        daily_returns = pd.Series(values, dtype=float).pct_change().dropna()
        metrics = {
            "initial_capital": engine.initial_capital,
            "final_value": float(values[-1]),
            "total_return": float(values[-1] / engine.initial_capital - 1),
            "annualized_return": annualized_return(daily_returns) if len(daily_returns) else 0.0,
            "sharpe_ratio": sharpe_ratio(daily_returns) if len(daily_returns) else 0.0,
            "max_drawdown": max_drawdown(daily_returns) if len(daily_returns) else 0.0,
            "win_rate": win_rate(daily_returns) if len(daily_returns) else 0.0,
            "profit_loss_ratio": profit_loss_ratio(daily_returns) if len(daily_returns) else 0.0,
            "calmar_ratio": calmar_ratio(daily_returns) if len(daily_returns) else 0.0,
            "sortino_ratio": sortino_ratio(daily_returns) if len(daily_returns) else 0.0,
            "n_trading_days": int(len(daily_returns)),
            "n_trades": len(engine.trades),
            "values": values,
        }
        self._save_backtest_results(results, metrics)
        self.log.info(
            f"回测完成 | 总收益: {metrics['total_return']:.2%} | "
            f"夏普: {metrics['sharpe_ratio']:.3f} | 成交笔数: {metrics['n_trades']}"
        )
        return {
            "results": results,
            "metrics": metrics,
            "period": {"start": trading_days[0], "end": trading_days[-1]},
        }

    def run_live(
        self,
        stock_pool: str = "all_a_share",
        max_days: int = None
    ) -> Dict:
        """
        执行实盘交易

        Args:
            stock_pool: 股票池
            max_days: 最大交易日数

        Returns:
            实盘结果
        """
        self.log.info("开始实盘交易模式")

        day_count = 0
        results = []

        while True:
            date = datetime.now().strftime("%Y-%m-%d")

            # 检查是否已交易
            log_dir = Path("../logs")
            trade_log = log_dir / f"trades_{date}.log"
            if max_days and day_count >= max_days:
                break

            result = self.run_daily(date, mode="live")
            results.append(result)
            day_count += 1

            # 风控退出
            if result.get("risk_level") == "risk_off":
                self.log.warning("触发风控退出条件，停止交易")
                break

            # 非交易日跳过
            if not result.get("trades"):
                continue

            self.log.info(f"实盘第{day_count}天 | {date} | 净值: {result['portfolio']['value']:,.0f}")

        return {
            "results": results,
            "n_days": day_count,
            "final_value": results[-1]["portfolio"]["value"] if results else 0,
        }

    # ========== 内部方法 ==========

    def _get_market_state(self, date: str) -> dict:
        """获取市场状态"""
        return {
            "date": date,
            "volatility": 0.15,  # 默认值
            "momentum": 0.0,
            "risk_score": 0.0,
            "daily_drop": 0.0,
        }

    def _compute_daily_returns(self, date: str) -> dict:
        """计算当日各股票收益率"""
        return {}

    def _execute_trades(self, positions: dict, date: str) -> list:
        """执行交易"""
        trades = []
        commission_rate = self.backtest_config.get("commission", 0.0003)
        slippage_rate = self.backtest_config.get("slippage", 0.001)

        # 卖出不在新仓位中的股票
        old_codes = set(self.portfolio["positions"].keys())
        new_codes = set(positions.keys())

        for code in old_codes - new_codes:
            weight = self.portfolio["positions"].get(code, 0)
            trade_value = self.portfolio["value"] * weight
            trade = {
                "date": date,
                "code": code,
                "action": "sell",
                "weight": weight,
                "value": trade_value,
                "commission": trade_value * commission_rate,
                "slippage": trade_value * slippage_rate,
            }
            trades.append(trade)
            self.portfolio["cash"] += trade_value * (1 - commission_rate - slippage_rate)
            del self.portfolio["positions"][code]

        # 买入新仓位
        for code, weight in positions.items():
            if code in new_codes and code not in self.portfolio["positions"]:
                trade_value = self.portfolio["value"] * weight
                if trade_value > 0 and self.portfolio["cash"] > trade_value * (1 + commission_rate):
                    trade = {
                        "date": date,
                        "code": code,
                        "action": "buy",
                        "weight": weight,
                        "value": trade_value,
                        "commission": trade_value * commission_rate,
                        "slippage": trade_value * slippage_rate,
                    }
                    trades.append(trade)
                    self.portfolio["cash"] -= trade_value * (1 + commission_rate + slippage_rate)
                    self.portfolio["positions"][code] = weight
                    self.portfolio["entry_date"][code] = date

        self.trade_history.extend(trades)
        return trades

    def _emergency_close(self, date: str) -> list:
        """紧急平仓"""
        trades = []
        for code in list(self.portfolio["positions"].keys()):
            weight = self.portfolio["positions"][code]
            trade_value = self.portfolio["value"] * weight
            trade = {
                "date": date,
                "code": code,
                "action": "emergency_sell",
                "weight": weight,
                "value": trade_value,
                "reason": "风控熔断",
            }
            trades.append(trade)
            self.portfolio["cash"] += trade_value * 0.999  # 简化滑点
            del self.portfolio["positions"][code]

        self.portfolio["value"] = self.portfolio["cash"]
        self.trade_history.extend(trades)
        return trades

    def _update_portfolio(self, date: str) -> None:
        """更新组合状态"""
        # 简化: 组合净值 = 现金 + 持仓市值
        # 实际应连接行情数据计算持仓市值
        self.portfolio["value"] = self.portfolio["cash"]

    def _calculate_performance(self, date: str) -> dict:
        """计算绩效指标"""
        if not self.performance_history:
            return {"date": date, "value": self.portfolio["value"], "return": 0}

        initial_capital = self.backtest_config.get("initial_capital", 1000000)
        values = [p["value"] for p in self.performance_history]
        returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]

        return {
            "date": date,
            "value": self.portfolio["value"],
            "cash": self.portfolio["cash"],
            "n_positions": len(self.portfolio["positions"]),
            "total_return": (self.portfolio["value"] / initial_capital - 1) if initial_capital > 0 else 0,
            "daily_return": returns[-1] if returns else 0,
        }

    def _calculate_backtest_metrics(self, results: list) -> dict:
        """计算回测指标"""
        values = [r["portfolio"]["value"] for r in results]
        daily_returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]

        if not daily_returns:
            return {"status": "no_data"}

        initial_capital = self.backtest_config.get("initial_capital", 1000000)

        metrics = {
            "total_return": (values[-1] / initial_capital - 1) if initial_capital > 0 else 0,
            "annualized_return": annualized_return(pd.Series(daily_returns)),
            "sharpe_ratio": sharpe_ratio(pd.Series(daily_returns)),
            "max_drawdown": max_drawdown(pd.Series(daily_returns)),
            "win_rate": win_rate(pd.Series(daily_returns)),
            "profit_loss_ratio": profit_loss_ratio(pd.Series(daily_returns)),
            "calmar_ratio": calmar_ratio(pd.Series(daily_returns)),
            "sortino_ratio": sortino_ratio(pd.Series(daily_returns)),
            "n_trading_days": len(daily_returns),
            "n_trades": len(self.trade_history),
            "final_value": values[-1],
            "values": values,
        }

        return metrics

    def _generate_trading_days(self, start: str, end: str) -> list:
        """Read actual exchange sessions; fail rather than fabricate weekdays."""
        dates = self.screening.data_fetcher.fetch_trading_calendar(start, end)
        return [pd.Timestamp(day).strftime("%Y-%m-%d") for day in dates]

    def _save_backtest_results(self, results: list, metrics: dict) -> None:
        """保存回测结果"""
        output_dir = Path("../output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        save_data = {
            "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items() if k != "values"},
            "trades": self.trade_history,
            "daily_results": [
                {k: v for k, v in r.items() if k not in ("trades", "performance")}
                for r in results
            ],
        }

        with open(output_dir / "backtest_results.json", "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

        # 保存绩效曲线
        values = [r["portfolio"]["value"] for r in results]
        dates = [r["date"] for r in results]
        df = pd.DataFrame({"date": dates, "value": values})
        df.to_csv(output_dir / "performance_curve.csv", index=False)

        self.log.info(f"回测结果已保存到: {output_dir}")

    def get_summary(self) -> dict:
        """获取系统摘要"""
        return {
            "system": self.config.get("system", {}).get("name", "AI Quant System"),
            "version": self.config.get("system", {}).get("version", "1.0.0"),
            "layers": {
                "screening": "LightGBM因子选股",
                "research": "TradingAgents研报分析",
                "execution": "PPO仓位管理",
                "risk": "DoubleEnsemble风控",
            },
            "performance": self.performance_history[-1] if self.performance_history else {},
            "total_trades": len(self.trade_history),
        }
