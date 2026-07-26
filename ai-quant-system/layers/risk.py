"""
AI量化交易系统 - 风控层 (Risk Layer)
DoubleEnsemble风格的低过拟合设计做安全垫
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from utils.data_helper import (
    annualized_return, sharpe_ratio, max_drawdown, calmar_ratio, sortino_ratio, win_rate, profit_loss_ratio
)


class DoubleEnsembleModel:
    """
    DoubleEnsemble模型 - 双重集成学习
    核心思想: 多模型投票 + 动态权重,降低单模型过拟合风险
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.n_models = self.config.get("n_models", 5)
        self.base_estimator = self.config.get("base_estimator", "lightgbm")
        self.max_depth = self.config.get("max_depth", 4)
        self.min_samples_leaf = self.config.get("min_samples_leaf", 200)
        self.bootstrap = self.config.get("bootstrap", True)

        self.models = []
        self.model_weights = []
        self.oob_scores = []
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        训练DoubleEnsemble模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)

        Returns:
            训练结果
        """
        n_samples, n_features = X.shape
        self.models = []
        self.model_weights = []
        self.oob_scores = []

        logger.info(f"训练DoubleEnsemble | 样本数: {n_samples} | 特征数: {n_features} | 模型数: {self.n_models}")

        for i in range(self.n_models):
            # 每次用不同的随机子集
            if self.bootstrap and n_samples > 100:
                n_train = int(n_samples * 0.8)
                indices = np.random.choice(n_samples, n_train, replace=True)
                X_train, y_train = X[indices], y[indices]
                X_val = np.delete(X, indices, axis=0)
            else:
                X_train, y_train = X, y
                X_val = X[:min(100, n_samples)]

            # 创建基学习器
            if self.base_estimator == "lightgbm":
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=self.max_depth,
                    min_child_samples=max(10, self.min_samples_leaf // 10),
                    learning_rate=0.05,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    verbose=-1,
                    random_state=42 + i,
                    n_jobs=-1,
                )
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=self.max_depth,
                    verbose=-1,
                    random_state=42 + i,
                    n_jobs=-1,
                )

            model.fit(X_train, y_train)
            self.models.append(model)

            # OOB分数
            if X_val.shape[0] > 0:
                y_pred = model.predict_proba(X_val)[:, 1]
                oob_score = float(np.mean(y_val > 0.5 if hasattr(y_val, '__iter__') else 0))
                self.oob_scores.append(oob_score)

            # 初始化权重
            self.model_weights.append(1.0 / self.n_models)

            logger.debug(f"模型{i+1}训练完成 | OOB: {self.oob_scores[-1]:.4f}")

        self.is_trained = True
        return {
            "n_models": len(self.models),
            "oob_scores": [round(s, 4) for s in self.oob_scores],
            "avg_oob": round(np.mean(self.oob_scores), 4) if self.oob_scores else 0,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率 (加权平均)"""
        if not self.is_trained or not self.models:
            return np.ones(X.shape[0]) * 0.5

        # 各模型预测
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_models, n_samples)

        # 加权平均
        weights = np.array(self.model_weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            avg_pred = np.average(predictions, axis=0, weights=weights)
        else:
            avg_pred = predictions.mean(axis=0)

        return avg_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_consensus(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测共识
        返回: (预测结果, 共识度)
        """
        if not self.is_trained or not self.models:
            return np.ones(X.shape[0]), 0.0

        # 各模型投票
        votes = []
        for model in self.models:
            pred = model.predict(X)
            votes.append(pred)

        votes = np.array(votes)  # (n_models, n_samples)

        # 多数投票
        consensus = (votes.mean(axis=0) >= 0.5).astype(int)

        # 共识度:  agreeing models的比例
        agreement = np.abs(votes.mean(axis=0) - 0.5) * 2  # 归一化到[0, 1]

        return consensus, agreement

    def update_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """基于验证集更新模型权重"""
        if X_val.shape[0] == 0:
            return

        new_weights = []
        for model in self.models:
            proba = model.predict_proba(X_val)[:, 1]
            # 简单用AUC作为权重
            y_binary = (y_val > 0.5).astype(int) if hasattr(y_val, '__iter__') else np.array([y_val > 0.5])
            # 简化: 用准确率
            pred = (proba > 0.5).astype(int)
            accuracy = np.mean(pred == y_binary) if hasattr(y_binary, '__iter__') else 0.5
            new_weights.append(max(accuracy, 0.1))

        # 归一化
        total = sum(new_weights)
        self.model_weights = [w / total for w in new_weights]


class RiskModel:
    """
    风险预测模型
    预测未来N日是否会发生大幅回撤
    """

    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self.features = []
        self.labels = []

    def prepare_data(self, stock_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备风险预测数据"""
        close = stock_data["close"]
        volume = stock_data["volume"]

        # 特征: 波动率、动量、成交量变化
        features = []
        labels = []

        for i in range(len(close) - self.horizon):
            feat = [
                close.iloc[i:i+20].pct_change().std() if i+20 <= len(close) else 0,
                (close.iloc[i] / close.iloc[max(0, i-20)] - 1) if i >= 20 else 0,
                volume.iloc[i] / volume.iloc[max(0, i-20):i].mean() if i >= 20 else 1,
                (close.iloc[i] - close.iloc[max(0, i-5)]) / close.iloc[max(0, i-5)] if i >= 5 else 0,
            ]
            features.append(feat)

            # 标签: 未来horizon日最大回撤 > 3%
            future_close = close.iloc[i+1:i+self.horizon+1]
            peak = future_close.max()
            max_dd = (future_close.min() - peak) / peak if peak > 0 else 0
            label = 1 if max_dd < -0.03 else 0
            labels.append(label)

        return np.array(features), np.array(labels)


class RiskLayer:
    """
    风控层: DoubleEnsemble低过拟合设计
    多层风控: 个股风控 + 组合风控 + 市场风控
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ensemble_config = self.config.get("double_ensemble", {})
        self.metrics_config = self.config.get("metrics", {})
        self.position_risk_config = self.config.get("position_risk", {})
        self.market_risk_config = self.config.get("market_risk", {})

        # DoubleEnsemble模型
        self.ensemble_model = None
        self.risk_model = None

        # 风控指标
        self.risk_metrics = {
            "max_drawdown": self.metrics_config.get("max_drawdown", 0.15),
            "daily_loss_limit": self.metrics_config.get("daily_loss_limit", 0.03),
            "var_confidence": self.metrics_config.get("var_confidence", 0.95),
            "tracking_error": self.metrics_config.get("tracking_error", 0.10),
        }

        # 仓位风控阈值
        self.position_limits = {
            "max_single_position": self.position_risk_config.get("max_single_position", 0.10),
            "max_industry_position": self.position_risk_config.get("max_industry_position", 0.30),
            "max_correlation": self.position_risk_config.get("max_correlation", 0.7),
            "stop_loss": self.position_risk_config.get("stop_loss", -0.08),
            "trailing_stop": self.position_risk_config.get("trailing_stop", -0.05),
        }

        # 市场风控
        self.market_limits = {
            "circuit_breaker": self.market_risk_config.get("circuit_breaker", True),
            "volatility_target": self.market_risk_config.get("volatility_target", 0.15),
            "risk_off_threshold": self.market_risk_config.get("risk_off_threshold", 0.25),
        }

        # 风控历史
        self.risk_history = []
        self.stop_events = []

    def check_risk(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        daily_returns: Dict[str, float],
        market_data: dict = None
    ) -> Dict:
        """
        执行风控检查

        Args:
            positions: 当前仓位 {stock_code: weight}
            portfolio_value: 当前组合净值
            daily_returns: 当日各股票收益率 {stock_code: return}
            market_data: 市场数据

        Returns:
            {
                "risk_level": "normal" / "warning" / "danger" / "risk_off",
                "actions": [...],
                "metrics": {...},
                "triggers": [...]
            }
        """
        triggers = []
        actions = []
        metrics = {}

        # === 1. 个股风控 ===
        individual_risk = self._check_individual_risk(positions, daily_returns)
        triggers.extend(individual_risk["triggers"])
        actions.extend(individual_risk["actions"])

        # === 2. 组合风控 ===
        portfolio_metrics = self._calculate_portfolio_metrics(positions, daily_returns, portfolio_value)
        metrics.update(portfolio_metrics)

        portfolio_risk = self._check_portfolio_risk(portfolio_metrics, daily_returns)
        triggers.extend(portfolio_risk["triggers"])
        actions.extend(portfolio_risk["actions"])

        # === 3. 市场风控 ===
        if market_data:
            market_risk = self._check_market_risk(market_data)
            triggers.extend(market_risk["triggers"])
            actions.extend(market_risk["actions"])
            metrics["market_regime"] = market_risk.get("regime", "normal")

        # === 4. 风控分级 ===
        risk_level = self._determine_risk_level(triggers)

        # === 5. 生成风控动作 ===
        if risk_level == "risk_off":
            actions = ["全部平仓", "启动熔断"]
        elif risk_level == "danger":
            actions.extend(["减仓50%", "收紧止损"])
        elif risk_level == "warning":
            actions.extend(["监控预警", "准备减仓"])

        result = {
            "risk_level": risk_level,
            "triggers": triggers,
            "actions": actions,
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.risk_history.append(result)

        if triggers:
            logger.warning(f"风控触发 | 级别: {risk_level} | 触发: {triggers}")

        return result

    def _check_individual_risk(
        self,
        positions: Dict[str, float],
        daily_returns: Dict[str, float]
    ) -> Dict:
        """检查个股风险"""
        triggers = []
        actions = []
        stop_codes = []

        for code, weight in positions.items():
            ret = daily_returns.get(code, 0)

            # 止损检查
            if ret <= self.position_limits["stop_loss"]:
                triggers.append(f"{code} 触发止损 ({ret:.1%})")
                actions.append(f"卖出 {code}")
                stop_codes.append(code)

            # 移动止损检查
            if hasattr(self, '_peak_prices') and code in self._peak_prices:
                peak = self._peak_prices[code]
                current = peak * (1 + ret)
                if (current - peak) / peak <= self.position_limits["trailing_stop"]:
                    triggers.append(f"{code} 触发移动止损")
                    actions.append(f"减仓 {code}")

        if stop_codes:
            # 更新仓位
            for code in stop_codes:
                if code in positions:
                    del positions[code]

        return {"triggers": triggers, "actions": actions}

    def _calculate_portfolio_metrics(
        self,
        positions: Dict[str, float],
        daily_returns: Dict[str, float],
        portfolio_value: float
    ) -> Dict:
        """计算组合风险指标"""
        # 组合日收益
        weights = list(positions.values())
        returns = [daily_returns.get(code, 0) for code in positions.keys()]

        if weights and sum(weights) > 0:
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            portfolio_return = sum(w * r for w, r in zip(weights, returns))
        else:
            portfolio_return = 0

        # 最大回撤估算
        max_dd = self.risk_metrics.get("max_drawdown", 0.15)

        # VaR (参数法)
        var_95 = self._calculate_var(returns)

        # 波动率
        vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # 相关性检查
        correlation_risk = self._check_correlation_risk(positions)

        return {
            "portfolio_return": round(portfolio_return, 6),
            "annualized_vol": round(vol, 4),
            "var_95": round(var_95, 6),
            "max_drawdown": round(max_dd, 4),
            "n_positions": len(positions),
            "concentration": sum(w**2 for w in weights) if weights else 0,
            "correlation_risk": correlation_risk,
        }

    def _check_portfolio_risk(
        self,
        metrics: dict,
        daily_returns: dict
    ) -> Dict:
        """检查组合风险"""
        triggers = []
        actions = []

        # 单日亏损限额
        daily_loss = metrics.get("portfolio_return", 0)
        if daily_loss < -self.risk_metrics["daily_loss_limit"]:
            triggers.append(f"单日亏损超限: {daily_loss:.2%}")
            actions.append("触发日亏损熔断")

        # 波动率目标
        ann_vol = metrics.get("annualized_vol", 0)
        if ann_vol > self.market_limits["volatility_target"]:
            triggers.append(f"波动率过高: {ann_vol:.2%}")
            actions.append("降低仓位")

        # 集中度检查
        concentration = metrics.get("concentration", 0)
        if concentration > 0.15:  # HHI > 0.15 表示过度集中
            triggers.append(f"组合过度集中: HHI={concentration:.4f}")
            actions.append("分散持仓")

        # VaR检查
        var = metrics.get("var_95", 0)
        if abs(var) > self.risk_metrics["var_confidence"] * 0.05:
            triggers.append(f"VaR超限: {var:.2%}")
            actions.append("降低风险敞口")

        return {"triggers": triggers, "actions": actions}

    def _check_market_risk(self, market_data: dict) -> Dict:
        """检查市场风险"""
        triggers = []
        regime = "normal"

        # 市场波动率
        market_vol = market_data.get("volatility", 0)
        if market_vol > self.market_limits["volatility_target"]:
            regime = "high_volatility"
            triggers.append(f"市场波动率高: {market_vol:.2%}")

        # 市场动量
        market_momentum = market_data.get("momentum", 0)
        if market_momentum < -0.1:
            regime = "bearish"
            triggers.append(f"市场趋势向下: {market_momentum:.2%}")

        # 熔断机制
        if self.market_limits["circuit_breaker"]:
            market_drop = market_data.get("daily_drop", 0)
            if market_drop < -0.07:  # 市场跌7%
                regime = "circuit_breaker"
                triggers.append(f"市场熔断: {market_drop:.1%}")

        # 风险退出阈值
        if regime in ["high_volatility", "bearish"]:
            risk_score = market_data.get("risk_score", 0)
            if risk_score > self.market_limits["risk_off_threshold"]:
                regime = "risk_off"
                triggers.append("市场风险过高,建议退出")

        return {
            "triggers": triggers,
            "regime": regime,
        }

    def _determine_risk_level(self, triggers: list) -> str:
        """确定风险等级"""
        if not triggers:
            return "normal"

        # 熔断触发
        if any("熔断" in t for t in triggers):
            return "risk_off"

        # 多个危险触发
        danger_count = sum(1 for t in triggers if any(kw in t for kw in ["止损", "超限", "熔断"]))
        if danger_count >= 2:
            return "danger"

        # 单一危险触发
        if danger_count >= 1:
            return "danger"

        # 警告
        if len(triggers) >= 2:
            return "warning"

        return "warning"

    def _calculate_var(self, returns: list, confidence: float = 0.95) -> float:
        """计算VaR (历史模拟法)"""
        if not returns or len(returns) < 10:
            return 0.05  # 默认5%

        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return sorted_returns[max(0, index)]

    def _check_correlation_risk(self, positions: Dict[str, float]) -> str:
        """检查相关性风险"""
        max_corr = self.position_limits.get("max_correlation", 0.7)

        # 简化: 按代码前缀分组
        groups = defaultdict(list)
        for code in positions.keys():
            if code.startswith("6"):
                groups["SH"].append(code)
            elif code.startswith("0") or code.startswith("3"):
                groups["SZ"].append(code)
            else:
                groups["OTHER"].append(code)

        # 同板块高度集中
        for sector, codes in groups.items():
            if len(codes) > 5:
                return "high"

        return "normal"

    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> dict:
        """训练DoubleEnsemble模型"""
        self.ensemble_model = DoubleEnsembleModel(self.ensemble_config)
        result = self.ensemble_model.fit(X, y)
        result["status"] = "success"
        logger.info(f"DoubleEnsemble训练完成 | {result}")
        return result

    def get_risk_report(self) -> dict:
        """获取风控报告"""
        if not self.risk_history:
            return {"status": "no_data"}

        risk_levels = [r["risk_level"] for r in self.risk_history]
        total_events = len(self.risk_history)
        trigger_counts = defaultdict(int)

        for r in self.risk_history:
            for t in r.get("triggers", []):
                trigger_counts[t] += 1

        return {
            "total_events": total_events,
            "risk_level_distribution": {
                level: risk_levels.count(level)
                for level in set(risk_levels)
            },
            "top_triggers": dict(sorted(trigger_counts.items(), key=lambda x: -x[1])[:10]),
            "latest_risk_level": risk_levels[-1] if risk_levels else "normal",
        }

    def add_stop_event(self, event: dict) -> None:
        """记录风控事件"""
        self.stop_events.append({
            **event,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
