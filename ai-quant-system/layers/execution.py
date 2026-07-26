"""
AI量化交易系统 - 执行层 (Execution Layer)
PPO强化学习做动态仓位微调
强化学习负责"调仓位"
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.utils import seeding
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gymnasium未安装,环境将退回为非标准接口")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    logger.warning("stable-baselines3未安装,使用规则仓位管理")

from utils.data_helper import (
    normalize, z_score, annualized_return, sharpe_ratio, max_drawdown
)


class PortfolioEnv(gym.Env if GYM_AVAILABLE else object):
    """
    强化学习环境 - 投资组合管理
    状态空间: 市场状态 + 持仓状态
    动作空间: 连续仓位调整 (-1 ~ 1)
    奖励函数: 风险调整后收益
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict = None, stock_pool: List[dict] = None, market_data: pd.DataFrame = None):
        self.config = config or {}
        self.state_space_config = self.config.get("state_space", {})
        self.position_config = self.config.get("position", {})

        self.stock_pool = stock_pool or []
        self.market_data = market_data.copy() if isinstance(market_data, pd.DataFrame) else pd.DataFrame()
        self.market_returns = self._build_market_returns(self.market_data)

        self.current_step = 0
        self.portfolio_value = 0.0
        self.position_weights = {}
        self.entry_prices = {}
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.history = deque(maxlen=1000)
        self.state_history = []

        self.initial_capital = self.position_config.get("initial_capital", 1000000)
        self.commission_rate = self.position_config.get("commission_rate", 0.0003)
        self.slippage_rate = self.position_config.get("slippage_rate", 0.001)

        self.max_steps = min(
            self.position_config.get("max_steps", 252 * 5),
            len(self.market_returns) if len(self.market_returns) > 0 else 252 * 5,
        )

        self.np_random = np.random.default_rng()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._state_dim(),),
            dtype=np.float32,
        ) if GYM_AVAILABLE else None

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        ) if GYM_AVAILABLE else None

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.portfolio_value = float(self.initial_capital)
        self.position_weights = {}
        self.entry_prices = {}
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.history.clear()
        self.state_history.clear()

        if self.stock_pool:
            n_positions = min(self.position_config.get("max_positions", 20), len(self.stock_pool))
            equal_weight = 1.0 / n_positions if n_positions > 0 else 0.0
            for i, stock in enumerate(self.stock_pool[:n_positions]):
                code = stock.get("code", f"stock_{i}")
                self.position_weights[code] = equal_weight
                self.entry_prices[code] = float(stock.get("price", 100.0))

        state = self._get_state()
        return state, {}

    def step(self, action: np.ndarray):
        self.current_step += 1

        portfolio_adjustment = self._apply_action(action)
        market_return = self.market_returns[self.current_step] if 0 <= self.current_step < len(self.market_returns) else 0.0
        self.portfolio_value *= max(0.0, 1.0 + market_return + portfolio_adjustment)

        reward = self._calculate_reward(market_return, portfolio_adjustment)

        self.history.append({
            "step": self.current_step,
            "value": self.portfolio_value,
            "reward": reward,
            "market_return": market_return,
            "adjustment": portfolio_adjustment,
        })

        terminated = self._check_terminated()
        truncated = self._check_truncated()

        info = {
            "portfolio_value": self.portfolio_value,
            "n_positions": len(self.position_weights),
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }

        observation = self._get_state() if not (terminated or truncated) else np.zeros(self._state_dim(), dtype=np.float32)
        return observation, float(reward), terminated, truncated, info

    def render(self, mode: str = "human"):
        if mode == "human":
            print(f"Step={self.current_step} Value={self.portfolio_value:.2f} Positions={len(self.position_weights)}")

    def close(self):
        pass

    def _build_market_returns(self, market_data: pd.DataFrame) -> List[float]:
        if market_data is None or market_data.empty:
            return []
        if "close" not in market_data.columns:
            return []
        returns = market_data["close"].pct_change().fillna(0.0).tolist()
        return [float(r) for r in returns]

    def _get_state(self) -> np.ndarray:
        features = self.state_space_config.get("features", [])
        state = []

        if "portfolio_value" in features:
            state.append(np.log(max(self.portfolio_value, 1e-6)))

        if "position_pnl" in features:
            current_pnl = self._calculate_portfolio_pnl()
            state.append(current_pnl / max(self.portfolio_value, 1e-6))

        if "market_regime" in features:
            if len(self.history) >= 20:
                returns = [h["market_return"] for h in list(self.history)[-20:]]
                regime = np.std(returns) if returns else 0.0
                state.append(regime)
            else:
                state.append(0.0)

        if "volatility" in features:
            if len(self.history) >= 20:
                returns = [h["market_return"] for h in list(self.history)[-20:]]
                vol = np.std(returns) if len(returns) > 1 else 0.0
                state.append(vol)
            else:
                state.append(0.0)

        if "momentum" in features:
            if len(self.history) >= 10:
                recent = [h["market_return"] for h in list(self.history)[-10:]]
                state.append(np.sum(recent))
            else:
                state.append(0.0)

        if "volume_profile" in features:
            state.append(len(self.position_weights) / max(self.position_config.get("max_positions", 20), 1))

        state.append(len(self.position_weights) / max(self.position_config.get("max_positions", 20), 1))

        if self.position_weights:
            pnls = []
            current_factor = 1.0 + (self.market_returns[self.current_step] if 0 <= self.current_step < len(self.market_returns) else 0.0)
            for code, weight in self.position_weights.items():
                entry = self.entry_prices.get(code, 100.0)
                current = entry * current_factor
                pnls.append((current - entry) / entry)
            state.append(np.mean(pnls))
            state.append(np.std(pnls) if len(pnls) > 1 else 0.0)

        target_dim = self._state_dim()
        while len(state) < target_dim:
            state.append(0.0)
        state = state[:target_dim]
        return np.array(state, dtype=np.float32)

    def _calculate_portfolio_pnl(self) -> float:
        if not self.position_weights:
            return 0.0
        current_factor = 1.0 + (self.market_returns[self.current_step] if 0 <= self.current_step < len(self.market_returns) else 0.0)
        total = 0.0
        for code, weight in self.position_weights.items():
            entry = self.entry_prices.get(code, 100.0)
            current = entry * current_factor
            total += weight * (current - entry)
        return total

    def _state_dim(self) -> int:
        return max(16, self.position_config.get("state_dim", 16))

    def _apply_action(self, action: np.ndarray) -> float:
        adjustment = np.clip(action[0] if len(action) > 0 else 0.0, -0.1, 0.1)
        return float(adjustment)

    def _calculate_reward(self, market_return: float, adjustment: float) -> float:
        if len(self.history) < 2:
            return 0.0

        values = [h["value"] for h in self.history]
        if len(values) < 2:
            return 0.0

        total_return = (values[-1] / values[0]) - 1.0
        cummax = max(values)
        drawdown = (values[-1] - cummax) / max(cummax, 1e-6)
        cost_penalty = abs(adjustment) * 10.0
        reward = total_return * 100.0 + drawdown * 5.0 - cost_penalty
        return float(reward)

    def _check_terminated(self) -> bool:
        if len(self.history) >= 10:
            values = [h["value"] for h in self.history]
            peak = max(values)
            drawdown = (values[-1] - peak) / max(peak, 1e-6)
            if drawdown < -0.20:
                return True

        if self.portfolio_value < self.initial_capital * 0.01:
            return True

        return False

    def _check_truncated(self) -> bool:
        return self.current_step >= self.max_steps


class ExecutionLayer:
    """
    执行层: PPO强化学习动态仓位管理
    负责根据市场状态动态调整仓位
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ppo_config = self.config.get("ppo", {})
        self.position_config = self.config.get("position", {})

        self.env_config = self.config
        self.ppo_model = None
        self.is_trained = False

        # 自动尝试加载已保存模型
        if self.config.get("auto_load_model", True):
            self.load_model()

        # 仓位管理参数
        self.max_positions = self.position_config.get("max_positions", 20)
        self.min_position = self.position_config.get("min_position", 0.02)
        self.max_position = self.position_config.get("max_position", 0.10)
        self.equal_weight = self.position_config.get("equal_weight", 0.10)

        # 训练/回测历史
        self.training_history = []
        self.trading_log = []

    def train(
        self,
        stock_pool: List[dict],
        market_data: pd.DataFrame,
        n_episodes: int = 100,
        verbose: int = 1
    ) -> dict:
        """
        训练PPO模型

        Args:
            stock_pool: 候选股票池
            market_data: 市场数据
            n_episodes: 训练轮数
            verbose: 日志级别

        Returns:
            训练结果
        """
        if not HAS_SB3:
            logger.warning("stable-baselines3未安装,跳过PPO训练")
            self.is_trained = False
            return {"status": "skipped", "reason": "依赖未安装"}

        logger.info("=" * 60)
        logger.info(f"执行层开始训练PPO模型 | 训练轮数: {n_episodes}")
        logger.info("=" * 60)

        if market_data is None or market_data.empty:
            logger.error("market_data为空,无法训练PPO模型")
            return {"status": "failed", "error": "market_data missing", "is_trained": False}

        try:
            env_kwargs = {
                "config": self.env_config,
                "stock_pool": stock_pool,
                "market_data": market_data,
            }
            env_factory = lambda: PortfolioEnv(**env_kwargs)
            n_envs = self.ppo_config.get("n_envs", 4)
            vec_env = make_vec_env(env_factory, n_envs=n_envs, seed=42)

            self.ppo_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=self.ppo_config.get("learning_rate", 3e-4),
                n_steps=self.ppo_config.get("n_steps", 2048),
                batch_size=self.ppo_config.get("batch_size", 64),
                n_epochs=self.ppo_config.get("n_epochs", 10),
                gamma=self.ppo_config.get("gamma", 0.99),
                gae_lambda=self.ppo_config.get("gae_lambda", 0.95),
                clip_range=self.ppo_config.get("clip_range", 0.2),
                ent_coef=self.ppo_config.get("ent_coef", 0.01),
                vf_coef=self.ppo_config.get("vf_coef", 0.5),
                max_grad_norm=self.ppo_config.get("max_grad_norm", 0.5),
                verbose=verbose,
            )

            total_steps = n_episodes * self.ppo_config.get("n_steps", 2048)
            self.ppo_model.learn(
                total_timesteps=total_steps,
                progress_bar=True,
            )

            self.is_trained = True
            logger.info("PPO模型训练完成")

            result = {
                "status": "success",
                "n_episodes": n_episodes,
                "total_steps": total_steps,
                "is_trained": True,
            }

            self.save_model()
            return result

        except Exception as e:
            logger.error(f"PPO训练失败: {e}")
            self.is_trained = False
            return {
                "status": "failed",
                "error": str(e),
                "is_trained": False,
            }

    def optimize_positions(
        self,
        verified_stocks: List[dict],
        market_state: dict = None,
        date: str = None
    ) -> Dict:
        """
        优化仓位权重

        Args:
            verified_stocks: 精研层通过的股票
            market_state: 当前市场状态
            date: 优化日期

        Returns:
            {
                "date": "2026-07-21",
                "positions": {stock_code: weight},
                "total_weight": 1.0,
                "method": "ppo" / "equal_weight" / "custom",
                "metrics": {...}
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"执行层: 优化仓位 | 候选股票: {len(verified_stocks)}只")

        if not verified_stocks:
            return {
                "date": date,
                "positions": {},
                "total_weight": 0,
                "method": "empty",
                "metrics": {},
            }

        # 方法选择
        if self.is_trained and self.ppo_model is not None:
            positions = self._ppo_optimize(verified_stocks, market_state)
            method = "ppo"
        else:
            positions = self._equal_weight_optimize(verified_stocks)
            method = "equal_weight"
            logger.info("使用等权仓位分配")

        # 归一化权重
        total_weight = sum(positions.values())
        if total_weight > 0:
            positions = {k: v / total_weight for k, v in positions.items()}

        # 应用仓位约束
        positions = self._apply_position_limits(positions)

        # 计算指标
        metrics = {
            "n_positions": len(positions),
            "max_weight": max(positions.values()) if positions else 0,
            "min_weight": min(positions.values()) if positions else 0,
            "entropy": self._portfolio_entropy(positions),
            "herfindahl": self._herfindahl(positions),
        }

        result = {
            "date": date,
            "positions": {k: round(v, 6) for k, v in positions.items()},
            "total_weight": round(sum(positions.values()), 6),
            "method": method,
            "metrics": metrics,
        }

        self.trading_log.append({
            "date": date,
            "method": method,
            "n_positions": len(positions),
            "total_weight": sum(positions.values()),
        })

        logger.info(f"仓位优化完成 | {len(positions)}只 | 总权重: {sum(positions.values()):.2%} | 熵: {metrics['entropy']:.4f}")

        return result

    def _ppo_optimize(
        self,
        verified_stocks: List[dict],
        market_state: dict = None
    ) -> Dict[str, float]:
        """使用PPO模型优化仓位"""
        positions = {}

        if not verified_stocks or self.ppo_model is None:
            return positions

        state = self._build_ppo_state(verified_stocks, market_state)
        action, _ = self.ppo_model.predict(state, deterministic=True)

        base_weight = 1.0 / len(verified_stocks)
        adjustment = float(np.clip(action[0] if hasattr(action, '__getitem__') else action, -0.1, 0.1))

        for i, stock in enumerate(verified_stocks):
            code = stock.get("code", f"stock_{i}")
            score = stock.get("final_score", stock.get("score", 0.5))
            weight = base_weight * (1 + score * 0.5 + adjustment)
            weight = max(self.min_position, min(self.max_position, weight))
            positions[code] = weight

        return positions

    def _build_ppo_state(
        self,
        verified_stocks: List[dict],
        market_state: dict = None
    ) -> np.ndarray:
        """构建与训练环境一致的PPO输入状态向量"""
        state_dim = self.position_config.get("state_dim", 16)
        state_values = [
            len(verified_stocks) / max(self.max_positions, 1),
            float(np.mean([s.get("score", 0.0) for s in verified_stocks])) if verified_stocks else 0.0,
            float(np.std([s.get("score", 0.0) for s in verified_stocks])) if len(verified_stocks) > 1 else 0.0,
            float(market_state.get("volatility", 0.0) if market_state else 0.0),
            float(market_state.get("momentum", 0.0) if market_state else 0.0),
        ]

        while len(state_values) < state_dim:
            state_values.append(0.0)
        state_values = state_values[:state_dim]
        return np.array(state_values, dtype=np.float32).reshape(1, -1)

    def _equal_weight_optimize(
        self,
        verified_stocks: List[dict]
    ) -> Dict[str, float]:
        """等权仓位分配"""
        n = min(len(verified_stocks), self.max_positions)
        weight = 1.0 / n if n > 0 else 0

        positions = {}
        for i, stock in enumerate(verified_stocks[:n]):
            code = stock.get("code", f"stock_{i}")
            positions[code] = weight

        return positions

    def _apply_position_limits(self, positions: Dict[str, float]) -> Dict[str, float]:
        """应用仓位限制"""
        result = {}
        for code, weight in positions.items():
            # 确保在[min_position, max_position]范围内
            weight = max(self.min_position, min(self.max_position, weight))
            result[code] = weight

        # 重新归一化
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def _portfolio_entropy(self, weights: Dict[str, float]) -> float:
        """计算组合熵 (衡量分散度)"""
        if not weights:
            return 0
        entropy = -sum(w * np.log(max(w, 1e-10)) for w in weights.values())
        # 归一化
        max_entropy = np.log(max(len(weights), 1))
        return entropy / max_entropy if max_entropy > 0 else 0

    def _herfindahl(self, weights: Dict[str, float]) -> float:
        """计算Herfindahl指数 (衡量集中度)"""
        if not weights:
            return 0
        return sum(w ** 2 for w in weights.values())

    def save_model(self) -> bool:
        """保存PPO模型"""
        if not self.is_trained or self.ppo_model is None:
            return False

        try:
            model_path = Path("../models/ppo_model")
            model_path.mkdir(parents=True, exist_ok=True)
            model_file = model_path / "ppo.zip"
            self.ppo_model.save(str(model_file))

            meta = {
                "is_trained": True,
                "train_date": datetime.now().strftime("%Y-%m-%d"),
                "ppo_config": self.ppo_config,
                "position_config": self.position_config,
                "env_config": self.env_config,
                "state_dim": self.position_config.get("state_dim", 16),
            }
            with open(model_path / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info("PPO模型已保存")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False

    def load_model(self) -> bool:
        """加载PPO模型"""
        if not HAS_SB3:
            return False

        try:
            model_path = Path("../models/ppo_model")
            model_file = model_path / "ppo.zip"

            if not model_file.exists():
                logger.warning("PPO模型文件不存在")
                return False

            self.ppo_model = PPO.load(str(model_file))
            self.is_trained = True

            meta_path = model_path / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.ppo_config.update(meta.get("ppo_config", {}))
                self.position_config.update(meta.get("position_config", {}))
                self.env_config.update(meta.get("env_config", {}))

            logger.info("PPO模型加载成功")
            return True
        except Exception as e:
            logger.error(f"加载PPO模型失败: {e}")
            self.is_trained = False
            return False

    def get_trading_log(self) -> List[dict]:
        """获取交易日志"""
        return self.trading_log
