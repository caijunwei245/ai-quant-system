"""
AI量化交易系统 - 初筛层 (Screening Layer)
LightGBM + 全市场因子选股
树模型负责"挖Alpha"
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics import make_scorer
from loguru import logger

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM未安装,使用模拟模式")

from data.fetcher import DataFetcher
from data.processor import FactorProcessor
from utils.validator import validate_stock_data, filter_unqualified_stocks
from utils.data_helper import (
    normalize, z_score, annualized_return, sharpe_ratio, max_drawdown
)


class ScreeningLayer:
    """
    初筛层: 全市场因子选股
    使用LightGBM模型对全市场股票进行因子计算和预测,选出候选股票池
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.lgb_config = self.config.get("lightgbm", {})
        self.selection_config = self.config.get("selection", {})
        self.factors_config = self.config.get("factors", {})

        self.data_fetcher = DataFetcher(self.config.get("data", {}))
        self.factor_processor = FactorProcessor()

        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.model_dir = Path("../models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "screening_model.json"

        # 因子权重 (可学习)
        self.factor_weights = {}
        self._compute_default_weights()

    def _compute_default_weights(self):
        """计算默认因子权重"""
        weights = {}
        technical = self.factors_config.get("technical", [])
        fundamental = self.factors_config.get("fundamental", [])
        price_volume = self.factors_config.get("price_volume", [])

        for f in technical:
            name = f.get("name", "") if isinstance(f, dict) else f
            if name:
                weights[name] = 1.0
        for f in fundamental:
            name = f.get("name", "") if isinstance(f, dict) else f
            if name:
                weights[name] = 1.2  # 基本面因子权重略高
        for f in price_volume:
            name = f.get("name", "") if isinstance(f, dict) else f
            if name:
                weights[name] = 0.8

        self.factor_weights = weights

    def run_screening(
        self,
        date: str = None,
        universe: str = None,
        top_k: int = None,
        min_score: float = None
    ) -> Dict:
        """
        执行初筛

        Args:
            date:  screening日期, None表示最新
            universe: 股票池范围
            top_k: 选前K只
            min_score: 最低预测分

        Returns:
            {
                "date": "2026-07-21",
                "candidate_stocks": [...],
                "scores": {...},
                "model_metrics": {...},
                "factor_importance": {...}
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        top_k = top_k or self.selection_config.get("top_k", 50)
        min_score = min_score if min_score is not None else self.selection_config.get("min_score", 0.3)
        universe = universe or self.config.get("universe", "all_a_share")

        logger.info(f"=" * 60)
        logger.info(f"初筛层开始工作 | 日期: {date} | 股票池: {universe}")
        logger.info(f"=" * 60)

        # Step 1: 获取股票池
        stock_list = self._get_universe(universe)
        logger.info(f"股票池数量: {len(stock_list)}")

        # Step 2: 获取数据并计算因子
        all_features = {}
        all_labels = {}
        valid_stocks = []
        failed_stocks = []

        for i, stock in enumerate(stock_list):
            code = stock.get("code", "")
            try:
                df = self.data_fetcher.fetch_stock_data(
                    code,
                    start_date="2021-01-01",
                    end_date=date,
                    adjust="qfq"
                )

                if df.empty or len(df) < 60:
                    failed_stocks.append({"code": code, "reason": "数据不足"})
                    continue

                # 验证数据
                validation = validate_stock_data(df, code)
                if not validation["valid"]:
                    failed_stocks.append({"code": code, "reason": validation["issues"]})
                    continue

                # 计算因子
                factor_list = list(self.factor_weights.keys()) if self.factor_weights else None
                factors = self.factor_processor.compute_all_factors(df, factor_list)

                # 修复: 使用 NaN 比例而非绝对数量 (rolling/ewm 预热期会产生大量 NaN)
                nan_ratio = factors.isna().mean().mean() if factors.size > 0 else 1.0
                if factors.empty or nan_ratio > 0.8:  # 超过80%的因子值为NaN才判定失败
                    failed_stocks.append({"code": code, "reason": f"因子计算失败 (NaN比例:{nan_ratio:.1%})"})
                    continue

                # 生成标签 (未来5日收益率)
                label = self.factor_processor.generate_labels(df, forward_days=5)

                all_features[code] = factors
                all_labels[code] = label
                valid_stocks.append(code)

                if (i + 1) % 100 == 0:
                    logger.info(f"已处理: {i + 1}/{len(stock_list)}, 有效: {len(valid_stocks)}")

            except Exception as e:
                failed_stocks.append({"code": code, "reason": str(e)})
                continue

        logger.info(f"有效股票: {len(valid_stocks)}, 失败: {len(failed_stocks)}")

        if len(valid_stocks) < 10:
            logger.error("有效股票数量不足,无法进行筛选")
            return {"date": date, "candidate_stocks": [], "scores": {}, "model_metrics": {}, "valid_stocks": len(valid_stocks), "failed_stocks": len(failed_stocks), "failed_stocks_detail": failed_stocks}

        # Step 3: 训练LightGBM模型 (或用已有模型)
        if HAS_LIGHTGBM and not self.is_trained:
            self._train_model(valid_stocks, all_features, all_labels)
        elif self.is_trained:
            logger.info("使用已有模型进行预测")
        else:
            logger.warning("LightGBM不可用,使用评分法")

        # Step 4: 预测得分
        scores = self._predict_scores(valid_stocks, date)

        # Step 5: 排序选股
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidate_stocks = [
            {
                "code": code,
                "score": round(score, 4),
                "rank": i + 1,
            }
            for i, (code, score) in enumerate(sorted_scores[:top_k])
            if score >= min_score
        ]

        logger.info(f"初筛完成 | 选出 {len(candidate_stocks)} 只股票")
        if candidate_stocks:
            logger.info(f"Top5: {[s['code'] for s in candidate_stocks[:5]]}")

        # Step 6: 行业分散检查
        if self.selection_config.get("industry_diversify", False):
            candidate_stocks = self._apply_industry_diversify(candidate_stocks)

        result = {
            "date": date,
            "candidate_stocks": candidate_stocks,
            "scores": {k: round(v, 4) for k, v in sorted_scores},
            "model_metrics": self.model.get("metrics", {}) if self.model else {},
            "factor_importance": self.model.get("feature_importance", {}) if self.model else {},
            "valid_stocks": len(valid_stocks),
            "failed_stocks": len(failed_stocks),
        }

        return result

    def _get_universe(self, universe_type: str) -> List[dict]:
        """获取股票池"""
        market_map = {
            "all_a_share": "a_share",
            "hs300": "hs300",
            "zz500": "zz500",
            "gem": "gem",
        }
        market = market_map.get(universe_type, "a_share")
        return self.data_fetcher.fetch_stock_list(market)

    def build_training_dataset(
        self, valid_stocks, all_features, all_labels,
        lookback_days=0, min_samples=100, forward_days=5,
    ):
        """Build date-sorted cross-sectional relevance groups for LambdaRank."""
        from data.alpha_training import build_rank_dataset
        result = build_rank_dataset(
            valid_stocks, all_features, all_labels,
            min_group_size=int(self.lgb_config.get("min_group_size", 5)),
        )
        X, relevance, raw_returns, stock_codes, feature_names, sample_dates, groups = result
        if lookback_days > 0:
            dates = pd.DatetimeIndex(pd.to_datetime(sample_dates)).normalize()
            keep_dates = sorted(dates.unique())[-int(lookback_days):]
            mask = dates.isin(keep_dates)
            X, relevance, raw_returns = X[mask], relevance[mask], raw_returns[mask]
            stock_codes = [c for c, keep in zip(stock_codes, mask) if keep]
            sample_dates = [d for d, keep in zip(sample_dates, mask) if keep]
            dates = pd.DatetimeIndex(pd.to_datetime(sample_dates)).normalize()
            groups = [int((dates == d).sum()) for d in dates.unique()]
        if len(X) < min_samples:
            raise ValueError(f"training samples insufficient: {len(X)} < {min_samples}")
        return X, relevance, raw_returns, stock_codes, feature_names, sample_dates, groups

    def _purged_kfold_cv(
        self, X, y, sample_dates, raw_returns, trading_calendar,
        n_splits=5, embargo_days=2, forward_days=5,
    ):
        """Strict expanding-window LambdaRank CV on exchange-session ranks."""
        from data.alpha_training import purged_walk_forward_splits, rank_metrics
        report = {"folds": [], "mean_rank_ic": None, "mean_top_bottom_return": None,
                  "n_folds": 0}
        splits = list(purged_walk_forward_splits(
            sample_dates, trading_calendar, n_splits=n_splits,
            forward_days=forward_days, embargo_days=embargo_days,
            min_train_samples=int(self.lgb_config.get("min_cv_train_samples", 50)),
            min_validation_samples=int(self.lgb_config.get("min_cv_validation_samples", 10)),
        ))
        params = {
            "objective": "lambdarank", "metric": "ndcg",
            "ndcg_eval_at": [5, 10], "verbosity": -1,
            "learning_rate": self.lgb_config.get("learning_rate", 0.05),
            "n_estimators": max(50, int(self.lgb_config.get("n_estimators", 500) // 5)),
            "num_leaves": self.lgb_config.get("num_leaves", 31),
            "random_state": 42, "n_jobs": -1,
        }
        dates = pd.DatetimeIndex(pd.to_datetime(sample_dates)).normalize()
        for fold, train_idx, val_idx, val_start, val_end in splits:
            train_groups = [int((dates[train_idx] == d).sum()) for d in dates[train_idx].unique()]
            val_groups = [int((dates[val_idx] == d).sum()) for d in dates[val_idx].unique()]
            if not train_groups or not val_groups:
                continue
            model = lgb.LGBMRanker(**params)
            model.fit(X[train_idx], y[train_idx], group=train_groups)
            scores = model.predict(X[val_idx])
            metrics = rank_metrics(dates[val_idx], raw_returns[val_idx], scores)
            report["folds"].append({
                "fold": int(fold), "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)), "train_groups": train_groups,
                "validation_groups": val_groups, "val_start": val_start,
                "val_end": val_end, **metrics,
            })
        report["n_folds"] = len(report["folds"])
        if report["folds"]:
            ics = [x["rank_ic"] for x in report["folds"] if x["rank_ic"] is not None]
            spreads = [x["top_bottom_return"] for x in report["folds"] if x["top_bottom_return"] is not None]
            report["mean_rank_ic"] = float(np.mean(ics)) if ics else None
            report["mean_top_bottom_return"] = float(np.mean(spreads)) if spreads else None
        return report

    def _train_model(self, valid_stocks, all_features, all_labels) -> None:
        """Train LambdaRank with purged walk-forward CV and a final untouched OOS period."""
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM is required for Alpha Ranker training")
        from data.alpha_training import rank_metrics

        X, y, raw_returns, stock_codes, feature_names, sample_dates, groups = self.build_training_dataset(
            valid_stocks, all_features, all_labels,
            lookback_days=int(self.lgb_config.get("lookback_days", 0)),
            min_samples=int(self.lgb_config.get("min_samples", 100)),
        )
        dates = pd.DatetimeIndex(pd.to_datetime(sample_dates)).normalize()
        start_date, end_date = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
        calendar = pd.DatetimeIndex(pd.to_datetime(
            self.data_fetcher.fetch_trading_calendar(start_date, end_date)
        )).normalize()
        ranks = calendar.get_indexer(dates)
        if (ranks < 0).any():
            raise ValueError("training sample date missing from exchange trading calendar")

        unique_dates = dates.unique().sort_values()
        oos_fraction = float(self.lgb_config.get("oos_fraction", 0.2))
        if not 0.1 <= oos_fraction <= 0.4:
            raise ValueError("oos_fraction must be between 0.10 and 0.40")
        split_at = int(np.floor(len(unique_dates) * (1.0 - oos_fraction)))
        if split_at <= 0 or split_at >= len(unique_dates):
            raise ValueError("not enough distinct dates for independent OOS split")
        oos_start = unique_dates[split_at]
        oos_rank = int(calendar.get_loc(oos_start))
        forward_days = int(self.lgb_config.get("forward_days", 5))
        embargo_days = int(self.lgb_config.get("embargo_days", 2))
        dev_mask = (dates < oos_start) & ((ranks + forward_days) < (oos_rank - embargo_days))
        oos_mask = dates >= oos_start
        dev_idx, oos_idx = np.flatnonzero(dev_mask), np.flatnonzero(oos_mask)
        if len(dev_idx) < int(self.lgb_config.get("min_samples", 100)) or len(oos_idx) < 20:
            raise ValueError("development or independent OOS sample is too small")

        cv = self._purged_kfold_cv(
            X[dev_idx], y[dev_idx], [sample_dates[i] for i in dev_idx],
            raw_returns[dev_idx], calendar,
            n_splits=int(self.lgb_config.get("n_splits", 5)),
            embargo_days=embargo_days, forward_days=forward_days,
        )
        if cv["n_folds"] < max(2, int(self.lgb_config.get("n_splits", 5)) - 1):
            raise RuntimeError(f"insufficient valid purged walk-forward folds: {cv['n_folds']}")

        dates_dev = dates[dev_idx]
        group_sizes = [int((dates_dev == d).sum()) for d in dates_dev.unique()]
        params = {
            "objective": "lambdarank", "metric": "ndcg",
            "ndcg_eval_at": [5, 10], "verbosity": -1,
            "learning_rate": self.lgb_config.get("learning_rate", 0.05),
            "n_estimators": self.lgb_config.get("n_estimators", 500),
            "num_leaves": self.lgb_config.get("num_leaves", 31),
            "max_depth": self.lgb_config.get("max_depth", -1),
            "min_child_samples": self.lgb_config.get("min_child_samples", 30),
            "random_state": 42, "n_jobs": -1,
        }
        model = lgb.LGBMRanker(**params)
        model.fit(X[dev_idx], y[dev_idx], group=group_sizes)
        oos = rank_metrics(dates[oos_idx], raw_returns[oos_idx], model.predict(X[oos_idx]))
        min_oos_rank_ic = float(self.lgb_config.get("min_oos_rank_ic", 0.0))
        if oos["rank_ic"] is None or oos["rank_ic"] < min_oos_rank_ic:
            raise RuntimeError(
                f"independent OOS RankIC {oos['rank_ic']} is below required {min_oos_rank_ic}; refusing deployment"
            )

        self.feature_names = feature_names
        importance = dict(zip(feature_names, model.feature_importances_.tolist()))
        self.model = {
            "metrics": {"trained": True, "samples": int(len(dev_idx)), "cv": cv,
                        "oos": oos, "oos_start": str(oos_start.date()),
                        "oos_end": str(dates[oos_idx].max().date()),
                        "oos_samples": int(len(oos_idx))},
            "feature_importance": importance, "model_obj": model,
        }
        self.is_trained = True
        self.model_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "is_trained": True, "model_type": "LGBMRanker",
            "feature_names": feature_names, "feature_importance": importance,
            "params": params, "train_date": datetime.now().strftime("%Y-%m-%d"),
            **self.model["metrics"],
        }
        with open(self.model_dir / "screening_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(self.model_dir / "screening_cv_report.json", "w", encoding="utf-8") as f:
            json.dump({"cv": cv, "oos": oos}, f, ensure_ascii=False, indent=2)
        logger.info(f"LambdaRank trained; CV RankIC={cv.get('mean_rank_ic')}, OOS RankIC={oos.get('rank_ic')}")

    def _predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, float]:
        """
        对候选股票进行预测打分
        返回 {stock_code: score}
        """
        scores = {}

        if not HAS_LIGHTGBM or not self.is_trained:
            # 回退方案: 基于因子评分法
            return self._scoring_method(stock_codes, date)

        try:
            import lightgbm as lgb

            # 获取模型对象
            model_obj = self.model.get("model_obj", self.model) if isinstance(self.model, dict) else self.model

            # 构建当日特征 (使用 adjust="qfq" 匹配训练阶段缓存)
            X_day = []
            scored_codes = []
            for code in stock_codes:
                df = self.data_fetcher.fetch_stock_data(code, start_date="2021-01-01", end_date=date, adjust="qfq")
                if df.empty:
                    continue

                factor_list = list(self.factor_weights.keys()) if self.factor_weights else None
                factors = self.factor_processor.compute_all_factors(df, factor_list)

                if factors.empty:
                    continue

                last_row = factors.iloc[-1]
                feat_vals = last_row.dropna()

                # 对齐特征到统一列名（与训练阶段一致）
                if len(feat_vals) > 0 and self.feature_names is not None:
                    aligned = pd.Series(0.0, index=self.feature_names)
                    overlap = feat_vals.index.intersection(aligned.index)
                    aligned[overlap] = feat_vals[overlap]
                    X_day.append(aligned.values)
                    scored_codes.append(code)

            if len(X_day) == 0:
                return self._scoring_method(stock_codes, date)

            X_day = np.nan_to_num(np.array(X_day, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            # Ranker emits a continuous cross-sectional ranking score (not a probability).
            rank_scores = model_obj.predict(X_day)

            for code, score in zip(scored_codes, rank_scores):
                scores[code] = float(score)

        except Exception as e:
            logger.error(f"LightGBM预测失败: {e}, 使用评分法回退")
            scores = self._scoring_method(stock_codes, date)

        return scores

    def _scoring_method(self, stock_codes: List[str], date: str) -> Dict[str, float]:
        """
        回退评分法: 基于多因子加权评分
        """
        scores = {}

        for code in stock_codes:
            df = self.data_fetcher.fetch_stock_data(code, start_date="2021-01-01", end_date=date, adjust="qfq")
            if df.empty or len(df) < 20:
                continue

            # 计算综合评分
            score = self._compute_composite_score(df)
            if score is not None:
                scores[code] = score

        return scores

    def _compute_composite_score(self, df: pd.DataFrame) -> Optional[float]:
        """
        计算综合评分 (回退方案)
        基于动量、波动率、流动性、技术面等多维度
        """
        close = df["close"]
        volume = df["volume"]

        # 1. 动量评分 (近20日涨幅)
        mom_20 = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0

        # 2. 动量评分 (近5日涨幅,短期强势)
        mom_5 = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0

        # 3. 波动率 (越低越好,但不要太低)
        returns = close.pct_change().dropna()
        vol_20 = returns.tail(20).std() if len(returns) >= 20 else 0

        # 4. 成交量稳定性
        vol_ratio = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(20).mean() > 0 else 1

        # 5. 均线位置 (价格在MA20上方)
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
        price_above_ma20 = 1 if ma20 > 0 and close.iloc[-1] > ma20 else 0

        # 6. RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).tail(14).mean()
        loss = (-delta).where(delta < 0, 0.0).tail(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # 综合评分 (加权)
        score = (
            np.tanh(mom_20 * 3) * 0.25 +       # 20日动量
            np.tanh(mom_5 * 5) * 0.20 +          # 5日动量
            (1 - min(vol_20 * 10, 1)) * 0.15 +  # 低波动偏好
            np.tanh(vol_ratio) * 0.10 +           # 量比
            price_above_ma20 * 0.15 +             # 均线位置
            (1 - abs(rsi - 55) / 55) * 0.15      # RSI适中
        )

        return float(np.clip(score, -1, 1))

    def _apply_industry_diversify(self, candidate_stocks: List[dict]) -> List[dict]:
        """
        应用行业分散约束
        单行业最多占30%
        """
        max_industry_weight = self.selection_config.get("max_industry_weight", 0.3)

        # 简化: 按股票代码前缀分组行业
        industry_groups = {}
        for stock in candidate_stocks:
            code = stock["code"]
            # 简化行业分组
            if code.startswith("6"):
                industry = "SH"
            elif code.startswith("0"):
                industry = "SZ"
            elif code.startswith("3"):
                industry = "GEM"
            elif code.startswith("688"):
                industry = "STAR"
            else:
                industry = "OTHER"

            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(stock)

        # 每行业选top
        result = []
        for industry, stocks in industry_groups.items():
            n_select = max(1, int(len(candidate_stocks) * max_industry_weight))
            result.extend(stocks[:n_select])

        # 按score排序
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    def save_model(self) -> None:
        """保存模型"""
        if not HAS_LIGHTGBM or not self.is_trained:
            return

        model_obj = self.model.get("model_obj", self.model) if isinstance(self.model, dict) else self.model

        # LightGBM 4.x: LGBMClassifier exposes the booster via `.booster_`
        # and only the Booster has `save_model`. Fall back gracefully.
        booster = getattr(model_obj, "booster_", None) or (
            model_obj if hasattr(model_obj, "save_model") else None
        )
        if booster is not None and hasattr(booster, "save_model"):
            booster.save_model(self.model_dir / "screening_model.txt")
            logger.info("模型已保存到: screening_model.txt")
        else:
            logger.warning("无法保存模型: 未找到可用的 Booster 对象")

    def load_model(self) -> bool:
        """加载模型"""
        meta_path = self.model_dir / "screening_meta.json"
        if not meta_path.exists():
            logger.warning("模型文件不存在")
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.is_trained = meta.get("is_trained", False)
            self.feature_names = meta.get("feature_names", [])

            if HAS_LIGHTGBM:
                model_path = self.model_dir / "screening_model.txt"
                if model_path.exists():
                    self.model = lgb.Booster(model_file=str(model_path))
                    logger.info("模型加载成功")
                    return True

            return False
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def get_factor_importance(self) -> Dict[str, float]:
        """获取因子重要性"""
        if isinstance(self.model, dict):
            return self.model.get("feature_importance", {})
        return {}
