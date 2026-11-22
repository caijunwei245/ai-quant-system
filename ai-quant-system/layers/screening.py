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
from sklearn.model_selection import TimeSeriesSplit
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

    def _time_series_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
        """
        基于TimeSeriesSplit的简单时序交叉验证
        返回: 字典 {"mean_auc":..., "std_auc":...}
        (保留为回退方法, 主要使用_purged_kfold_cv)
        """
        from sklearn.metrics import roc_auc_score

        if X.size == 0 or len(y) < 20:
            return {"mean_auc": 0.0, "std_auc": 0.0, "n_splits": 0}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            try:
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                model = lgb.LGBMClassifier(**{
                    "objective": "binary",
                    "n_estimators": max(50, int(self.lgb_config.get("n_estimators", 500) // 5)),
                    "learning_rate": self.lgb_config.get("learning_rate", 0.05),
                    "random_state": 42,
                })
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_val)[:, 1]
                auc = float(roc_auc_score(y_val, probs)) if len(np.unique(y_val)) > 1 else 0.5
                aucs.append(auc)
            except Exception as e:
                logger.debug(f"CV fold failed: {e}")
                continue

        if not aucs:
            return {"mean_auc": 0.0, "std_auc": 0.0, "n_splits": 0}

        return {"mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)), "n_splits": len(aucs)}

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

    def _train_model(
        self,
        valid_stocks: List[str],
        all_features: Dict[str, pd.DataFrame],
        all_labels: Dict[str, pd.Series]
    ) -> None:
        """训练LightGBM模型"""
        if not HAS_LIGHTGBM:
            return

        logger.info("开始训练LightGBM模型...")

        # 构建训练数据 (lookback_days 可配置, 默认 30 个交易日窗口)
        lookback_days = int(self.lgb_config.get("lookback_days", 30))
        X, y, stock_codes, feature_names, sample_dates = self.build_training_dataset(valid_stocks, all_features, all_labels, lookback_days=lookback_days, min_samples=100)

        if X.size == 0 or y.size == 0:
            logger.warning("未能构建有效训练集,跳过训练")
            return

        self.feature_names = feature_names
        logger.info(f"训练数据: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"正样本比例: {y.mean():.2%}")

        # 验证: 使用 Purged K-Fold + Embargo
        embargo_days = int(self.lgb_config.get("embargo_days", 2))
        forward_days = int(self.lgb_config.get("forward_days", 5))
        n_splits = int(self.lgb_config.get("n_splits", 5))

        cv_result = self._purged_kfold_cv(X, y, sample_dates, n_splits=n_splits, embargo_days=embargo_days, forward_days=forward_days)
        logger.info(f"Purged K-Fold CV 结果: mean_auc={cv_result.get('mean_auc'):.4f}, std={cv_result.get('std_auc'):.4f}, folds={cv_result.get('n_folds')}")

        # 保存 CV 报告到文件（JSON + 简易 HTML）
        try:
            report_path = self.model_dir / "screening_cv_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(cv_result, f, ensure_ascii=False, indent=2)
            # 生成带图像的 HTML 报告 (将图像保存在 model_dir 下)
            html_path = self.model_dir / "screening_cv_report.html"
            try:
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write('<html><head><meta charset="utf-8"><title>Screening CV Report</title></head><body>')
                    f.write(f'<h2>Purged K-Fold CV Report</h2>')
                    f.write(f'<p>mean_auc: {cv_result.get("mean_auc"):.4f}, std: {cv_result.get("std_auc"):.4f}, folds: {cv_result.get("n_folds")}</p>')

                    # 总体ROC图 (如果存在)
                    overall_img = cv_result.get('overall_roc_image')
                    if overall_img:
                        f.write('<h3>Overall ROC</h3>')
                        f.write(f'<img src="{overall_img}" alt="overall roc" style="max-width:600px; height:auto;"><br>')

                    f.write('<h3>Per-fold details</h3>')
                    f.write('<table border="1" cellpadding="4"><tr><th>fold</th><th>n_train</th><th>n_val</th><th>train_pos</th><th>train_neg</th><th>val_pos</th><th>val_neg</th><th>auc</th><th>val_start</th><th>val_end</th><th>ROC</th></tr>')
                    for fold in cv_result.get('folds', []):
                        f.write('<tr>')
                        f.write(f"<td>{fold.get('fold')}</td>")
                        f.write(f"<td>{fold.get('n_train')}</td>")
                        f.write(f"<td>{fold.get('n_val')}</td>")
                        f.write(f"<td>{fold.get('train_pos')}</td>")
                        f.write(f"<td>{fold.get('train_neg')}</td>")
                        f.write(f"<td>{fold.get('val_pos')}</td>")
                        f.write(f"<td>{fold.get('val_neg')}</td>")
                        f.write(f"<td>{fold.get('auc'):.4f}</td>")
                        f.write(f"<td>{fold.get('val_start')}</td>")
                        f.write(f"<td>{fold.get('val_end')}</td>")
                        roc_img = fold.get('roc_image')
                        if roc_img:
                            f.write(f"<td><img src=\"{roc_img}\" alt=\"roc_fold_{fold.get('fold')}\" style=\"max-width:200px; height:auto;\"></td>")
                        else:
                            f.write('<td>n/a</td>')
                        f.write('</tr>')
                    f.write('</table>')
                    f.write(f'<p>Report generated at: {cv_result.get("generated_at")}</p>')
                    f.write('</body></html>')
                logger.info(f"CV报告已保存到: {report_path} 和 {html_path}")
            except Exception as e:
                logger.error(f"保存HTML CV报告失败: {e}")
        except Exception as e:
            logger.error(f"保存CV报告失败: {e}")

        min_auc = float(self.lgb_config.get("min_cv_auc", 0.60))
        if cv_result.get("mean_auc", 0) < min_auc:
            logger.warning(f"Purged CV mean AUC {cv_result.get('mean_auc'):.4f} 小于阈值 {min_auc}, 不建议训练/部署模型")
            # 仍可选择继续训练，但为了稳健性这里直接返回不训练
            return

        # LightGBM参数
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": self.lgb_config.get("learning_rate", 0.05),
            "n_estimators": self.lgb_config.get("n_estimators", 500),
            "max_depth": self.lgb_config.get("max_depth", 6),
            "num_leaves": self.lgb_config.get("num_leaves", 63),
            "min_child_samples": self.lgb_config.get("min_child_samples", 100),
            "feature_fraction": self.lgb_config.get("feature_fraction", 0.8),
            "bagging_fraction": self.lgb_config.get("bagging_fraction", 0.8),
            "bagging_freq": self.lgb_config.get("bagging_freq", 5),
            "reg_alpha": self.lgb_config.get("reg_alpha", 0.1),
            "reg_lambda": self.lgb_config.get("reg_lambda", 0.1),
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

        # 训练最终模型
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            callbacks=[lgb.log_evaluation(period=0)],  # 静默训练
        )

        self.is_trained = True

        # 保存特征重要性
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist()
        ))
        self.model = {
            "metrics": {"trained": True, "samples": len(X), "positive_rate": float(y.mean()), "cv": cv_result},
            "feature_importance": importance,
            "model_obj": self.model,  # 内部引用
        }

        # 持久化
        model_data = {
            "is_trained": True,
            "feature_names": self.feature_names,
            "feature_importance": importance,
            "params": params,
            "train_date": datetime.now().strftime("%Y-%m-%d"),
            "samples": int(X.shape[0]),
            "cv": cv_result,
        }

        # 保存元数据
        meta_path = self.model_dir / "screening_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        logger.info(f"LightGBM模型训练完成 | 样本数: {len(X)} | 特征数: {X.shape[1]} | CV AUC: {cv_result.get('mean_auc'):.4f}")

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

            if len(X_day) == 0:
                return self._scoring_method(stock_codes, date)

            X_day = np.nan_to_num(np.array(X_day, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            # 预测概率
            probs = model_obj.predict_proba(X_day)[:, 1]

            for code, prob in zip(stock_codes, probs):
                scores[code] = float(prob)

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
