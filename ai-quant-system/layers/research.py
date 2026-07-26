"""
AI量化交易系统 - 精研层 (Research Layer)
TradingAgents 读研报/新闻做逻辑验真,排除风险标的
多智能体负责"读信息"
"""
import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
from loguru import logger
from utils.llm_client import LLMClient, MockLLMClient

try:
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from data.fetcher import DataFetcher
from utils.validator import filter_unqualified_stocks
from utils.data_helper import safe_float


class TradingAgent:
    """
    单个TradingAgent - 负责特定领域的分析
    """

    def __init__(self, agent_id: str, role: str, config: dict = None):
        self.agent_id = agent_id
        self.role = role
        self.config = config or {}
        self.analysis_history = []

    def analyze(self, stock_code: str, context: dict) -> dict:
        """执行分析"""
        raise NotImplementedError


class SentimentAgent(TradingAgent):
    """情感分析Agent - 优先使用LLM做 prompt-based sentiment 分析，回退到关键词规则。

    如果传入 llm_client（具有 sentiment(texts) 方法），则调用 LLM 返回每段文本的情感分数（-1..1）和简短理由。
    否则回退到原有的关键词+规则实现。
    """

    def __init__(self, config: dict = None, llm_client: Optional[object] = None):
        super().__init__("sentiment", "sentiment_analysis", config)
        self.config = config or {}
        keywords = self.config.get("keywords", {})
        self.positive_words = keywords.get("positive", ["增长", "突破", "利好", "景气", "超预期", "盈利", "扩张"])
        self.negative_words = keywords.get("negative", ["亏损", "退市", "处罚", "调查", "暴雷", "减持", "质押", "下滑"])
        self.intensifiers = {"非常": 1.5, "大幅": 1.3, "明显": 1.2, "显著": 1.2}
        self.negators = {"不": -1, "未": -1, "没有": -1, "缺乏": -1}
        # 可注入的 LLM 客户端，用于 prompt-based sentiment
        self.llm_client = llm_client

    def analyze(self, stock_code: str, context: dict) -> dict:
        """分析股票相关新闻/研报情感

        优先使用 llm_client（如果可用且 provider != 'mock'），否则使用本地关键词规则。
        返回格式与原实现兼容。
        """
        texts = context.get("texts", [])
        if not texts:
            return {"agent": "sentiment", "score": 0, "confidence": 0, "details": {}}

        details = {"positive": [], "negative": [], "neutral": []}
        scores = []

        # 如果有 llm_client，则通过 LLM 获取情感分数
        if self.llm_client is not None:
            try:
                llm_results = self.llm_client.sentiment(texts)
                # 期望 llm_results 为 [{'text':..., 'score': -1..1, 'reason':...}, ...]
                for res in llm_results:
                    score = float(res.get("score", 0))
                    txt = res.get("text", "")
                    reason = res.get("reason", "")
                    scores.append(score)
                    if score > 0.1:
                        details["positive"].append({"text": txt[:50], "score": score, "reason": reason})
                    elif score < -0.1:
                        details["negative"].append({"text": txt[:50], "score": score, "reason": reason})
                    else:
                        details["neutral"].append({"text": txt[:50], "score": score, "reason": reason})

                final_score = float(np.mean(scores)) if scores else 0
                confidence = float(np.std(scores))

                return {
                    "agent": "sentiment",
                    "score": final_score,
                    "confidence": min(confidence, 1.0),
                    "details": details,
                    "n_texts": len(texts),
                }
            except Exception as e:
                logger.warning(f"LLM 情感分析失败，回退到关键词规则: {e}")
                # 回退到规则

        # 原有关键词规则回退实现
        for text in texts:
            score, pos_words, neg_words = self._analyze_text(text)
            if score > 0.1:
                details["positive"].append({"text": text[:50], "score": score})
            elif score < -0.1:
                details["negative"].append({"text": text[:50], "score": score})
            else:
                details["neutral"].append({"text": text[:50], "score": score})
            scores.append(score)

        final_score = float(np.mean(scores)) if scores else 0
        confidence = float(np.std(scores))  # 标准差越小越一致

        return {
            "agent": "sentiment",
            "score": final_score,
            "confidence": min(confidence, 1.0),
            "details": details,
            "n_texts": len(texts),
        }

    def _analyze_text(self, text: str) -> Tuple[float, list, list]:
        """分析单段文本情感 (原有关键词规则)"""
        pos_found = []
        neg_found = []
        score = 0

        # 找 intensifiers
        for word, weight in self.intensifiers.items():
            if word in text:
                score += weight * 0.1

        # 找 negators
        for word, weight in self.negators.items():
            if word in text:
                # 检查后续是否有关键词
                idx = text.find(word)
                subsequent = text[idx:idx+10]
                for neg_word in self.negative_words:
                    if neg_word in subsequent:
                        score -= 0.2
                        neg_found.append(neg_word)

        # 找正面关键词
        for word in self.positive_words:
            count = text.count(word)
            if count > 0:
                score += count * 0.15
                pos_found.extend([word] * count)

        # 找负面关键词
        for word in self.negative_words:
            count = text.count(word)
            if count > 0:
                score -= count * 0.2
                neg_found.extend([word] * count)

        # 归一化到 [-1, 1]
        score = np.tanh(score)

        return score, pos_found, neg_found


class RiskAgent(TradingAgent):
    """风险识别Agent - 识别各类风险"""

    def __init__(self, config: dict = None):
        super().__init__("risk", "risk_identification", config)
        self.config = config or {}

    def analyze(self, stock_code: str, context: dict) -> dict:
        """分析股票风险"""
        risks = []
        risk_score = 0

        # 1. ST风险
        if context.get("is_st", False):
            risks.append({"type": "ST", "severity": "high", "score": -0.8})
            risk_score += 0.8

        # 2. 停牌风险
        if context.get("is_suspended", False):
            risks.append({"type": "suspended", "severity": "high", "score": -0.9})
            risk_score += 0.9

        # 3. 高质押风险
        pledge_ratio = context.get("pledge_ratio", 0)
        if pledge_ratio > 0.3:
            severity = "high" if pledge_ratio > 0.5 else "medium"
            risks.append({"type": "high_pledge", "severity": severity, "score": -0.3 * pledge_ratio})
            risk_score += 0.3 * pledge_ratio

        # 4. 监管风险
        if context.get("regulatory", False):
            risks.append({"type": "regulatory", "severity": "high", "score": -0.7})
            risk_score += 0.7

        # 5. 次新股风险
        if context.get("is_new_stock", False):
            risks.append({"type": "new_stock", "severity": "low", "score": -0.1})
            risk_score += 0.1

        # 6. 流动性风险
        avg_volume = context.get("avg_volume_20", 0)
        if avg_volume < 1000:
            risks.append({"type": "low_liquidity", "severity": "medium", "score": -0.3})
            risk_score += 0.3

        # 7. 财务风险
        if context.get("revenue_decline", False):
            risks.append({"type": "revenue_decline", "severity": "medium", "score": -0.4})
            risk_score += 0.4

        if context.get("profit_decline", False):
            risks.append({"type": "profit_decline", "severity": "medium", "score": -0.5})
            risk_score += 0.5

        # 综合风险评分
        final_score = -min(risk_score, 1.0)

        return {
            "agent": "risk",
            "score": final_score,
            "confidence": min(risk_score + 0.3, 1.0),
            "risks": risks,
            "n_risks": len(risks),
        }


class NewsAgent(TradingAgent):
    """新闻分析Agent - 分析新闻时效性和影响力"""

    def __init__(self, config: dict = None):
        super().__init__("news", "news_analysis", config)
        self.config = config or {}

    def analyze(self, stock_code: str, context: dict) -> dict:
        """分析新闻"""
        news_list = context.get("news", [])
        if not news_list:
            return {"agent": "news", "score": 0, "confidence": 0, "news_count": 0}

        scores = []
        news_details = []

        for news in news_list:
            score = self._score_news(news)
            scores.append(score)
            news_details.append({
                "title": news.get("title", "")[:30],
                "date": news.get("published_at", ""),
                "score": score,
            })

        # 时间衰减: 最近的新闻权重更高
        n = len(scores)
        if n > 0:
            weighted_score = sum(s * (0.9 ** (n - i - 1)) for i, s in enumerate(scores))
            weight_sum = sum(0.9 ** (n - i - 1) for i in range(n))
            final_score = weighted_score / weight_sum if weight_sum > 0 else 0
        else:
            final_score = 0

        return {
            "agent": "news",
            "score": float(final_score),
            "confidence": float(np.std(scores)) if n > 1 else 0.5,
            "news_count": n,
            "news_details": news_details[:10],
            "positive_ratio": sum(1 for s in scores if s > 0) / n if n > 0 else 0,
        }

    def _score_news(self, news: dict) -> float:
        """评估单条新闻的情感分数"""
        title = news.get("title", "")
        content = news.get("content", "")
        text = title + content

        score = 0

        # 正面词汇
        positive_keywords = ["上涨", "增长", "盈利", "利好", "突破", "景气", "超预期", "中标", "合作", "扩张"]
        negative_keywords = ["下跌", "亏损", "处罚", "调查", "减持", "质押", "违约", "暴雷", "退市", "立案"]

        for kw in positive_keywords:
            if kw in text:
                score += 0.15

        for kw in negative_keywords:
            if kw in text:
                score -= 0.2

        # 标题权重更高
        for kw in positive_keywords:
            if kw in title:
                score += 0.05
        for kw in negative_keywords:
            if kw in title:
                score -= 0.1

        return float(np.tanh(score))


class ResearchLayer:
    """
    精研层: TradingAgents多智能体系统
    读研报/新闻做逻辑验真,排除风险标的
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.nlp_config = self.config.get("nlp", {})
        self.report_config = self.config.get("report_analysis", {})
        self.news_config = self.config.get("news_analysis", {})
        self.exclusion_config = self.config.get("exclusion", {})

        self.data_fetcher = DataFetcher()

        # 初始化 LLM 客户端（可选）
        llm_cfg = self.nlp_config.get("llm", {})
        self.llm_client = None
        try:
            if llm_cfg:
                self.llm_client = LLMClient(llm_cfg)
        except Exception as e:
            logger.warning(f"初始化 LLM 客户端失败: {e}. 将回退到规则情感分析。")
            self.llm_client = None

        # 初始化Agents（将 llm_client 注入 SentimentAgent）
        self.agents = {
            "sentiment": SentimentAgent(self.report_config, llm_client=self.llm_client),
            "risk": RiskAgent(),
            "news": NewsAgent(self.news_config),
        }

        # 排除规则
        self.exclusion_rules = {
            "st_stocks": self.exclusion_config.get("st_stocks", True),
            "suspended": self.exclusion_config.get("suspended", True),
            "newly_ipo": self.exclusion_config.get("newly_ipo", True),
            "high_pledged": self.exclusion_config.get("high_pledged", True),
            "regulatory": self.exclusion_config.get("regulatory", True),
        }

        # 最小情感分数阈值
        self.min_sentiment = self.report_config.get("min_sentiment_score", 0.2)
        self.min_news_count = self.news_config.get("min_news_count", 5)

        # 结果缓存
        self.research_results = {}

    def run_research(
        self,
        candidate_stocks: List[dict],
        date: str = None
    ) -> Dict:
        """
        执行精研分析

        Args:
            candidate_stocks: 初筛层候选股票列表
            date: 分析日期

        Returns:
            {
                "date": "2026-07-21",
                "verified_stocks": [...],
                "excluded_stocks": [...],
                "agent_results": {...},
                "summary": {...}
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"=" * 60)
        logger.info(f"精研层开始工作 | 日期: {date} | 候选股票: {len(candidate_stocks)}只")
        logger.info(f"=" * 60)

        verified_stocks = []
        excluded_stocks = []
        agent_results = {}

        for stock in candidate_stocks:
            code = stock.get("code", "")
            stock_result = {
                "code": code,
                "name": stock.get("name", ""),
                "initial_score": stock.get("score", 0),
                "agents": {},
                "final_score": 0,
                "verified": False,
                "exclusion_reasons": [],
            }

            # Step 1: 排除规则检查
            exclusion_reasons = self._check_exclusions(code)
            stock_result["exclusion_reasons"] = exclusion_reasons

            if exclusion_reasons:
                stock_result["verified"] = False
                excluded_stocks.append({
                    "code": code,
                    "name": stock.get("name", ""),
                    "reasons": exclusion_reasons,
                    "initial_score": stock.get("score", 0),
                })
                logger.warning(f"排除股票: {code} - {exclusion_reasons}")
                self.research_results[code] = stock_result
                continue

            # Step 2: 获取股票新闻/研报数据
            news = self.data_fetcher.fetch_news(code, days=self.news_config.get("lookback_days", 30))

            # Step 3: 获取基本面信息
            financial_data = self.data_fetcher.fetch_financial_data(code)

            # Step 4: 运行各Agent分析
            context = {
                "texts": [n.get("content", n.get("title", "")) for n in news],
                "news": news,
                "is_st": "ST" in stock.get("name", ""),
                "is_suspended": False,  # 需实时检查
                "pledge_ratio": safe_float(stock.get("pledge_ratio", 0)),
                "regulatory": False,  # 需检查
                "is_new_stock": False,  # 需检查
                "revenue_decline": False,
                "profit_decline": False,
            }

            agent_scores = {}
            for agent_name, agent in self.agents.items():
                try:
                    result = agent.analyze(code, context)
                    agent_scores[agent_name] = result
                    logger.debug(f"Agent [{agent_name}] 分析 {code}: score={result.get('score', 0):.3f}")
                except Exception as e:
                    logger.error(f"Agent [{agent_name}] 分析失败: {code}, 错误: {e}")
                    agent_scores[agent_name] = {"agent": agent_name, "score": 0, "confidence": 0}

            stock_result["agents"] = agent_scores

            # Step 5: 综合各Agent评分
            final_score = self._aggregate_agent_scores(agent_scores)
            stock_result["final_score"] = final_score

            # Step 6: 判断是否通过精研
            if self._pass_research(final_score, agent_scores):
                verified_stocks.append({
                    "code": code,
                    "name": stock.get("name", ""),
                    "initial_score": stock.get("score", 0),
                    "final_score": round(final_score, 4),
                    "agent_scores": {k: round(v.get("score", 0), 4) for k, v in agent_scores.items()},
                })
                logger.info(f"通过精研: {code} | 综合分: {final_score:.3f}")
            else:
                excluded_stocks.append({
                    "code": code,
                    "name": stock.get("name", ""),
                    "reasons": ["精研未通过"],
                    "agent_scores": agent_scores,
                    "initial_score": stock.get("score", 0),
                    "final_score": round(final_score, 4),
                })
                logger.info(f"精研未通过: {code} | 综合分: {final_score:.3f}")

            self.research_results[code] = stock_result

        result = {
            "date": date,
            "candidate_count": len(candidate_stocks),
            "verified_count": len(verified_stocks),
            "excluded_count": len(excluded_stocks),
            "verified_stocks": verified_stocks,
            "excluded_stocks": excluded_stocks,
            "agent_results": agent_results,
            "summary": {
                "pass_rate": len(verified_stocks) / len(candidate_stocks) if candidate_stocks else 0,
                "avg_final_score": np.mean([s["final_score"] for s in verified_stocks]) if verified_stocks else 0,
            },
        }

        logger.info(f"精研完成 | 通过: {len(verified_stocks)}/{len(candidate_stocks)} | 排除: {len(excluded_stocks)}")

        return result

    def _check_exclusions(self, stock_code: str) -> List[str]:
        """检查排除规则"""
        reasons = []

        # 获取股票信息
        stock_list = self.data_fetcher.fetch_stock_list("a_share")
        stock_info = None
        for s in stock_list:
            if s.get("code") == stock_code:
                stock_info = s
                break

        if not stock_info:
            return ["股票信息获取失败"]

        # ST检查
        if self.exclusion_rules.get("st_stocks", True):
            if "ST" in stock_info.get("name", "") or "*ST" in stock_info.get("name", ""):
                reasons.append("ST股")

        # 高质押检查
        if self.exclusion_rules.get("high_pledged", True):
            pledge = stock_info.get("pledge_ratio", 0)
            if pledge > 0.3:
                reasons.append(f"高质押({pledge:.1%})")

        # 监管检查
        if self.exclusion_rules.get("regulatory", True):
            if stock_info.get("status") in ["立案调查", "监管函", "警告"]:
                reasons.append("监管中")

        return reasons

    def _aggregate_agent_scores(self, agent_scores: dict) -> float:
        """
        综合各Agent评分
        权重: 情感40%, 风险30%, 新闻30%
        """
        weights = {
            "sentiment": 0.4,
            "risk": 0.3,
            "news": 0.3,
        }

        total_score = 0
        total_weight = 0

        for agent_name, score_dict in agent_scores.items():
            w = weights.get(agent_name, 0)
            if w > 0:
                score = score_dict.get("score", 0)
                confidence = score_dict.get("confidence", 0.5)
                # 置信度加权
                weighted_score = score * (0.5 + 0.5 * confidence)
                total_score += weighted_score * w
                total_weight += w

        if total_weight > 0:
            return total_score / total_weight
        return 0

    def _pass_research(self, final_score: float, agent_scores: dict) -> bool:
        """判断是否通过精研"""
        # 综合分数阈值
        if final_score < self.min_sentiment:
            return False

        # 风险Agent否决
        risk_score = agent_scores.get("risk", {}).get("score", 0)
        if risk_score < -0.5:
            return False

        # 情感Agent否决
        sentiment_score = agent_scores.get("sentiment", {}).get("score", 0)
        if sentiment_score < -0.3:
            return False

        return True

    def get_research_summary(self) -> dict:
        """获取精研总结"""
        if not self.research_results:
            return {}

        verified = [r for r in self.research_results.values() if r.get("verified")]
        excluded = [r for r in self.research_results.values() if not r.get("verified")]

        return {
            "total_processed": len(self.research_results),
            "verified": len(verified),
            "excluded": len(excluded),
            "exclusion_reasons": self._get_exclusion_stats(),
            "agent_avg_scores": self._get_agent_avg_scores(),
        }

    def _get_exclusion_stats(self) -> dict:
        """获取排除原因统计"""
        stats = {}
        for result in self.research_results.values():
            for reason in result.get("exclusion_reasons", []):
                stats[reason] = stats.get(reason, 0) + 1
        return stats

    def _get_agent_avg_scores(self) -> dict:
        """获取各Agent平均分数"""
        agent_sums = {}
        agent_counts = {}
        for result in self.research_results.values():
            for agent_name, agent_result in result.get("agents", {}).items():
                score = agent_result.get("score", 0)
                agent_sums[agent_name] = agent_sums.get(agent_name, 0) + score
                agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1

        return {
            name: agent_sums[name] / agent_counts[name]
            for name in agent_sums
        }
