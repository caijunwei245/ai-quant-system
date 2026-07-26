"""
Light-weight LLM client adapter for the project.
Supports multiple providers via a simple interface.

Interfaces:
- LLMClient(config).sentiment(texts: List[str]) -> List[dict]

Providers supported in this adapter:
- 'mock' : returns neutral 0 scores (useful for testing)
- 'lm-studio' : expects a HTTP endpoint in config['endpoint'] that accepts {"prompt":..., "model":...}

This adapter is intentionally conservative: if the configured provider is unavailable,
it raises a descriptive error (so the caller can fallback to rule-based analysis).
"""
from typing import List, Dict, Optional
import json
import logging

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.provider = config.get("provider", "mock")
        self.model = config.get("model")
        self.endpoint = config.get("endpoint")
        self.use_gpu = config.get("use_gpu", False)
        # tuning params
        self.timeout = config.get("timeout", 30)

        if self.provider == "lm-studio" and not self.endpoint:
            raise ValueError("lm-studio provider requires 'endpoint' in config")
        if self.provider == "lm-studio" and not HAS_REQUESTS:
            raise RuntimeError("requests library required for lm-studio provider but not installed")

    def sentiment(self, texts: List[str]) -> List[Dict]:
        """Return list of dicts: [{'text':..., 'score': -1..1, 'reason': ...}, ...]"""
        if self.provider == "mock":
            return [{"text": t, "score": 0.0, "reason": "mocked"} for t in texts]

        if self.provider == "lm-studio":
            # build a compact prompt asking for JSON output
            examples = [
                {"text": "公司业绩大幅增长，盈利超预期", "score": 0.9, "reason": "业绩向好"},
                {"text": "公司遭遇监管调查，利润下滑", "score": -0.9, "reason": "监管风险"},
            ]
            prompt = (
                "请对下面的每条新闻/研报文本做情感评分，返回 JSON 列表，\n"
                "每项包含字段: text, score (取值范围 -1 到 1), reason (一句话说明理由)。\n"
                "只输出 JSON，不要额外说明。示例: "
                + json.dumps(examples, ensure_ascii=False)
                + "\n要评分的文本:\n"
            )
            prompt += "\n".join(texts)

            payload = {"prompt": prompt, "model": self.model}
            try:
                resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                # try to extract JSON from the model output - common shapes vary
                text_out = None
                if isinstance(data, dict):
                    # Common: {'choices':[{'text': '...'}]}
                    choices = data.get("choices")
                    if choices and isinstance(choices, list):
                        text_out = choices[0].get("text")
                    else:
                        # Some services return {'result': '...'}
                        text_out = data.get("result") or data.get("output") or None
                elif isinstance(data, list):
                    # already list
                    return data

                if not text_out:
                    # fallback: try raw text
                    text_out = resp.text

                # attempt to parse first JSON structure found
                parsed = None
                try:
                    parsed = json.loads(text_out)
                except Exception:
                    # try to locate JSON substring
                    import re

                    m = re.search(r"(\[\s*\{.*\}\s*\])", text_out, flags=re.S)
                    if m:
                        parsed = json.loads(m.group(1))

                if isinstance(parsed, list):
                    return parsed
                else:
                    raise ValueError("无法解析 LLM 返回的 JSON 列表")

            except Exception as e:
                logger.warning(f"调用 lm-studio 接口失败: {e}")
                # 让上层决定如何回退
                raise

        # 未知 provider
        raise ValueError(f"Unsupported LLM provider: {self.provider}")


class MockLLMClient(LLMClient):
    def __init__(self):
        super().__init__({"provider": "mock"})

    def sentiment(self, texts: List[str]) -> List[Dict]:
        # simple rule: positive if 包含 '增' or '利好'，negative if 包含 '亏' '下跌' '调查'
        out = []
        for t in texts:
            score = 0.0
            reason = "neutral"
            if any(w in t for w in ["增", "利好", "盈利", "上涨", "突破"]):
                score = 0.7
                reason = "包含正面关键词"
            if any(w in t for w in ["亏", "下跌", "调查", "处罚", "暴雷", "违约"]):
                score = -0.8
                reason = "包含负面关键词"
            out.append({"text": t, "score": score, "reason": reason})
        return out
