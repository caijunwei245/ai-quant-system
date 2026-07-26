"""
Self-test script for LLM-based SentimentAgent integration.
Runs a small example using the MockLLMClient so it works without external services.

Usage:
    python scripts/selftest_llm.py
"""
import json
import importlib.util
from pathlib import Path

# Load SentimentAgent directly from file to avoid importing the whole package (and heavy deps)
research_path = Path(__file__).resolve().parents[1] / 'layers' / 'research.py'
spec = importlib.util.spec_from_file_location('layers.research', str(research_path))
research = importlib.util.module_from_spec(spec)
import sys, types
# Provide a minimal mock for loguru.logger to avoid requiring the dependency in the self-test
if 'loguru' not in sys.modules:
    mock_loguru = types.ModuleType('loguru')
    class _MockLogger:
        def debug(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def exception(self, *a, **k): pass
    mock_loguru.logger = _MockLogger()
    sys.modules['loguru'] = mock_loguru

# Also provide minimal numpy and pandas if missing to avoid heavy deps for this simple test
try:
    import numpy as np  # prefer real if available
except Exception:
    mock_np = types.SimpleNamespace(tanh=lambda x: x if hasattr(x, '__iter__') else x, mean=lambda x: sum(x)/len(x) if x else 0, std=lambda x: 0.0)
    sys.modules['numpy'] = mock_np

try:
    import pandas as pd
except Exception:
    mock_pd = types.ModuleType('pandas')
    sys.modules['pandas'] = mock_pd

spec.loader.exec_module(research)
SentimentAgent = research.SentimentAgent

from utils.llm_client import MockLLMClient


def main():
    llm = MockLLMClient()
    agent = SentimentAgent(config={
        "keywords": {
            "positive": ["增长", "突破", "利好"],
            "negative": ["亏损", "退市", "处罚"]
        }
    }, llm_client=llm)

    texts = [
        "公司一季度业绩大幅增长，盈利超预期，主营业务扩张明显。",
        "公司被监管调查，存在财务造假嫌疑，股价大幅下跌。",
        "短期内暂无重大利空或利好，运营平稳。",
    ]

    ctx = {"texts": texts}
    res = agent.analyze("000001", ctx)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
