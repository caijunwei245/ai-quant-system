"""快速测试初筛流程"""
import sys
sys.path.insert(0, '.')

from layers.screening import ScreeningLayer
from utils.config_loader import load_config

cfg = load_config()
sl = ScreeningLayer(cfg.get('screening', {}))

# 先用小股票池测试
result = sl.run_screening(date='2026-07-22', universe='hs300', top_k=50)

print(f"\n{'='*40}")
print(f"有效股票: {result.get('valid_stocks', 0)}")
print(f"失败股票: {result.get('failed_stocks', 0)}")
print(f"选出股票: {len(result.get('candidate_stocks', []))}")

if result.get('candidate_stocks'):
    print("\nTop10:")
    for s in result['candidate_stocks'][:10]:
        print(f"  #{s['rank']} {s['code']} score={s['score']}")
else:
    print("未选出股票，检查失败原因...")
    failed = result.get('failed_stocks_detail', [])
    if failed:
        print(f"失败股票详情 (前10个):")
        for f in failed[:10]:
            print(f"  {f['code']}: {f['reason']}")
    else:
        print("无失败详情")
