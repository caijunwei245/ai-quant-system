"""极简测试初筛流程 - 验证端到端链路"""
import sys
sys.path.insert(0, '.')
from layers.screening import ScreeningLayer
from utils.config_loader import load_config
import pandas as pd
import numpy as np

# 1. 加载配置
cfg = load_config()
sl = ScreeningLayer(cfg.get('screening', {}))

# 2. Mock 一个极小规模的股票池 (仅用5只股票)
# 我们通过覆盖 _get_universe 方法来实现快速测试
sl._get_universe = lambda universe: [
    {"code": "600000"}, {"code": "600001"}, {"code": "600002"}, 
    {"code": "600003"}, {"code": "600004"}
]

print("开始极简端到端测试 (5只股票)...")
try:
    # 使用一个确定的日期，确保能命中缓存或快速获取
    result = sl.run_screening(date='2026-07-22', universe='mini', top_k=3)
    
    print(f"\n{'='*40}")
    print(f"测试结果:")
    print(f"有效股票: {result.get('valid_stocks', 0)}")
    print(f"失败股票: {result.get('failed_stocks', 0)}")
    print(f"最终选出: {len(result.get('candidate_stocks', []))}")
    
    if result.get('candidate_stocks'):
        print("\nTop Candidates:")
        for s in result['candidate_stocks']:
            print(f"  #{s['rank']} {s['code']} score={s['score']}")
    else:
        print("\n未选出股票，详情如下:")
        for f in result.get('failed_stocks_detail', [])[:5]:
            print(f"  {f['code']}: {f['reason']}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n测试过程中发生错误: {e}")