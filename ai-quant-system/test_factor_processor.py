"""测试 FactorProcessor 因子计算"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from data.processor import FactorProcessor
from datetime import datetime

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range('2021-01-01', periods=200, freq='B')
base_price = 10.0
returns = np.random.normal(0.0005, 0.02, 200)
prices = base_price * np.cumprod(1 + returns)

df = pd.DataFrame({
    'open': prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
    'high': prices * (1 + np.abs(np.random.uniform(0, 0.02, 200))),
    'low': prices * (1 - np.abs(np.random.uniform(0, 0.02, 200))),
    'close': prices,
    'volume': np.random.randint(100000, 1000000, 200),
    'amount': prices * np.random.randint(100000, 1000000, 200),
}, index=dates)

print(f"模拟数据: {len(df)} 条")
print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")

# 测试 FactorProcessor
fp = FactorProcessor()

# 测试所有因子
print("\n计算所有因子...")
factors = fp.compute_all_factors(df)

if factors.empty:
    print("ERROR: factors 为空 DataFrame!")
else:
    print(f"因子数量: {len(factors.columns)}")
    print(f"因子列: {list(factors.columns)}")
    print(f"\n最后5行因子数据:")
    print(factors.tail().to_string())
    
    # 检查 NaN
    nan_count = factors.isna().sum()
    total_nan = nan_count.sum()
    print(f"\nNaN 统计: 总计 {total_nan} 个, 占比 {total_nan / (len(factors) * len(factors.columns)):.1%}")
    
    # 检查各因子 NaN 比例
    for col in factors.columns:
        nan_pct = factors[col].isna().mean()
        if nan_pct > 0.5:
            print(f"  警告: {col} 有 {nan_pct:.0%} 的 NaN 值")

print("\n" + "="*40)
print("FactorProcessor 测试完成!")
