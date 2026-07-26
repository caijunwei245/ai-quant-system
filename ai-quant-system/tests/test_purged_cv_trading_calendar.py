import numpy as np
import pandas as pd
from layers.screening import ScreeningLayer


def test_purged_kfold_trading_calendar():
    # 构建交易日历为工作日 (business days)
    trading_index = pd.bdate_range(start="2026-07-01", periods=30)

    # 样本日期与交易日一一对应 (每个交易日1个样本)
    sample_dates = list(trading_index)

    # 合成特征 X 和标签 y（单个样本对应一个交易日）
    rng = np.random.RandomState(42)
    X = rng.randn(len(sample_dates), 5).astype("float32")
    # 让标签在不同时段有正负样本
    y = np.array([1 if (i % 7 == 0) else 0 for i in range(len(sample_dates))], dtype=int)

    layer = ScreeningLayer(config={})

    # 运行 purged cv，显式传入 trading_calendar
    result = layer._purged_kfold_cv(X, y, sample_dates, n_splits=3, embargo_days=1, forward_days=2, trading_calendar=trading_index)

    # 确认有 folds 返回
    assert isinstance(result, dict)
    assert result.get("n_folds", 0) >= 1

    # 找到 val_start 为 trading_index[10] 的那一折，检查训练样本数是否符合预期
    expected_val_start = str(pd.to_datetime(trading_index[10]).date())
    target_fold = None
    for f in result.get("folds", []):
        if f.get("val_start") == expected_val_start:
            target_fold = f
            break

    assert target_fold is not None, f"未找到 val_start={expected_val_start} 的折"

    # 计算期望训练数: ranks 0..9 为训练样本, purge 条件为 sample_rank + fwd >= val_start_rank - embargo
    # val_start_rank = 10, embargo=1, fwd=2 => purge sample_rank >= 7 => remaining sample_rank 0..6 => 7 样本
    assert target_fold.get("n_train") == 7, f"期望 n_train=7, 实际 {target_fold.get('n_train')}"
