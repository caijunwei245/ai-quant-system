"""
AI量化交易系统 - 数据验证模块
验证股票数据完整性、交易状态等
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def validate_stock_data(df: pd.DataFrame, stock_code: str = "") -> dict:
    """
    验证股票数据完整性
    返回: {valid: bool, issues: list, warnings: list}
    """
    issues = []
    warnings = []

    if df.empty:
        return {"valid": False, "issues": ["数据为空"], "warnings": []}

    # 检查必需列
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"缺少必需列: {missing_cols}")
        return {"valid": False, "issues": issues, "warnings": warnings}

    # 检查日期索引
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        else:
            issues.append("无日期索引")

    # 检查NaN值比例
    nan_ratio = df[required_cols].isna().sum().sum() / (len(df) * len(required_cols))
    if nan_ratio > 0.1:
        issues.append(f"NaN比例过高: {nan_ratio:.2%}")
    elif nan_ratio > 0.05:
        warnings.append(f"NaN比例偏高: {nan_ratio:.2%}")

    # 检查价格为负
    negative_cols = [col for col in ["open", "high", "low", "close"] if (df[col] < 0).any()]
    if negative_cols:
        issues.append(f"存在负价格: {negative_cols}")

    # 检查成交量为负
    if (df["volume"] < 0).any():
        issues.append("存在负成交量")

    # 检查数据连续性（停牌检测）
    if len(df) > 1:
        consecutive_zeros = (df["volume"] == 0).astype(int).diff()
        suspended_days = (consecutive_zeros == 0).sum()
        if suspended_days > 5:
            warnings.append(f"检测到连续停牌: {suspended_days}天")

    # 检查价格是否全相等（数据异常）
    if df["close"].nunique() == 1 and len(df) > 1:
        issues.append("收盘价全相等，数据可能异常")

    # 检查日期范围
    if isinstance(df.index, pd.DatetimeIndex):
        date_range = df.index.max() - df.index.min()
        if date_range.days < 30:
            warnings.append(f"数据时间跨度不足30天: {date_range.days}天")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "data_points": len(df),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())] if isinstance(df.index, pd.DatetimeIndex) else None,
    }


def check_trading_status(df: pd.DataFrame, date: str = None) -> dict:
    """
    检查股票在指定日期的交易状态
    返回: {trading: bool, status: str, reason: str}
    """
    if df.empty:
        return {"trading": False, "status": "no_data", "reason": "无数据"}

    if date:
        target_date = pd.to_datetime(date)
        # 找最接近的交易日
        available_dates = df.index[df.index <= target_date]
        if len(available_dates) == 0:
            return {"trading": False, "status": "future_date", "reason": f"日期{date}超出数据范围"}
        latest_date = available_dates[-1]
    else:
        latest_date = df.index[-1]

    latest_row = df.loc[latest_date]

    # 检查是否停牌 (成交量为0)
    if latest_row.get("volume", 0) == 0:
        return {"trading": False, "status": "suspended", "reason": f"{latest_date.date()} 停牌"}

    # 检查涨跌停
    close = latest_row.get("close", 0)
    open_price = latest_row.get("open", 0)
    if open_price > 0 and close > 0:
        change_pct = (close - open_price) / open_price * 100
        if change_pct >= 9.8:
            return {"trading": True, "status": "limit_up", "reason": f"涨停 {change_pct:.2f}%"}
        if change_pct <= -9.8:
            return {"trading": True, "status": "limit_down", "reason": f"跌停 {change_pct:.2f}%"}

    return {"trading": True, "status": "normal", "reason": f"正常交易 {latest_date.date()}"}


def is_st_stock(stock_info: dict) -> bool:
    """判断是否为ST股票"""
    name = stock_info.get("name", "")
    code = stock_info.get("code", "")

    if "ST" in name or "*ST" in name:
        return True
    if code.startswith("300") or code.startswith("688"):
        # 创业板和科创板也有ST
        pass
    return False


def is_new_stock(stock_info: dict, lookback_days: int = 60) -> bool:
    """判断是否为次新股（上市不足lookback_days天）"""
    listed_date = stock_info.get("listed_date", "")
    if not listed_date:
        return False

    try:
        listed_dt = pd.to_datetime(listed_date)
        days_since_ipo = (pd.Timestamp.now() - listed_dt).days
        return days_since_ipo < lookback_days
    except:
        return False


def check_high_pledge(stock_info: dict, threshold: float = 0.3) -> bool:
    """检查是否高质押（质押比例超过阈值）"""
    pledge_ratio = stock_info.get("pledge_ratio", 0)
    return pledge_ratio > threshold


def check_regulatory(stock_info: dict) -> bool:
    """检查是否正在被监管"""
    status = stock_info.get("status", "")
    regulatory = stock_info.get("regulatory", False)
    return regulatory or status in ["立案调查", "监管函", "警告"]


def filter_unqualified_stocks(stock_pool: list, config: dict = None) -> list:
    """
    根据排除规则过滤不合格股票
    """
    if config is None:
        config = {
            "st_stocks": True,
            "suspended": True,
            "newly_ipo": True,
            "high_pledged": True,
            "regulatory": True,
        }

    filtered = []
    excluded_reasons = {}

    for stock in stock_pool:
        reasons = []

        if config.get("st_stocks", True) and is_st_stock(stock):
            reasons.append("ST")

        if config.get("newly_ipo", True) and is_new_stock(stock):
            reasons.append("次新股")

        if config.get("high_pledged", True) and check_high_pledge(stock):
            reasons.append("高质押")

        if config.get("regulatory", True) and check_regulatory(stock):
            reasons.append("监管中")

        if reasons:
            excluded_reasons[stock.get("code", "unknown")] = reasons
        else:
            filtered.append(stock)

    return filtered, excluded_reasons


def validate_portfolio(weights: dict, limits: dict = None) -> dict:
    """
    验证投资组合合规性
    """
    if limits is None:
        limits = {
            "max_single_position": 0.10,
            "max_industry_position": 0.30,
            "max_correlation": 0.7,
        }

    issues = []
    warnings = []

    if not weights:
        return {"valid": False, "issues": ["空组合"], "warnings": []}

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        issues.append(f"仓位总和异常: {total_weight:.4f}")

    for code, weight in weights.items():
        if weight > limits["max_single_position"]:
            issues.append(f"{code} 仓位{weight:.2%}超过上限{limits['max_single_position']:.2%}")
        if weight < 0.001 and weight > 0:
            warnings.append(f"{code} 仓位过小: {weight:.4f}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "total_weight": total_weight,
        "n_positions": len(weights),
    }
