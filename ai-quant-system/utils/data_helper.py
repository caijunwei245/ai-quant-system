"""
AI量化交易系统 - 数据辅助函数
通用数学/统计辅助函数
"""
import numpy as np
import pandas as pd


def safe_float(value, default=0.0):
    """安全转换为float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def normalize(values: np.ndarray) -> np.ndarray:
    """Min-Max归一化到[0, 1]"""
    if len(values) == 0:
        return values
    min_val = values.min()
    max_val = values.max()
    if max_val - min_val == 0:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def z_score(values: np.ndarray) -> np.ndarray:
    """Z-Score标准化"""
    if len(values) == 0:
        return values
    mean = values.mean()
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动Z-Score"""
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


def calculate_returns(prices: pd.Series) -> pd.Series:
    """计算收益率序列"""
    return prices.pct_change().dropna()


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """年化收益率"""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod()
    n_days = len(returns)
    if n_days == 0:
        return 0.0
    annualized = total_return ** (periods / n_days) - 1
    return float(annualized)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03, periods: int = 252) -> float:
    """夏普比率"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods
    return float(excess_returns.mean() / returns.std() * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """最大回撤"""
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.03, periods: int = 252) -> float:
    """Calmar比率 (年化收益率/最大回撤)"""
    ann_ret = annualized_return(returns, periods)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03, periods: int = 252) -> float:
    """Sortino比率"""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return float("inf")
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    return float(excess_returns.mean() / downside_std * np.sqrt(periods))


def win_rate(returns: pd.Series) -> float:
    """胜率"""
    if len(returns) == 0:
        return 0.0
    positive = (returns > 0).sum()
    total = (returns != 0).sum()
    if total == 0:
        return 0.0
    return float(positive / total)


def profit_loss_ratio(returns: pd.Series) -> float:
    """盈亏比"""
    if len(returns) == 0:
        return 0.0
    profits = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or losses.abs().sum() == 0:
        return float("inf") if len(profits) > 0 else 0.0
    return float(profits.mean() / losses.abs().mean())


def compute_factor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """计算因子相关性矩阵"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr().round(4)


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """IQR方法检测异常值"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def rolling_correlation(x: pd.Series, y: pd.Series, window: int = 20) -> pd.Series:
    """滚动相关性"""
    return x.rolling(window).corr(y)


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> dict:
    """计算布林带"""
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return {"upper": upper, "middle": middle, "lower": lower}


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算ATR (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr
