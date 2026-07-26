"""
AI量化交易系统 - 因子处理模块
计算技术因子、基本面因子、量价因子
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger
from utils.data_helper import (
    compute_bollinger_bands, compute_atr, z_score, normalize, rolling_zscore
)


class FactorProcessor:
    """
    因子计算器
    计算技术因子、基本面因子、量价因子
    """

    def __init__(self):
        self.factor_functions = {
            # 技术指标因子
            "ma_5": self._ma,
            "ma_10": self._ma,
            "ma_20": self._ma,
            "ma_60": self._ma,
            "ma_120": self._ma,
            "ma_250": self._ma,
            "rsi_14": self._rsi,
            "rsi_6": self._rsi,
            "rsi_24": self._rsi,
            "macd_hist": self._macd_histogram,
            "boll_position": self._boll_position,
            "atr_14": self._atr,
            "volume_ratio": self._volume_ratio,
            "turnover_rate": self._turnover_rate,
            "price_ma_ratio": self._price_ma_ratio,
            "volatility": self._volatility,
            # 量价因子
            "amihud": self._amihud,
            "realized_volatility": self._realized_volatility,
            "skewness": self._skewness,
            "kurtosis": self._kurtosis,
            "volatility_20": self._volatility,
            "skewness_20": self._skewness,
            "kurtosis_20": self._kurtosis,
            "max_drawdown_20": self._max_drawdown_window,
            "max_gain_20": self._max_gain_window,
            # 动量因子
            "momentum_5": self._momentum,
            "momentum_10": self._momentum,
            "momentum_20": self._momentum,
            "momentum_60": self._momentum,
            # 反转因子
            "reversal_5": self._reversal,
            "reversal_10": self._reversal,
            "reversal_20": self._reversal,
            # 换手率因子
            "avg_turnover_5": self._avg_turnover,
            "avg_turnover_20": self._avg_turnover,
            # 量价关系
            "volume_price_trend": self._vpt,
            "obv": self._obv,
            # 基本面因子 (基于价格数据模拟)
            "pe_ratio": self._pe_ratio,
            "pb_ratio": self._pb_ratio,
            "roe": self._roe,
            "revenue_growth": self._revenue_growth,
            "net_profit_growth": self._net_profit_growth,
        }

    def compute_all_factors(
        self,
        df: pd.DataFrame,
        factor_list: List[str] = None
    ) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            df: 股票日线数据 (index=date, columns=open/high/low/close/volume/amount)
            factor_list: 要计算的因子列表, None表示全部

        Returns:
            DataFrame with computed factors
        """
        if df.empty or len(df) < 60:
            logger.warning(f"数据不足,至少需要60条,当前{len(df)}条")
            return pd.DataFrame()

        factors_dict = {}
        factors_list = factor_list or list(self.factor_functions.keys())

        for factor_name in factors_list:
            if factor_name not in self.factor_functions:
                logger.warning(f"未知因子: {factor_name}")
                continue

            try:
                func = self.factor_functions[factor_name]
                result = func(df)
                if result is None:
                    continue
                # 如果返回 DataFrame，展开为多列
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        factors_dict[col] = result[col]
                elif isinstance(result, pd.Series):
                    factors_dict[factor_name] = result
                else:
                    # 标量值，广播到整个 DataFrame
                    factors_dict[factor_name] = result
            except Exception as e:
                logger.debug(f"因子计算失败: {factor_name}, 错误: {e}")

        if not factors_dict:
            return pd.DataFrame(index=df.index)
        factor_df = pd.DataFrame(factors_dict, index=df.index)
        return factor_df

    def compute_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标因子"""
        technical_factors = [
            "ma_5", "ma_10", "ma_20", "ma_60", "ma_120", "ma_250",
            "rsi_14", "rsi_6", "rsi_24",
            "macd_hist",
            "boll_position",
            "atr_14",
            "volume_ratio",
            "turnover_rate",
            "price_ma_ratio",
            "volatility",
        ]
        return self.compute_all_factors(df, technical_factors)

    def compute_price_volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算量价因子"""
        pv_factors = [
            "amihud", "realized_volatility", "skewness", "kurtosis",
            "max_drawdown_20", "max_gain_20",
            "momentum_5", "momentum_10", "momentum_20", "momentum_60",
            "reversal_5", "reversal_10", "reversal_20",
            "volume_price_trend", "obv",
        ]
        return self.compute_all_factors(df, pv_factors)

    # ========== 因子计算函数 ==========

    def _ma(self, df: pd.DataFrame) -> pd.Series:
        """移动平均线"""
        # 返回多个周期
        result = pd.DataFrame({
            "ma_5": df["close"].rolling(5).mean(),
            "ma_10": df["close"].rolling(10).mean(),
            "ma_20": df["close"].rolling(20).mean(),
            "ma_60": df["close"].rolling(60).mean(),
            "ma_120": df["close"].rolling(120).mean(),
            "ma_250": df["close"].rolling(250).mean(),
        })
        return result

    def _rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI指标"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # 多周期RSI
        periods = [6, 14, 24]
        result = {}
        for period in periods:
            if len(df) < period + 1:
                continue
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            result[f"rsi_{period}"] = rsi
        return pd.DataFrame(result)

    def _macd_histogram(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD直方图"""
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        dif = ema_12 - ema_26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = 2 * (dif - dea)
        return pd.DataFrame({"macd_hist": hist}, index=df.index)

    def _boll_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """布林带位置"""
        bb = compute_bollinger_bands(df["close"], window=20, num_std=2.0)
        boll_pos = (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"] + 1e-10)
        return pd.DataFrame({"boll_position": boll_pos}, index=df.index)

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR指标"""
        return compute_atr(df["high"], df["low"], df["close"], period=14)

    def _volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """量比"""
        vol_ma5 = df["volume"].rolling(5).mean()
        vol_ma20 = df["volume"].rolling(20).mean()
        vr = vol_ma5 / (vol_ma20 + 1e-10)
        return pd.DataFrame({"volume_ratio": vr}, index=df.index)

    def _turnover_rate(self, df: pd.DataFrame) -> pd.Series:
        """换手率"""
        return df.get("turnover", pd.Series(index=df.index))

    def _price_ma_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格与均线比值"""
        result = {}
        for period in [5, 10, 20, 60]:
            ma = df["close"].rolling(period).mean()
            result[f"price_ma_{period}"] = df["close"] / (ma + 1e-10)
        return pd.DataFrame(result, index=df.index)

    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率"""
        returns = df["close"].pct_change()
        vol_10 = returns.rolling(10).std()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        return pd.DataFrame({
            "volatility_10": vol_10,
            "volatility_20": vol_20,
            "volatility_60": vol_60,
        }, index=df.index)

    def _amihud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Amihud流动性指标"""
        amount = df.get("amount", df["volume"] * df["close"])
        abs_return = df["close"].pct_change().abs()
        amihud = (abs_return / (amount + 1e-10)).rolling(20).mean()
        return pd.DataFrame({"amihud": amihud}, index=df.index)

    def _realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """实现波动率"""
        returns = df["close"].pct_change()
        result = {}
        for window in [10, 20, 60]:
            rv = returns.rolling(window).std() * np.sqrt(252)
            result[f"realized_volatility_{window}"] = rv
        return pd.DataFrame(result, index=df.index)

    def _skewness(self, df: pd.DataFrame) -> pd.DataFrame:
        """收益率偏度"""
        returns = df["close"].pct_change()
        skew = returns.rolling(20).skew()
        return pd.DataFrame({"skewness_20": skew}, index=df.index)

    def _kurtosis(self, df: pd.DataFrame) -> pd.DataFrame:
        """收益率峰度"""
        returns = df["close"].pct_change()
        kurt = returns.rolling(20).kurt()
        return pd.DataFrame({"kurtosis_20": kurt}, index=df.index)

    def _max_drawdown_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """滚动最大回撤"""
        returns = df["close"].pct_change()
        cum = (1 + returns).cumprod()
        rolling_max = cum.rolling(20).max()
        dd = (cum - rolling_max) / rolling_max
        # 用 expanding min 计算滚动最大回撤
        max_dd = dd.expanding().min()
        return pd.DataFrame({"max_drawdown_20": max_dd}, index=df.index)

    def _max_gain_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """滚动最大涨幅"""
        returns = df["close"].pct_change()
        mg = returns.rolling(20).max()
        return pd.DataFrame({"max_gain_20": mg}, index=df.index)

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量因子"""
        result = {}
        for period in [5, 10, 20, 60]:
            mom = df["close"] / df["close"].shift(period) - 1
            result[f"momentum_{period}"] = mom
        return pd.DataFrame(result, index=df.index)

    def _reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """反转因子"""
        result = {}
        for period in [5, 10, 20]:
            rev = df["close"] / df["close"].shift(period) - 1
            result[f"reversal_{period}"] = rev
        return pd.DataFrame(result, index=df.index)

    def _avg_turnover(self, df: pd.DataFrame) -> pd.DataFrame:
        """平均换手率"""
        turnover = df.get("turnover", pd.Series(0, index=df.index))
        result = {}
        for period in [5, 20]:
            result[f"avg_turnover_{period}"] = turnover.rolling(period).mean()
        return pd.DataFrame(result, index=df.index)

    def _vpt(self, df: pd.DataFrame) -> pd.Series:
        """量价趋势指标"""
        returns = df["close"].pct_change()
        volume = df["volume"]
        vpt = (returns * volume).cumsum()
        return vpt

    def _obv(self, df: pd.DataFrame) -> pd.Series:
        """OBV指标"""
        direction = np.sign(df["close"].diff())
        # 处理第一个NaN
        direction = direction.fillna(0)
        obv = (direction * df["volume"]).cumsum()
        return obv

    # ========== 基本面因子 (基于价格数据模拟) ==========
    # 注意: 这些是简化版基本面因子,基于价格/成交量数据估算
    # 实际生产环境应接入真实财务数据源 (如 Tushare, Wind, AKShare 财务接口)

    def _pe_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        PE比率 (简化估算版)
        基于价格动量和波动率模拟PE因子
        实际应使用: 市值 / 净利润
        """
        # 使用价格相对强度作为PE代理 (高动量通常对应高PE)
        returns_20 = df["close"] / df["close"].shift(20) - 1
        returns_60 = df["close"] / df["close"].shift(60) - 1
        # PE越高表示估值越高,动量越大
        pe = returns_20 * 10 + returns_60 * 5  # 缩放为合理范围
        return pe

    def _pb_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        PB比率 (简化估算版)
        基于波动率模拟PB因子
        实际应使用: 市值 / 净资产
        """
        # 高波动股票通常有更高PB
        returns = df["close"].pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        pb = vol_20 * 100 + vol_60 * 50  # 缩放为合理范围
        return pb

    def _roe(self, df: pd.DataFrame) -> pd.Series:
        """
        ROE (简化估算版)
        基于价格趋势稳定性模拟ROE
        实际应使用: 净利润 / 净资产
        """
        # 稳定的上升趋势股票ROE较高
        returns = df["close"].pct_change()
        # 正收益天数比例 (越高表示ROE越高)
        positive_days = returns.rolling(60).apply(lambda x: (x > 0).sum()) / 60
        # 收益一致性 (标准差越小越一致)
        vol = returns.rolling(60).std()
        roe = positive_days * 100 / (vol + 0.01)
        return roe

    def _revenue_growth(self, df: pd.DataFrame) -> pd.Series:
        """
        营收增长率 (简化估算版)
        基于量价关系模拟营收增长
        实际应使用: (本期营收 - 上期营收) / 上期营收
        """
        # 成交量增长作为营收增长代理
        volume_ma5 = df["volume"].rolling(5).mean()
        volume_ma20 = df["volume"].rolling(20).mean()
        volume_growth = (volume_ma5 / (volume_ma20 + 1e-10) - 1) * 100
        
        # 成交额增长
        amount = df["volume"] * df["close"]
        amount_ma5 = amount.rolling(5).mean()
        amount_ma20 = amount.rolling(20).mean()
        amount_growth = (amount_ma5 / (amount_ma20 + 1e-10) - 1) * 100
        
        return (volume_growth + amount_growth) / 2

    def _net_profit_growth(self, df: pd.DataFrame) -> pd.Series:
        """
        净利润增长率 (简化估算版)
        基于价格趋势强度模拟利润增长
        实际应使用: (本期利润 - 上期利润) / 上期利润
        """
        # 趋势强度作为利润增长代理
        returns = df["close"].pct_change()
        # 短期动量 vs 长期动量
        mom_5 = returns.rolling(5).sum()
        mom_20 = returns.rolling(20).sum()
        mom_60 = returns.rolling(60).sum()
        
        # 利润增长 = 短期动量 - 长期动量 (加速增长)
        profit_growth = mom_5 * 5 - mom_60 * 2
        return profit_growth

    def standardize_factors(self, df: pd.DataFrame, method: str = "z_score") -> pd.DataFrame:
        """
        因子标准化

        Args:
            df: 因子矩阵
            method: "z_score" / "min_max" / "winsorize"

        Returns:
            标准化后的因子矩阵
        """
        result = df.copy()

        for col in df.columns:
            series = df[col]
            valid = series.dropna()

            if len(valid) == 0:
                result[col] = 0
                continue

            if method == "z_score":
                mean = valid.mean()
                std = valid.std()
                if std > 0:
                    result[col] = (series - mean) / std
                else:
                    result[col] = 0

            elif method == "min_max":
                min_val = valid.min()
                max_val = valid.max()
                if max_val > min_val:
                    result[col] = (series - min_val) / (max_val - min_val)
                else:
                    result[col] = 0

            elif method == "winsorize":
                lower = valid.quantile(0.01)
                upper = valid.quantile(0.99)
                result[col] = series.clip(lower, upper)

        return result

    def cross_section_normalize(self, df: pd.DataFrame, date: pd.Timestamp = None) -> pd.DataFrame:
        """
        截面标准化
        对某一天的所有股票因子做截面标准化
        """
        if date:
            if isinstance(date, str):
                date = pd.Timestamp(date)
            if date in df.index:
                row = df.loc[date]
            else:
                # 找最近的日期
                nearby = df.index[df.index <= date]
                if len(nearby) == 0:
                    return df
                row = df.loc[nearby[-1]]
        else:
            row = df.iloc[-1]

        valid_cols = row.dropna().index
        values = row[valid_cols]

        mean = values.mean()
        std = values.std()

        if std > 0:
            df[valid_cols] = (df[valid_cols] - mean) / std

        return df

    def compute_fundamental_factors(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本面因子
        (需要外部传入财务数据)
        """
        factors = {}

        if financial_data.empty:
            return pd.DataFrame(factors)

        # 从财务数据中提取PE/PB/ROE等
        for col in financial_data.columns:
            factors[col] = financial_data[col]

        return pd.DataFrame(factors)

    def generate_labels(self, df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
        """
        生成标签 (用于监督学习)

        Args:
            df: 股票日线数据
            forward_days: 向前天数

        Returns:
            Series: 未来收益率 (标签)
        """
        future_return = df["close"].shift(-forward_days) / df["close"] - 1
        return future_return

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        factor_list: List[str] = None
    ) -> pd.DataFrame:
        """
        准备训练数据集

        Args:
            df: 股票日线数据
            forward_days: 向前几天作为标签
            factor_list: 因子列表

        Returns:
            DataFrame with features and label
        """
        # 计算因子
        features = self.compute_all_factors(df, factor_list)

        # 生成标签
        labels = self.generate_labels(df, forward_days)

        # 合并
        dataset = pd.concat([features, labels.rename("label")], axis=1)

        # 删除全NaN列
        dataset = dataset.dropna(axis=1, how="all")

        return dataset
