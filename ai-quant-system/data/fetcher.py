"""
AI量化交易系统 - 数据获取模块
支持多种数据源: AKShare, yfinance, CSV本地数据
 支持AKShare、yfinance、Tushare MCP、CSV本地数据
"""
import os
import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger


class DataFetcher:
    """
    统一数据获取接口
    支持AKShare、yfinance、CSV等多种数据源
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.primary_source = self.config.get("primary_source", "akshare")
        self.max_retries = self.config.get("max_retries", 2)
        # 减少重试延迟到2秒
        self.retry_delay = self.config.get("retry_delay", 2)
        self._cache = {}
        self._cache_dir = Path("../data/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, code: str, start: str, end: str, source: str = None, adjust: str = "") -> str:
        """生成缓存键"""
        src = source or self.primary_source
        # 缓存键包含: code, date range, source, adjust
        return f"{code}_{start}_{end}_{src}_{adjust}"

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        """缓存到文件"""
        cache_file = self._cache_dir / f"{key}.pkl"
        try:
            df.to_pickle(cache_file)
        except Exception as e:
            logger.debug(f"缓存保存失败: {e}")

    def _load_cache(self, key: str) -> pd.DataFrame:
        """从文件加载缓存"""
        cache_file = self._cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                logger.debug(f"缓存命中: {key}")
                return df
            except Exception as e:
                logger.debug(f"缓存加载失败: {e}")
        return None

    def fetch_stock_data(
        self,
        stock_code: str,
        start_date: str = "2020-01-01",
        end_date: str = None,
        source: str = None,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        获取股票日线数据 (带本地缓存)

        Args:
            stock_code: 股票代码, 如 "000001"
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源 "akshare" / "yfinance" / "tushare" / "csv"
            adjust: 复权方式 "qfq"(前复权) / "hfq"(后复权) / ""(不复权)

        Returns:
            DataFrame with columns: open, high, low, close, volume, amount, turnover
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        cache_key = self._get_cache_key(stock_code, start_date, end_date, source, adjust)

        # 1. 内存缓存
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # 2. 文件缓存 (跨进程/重启有效)
        cached_df = self._load_cache(cache_key)
        if cached_df is not None and not cached_df.empty:
            self._cache[cache_key] = cached_df
            return cached_df.copy()

        source = source or self.primary_source
        df = None

        if source == "akshare":
            df = self._fetch_akshare(stock_code, start_date, end_date, adjust)
            # AKShare 失败时不再回退 yfinance（A股数据源优先使用AKShare）
            if df.empty:
                logger.debug(f"AKShare数据获取失败: {stock_code}")
        elif source == "yfinance":
            df = self._fetch_yfinance(stock_code, start_date, end_date, adjust)
        elif source == "csv":
            df = self._fetch_csv(stock_code, start_date, end_date)
        elif source == "tushare":
            df = self._fetch_tushare_mcp(stock_code, start_date, end_date)
        else:
            logger.warning(f"未知数据源: {source}, 使用默认 akshare")
            df = self._fetch_akshare(stock_code, start_date, end_date, adjust)


        if df is not None and not df.empty:
            self._cache[cache_key] = df
            self._save_cache(cache_key, df)
            logger.info(f"获取股票数据成功: {stock_code}, {len(df)}条记录")
        else:
            logger.warning(f"获取股票数据失败: {stock_code}")

        return df if df is not None else pd.DataFrame()

    def fetch_stock_list(self, market: str = "a_share") -> List[dict]:
        """
        获取股票列表 (带本地缓存)

        Args:
            market: 市场 "a_share" / "hs300" / "zz500" / "gem"(创业板)

        Returns:
            List of stock info dicts
        """
        cache_key = f"stock_list_{market}"

        # 1. 内存缓存
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        # 2. 文件缓存 (股票列表变化频率低)
        cached_df = self._load_cache(cache_key)
        if cached_df is not None and len(cached_df) > 0:
            stocks = [
                {"code": str(row["code"]), "name": str(row.get("name", ""))}
                for _, row in cached_df.iterrows()
            ]
            self._cache[cache_key] = stocks
            return stocks

        stocks = []

        if market == "a_share":
            stocks = self._get_a_share_list()
        elif market == "hs300":
            stocks = self._get_index_constituents("000300.SH")
        elif market == "zz500":
            stocks = self._get_index_constituents("000905.SH")
        elif market == "gem":
            stocks = self._get_gem_stocks()

        # 保存为 DataFrame 缓存
        df_cache = pd.DataFrame(stocks)
        self._save_cache(cache_key, df_cache)
        self._cache[cache_key] = stocks
        return stocks

    @staticmethod
    def _normalize_calendar_dates(raw_dates, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Normalize, de-duplicate, and bound provider-supplied exchange sessions."""
        dates = pd.DatetimeIndex(pd.to_datetime(raw_dates, errors="coerce"))
        dates = dates[~dates.isna()]
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start.tz is not None:
            start = start.tz_localize(None)
        if end.tz is not None:
            end = end.tz_localize(None)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("calendar start/end must be valid and start must not exceed end")
        dates = dates.normalize().unique().sort_values()
        return dates[(dates >= start.normalize()) & (dates <= end.normalize())]

    def fetch_trading_calendar(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Fetch the exchange trading calendar from AKShare; never fabricate weekdays."""
        self._normalize_calendar_dates([], start_date, end_date)  # validate requested bounds
        try:
            import akshare as ak
            calendar = ak.tool_trade_date_hist_sina()
            if calendar is None or calendar.empty:
                raise ValueError("AKShare returned an empty exchange calendar")
            date_col = "trade_date" if "trade_date" in calendar.columns else calendar.columns[0]
            dates = self._normalize_calendar_dates(calendar[date_col], start_date, end_date)
            if len(dates):
                return dates
        except Exception as exc:
            logger.warning(f"获取交易所日历失败，尝试基准指数行情日历: {exc}")

        benchmark = self.config.get("benchmark", "000300.SH")
        frame = self.fetch_stock_data(benchmark, start_date=start_date, end_date=end_date, adjust="")
        if frame is None or frame.empty:
            raise RuntimeError("无法获取交易所日历或基准指数交易日数据")
        dates = self._normalize_calendar_dates(frame.index, start_date, end_date)
        if not len(dates):
            raise RuntimeError("基准指数数据中没有请求区间的交易日")
        return dates

    def fetch_market_index(self, index_code: str = "000300.SH") -> pd.DataFrame:
        """获取指数数据 (带缓存)"""
        cache_key = self._get_cache_key(index_code, "2020-01-01", datetime.now().strftime("%Y-%m-%d"), adjust="")
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        df = self.fetch_stock_data(index_code, adjust="")
        if not df.empty:
            self._cache[cache_key] = df
        return df

    def fetch_financial_data(self, stock_code: str) -> pd.DataFrame:
        """获取基本面财务数据 (带缓存)"""
        cache_key = f"financial_{stock_code}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        try:
            import akshare as ak
            df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"获取财务数据失败: {stock_code}, 错误: {e}")
            return pd.DataFrame()

    def fetch_industry_data(self) -> pd.DataFrame:
        """获取行业分类数据 (带缓存)"""
        cache_key = "industry_data"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        try:
            import akshare as ak
            df = ak.stock_board_industry_name_em()
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"获取行业数据失败: {e}")
            return pd.DataFrame()

    def fetch_news(self, stock_code: str, days: int = 30) -> List[dict]:
        """获取股票相关新闻 (带缓存)"""
        cache_key = f"news_{stock_code}_{days}"
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        news_list = []
        try:
            import akshare as ak
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            df = ak.stock_news_em(symbol=stock_code)
            if not df.empty:
                for _, row in df.iterrows():
                    news_list.append({
                        "title": row.get("标题", ""),
                        "content": row.get("新闻内容", ""),
                        "url": row.get("新闻链接", ""),
                        "published_at": str(row.get("发布时间", "")),
                        "source": row.get("文章来源", ""),
                    })
        except Exception as e:
            logger.error(f"获取新闻失败: {stock_code}, 错误: {e}")
        self._cache[cache_key] = news_list
        return news_list

    def fetch_bond_data(self, stock_code: str) -> dict:
        """获取质押数据"""
        try:
            import akshare as ak
            df = ak.stock_gdp_by_area(symbol=stock_code)
            return {"pledge_ratio": float(df.iloc[0].get("比率", 0)) if not df.empty else 0}
        except:
            return {"pledge_ratio": 0}

    # ========== 内部方法 ==========

    def _fetch_akshare(self, code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
        """AKShare数据源 (带重试机制 + 自动回退到 yfinance)"""
        df = None
        try:
            import akshare as ak

            for attempt in range(1, self.max_retries + 1):
                try:
                    # 获取日线数据
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""),
                        adjust=adjust
                    )

                    if not df.empty:
                        # 统一列名为英文 (akshare 返回中文列名)
                        col_map = {
                            "日期": "date",
                            "开盘": "open",
                            "收盘": "close",
                            "最高": "high",
                            "最低": "low",
                            "成交量": "volume",
                            "成交额": "amount",
                            "振幅": "amplitude",
                            "涨跌幅": "pct_change",
                            "涨跌额": "change",
                            "换手率": "turnover",
                        }
                        df = df.rename(columns=col_map)
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        return df

                    logger.warning(f"AKShare返回空数据: {code} (尝试 {attempt}/{self.max_retries})")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                    continue

                except Exception as conn_err:
                    logger.debug(f"AKShare连接失败: {code} 尝试{attempt}: {conn_err}")
                    if attempt < self.max_retries:
                        # 指数退避：每次等待时间递增
                        time.sleep(self.retry_delay * attempt)
                    else:
                        # 所有重试失败，返回空 DataFrame 触发回退
                        return pd.DataFrame()

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"AKShare获取数据失败: {code}, 错误: {e}")
            return pd.DataFrame()

    def _fetch_yfinance(self, code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
        """yfinance数据源"""
        try:
            import yfinance as yf

            # A股代码转换
            if code.startswith("6"):
                yf_code = f"{code}.SS"  # 上证
            else:
                yf_code = f"{code}.SZ"  # 深证

            df = yf.download(yf_code, start=start, end=end, auto_adjust=(adjust != ""))

            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            df.index.name = "date"
            return df.reset_index()

        except Exception as e:
            logger.error(f"yfinance获取数据失败: {code}, 错误: {e}")
            return pd.DataFrame()

    def _fetch_csv(self, code: str, start: str, end: str) -> pd.DataFrame:
        """CSV本地数据源"""
        csv_dir = Path("../data/csv")
        csv_path = csv_dir / f"{code}.csv"

        if not csv_path.exists():
            logger.warning(f"CSV文件不存在: {csv_path}")
            return pd.DataFrame()

        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()

        # 日期过滤
        mask = (df.index >= start) & (df.index <= end)
        return df[mask]


    def _fetch_tushare_mcp(self, code: str, start: str, end: str, adjust: str = "") -> pd.DataFrame:
        """Tushare MCP 数据源 (通过 HTTP API)"""
        import json as json_lib
        import requests as req_lib

        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            logger.warning("[Tushare] 环境变量 TUSHARE_TOKEN 未设置")
            return pd.DataFrame()

        # 获取配置
        try:
            from utils.config_loader import ConfigLoader
            cfg = ConfigLoader()
            t_cfg = cfg.get("data.tushare_mcp", {})
            url = t_cfg.get("url", "https://api.tushare.pro/mcp/")
            timeout = t_cfg.get("timeout", 30)
            retry_delay = t_cfg.get("retry_delay", 4)
        except Exception:
            url = "https://api.tushare.pro/mcp/"
            timeout = 30
            retry_delay = 4

        # 股票代码转换：AKShare格式 → Tushare格式
        if "." in code:
            ts_code = code
        elif code.startswith("6"):
            ts_code = f"{code}.SH"
        elif code.startswith("9") or code.startswith("4"):
            ts_code = f"{code}.BJ"
        else:
            ts_code = f"{code}.SZ"

        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "")

        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time()),
            "method": "tools/call",
            "params": {
                "name": "daily",
                "args": {
                    "ts_code": ts_code,
                    "start_date": start_fmt,
                    "end_date": end_fmt,
                }
            }
        }

        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

        for attempt in range(1, 4):
            try:
                resp = req_lib.post(url, json=payload, headers=headers, timeout=timeout)
                if resp.status_code != 200:
                    logger.warning(f"[Tushare] HTTP {resp.status_code}: {resp.text[:200]}")
                    continue

                text = resp.text.strip()
                # MCP SSE 格式：event: message\ndata: {...}
                if "data:" in text:
                    data_str = text.split("data:", 1)[1].strip()
                else:
                    data_str = text

                result = json_lib.loads(data_str)

                # 检查是否有错误
                if "error" in result:
                    err_msg = result["error"].get("message", "")
                    logger.warning(f"[Tushare] API错误 [{result['error'].get('code', '')}]: {err_msg}")
                    return pd.DataFrame()

                content = result.get("result", {}).get("content", [])
                if not content:
                    logger.debug(f"[Tushare] 返回空内容: {code}")
                    return pd.DataFrame()

                # 解析文本格式的数据
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_content += item["text"]
                    elif isinstance(item, str):
                        text_content += item

                if not text_content or "tushare API 错误" in text_content:
                    logger.warning(f"[Tushare] {code}: {text_content[:200]}")
                    return pd.DataFrame()

                # Tushare 返回的是表格文本，解析为 DataFrame
                df = self._parse_tushare_table(text_content)
                if df is not None and not df.empty:
                    # 统一列名和格式 (与 AKShare 一致)
                    col_map = {
                        "ts_code": "ts_code",
                        "trade_date": "date",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "pre_close": "pre_close",
                        "change": "change",
                        "pct_chg": "pct_change",
                        "vol": "volume",
                        "amount": "amount",
                    }
                    df = df.rename(columns=col_map)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        # 只保留标准列
                        keep_cols = ["open", "high", "low", "close", "volume", "amount", "pct_change", "change"]
                        available_cols = [c for c in keep_cols if c in df.columns]
                        return df[available_cols]

            except Exception as e:
                logger.warning(f"[Tushare] 请求失败 [{attempt}/3]: {e}")
                if attempt < 3:
                    time.sleep(retry_delay * attempt)
                continue

        return pd.DataFrame()

    def _parse_tushare_table(self, text: str) -> Optional[pd.DataFrame]:
        """解析 Tushare 返回的表格文本为 DataFrame"""
        import io

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return None

        first_line = lines[0]
        if "\t" in first_line:
            sep = "\t"
        elif "|" in first_line:
            sep = "|"
        else:
            sep = None

        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", header=0)
            return df
        except Exception:
            logger.debug(f"[Tushare] 表格解析失败")
            return None

    def _get_a_share_list(self) -> List[dict]:
        """获取A股列表"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            stocks = []
            for _, row in df.iterrows():
                stocks.append({
                    "code": str(row.get("代码", "")),
                    "name": row.get("名称", ""),
                    "price": float(row.get("最新价", 0)),
                    "pct_change": float(row.get("涨跌幅", 0)),
                    "volume": float(row.get("成交量", 0)),
                    "amount": float(row.get("成交额", 0)),
                    "industry": row.get("所属板块", ""),
                    "market_cap": float(row.get("总市值", 0)),
                })
            return stocks
        except Exception as e:
            logger.error(f"获取A股列表失败: {e}")
            return []

    def _get_index_constituents(self, index_code: str) -> List[dict]:
        """获取指数成分股"""
        try:
            import akshare as ak
            if index_code == "000300.SH":
                df = ak.index_stock_cons(symbol="000300")
            elif index_code == "000905.SH":
                df = ak.index_stock_cons(symbol="000905")
            else:
                df = ak.index_stock_cons(symbol=index_code)

            # 兼容不同版本的列名 (品种代码/成分券代码)
            code_col = "成分券代码" if "成分券代码" in df.columns else "品种代码"
            name_col = "成分券名称" if "成分券名称" in df.columns else "品种名称"

            stocks = []
            for _, row in df.iterrows():
                code = str(row.get(code_col, ""))
                name = row.get(name_col, "")
                if code:
                    stocks.append({"code": code, "name": name})
            return stocks
        except Exception as e:
            logger.error(f"获取指数成分股失败: {index_code}, 错误: {e}")
            return []

    def _get_gem_stocks(self) -> List[dict]:
        """获取创业板股票"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            gem_df = df[df["代码"].str.startswith("30")]
            return [
                {"code": str(row["代码"]), "name": row["名称"], "market": "gem"}
                for _, row in gem_df.iterrows()
            ]
        except Exception as e:
            logger.error(f"获取创业板股票失败: {e}")
            return []

    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache.clear()
        for f in self._cache_dir.glob("*.pkl"):
            f.unlink()
