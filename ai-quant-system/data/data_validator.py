"""
AI量化交易系统 - 数据完整性验证模块
对比 AKShare 和 Tushare MCP 双源数据的一致性
"""
import os
import time
import json
import requests  # 可选依赖，仅在启用 Tushare MCP 时需要
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ValidationReport:
    """数据完整性验证报告"""
    stock_code: str
    start_date: str
    end_date: str
    source_akshare_rows: int = 0
    source_tushare_rows: int = 0
    common_dates: int = 0
    missing_in_akshare: List[str] = field(default_factory=list)   # Tushare有但AKShare缺失的日期
    missing_in_tushare: List[str] = field(default_factory=list)   # AKShare有但Tushare缺失的日期
    price_diff_count: int = 0       # 收盘价差异超过阈值的记录数
    volume_diff_count: int = 0      # 成交量差异超过阈值的记录数
    max_price_diff_pct: float = 0.0
    max_volume_diff_pct: float = 0.0
    passed: bool = True             # 整体是否通过校验
    details: List[str] = field(default_factory=list)


class DataValidator:
    """
    数据完整性验证器
    对比 AKShare 和 Tushare MCP 双源数据的一致性
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.tolerance = self.config.get("tolerance", {})
        self.close_pct_tol = self.tolerance.get("close_pct", 0.005)      # 收盘价 ±0.5%
        self.volume_pct_tol = self.tolerance.get("volume_pct", 0.01)     # 成交量 ±1%
        self.open_pct_tol = self.tolerance.get("open_pct", 0.005)       # 开盘价 ±0.5%
        self.high_low_pct_tol = self.tolerance.get("high_low_pct", 0.005)  # 最高/最低价 ±0.5%
        self.auto_save = self.config.get("auto_save", True)
        self.sample_size = self.config.get("sample_size", 10)

    def validate_single_stock(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        source_akshare_df=None,
        source_tushare_df=None,
    ) -> ValidationReport:
        """
        验证单只股票的双源数据一致性

        Args:
            stock_code: 股票代码，如 "000001"
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"
            source_akshare_df: AKShare DataFrame（可选，传入已缓存的数据可避免重复请求）
            source_tushare_df: Tushare DataFrame（可选，传入已缓存的数据可避免重复请求）

        Returns:
            ValidationReport
        """
        report = ValidationReport(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
        )

        if not self.enabled:
            logger.debug("数据校验已禁用，跳过验证")
            report.passed = True
            return report

        # 1. 获取 AKShare 数据（如果未传入）
        if source_akshare_df is None:
            from .fetcher import DataFetcher
            fetcher = DataFetcher()
            source_akshare_df = fetcher.fetch_stock_data(
                stock_code, start_date=start_date, end_date=end_date, source="akshare"
            )

        # 2. 获取 Tushare 数据（如果未传入）
        if source_tushare_df is None:
            source_tushare_df = self._fetch_tushare_daily(stock_code, start_date, end_date)

        report.source_akshare_rows = len(source_akshare_df) if source_akshare_df is not None and not source_akshare_df.empty else 0
        report.source_tushare_rows = len(source_tushare_df) if source_tushare_df is not None and not source_tushare_df.empty else 0

        # 3. 数据不足则跳过校验
        if report.source_akshare_rows == 0 or report.source_tushare_rows == 0:
            report.details.append(f"数据不足: AKShare={report.source_akshare_rows}条, Tushare={report.source_tushare_rows}条")
            logger.warning(f"[校验] {stock_code}: 数据不足，跳过校验")
            return report

        # 4. 统一日期索引进行对比
        ak_dates = set()
        ts_dates = set()

        if "date" in source_akshare_df.columns:
            ak_dates = set(source_akshare_df["date"].dt.strftime("%Y-%m-%d"))
        elif source_akshare_df.index.name == "date":
            ak_dates = set(source_akshare_df.index.strftime("%Y-%m-%d"))

        if "trade_date" in source_tushare_df.columns:
            ts_dates = set(pd.to_datetime(source_tushare_df["trade_date"]).dt.strftime("%Y-%m-%d"))
        elif source_tushare_df.index.name == "trade_date":
            ts_dates = set(source_tushare_df.index.strftime("%Y-%m-%d"))

        common_dates = ak_dates & ts_dates
        report.common_dates = len(common_dates)
        report.missing_in_akshare = sorted(ts_dates - ak_dates)
        report.missing_in_tushare = sorted(ak_dates - ts_dates)

        # 5. 对比价格/成交量（仅对共同日期）
        if common_dates:
            price_diff_count, volume_diff_count, max_price_pct, max_volume_pct = self._compare_data(
                source_akshare_df, source_tushare_df, common_dates
            )
            report.price_diff_count = price_diff_count
            report.volume_diff_count = volume_diff_count
            report.max_price_diff_pct = max_price_pct
            report.max_volume_diff_pct = max_volume_pct

        # 6. 判定是否通过
        issues = []
        if report.missing_in_akshare:
            issues.append(f"AKShare缺失{len(report.missing_in_akshare)}天")
        if report.missing_in_tushare:
            issues.append(f"Tushare缺失{len(report.missing_in_tushare)}天")
        if report.price_diff_count > 0:
            issues.append(f"收盘价差异>{self.close_pct_tol*100:.1f}%的记录有{report.price_diff_count}条")
        if report.volume_diff_count > 0:
            issues.append(f"成交量差异>{self.volume_pct_tol*100:.1f}%的记录有{report.volume_diff_count}条")

        if issues:
            report.passed = False
            report.details.extend(issues)
            logger.warning(f"[校验] {stock_code}: 未通过 - {'; '.join(issues)}")
        else:
            report.passed = True
            logger.info(f"[校验] {stock_code}: ✅ 通过 (共同日期={report.common_dates}天)")

        return report

    def validate_batch(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        sample_size: int = None,
    ) -> Dict[str, ValidationReport]:
        """
        批量验证多只股票的数据一致性

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"
            sample_size: 抽样数量，None表示全部

        Returns:
            {stock_code: ValidationReport}
        """
        if sample_size and len(stock_codes) > sample_size:
            import random
            sampled = random.sample(stock_codes, min(sample_size, len(stock_codes)))
            logger.info(f"[校验] 批量验证抽样: {len(stock_codes)}只 → {len(sampled)}只")
        else:
            sampled = stock_codes

        results = {}
        passed_count = 0
        failed_count = 0

        for code in sampled:
            report = self.validate_single_stock(code, start_date, end_date)
            results[code] = report
            if report.passed:
                passed_count += 1
            else:
                failed_count += 1

        logger.info(f"[校验] 批量验证完成: 通过={passed_count}, 失败={failed_count}, 总计={len(sampled)}")
        return results

    def _compare_data(
        self,
        ak_df: pd.DataFrame,
        ts_df: pd.DataFrame,
        common_dates: set,
    ) -> Tuple[int, int, float, float]:
        """对比双源数据的价格和成交量"""
        price_diff_count = 0
        volume_diff_count = 0
        max_price_pct = 0.0
        max_volume_pct = 0.0

        # 构建 AKShare 查找字典 {date: row}
        ak_lookup = {}
        if "date" in ak_df.columns:
            for _, row in ak_df.iterrows():
                d = row["date"].strftime("%Y-%m-%d")
                ak_lookup[d] = row
        elif ak_df.index.name == "date":
            for idx, row in ak_df.iterrows():
                d = idx.strftime("%Y-%m-%d")
                ak_lookup[d] = row

        # 构建 Tushare 查找字典 {trade_date: row}
        ts_lookup = {}
        if "trade_date" in ts_df.columns:
            for _, row in ts_df.iterrows():
                d = pd.to_datetime(row["trade_date"]).strftime("%Y-%m-%d")
                ts_lookup[d] = row
        elif ts_df.index.name == "trade_date":
            for idx, row in ts_df.iterrows():
                d = idx.strftime("%Y-%m-%d")
                ts_lookup[d] = row

        for date_str in sorted(common_dates):
            ak_row = ak_lookup.get(date_str)
            ts_row = ts_lookup.get(date_str)
            if not ak_row or not ts_row:
                continue

            # 对比收盘价
            ak_close = float(ak_row.get("close", 0))
            ts_close = float(ts_row.get("close", 0))
            if ak_close > 0 and ts_close > 0:
                close_diff_pct = abs(ak_close - ts_close) / ak_close
                max_price_pct = max(max_price_pct, close_diff_pct)
                if close_diff_pct > self.close_pct_tol:
                    price_diff_count += 1

            # 对比成交量
            ak_vol = float(ak_row.get("volume", 0))
            ts_vol = float(ts_row.get("vol", ts_row.get("amount", 0)))
            if ak_vol > 0 and ts_vol > 0:
                vol_diff_pct = abs(ak_vol - ts_vol) / ak_vol
                max_volume_pct = max(max_volume_pct, vol_diff_pct)
                if vol_diff_pct > self.volume_pct_tol:
                    volume_diff_count += 1

        return price_diff_count, volume_diff_count, max_price_pct, max_volume_pct

    def _fetch_tushare_daily(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        通过 Tushare MCP HTTP API 获取日线数据

        Args:
            stock_code: 股票代码，如 "000001" → 转为 "000001.SZ"
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"

        Returns:
            DataFrame with columns: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        """
        import pandas as pd

        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            logger.warning("[Tushare] 环境变量 TUSHARE_TOKEN 未设置")
            return None

        # 获取配置中的 URL（如果存在）
        try:
            from utils.config_loader import ConfigLoader
            cfg = ConfigLoader()
            tushare_cfg = cfg.get("data.tushare_mcp", {})
            url = tushare_cfg.get("url", "https://api.tushare.pro/mcp/")
            timeout = tushare_cfg.get("timeout", 30)
            retry_delay = tushare_cfg.get("retry_delay", 4)
        except Exception:
            url = "https://api.tushare.pro/mcp/"
            timeout = 30
            retry_delay = 4

        # 股票代码转换：AKShare格式 → Tushare格式
        ts_code = self._code_to_tushare(stock_code)

        # 日期格式转换：YYYY-MM-DD → YYYYMMDD
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")

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
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                if resp.status_code != 200:
                    logger.warning(f"[Tushare] HTTP {resp.status_code}: {resp.text[:200]}")
                    continue

                text = resp.text.strip()
                # MCP SSE 格式：event: message\ndata: {...}
                if "data:" in text:
                    data_str = text.split("data:", 1)[1].strip()
                else:
                    data_str = text

                result = json.loads(data_str)

                # 检查是否有错误
                if "error" in result:
                    err_msg = result["error"].get("message", "")
                    logger.warning(f"[Tushare] API错误 [{result['error'].get('code', '')}]: {err_msg}")
                    return None

                content = result.get("result", {}).get("content", [])
                if not content:
                    logger.debug(f"[Tushare] 返回空内容: {stock_code}")
                    return pd.DataFrame()

                # 解析文本格式的数据
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_content += item["text"]
                    elif isinstance(item, str):
                        text_content += item

                if not text_content or "tushare API 错误" in text_content:
                    logger.warning(f"[Tushare] {stock_code}: {text_content[:200]}")
                    return None

                # Tushare 返回的是表格文本，解析为 DataFrame
                df = self._parse_tushare_table(text_content)
                if df is not None and not df.empty:
                    logger.debug(f"[Tushare] 获取 {stock_code}: {len(df)}条记录")
                    return df

            except Exception as e:
                logger.warning(f"[Tushare] 请求失败 [{attempt}/3]: {e}")
                if attempt < 3:
                    time.sleep(retry_delay * attempt)
                continue

        return pd.DataFrame()

    def _parse_tushare_table(self, text: str) -> Optional[pd.DataFrame]:
        """解析 Tushare 返回的表格文本为 DataFrame"""
        import pandas as pd
        import io

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return None

        # 尝试检测分隔符（空格、制表符等）
        first_line = lines[0]
        if "\t" in first_line:
            sep = "\t"
        elif "|" in first_line:
            sep = "|"
        else:
            # 多空格分隔
            sep = None

        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", header=0)
            return df
        except Exception:
            logger.debug(f"[Tushare] 表格解析失败，返回原始文本")
            return None

    def _code_to_tushare(self, code: str) -> str:
        """AKShare股票代码 → Tushare格式"""
        if "." in code:
            return code  # 已经是 Tushare 格式
        if code.startswith("6"):
            return f"{code}.SH"  # 上交所
        elif code.startswith("9") or code.startswith("4"):
            return f"{code}.BJ"  # 北交所
        else:
            return f"{code}.SZ"  # 深证

    def save_validation_report(
        self,
        reports: Dict[str, ValidationReport],
        output_dir: str = None,
    ) -> str:
        """保存验证报告到本地"""
        if not self.auto_save:
            return ""

        if output_dir is None:
            output_dir = "../output/validation"

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        report_file = path / f"validation_{date_str}.json"

        summary = {
            "date": datetime.now().isoformat(),
            "total": len(reports),
            "passed": sum(1 for r in reports.values() if r.passed),
            "failed": sum(1 for r in reports.values() if not r.passed),
            "stocks": {}
        }

        for code, report in reports.items():
            summary["stocks"][code] = {
                "start_date": report.start_date,
                "end_date": report.end_date,
                "source_akshare_rows": report.source_akshare_rows,
                "source_tushare_rows": report.source_tushare_rows,
                "common_dates": report.common_dates,
                "missing_in_akshare_count": len(report.missing_in_akshare),
                "missing_in_tushare_count": len(report.missing_in_tushare),
                "price_diff_count": report.price_diff_count,
                "volume_diff_count": report.volume_diff_count,
                "max_price_diff_pct": round(report.max_price_diff_pct, 6),
                "max_volume_diff_pct": round(report.max_volume_diff_pct, 6),
                "passed": report.passed,
            }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[校验] 验证报告已保存: {report_file}")
        return str(report_file)


# 需要 pandas 在模块级别导入（避免循环依赖）
import pandas as pd