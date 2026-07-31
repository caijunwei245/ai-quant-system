"""
沪深300成分股数据下载脚本
使用 AKShare 真实接口，下载成分股列表及历史K线数据
"""
import os
import time
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_hs300_constituents() -> list:
    """
    获取沪深300成分股列表
    返回: [{"code": "600519", "name": "贵州茅台"}, ...]
    """
    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol="000300")

        code_col = "成分券代码" if "成分券代码" in df.columns else "品种代码"
        name_col = "成分券名称" if "成分券名称" in df.columns else "品种名称"

        stocks = []
        for _, row in df.iterrows():
            code = str(row.get(code_col, "")).strip()
            name = str(row.get(name_col, "")).strip()
            if code:
                stocks.append({"code": code, "name": name})

        print(f"✅ 成功获取沪深300成分股列表，共 {len(stocks)} 只股票")
        return stocks

    except ImportError:
        print("错误: 请先安装 akshare: pip install akshare")
        return []
    except Exception as e:
        print(f"❌ 获取成分股列表失败: {e}")
        return []


def download_stock_history(
    stock_code: str,
    stock_name: str,
    save_dir: str,
    max_retries: int = 2,
    timeout: int = 10,
) -> bool:
    """
    下载单只股票的历史日K线数据并保存为CSV

    Args:
        stock_code: 股票代码 (如 "600519")
        stock_name: 股票名称 (用于进度显示)
        save_dir: CSV 保存目录
        max_retries: 最大重试次数
        timeout: 请求超时时间(秒)

    Returns:
        bool: 是否成功
    """
    import akshare as ak

    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="hfq",
                timeout=timeout,
            )

            if df.empty:
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                print(f"  [{stock_name}] 返回空数据，跳过")
                return False

            # 统一列名并保存
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pct_change", "涨跌额": "change",
                "换手率": "turnover",
            }
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            save_path = os.path.join(save_dir, f"{stock_code}.csv")
            df.to_csv(save_path, index=False)
            return True

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)

    return False


def download_historical_data(constituents: list, save_dir: str = None):
    """
    批量下载沪深300成分股历史数据

    Args:
        constituents: 成分股列表 [{code, name}, ...]
        save_dir: 保存目录，默认 DATA_DIR
    """
    if save_dir is None:
        save_dir = DATA_DIR

    total = len(constituents)
    success = 0
    failed = []

    print(f"\n开始批量下载 {total} 只股票的历史数据...")
    print(f"数据保存到: {save_dir}")

    for idx, stock in enumerate(constituents, 1):
        code = stock["code"]
        name = stock["name"]
        progress = f"[{idx}/{total}]"

        print(f"\n{progress} 正在下载 {name} ({code})...", end="", flush=True)

        ok = download_stock_history(code, name, save_dir)

        if ok:
            success += 1
            print(f" ✓")
        else:
            failed.append(f"{code} ({name})")
            print(f" ✗")

        # 避免请求过快导致被限流
        if idx < total:
            time.sleep(0.3)

    # 统计结果
    print(f"\n{'='*40}")
    print(f"下载完成！成功: {success}/{total}")
    if failed:
        print(f"失败: {len(failed)} 只:")
        for s in failed[:10]:
            print(f"  - {s}")
        if len(failed) > 10:
            print(f"  ... 还有 {len(failed) - 10} 只")


def check_market_hours() -> str:
    """检查当前是否在交易时段，返回提示信息"""
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=周一, 6=周日

    if weekday >= 5:
        return "⚠️  当前为周末，不在交易时段，历史数据下载可能受限"

    if hour < 9 or hour >= 17:
        return "⚠️  当前为非交易时段（9:00-17:00），东方财富历史K线接口可能限流"

    return ""


def main():
    print("=" * 45)
    print("  沪深300成分股数据下载器")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 45)

    # 0. 环境提示
    hint = check_market_hours()
    if hint:
        print(hint)
        print()

    # 1. 获取成分股列表
    constituents = fetch_hs300_constituents()
    if not constituents:
        print("未获取到成分股列表，退出。")
        return

    # 2. 保存成分股列表为 CSV
    list_path = os.path.join(DATA_DIR, "hs300_constituents.csv")
    df_list = pd.DataFrame(constituents)
    df_list.to_csv(list_path, index=False)
    print(f"成分股列表已保存: {list_path}")
    print(f"前5只成分股: {[s['name'] for s in constituents[:5]]}")

    # 3. 批量下载历史数据
    download_historical_data(constituents)

    print("\n✅ 全部任务完成！")
    if hint:
        print(f"💡 提示: 历史数据下载因{hint[1:]}，建议在交易时段重新运行以获取K线数据")
    else:
        print("📁 历史K线数据已保存至 data/ 目录")


if __name__ == "__main__":
    main()