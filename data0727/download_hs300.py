#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import time
import traceback

print("开始：准备查询并下载沪深300成分股的历史数据（2024-01-01 到 2026-07-21）")

# Ensure akshare is installed
try:
    import akshare as ak
    print("已检测到 akshare 库。")
except Exception as e:
    print("未检测到 akshare，尝试安装 akshare ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "akshare"], stdout=subprocess.DEVNULL)
        print("akshare 安装完成，重新导入 akshare ...")
        import akshare as ak  # try import again
        print("akshare 导入成功。")
    except Exception as ie:
        print("自动安装 akshare 失败。请手动运行：pip install akshare")
        print("错误信息：", ie)
        sys.exit(1)

import pandas as pd

START_DATE = "20240101"  # akshare 常用的 YYYYMMDD 格式
END_DATE = "20260721"
INDEX_CODE = "000300"
OUTPUT_FILENAME = "沪深300成分股.csv"

def get_hs300_components():
    """
    尝试通过 akshare 获取沪深300成分股。
    akshare 的函数在不同版本中名称不同，尝试多种常见接口。
    返回 DataFrame（至少包含代码列）。
    """
    print("查询沪深300成分股列表（指数代码：{}）...".format(INDEX_CODE))
    df = None
    tried = []
    # 几个常见的候选函数名（按优先顺序）
    candidates = [
        ("index_stock_cons_ths", ((INDEX_CODE,), {})),  # ak.index_stock_cons_ths("000300")
        ("index_stock_cons_ths", ((), {"index": INDEX_CODE})),  # ak.index_stock_cons_ths(index="000300")
        ("index_stock_cons_csindex", ((INDEX_CODE,), {})),
        ("index_stock_cons_csindex", ((), {"index": INDEX_CODE})),
        ("index_stock_cons", ((INDEX_CODE,), {})),
        ("index_stock_cons", ((), {"index": INDEX_CODE})),
    ]
    for name, params in candidates:
        func = getattr(ak, name, None)
        if not func:
            tried.append(name + " (不存在)")
            continue
        args, kwargs = params
        try:
            df = func(*args, **kwargs)
            tried.append(name + " (成功调用)")
            if df is not None:
                print("使用 akshare.{} 获取到成分股（rows={}）".format(name, len(df)))
                break
        except Exception as e:
            tried.append(name + " (调用失败: {})".format(e))
            # keep trying others
    if df is None:
        print("尝试的候选函数及结果：")
        for t in tried:
            print("  -", t)
        raise RuntimeError("无法通过 akshare 获取沪深300成分股，请确认 akshare 版本和网络可用性。")
    return df

def normalize_code(raw_code):
    """把各种格式的股票代码标准化为纯6位数字字符串"""
    if pd.isna(raw_code):
        return None
    s = str(raw_code).strip()
    # 一些返回值可能带有后缀，如 .SH/.SZ 或 "sh600000" 等
    s = s.upper().replace(".SH", "").replace(".SZ", "")
    # remove optional prefix
    if s.startswith("SH") or s.startswith("SZ"):
        s = s[2:]
    # ensure 6-digit by zero-fill (defensive)
    if s.isdigit():
        s = s.zfill(6)
    return s

def make_ak_symbol(code):
    """根据 6位代码生成 akshare 要求的 symbol，例如 'sh600000' 或 'sz000001'"""
    if not code or len(code) != 6:
        return None
    if code.startswith(("6",)):
        return "sh" + code
    else:
        return "sz" + code

# 获取成分股
try:
    comps_df = get_hs300_components()
except Exception as e:
    print("获取成分股失败：", e)
    traceback.print_exc()
    sys.exit(1)

# 找到代码列（常见列名：'股票代码','code','代码','ts_code'，以及 akshare 实际返回的 '成分券代码'）
possible_code_cols = [
    "股票代码", "code", "代码", "ts_code", "stock_code",
    "成分券代码", "成份券代码", "券代码", "stock_code", "symbol"
]
code_col = None
for col in comps_df.columns:
    key = str(col).strip()
    if key in possible_code_cols or key.lower() in [c.lower() for c in possible_code_cols]:
        code_col = col
        break
# If not directly found, infer by checking if values look like stock codes
if code_col is None:
    for col in comps_df.columns:
        values = comps_df[col].dropna().astype(str).head(20)
        if values.empty:
            continue
        if values.str.fullmatch(r"\d{6}").any() or values.str.fullmatch(r"[A-Z]{2}\d{6}").any():
            code_col = col
            break

if code_col is None:
    print("无法在成分股表中识别代码列，表头为：", list(comps_df.columns))
    print("请手动检查 akshare 的返回格式。")
    sys.exit(1)

print("识别到代码所在列为：", code_col)

# Extract codes and (optionally) names
codes = []
names = {}
for _, row in comps_df.iterrows():
    raw = row.get(code_col)
    code = normalize_code(raw)
    if not code:
        continue
    # try to get name from common columns
    name = None
    for ncol in ["股票简称", "名称", "简称", "name", "stock_name", "成分券名称", "成份券名称"]:
        if ncol in comps_df.columns:
            name = row.get(ncol)
            break
    names[code] = name
    codes.append(code)

codes = sorted(list(set(codes)))
print("共识别到成分股数量：", len(codes))

# Prepare to fetch historical data
all_dfs = []
failed = []

# Choose historical fetch function: try common akshare functions
hist_func_candidates = ["stock_zh_a_daily", "stock_zh_a_hist", "stock_zh_a_tick_tx", "stock_zh_a_minute", "stock_zh_a_daily_qfq"]

def fetch_history_for_symbol(ak_sym, code):
    """
    尝试通过 akshare 的常见接口获取历史日线数据。
    返回 DataFrame 或 None
    """
    last_exc = None
    # Try stock_zh_a_daily(symbol=..., start_date=..., end_date=...)
    if hasattr(ak, "stock_zh_a_daily"):
        try:
            return ak.stock_zh_a_daily(symbol=ak_sym, start_date=START_DATE, end_date=END_DATE)
        except Exception as e:
            last_exc = e
    # Some akshare versions use stock_zh_a_hist
    if hasattr(ak, "stock_zh_a_hist"):
        try:
            return ak.stock_zh_a_hist(symbol=ak_sym, period="daily", start_date=START_DATE, end_date=END_DATE)
        except Exception as e:
            last_exc = e
    # fallback: try stock_zh_a_daily with adjust param if available (qfq or hfq)
    if hasattr(ak, "stock_zh_a_daily"):
        try:
            return ak.stock_zh_a_daily(symbol=ak_sym, start_date=START_DATE, end_date=END_DATE, adjust="")
        except Exception as e:
            last_exc = e
    # 最后返回 None，并记录最后异常
    if last_exc is not None:
        raise last_exc
    return None

print("开始下载各成分股历史数据（每一步会输出状态，遇到失败会继续下一个）...")
total = len(codes)
for idx, code in enumerate(codes, start=1):
    print("[{}/{}] 处理代码：{}".format(idx, total, code))
    ak_sym = make_ak_symbol(code)
    if not ak_sym:
        print("  无法生成 akshare 符号，跳过：", code)
        failed.append((code, "symbol_generation_failed"))
        continue
    print("  akshare 符号：", ak_sym, "，开始下载历史数据范围：{} - {}".format(START_DATE, END_DATE))
    try:
        df_hist = fetch_history_for_symbol(ak_sym, code)
        if df_hist is None or df_hist.empty:
            print("  警告：未获取到数据或数据为空，可能该股票在该时间段停牌或接口限制。")
            failed.append((code, "no_data"))
            # 继续但也可以选择保存空信息（这里跳过）
            time.sleep(0.3)
            continue
        # 标准化列名并添加代码列，便于后续合并
        # 常见返回列包含 'date'，或者索引为日期。把索引转为列。
        if isinstance(df_hist.index, pd.DatetimeIndex) and 'date' not in df_hist.columns:
            df_hist = df_hist.reset_index()
            # some akshare versions name it '日期' instead of 'date'
        # ensure there's a date column named 'date' or '日期'
        if 'date' not in df_hist.columns and '日期' in df_hist.columns:
            df_hist = df_hist.rename(columns={'日期': 'date'})
        # add code and name columns
        df_hist['stock_code'] = code
        df_hist['stock_name'] = names.get(code)
        # normalize date format to YYYY-MM-DD (if pandas datetime)
        if 'date' in df_hist.columns:
            try:
                df_hist['date'] = pd.to_datetime(df_hist['date']).dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        # append
        all_dfs.append(df_hist)
        print("  成功：获取到 {} 行数据".format(len(df_hist)))
    except Exception as e:
        print("  错误：下载失败，原因：", e)
        # 打印简短的 traceback
        tb = traceback.format_exc(limit=1)
        print("  Traceback（简短）：", tb.splitlines()[-1])
        failed.append((code, str(e)))
    # Respectful delay to avoid被封/速率限制
    time.sleep(0.5)

# 合并并保存
if len(all_dfs) == 0:
    print("未成功下载到任何股票的历史数据。失败列表：", failed)
    sys.exit(1)

print("合并 {} 支股票的数据...".format(len(all_dfs)))
try:
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    # optional: 将列顺序调整为常见顺序（stock_code, stock_name, date, open, high, low, close, volume, ...）
    cols = combined.columns.tolist()
    desired_start = ['stock_code', 'stock_name', 'date']
    rest = [c for c in cols if c not in desired_start]
    combined = combined[desired_start + rest]
except Exception as e:
    print("合并数据出错：", e)
    traceback.print_exc()
    sys.exit(1)

print("保存为本地文件：", OUTPUT_FILENAME)
try:
    # 使用 utf-8-sig 以便 Excel 打开中文文件名/内容不展示 BOM 问题
    combined.to_csv(OUTPUT_FILENAME, index=False, encoding="utf-8-sig")
    print("保存成功，文件路径：", OUTPUT_FILENAME)
except Exception as e:
    print("保存 CSV 失败：", e)
    traceback.print_exc()
    sys.exit(1)

# 总结信息
print("全部完成。")
print("成功下载的股票数量：", len(all_dfs))
print("失败的股票数：", len(failed))
if failed:
    print("失败清单（代码, 原因）：")
    for c, reason in failed:
        print("  -", c, reason)

print("脚本结束。")