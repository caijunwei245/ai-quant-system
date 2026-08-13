"""
CSI300 日线数据面板加载器
读取已准备好的 "csi300_daily_2010_2026.csv(.gz)" 宽/长表面板，
转换为初筛层 (ScreeningLayer) 可直接使用的逐股票 DataFrame。

面板字段:
    code, name, date, open, high, low, close, volume, amount, turnover
其中 code 形如 "sh600000" / "sz000001"，会自动归一化为 6 位纯数字代码。
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path


def normalize_code(code) -> str:
    """将 "sh600000" / "sz000001" / "600000" 归一化为 6 位代码。"""
    s = str(code).strip().lower()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits


def load_panel(path) -> pd.DataFrame:
    """
    加载 CSI300 日线面板 (支持 .gz / .csv)。

    Returns:
        DataFrame, 列: code, name, date, open, high, low, close, volume, amount, turnover
        date 已解析为 datetime, code 已归一化为 6 位字符串。
    """
    path = Path(path)
    if path.suffix == ".gz":
        compression = "gzip"
    elif str(path).endswith(".zip"):
        compression = "zip"
    else:
        compression = None

    df = pd.read_csv(path, compression=compression)
    # 去除列名中的 BOM / 空白
    df.columns = [c.strip().lstrip("﻿").strip() for c in df.columns]

    required = {"code", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"面板缺少必要列: {missing}; 现有列: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].map(normalize_code)
    return df


def build_stock_frame(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """从面板中抽取单只股票的日线 DataFrame (以 date 为索引)。"""
    cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"] if c in df.columns]
    sub = (
        df[df["code"] == code][cols]
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .set_index("date")
        .sort_index()
    )
    return sub


def iter_stock_frames(df: pd.DataFrame, top_n: int | None = None):
    """
    逐股票产出 (code, frame) 迭代器。
    仅返回历史长度 >= 60 个交易日的股票。
    """
    codes = list(df["code"].dropna().unique())
    if top_n:
        codes = codes[:top_n]
    for code in codes:
        sdf = build_stock_frame(df, code)
        if len(sdf) >= 60:
            yield code, sdf


def list_universe(df: pd.DataFrame) -> list[dict]:
    """返回面板中的股票池 [{code, name}, ...]。"""
    names = df.drop_duplicates("code").set_index("code")["name"].to_dict()
    return [{"code": c, "name": names.get(c, "")} for c in df["code"].dropna().unique()]
