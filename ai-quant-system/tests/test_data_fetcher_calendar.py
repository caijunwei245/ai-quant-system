import pandas as pd
import pytest

from data.fetcher import DataFetcher


def test_calendar_provider_dates_are_normalized_and_deduplicated():
    dates = DataFetcher._normalize_calendar_dates(
        [
            "2024-02-08 09:30:00+08:00",
            "2024-02-08 15:00:00+08:00",
            "2024-02-19 09:30:00+08:00",
        ],
        "2024-02-08",
        "2024-02-19",
    )
    assert list(dates.strftime("%Y-%m-%d")) == ["2024-02-08", "2024-02-19"]


def test_calendar_provider_rejects_invalid_date_window():
    with pytest.raises(ValueError):
        DataFetcher._normalize_calendar_dates([], "2024-02-20", "2024-02-19")

from types import SimpleNamespace
from unittest.mock import patch
import sys


def test_fetch_trading_calendar_uses_akshare_provider(monkeypatch):
    provider_dates = pd.DataFrame({
        "trade_date": ["2024-02-08", "2024-02-08", "2024-02-19"]
    })
    fake_akshare = SimpleNamespace(
        tool_trade_date_hist_sina=lambda: provider_dates
    )
    with patch.dict(sys.modules, {"akshare": fake_akshare}):
        fetcher = DataFetcher({"primary_source": "csv"})
        result = fetcher.fetch_trading_calendar("2024-02-08", "2024-02-19")
    assert list(result.strftime("%Y-%m-%d")) == ["2024-02-08", "2024-02-19"]


def test_fetch_trading_calendar_falls_back_to_benchmark_sessions():
    fake_akshare = SimpleNamespace(
        tool_trade_date_hist_sina=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )

    class FallbackFetcher(DataFetcher):
        def fetch_stock_data(self, stock_code, start_date, end_date, source=None, adjust="hfq"):
            assert stock_code == "000300.SH"
            return pd.DataFrame(
                {"close": [10.0, 11.0, 12.0]},
                index=pd.DatetimeIndex(["2024-02-08", "2024-02-08", "2024-02-19"]),
            )

    with patch.dict(sys.modules, {"akshare": fake_akshare}):
        fetcher = FallbackFetcher({"benchmark": "000300.SH"})
        result = fetcher.fetch_trading_calendar("2024-02-08", "2024-02-19")
    assert list(result.strftime("%Y-%m-%d")) == ["2024-02-08", "2024-02-19"]
