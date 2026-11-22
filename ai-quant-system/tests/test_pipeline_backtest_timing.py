from types import SimpleNamespace

import pandas as pd

from pipeline import QuantPipeline


class _FakeFetcher:
    primary_source = "csv"

    def fetch_trading_calendar(self, start_date, end_date):
        return pd.DatetimeIndex(["2024-01-02", "2024-01-03"])

    def fetch_stock_data(self, code, start_date, end_date, adjust=""):
        return pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [10.2, 10.2],
                "low": [9.8, 9.8],
                "close": [10.0, 11.0],
                "volume": [100000, 100000],
            },
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        )


def test_pipeline_executes_close_signal_at_next_session_open():
    pipeline = QuantPipeline.__new__(QuantPipeline)
    pipeline.backtest_config = {
        "initial_capital": 100000,
        "commission": 0,
        "min_commission": 0,
        "stamp_tax": 0,
        "slippage": 0,
        "volume_unit": "shares",
        "max_volume_participation": 1.0,
    }
    pipeline.screening = SimpleNamespace(data_fetcher=_FakeFetcher())
    pipeline.log = SimpleNamespace(info=lambda *args, **kwargs: None)
    pipeline.trade_history = []
    pipeline.performance_history = []
    pipeline._get_backtest_target_weights = lambda signal_date, stock_pool: {
        "600000": 0.5
    }
    pipeline._save_backtest_results = lambda results, metrics: None

    result = pipeline.run_backtest("2024-01-02", "2024-01-03")

    first_day, second_day = result["results"]
    assert first_day["date"] == "2024-01-02"
    assert first_day["trades"] == []
    assert second_day["date"] == "2024-01-03"
    assert len(second_day["trades"]) == 1
    trade = second_day["trades"][0]
    assert trade["date"] == "2024-01-03"
    assert trade["side"] == "buy"
    assert trade["reference_price"] == 10.0
    assert trade["fill_price"] == 10.0
    assert second_day["positions"]["600000"]["shares"] == 5000
