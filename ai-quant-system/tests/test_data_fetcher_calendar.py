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
