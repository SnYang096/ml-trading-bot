"""OHLCV reader: bars_1min resampling across UI timeframes."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.ohlcv_reader import (
    OhlcvWindowError,
    fetch_ohlcv,
    load_bars_1min,
    resample_ohlcv,
)


def test_load_bars_1min_filters_range(bus_root):
    df = load_bars_1min(
        bus_root,
        "ETHUSDT",
        start=pd.Timestamp("2024-01-01 12:00", tz="UTC"),
        end=pd.Timestamp("2024-01-01 13:00", tz="UTC"),
    )
    assert len(df) == 61
    assert df["timestamp"].min() >= pd.Timestamp("2024-01-01 12:00", tz="UTC")


@pytest.mark.parametrize("timeframe", ["1min", "15min", "2h", "1d"])
def test_resample_all_timeframes(bus_root, timeframe: str):
    raw = load_bars_1min(bus_root, "ETHUSDT")
    out, degraded = resample_ohlcv(raw, timeframe)
    assert not out.empty
    assert not degraded
    assert {"open", "high", "low", "close"}.issubset(out.columns)


def test_fetch_ohlcv_source_is_bars_1min(bus_root):
    data = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-01 23:59", tz="UTC"),
        max_days=90,
    )
    assert data["source"] == "bars_1min"
    assert len(data["candles"]) >= 1
    assert data["candles"][0]["time"] > 0


def test_fetch_ohlcv_missing_symbol_returns_empty(tmp_path):
    data = fetch_ohlcv(
        tmp_path,
        "NOPE",
        "2h",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-02", tz="UTC"),
        max_days=90,
    )
    assert data["candles"] == []
    assert data["row_count"] == 0


def test_fetch_ohlcv_window_limit(bus_root):
    with pytest.raises(OhlcvWindowError):
        fetch_ohlcv(
            bus_root,
            "ETHUSDT",
            "1min",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-06-01", tz="UTC"),
            max_days=7,
        )
