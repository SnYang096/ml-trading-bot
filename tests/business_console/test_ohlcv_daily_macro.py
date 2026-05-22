"""1d OHLCV from macro spot_klines (not limited to feature-bus 180d window)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mlbot_console.services.macro_spot_daily import MacroSpotDailyLoader
from mlbot_console.services.ohlcv_reader import fetch_ohlcv


def test_macro_loader_reads_monthly(macro_kline_root: Path) -> None:
    loader = MacroSpotDailyLoader(macro_kline_root)
    daily = loader.load_symbol_daily(
        "ETHUSDT",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
    assert len(daily) == 31


def test_fetch_ohlcv_1d_uses_macro_not_180d_cap(
    bus_root: Path, macro_kline_root: Path
) -> None:
    data = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "1d",
        macro_kline_root=macro_kline_root,
        daily_ohlcv_start=date(2024, 1, 1),
        max_daily_ohlcv_days=3650,
        full_range=True,
    )
    assert data["source"] == "macro_spot_klines"
    assert data["row_count"] == 31
    assert len(data["candles"]) == 31


def test_fetch_ohlcv_1w_uses_macro(bus_root: Path, macro_kline_root: Path) -> None:
    data = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "1w",
        macro_kline_root=macro_kline_root,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-02-01", tz="UTC"),
        full_range=False,
    )
    assert data["timeframe"] == "1w"
    assert data["source"] == "macro_spot_klines"
    assert len(data["candles"]) >= 4


def test_fetch_ohlcv_2h_still_bars_1min(bus_root: Path, macro_kline_root: Path) -> None:
    data = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        macro_kline_root=macro_kline_root,
        full_range=True,
        max_days=7,
    )
    assert data["source"] == "bars_1min"
