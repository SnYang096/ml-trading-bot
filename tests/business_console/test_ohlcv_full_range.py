"""Full-range OHLCV loads entire bars_1min parquet (within max_days cap)."""

from __future__ import annotations

import pandas as pd

from mlbot_console.services.ohlcv_reader import (
    bars_1min_bounds,
    fetch_ohlcv,
    load_bars_1min,
)


def test_full_range_loads_all_fixture_bars(bus_root):
    path = bus_root / "bars_1min" / "ETHUSDT.parquet"
    _, _, rows = bars_1min_bounds(path)
    data = fetch_ohlcv(bus_root, "ETHUSDT", "2h", max_days=180, full_range=True)
    raw = load_bars_1min(bus_root, "ETHUSDT")
    resampled_count = len(data["candles"])
    assert rows == len(raw)
    assert resampled_count >= 12
    assert data["bars_1min_rows"] == rows


def test_explicit_range_still_works(bus_root):
    data = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-01 12:00", tz="UTC"),
        max_days=90,
        full_range=False,
    )
    assert len(data["candles"]) >= 1
