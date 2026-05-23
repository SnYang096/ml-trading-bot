"""Main chart slow MA overlays (EMA1200, weekly EMA200)."""

from __future__ import annotations

import pandas as pd
import pytest

from mlbot_console.services.main_chart_overlays import (
    _align_ma_to_candles,
    load_main_chart_overlays,
)
from mlbot_console.services.ohlcv_reader import resolve_trade_map_window


def test_resolve_trade_map_window_defaults(bus_root):
    start, end, full = resolve_trade_map_window("2h", full_range=False)
    assert full is False
    assert start is not None
    assert (end - start).days >= 59


def test_align_ma_to_candles_merges_second_and_nanosecond_timestamps() -> None:
    """Parquet features are ns; candle times from unit=s must still merge_asof."""
    t0 = pd.Timestamp("2024-06-01 00:00", tz="UTC")
    feat = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([t0, t0 + pd.Timedelta(hours=2)], utc=True),
            "close": [100.0, 101.0],
            "ema_1200_position": [0.01, 0.02],
        }
    )
    candles = [
        {"time": int(t0.timestamp())},
        {"time": int((t0 + pd.Timedelta(hours=2)).timestamp())},
    ]
    points = _align_ma_to_candles(feat, "ema_1200_position", candles)
    assert len(points) == 2
    assert points[0]["value"] == pytest.approx(99.0)
    assert points[1]["value"] == pytest.approx(98.98)


def test_load_main_overlays_aligns_to_candles(bus_root):
    candles = [
        {"time": int(pd.Timestamp("2024-01-01 10:00", tz="UTC").timestamp())},
        {"time": int(pd.Timestamp("2024-01-01 14:00", tz="UTC").timestamp())},
    ]
    out = load_main_chart_overlays(
        bus_root,
        "ETHUSDT",
        candles,
        ["ema_1200", "weekly_ema_200"],
    )
    assert out["ema_1200"]["available"]
    assert len(out["ema_1200"]["points"]) == 2
    assert out["ema_1200"]["points"][0]["value"] > 90
