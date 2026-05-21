"""Main chart slow MA overlays (EMA1200, weekly EMA200)."""

from __future__ import annotations

import pandas as pd

from mlbot_console.services.main_chart_overlays import load_main_chart_overlays
from mlbot_console.services.ohlcv_reader import resolve_trade_map_window


def test_resolve_trade_map_window_defaults(bus_root):
    start, end, full = resolve_trade_map_window("2h", full_range=False)
    assert full is False
    assert start is not None
    assert (end - start).days >= 59


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
