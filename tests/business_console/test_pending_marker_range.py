"""Pending markers must survive OHLCV window filtering and align to visible bars."""

from __future__ import annotations

import pandas as pd

from mlbot_console.services.trade_markers import (
    align_pending_markers_to_candles,
    collect_markers,
)


def test_pending_included_outside_ohlcv_window(trend_db, spot_db, multi_leg_db):
    """Fixture pending orders are 2024-01-*; chart window is 2026 only."""
    start = int(pd.Timestamp("2026-05-18", tz="UTC").timestamp())
    end = int(pd.Timestamp("2026-05-20", tz="UTC").timestamp())
    without = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
        start_ts=start,
        end_ts=end,
        include_pending=False,
    )
    assert all(m.get("status") != "pending" for m in without)

    with_pending = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
        start_ts=start,
        end_ts=end,
        include_pending=True,
    )
    pending = [m for m in with_pending if m.get("status") == "pending"]
    assert len(pending) >= 2


def test_align_pending_to_last_candle():
    candle_times = [100, 200, 300]
    markers = [
        {
            "id": "trend:orders:1",
            "time": 50,
            "status": "pending",
            "scope": "trend",
            "event": "entry",
        }
    ]
    out = align_pending_markers_to_candles(markers, candle_times)
    assert out[0]["time"] == 300
    assert out[0]["detail"]["order_time"] == 50
