"""OHLCV: live_storage archive + feature-bus tail."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from mlbot_console.services.ohlcv_reader import fetch_ohlcv, stitch_live_storage_and_bus


@pytest.fixture
def stitched_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Archive: Jan 1–5; bus snapshot: Jan 6–7 only (2 days)."""
    sym = "ETHUSDT"
    bars_root = tmp_path / "bars" / sym
    bars_root.mkdir(parents=True)
    for day in range(5):
        d = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
        rows = [
            {
                "timestamp": d + pd.Timedelta(hours=h),
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.5,
                "volume": 1.0,
            }
            for h in range(24)
        ]
        pd.DataFrame(rows).to_parquet(
            bars_root / f"{d.strftime('%Y-%m-%d')}.parquet", index=False
        )

    bus_root = tmp_path / "bus"
    bus_dir = bus_root / "bars_1min"
    bus_dir.mkdir(parents=True)
    bus_rows = []
    for day in (5, 6):
        d = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
        for i in range(24 * 60):
            ts = d + pd.Timedelta(minutes=i)
            bus_rows.append(
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 100.5,
                    "low": 99.5,
                    "close": 100.1,
                    "volume": 10.0,
                }
            )
    pd.DataFrame(bus_rows).to_parquet(bus_dir / f"{sym}.parquet", index=False)
    return bus_root, tmp_path / "bars"


def test_stitch_merge_dedupes_bus_wins():
    t0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    hist = pd.DataFrame(
        {
            "timestamp": [t0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    bus = pd.DataFrame(
        {
            "timestamp": [t0],
            "open": [2.0],
            "high": [2.0],
            "low": [2.0],
            "close": [2.0],
            "volume": [2.0],
        }
    )
    out = stitch_live_storage_and_bus(hist, bus)
    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 2.0


def test_fetch_ohlcv_stitch_more_2h_bars_than_bus_only(stitched_roots):
    bus_root, live_bars_root = stitched_roots
    bus_only = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        max_days=30,
        full_range=True,
        stitch_live_storage=False,
    )
    stitched = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        max_days=30,
        full_range=True,
        live_storage_bars_root=live_bars_root,
        stitch_live_storage=True,
    )
    assert bus_only["source"] == "bars_1min"
    assert stitched["source"] == "live_storage+bars_1min"
    assert stitched["live_storage_1m_rows"] > 0
    assert len(stitched["candles"]) > len(bus_only["candles"])


def test_fetch_ohlcv_stitch_with_explicit_window(stitched_roots):
    """Windowed from/to must still merge live_storage archive (regression)."""
    bus_root, live_bars_root = stitched_roots
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-07 23:59", tz="UTC")
    bus_only = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        start=start,
        end=end,
        max_days=30,
        full_range=False,
        stitch_live_storage=False,
    )
    stitched = fetch_ohlcv(
        bus_root,
        "ETHUSDT",
        "2h",
        start=start,
        end=end,
        max_days=30,
        full_range=False,
        live_storage_bars_root=live_bars_root,
        stitch_live_storage=True,
    )
    assert stitched["source"] == "live_storage+bars_1min"
    assert len(stitched["candles"]) > len(bus_only["candles"])
