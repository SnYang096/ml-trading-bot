"""Main chart slow MA overlays (EMA1200, weekly EMA200)."""

from __future__ import annotations

import pandas as pd
import pytest

from mlbot_console.services.main_chart_overlays import (
    _align_ma_to_candles,
    _align_weekly_ema_seed_to_candles,
    load_main_chart_overlays,
)
from src.live_data_stream.spot_weekly_ema_seed import seed_parquet_path
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


def test_ema1200_overlay_uses_feature_bus_price_column(bus_root) -> None:
    """EMA1200 must plot parquet ema_1200 (~664), not position×close inversion."""
    t0 = pd.Timestamp("2024-01-01 02:00", tz="UTC")
    candles = [{"time": int(t0.timestamp()), "close": 700.0}]
    out = load_main_chart_overlays(
        bus_root,
        "ETHUSDT",
        candles,
        ["ema_1200"],
    )
    assert out["ema_1200"]["available"]
    assert out["ema_1200"]["source"] == "feature_bus_price"
    assert out["ema_1200"]["parquet_column"] == "ema_1200"
    # conftest @ 02:00: close=102.4, ema_1200=close*0.95
    assert out["ema_1200"]["latest"] == pytest.approx(102.4 * 0.95)


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


def test_weekly_ema_overlay_uses_macro_seed_not_bus_close(tmp_path, bus_root) -> None:
    """Seed weekly_ema_200 (~565) must not be inverted with stale 2h bus close (~374)."""
    seed_root = tmp_path / "macro" / "spot_weekly_ema200"
    seed_root.mkdir(parents=True, exist_ok=True)
    week = pd.Timestamp("2024-01-01", tz="UTC")
    pd.DataFrame(
        {
            "week_ts": [week],
            "weekly_ema_200": [565.0],
        }
    ).to_parquet(seed_parquet_path(seed_root, "ETHUSDT"), index=False)

    t_bar = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    candles = [{"time": int(t_bar.timestamp()), "close": 617.0}]
    points = _align_weekly_ema_seed_to_candles(seed_root, "ETHUSDT", candles)
    assert len(points) == 1
    assert points[0]["value"] == pytest.approx(565.0)

    out = load_main_chart_overlays(
        bus_root,
        "ETHUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=seed_root,
    )
    assert out["weekly_ema_200"]["available"]
    assert out["weekly_ema_200"]["source"] == "macro_seed"
    assert out["weekly_ema_200"]["latest"] == pytest.approx(565.0)


def test_weekly_ema_fallback_uses_chart_candle_close(bus_root) -> None:
    """Without seed, invert position with chart close (not 2h bus close)."""
    t0 = pd.Timestamp("2024-01-01 02:00", tz="UTC")
    candles = [{"time": int(t0.timestamp()), "close": 617.0}]
    out = load_main_chart_overlays(
        bus_root,
        "ETHUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=None,
    )
    assert out["weekly_ema_200"]["available"]
    # position -0.05 at first bar => 617 * 1.05 = 647.85
    assert out["weekly_ema_200"]["latest"] == pytest.approx(647.85)
