from __future__ import annotations

import pandas as pd

from src.live_data_stream.classic_feature_bus_provider import (
    ClassicFeatureBusProvider,
)
from src.live_data_stream.feature_bus import FeatureBusWriter


def test_classic_feature_bus_provider_reconstructs_multitimeframe_event(tmp_path):
    writer = FeatureBusWriter(tmp_path, max_rows=10)
    writer.append_bar_1m(
        "BTCUSDT",
        {
            "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        },
    )
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="240T",
        features={"close": 100.5, "atr": 1.2, "bpc_signal": 1.0},
        timestamp=pd.Timestamp("2026-01-01T00:15:00Z"),
    )
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="120T",
        features={"close": 100.5, "srb_signal": 1.0},
        timestamp=pd.Timestamp("2026-01-01T00:15:00Z"),
    )

    provider = ClassicFeatureBusProvider(
        feature_bus_root=tmp_path,
        symbols=["BTCUSDT"],
        primary_timeframe="240T",
        timeframes=["240T", "120T"],
        max_staleness_seconds=0,
    )

    events = provider.poll()

    assert len(events) == 1
    event = events[0]
    assert event.symbol == "BTCUSDT"
    assert float(event.features["bpc_signal"]) == 1.0
    assert set(event.features_by_timeframe) == {"240T", "120T"}
    assert float(event.features_by_timeframe["120T"]["srb_signal"]) == 1.0
    assert len(event.bars) == 1
    assert float(event.bars[0]["close"]) == 100.5

    assert provider.poll() == []


def test_classic_feature_bus_provider_polls_execution_bars_without_features(tmp_path):
    writer = FeatureBusWriter(tmp_path, max_rows=10)
    writer.append_bar_1m(
        "BTCUSDT",
        {
            "timestamp": pd.Timestamp("2026-01-01T00:00:30Z"),
            "open": 100.0,
            "high": 104.0,
            "low": 100.0,
            "close": 103.5,
            "_bar_kind": "fast_intraminute",
        },
    )
    provider = ClassicFeatureBusProvider(
        feature_bus_root=tmp_path,
        symbols=["BTCUSDT"],
        primary_timeframe="240T",
        timeframes=["240T"],
        max_staleness_seconds=0,
    )

    first = provider.poll_bars()
    second = provider.poll_bars()

    assert list(first) == ["BTCUSDT"]
    assert first["BTCUSDT"][0]["_bar_kind"] == "fast_intraminute"
    assert second == {}


def test_classic_feature_bus_provider_limits_initial_bar_backfill(tmp_path):
    writer = FeatureBusWriter(tmp_path, max_rows=10)
    for minute in range(5):
        writer.append_bar_1m(
            "BTCUSDT",
            {
                "timestamp": pd.Timestamp(f"2026-01-01T00:0{minute}:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + minute,
            },
        )
    provider = ClassicFeatureBusProvider(
        feature_bus_root=tmp_path,
        symbols=["BTCUSDT"],
        primary_timeframe="240T",
        timeframes=["240T"],
        max_staleness_seconds=0,
        initial_bars_lookback=1,
    )

    first = provider.poll_bars()

    assert [str(row["timestamp"]) for row in first["BTCUSDT"]] == [
        "2026-01-01 00:04:00+00:00"
    ]


def test_classic_feature_bus_provider_skips_stale_rows(tmp_path):
    writer = FeatureBusWriter(tmp_path, max_rows=10)
    writer.append_features(
        symbol="ETHUSDT",
        timeframe="240T",
        features={"close": 1.0},
        timestamp=pd.Timestamp("2020-01-01T00:00:00Z"),
    )

    provider = ClassicFeatureBusProvider(
        feature_bus_root=tmp_path,
        symbols=["ETHUSDT"],
        primary_timeframe="240T",
        timeframes=["240T"],
        max_staleness_seconds=1,
    )

    assert provider.poll() == []
