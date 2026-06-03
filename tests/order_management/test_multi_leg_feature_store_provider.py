from __future__ import annotations

import pandas as pd

from src.live_data_stream.feature_bus import FeatureBusWriter
from src.order_management.multi_leg_feature_store_provider import (
    FeatureStoreBarProvider,
)


def test_feature_store_bar_provider_returns_new_events_once(tmp_path):
    writer = FeatureBusWriter(tmp_path)
    writer.append_bar_1m(
        "BTCUSDT",
        {
            "timestamp": pd.Timestamp("2024-01-01T00:01:00Z"),
            "open": 99.0,
            "high": 106.0,
            "low": 98.0,
            "close": 104.0,
        },
    )
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="2h",
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        features={
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "atr14": 3.0,
            "semantic_chop": 0.55,
            "box_prefilter": False,
            "trend_confidence": 0.8,
            "trend_direction": "UP",
        },
    )

    provider = FeatureStoreBarProvider(feature_bus_root=tmp_path, timeframe="2h")
    first = provider.latest_closed_bars(["BTCUSDT"])
    second = provider.latest_closed_bars(["BTCUSDT"])

    assert len(first) == 1
    assert first[0].symbol == "BTCUSDT"
    assert first[0].atr == 3.0
    assert first[0].timestamp == "2024-01-01 00:01:00+00:00"
    assert first[0].high == 106.0
    assert first[0].close == 104.0
    assert first[0].features["semantic_chop"] == 0.55
    assert first[0].features["_signal_timestamp"] == "2024-01-01 00:00:00+00:00"
    assert second == []


def test_feature_store_bar_provider_decodes_numeric_trend_direction(tmp_path):
    writer = FeatureBusWriter(tmp_path)
    writer.append_bar_1m(
        "BTCUSDT",
        {
            "timestamp": pd.Timestamp("2024-01-01T00:01:00Z"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        },
    )
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="2h",
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        features={
            "close": 100.0,
            "atr14": 2.0,
            "trend_confidence": 0.9,
            "trend_direction": -1.0,
            "trend_direction_raw": -1.0,
        },
    )
    provider = FeatureStoreBarProvider(feature_bus_root=tmp_path, timeframe="2h")
    events = provider.latest_closed_bars(["BTCUSDT"])
    assert events[0].features["trend_direction"] == "DOWN"


def test_feature_store_bar_provider_skips_historical_startup_bars(tmp_path):
    writer = FeatureBusWriter(tmp_path)
    for minute in range(1, 6):
        writer.append_bar_1m(
            "BTCUSDT",
            {
                "timestamp": pd.Timestamp(f"2024-01-01T00:0{minute}:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + minute,
            },
        )
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="2h",
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        features={
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "atr14": 3.0,
        },
    )

    provider = FeatureStoreBarProvider(
        feature_bus_root=tmp_path,
        timeframe="2h",
        initial_backfill_bars=1,
    )
    first = provider.latest_closed_bars(["BTCUSDT"])

    assert [bar.timestamp for bar in first] == ["2024-01-01 00:05:00+00:00"]
