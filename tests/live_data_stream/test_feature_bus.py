from __future__ import annotations

import pandas as pd

from src.live_data_stream.feature_bus import FeatureBusReader, FeatureBusWriter


def test_feature_bus_writer_reader_latest_features(tmp_path):
    writer = FeatureBusWriter(tmp_path, max_rows=2)
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="120T",
        features={"close": 100.0, "semantic_chop": 0.5},
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="120T",
        features={"close": 101.0, "semantic_chop": 0.6},
        timestamp=pd.Timestamp("2024-01-01T02:00:00Z"),
    )

    row = FeatureBusReader(tmp_path).latest_features(symbol="BTCUSDT", timeframe="120T")

    assert row is not None
    assert float(row["close"]) == 101.0
    assert float(row["semantic_chop"]) == 0.6


def test_feature_bus_writer_trims_rows(tmp_path):
    writer = FeatureBusWriter(tmp_path, max_rows=1)
    writer.append_bar_1m(
        "ETHUSDT",
        {
            "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
        },
    )
    writer.append_bar_1m(
        "ETHUSDT",
        {
            "timestamp": pd.Timestamp("2024-01-01T00:01:00Z"),
            "open": 1.5,
            "high": 2.5,
            "low": 1.0,
            "close": 2.0,
        },
    )

    bars = FeatureBusReader(tmp_path).latest_bars_1m(symbol="ETHUSDT")

    assert len(bars) == 1
    assert float(bars.iloc[0]["close"]) == 2.0


def test_merge_bars_1m_fills_bus_holes(tmp_path):
    writer = FeatureBusWriter(tmp_path, max_rows=10)
    writer.append_bar_1m(
        "ETHUSDT",
        {
            "timestamp": pd.Timestamp("2026-05-21T00:00:00Z"),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
        },
    )
    writer.append_bar_1m(
        "ETHUSDT",
        {
            "timestamp": pd.Timestamp("2026-05-21T00:02:00Z"),
            "open": 2.0,
            "high": 3.0,
            "low": 1.5,
            "close": 2.5,
        },
    )
    repair = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-21T00:01:00Z"], utc=True),
            "open": [1.6],
            "high": [2.1],
            "low": [1.4],
            "close": [1.9],
        }
    )
    n = writer.merge_bars_1m("ETHUSDT", repair)
    assert n == 1
    bars = FeatureBusReader(tmp_path).latest_bars_1m(symbol="ETHUSDT")
    assert len(bars) == 3
    assert float(bars.iloc[1]["close"]) == 1.9
