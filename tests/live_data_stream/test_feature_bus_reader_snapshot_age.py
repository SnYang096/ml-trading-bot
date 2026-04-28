from __future__ import annotations

import time

import pandas as pd

from src.live_data_stream.feature_bus import FeatureBusReader, FeatureBusWriter


def test_latest_snapshot_age_seconds_reads_latest_json(tmp_path) -> None:
    writer = FeatureBusWriter(tmp_path, max_rows=10)
    ts = pd.Timestamp("2026-01-01T12:00:00Z")
    writer.append_features(
        symbol="BTCUSDT",
        timeframe="2h",
        features={"close": 1.0},
        timestamp=ts,
    )
    reader = FeatureBusReader(tmp_path)
    age = reader.latest_snapshot_age_seconds(symbol="BTCUSDT", timeframe="2h")
    assert age is not None
    assert age >= 0.0
    time.sleep(0.05)
    age2 = reader.latest_snapshot_age_seconds(symbol="BTCUSDT", timeframe="2h")
    assert age2 is not None and age2 >= age


def test_latest_snapshot_age_seconds_missing_returns_none(tmp_path) -> None:
    reader = FeatureBusReader(tmp_path)
    assert reader.latest_snapshot_age_seconds(symbol="ETHUSDT", timeframe="2h") is None
