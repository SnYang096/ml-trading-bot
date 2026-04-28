from __future__ import annotations

import pandas as pd

from src.live_data_stream.feature_bus import FeatureBusWriter
from src.order_management.multi_leg_feature_store_provider import (
    FeatureStoreBarProvider,
)


def test_feature_store_bar_provider_returns_new_events_once(tmp_path):
    writer = FeatureBusWriter(tmp_path)
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
    assert first[0].features["semantic_chop"] == 0.55
    assert second == []
