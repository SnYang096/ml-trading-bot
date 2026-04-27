from __future__ import annotations

from src.order_management.multi_leg_ws_provider import MultiLegWebSocketBarProvider


def test_websocket_provider_maps_features_to_multi_leg_bar_event(tmp_path) -> None:
    provider = MultiLegWebSocketBarProvider(
        symbols=["BTCUSDT"],
        storage_base_path=str(tmp_path / "live_storage"),
    )
    provider._latest_1m_bar["BTCUSDT"] = {
        "timestamp": "2026-01-01T00:00:00Z",
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
    }

    event = provider._feature_event(
        "BTCUSDT",
        {
            "timestamp": "2026-01-01T00:15:00Z",
            "semantic_chop": 0.5,
            "trend_confidence": 0.9,
            "atr14": 2.5,
        },
    )

    assert event.symbol == "BTCUSDT"
    assert event.close == 100.0
    assert event.high == 101.0
    assert event.low == 99.0
    assert event.atr == 2.5
    assert event.features["semantic_chop"] == 0.5
