"""Feature column discovery and multi-overlay bundle."""

from __future__ import annotations


def test_bus_feature_columns(client):
    r = client.get(
        "/api/bus/features/columns",
        params={"symbol": "ETHUSDT", "timeframe": "2h"},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["available"] is True
    assert "weekly_ema_200_position" in data["columns"]
    assert "regime_score" in data["columns"]
    assert "weekly_ema_200_position" in data["defaults"]


def test_bundle_multi_feature_overlays(client):
    r = client.get(
        "/api/trade-map/bundle",
        params={
            "symbol": "ETHUSDT",
            "timeframe": "2h",
            "scopes": "trend",
            "from": "2024-01-01T00:00:00Z",
            "to": "2024-01-02T00:00:00Z",
            "feature_columns": "weekly_ema_200_position,regime_score",
        },
    )
    assert r.status_code == 200
    overlays = r.json()["data"]["overlays"]
    assert overlays["weekly_ema_200_position"]["available"] is True
    assert overlays["regime_score"]["available"] is True
    assert len(overlays["weekly_ema_200_position"]["points"]) >= 1
    meta = r.json()["meta"]
    assert "weekly_ema_200_position" in meta["feature_columns"]
