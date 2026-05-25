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
    assert "taxonomy" in data
    assert data["taxonomy"]["strategies"]
    meta = data["column_meta"].get("weekly_ema_200_position")
    assert meta and meta[0]["stage"] == "prefilter"


def test_bus_feature_taxonomy_endpoint(client):
    r = client.get("/api/bus/features/taxonomy")
    assert r.status_code == 200
    tax = r.json()["data"]
    assert any(s["id"] == "tpc" for s in tax["strategies"])
    assert any(s["id"] == "chop_grid" for s in tax["strategies"])
    assert any(s["id"] == "trend_scalp" for s in tax["strategies"])
    assert "chop_grid" in tax.get("live_strategy_ids", [])
    assert "tpc" in tax.get("live_strategy_ids", [])
    assert "bpc" not in [s["id"] for s in tax["strategies"]]
    assert tax.get("constitution_source", "").endswith("constitution.yaml")
    assert "tpc_pullback_depth" in tax["index"]


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
