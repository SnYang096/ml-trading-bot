"""HTTP-level web tests (no browser): HTML/JS assets and API from page context."""

from __future__ import annotations


def test_trade_map_html_served(client):
    r = client.get("/trade-map")
    assert r.status_code == 200
    body = r.text
    assert "Trade Map Live" in body
    assert "trade-map-core.js" in body
    assert "layerMultiLeg" in body
    assert "module-accounts" in body
    assert "featureColumnList" in body
    assert "subchartStack" in body
    assert "eligibilityPanel" in body


def test_static_core_js(client):
    r = client.get("/static/trade-map-core.js")
    assert r.status_code == 200
    assert "MLBotTradeMapCore" in r.text
    assert "markersToLwc" in r.text


def test_bundle_json_shape_for_frontend(client):
    r = client.get(
        "/api/trade-map/bundle",
        params={
            "symbol": "ETHUSDT",
            "timeframe": "2h",
            "scopes": "trend,spot,multi_leg",
            "from": "2024-01-01T00:00:00Z",
            "to": "2024-01-02T00:00:00Z",
            "include_pending": "true",
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert "ohlcv" in payload["data"]
    assert "markers" in payload["data"]
    assert "overlays" in payload["data"]
    candle = payload["data"]["ohlcv"]["candles"][0]
    assert {"time", "open", "high", "low", "close"}.issubset(candle.keys())
