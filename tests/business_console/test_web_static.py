"""HTTP-level web tests (no browser): HTML/JS assets and API from page context."""

from __future__ import annotations


def test_trade_map_html_served(client):
    r = client.get("/trade-map")
    assert r.status_code == 200
    body = r.text
    assert "交易地图" in body or "trade-map" in body
    assert "console-shell.js" in body
    assert "trade-map-core.js" in body
    assert "layerMultiLeg" in body
    assert "appNav" in body
    assert "featureColumnList" in body
    assert "featurePanelBtn" in body
    assert "featureSearch" in body
    assert "subchartStack" in body
    assert 'data-feature-action="preset-tpc"' in body
    assert 'data-feature-action="preset-spot"' in body
    assert "marker-detail-drawer" in body
    assert "ordersDock" in body
    assert "ordersDockToggle" in body
    assert "statusGrid" in body
    assert "statusClock" in body
    assert "side-panels" not in body
    assert "eligibilityPanel" not in body


def test_orders_html_served(client):
    r = client.get("/orders")
    assert r.status_code == 200
    body = r.text
    assert "订单" in body
    assert "orders-page.js" in body
    assert "ordersTable" in body
    assert "ordersThSymbol" in body
    assert "order-detail-body" in body
    assert "appNav" in body


def test_root_redirects_to_signals(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert "/signals" in r.text


def test_static_core_js(client):
    r = client.get("/static/trade-map-core.js")
    assert r.status_code == 200
    assert "MLBotTradeMapCore" in r.text
    assert "markersToLwc" in r.text


def test_trade_map_js_layer_toggle_does_not_reset_history(client):
    """Regression: EMA/layer changes must not call resetOhlcvLoadedRange."""
    r = client.get("/static/trade-map.js")
    assert r.status_code == 200
    body = r.text
    assert "resetChartRangeIds" in body
    assert "opts.resetMarkerRange" in body
    idx = body.find('"mainEma1200"')
    assert idx >= 0
    block = body[idx : idx + 1200]
    assert "resetOhlcvLoadedRange" not in block
    assert "rerunAll" not in block


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
