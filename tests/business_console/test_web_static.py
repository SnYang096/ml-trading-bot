"""HTTP-level web tests (no browser): HTML/JS assets and API from page context."""

from __future__ import annotations


def test_trade_map_html_served(client):
    r = client.get("/trade-map")
    assert r.status_code == 200
    body = r.text
    assert "交易地图" in body or "trade-map" in body
    assert "console-shell.js" in body
    assert "trade-map/core/00-constants.js" in body
    assert "trade-map/core/20-markers.js" in body
    assert "trade-map/state.js" in body
    assert "trade-map/bootstrap.js" in body
    assert "chopGridLabelLayer" in body
    assert "layerMultiLeg" in body
    assert "appNav" in body
    assert "featureColumnList" in body
    assert "featurePanelBtn" in body
    assert "featureDrawer" in body
    assert "toolbar-chart" in body
    assert "toolbar-global" in body
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
    assert 'id="layerMultiLeg" checked' in body


def test_orders_html_served(client):
    r = client.get("/orders")
    assert r.status_code == 200
    body = r.text
    assert "订单" in body
    assert "orders-page.js" in body
    assert "ordersTable" in body
    assert "ordersThSymbol" in body
    assert "止损价" in body
    assert "盈亏" in body
    assert "order-detail-body" in body
    assert "appNav" in body


def test_account_html_served(client):
    r = client.get("/account")
    assert r.status_code == 200
    body = r.text
    assert "账户总览" in body
    assert "account-page.js" in body
    assert "kpiRow" in body
    assert "account-global-section" in body
    assert "account-scoped-section" in body
    assert "scopedKpiRow" in body
    assert "appNav" in body


def test_signals_html_overview_first(client):
    r = client.get("/signals")
    assert r.status_code == 200
    body = r.text
    assert "signals-overview-panel" in body
    assert "funnel-collapsed" in body
    overview_pos = body.find("signals-overview-panel")
    funnel_pos = body.find("funnel-collapsed")
    assert overview_pos >= 0 and funnel_pos > overview_pos


def test_console_shell_has_account_nav(client):
    r = client.get("/static/console-shell.js")
    assert r.status_code == 200
    assert 'href: "/account"' in r.text
    assert "formatPnl" in r.text
    assert "mlbot_orders_filter_v2" in r.text
    assert "hideExpired: false" in r.text
    assert "loadOrdersFilter" in r.text


def test_root_redirects_to_trade_map(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert "/trade-map" in r.text


def test_static_core_js(client):
    r = client.get("/static/trade-map/core/20-markers.js")
    assert r.status_code == 200
    assert "markersToLwc" in r.text
    r2 = client.get("/static/trade-map/core/40-features.js")
    assert r2.status_code == 200
    assert "resolveSubchartColumns" in r2.text


def test_trade_map_js_layer_toggle_does_not_reset_history(client):
    """Regression: EMA/layer changes must not call resetOhlcvLoadedRange."""
    boot = client.get("/static/trade-map/bootstrap.js")
    assert boot.status_code == 200
    body = boot.text
    assert "resetChartRangeIds" in body
    bundle = client.get("/static/trade-map/bundle.js").text
    assert "opts.resetMarkerRange" in bundle
    assert '!S.ohlcvLoadedFrom || mode === "full"' not in bundle
    assert "!S.ohlcvLoadedFrom || opts.resetOhlcvRange" in bundle
    assert 'resetChartRangeIds = new Set(["symbolSelect", "timeframeSelect"])' in body
    assert '"layerChopGrid"' in body
    assert '"mainEma1200"' in body
    assert "main_overlays" in bundle
    assert "lastMarkerPollSince" in bundle
    assert 'mode === "poll"' in bundle
    assert "mergeMarkersById" in client.get("/static/trade-map/markers.js").text
    assert "featureDrawer" in client.get("/static/trade-map/features.js").text
    subcharts = client.get("/static/trade-map/subcharts.js").text
    assert "pane.chart" in subcharts
    assert "pane.S.chart" not in subcharts
    assert "mergeChopMapPayload" in bundle
    assert "stage_regions" in bundle
    chop = client.get("/static/trade-map/chop.js").text
    assert "S.candleSeries.priceToCoordinate" in chop
    assert 'priceScale("right").priceToCoordinate' not in chop
    assert "ps.priceToCoordinate" not in chop


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
