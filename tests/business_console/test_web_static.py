"""HTTP-level web tests: React SPA shell + static assets + bundle API."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
DIST_ROOT = REPO_ROOT / "src" / "mlbot_console" / "static" / "dist"

SPA_ROUTES = (
    "/trade-map",
    "/trade-map-grid",
    "/orders",
    "/signals",
    "/account",
    "/regime",
    "/monitoring",
)


def _read(rel: str) -> str:
    return (FRONTEND_SRC / rel).read_text(encoding="utf-8")


def test_spa_routes_serve_index(client) -> None:
    for path in SPA_ROUTES:
        r = client.get(path)
        assert r.status_code == 200, path
        assert 'id="root"' in r.text, path
        assert "/static/assets/" in r.text, path


def test_root_redirects_to_trade_map(client) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "/trade-map" in r.headers.get("location", "")


def test_vite_bundle_asset_reachable(client) -> None:
    index = DIST_ROOT / "index.html"
    if not index.is_file():
        pytest.skip("Run: make frontend-build")
    html = index.read_text(encoding="utf-8")
    match = re.search(r'src="(/static/assets/[^"]+\.js)"', html)
    assert match, "missing Vite JS asset in index.html"
    r = client.get(match.group(1))
    assert r.status_code == 200
    assert len(r.content) > 1000


def test_vite_lazy_trade_map_chunk_reachable(client) -> None:
    """index.html main chunk must reference an existing TradeMapPage lazy chunk."""
    index = DIST_ROOT / "index.html"
    if not index.is_file():
        pytest.skip("Run: make frontend-build")
    main_match = re.search(r'src="(/static/assets/index-[^"]+\.js)"', index.read_text())
    assert main_match
    main_js = (DIST_ROOT / main_match.group(1).removeprefix("/static/")).read_text(
        encoding="utf-8"
    )
    chunk_match = re.search(r"TradeMapPage-[A-Za-z0-9_-]+\.js", main_js)
    assert chunk_match, "main bundle missing TradeMapPage lazy import"
    r = client.get(f"/static/assets/{chunk_match.group(0)}")
    assert r.status_code == 200
    assert len(r.content) > 500


def test_app_shell_nav_in_source() -> None:
    shell = _read("components/AppShell/AppShell.tsx")
    pages = _read("lib/shell.ts")
    assert "root@mlbot" in shell
    assert "/account" in pages
    assert "/trade-map" in pages
    assert "mlbot_orders_filter_v3" in pages


def test_trade_map_two_phase_bundle_in_source() -> None:
    bundle = _read("hooks/useTradeMapBundle.ts")
    query = _read("lib/tradeMap/bundleQuery.ts")
    assert "Promise.all" in bundle
    assert "buildFullShellQuery" in query
    assert "buildFullMarkersQuery" in query
    assert "buildFullFeaturesQuery" in query
    assert "include_markers: 'true'" in query
    assert "include_features: 'true'" in query
    assert "include_ohlcv: 'none'" in query
    assert "lastTradeLinks" in bundle


def test_trade_map_history_pan_in_source() -> None:
    history = _read("hooks/useTradeMapHistory.ts")
    assert "loadMoreHistory" in history
    assert "subscribeVisibleLogicalRangeChange" in history
    assert "unsubscribeVisibleLogicalRangeChange" in history
    assert "mergeCandlesByTime" in history
    assert "refreshMarkersOnly" in history
    assert "mergeFeatureOverlays" in history


def test_trade_map_grid_page_in_source() -> None:
    grid = _read("pages/TradeMapGrid/TradeMapGridPage.tsx")
    assert "GRID_SYMBOLS" in grid
    assert "MiniTradeMapChart" in grid
    assert "useStaggeredGridQueries" in grid
    assert "trade-map-grid" in _read("routes.tsx")


def test_routes_lazy_load_pages() -> None:
    routes = _read("routes.tsx")
    assert "React.lazy" in routes or "lazy(" in routes
    assert "Suspense" in routes
    assert "PageFallback" in routes


def test_page_visibility_polling() -> None:
    hook = _read("hooks/usePageVisible.ts")
    assert "usePageVisible" in hook
    assert "visibleRefetchInterval" in hook
    assert "usePageVisible" in _read("pages/Orders/OrdersPage.tsx")
    assert "usePageVisible" in _read("pages/TradeMap/TradeMapPage.tsx")


def test_trade_map_ema_overlay_refresh() -> None:
    bundle = _read("hooks/useTradeMapBundle.ts")
    page = _read("pages/TradeMap/TradeMapPage.tsx")
    assert "refreshMainOverlays" in bundle
    assert "refreshMainOverlays" in page
    assert "mainEma1200" in page
    full_m = re.search(
        r"refreshFull\(\)\.catch\(\(\) => \{\}\);\s*\n\s*\}, \[([^\]]+)\]",
        page,
    )
    assert full_m, "refreshFull effect deps not found"
    full_deps = full_m.group(1)
    assert "mainEma1200" not in full_deps
    assert "mainWeeklyEma200" not in full_deps
    ema_m = re.search(
        r"refreshMainOverlays\(\)\.catch\(\(\) => \{\}\);\s*\n\s*\}, \[([^\]]+)\]",
        page,
    )
    assert ema_m, "refreshMainOverlays effect deps not found"
    ema_deps = ema_m.group(1)
    assert "mainEma1200" in ema_deps
    assert "mainWeeklyEma200" in ema_deps


def test_orders_client_pagination() -> None:
    orders = _read("pages/Orders/OrdersPage.tsx")
    assert "PAGE_SIZE" in orders
    assert "上一页" in orders
    assert "下一页" in orders


def test_account_recon_lazy_load() -> None:
    account = _read("pages/Account/AccountPage.tsx")
    assert "reconOpen" in account
    assert "展开对账" in account
    assert "enabled: reconOpen" in account
    assert "/api/account/reconciliation/all" in account
    assert "SCOPE_LABELS" in account


def test_mini_grid_markers_hide_text() -> None:
    mini = _read("pages/TradeMapGrid/MiniTradeMapChart.tsx")
    assert "showText: false" in mini
    assert "prepareChartMarkers" in mini


def test_grid_tail_poll_query() -> None:
    query = _read("lib/tradeMap/bundleQuery.ts")
    stagger = _read("hooks/useStaggeredGridQueries.ts")
    assert "buildGridPollQuery" in query
    assert "include_ohlcv: 'tail'" in query
    assert "buildGridPollQuery" in stagger
    assert "STAGGER_MS" in stagger


def test_trade_map_poll_uses_tail_bundle() -> None:
    bundle = _read("hooks/useTradeMapBundle.ts")
    assert "refreshPoll" in bundle
    assert "include_ohlcv: 'tail'" in _read("lib/tradeMap/bundleQuery.ts")
    links = _read("lib/tradeMap/tradeLinks.ts")
    main = _read("hooks/useTradeMapMainChart.ts")
    assert "buildTradeLinkLines" in links
    assert "mergeTradeLinks" in links
    assert "tradeLinksForDisplay" in links
    assert "applyTradeLinks" in main
    assert "buildTradeLinkLines" in main


def test_trade_map_layer_toggle_does_not_reset_history() -> None:
    page = _read("pages/TradeMap/TradeMapPage.tsx")
    assert "ohlcvLoadedFrom: null" in page
    sym_idx = page.index("setStoreSymbol(e.target.value)")
    sym_block = page[sym_idx : sym_idx + 220]
    assert "ohlcvLoadedFrom: null" in sym_block
    layer_idx = page.index("layers.trend, layers.spot")
    layer_block = page[max(0, layer_idx - 80) : layer_idx + 80]
    assert "refreshMarkersOnly" in layer_block
    assert "ohlcvLoadedFrom" not in layer_block


def test_trade_map_core_ts_exports() -> None:
    markers = _read("lib/tradeMap/markers.ts")
    features = _read("lib/tradeMap/features.ts")
    ohlcv = _read("lib/tradeMap/ohlcv.ts")
    assert "markersToLwc" in markers
    assert "isFeatureBusRegimeExitMarker" in markers
    assert "chopRegimeExitBarTimes" in markers
    assert "overlayAsOfAtCandleTimes" in ohlcv
    assert "chopGridMetricsRowSpecs" in features


def test_global_styles_use_tokens_not_bare_main() -> None:
    css = _read("styles/global.css")
    assert "scrollbar-color" in css
    assert not re.search(r"(?m)^main\s*\{", css)


def test_lwc_hook_avoids_click_scroll() -> None:
    hook = _read("hooks/useTradeMapMainChart.ts")
    assert "scrollChartToBarTime" not in hook
    assert "subscribeClick" not in hook


def test_bundle_json_shape_for_frontend(client) -> None:
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
