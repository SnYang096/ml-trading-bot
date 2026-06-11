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


def test_app_shell_nav_in_source() -> None:
    shell = _read("components/AppShell/AppShell.tsx")
    pages = _read("lib/shell.ts")
    assert "MLBot Console" in shell
    assert "/account" in pages
    assert "/trade-map" in pages
    assert "mlbot_orders_filter_v3" in pages


def test_trade_map_two_phase_bundle_in_source() -> None:
    bundle = _read("hooks/useTradeMapBundle.ts")
    assert "include_markers: 'true'" in bundle
    assert "include_features: 'true'" in bundle
    assert "Promise.all" in bundle
    assert "include_ohlcv: 'none'" in bundle


def test_trade_map_layer_toggle_does_not_reset_history() -> None:
    page = _read("pages/TradeMap/TradeMapPage.tsx")
    assert "ohlcvLoadedFrom: null" in page
    sym_idx = page.index("setStoreSymbol(e.target.value)")
    sym_block = page[sym_idx : sym_idx + 220]
    assert "ohlcvLoadedFrom: null" in sym_block
    layer_idx = page.index("layers.trend")
    layer_block = page[max(0, layer_idx - 120) : layer_idx + 120]
    assert "setLayers" in layer_block
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
    hook = _read("hooks/useLightweightChart.ts")
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
