"""Smoke tests for every business-console HTTP API route."""

from __future__ import annotations

import pytest

# (path, query_params, expected_status)
CMS_GET_ROUTES: list[tuple[str, dict, int]] = [
    ("/api/health", {}, 200),
    ("/api/overview", {}, 200),
    ("/api/constitution/summary", {}, 200),
    ("/api/links", {}, 200),
    ("/api/bus/features/columns", {"symbol": "ETHUSDT", "timeframe": "120T"}, 200),
    ("/api/bus/features/taxonomy", {}, 200),
    ("/api/trade-map/symbols", {}, 200),
    (
        "/api/trade-map/bundle",
        {
            "symbol": "ETHUSDT",
            "timeframe": "2h",
            "scopes": "trend",
            "from": "2024-01-01T00:00:00Z",
            "to": "2024-01-02T00:00:00Z",
        },
        200,
    ),
    (
        "/api/trade-map/markers",
        {
            "symbol": "ETHUSDT",
            "scopes": "trend,spot,multi_leg",
            "from": "2024-01-01T00:00:00Z",
            "to": "2024-01-02T00:00:00Z",
        },
        200,
    ),
    ("/api/trade-map/marker-detail", {"marker_id": "trend:positions:p1:entry"}, 200),
    ("/api/spot/eligibility", {"symbol": "ETHUSDT"}, 200),
    (
        "/api/orders/list",
        {"symbol": "*", "scopes": "trend,spot,multi_leg", "limit": "50"},
        200,
    ),
    ("/api/trend/orders", {"symbol": "ETHUSDT", "limit": "50"}, 200),
    ("/api/trend/funnel", {"symbol": "ETHUSDT", "limit": "5"}, 200),
    ("/api/spot/orders", {"symbol": "ETHUSDT", "limit": "50"}, 200),
    ("/api/multileg/orders", {"symbol": "ETHUSDT", "limit": "50"}, 200),
    ("/api/account/summary", {"lookback_days": "365"}, 200),
    ("/api/account/reconciliation/all", {"symbol": "*", "lookback_days": "0"}, 200),
    ("/api/trend/regime-ops", {"limit": "5"}, 200),
    ("/api/regime/cockpit", {"symbol": "BTCUSDT"}, 200),
    (
        "/api/bus/ohlcv",
        {
            "symbol": "ETHUSDT",
            "timeframe": "2h",
            "from": "2024-01-01T00:00:00Z",
            "to": "2024-01-02T00:00:00Z",
        },
        200,
    ),
]


@pytest.mark.parametrize("path,params,status_code", CMS_GET_ROUTES)
def test_cms_get_route_returns_expected_status(
    client, path: str, params: dict, status_code: int
) -> None:
    r = client.get(path, params=params)
    assert r.status_code == status_code, r.text[:500]


def test_cms_bus_ohlcv_rejects_oversized_window(client) -> None:
    r = client.get(
        "/api/bus/ohlcv",
        params={
            "symbol": "ETHUSDT",
            "timeframe": "1min",
            "from": "2024-01-01T00:00:00Z",
            "to": "2024-06-01T00:00:00Z",
        },
    )
    assert r.status_code == 400
