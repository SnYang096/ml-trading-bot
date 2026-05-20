"""FastAPI business console endpoints."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from app.config import ConsoleSettings


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["service"] == "mlbot-business-console"
    assert body["data"]["databases"]["trend_order"]["readable"] is True


def test_trade_map_symbols(client):
    r = client.get("/api/trade-map/symbols")
    assert r.status_code == 200
    syms = [x["symbol"] for x in r.json()["data"]]
    assert "ETHUSDT" in syms


def test_trade_map_bundle_full_range_default(client):
    r = client.get(
        "/api/trade-map/bundle",
        params={"symbol": "ETHUSDT", "timeframe": "2h", "scopes": "trend"},
    )
    assert r.status_code == 200
    body = r.json()
    meta = body["meta"]
    candles = body["data"]["ohlcv"]["candles"]
    assert meta["bars_1min_rows"] == 24 * 60 * 3
    assert len(candles) >= 30
    assert meta["range_clipped"] is False


def test_trade_map_bundle(client):
    r = client.get(
        "/api/trade-map/bundle",
        params={
            "symbol": "ETHUSDT",
            "timeframe": "2h",
            "scopes": "trend,spot,multi_leg",
            "from": "2024-01-01T00:00:00Z",
            "to": "2024-01-02T00:00:00Z",
            "include_pending": "true",
            "feature_columns": "weekly_ema_200_position",
        },
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["ohlcv"]["source"] == "bars_1min"
    assert len(data["ohlcv"]["candles"]) >= 1
    assert len(data["markers"]) >= 2
    assert "weekly_ema_200_position" in data["overlays"]
    assert data["overlays"]["weekly_ema_200_position"]["available"] is True


def test_spot_eligibility(client):
    r = client.get("/api/spot/eligibility", params={"symbol": "ETHUSDT"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["symbol"] == "ETHUSDT"
    assert "pending_spot_orders" in data["blockers"]
    assert data["can_buy"] is False


def test_marker_detail(client):
    r = client.get(
        "/api/trade-map/marker-detail",
        params={"marker_id": "trend:positions:p1:entry"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["found"] is True


def test_links(client):
    r = client.get("/api/links")
    assert r.status_code == 200
    ids = {x["id"] for x in r.json()["data"]["links"]}
    assert "grafana" in ids
    assert "rolling_backtest" in ids


def test_bus_ohlcv_window_error(client):
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


def test_markers_missing_db(client, tmp_path, monkeypatch):
    from app.main import app
    from fastapi.testclient import TestClient

    settings = ConsoleSettings(
        repo_root=tmp_path,
        feature_bus_root=tmp_path,
        live_data_root=tmp_path,
        engine_data_root=tmp_path,
        live_root=tmp_path,
        constitution_yaml=tmp_path / "constitution.yaml",
        universe_yaml=tmp_path / "universe.yaml",
        trend_order_db=tmp_path / "nope.db",
        live_monitor_db=tmp_path / "nope.db",
        spot_order_db=tmp_path / "nope2.db",
        spot_ledger_db=tmp_path / "nope3.db",
        multi_leg_db=tmp_path / "nope4.db",
        max_ohlcv_days=90,
        map_poll_seconds=10.0,
        grafana_url="http://localhost:3000",
        rolling_backtest_url="",
        basic_auth_user=None,
        basic_auth_password=None,
    )
    monkeypatch.setattr("app.routers.trade_map.SETTINGS", settings)
    c = TestClient(app)
    r = c.get(
        "/api/trade-map/markers",
        params={"symbol": "ETHUSDT", "scopes": "trend,spot"},
    )
    assert r.status_code == 200
    assert r.json()["data"] == []


def test_basic_auth_required(console_settings):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    replace(console_settings, basic_auth_user="tester", basic_auth_password="secret")

    from app.auth import BasicAuthMiddleware
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    mini = Starlette(
        routes=[Route("/api/trade-map/symbols", lambda r: PlainTextResponse("ok"))],
    )
    mini.add_middleware(
        BasicAuthMiddleware,
        user="tester",
        password="secret",
    )
    c = TestClient(mini)
    assert c.get("/api/trade-map/symbols").status_code == 401
    token = base64.b64encode(b"tester:secret").decode()
    assert (
        c.get(
            "/api/trade-map/symbols",
            headers={"Authorization": f"Basic {token}"},
        ).status_code
        == 200
    )
