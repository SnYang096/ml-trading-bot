"""Tests for open positions list (CMS holdings view)."""

from __future__ import annotations

import sqlite3

import pytest

from mlbot_console.services.open_positions_list import collect_open_positions


def test_collect_open_positions_trend_and_multileg(
    trend_db, spot_db, multi_leg_db, bus_root
) -> None:
    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_open', 'ETHUSDT', 'long',
            '2024-01-02T10:00:00+00:00', NULL,
            100.0, NULL, NULL, 'open', 'tpc', 98.5, 106.0, 0.5, NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_tp', 'ETHUSDT', 'SELL', 'open', 'limit',
            0.5, 106.0, NULL,
            NULL, '2024-01-02T11:00:00+00:00', '2024-01-02T11:00:00+00:00',
            NULL, 0.0, 'p_open'
        )
        """
    )
    conn.commit()
    conn.close()

    rows = collect_open_positions(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot", "multi_leg"],
        limit=50,
        feature_bus_root=bus_root,
    )
    by_scope = {r["scope"]: r for r in rows if r["symbol"] == "ETHUSDT"}
    assert "trend" in by_scope
    trend = by_scope["trend"]
    assert trend["strategy"] == "tpc"
    assert trend["side"] == "long"
    assert trend["quantity"] == pytest.approx(0.5)
    assert trend["pending_exit_orders"] == 1
    assert trend["unrealized_pnl_usdt"] is not None
    assert trend["entry_marker_id"] == "trend:positions:p_open:entry"

    ml_rows = [r for r in rows if r["scope"] == "multi_leg"]
    assert len(ml_rows) >= 1
    assert ml_rows[0]["strategy"] == "chop_grid"
    assert ml_rows[0]["unrealized_pnl_usdt"] is not None


def test_collect_open_positions_spot_lot(
    spot_db, trend_db, multi_leg_db, bus_root
) -> None:
    rows = collect_open_positions(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["spot"],
        limit=20,
        feature_bus_root=bus_root,
    )
    assert len(rows) == 1
    assert rows[0]["scope"] == "spot"
    assert rows[0]["quantity"] == pytest.approx(0.1)
    assert rows[0]["unrealized_pnl_usdt"] is not None


def test_open_positions_api(client, trend_db, bus_root) -> None:
    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_api', 'ETHUSDT', 'short',
            '2024-01-03T08:00:00+00:00', NULL,
            105.0, NULL, NULL, 'open', 'tpc', 108.0, 102.0, 0.2, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    r = client.get(
        "/api/orders/open-positions",
        params={"symbol": "ETHUSDT", "scopes": "trend", "strategy": "tpc"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["meta"]["symbol"] == "ETHUSDT"
    rows = payload["data"]
    assert any(x["position_id"] == "p_api" and x["side"] == "short" for x in rows)
