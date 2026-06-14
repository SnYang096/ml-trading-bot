"""Tests for open positions list (CMS holdings view)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from mlbot_console.services.open_positions_list import (
    _exchange_has_position,
    _exchange_position_map,
    collect_open_positions,
)


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


# ── Exchange cross-reference helpers ──────────────────────────

def test_exchange_position_map_from_ledger():
    ledger = {
        "accounts": [
            {
                "exchange_open_positions": [
                    {"symbol": "ETHUSDT", "position_amt": 2.884},
                    {"symbol": "ETHUSDT", "position_amt": -2.882},
                    {"symbol": "XRPUSDT", "position_amt": 12585.0},
                    {"symbol": "BNBUSDT", "position_amt": 0},  # dust — skip
                    {},
                ]
            },
            {
                "exchange_open_positions": [
                    {"symbol": "BNBUSDT", "position_amt": 0.31},
                ]
            },
        ]
    }
    m = _exchange_position_map(ledger)
    assert m[("ETHUSDT", "long")] == 2.884
    assert m[("ETHUSDT", "short")] == 2.882
    assert m[("XRPUSDT", "long")] == 12585.0
    assert m[("BNBUSDT", "long")] == 0.31
    # Dust (qty=0) filtered
    assert ("BNBUSDT", "short") not in m


def test_exchange_position_map_empty_or_broken():
    assert _exchange_position_map(None) == {}
    assert _exchange_position_map({}) == {}
    assert _exchange_position_map({"accounts": []}) == {}
    assert _exchange_position_map({"accounts": [{}]}) == {}
    # position_amt missing
    assert _exchange_position_map({
        "accounts": [{"exchange_open_positions": [{"symbol": "X", "position_amt": 0.0}]}]
    }) == {}


def test_exchange_has_position_threshold():
    m = {("ETHUSDT", "long"): 0.31, ("BNBUSDT", "long"): 0.0}
    assert _exchange_has_position(m, "ETHUSDT", "long")
    assert not _exchange_has_position(m, "SOLUSDT", "long")
    # Symbol not in map
    assert not _exchange_has_position(m, "XRPUSDT", "short")
    # Zero quantity — still below min_qty of 0.0001
    m2 = {("BNBUSDT", "long"): 0.0}
    assert not _exchange_has_position(m2, "BNBUSDT", "long")
    # Case-insensitive
    assert _exchange_has_position(m, "ethusdt", "LONG")


# ── Trend dedup ──────────────────────────────────────────────

def test_trend_dedup_keeps_most_recent(trend_db, spot_db, multi_leg_db, bus_root):
    """Two entries for same (symbol, side) → only most recent survives."""
    ts1 = int(datetime(2026, 6, 12, 2, 21, 53, tzinfo=timezone.utc).timestamp())
    ts2 = int(datetime(2026, 6, 12, 2, 33, 38, tzinfo=timezone.utc).timestamp())

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'BNB:exchange_sync_1', 'BNBUSDT', 'long',
            ?, NULL,
            630.68, NULL, NULL, 'open', 'tpc',
            NULL, NULL, 0.31, NULL
        )
        """,
        (datetime.fromtimestamp(ts1, tz=timezone.utc).isoformat(),),
    )
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'BNB:bootstrap_2', 'BNBUSDT', 'long',
            ?, NULL,
            630.68, NULL, NULL, 'open', 'tpc',
            NULL, NULL, 0.31, NULL
        )
        """,
        (datetime.fromtimestamp(ts2, tz=timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()

    rows = collect_open_positions(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="*",
        scopes=["trend"],
        limit=50,
    )
    bnb_rows = [r for r in rows if r["symbol"] == "BNBUSDT"]
    assert len(bnb_rows) == 1, f"expected 1 BNBUSDT after dedup, got {len(bnb_rows)}"
    # Kept the most recent (bootstrap with later entry_time)
    assert bnb_rows[0]["position_id"] == "BNB:bootstrap_2"


def test_trend_dedup_preserves_non_trend_scopes(trend_db, spot_db, multi_leg_db, bus_root):
    """Multi-leg entries must survive trend dedup untouched."""
    conn = sqlite3.connect(trend_db)
    ts = datetime(2026, 6, 12, 2, 21, 53, tzinfo=timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'BNB:dup_a', 'BNBUSDT', 'long',
            ?, NULL,
            630.68, NULL, NULL, 'open', 'tpc',
            NULL, NULL, 0.31, NULL
        )
        """,
        (ts,),
    )
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'BNB:dup_b', 'BNBUSDT', 'long',
            ?, NULL,
            630.68, NULL, NULL, 'open', 'tpc',
            NULL, NULL, 0.31, NULL
        )
        """,
        (ts,),
    )
    conn.commit()
    conn.close()

    rows = collect_open_positions(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="*",
        scopes=["trend", "multi_leg", "spot"],
        limit=50,
    )
    # Multi-leg rows must still be present (trend dedup only targets trend scope)
    ml = [r for r in rows if r["scope"] == "multi_leg" and r["symbol"] == "ETHUSDT"]
    assert len(ml) >= 1, f"multi_leg ETHUSDT entries should survive dedup, got {len(ml)}"


# ── Exchange-ledger cross-ref filter ─────────────────────────

def test_exchange_cross_ref_drops_stale_symbols(trend_db, spot_db, multi_leg_db, bus_root):
    """BNBUSDT in local DB but NOT on exchange → filter drops it."""
    import sqlite3
    from datetime import datetime, timezone

    ts = datetime(2026, 6, 12, 2, 21, 53, tzinfo=timezone.utc).isoformat()
    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'SOL:fake_open', 'SOLUSDT', 'long',
            ?, NULL,
            64.80, NULL, NULL, 'open', 'tpc',
            NULL, NULL, 3.35, NULL
        )
        """,
        (ts,),
    )
    # Also insert an OPEN ETHUSDT position that should survive the filter
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'ETH:keep_open', 'ETHUSDT', 'long',
            ?, NULL,
            2000.0, NULL, NULL, 'open', 'tpc',
            NULL, NULL, 0.1, NULL
        )
        """,
        (ts,),
    )
    conn.commit()
    conn.close()

    # Exchange ledger: only ETHUSDT, no SOLUSDT
    exchange_ledger = {
        "accounts": [
            {"exchange_open_positions": [
                {"symbol": "ETHUSDT", "position_amt": 2.884}
            ]}
        ]
    }

    rows = collect_open_positions(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="*",
        scopes=["trend"],
        limit=50,
        exchange_ledger=exchange_ledger,
    )
    sol = [r for r in rows if r["symbol"] == "SOLUSDT"]
    assert len(sol) == 0, f"SOLUSDT should be filtered (exchange shows 0), got {len(sol)}"

    eth = [r for r in rows if r["symbol"] == "ETHUSDT"]
    assert len(eth) >= 1, f"ETHUSDT should survive (exchange has it), got {len(eth)}"
