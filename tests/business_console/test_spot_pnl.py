"""Spot FIFO PnL — buy lots by time, realized on sell."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mlbot_console.services.spot_pnl import compute_spot_order_pnl


def _spot_db(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE spot_orders (
            order_id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            symbol TEXT,
            side TEXT,
            order_type TEXT,
            quantity REAL,
            price REAL,
            status TEXT,
            filled_quantity REAL,
            filled_quote_usdt REAL
        );
        """
    )
    for row in rows:
        conn.execute(
            """
            INSERT INTO spot_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
    conn.commit()
    conn.close()


def test_spot_fifo_sell_matches_oldest_buys(tmp_path: Path) -> None:
    db = tmp_path / "spot.db"
    _spot_db(
        db,
        [
            (
                "b1",
                "2024-01-01T08:00:00+00:00",
                "2024-01-01T08:01:00+00:00",
                "ETHUSDT",
                "buy",
                "market",
                0.2,
                1000.0,
                "filled",
                0.1,
                100.0,
            ),
            (
                "b2",
                "2024-01-01T09:00:00+00:00",
                "2024-01-01T09:01:00+00:00",
                "ETHUSDT",
                "buy",
                "market",
                0.2,
                1100.0,
                "filled",
                0.1,
                120.0,
            ),
            (
                "s1",
                "2024-01-01T10:00:00+00:00",
                "2024-01-01T10:01:00+00:00",
                "ETHUSDT",
                "sell",
                "market",
                0.2,
                1300.0,
                "filled",
                0.15,
                210.0,
            ),
        ],
    )
    pnl = compute_spot_order_pnl(db, mark_prices={"ETHUSDT": 1100.0})
    sell = pnl["s1"]
    assert sell["pnl_hint"] == "已实现"
    assert sell["matched_buy_orders"] == ["b1", "b2"]
    # 0.1 @ 100 + 0.05 @ 120 cost = 160; proceeds 210
    assert sell["realized_pnl"] == pytest.approx(50.0)
    assert pnl["b2"]["unrealized_pnl"] is not None


def test_spot_open_buy_unrealized_with_mark(tmp_path: Path) -> None:
    db = tmp_path / "spot.db"
    _spot_db(
        db,
        [
            (
                "b1",
                "2024-01-01T08:00:00+00:00",
                "2024-01-01T08:01:00+00:00",
                "ETHUSDT",
                "buy",
                "market",
                0.1,
                2000.0,
                "filled",
                0.1,
                200.0,
            ),
        ],
    )
    pnl = compute_spot_order_pnl(db, mark_prices={"ETHUSDT": 2500.0})
    assert pnl["b1"]["pnl_hint"] == "持仓浮盈"
    assert pnl["b1"]["unrealized_pnl"] == pytest.approx(50.0)


def test_spot_fully_sold_buy_has_no_stale_unrealized(tmp_path: Path) -> None:
    """Regression: closed buy lots must not keep pre-sell unrealized rows."""
    db = tmp_path / "spot.db"
    _spot_db(
        db,
        [
            (
                "b1",
                "2024-01-01T08:00:00+00:00",
                "2024-01-01T08:01:00+00:00",
                "ETHUSDT",
                "buy",
                "market",
                0.1,
                1000.0,
                "filled",
                0.1,
                100.0,
            ),
            (
                "s1",
                "2024-01-01T10:00:00+00:00",
                "2024-01-01T10:01:00+00:00",
                "ETHUSDT",
                "sell",
                "market",
                0.1,
                1200.0,
                "filled",
                0.1,
                120.0,
            ),
        ],
    )
    pnl = compute_spot_order_pnl(db, mark_prices={"ETHUSDT": 2000.0})
    assert "b1" not in pnl
    assert pnl["s1"]["realized_pnl"] == pytest.approx(20.0)


def test_spot_sell_without_inventory_omitted(tmp_path: Path) -> None:
    db = tmp_path / "spot.db"
    _spot_db(
        db,
        [
            (
                "s1",
                "2024-01-01T10:00:00+00:00",
                "2024-01-01T10:01:00+00:00",
                "ETHUSDT",
                "sell",
                "market",
                0.1,
                1200.0,
                "filled",
                0.1,
                120.0,
            ),
        ],
    )
    assert compute_spot_order_pnl(db) == {}


def test_spot_strategy_from_db_column(tmp_path: Path) -> None:
    db = tmp_path / "spot.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE spot_orders (
            order_id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            symbol TEXT,
            side TEXT,
            order_type TEXT,
            quantity REAL,
            price REAL,
            status TEXT,
            filled_quantity REAL,
            filled_quote_usdt REAL,
            strategy TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO spot_orders VALUES (
            'b1', '2024-01-01T08:00:00+00:00', '2024-01-01T08:01:00+00:00',
            'ETHUSDT', 'buy', 'market', 0.1, 1000.0, 'filled', 0.1, 100.0,
            'spot_accum_simple'
        )
        """
    )
    conn.commit()
    conn.close()
    pnl = compute_spot_order_pnl(db, mark_prices={"ETHUSDT": 1100.0})
    assert pnl["b1"]["strategy"] == "spot_accum_simple"


def test_spot_fifo_respects_fill_time_not_insert_order(tmp_path: Path) -> None:
    """Later insert with earlier updated_at still consumes oldest lot first."""
    db = tmp_path / "spot.db"
    _spot_db(
        db,
        [
            (
                "b_new",
                "2024-01-02T09:00:00+00:00",
                "2024-01-01T08:00:00+00:00",
                "ETHUSDT",
                "buy",
                "market",
                0.1,
                900.0,
                "filled",
                0.1,
                90.0,
            ),
            (
                "b_old",
                "2024-01-01T07:00:00+00:00",
                "2024-01-01T07:01:00+00:00",
                "ETHUSDT",
                "buy",
                "market",
                0.1,
                800.0,
                "filled",
                0.1,
                80.0,
            ),
            (
                "s1",
                "2024-01-01T10:00:00+00:00",
                "2024-01-01T10:01:00+00:00",
                "ETHUSDT",
                "sell",
                "market",
                0.1,
                1000.0,
                "filled",
                0.1,
                100.0,
            ),
        ],
    )
    pnl = compute_spot_order_pnl(db)
    assert pnl["s1"]["matched_buy_orders"] == ["b_old"]
    assert pnl["s1"]["realized_pnl"] == pytest.approx(20.0)
