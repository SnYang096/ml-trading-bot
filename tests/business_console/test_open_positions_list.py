"""Tests for open_positions_list ghost position filtering."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mlbot_console.services.open_positions_list import (
    _multileg_open_leg_ids,
    _multileg_open_rows,
    _get_multileg_tp_sl_orders,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary multi-leg database with positions and orders tables."""
    db_path = tmp_path / "test_multi_leg.db"
    conn = sqlite3.connect(db_path)

    # Create multi_leg_positions table
    conn.execute(
        """
        CREATE TABLE multi_leg_positions (
            run_id TEXT,
            strategy TEXT,
            leg_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            quantity REAL,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Create multi_leg_orders table
    conn.execute(
        """
        CREATE TABLE multi_leg_orders (
            local_order_id TEXT PRIMARY KEY,
            strategy TEXT,
            purpose TEXT,
            status TEXT,
            side TEXT,
            position_side TEXT,
            symbol TEXT,
            price REAL,
            quantity REAL,
            filled_quantity REAL,
            average_price REAL,
            leg_id TEXT,
            exchange_order_id TEXT,
            filled_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


class TestMultilegOpenLegIds:
    """Tests for _multileg_open_leg_ids function."""

    def test_returns_empty_set_when_no_positions(self, temp_db: Path) -> None:
        """When no open positions exist, return empty set."""
        result = _multileg_open_leg_ids(temp_db, "HYPEUSDT")
        assert result == set()

    def test_returns_leg_ids_with_open_status(self, temp_db: Path) -> None:
        """Return leg_ids that have status='open'."""
        conn = sqlite3.connect(temp_db)
        conn.executemany(
            "INSERT INTO multi_leg_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # open position
                (
                    "r1",
                    "chop_grid",
                    "leg_open",
                    "HYPEUSDT",
                    "SHORT",
                    64.0,
                    77.12,
                    "open",
                    "2026-06-15",
                ),
                # closed position
                (
                    "r1",
                    "chop_grid",
                    "leg_closed",
                    "HYPEUSDT",
                    "LONG",
                    63.0,
                    75.88,
                    "closed",
                    "2026-06-14",
                ),
                # open position for different symbol
                (
                    "r1",
                    "chop_grid",
                    "leg_btc",
                    "BTCUSDT",
                    "LONG",
                    65000.0,
                    0.01,
                    "open",
                    "2026-06-15",
                ),
            ],
        )
        conn.commit()
        conn.close()

        result = _multileg_open_leg_ids(temp_db, "HYPEUSDT")
        assert result == {"leg_open"}

    def test_filters_by_symbol(self, temp_db: Path) -> None:
        """When symbol is specified, only return leg_ids for that symbol."""
        conn = sqlite3.connect(temp_db)
        conn.executemany(
            "INSERT INTO multi_leg_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "r1",
                    "chop_grid",
                    "leg_hype",
                    "HYPEUSDT",
                    "SHORT",
                    64.0,
                    77.12,
                    "open",
                    "2026-06-15",
                ),
                (
                    "r1",
                    "chop_grid",
                    "leg_btc",
                    "BTCUSDT",
                    "LONG",
                    65000.0,
                    0.01,
                    "open",
                    "2026-06-15",
                ),
            ],
        )
        conn.commit()
        conn.close()

        result = _multileg_open_leg_ids(temp_db, "HYPEUSDT")
        assert result == {"leg_hype"}
        # All symbols
        result_all = _multileg_open_leg_ids(temp_db, "*")
        assert result_all == {"leg_hype", "leg_btc"}

    def test_case_insensitive_symbol(self, temp_db: Path) -> None:
        """Symbol matching should be case-insensitive."""
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO multi_leg_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "r1",
                "chop_grid",
                "leg_hype",
                "HYPEUSDT",
                "SHORT",
                64.0,
                77.12,
                "open",
                "2026-06-15",
            ),
        )
        conn.commit()
        conn.close()

        assert _multileg_open_leg_ids(temp_db, "hypeusdt") == {"leg_hype"}
        assert _multileg_open_leg_ids(temp_db, "HYPEUSDT") == {"leg_hype"}


class TestGhostPositionFiltering:
    """Tests that _multileg_open_rows filters ghost positions correctly."""

    def test_entry_order_without_matching_position_is_filtered(
        self, temp_db: Path
    ) -> None:
        """Ghost entry order (filled but position closed) should not appear."""
        conn = sqlite3.connect(temp_db)
        # Insert a filled entry order (ghost)
        conn.execute(
            """
            INSERT INTO multi_leg_orders
            (local_order_id, strategy, purpose, status, side, position_side,
             symbol, price, quantity, filled_quantity, leg_id, exchange_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "order_ghost",
                "chop_grid",
                "entry",
                "filled",
                "SELL",
                "SHORT",
                "HYPEUSDT",
                64.0,
                77.12,
                77.12,
                "ghost_leg",
                "ex_ghost",
            ),
        )
        # No matching open position in multi_leg_positions
        conn.commit()
        conn.close()

        result = _multileg_open_leg_ids(temp_db, "HYPEUSDT")
        assert result == set()  # No open positions

    def test_entry_order_with_matching_position_is_kept(self, temp_db: Path) -> None:
        """Entry order with matching open position should appear."""
        conn = sqlite3.connect(temp_db)
        # Insert open position
        conn.execute(
            "INSERT INTO multi_leg_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "r1",
                "chop_grid",
                "active_leg",
                "HYPEUSDT",
                "SHORT",
                64.0,
                77.12,
                "open",
                "2026-06-15",
            ),
        )
        # Insert matching entry order
        conn.execute(
            """
            INSERT INTO multi_leg_orders
            (local_order_id, strategy, purpose, status, side, position_side,
             symbol, price, quantity, filled_quantity, leg_id, exchange_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "order_active",
                "chop_grid",
                "entry",
                "filled",
                "SELL",
                "SHORT",
                "HYPEUSDT",
                64.0,
                77.12,
                77.12,
                "active_leg",
                "ex_active",
            ),
        )
        conn.commit()
        conn.close()

        result = _multileg_open_leg_ids(temp_db, "HYPEUSDT")
        assert result == {"active_leg"}

    def test_closed_entry_order_is_filtered(self, temp_db: Path) -> None:
        """Pruned entry orders marked closed must not appear as open positions."""
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO multi_leg_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "r1",
                "chop_grid",
                "active_leg",
                "HYPEUSDT",
                "LONG",
                63.0,
                75.88,
                "open",
                "2026-06-15",
            ),
        )
        conn.executemany(
            """
            INSERT INTO multi_leg_orders
            (local_order_id, strategy, purpose, status, side, position_side,
             symbol, price, quantity, filled_quantity, leg_id, exchange_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "order_active",
                    "chop_grid",
                    "entry",
                    "filled",
                    "BUY",
                    "LONG",
                    "HYPEUSDT",
                    63.0,
                    75.88,
                    75.88,
                    "active_leg",
                    "ex_active",
                ),
                (
                    "order_ghost",
                    "chop_grid",
                    "entry",
                    "closed",
                    "BUY",
                    "LONG",
                    "HYPEUSDT",
                    59.999,
                    79.98,
                    79.98,
                    "ghost_leg",
                    "ex_ghost",
                ),
            ],
        )
        conn.commit()
        conn.close()

        rows = _multileg_open_rows(
            temp_db,
            symbol="HYPEUSDT",
            mark_prices={"HYPEUSDT": 67.0},
            pending_exits={},
        )
        leg_ids = {str(r.get("leg") or "") for r in rows}
        assert leg_ids == {"active_leg"}

    def test_filled_entry_without_open_position_is_filtered(
        self, temp_db: Path
    ) -> None:
        """Ghost filled entry with no open position row must not appear."""
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """
            INSERT INTO multi_leg_orders
            (local_order_id, strategy, purpose, status, side, position_side,
             symbol, price, quantity, filled_quantity, leg_id, exchange_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "order_ghost",
                "chop_grid",
                "entry",
                "filled",
                "BUY",
                "LONG",
                "HYPEUSDT",
                59.999,
                79.98,
                79.98,
                "ghost_leg",
                "ex_ghost",
            ),
        )
        # Closed position row proves positions table is in use; leg is not open.
        conn.execute(
            "INSERT INTO multi_leg_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "r1",
                "chop_grid",
                "ghost_leg",
                "HYPEUSDT",
                "LONG",
                59.999,
                79.98,
                "closed",
                "2026-06-14",
            ),
        )
        conn.commit()
        conn.close()

        rows = _multileg_open_rows(
            temp_db,
            symbol="HYPEUSDT",
            mark_prices={"HYPEUSDT": 67.0},
            pending_exits={},
        )
        assert rows == []


class TestMultilegTpSlOrders:
    """Tests for _get_multileg_tp_sl_orders function."""

    def test_returns_tp_and_sl_orders(self, temp_db: Path) -> None:
        """Should return TP and SL orders grouped by leg_id."""
        conn = sqlite3.connect(temp_db)
        conn.executemany(
            """
            INSERT INTO multi_leg_orders
            (local_order_id, strategy, purpose, status, side, position_side,
             symbol, price, quantity, filled_quantity, leg_id, exchange_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                # TP order for leg_1 (status=open, not filled)
                (
                    "tp_1",
                    "chop_grid",
                    "take_profit",
                    "open",
                    "BUY",
                    "SHORT",
                    "HYPEUSDT",
                    64.5,
                    77.12,
                    0,
                    "leg_1",
                    "ex_tp_1",
                ),
                # SL order for leg_1 (status=open, not filled)
                (
                    "sl_1",
                    "chop_grid",
                    "stop_loss",
                    "open",
                    "BUY",
                    "SHORT",
                    "HYPEUSDT",
                    62.0,
                    77.12,
                    0,
                    "leg_1",
                    "ex_sl_1",
                ),
                # TP order for leg_2 (status=open)
                (
                    "tp_2",
                    "chop_grid",
                    "take_profit",
                    "open",
                    "SELL",
                    "LONG",
                    "HYPEUSDT",
                    66.0,
                    75.88,
                    0,
                    "leg_2",
                    "ex_tp_2",
                ),
            ],
        )
        conn.commit()
        conn.close()

        result = _get_multileg_tp_sl_orders(temp_db, "HYPEUSDT")

        assert "leg_1" in result
        assert len(result["leg_1"]) == 2
        tp = next(o for o in result["leg_1"] if o["order_type"] == "TAKE_PROFIT")
        assert tp["price"] == 64.5
        assert tp["order_id"] == "ex_tp_1"

        sl = next(o for o in result["leg_1"] if o["order_type"] == "STOP_LOSS")
        assert sl["price"] == 62.0
        assert sl["order_id"] == "ex_sl_1"

    def test_returns_empty_dict_when_no_orders(self, temp_db: Path) -> None:
        """When no TP/SL orders exist, return empty dict."""
        result = _get_multileg_tp_sl_orders(temp_db, "HYPEUSDT")
        assert result == {}
