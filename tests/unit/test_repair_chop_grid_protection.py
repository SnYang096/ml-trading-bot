from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.repair_chop_grid_protection import _db_entry_tp_actions


def _make_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE multi_leg_orders (
            local_order_id TEXT,
            symbol TEXT,
            side TEXT,
            purpose TEXT,
            status TEXT,
            quantity REAL,
            filled_quantity REAL,
            price REAL,
            average_price REAL,
            exchange_order_id TEXT,
            client_order_id TEXT,
            leg_id TEXT
        )
        """
    )
    con.commit()
    con.close()


def _insert(path: Path, **row: object) -> None:
    con = sqlite3.connect(str(path))
    cols = [
        "local_order_id",
        "symbol",
        "side",
        "purpose",
        "status",
        "quantity",
        "filled_quantity",
        "price",
        "average_price",
        "exchange_order_id",
        "client_order_id",
        "leg_id",
    ]
    con.execute(
        f"INSERT INTO multi_leg_orders ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
        [row.get(c) for c in cols],
    )
    con.commit()
    con.close()


def test_db_entry_repair_plans_only_missing_s2_tp(tmp_path: Path) -> None:
    db = tmp_path / "orders.db"
    _make_db(db)
    grid = "BNBUSDT_2026-05-19 08:40:00+00:00"
    _insert(
        db,
        local_order_id=f"{grid}_L1",
        symbol="BNBUSDT",
        side="BUY",
        purpose="entry",
        status="filled",
        quantity=0.31,
        filled_quantity=0.31,
        price=637.11,
        average_price=637.11,
    )
    _insert(
        db,
        local_order_id=f"{grid}_S1",
        symbol="BNBUSDT",
        side="SELL",
        purpose="entry",
        status="filled",
        quantity=0.31,
        filled_quantity=0.31,
        price=649.99,
        average_price=649.99,
    )
    _insert(
        db,
        local_order_id=f"{grid}_S1_tp",
        symbol="BNBUSDT",
        side="SHORT",
        purpose="take_profit",
        status="open",
        quantity=0.31,
        price=643.55,
        exchange_order_id="90489849398",
        client_order_id="cg_s1tp",
        leg_id=f"{grid}_S1",
    )
    _insert(
        db,
        local_order_id=f"{grid}_S2",
        symbol="BNBUSDT",
        side="SELL",
        purpose="entry",
        status="filled",
        quantity=0.31,
        filled_quantity=0.31,
        price=656.42,
        average_price=656.42,
    )

    actions = _db_entry_tp_actions(
        db_path=db,
        symbol="BNBUSDT",
        grid_id=grid,
        spacing=6.4355,
        positions=[{"symbol": "BNB/USDT:USDT", "side": "short", "size": 0.62}],
        open_orders=[
            {
                "order_id": "90489849398",
                "client_order_id": "cg_s1tp",
                "side": "buy",
                "price": 643.55,
                "remaining": 0.31,
                "position_side": "SHORT",
            }
        ],
    )

    assert len(actions) == 1
    assert actions[0]["order_id"] == f"{grid}_S2_tp"
    assert actions[0]["leg_id"] == f"{grid}_S2"
    assert actions[0]["side"] == "SHORT"
    assert actions[0]["quantity"] == pytest.approx(0.31)
    assert actions[0]["price"] == pytest.approx(649.98)
