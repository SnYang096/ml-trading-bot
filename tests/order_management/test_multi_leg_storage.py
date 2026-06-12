from __future__ import annotations

import sqlite3

import pytest

from src.order_management.multi_leg_storage import MultiLegStorage


def test_multi_leg_storage_creates_isolated_tables_and_records(tmp_path) -> None:
    db_path = tmp_path / "multi_leg.db"
    storage = MultiLegStorage(str(db_path))

    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BTCUSDT"],
        account_label="multi_leg_testnet",
    )
    order_id = storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "leg_1_entry",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "position_side": "LONG",
            "order_type": "limit",
            "purpose": "entry",
            "quantity": 0.01,
            "price": 100000.0,
            "client_order_id": "cg_1",
            "exchange_order_id": "123",
            "status": "open",
        }
    )
    leg_id = storage.upsert_position(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "leg_id": "leg_1",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 100000.0,
            "quantity": 0.01,
            "status": "open",
        }
    )
    event_id = storage.record_execution_report(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "symbol": "BTCUSDT",
            "order_id": "123",
            "client_order_id": "cg_1",
            "status": "FILLED",
            "execution_type": "TRADE",
        }
    )
    snap_id = storage.record_reconciliation_snapshot(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "symbol": "BTCUSDT",
            "ok": True,
        }
    )
    storage.finish_run(run_id)

    assert order_id == "leg_1_entry"
    assert leg_id == "leg_1"
    assert event_id.startswith("mle_")
    assert snap_id.startswith("mlr_")

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM multi_leg_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM multi_leg_orders").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM multi_leg_positions").fetchone()[0] == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM multi_leg_execution_reports").fetchone()[
                0
            ]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM multi_leg_reconciliation_snapshots"
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_get_open_orders_for_reconcile_filters_terminal(tmp_path) -> None:
    db_path = tmp_path / "multi_leg.db"
    storage = MultiLegStorage(str(db_path))
    run_id = storage.create_run(
        mode="testnet", strategies=["chop_grid"], symbols=["BNBUSDT"]
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "open_1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "90489849398",
            "client_order_id": "cg_16738f8fae98",
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "stale_l2",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "90414533226",
            "client_order_id": "cg_5b030aa6a6ac",
            "status": "expired",
        }
    )

    rows = storage.get_open_orders_for_reconcile(strategy="chop_grid", symbol="BNBUSDT")
    assert len(rows) == 1
    assert rows[0]["local_order_id"] == "open_1"


def test_apply_execution_report_reopens_expired_order_and_clears_error(
    tmp_path,
) -> None:
    db_path = tmp_path / "multi_leg.db"
    storage = MultiLegStorage(str(db_path))
    run_id = storage.create_run(
        mode="testnet", strategies=["chop_grid"], symbols=["BNBUSDT"]
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "stale_l2",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "90414533226",
            "client_order_id": "cg_5b030aa6a6ac",
            "status": "expired",
            "error_message": "exchange_order_missing",
        }
    )

    changed = storage.apply_execution_report(
        {
            "order_id": "90414533226",
            "client_order_id": "cg_5b030aa6a6ac",
            "status": "open",
            "event_time": "2026-05-22T00:00:00+00:00",
        }
    )
    assert changed == 1

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT status, canceled_at, error_message FROM multi_leg_orders "
            "WHERE local_order_id = 'stale_l2'"
        ).fetchone()
        assert row == ("open", None, None)
    finally:
        conn.close()


def test_close_absent_positions_marks_stale_open_rows_closed(tmp_path) -> None:
    db_path = tmp_path / "multi_leg.db"
    storage = MultiLegStorage(str(db_path))
    storage.upsert_position(
        {
            "run_id": "run_1",
            "strategy": "chop_grid",
            "leg_id": "stale_leg",
            "symbol": "BNBUSDT",
            "side": "LONG",
            "entry_price": 650.0,
            "quantity": 0.1,
            "status": "open",
        }
    )
    storage.upsert_position(
        {
            "run_id": "run_1",
            "strategy": "chop_grid",
            "leg_id": "active_leg",
            "symbol": "BNBUSDT",
            "side": "LONG",
            "entry_price": 640.0,
            "quantity": 0.1,
            "status": "open",
        }
    )

    changed = storage.close_absent_positions(
        strategy="chop_grid",
        symbol="BNBUSDT",
        active_leg_ids=["active_leg"],
        run_id="run_2",
    )

    assert changed == 1
    conn = sqlite3.connect(db_path)
    try:
        rows = dict(
            conn.execute("SELECT leg_id, status FROM multi_leg_positions").fetchall()
        )
        assert rows == {"active_leg": "open", "stale_leg": "closed"}
    finally:
        conn.close()


def test_apply_execution_report_does_not_zero_market_fill(tmp_path) -> None:
    """User-stream z=0 after place-time fill must not erase filled_quantity."""
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    storage.upsert_order(
        {
            "local_order_id": "trend_entry",
            "run_id": "run",
            "strategy": "trend_scalp",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "order_type": "market",
            "purpose": "entry",
            "quantity": 0.126,
            "exchange_order_id": "8389766200983343994",
            "client_order_id": "dat_23ac20a34a6c",
            "status": "closed",
            "filled_quantity": 0.126,
            "average_price": 1685.77,
        }
    )
    assert (
        storage.apply_execution_report(
            {
                "order_id": "8389766200983343994",
                "client_order_id": "dat_23ac20a34a6c",
                "status": "NEW",
                "filled_qty": 0.0,
            }
        )
        == 1
    )
    conn = sqlite3.connect(storage.db_path)
    try:
        row = conn.execute(
            "SELECT filled_quantity, average_price FROM multi_leg_orders "
            "WHERE local_order_id = 'trend_entry'"
        ).fetchone()
        assert float(row[0]) == pytest.approx(0.126)
        assert float(row[1]) == pytest.approx(1685.77)
    finally:
        conn.close()


def test_legacy_db_gets_position_side_column(tmp_path) -> None:
    db_path = tmp_path / "legacy_multi_leg.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE multi_leg_orders (
            local_order_id TEXT PRIMARY KEY,
            run_id TEXT,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            leg_id TEXT,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            purpose TEXT,
            quantity REAL NOT NULL,
            price REAL,
            stop_price REAL,
            client_order_id TEXT,
            exchange_order_id TEXT,
            status TEXT NOT NULL,
            raw_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

    MultiLegStorage(str(db_path))

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(multi_leg_orders)")}
        assert "position_side" in cols
    finally:
        conn.close()


def test_lookup_order_purpose_by_exchange_and_client_id(tmp_path) -> None:
    db_path = tmp_path / "multi_leg.db"
    storage = MultiLegStorage(str(db_path))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["SOLUSDT"],
        account_label="multi_leg_testnet",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "SOLUSDT_grid_L1_sl",
            "leg_id": "SOLUSDT_grid_L1",
            "symbol": "SOLUSDT",
            "side": "LONG",
            "position_side": "LONG",
            "order_type": "stop_market",
            "purpose": "stop_loss",
            "quantity": 1.13,
            "client_order_id": "cg_de1f197df8e3",
            "exchange_order_id": "2000001079676592",
            "status": "filled",
        }
    )

    assert (
        storage.lookup_order_purpose(exchange_order_id="2000001079676592")
        == "stop_loss"
    )
    assert (
        storage.lookup_order_purpose(client_order_id="cg_de1f197df8e3") == "stop_loss"
    )
    assert storage.lookup_order_purpose(leg_id="SOLUSDT_grid_L1_sl") == "stop_loss"
    assert storage.lookup_order_purpose(exchange_order_id="missing") is None
