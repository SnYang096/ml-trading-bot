from __future__ import annotations

import sqlite3

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
