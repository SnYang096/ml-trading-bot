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
