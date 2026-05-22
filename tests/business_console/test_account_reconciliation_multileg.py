"""Multi-leg reconciliation reads engine snapshot raw_json."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from mlbot_console.services.account_reconciliation import reconcile_account


def test_reconcile_multi_leg_from_snapshot(tmp_path: Path) -> None:
    db = tmp_path / "multi_leg.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE multi_leg_reconciliation_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT,
            strategy TEXT,
            symbol TEXT,
            ok INTEGER NOT NULL,
            raw_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    report = {
        "ok": False,
        "missing_exchange_orders": [
            {"order_id": "BNBUSDT_L2", "symbol": "BNBUSDT", "side": "BUY"},
        ],
        "orphan_exchange_orders": [],
        "position_mismatches": [],
    }
    conn.execute(
        """
        INSERT INTO multi_leg_reconciliation_snapshots
        (snapshot_id, run_id, strategy, symbol, ok, raw_json)
        VALUES ('s1', 'r1', 'chop_grid', 'BNBUSDT', 0, ?)
        """,
        (json.dumps(report),),
    )
    conn.commit()
    conn.close()

    fake_exchange = {"ok": True, "equity_usdt": 1000.0}

    with patch(
        "mlbot_console.services.account_reconciliation.fetch_scope_exchange_balance",
        return_value=fake_exchange,
    ):
        res = reconcile_account("multi_leg", multi_leg_db=db)

    assert res["ok"] is False
    assert len(res["issues"]) == 1
    assert res["issues"][0]["kind"] == "missing_exchange_order"
    assert res["issues"][0]["order_id"] == "BNBUSDT_L2"
