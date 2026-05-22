"""Tests for account reconciliation."""

from pathlib import Path
from unittest.mock import patch
import pytest

from mlbot_console.services.account_reconciliation import reconcile_account


def test_reconcile_account_spot(spot_ledger_db: Path) -> None:
    import sqlite3
    import json

    conn = sqlite3.connect(spot_ledger_db)
    positions = {
        "lot1": {
            "symbol": "BTCUSDT",
            "qty_base": 0.5,
            "vwap_entry": 60000.0,
            "entry_notional_usdt": 30000.0,
        },
    }
    conn.execute(
        "INSERT INTO state_kv (k, v) VALUES (?, ?)",
        ("positions", json.dumps(positions)),
    )
    conn.commit()
    conn.close()

    fake_exchange = {
        "ok": True,
        "holdings": [
            {"asset": "BTC", "qty": 0.51, "value_usdt": 33150.0},
        ],
    }

    with patch(
        "mlbot_console.services.account_reconciliation.fetch_scope_exchange_balance",
        return_value=fake_exchange,
    ):
        res = reconcile_account(
            "spot",
            spot_ledger_db=spot_ledger_db,
            mark_prices={"BTCUSDT": 65000.0},
        )

    assert res["ok"] is False
    assert len(res["issues"]) == 1
    issue = res["issues"][0]
    assert issue["kind"] == "qty_mismatch"
    assert issue["asset"] == "BTC"
    assert issue["exchange"] == 0.51
    assert issue["local"] == 0.5
    assert issue["delta"] == pytest.approx(0.01)
