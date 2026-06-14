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


def test_reconcile_account_trend_flags_exchange_position_missing_local(
    trend_db: Path,
) -> None:
    fake_exchange = {
        "ok": True,
        "unrealized_pnl_usdt": -13.95,
        "exchange_open_positions": [
            {
                "symbol": "XRPUSDT",
                "side": "short",
                "quantity": 100.0,
                "unrealized_pnl_usdt": -13.95,
            }
        ],
        "exchange_open_position_count": 1,
    }
    with patch(
        "mlbot_console.services.account_reconciliation.fetch_scope_exchange_balance",
        return_value=fake_exchange,
    ):
        res = reconcile_account("trend", trend_db=trend_db)

    assert res["ok"] is False
    kinds = {i["kind"] for i in res["issues"]}
    assert "exchange_position_not_in_local_db" in kinds


def test_reconcile_account_trend_symbol_filter_ignores_other_symbols(
    trend_db: Path,
) -> None:
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_bnb', 'BNBUSDT', 'long',
            '2026-06-10T08:00:00+00:00', NULL,
            1.0, NULL, NULL, 'open', 'tpc', 700.0, NULL, NULL, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    fake_exchange = {
        "ok": True,
        "exchange_open_positions": [],
        "exchange_open_position_count": 0,
    }
    with patch(
        "mlbot_console.services.account_reconciliation.fetch_scope_exchange_balance",
        return_value=fake_exchange,
    ) as fetch_mock:
        res = reconcile_account("trend", trend_db=trend_db, symbol="XRPUSDT")

    fetch_mock.assert_called_once()
    assert fetch_mock.call_args.kwargs.get("symbol") == "XRPUSDT"
    assert res["ok"] is True


def test_local_trend_open_positions_skips_zero_qty_without_entry(
    trend_db: Path,
) -> None:
    import sqlite3

    from mlbot_console.services.account_reconciliation import (
        _local_trend_open_positions,
    )

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_ghost', 'ETHUSDT', 'long',
            '2026-06-10T08:00:00+00:00', NULL,
            0.0, NULL, NULL, 'open', 'tpc', 2100.0, NULL, NULL, NULL
        )
        """
    )
    conn.commit()
    conn.close()

    rows = _local_trend_open_positions(trend_db)
    assert all(r["position_id"] != "p_ghost" for r in rows)
