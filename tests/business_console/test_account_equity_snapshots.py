"""Tests for daily account equity snapshots."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mlbot_console.services.account_equity_snapshots import (
    capture_daily_account_snapshots,
    load_daily_equity_curves,
    merge_live_into_curves,
)
from mlbot_console.services.account_summary import build_wallet_equity_curves


_MOCK_LEDGER = {
    "accounts": [
        {
            "scope": "trend",
            "ok": True,
            "wallet_balance_usdt": 900.0,
            "equity_usdt": 920.0,
            "account_unrealized_pnl_usdt": 20.0,
        },
        {
            "scope": "spot",
            "ok": True,
            "wallet_balance_usdt": 100.0,
            "equity_usdt": 105.0,
            "unrealized_pnl_usdt": 5.0,
        },
    ],
    "totals": {
        "wallet_balance_usdt": 1000.0,
        "equity_usdt": 1025.0,
        "exchange_unrealized_pnl_usdt": 25.0,
        "accounts_ok": 2,
    },
}


@patch(
    "mlbot_console.services.account_equity_snapshots.build_exchange_ledger",
    return_value=_MOCK_LEDGER,
)
def test_capture_daily_account_snapshots_upserts(mock_ledger, tmp_path: Path) -> None:
    db = tmp_path / "account_equity.db"
    r1 = capture_daily_account_snapshots(db, snapshot_date="2026-06-10")
    assert r1["rows_written"] == 3  # trend, spot, all
    curves = load_daily_equity_curves(db, scope="all")
    assert len(curves["balance"]) == 1
    assert curves["balance"][0]["value_usdt"] == pytest.approx(1000.0)
    assert curves["equity"][0]["value_usdt"] == pytest.approx(1025.0)

    updated = dict(_MOCK_LEDGER)
    updated["totals"] = dict(updated["totals"])
    updated["totals"]["wallet_balance_usdt"] = 1100.0
    updated["totals"]["equity_usdt"] = 1130.0
    mock_ledger.return_value = updated
    r2 = capture_daily_account_snapshots(db, snapshot_date="2026-06-10")
    assert r2["rows_written"] == 3
    curves2 = load_daily_equity_curves(db, scope="all")
    assert curves2["balance"][0]["value_usdt"] == pytest.approx(1100.0)


def test_merge_live_into_curves_refreshes_today() -> None:
    base = {
        "balance": [{"date": "2026-06-10", "value_usdt": 1000.0}],
        "equity": [{"date": "2026-06-10", "value_usdt": 1020.0}],
        "note": "来自 account_equity 日快照",
    }
    merged = merge_live_into_curves(base, wallet_usdt=1005.0, equity_usdt=1030.0)
    assert merged["balance"][-1]["value_usdt"] == pytest.approx(1005.0)
    assert merged["equity"][-1]["value_usdt"] == pytest.approx(1030.0)


@patch(
    "mlbot_console.services.account_equity_snapshots.build_exchange_ledger",
    return_value=_MOCK_LEDGER,
)
def test_build_wallet_equity_curves_prefers_snapshots(
    mock_ledger, tmp_path: Path
) -> None:
    db = tmp_path / "account_equity.db"
    capture_daily_account_snapshots(db, snapshot_date="2026-06-09")
    capture_daily_account_snapshots(db, snapshot_date="2026-06-10")

    daily = [{"date": "2026-06-10", "pnl": 50.0}]
    curves = build_wallet_equity_curves(
        daily,
        wallet_usdt=1100.0,
        equity_usdt=1130.0,
        snapshot_db=db,
    )
    assert len(curves["balance"]) >= 2
    assert curves["balance"][0]["value_usdt"] == pytest.approx(1000.0)
    assert curves["balance"][-1]["value_usdt"] == pytest.approx(1100.0)
    assert "日快照" in str(curves.get("note") or "")
