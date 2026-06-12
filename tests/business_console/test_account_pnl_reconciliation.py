"""Tests for PnL vs exchange equity reconciliation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mlbot_console.services.account_pnl_reconciliation import (
    reconcile_pnl_vs_exchange,
    reconcile_scope_pnl,
)


def test_reconcile_scope_pnl_skips_spot_unrealized_mismatch() -> None:
    issues = reconcile_scope_pnl(
        "spot",
        scope_block={
            "scope": "spot",
            "realized_pnl": 0.0,
            "unrealized_pnl": 12.5,
            "open_positions": 0,
            "closed_trades": 3,
        },
        exchange_row={
            "ok": True,
            "equity_usdt": 100.0,
            "wallet_balance_usdt": 100.0,
            "unrealized_pnl_usdt": 0.0,
            "holdings_value_usdt": 100.0,
        },
        strategy_rows=[],
    )
    assert not any(i["kind"] == "unrealized_pnl_mismatch" for i in issues)


def test_reconcile_scope_pnl_flags_unrealized_mismatch() -> None:
    issues = reconcile_scope_pnl(
        "multi_leg",
        scope_block={
            "scope": "multi_leg",
            "realized_pnl": -7.0,
            "unrealized_pnl": -8.0,
            "open_positions": 0,
            "closed_trades": 41,
        },
        exchange_row={
            "ok": True,
            "equity_usdt": 500.0,
            "wallet_balance_usdt": 510.0,
            "unrealized_pnl_usdt": 2.0,
        },
        strategy_rows=[
            {
                "scope": "multi_leg",
                "strategy": "trend_scalp",
                "realized_pnl": -7.0,
                "unrealized_pnl": -8.0,
                "open_positions": 0,
                "closed_trades": 41,
            }
        ],
    )
    kinds = {i["kind"] for i in issues}
    assert "unrealized_pnl_mismatch" in kinds
    assert "orphan_local_unrealized" in kinds
    assert "strategy_orphan_unrealized" in kinds


def test_reconcile_scope_pnl_symbol_filter_uses_symbol_exchange_upnl() -> None:
    """Single-symbol reconcile should not compare against full-account unrealized."""
    issues = reconcile_scope_pnl(
        "multi_leg",
        scope_block={
            "scope": "multi_leg",
            "realized_pnl": 0.0,
            "unrealized_pnl": -2.0,
            "open_positions": 1,
            "closed_trades": 0,
        },
        exchange_row={
            "ok": True,
            "equity_usdt": 500.0,
            "wallet_balance_usdt": 510.0,
            "unrealized_pnl_usdt": -2.0,
            "unrealized_pnl_basis": "symbol",
        },
        strategy_rows=[],
        symbol="XRPUSDT",
    )
    assert issues == []


def test_reconcile_scope_pnl_symbol_filter_skips_equity_identity() -> None:
    issues = reconcile_scope_pnl(
        "trend",
        scope_block={
            "scope": "trend",
            "realized_pnl": 0.0,
            "unrealized_pnl": 1.0,
            "open_positions": 1,
            "closed_trades": 0,
        },
        exchange_row={
            "ok": True,
            "equity_usdt": 1000.0,
            "wallet_balance_usdt": 990.0,
            "unrealized_pnl_usdt": 1.0,
        },
        strategy_rows=[],
        symbol="XRPUSDT",
    )
    kinds = {i["kind"] for i in issues}
    assert "exchange_equity_identity" not in kinds


def test_reconcile_pnl_vs_exchange_symbol_filter_skips_global_mismatch() -> None:
    fake_summary = {
        "symbol": "XRPUSDT",
        "lookback_days": 0,
        "totals": {
            "realized_pnl": 0.0,
            "unrealized_pnl": -2.0,
            "open_positions": 1,
            "closed_trades": 0,
        },
        "scopes": [
            {
                "scope": "multi_leg",
                "realized_pnl": 0.0,
                "unrealized_pnl": -2.0,
                "open_positions": 1,
                "closed_trades": 0,
            },
        ],
        "strategies": [],
        "exchange_ledger": {
            "symbol": "XRPUSDT",
            "accounts": [
                {
                    "scope": "multi_leg",
                    "ok": True,
                    "equity_usdt": 500.0,
                    "wallet_balance_usdt": 510.0,
                    "unrealized_pnl_usdt": -2.0,
                    "unrealized_pnl_basis": "symbol",
                },
            ],
            "totals": {
                "exchange_unrealized_pnl_usdt": 50.0,
                "accounts_ok": 1,
            },
            "fetched_at": "2026-06-09T00:00:00+00:00",
        },
    }
    with patch(
        "mlbot_console.services.account_pnl_reconciliation.build_account_summary",
        return_value=fake_summary,
    ):
        report = reconcile_pnl_vs_exchange(
            trend_db=Path("/dev/null"),
            spot_db=Path("/dev/null"),
            spot_ledger_db=Path("/dev/null"),
            multi_leg_db=Path("/dev/null"),
            feature_bus_root=Path("/dev/null"),
            symbol="XRPUSDT",
        )
    kinds = {i["kind"] for i in report["issues"]}
    assert "global_unrealized_pnl_mismatch" not in kinds
    assert report["ok"] is True


def test_reconcile_scope_pnl_ok_when_aligned() -> None:
    issues = reconcile_scope_pnl(
        "trend",
        scope_block={
            "scope": "trend",
            "realized_pnl": 10.0,
            "unrealized_pnl": 1.5,
            "open_positions": 1,
            "closed_trades": 5,
        },
        exchange_row={
            "ok": True,
            "equity_usdt": 1000.0,
            "wallet_balance_usdt": 998.5,
            "unrealized_pnl_usdt": 1.5,
        },
        strategy_rows=[],
    )
    assert issues == []


def test_reconcile_scope_pnl_flags_trend_exchange_float_without_local_open() -> None:
    issues = reconcile_scope_pnl(
        "trend",
        scope_block={
            "scope": "trend",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_positions": 0,
            "closed_trades": 0,
        },
        exchange_row={
            "ok": True,
            "equity_usdt": 12620.0,
            "wallet_balance_usdt": 12634.0,
            "unrealized_pnl_usdt": -13.95,
        },
        strategy_rows=[],
    )
    kinds = {i["kind"] for i in issues}
    assert "unrealized_pnl_mismatch" in kinds
    assert "exchange_position_not_in_local_db" in kinds


def test_reconcile_pnl_vs_exchange_integration(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    fake_ledger = {
        "accounts": [
            {
                "scope": "trend",
                "ok": True,
                "equity_usdt": 100.0,
                "wallet_balance_usdt": 98.0,
                "unrealized_pnl_usdt": 5.0,
            },
            {
                "scope": "spot",
                "ok": True,
                "equity_usdt": 50.0,
                "wallet_balance_usdt": 50.0,
                "holdings_value_usdt": 0.0,
            },
            {
                "scope": "multi_leg",
                "ok": True,
                "equity_usdt": 200.0,
                "wallet_balance_usdt": 200.0,
                "unrealized_pnl_usdt": 0.0,
            },
        ],
        "totals": {
            "equity_usdt": 350.0,
            "wallet_balance_usdt": 348.0,
            "exchange_unrealized_pnl_usdt": 5.0,
            "accounts_ok": 3,
        },
        "fetched_at": "2026-06-09T00:00:00+00:00",
    }
    fake_summary = {
        "symbol": "ALL",
        "lookback_days": 0,
        "totals": {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_positions": 0,
            "closed_trades": 0,
        },
        "scopes": [
            {
                "scope": "trend",
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_positions": 0,
                "closed_trades": 0,
            },
            {
                "scope": "spot",
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_positions": 0,
                "closed_trades": 0,
                "exchange": {"ledger_holdings_value_usdt": 0.0},
            },
            {
                "scope": "multi_leg",
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_positions": 0,
                "closed_trades": 0,
            },
        ],
        "strategies": [],
        "exchange_ledger": fake_ledger,
    }

    with patch(
        "mlbot_console.services.account_pnl_reconciliation.build_account_summary",
        return_value=fake_summary,
    ):
        report = reconcile_pnl_vs_exchange(
            trend_db=trend_db,
            spot_db=spot_db,
            spot_ledger_db=spot_ledger_db,
            multi_leg_db=multi_leg_db,
            feature_bus_root=bus_root,
        )

    kinds = {i["kind"] for i in report["issues"]}
    assert "unrealized_pnl_mismatch" in kinds
    assert report["ok"] is False
