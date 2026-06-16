"""Unit tests for multileg timeline account ledger integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.multileg_timeline_account import MultilegTimelineAccount
from src.order_management.grid_execution_adapter import MultiLegExecutionResult
from src.order_management.mock_binance_api import MockBinanceAPI


def test_record_execution_results_open_and_close():
    mock = MockBinanceAPI(initial_wallet_usdt=10000.0)
    mock.set_price("BTCUSDT", 50000.0)
    acct = MultilegTimelineAccount(initial_equity=10000.0, mock=mock)

    open_res = MultiLegExecutionResult(
        action="place",
        symbol="BTCUSDT",
        status="filled",
        raw={
            "local_order_id": "lot-1",
            "side": "BUY",
            "quantity": 0.1,
            "average_price": 50000.0,
        },
    )
    acct.record_execution_results([open_res], strategy="chop_grid", fee_bps=4.0)
    assert acct.ledger.get_lot("lot-1") is not None

    mock.set_price("BTCUSDT", 51000.0)
    exit_res = MultiLegExecutionResult(
        action="market_exit",
        symbol="BTCUSDT",
        status="filled",
        raw={
            "local_order_id": "lot-1",
            "side": "SELL",
            "quantity": 0.1,
            "average_price": 51000.0,
        },
    )
    acct.record_execution_results([exit_res], strategy="chop_grid", fee_bps=4.0)
    assert acct.ledger.get_lot("lot-1") is None
    assert acct.ledger.realized_pnl_usdt != 0.0


def test_sync_engine_realized_bridge_credits_delta():
    mock = MockBinanceAPI(initial_wallet_usdt=10000.0)
    acct = MultilegTimelineAccount(initial_equity=10000.0, mock=mock)
    eng = MagicMock()
    eng.bar_simulation = True
    eng.state.realized_pnl = 25.0
    engines = {"BTCUSDT": {"chop": eng}}
    acct.sync_engine_realized_bridge(engines)
    assert mock.wallet_usdt == 10025.0
    eng.state.realized_pnl = 40.0
    acct.sync_engine_realized_bridge(engines)
    assert mock.wallet_usdt == 10040.0


def test_sync_engine_bridge_does_not_double_count_ledger():
    """Engine realized_pnl bridge must NOT touch ledger — lot close already records it.

    Regression test: previously sync_engine_realized_bridge added the same
    delta to ledger.realized_pnl_usdt, causing TP/SL exits to be double-counted.
    """
    mock = MockBinanceAPI(initial_wallet_usdt=10000.0)
    mock.set_price("BTCUSDT", 50000.0)
    acct = MultilegTimelineAccount(initial_equity=10000.0, mock=mock)

    # 1. Open a lot via execution results (simulates orchestrator entry)
    open_res = MultiLegExecutionResult(
        action="place",
        symbol="BTCUSDT",
        status="filled",
        raw={
            "local_order_id": "lot-dbl-1",
            "side": "BUY",
            "quantity": 0.1,
            "average_price": 50000.0,
        },
    )
    acct.record_execution_results([open_res], strategy="chop_grid", fee_bps=4.0)
    ledger_pnl_after_open = acct.ledger.realized_pnl_usdt

    # 2. Close the lot (simulates TP/SL fill via pending orders)
    mock.set_price("BTCUSDT", 51000.0)
    exit_res = MultiLegExecutionResult(
        action="market_exit",
        symbol="BTCUSDT",
        status="filled",
        raw={
            "local_order_id": "lot-dbl-1",
            "side": "SELL",
            "quantity": 0.1,
            "average_price": 51000.0,
        },
    )
    acct.record_execution_results([exit_res], strategy="chop_grid", fee_bps=4.0)
    ledger_pnl_after_close = acct.ledger.realized_pnl_usdt
    assert ledger_pnl_after_close != ledger_pnl_after_open  # lot close added P&L

    # 3. Now simulate engine also tracking the same P&L (bar_simulation=True)
    eng = MagicMock()
    eng.bar_simulation = True
    eng.state.realized_pnl = ledger_pnl_after_close  # engine mirrors same P&L
    engines = {"BTCUSDT": {"chop": eng}}
    acct.sync_engine_realized_bridge(engines)

    # KEY ASSERTION: ledger should NOT change — the P&L was already recorded
    assert acct.ledger.realized_pnl_usdt == ledger_pnl_after_close
    # Wallet should get the delta (engine bridge is for wallet only)
    assert mock.wallet_usdt != 10000.0  # wallet was credited


def test_drawdown_and_halt():
    mock = MockBinanceAPI(initial_wallet_usdt=10000.0)
    acct = MultilegTimelineAccount(initial_equity=10000.0, mock=mock)
    acct.peak_equity = 10000.0
    mock.wallet_usdt = 7500.0
    acct.on_bar_close(
        day_key="2026-01-01",
        max_dd=0.20,
        daily_loss_limit=600.0,
        ts_label="2026-01-01",
    )
    assert acct.halted is True
    assert "dd>" in acct.halt_reason
