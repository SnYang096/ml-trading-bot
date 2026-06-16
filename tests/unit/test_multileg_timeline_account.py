"""Unit tests for multileg timeline account (mock-wallet sourced equity)."""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.multileg_timeline_account import MultilegTimelineAccount
from src.order_management.mock_binance_api import MockBinanceAPI


def test_no_side_ledger_attribute():
    """The misleading side ledger was removed; equity is wallet-sourced only."""
    mock = MockBinanceAPI(initial_wallet_usdt=10000.0)
    acct = MultilegTimelineAccount(initial_equity=10000.0, mock=mock)
    assert not hasattr(acct, "ledger")
    assert acct.current == 10000.0


def test_summary_fees_from_mock_and_realized_from_wallet():
    """total_fees comes from the mock (real per-fill fees); realized = equity delta."""
    mock = MockBinanceAPI(initial_wallet_usdt=10000.0, fee_bps=4.0)
    mock.set_price("BTCUSDT", 50000.0)
    acct = MultilegTimelineAccount(initial_equity=10000.0, mock=mock)

    # One real round-trip through the mock updates wallet + accumulates fees.
    mock._apply_open(
        symbol="BTCUSDT", position_side="LONG", qty=0.1, fill_price=50000.0
    )
    mock.set_price("BTCUSDT", 51000.0)
    mock._apply_reduce(
        symbol="BTCUSDT", position_side="LONG", qty=0.1, fill_price=51000.0
    )

    summary = acct.to_summary()
    assert "ledger_realized_pnl" not in summary
    # entry fee 5000*4bps=2.0 + exit fee 5100*4bps=2.04
    assert abs(summary["total_fees_usdt"] - 4.04) < 1e-6
    # realized = wallet delta = gross 100 - fees 4.04
    assert abs(summary["realized_pnl"] - (100.0 - 4.04)) < 1e-6


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
