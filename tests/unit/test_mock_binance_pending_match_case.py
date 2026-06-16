"""Unit tests for MockBinanceAPI.match_pending_orders case normalization.

Covers fatal bug #1: ``place_order`` stores lowercase ``"buy"/"sell"`` from
the grid adapter, but ``match_pending_orders`` previously compared against
uppercase ``"BUY"/"SELL"``  → zero fills in backtest.

Fix: ``str(order["side"]).upper()`` before comparison.
"""

from __future__ import annotations

import pytest

from src.order_management.mock_binance_api import MockBinanceAPI

# ------------------------------------------------------------------ helpers


def _make_api() -> MockBinanceAPI:
    api = MockBinanceAPI(initial_wallet_usdt=100_000.0)
    api.set_price("BTCUSDT", 50_000.0)
    return api


# ---------------------------------------------------------- limit orders


class TestMatchPendingOrdersCaseNormalization:
    """Verify that side comparison is case-insensitive."""

    def test_lowercase_buy_limit_matches(self) -> None:
        """Lowercase 'buy' limit should match when low <= price."""
        api = _make_api()
        # Simulate what the adapter does: store lowercase side
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "limit",
                "side": "buy",  # lowercase — adapter output
                "price": 50_000.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": False,
                "order_id": "local-buy-1",
                "trigger_price": 0.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=51_000.0, low=49_000.0)
        assert len(fills) == 1
        assert fills[0]["average_price"] == 50_000.0

    def test_lowercase_sell_limit_matches(self) -> None:
        """Lowercase 'sell' limit should match when high >= price."""
        api = _make_api()
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "limit",
                "side": "sell",  # lowercase
                "price": 51_000.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": False,
                "order_id": "local-sell-1",
                "trigger_price": 0.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=52_000.0, low=50_000.0)
        assert len(fills) == 1
        assert fills[0]["average_price"] == 51_000.0

    def test_uppercase_buy_limit_still_works(self) -> None:
        """Uppercase 'BUY' should continue working (backward compat)."""
        api = _make_api()
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "limit",
                "side": "BUY",
                "price": 50_000.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": False,
                "order_id": "local-buy-2",
                "trigger_price": 0.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=51_000.0, low=49_000.0)
        assert len(fills) == 1

    def test_mixed_case_sell_limit(self) -> None:
        """Mixed-case 'Sell' should also match."""
        api = _make_api()
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "limit",
                "side": "Sell",
                "price": 51_000.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": False,
                "order_id": "local-sell-3",
                "trigger_price": 0.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=52_000.0, low=50_000.0)
        assert len(fills) == 1


# -------------------------------------------------------- stop_market orders


class TestStopMarketCaseNormalization:
    """Stop-market (stop-loss) orders with various side casings."""

    def test_lowercase_sell_stop_market_triggers(self) -> None:
        """Lowercase 'sell' stop-market: LONG SL triggers when low <= trigger."""
        api = _make_api()
        # Seed a LONG position (stop-loss needs a position to reduce)
        api._hedge_positions[("BTCUSDT", "LONG")] = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "qty": 1.0,
            "entry_price": 50_000.0,
        }
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "stop_market",
                "side": "sell",  # lowercase
                "price": 0.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": True,
                "order_id": "local-sl-1",
                "trigger_price": 48_000.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=50_000.0, low=47_500.0)
        assert len(fills) == 1
        assert fills[0]["average_price"] == 48_000.0

    def test_lowercase_buy_stop_market_triggers(self) -> None:
        """Lowercase 'buy' stop-market: SHORT SL triggers when high >= trigger."""
        api = _make_api()
        # Seed a SHORT position
        api._hedge_positions[("BTCUSDT", "SHORT")] = {
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "qty": 1.0,
            "entry_price": 50_000.0,
        }
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "stop_market",
                "side": "buy",  # lowercase
                "price": 0.0,
                "quantity": 0.1,
                "position_side": "SHORT",
                "reduce_only": True,
                "order_id": "local-sl-2",
                "trigger_price": 52_000.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=53_000.0, low=50_000.0)
        assert len(fills) == 1
        assert fills[0]["average_price"] == 52_000.0


# --------------------------------------------- take_profit_market orders


class TestTakeProfitMarketCaseNormalization:
    """Take-profit-market orders with various side casings."""

    def test_lowercase_sell_tp_market_triggers(self) -> None:
        """Lowercase 'sell' TP: LONG TP triggers when high >= trigger."""
        api = _make_api()
        # Seed a LONG position (TP needs a position to reduce)
        api._hedge_positions[("BTCUSDT", "LONG")] = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "qty": 1.0,
            "entry_price": 50_000.0,
        }
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "take_profit_market",
                "side": "sell",  # lowercase
                "price": 0.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": True,
                "order_id": "local-tp-1",
                "trigger_price": 55_000.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=56_000.0, low=54_000.0)
        assert len(fills) == 1
        assert fills[0]["average_price"] == 55_000.0

    def test_lowercase_buy_tp_market_triggers(self) -> None:
        """Lowercase 'buy' TP: SHORT TP triggers when low <= trigger."""
        api = _make_api()
        # Seed a SHORT position
        api._hedge_positions[("BTCUSDT", "SHORT")] = {
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "qty": 1.0,
            "entry_price": 50_000.0,
        }
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "take_profit_market",
                "side": "buy",  # lowercase
                "price": 0.0,
                "quantity": 0.1,
                "position_side": "SHORT",
                "reduce_only": True,
                "order_id": "local-tp-2",
                "trigger_price": 45_000.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=46_000.0, low=44_000.0)
        assert len(fills) == 1
        assert fills[0]["average_price"] == 45_000.0


# --------------------------------------------------------- no-match guard


class TestNoMatchWhenPriceOutOfRange:
    """Ensure orders are NOT matched when price doesn't reach."""

    def test_buy_limit_above_low_no_match(self) -> None:
        api = _make_api()
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "limit",
                "side": "buy",
                "price": 48_000.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": False,
                "order_id": "local-nm-1",
                "trigger_price": 0.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=51_000.0, low=49_000.0)
        assert len(fills) == 0

    def test_sell_limit_below_high_no_match(self) -> None:
        api = _make_api()
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "limit",
                "side": "sell",
                "price": 52_000.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": False,
                "order_id": "local-nm-2",
                "trigger_price": 0.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=51_000.0, low=49_000.0)
        assert len(fills) == 0


class TestMarginGate:
    """Broke account (equity <= 0) cannot open new risk; reduce-only is unaffected."""

    def test_market_open_rejected_when_broke(self) -> None:
        api = MockBinanceAPI(initial_wallet_usdt=0.0)
        api.set_price("BTCUSDT", 50_000.0)
        res = api.place_order(
            "BTCUSDT", "BUY", "market", 0.1, price=50_000.0, position_side="LONG"
        )
        assert res["status"] == "rejected"
        assert res["reason"] == "insufficient_margin"
        assert not api._hedge_positions

    def test_pending_entry_dropped_when_broke(self) -> None:
        api = MockBinanceAPI(initial_wallet_usdt=0.0)
        api.set_price("BTCUSDT", 50_000.0)
        api._pending_orders.append(
            {
                "symbol": "BTCUSDT",
                "type": "limit",
                "side": "buy",
                "price": 50_000.0,
                "quantity": 0.1,
                "position_side": "LONG",
                "reduce_only": False,
                "order_id": "local-broke-1",
                "trigger_price": 0.0,
            }
        )
        fills = api.match_pending_orders("BTCUSDT", high=51_000.0, low=49_000.0)
        assert fills == []
        assert not api._hedge_positions

    def test_reduce_only_allowed_when_broke(self) -> None:
        api = MockBinanceAPI(initial_wallet_usdt=1_000.0)
        api.set_price("BTCUSDT", 50_000.0)
        # Open a position while solvent, then drain the wallet.
        assert (
            api.place_order(
                "BTCUSDT", "BUY", "market", 0.1, price=50_000.0, position_side="LONG"
            )["status"]
            == "filled"
        )
        api.wallet_usdt = -100.0  # simulate underwater account
        res = api.place_order(
            "BTCUSDT",
            "SELL",
            "market",
            0.1,
            price=49_000.0,
            position_side="LONG",
            reduce_only=True,
        )
        assert res["status"] == "filled"  # reduce-only must still execute
        assert not api._hedge_positions  # position closed

    def test_healthy_account_opens_normally(self) -> None:
        api = MockBinanceAPI(initial_wallet_usdt=10_000.0)
        api.set_price("BTCUSDT", 50_000.0)
        res = api.place_order(
            "BTCUSDT", "BUY", "market", 0.1, price=50_000.0, position_side="LONG"
        )
        assert res["status"] == "filled"
        assert api._hedge_positions
