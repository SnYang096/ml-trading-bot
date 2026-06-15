"""Exchange balance fetch for account overview."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mlbot_console.services.exchange_balances import (
    build_exchange_ledger,
    fetch_scope_exchange_balance,
    futures_open_positions,
    futures_symbol_unrealized_pnl,
    parse_futures_account,
    spot_symbol_holdings_value,
)


def test_futures_open_positions_filters_flat_legs() -> None:
    raw = {
        "positions": [
            {"symbol": "XRPUSDT", "positionAmt": "-100", "unRealizedProfit": "-13.95"},
            {"symbol": "ETHUSDT", "positionAmt": "0", "unRealizedProfit": "0"},
        ]
    }
    legs = futures_open_positions(raw)
    assert len(legs) == 1
    assert legs[0]["symbol"] == "XRPUSDT"
    assert legs[0]["side"] == "short"
    assert legs[0]["unrealized_pnl_usdt"] == pytest.approx(-13.95)


def test_futures_symbol_unrealized_pnl_sums_hedge_legs() -> None:
    raw = {
        "positions": [
            {
                "symbol": "XRPUSDT",
                "positionAmt": "100",
                "unRealizedProfit": "1.5",
            },
            {
                "symbol": "XRPUSDT",
                "positionAmt": "-50",
                "unRealizedProfit": "-0.3",
            },
            {"symbol": "BTCUSDT", "positionAmt": "0.01", "unRealizedProfit": "9.0"},
            {"symbol": "XRPUSDT", "positionAmt": "0", "unRealizedProfit": "0"},
        ]
    }
    assert futures_symbol_unrealized_pnl(raw, "XRPUSDT") == pytest.approx(1.2)


def test_futures_symbol_unrealized_pnl_computes_from_entry_mark_when_unrealized_zero() -> (
    None
):
    """When Binance returns unRealizedProfit=0 but entryPrice/markPrice are valid,
    compute PnL manually: short (entry-mark)*qty, long (mark-entry)*qty."""
    raw = {
        "positions": [
            {
                "symbol": "XRPUSDT",
                "positionAmt": "-161.2",
                "entryPrice": "1.104",
                "markPrice": "1.080",
                "unRealizedProfit": "0",  # Binance bug: per-leg PnL missing
            },
            {
                "symbol": "BNBUSDT",
                "positionAmt": "0.31",
                "entryPrice": "630.68",
                "markPrice": "600.0",
                "unRealizedProfit": "0",
            },
        ]
    }
    xrp_upnl = futures_symbol_unrealized_pnl(raw, "XRPUSDT")
    # short: (1.104 - 1.080) * 161.2 = 3.8688
    assert xrp_upnl == pytest.approx(3.8688)
    bnb_upnl = futures_symbol_unrealized_pnl(raw, "BNBUSDT")
    # long: (600 - 630.68) * 0.31 = -9.5108
    assert bnb_upnl == pytest.approx(-9.5108)


def test_futures_open_positions_computes_upnl_manually() -> None:
    raw = {
        "positions": [
            {
                "symbol": "XRPUSDT",
                "positionAmt": "-161.2",
                "entryPrice": "1.104",
                "markPrice": "1.080",
                "unRealizedProfit": "0",
            },
        ]
    }
    legs = futures_open_positions(raw)
    assert len(legs) == 1
    assert legs[0]["unrealized_pnl_usdt"] == pytest.approx(3.8688)
    assert legs[0]["mark_price"] == pytest.approx(1.080)
    assert legs[0]["entry_price"] == pytest.approx(1.104)


def test_spot_symbol_holdings_value_filters_asset() -> None:
    holdings = [
        {"asset": "XRP", "qty": 100.0, "value_usdt": 120.0},
        {"asset": "BTC", "qty": 0.1, "value_usdt": 6000.0},
    ]
    assert spot_symbol_holdings_value(holdings, "XRPUSDT") == pytest.approx(120.0)


def test_parse_futures_account() -> None:
    parsed = parse_futures_account(
        {
            "totalWalletBalance": "1000.5",
            "totalMarginBalance": "1008.2",
            "availableBalance": "900.1",
            "totalUnrealizedProfit": "7.7",
            "totalMaintMargin": "120.4",
            "totalPositionInitialMargin": "80.0",
            "totalOpenOrderInitialMargin": "28.1",
        }
    )
    assert parsed["wallet_balance_usdt"] == pytest.approx(1000.5)
    assert parsed["equity_usdt"] == pytest.approx(1008.2)
    assert parsed["available_usdt"] == pytest.approx(900.1)
    assert parsed["unrealized_pnl_usdt"] == pytest.approx(7.7)
    assert parsed["maint_margin_usdt"] == pytest.approx(120.4)
    assert parsed["margin_ratio"] == pytest.approx(0.119421, abs=1e-6)
    assert parsed["position_initial_margin_usdt"] == pytest.approx(80.0)
    assert parsed["open_order_initial_margin_usdt"] == pytest.approx(28.1)
    assert parsed["margin_locked_usdt"] == pytest.approx(108.1)


def test_parse_futures_account_gross_leverage_from_positions() -> None:
    legs = [
        {
            "symbol": "BTCUSDT",
            "notional_usdt": 30000.0,
            "leverage": 10,
            "initial_margin_usdt": 3000.0,
        }
    ]
    parsed = parse_futures_account(
        {
            "totalWalletBalance": "10000",
            "totalMarginBalance": "10000",
            "availableBalance": "7000",
            "totalUnrealizedProfit": "0",
            "totalMaintMargin": "100",
            "totalPositionInitialMargin": "3000",
            "totalOpenOrderInitialMargin": "0",
        },
        open_positions=legs,
    )
    assert parsed["gross_notional_usdt"] == pytest.approx(30000.0)
    assert parsed["gross_leverage"] == pytest.approx(3.0)


def test_futures_open_positions_includes_leverage_and_margin() -> None:
    raw = {
        "positions": [
            {
                "symbol": "ETHUSDT",
                "positionAmt": "1.5",
                "entryPrice": "3000",
                "markPrice": "3100",
                "leverage": "5",
                "positionInitialMargin": "930",
                "maintMargin": "120",
                "marginType": "cross",
                "unRealizedProfit": "150",
            },
        ]
    }
    legs = futures_open_positions(raw)
    assert len(legs) == 1
    assert legs[0]["leverage"] == 5
    assert legs[0]["notional_usdt"] == pytest.approx(4650.0)
    assert legs[0]["initial_margin_usdt"] == pytest.approx(930.0)
    assert legs[0]["margin_type"] == "cross"


def test_parse_futures_account_zero_equity_no_ratio() -> None:
    parsed = parse_futures_account(
        {
            "totalWalletBalance": "0",
            "totalMarginBalance": "0",
            "availableBalance": "0",
            "totalUnrealizedProfit": "0",
            "totalMaintMargin": "0",
        }
    )
    assert parsed["margin_ratio"] is None


def test_fetch_scope_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    row = fetch_scope_exchange_balance("trend")
    assert row["configured"] is False
    assert row["ok"] is False
    assert row["equity_usdt"] is None


def test_fetch_scope_futures_symbol_filter_keeps_account_unrealized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "totalWalletBalance": "1000",
        "totalMarginBalance": "1008",
        "availableBalance": "900",
        "totalUnrealizedProfit": "8",
        "positions": [
            {"symbol": "BTCUSDT", "positionAmt": "0.01", "unRealizedProfit": "8"},
            {"symbol": "ETHUSDT", "positionAmt": "0", "unRealizedProfit": "0"},
        ],
    }

    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    with patch(
        "mlbot_console.services.exchange_balances._fetch_futures_account_raw",
        return_value=raw,
    ):
        row = fetch_scope_exchange_balance("trend", symbol="ETHUSDT")

    assert row["ok"] is True
    assert row["account_unrealized_pnl_usdt"] == pytest.approx(8.0)
    assert row["symbol_unrealized_pnl_usdt"] == pytest.approx(0.0)
    assert row["unrealized_pnl_usdt"] == pytest.approx(0.0)
    assert row["unrealized_pnl_basis"] == "symbol"


def test_build_exchange_ledger_sums_ok_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake(scope: str, *, mark_prices=None, symbol=None):
        base = {
            "scope": scope,
            "label": scope,
            "configured": True,
            "ok": True,
            "error": None,
            "fetched_at": "2026-01-01T00:00:00+00:00",
        }
        if scope == "trend":
            return {
                **base,
                "wallet_balance_usdt": 1000.0,
                "equity_usdt": 1010.0,
                "available_usdt": 800.0,
                "unrealized_pnl_usdt": 10.0,
            }
        if scope == "spot":
            return {
                **base,
                "wallet_balance_usdt": 500.0,
                "equity_usdt": 520.0,
                "available_usdt": 500.0,
                "unrealized_pnl_usdt": 0.0,
            }
        return {**base, "configured": False, "ok": False, "error": "no keys"}

    with patch(
        "mlbot_console.services.exchange_balances.fetch_scope_exchange_balance",
        side_effect=_fake,
    ):
        ledger = build_exchange_ledger(mark_prices={})
    totals = ledger["totals"]
    assert totals["equity_usdt"] == pytest.approx(1530.0)
    assert totals["wallet_balance_usdt"] == pytest.approx(1500.0)
    assert totals["accounts_ok"] == 2
    assert totals["accounts_total"] == 3


def test_account_summary_includes_exchange_ledger(
    trend_db, spot_db, multi_leg_db, bus_root, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ledger = {
        "accounts": [
            {
                "scope": "trend",
                "ok": True,
                "wallet_balance_usdt": 2000.0,
                "equity_usdt": 2010.0,
                "available_usdt": 1800.0,
                "unrealized_pnl_usdt": 10.0,
            },
            {
                "scope": "spot",
                "ok": True,
                "wallet_balance_usdt": 300.0,
                "equity_usdt": 310.0,
                "available_usdt": 300.0,
                "unrealized_pnl_usdt": 0.0,
            },
            {"scope": "multi_leg", "ok": False, "configured": False},
        ],
        "totals": {
            "equity_usdt": 2320.0,
            "wallet_balance_usdt": 2300.0,
            "available_usdt": 2100.0,
            "exchange_unrealized_pnl_usdt": 10.0,
            "accounts_ok": 2,
            "accounts_total": 3,
        },
    }
    with patch(
        "mlbot_console.services.account_summary.build_exchange_ledger",
        return_value=fake_ledger,
    ), patch(
        "mlbot_console.services.spot_ledger_book.fetch_spot_ledger_holdings",
        return_value={"holdings": [], "holdings_value_usdt": 0.0},
    ):
        from mlbot_console.services.account_summary import build_account_summary
        from pathlib import Path

        data = build_account_summary(
            trend_db=trend_db,
            spot_db=spot_db,
            spot_ledger_db=Path("/dev/null"),
            multi_leg_db=multi_leg_db,
            feature_bus_root=bus_root,
            lookback_days=0,
        )
    assert data["totals"]["equity_usdt"] == pytest.approx(2320.0)
    trend_scope = next(s for s in data["scopes"] if s["scope"] == "trend")
    assert trend_scope["exchange"]["equity_usdt"] == pytest.approx(2010.0)
