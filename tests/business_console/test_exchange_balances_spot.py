"""Tests for spot equity calculation."""

from unittest.mock import MagicMock, patch
from mlbot_console.services.exchange_balances import _fetch_spot_equity


@patch("order_management.spot_binance_api.SpotBinanceAPI")
def test_fetch_spot_equity(mock_spot_api_class):
    mock_api = MagicMock()
    mock_spot_api_class.return_value = mock_api

    # Mock balance response
    mock_api.exchange.fetch_balance.return_value = {
        "USDT": {"free": 100.0, "total": 150.0},
        "total": {
            "USDT": 150.0,
            "BTC": 0.5,
            "ETH": 10.0,
            "SHIB": 1000.0,
            "DUST": 0.0,
        },
    }

    # Mock ticker fallback for SHIB
    mock_api.exchange.fetch_tickers.return_value = {"SHIB/USDT": {"last": 0.00001}}

    mark_prices = {
        "BTCUSDT": 60000.0,
        "ETHUSDT": 3000.0,
        # SHIB missing from mark_prices
    }

    res = _fetch_spot_equity(
        api_key="fake",
        api_secret="fake",
        mark_prices=mark_prices,
    )

    assert res["usdt_cash"] == 150.0
    assert res["available_usdt"] == 100.0

    # 150 (USDT) + 0.5 * 60000 (BTC) + 10 * 3000 (ETH) + 1000 * 0.00001 (SHIB)
    # 150 + 30000 + 30000 + 0.01 = 60150.01
    assert res["equity_usdt"] == 60150.01
    assert res["wallet_balance_usdt"] == 60150.01
    assert res["holdings_value_usdt"] == 60000.01

    assert len(res["holdings"]) == 3

    btc = next(h for h in res["holdings"] if h["asset"] == "BTC")
    assert btc["qty"] == 0.5
    assert btc["price_usdt"] == 60000.0
    assert btc["price_source"] == "bars_1min"

    shib = next(h for h in res["holdings"] if h["asset"] == "SHIB")
    assert shib["qty"] == 1000.0
    assert shib["price_usdt"] == 0.00001
    assert shib["price_source"] == "ticker"
