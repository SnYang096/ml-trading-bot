"""Tests for mark_prices fallback logic."""

from unittest.mock import MagicMock, patch
from mlbot_console.services.mark_prices import fetch_mark_prices


def test_fetch_mark_prices_stablecoins(tmp_path):
    marks = fetch_mark_prices(tmp_path, ["USDT", "USDC", "BUSD", "BTCUSDT"])
    assert marks.get("USDT") == 1.0
    assert marks.get("USDC") == 1.0
    assert marks.get("BUSD") == 1.0


@patch("mlbot_console.services.mark_prices.latest_close_prices")
@patch("mlbot_console.services.spot_ccxt.spot_binance_exchange")
def test_fetch_mark_prices_fallback(mock_spot_exchange, mock_latest_close, tmp_path):
    # Mock feature bus returning only BTC
    mock_latest_close.return_value = {"BTCUSDT": 65000.0}

    # Mock ccxt exchange ticker
    mock_exchange = MagicMock()
    mock_spot_exchange.return_value = mock_exchange
    mock_exchange.fetch_tickers.return_value = {
        "ETH/USDT": {"last": 3000.0},
        "SOL/USDT": {"close": 150.0},
    }

    marks = fetch_mark_prices(tmp_path, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])

    assert marks.get("BTCUSDT") == 65000.0
    assert marks.get("ETHUSDT") == 3000.0
    assert marks.get("SOLUSDT") == 150.0
    assert "DOGEUSDT" not in marks  # Not in feature bus nor ticker
