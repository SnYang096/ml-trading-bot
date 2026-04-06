from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.order_management.binance_user_stream import BinanceUserStream


def _make_stream() -> BinanceUserStream:
    stream = BinanceUserStream.__new__(BinanceUserStream)
    stream.on_execution_report = MagicMock()
    stream.on_account_update = MagicMock()
    return stream


def test_handle_message_dispatches_account_update():
    stream = _make_stream()
    msg = json.dumps(
        {
            "e": "ACCOUNT_UPDATE",
            "E": 1710000000000,
            "T": 1710000000500,
            "a": {
                "m": "ORDER",
                "B": [{"a": "USDT", "wb": "1234.5", "cw": "1200.0", "bc": "0.0"}],
                "P": [
                    {
                        "s": "BTCUSDT",
                        "pa": "0.01",
                        "ep": "50000",
                        "up": "4.2",
                        "ps": "BOTH",
                    }
                ],
            },
        }
    )

    stream._handle_message(msg)

    stream.on_account_update.assert_called_once()
    payload = stream.on_account_update.call_args[0][0]
    assert payload["wallet_balance"] == 1234.5
    assert payload["available_balance"] == 1200.0
    assert payload["positions"][0]["symbol"] == "BTCUSDT"
    assert payload["positions"][0]["position_amount"] == 0.01


def test_normalize_account_update_sums_unrealized_pnl():
    stream = _make_stream()
    data = {
        "e": "ACCOUNT_UPDATE",
        "E": 1710000000000,
        "a": {
            "B": [{"a": "USDT", "wb": "1000.0", "cw": "900.0"}],
            "P": [
                {"s": "BTCUSDT", "pa": "0.01", "up": "3.5"},
                {"s": "ETHUSDT", "pa": "-0.2", "up": "-1.2"},
            ],
        },
    }

    out = stream._normalize_account_update(data)

    assert out is not None
    assert out["wallet_balance"] == 1000.0
    assert out["available_balance"] == 900.0
    assert out["unrealized_pnl_total"] == 2.3
