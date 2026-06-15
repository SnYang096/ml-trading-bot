from __future__ import annotations

from src.monitoring.account_telegram_watch import (
    detect_new_positions,
    position_keys,
)
from src.monitoring.telegram import format_account_equity_change_message


def test_position_keys_from_exchange_rows() -> None:
    keys = position_keys(
        [
            {"symbol": "BTCUSDT", "positionAmt": "0.01"},
            {"symbol": "ETHUSDT", "positionAmt": "-1.5"},
        ]
    )
    assert keys == {"BTCUSDT:long", "ETHUSDT:short"}


def test_detect_new_positions() -> None:
    assert detect_new_positions(
        {"BTCUSDT:long"}, {"BTCUSDT:long", "ETHUSDT:short"}
    ) == ["ETHUSDT:short"]


def test_equity_change_message_threshold() -> None:
    assert (
        format_account_equity_change_message(
            scope="multi_leg",
            anchor=10000.0,
            current=10250.0,
            threshold_pct=0.03,
        )
        is None
    )
    msg = format_account_equity_change_message(
        scope="multi_leg",
        anchor=10000.0,
        current=9650.0,
        threshold_pct=0.03,
    )
    assert msg is not None
    assert "-3.50%" in msg
