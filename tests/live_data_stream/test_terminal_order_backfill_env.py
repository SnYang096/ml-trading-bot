"""Env + guard logic for trend terminal-order REST backfill."""

import pytest

from src.live_data_stream.terminal_order_backfill import (
    terminal_order_backfill_env_int,
    terminal_order_backfill_enabled_interval_seconds,
    terminal_order_backfill_should_run,
)


def test_interval_unset_defaults_60(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS", raising=False)
    assert terminal_order_backfill_enabled_interval_seconds() == 60.0


@pytest.mark.parametrize(
    "raw",
    ["0", "false", "FALSE", "no", "off", "disable", "disabled"],
)
def test_interval_disables(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS", raw)
    assert terminal_order_backfill_enabled_interval_seconds() == 0.0


def test_interval_custom_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS", "120")
    assert terminal_order_backfill_enabled_interval_seconds() == 120.0


def test_env_int_invalid_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLBOT_TERMINAL_ORDER_BACKFILL_LIMIT", "bad")
    assert (
        terminal_order_backfill_env_int("MLBOT_TERMINAL_ORDER_BACKFILL_LIMIT", 200)
        == 200
    )


def test_should_run_none() -> None:
    assert terminal_order_backfill_should_run(None) is False


def test_should_run_shadow() -> None:
    om = type(
        "OM",
        (),
        {
            "shadow": True,
            "binance_api": object(),
            "reconcile_recent_terminal_orders": lambda **kw: [],
        },
    )()
    assert terminal_order_backfill_should_run(om) is False


def test_should_run_no_api() -> None:
    om = type(
        "OM",
        (),
        {
            "shadow": False,
            "binance_api": None,
            "reconcile_recent_terminal_orders": lambda **kw: [],
        },
    )()
    assert terminal_order_backfill_should_run(om) is False


def test_should_run_ok() -> None:
    om = type(
        "OM",
        (),
        {
            "shadow": False,
            "binance_api": object(),
            "reconcile_recent_terminal_orders": lambda **kw: [],
        },
    )()
    assert terminal_order_backfill_should_run(om) is True
