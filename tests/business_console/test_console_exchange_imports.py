"""Ensure account exchange stack imports (Docker image deps)."""

import importlib


def test_exchange_balance_runtime_imports() -> None:
    importlib.import_module("requests")
    importlib.import_module("dotenv")
    importlib.import_module("ccxt")
    importlib.import_module("order_management.spot_binance_api")
    eb = importlib.import_module("mlbot_console.services.exchange_balances")
    assert hasattr(eb, "fetch_scope_exchange_balance")
    assert hasattr(eb, "_fetch_spot_equity")
