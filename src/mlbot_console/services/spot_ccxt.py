"""Minimal Binance spot ccxt client for read-only console (no order_management)."""

from __future__ import annotations

import os
from typing import Any

import ccxt


def spot_binance_exchange(
    *,
    api_key: str = "",
    api_secret: str = "",
) -> ccxt.binance:
    """Build a ccxt Binance spot exchange handle (console-only)."""
    options: dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"defaultType": "spot"},
    }
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if proxy:
        options["proxies"] = {"http": proxy, "https": proxy}
    params: dict[str, Any] = {**options}
    if api_key:
        params["apiKey"] = api_key
    if api_secret:
        params["secret"] = api_secret
    exchange = ccxt.binance(params)
    exchange.options["defaultType"] = "spot"
    exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
    return exchange
