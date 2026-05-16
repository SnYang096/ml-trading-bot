from __future__ import annotations

import os
from typing import Any, Optional


def init_order_manager_from_env() -> Optional[Any]:
    if os.getenv("MLBOT_ORDER_MANAGER_ENABLED", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return None
    testnet = os.getenv("MLBOT_ORDER_MANAGER_TESTNET", "").lower() in {
        "1",
        "true",
        "yes",
    }
    shadow = os.getenv("MLBOT_ORDER_SHADOW_MODE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    db_path = os.getenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", "data/order_management.db")
    if testnet:
        api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "")
        api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")
    else:
        api_key = os.getenv("BINANCE_API_KEY") or os.getenv(
            "BINANCE_FUTURES_API_KEY", ""
        )
        api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
            "BINANCE_FUTURES_API_SECRET", ""
        )
    if not api_key or not api_secret:
        if shadow:
            # Shadow 模式不需要真实 API key, 创建不连接交易所的实例
            try:
                from src.order_management.storage import Storage
                from src.order_management.order_manager import OrderManager

                storage = Storage(str(db_path))
                return OrderManager(storage, binance_api=None, shadow=True)
            except Exception:
                return None
        return None
    try:
        from src.order_management.storage import Storage
        from src.order_management.binance_api import BinanceAPI
        from src.order_management.order_manager import OrderManager

        storage = Storage(str(db_path))
        binance_api = BinanceAPI(
            api_key=str(api_key),
            api_secret=str(api_secret),
            testnet=bool(testnet),
            use_proxy=None,
        )
        return OrderManager(storage, binance_api, shadow=shadow)
    except Exception:
        return None
