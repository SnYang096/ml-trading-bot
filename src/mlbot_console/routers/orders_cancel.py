"""Cancel endpoint for the orders router — appended to orders.py."""

import hashlib
import hmac
import time

import requests

from fastapi import Query

from mlbot_console.responses import ok
from mlbot_console.services.exchange_balances import _env_first, _SCOPE_META
from .orders import router, logger


@router.delete("/api/orders/cancel")
def orders_cancel(
    scope: str = Query(..., description="trend | multi_leg | spot_accum"),
    symbol: str = Query(..., description="e.g. BNBUSDT"),
    order_id: int = Query(..., description="Binance order ID"),
) -> dict:
    """Cancel a single open order via Binance API using the scope's credentials."""
    if scope not in _SCOPE_META:
        return ok(None, meta={"error": f"unknown scope: {scope}"})

    meta = _SCOPE_META[scope]
    api_key = _env_first(*meta["key_envs"])
    api_secret = _env_first(*meta["secret_envs"])
    if not api_key or not api_secret:
        return ok(None, meta={"error": "API keys not configured"})

    try:
        session = requests.Session()
        base = "https://fapi.binance.com"
        srv = session.get(f"{base}/fapi/v1/time", timeout=8)
        srv.raise_for_status()
        server_ts = int(srv.json().get("serverTime", int(time.time() * 1000)))

        query = f"symbol={symbol.upper()}&orderId={order_id}&timestamp={server_ts}"
        sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        resp = session.delete(
            f"{base}/fapi/v1/order?{query}&signature={sig}",
            headers={"X-MBX-APIKEY": api_key},
            timeout=12,
        )
        data = resp.json()
        if resp.status_code == 200:
            logger.info("Cancelled order %s %d status=%s", symbol, order_id, data.get("status"))
            return ok(
                {
                    "order_id": order_id,
                    "symbol": symbol.upper(),
                    "status": data.get("status"),
                    "executed_qty": data.get("executedQty"),
                    "orig_qty": data.get("origQty"),
                },
                meta={"scope": scope},
            )
        else:
            logger.warning("Cancel order failed %s %d: %s", symbol, order_id, data.get("msg"))
            return ok(None, meta={"error": data.get("msg", "cancel failed"), "code": resp.status_code})
    except Exception as e:
        logger.error("Cancel order exception %s %d: %s", symbol, order_id, e)
        return ok(None, meta={"error": str(e)})
