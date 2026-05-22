from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import ccxt

logger = logging.getLogger(__name__)


class SpotBinanceAPI:
    """Small spot-only wrapper used by spot_accum_simple live runner."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ) -> None:
        options: Dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "spot"},
        }
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        if proxy:
            options["proxies"] = {"http": proxy, "https": proxy}

        if testnet:
            # Spot testnet host (ccxt still requires urls override for binance)
            options["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }

        self.exchange = ccxt.binance(
            {"apiKey": api_key, "secret": api_secret, **options}
        )
        self.exchange.options["defaultType"] = "spot"
        self.exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
        self.exchange.load_markets()

    def get_last_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol.upper())
        px = ticker.get("last") or ticker.get("close")
        return float(px or 0.0)

    def get_total_balances(self) -> Dict[str, float]:
        bal = self.exchange.fetch_balance()
        raw = bal.get("total") if isinstance(bal, dict) else {}
        out: Dict[str, float] = {}
        if isinstance(raw, dict):
            for asset, qty in raw.items():
                try:
                    fv = float(qty or 0.0)
                except (TypeError, ValueError):
                    continue
                if fv > 0.0:
                    out[str(asset).upper()] = fv
        return out

    def get_free_balances(self) -> Dict[str, float]:
        """Free (available) balances — use for buy affordability checks."""
        bal = self.exchange.fetch_balance()
        raw = bal.get("free") if isinstance(bal, dict) else {}
        out: Dict[str, float] = {}
        if isinstance(raw, dict):
            for asset, qty in raw.items():
                try:
                    fv = float(qty or 0.0)
                except (TypeError, ValueError):
                    continue
                if fv > 0.0:
                    out[str(asset).upper()] = fv
        return out

    def get_market_limits(self, symbol: str) -> Dict[str, float]:
        market = self.exchange.market(symbol.upper())
        limits = market.get("limits") if isinstance(market, dict) else {}
        out: Dict[str, float] = {}
        if isinstance(limits, dict):
            amount = limits.get("amount")
            cost = limits.get("cost")
            if isinstance(amount, dict) and amount.get("min") is not None:
                out["min_amount"] = float(amount["min"])
            if isinstance(cost, dict) and cost.get("min") is not None:
                out["min_cost"] = float(cost["min"])
        return out

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sym = symbol.upper()
        typ = str(order_type or "market").lower()
        s = str(side or "").lower()
        params: Dict[str, Any] = {}
        if client_order_id:
            params["newClientOrderId"] = str(client_order_id)[:36]
        if typ == "market":
            return self.exchange.create_order(sym, "market", s, quantity, None, params)
        if typ != "limit":
            raise ValueError(f"unsupported spot order_type={order_type}")
        if price is None or float(price) <= 0.0:
            raise ValueError("limit order requires positive price")
        params["timeInForce"] = "GTC"
        return self.exchange.create_order(sym, "limit", s, quantity, float(price), params)

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return float(self.exchange.amount_to_precision(symbol.upper(), amount))

    def price_to_precision(self, symbol: str, price: float) -> float:
        return float(self.exchange.price_to_precision(symbol.upper(), price))

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        sym = symbol.upper() if symbol else None
        return list(self.exchange.fetch_open_orders(sym) or [])

    def fetch_order(self, symbol: str, exchange_order_id: str) -> Dict[str, Any]:
        sym = symbol.upper()
        oid = str(exchange_order_id)
        try:
            return dict(self.exchange.fetch_order(oid, sym) or {})
        except Exception:
            return dict(self.exchange.fetch_order(oid) or {})

    def cancel_order(self, symbol: str, exchange_order_id: str) -> Dict[str, Any]:
        sym = symbol.upper()
        oid = str(exchange_order_id)
        try:
            return dict(self.exchange.cancel_order(oid, sym) or {})
        except Exception:
            return dict(self.exchange.cancel_order(oid) or {})
