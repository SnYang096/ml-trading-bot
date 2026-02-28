"""
Mock BinanceAPI for backtesting.

Simulates instant order fills without any network calls.
Used by event_backtest.py to integrate with order_management storage layer.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MockBinanceAPI:
    """
    BinanceAPI mock for backtesting — no network calls, instant fills.

    Compatible with the real BinanceAPI interface used by:
      - OrderManager.place_order()
      - OrderManager.cancel_order()
      - PositionManager._update_position_pnl()
      - PositionManager.update_stop_loss()
    """

    def __init__(self):
        """Initialize mock with empty state."""
        self._positions: Dict[str, Dict[str, Any]] = {}  # symbol → position info
        self._open_orders: Dict[str, Dict[str, Any]] = {}  # order_id → order
        self._last_prices: Dict[str, float] = {}  # symbol → last known price

    # ─── Price feed (called by backtest to update current prices) ───

    def set_price(self, symbol: str, price: float) -> None:
        """Update the mock price for a symbol (called each bar by backtest)."""
        self._last_prices[symbol] = price

    def set_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
    ) -> None:
        """Update mock position state (called when PositionSimulator changes)."""
        if size == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "size": size,
                "contracts": size,
                "entry_price": entry_price,
                "mark_price": self._last_prices.get(symbol, entry_price),
            }

    # ─── BinanceAPI interface (used by OrderManager / PositionManager) ───

    def place_order(
        self,
        symbol: str,
        side,  # OrderSide enum
        order_type,  # OrderType enum
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Simulate placing an order — instant fill at current price."""
        fill_price = price or self._last_prices.get(symbol, 0.0)
        order_id = f"mock_{uuid.uuid4().hex[:12]}"
        cid = client_order_id or f"mcid_{uuid.uuid4().hex[:10]}"

        side_val = side.value if hasattr(side, "value") else str(side)
        type_val = order_type.value if hasattr(order_type, "value") else str(order_type)

        result = {
            "order_id": order_id,
            "id": order_id,
            "client_order_id": cid,
            "symbol": symbol,
            "side": side_val,
            "type": type_val,
            "quantity": quantity,
            "price": fill_price,
            "average_price": fill_price,
            "filled": quantity,
            "filled_quantity": quantity,
            "status": "filled",
            "created_at": datetime.now().timestamp(),
        }

        logger.debug(
            "MockBinanceAPI.place_order: %s %s %s qty=%.6f @ %.4f",
            symbol, side_val, type_val, quantity, fill_price,
        )
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Simulate canceling an order."""
        self._open_orders.pop(order_id, None)
        return True

    def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get order status — always returns filled for mock."""
        return {
            "order_id": order_id,
            "status": "filled",
            "filled": 0,
            "average_price": 0,
            "created_at": datetime.now().timestamp(),
        }

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """No open orders in mock (everything fills instantly)."""
        return []

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get mock position for a symbol."""
        pos = self._positions.get(symbol)
        if pos:
            # Update mark price to latest
            pos["mark_price"] = self._last_prices.get(symbol, pos.get("entry_price", 0))
            return pos
        return {
            "symbol": symbol,
            "size": 0,
            "contracts": 0,
            "mark_price": self._last_prices.get(symbol, 0),
        }

    def get_balance(self) -> Dict[str, Any]:
        """Mock balance."""
        return {"total": 10000.0, "available": 10000.0}
