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
        # Multi-leg live (`GridExecutionAdapter.sync_positions`) expects this flag;
        # `scripts/run_multi_leg_live.py` sets hedge_mode=True for shadow runs.
        self.hedge_mode: bool = False
        self.hedge_mode_probe_error: Optional[str] = None

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
        time_in_force: Optional[str] = None,
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
            symbol,
            side_val,
            type_val,
            quantity,
            fill_price,
        )
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Simulate canceling an order."""
        self._open_orders.pop(order_id, None)
        return True

    def cancel_algo_order(self, order_id: str, symbol: str) -> bool:
        """Simulate canceling an algo (stop/tp) order."""
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

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return open positions in the same shape as ``BinanceAPI.get_positions``."""
        out: List[Dict[str, Any]] = []
        for sym, pos in self._positions.items():
            if symbol and str(sym).upper() != str(symbol).upper():
                continue
            try:
                qty = float(pos.get("contracts") or pos.get("size") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty == 0:
                continue
            raw_side = str(pos.get("side", "") or "").upper()
            if raw_side in {"LONG", "SHORT"}:
                side = raw_side
            else:
                side = "LONG" if qty >= 0 else "SHORT"
            mark = float(
                pos.get("mark_price") or self._last_prices.get(sym, 0.0) or 0.0
            )
            entry = float(pos.get("entry_price") or 0.0)
            out.append(
                {
                    "symbol": sym,
                    "side": side.lower(),
                    "position_side": side,
                    "positionSide": side,
                    "position_amount": abs(qty),
                    "positionAmt": abs(qty),
                    "contracts": abs(qty),
                    "mark_price": mark,
                    "markPrice": mark,
                    "entry_price": entry,
                    "entryPrice": entry,
                }
            )
        return out

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

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Mock symbol info with reasonable defaults."""
        return {
            "symbol": symbol,
            "price_precision": 2,
            "quantity_precision": 3,
            "min_notional": 5.0,
            "min_qty": 0.001,
            "step_size": 0.001,
            "tick_size": 0.01,
        }

    def get_open_orders_for_sl_cleanup(self, symbol: str) -> List[Dict[str, Any]]:
        """Mock: no open algo orders to clean up."""
        return []
