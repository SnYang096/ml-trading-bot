"""
Mock BinanceAPI for backtesting.

Simulates instant order fills without any network calls.
Used by event_backtest.py to integrate with order_management storage layer.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HedgeKey = Tuple[str, str]


class MockBinanceAPI:
    """
    BinanceAPI mock for backtesting — no network calls, instant fills.

    Compatible with the real BinanceAPI interface used by:
      - OrderManager.place_order()
      - OrderManager.cancel_order()
      - PositionManager._update_position_pnl()
      - PositionManager.update_stop_loss()
    """

    def __init__(self, *, initial_wallet_usdt: float = 10000.0, fee_bps: float = 4.0):
        """Initialize mock with empty state."""
        self._hedge_positions: Dict[HedgeKey, Dict[str, Any]] = {}
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._open_orders: Dict[str, Dict[str, Any]] = {}
        self._last_prices: Dict[str, float] = {}
        self.wallet_usdt = float(initial_wallet_usdt or 0.0)
        self.default_fee_bps = float(fee_bps or 0.0)
        self.hedge_mode: bool = False
        self.hedge_mode_probe_error: Optional[str] = None

    def set_wallet(self, amount: float) -> None:
        self.wallet_usdt = float(amount or 0.0)

    def set_fee_bps(self, fee_bps: float) -> None:
        self.default_fee_bps = float(fee_bps or 0.0)

    @staticmethod
    def _pos_key(symbol: str, position_side: str) -> HedgeKey:
        return (str(symbol).upper(), str(position_side).upper())

    def _fee_usdt(self, notional: float, fee_bps: Optional[float] = None) -> float:
        bps = self.default_fee_bps if fee_bps is None else float(fee_bps)
        return abs(notional) * max(0.0, bps) / 10000.0

    def unrealized_pnl_usdt(self) -> float:
        total = 0.0
        for (sym, side), pos in self._hedge_positions.items():
            qty = float(pos.get("qty") or 0.0)
            if qty <= 0:
                continue
            entry = float(pos.get("entry_price") or 0.0)
            mark = float(self._last_prices.get(sym, entry) or entry)
            if side == "LONG":
                total += (mark - entry) * qty
            else:
                total += (entry - mark) * qty
        return total

    def set_price(self, symbol: str, price: float) -> None:
        """Update the mock price for a symbol (called each bar by backtest)."""
        self._last_prices[symbol] = float(price)

    def set_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
    ) -> None:
        """Update mock position state (legacy single-side API)."""
        pside = str(side or "").upper()
        if pside not in {"LONG", "SHORT"}:
            pside = "LONG" if float(size or 0) >= 0 else "SHORT"
        if float(size or 0) == 0:
            self._hedge_positions.pop(self._pos_key(symbol, pside), None)
            self._positions.pop(symbol, None)
        else:
            self._hedge_positions[self._pos_key(symbol, pside)] = {
                "symbol": symbol,
                "side": pside,
                "qty": abs(float(size)),
                "entry_price": float(entry_price),
            }
            self._positions[symbol] = {
                "symbol": symbol,
                "side": pside,
                "size": abs(float(size)),
                "contracts": abs(float(size)),
                "entry_price": float(entry_price),
                "mark_price": self._last_prices.get(symbol, entry_price),
            }

    def _apply_open(
        self,
        *,
        symbol: str,
        position_side: str,
        qty: float,
        fill_price: float,
        fee_bps: Optional[float] = None,
    ) -> None:
        key = self._pos_key(symbol, position_side)
        fee = self._fee_usdt(qty * fill_price, fee_bps)
        self.wallet_usdt -= fee
        pos = self._hedge_positions.get(key)
        if pos is None or float(pos.get("qty") or 0) <= 0:
            self._hedge_positions[key] = {
                "symbol": symbol,
                "side": position_side,
                "qty": qty,
                "entry_price": fill_price,
            }
            return
        old_qty = float(pos["qty"])
        new_qty = old_qty + qty
        pos["entry_price"] = (pos["entry_price"] * old_qty + fill_price * qty) / new_qty
        pos["qty"] = new_qty

    def _apply_reduce(
        self,
        *,
        symbol: str,
        position_side: str,
        qty: float,
        fill_price: float,
        fee_bps: Optional[float] = None,
    ) -> float:
        key = self._pos_key(symbol, position_side)
        pos = self._hedge_positions.get(key)
        if pos is None or float(pos.get("qty") or 0) <= 0:
            return 0.0
        close_qty = min(float(qty), float(pos["qty"]))
        entry = float(pos["entry_price"])
        if position_side == "LONG":
            gross = (fill_price - entry) * close_qty
        else:
            gross = (entry - fill_price) * close_qty
        fee = self._fee_usdt(close_qty * fill_price, fee_bps)
        self.wallet_usdt += gross - fee
        pos["qty"] = float(pos["qty"]) - close_qty
        if float(pos["qty"]) <= 1e-12:
            self._hedge_positions.pop(key, None)
        return gross - fee

    def place_order(
        self,
        symbol: str,
        side,
        order_type,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        client_order_id: Optional[str] = None,
        time_in_force: Optional[str] = None,
        position_side: Optional[str] = None,
        working_type: Optional[str] = None,
        price_protect: Optional[bool] = None,
        post_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Simulate placing an order — instant fill at current price."""
        fill_price = float(price or self._last_prices.get(symbol, 0.0) or 0.0)
        order_id = f"mock_{uuid.uuid4().hex[:12]}"
        cid = client_order_id or f"mcid_{uuid.uuid4().hex[:10]}"

        side_val = side.value if hasattr(side, "value") else str(side)
        type_val = order_type.value if hasattr(order_type, "value") else str(order_type)
        qty = float(quantity or 0.0)

        pside = str(position_side or "").upper()
        if not pside:
            if reduce_only or close_position:
                pside = "LONG" if side_val.upper() == "SELL" else "SHORT"
            else:
                pside = "LONG" if side_val.upper() == "BUY" else "SHORT"

        if qty > 0 and fill_price > 0:
            if reduce_only or close_position:
                self._apply_reduce(
                    symbol=symbol,
                    position_side=pside,
                    qty=qty,
                    fill_price=fill_price,
                )
            else:
                self._apply_open(
                    symbol=symbol,
                    position_side=pside,
                    qty=qty,
                    fill_price=fill_price,
                )

        return {
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
            "position_side": pside,
        }

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        self._open_orders.pop(order_id, None)
        return True

    def cancel_algo_order(self, order_id: str, symbol: str) -> bool:
        self._open_orders.pop(order_id, None)
        return True

    def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        return {
            "order_id": order_id,
            "status": "filled",
            "filled": 0,
            "average_price": 0,
            "created_at": datetime.now().timestamp(),
        }

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for (sym, pside), pos in self._hedge_positions.items():
            if symbol and str(sym).upper() != str(symbol).upper():
                continue
            qty = float(pos.get("qty") or 0.0)
            if qty <= 0:
                continue
            mark = float(self._last_prices.get(sym, pos.get("entry_price", 0.0)) or 0.0)
            entry = float(pos.get("entry_price") or 0.0)
            out.append(
                {
                    "symbol": sym,
                    "side": pside.lower(),
                    "position_side": pside,
                    "positionSide": pside,
                    "position_amount": qty,
                    "positionAmt": qty,
                    "contracts": qty,
                    "mark_price": mark,
                    "markPrice": mark,
                    "entry_price": entry,
                    "entryPrice": entry,
                }
            )
        return out

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        positions = self.get_positions(symbol)
        if not positions:
            return {
                "symbol": symbol,
                "size": 0,
                "contracts": 0,
                "mark_price": self._last_prices.get(symbol, 0),
            }
        pos = positions[0]
        return {
            "symbol": symbol,
            "size": pos.get("contracts", 0),
            "contracts": pos.get("contracts", 0),
            "mark_price": pos.get("mark_price", 0),
            "entry_price": pos.get("entry_price", 0),
        }

    def get_balance(self) -> Dict[str, Any]:
        return {
            "total": self.wallet_usdt,
            "available": self.wallet_usdt,
            "wallet_balance": self.wallet_usdt,
        }

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
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
        return []
