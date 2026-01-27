"""
Signal -> Order Management bridge.

This module provides a minimal, explicit adapter that converts archetype signals
into OrderManager + PositionManager actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .models import PositionSide, OrderSide, OrderType
from .order_manager import OrderManager
from .position_manager import PositionManager


@dataclass(frozen=True)
class ExecutionSignal:
    """Normalized execution signal from strategy layer."""

    symbol: str
    archetype: str
    side: PositionSide
    size: float
    confidence: float
    direction_source: str
    dir_prob: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    decision_id: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManagementBridge:
    """
    Converts ExecutionSignal into order + position actions.

    This is intentionally minimal: it does not implement risk sizing logic.
    The caller should provide a fully sized signal.
    """

    def __init__(self, order_manager: OrderManager, position_manager: PositionManager):
        self._order_manager = order_manager
        self._position_manager = position_manager

    def submit_signal(
        self,
        signal: ExecutionSignal,
        *,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Submit an execution signal.

        Returns:
            (order_id, position_id)
        """
        if signal.size <= 0:
            raise ValueError("Signal size must be positive")

        order_side = OrderSide.BUY if signal.side == PositionSide.LONG else OrderSide.SELL
        order = self._order_manager.place_order(
            symbol=signal.symbol,
            side=order_side,
            order_type=order_type,
            quantity=signal.size,
            price=price or signal.entry_price,
        )

        position_id: Optional[str] = None
        if signal.entry_price is not None:
            position = self._position_manager.create_position(
                symbol=signal.symbol,
                side=signal.side,
                entry_price=signal.entry_price,
                size=signal.size,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
                strategy_id=signal.archetype,
                notes=signal.reason,
            )
            position_id = position.position_id

        return order.order_id, position_id

