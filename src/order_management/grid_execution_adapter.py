"""Execution adapter for standalone chop-grid order actions.

The adapter is intentionally thin: ``ChopGridLiveEngine`` owns grid state and
emits desired actions, while this module translates exchange-facing actions into
``BinanceAPI`` calls. It does not use ``TradeIntent`` or ``PositionTracker``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional

from src.order_management.binance_api import BinanceAPI
from src.order_management.models import OrderSide, OrderType

logger = logging.getLogger(__name__)


class GridExecutionError(RuntimeError):
    """Raised when a grid action cannot be executed safely."""


@dataclass(frozen=True)
class GridExecutionResult:
    """Result for one grid action processed by ``GridExecutionAdapter``."""

    action: str
    status: str
    symbol: str = ""
    order_id: str = ""
    client_order_id: str = ""
    raw: Optional[Dict[str, Any]] = None
    reason: str = ""


class GridExecutionAdapter:
    """Translate chop-grid actions into exchange orders.

    Real grid trading requires Binance Futures Hedge Mode so LONG and SHORT
    grid inventories remain separate. In one-way mode, exchange netting breaks
    the per-level accounting used by ``ChopGridLiveEngine``.
    """

    def __init__(
        self,
        binance_api: BinanceAPI,
        *,
        require_hedge_mode: bool = True,
        shadow: bool = False,
        client_id_prefix: str = "cg",
    ) -> None:
        self.binance_api = binance_api
        self.require_hedge_mode = bool(require_hedge_mode)
        self.shadow = bool(shadow)
        self.client_id_prefix = str(client_id_prefix or "cg")
        if self.require_hedge_mode and not bool(
            getattr(self.binance_api, "hedge_mode", False)
        ):
            raise GridExecutionError(
                "chop_grid real execution requires Binance Hedge Mode "
                "(dualSidePosition=true)"
            )

    def execute_actions(
        self, actions: Iterable[Dict[str, Any]]
    ) -> List[GridExecutionResult]:
        results: List[GridExecutionResult] = []
        for action in actions:
            results.append(self.execute_action(action))
        return results

    def execute_action(self, action: Dict[str, Any]) -> GridExecutionResult:
        kind = str(action.get("action", "") or "").strip().lower()
        if kind == "place":
            return self._place_limit(action)
        if kind == "cancel":
            return self._cancel(action)
        if kind == "market_exit":
            return self._market_exit(action)
        if kind in {"fill", "take_profit"}:
            # These are dry-run simulation events emitted by ChopGridLiveEngine.
            # Real fills must come from exchange order/user-stream sync.
            return GridExecutionResult(
                action=kind,
                status="ignored_simulation_event",
                symbol=str(action.get("symbol", "")),
                reason="fill/take_profit actions are not sent to exchange",
                raw=dict(action),
            )
        raise GridExecutionError(f"unsupported grid action: {kind or '<missing>'}")

    def sync_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open exchange orders for reconcile/reporting."""
        return self.binance_api.get_open_orders(symbol)

    def sync_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch exchange positions for reconcile/reporting."""
        return self.binance_api.get_positions(symbol)

    def _place_limit(self, action: Dict[str, Any]) -> GridExecutionResult:
        symbol = _required_str(action, "symbol")
        side = _order_side(_required_str(action, "side"))
        quantity = _required_positive_float(action, "quantity")
        price = _required_positive_float(action, "price")
        client_order_id = self._client_order_id(action)

        if self.shadow:
            logger.info(
                "shadow grid place: %s %s qty=%s price=%s cid=%s",
                symbol,
                side.value,
                quantity,
                price,
                client_order_id,
            )
            return GridExecutionResult(
                action="place",
                status="shadow",
                symbol=symbol,
                client_order_id=client_order_id,
                raw=dict(action),
            )

        order = self.binance_api.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
        )
        return GridExecutionResult(
            action="place",
            status=str(order.get("status", "submitted")),
            symbol=symbol,
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id") or client_order_id),
            raw=order,
        )

    def _cancel(self, action: Dict[str, Any]) -> GridExecutionResult:
        symbol = _required_str(action, "symbol", fallback="")
        order_id = str(action.get("exchange_order_id") or action.get("order_id") or "")
        if not order_id:
            raise GridExecutionError("cancel action requires order_id or exchange_order_id")
        if not symbol:
            raise GridExecutionError("cancel action requires symbol for exchange cancel")
        if self.shadow:
            return GridExecutionResult(
                action="cancel",
                status="shadow",
                symbol=symbol,
                order_id=order_id,
                raw=dict(action),
            )
        ok = self.binance_api.cancel_order(order_id, symbol)
        return GridExecutionResult(
            action="cancel",
            status="canceled" if ok else "not_canceled",
            symbol=symbol,
            order_id=order_id,
            raw=dict(action),
        )

    def _market_exit(self, action: Dict[str, Any]) -> GridExecutionResult:
        symbol = _required_str(action, "symbol")
        pos_side = _required_str(action, "side").upper()
        quantity = _required_positive_float(action, "quantity")
        side = OrderSide.SELL if pos_side == "LONG" else OrderSide.BUY
        if pos_side not in {"LONG", "SHORT"}:
            raise GridExecutionError(f"market_exit side must be LONG/SHORT: {pos_side}")

        client_order_id = self._client_order_id(action)
        if self.shadow:
            return GridExecutionResult(
                action="market_exit",
                status="shadow",
                symbol=symbol,
                client_order_id=client_order_id,
                raw=dict(action),
            )

        order = self.binance_api.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=True,
            client_order_id=client_order_id,
        )
        return GridExecutionResult(
            action="market_exit",
            status=str(order.get("status", "submitted")),
            symbol=symbol,
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id") or client_order_id),
            raw=order,
        )

    def _client_order_id(self, action: Dict[str, Any]) -> str:
        raw = str(
            action.get("client_order_id")
            or action.get("order_id")
            or f"{action.get('symbol','')}-{action.get('side','')}-{action.get('level','')}-{action.get('timestamp','')}"
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        prefix = self.client_id_prefix[:8]
        return f"{prefix}_{digest}"[:36]


def _required_str(action: Dict[str, Any], key: str, *, fallback: Any = None) -> str:
    value = action.get(key, fallback)
    if value is None or str(value).strip() == "":
        raise GridExecutionError(f"grid action requires {key}")
    return str(value).strip()


def _required_positive_float(action: Dict[str, Any], key: str) -> float:
    try:
        value = float(action.get(key))
    except (TypeError, ValueError) as exc:
        raise GridExecutionError(f"grid action requires numeric {key}") from exc
    if value <= 0:
        raise GridExecutionError(f"grid action requires positive {key}")
    return value


def _order_side(raw: str) -> OrderSide:
    side = str(raw or "").strip().upper()
    if side == "BUY":
        return OrderSide.BUY
    if side == "SELL":
        return OrderSide.SELL
    raise GridExecutionError(f"place side must be BUY/SELL: {raw}")
