"""Execution adapter for standalone multi-leg order actions.

The adapter is intentionally thin: multi-leg engines own their inventory state
and emit desired actions, while this module translates exchange-facing actions
into ``BinanceAPI`` calls. It does not use ``TradeIntent`` or
``PositionTracker``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional

from src.order_management.binance_api import BinanceAPI
from src.order_management.models import OrderSide, OrderType

logger = logging.getLogger(__name__)


class MultiLegExecutionError(RuntimeError):
    """Raised when a multi-leg action cannot be executed safely."""


@dataclass(frozen=True)
class MultiLegExecutionResult:
    """Result for one multi-leg action processed by ``MultiLegExecutionAdapter``."""

    action: str
    status: str
    symbol: str = ""
    order_id: str = ""
    client_order_id: str = ""
    raw: Optional[Dict[str, Any]] = None
    reason: str = ""


class MultiLegExecutionAdapter:
    """Translate multi-leg actions into exchange orders.

    Real multi-leg trading requires Binance Futures Hedge Mode so LONG and SHORT
    inventories remain separate. In one-way mode, exchange netting breaks the
    per-leg accounting used by the engines.
    """

    def __init__(
        self,
        binance_api: BinanceAPI,
        *,
        require_hedge_mode: bool = True,
        shadow: bool = False,
        client_id_prefix: str = "cg",
        default_symbol: Optional[str] = None,
        storage: Optional[Any] = None,
        run_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> None:
        self.binance_api = binance_api
        self.require_hedge_mode = bool(require_hedge_mode)
        self.shadow = bool(shadow)
        self.client_id_prefix = str(client_id_prefix or "cg")
        self.storage = storage
        self.run_id = run_id
        self.strategy_name = str(strategy_name or self.client_id_prefix)
        ds = str(default_symbol or "").strip().upper()
        self.default_symbol: Optional[str] = ds or None
        if self.require_hedge_mode and not bool(
            getattr(self.binance_api, "hedge_mode", False)
        ):
            raise MultiLegExecutionError(
                "multi-leg real execution requires Binance Hedge Mode "
                "(dualSidePosition=true)"
            )

    def execute_actions(
        self, actions: Iterable[Dict[str, Any]]
    ) -> List[MultiLegExecutionResult]:
        results: List[MultiLegExecutionResult] = []
        for action in actions:
            results.append(self.execute_action(action))
        return results

    def execute_action(self, action: Dict[str, Any]) -> MultiLegExecutionResult:
        kind = str(action.get("action", "") or "").strip().lower()
        if kind == "place":
            return self._place_entry(action)
        if kind == "cancel":
            return self._cancel(action)
        if kind == "cancel_protection":
            return self._cancel(action)
        if kind == "market_exit":
            return self._market_exit(action)
        if kind == "place_protection":
            return self._place_protection(action)
        if kind in {"fill", "take_profit"}:
            # These are dry-run simulation events emitted by ChopGridLiveEngine.
            # Real fills must come from exchange order/user-stream sync.
            return MultiLegExecutionResult(
                action=kind,
                status="ignored_simulation_event",
                symbol=str(action.get("symbol", "")),
                reason="fill/take_profit actions are not sent to exchange",
                raw=dict(action),
            )
        raise MultiLegExecutionError(
            f"unsupported multi-leg action: {kind or '<missing>'}"
        )

    def sync_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open exchange orders for reconcile/reporting."""
        sym = symbol or self.default_symbol
        return self.binance_api.get_open_orders(sym)

    def sync_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch exchange positions for reconcile/reporting."""
        sym = symbol or self.default_symbol
        return self.binance_api.get_positions(sym)

    def _place_entry(self, action: Dict[str, Any]) -> MultiLegExecutionResult:
        symbol = _required_str(action, "symbol")
        side = _order_side(_required_str(action, "side"))
        quantity = _required_positive_float(action, "quantity")
        raw_order_type = str(action.get("order_type") or "limit").strip().lower()
        order_type = OrderType.MARKET if raw_order_type == "market" else OrderType.LIMIT
        price = (
            None
            if order_type == OrderType.MARKET
            else _required_positive_float(action, "price")
        )
        client_order_id = self._client_order_id(action)
        time_in_force = str(action.get("time_in_force") or "").strip().upper() or None

        if self.shadow:
            logger.info(
                "shadow grid place: %s %s %s qty=%s price=%s tif=%s cid=%s",
                symbol,
                side.value,
                order_type.value,
                quantity,
                price,
                time_in_force,
                client_order_id,
            )
            result = MultiLegExecutionResult(
                action="place",
                status="shadow",
                symbol=symbol,
                client_order_id=client_order_id,
                raw=dict(action),
            )
            self._persist_order_result(action, result, purpose="entry")
            return result

        order = self.binance_api.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
            time_in_force=time_in_force,
        )
        result = MultiLegExecutionResult(
            action="place",
            status=str(order.get("status", "submitted")),
            symbol=symbol,
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id") or client_order_id),
            raw=order,
        )
        self._persist_order_result(action, result, purpose="entry")
        return result

    def _cancel(self, action: Dict[str, Any]) -> MultiLegExecutionResult:
        symbol = _required_str(action, "symbol", fallback="")
        order_id = str(action.get("exchange_order_id") or action.get("order_id") or "")
        if not order_id:
            raise MultiLegExecutionError(
                "cancel action requires order_id or exchange_order_id"
            )
        if not symbol:
            raise MultiLegExecutionError(
                "cancel action requires symbol for exchange cancel"
            )
        if self.shadow:
            result = MultiLegExecutionResult(
                action="cancel",
                status="shadow",
                symbol=symbol,
                order_id=order_id,
                raw=dict(action),
            )
            self._persist_order_result(action, result, purpose="cancel")
            return result
        ok = self.binance_api.cancel_order(order_id, symbol)
        result = MultiLegExecutionResult(
            action="cancel",
            status="canceled" if ok else "not_canceled",
            symbol=symbol,
            order_id=order_id,
            raw=dict(action),
        )
        self._persist_order_result(action, result, purpose="cancel")
        return result

    def _market_exit(self, action: Dict[str, Any]) -> MultiLegExecutionResult:
        symbol = _required_str(action, "symbol")
        pos_side = _required_str(action, "side").upper()
        quantity = _required_positive_float(action, "quantity")
        side = OrderSide.SELL if pos_side == "LONG" else OrderSide.BUY
        if pos_side not in {"LONG", "SHORT"}:
            raise MultiLegExecutionError(
                f"market_exit side must be LONG/SHORT: {pos_side}"
            )

        client_order_id = self._client_order_id(action)
        if self.shadow:
            result = MultiLegExecutionResult(
                action="market_exit",
                status="shadow",
                symbol=symbol,
                client_order_id=client_order_id,
                raw=dict(action),
            )
            self._persist_order_result(action, result, purpose="market_exit")
            return result

        order = self.binance_api.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=True,
            client_order_id=client_order_id,
        )
        result = MultiLegExecutionResult(
            action="market_exit",
            status=str(order.get("status", "submitted")),
            symbol=symbol,
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id") or client_order_id),
            raw=order,
        )
        self._persist_order_result(action, result, purpose="market_exit")
        return result

    def _place_protection(self, action: Dict[str, Any]) -> MultiLegExecutionResult:
        """Place a native reduce-only stop-loss or take-profit order.

        Protection is sized per logical leg. We intentionally avoid
        ``closePosition=True`` here because in Hedge Mode it closes the whole
        symbol side, not just the logical leg represented by this action.
        """

        symbol = _required_str(action, "symbol")
        pos_side = _required_str(action, "side").upper()
        if pos_side not in {"LONG", "SHORT"}:
            raise MultiLegExecutionError(
                f"place_protection side must be LONG/SHORT: {pos_side}"
            )
        quantity = _required_positive_float(action, "quantity")
        protection_price = _required_positive_float(
            action,
            "stop_price" if "stop_price" in action else "trigger_price",
        )
        raw_kind = str(
            action.get("protection_type")
            or action.get("order_type")
            or action.get("purpose")
            or ""
        ).lower()
        if raw_kind in {"stop", "sl", "stop_loss", "stop_market"}:
            order_type = OrderType.STOP_MARKET
            purpose = "stop_loss"
        elif raw_kind in {"take_profit", "tp", "take_profit_market"}:
            tp_order_type = str(action.get("order_type") or "").strip().lower()
            order_type = (
                OrderType.LIMIT
                if tp_order_type in {"limit", "post_only_limit"}
                or bool(action.get("post_only"))
                else OrderType.TAKE_PROFIT_MARKET
            )
            purpose = "take_profit"
        else:
            raise MultiLegExecutionError(
                "place_protection requires protection_type stop_loss/take_profit"
            )

        side = OrderSide.SELL if pos_side == "LONG" else OrderSide.BUY
        client_order_id = self._client_order_id({**action, "purpose": purpose})
        order_price = (
            _required_positive_float(action, "price")
            if order_type == OrderType.LIMIT
            else None
        )
        if self.shadow:
            result = MultiLegExecutionResult(
                action="place_protection",
                status="shadow",
                symbol=symbol,
                client_order_id=client_order_id,
                raw={**dict(action), "purpose": purpose, "order_type": order_type.value},
            )
            self._persist_order_result(action, result, purpose=purpose)
            return result

        order = self.binance_api.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=order_price,
            stop_price=None if order_type == OrderType.LIMIT else protection_price,
            reduce_only=True,
            close_position=False,
            client_order_id=client_order_id,
            position_side=pos_side,
            working_type=(
                None
                if order_type == OrderType.LIMIT
                else str(action.get("working_type") or "MARK_PRICE")
            ),
            price_protect=(
                None
                if order_type == OrderType.LIMIT
                else bool(action.get("price_protect", True))
            ),
            post_only=bool(action.get("post_only")),
            time_in_force=str(action.get("time_in_force") or "").strip().upper()
            or None,
        )
        result = MultiLegExecutionResult(
            action="place_protection",
            status=str(order.get("status", "submitted")),
            symbol=symbol,
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id") or client_order_id),
            raw={
                **order,
                "purpose": purpose,
                "local_order_id": action.get("order_id"),
                "leg_id": action.get("leg_id") or action.get("order_id"),
            },
        )
        self._persist_order_result(action, result, purpose=purpose)
        return result

    def _client_order_id(self, action: Dict[str, Any]) -> str:
        raw = str(
            action.get("client_order_id")
            or action.get("order_id")
            or (
                f"{action.get('symbol','')}-{action.get('side','')}-"
                f"{action.get('level','')}-{action.get('purpose','')}-"
                f"{action.get('action','')}-{action.get('timestamp','')}"
            )
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        prefix = self.client_id_prefix[:8]
        return f"{prefix}_{digest}"[:36]

    def _persist_order_result(
        self,
        action: Dict[str, Any],
        result: MultiLegExecutionResult,
        *,
        purpose: str,
    ) -> None:
        if self.storage is None:
            return
        try:
            self.storage.upsert_order(
                {
                    "run_id": self.run_id,
                    "strategy": self.strategy_name,
                    "local_order_id": action.get("order_id")
                    or result.client_order_id
                    or result.order_id,
                    "exchange_order_id": result.order_id,
                    "client_order_id": result.client_order_id,
                    "symbol": result.symbol or action.get("symbol"),
                    "leg_id": action.get("leg_id") or action.get("order_id"),
                    "side": action.get("side"),
                    "position_side": action.get("position_side")
                    or (
                        str(action.get("side")).upper()
                        if str(action.get("side")).upper() in {"LONG", "SHORT"}
                        else None
                    ),
                    "order_type": action.get("order_type") or action.get("action"),
                    "purpose": purpose,
                    "quantity": action.get("quantity", 0.0),
                    "price": action.get("price"),
                    "stop_price": action.get("stop_price")
                    or action.get("trigger_price"),
                    "status": result.status,
                    "raw": result.raw or dict(action),
                }
            )
        except Exception as exc:  # pragma: no cover - persistence must not block trading
            logger.warning("multi-leg order persistence failed: %s", exc)


def _required_str(action: Dict[str, Any], key: str, *, fallback: Any = None) -> str:
    value = action.get(key, fallback)
    if value is None or str(value).strip() == "":
        raise MultiLegExecutionError(f"multi-leg action requires {key}")
    return str(value).strip()


def _required_positive_float(action: Dict[str, Any], key: str) -> float:
    try:
        value = float(action.get(key))
    except (TypeError, ValueError) as exc:
        raise MultiLegExecutionError(
            f"multi-leg action requires numeric {key}"
        ) from exc
    if value <= 0:
        raise MultiLegExecutionError(f"multi-leg action requires positive {key}")
    return value


def _order_side(raw: str) -> OrderSide:
    side = str(raw or "").strip().upper()
    if side == "BUY":
        return OrderSide.BUY
    if side == "SELL":
        return OrderSide.SELL
    raise MultiLegExecutionError(f"place side must be BUY/SELL: {raw}")


# Backward-compatible aliases while callers migrate to multi-leg naming.
GridExecutionError = MultiLegExecutionError
GridExecutionResult = MultiLegExecutionResult
GridExecutionAdapter = MultiLegExecutionAdapter
