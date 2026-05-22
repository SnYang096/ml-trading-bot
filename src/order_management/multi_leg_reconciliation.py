"""Reconciliation helpers for multi-leg strategy state vs exchange truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


@dataclass(frozen=True)
class LocalOrderSnapshot:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    exchange_order_id: str = ""
    client_order_id: str = ""


@dataclass(frozen=True)
class LocalPositionSnapshot:
    symbol: str
    side: str
    quantity: float


@dataclass(frozen=True)
class PositionMismatch:
    symbol: str
    side: str
    local_quantity: float
    exchange_quantity: float


@dataclass(frozen=True)
class ReconciliationPolicy:
    client_id_prefixes: Set[str] = field(default_factory=set)
    cancel_orphan_exchange_orders: bool = True
    quantity_tolerance: float = 1e-9
    # When chop_grid + dual_add_trend run on the same symbol, comparing one
    # engine's inventory to combined exchange qty is meaningless (false mismatch).
    skip_position_reconciliation: bool = False


@dataclass(frozen=True)
class ReconciliationReport:
    missing_exchange_orders: List[LocalOrderSnapshot] = field(default_factory=list)
    orphan_exchange_orders: List[Dict[str, Any]] = field(default_factory=list)
    position_mismatches: List[PositionMismatch] = field(default_factory=list)
    suggested_actions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.missing_exchange_orders
            and not self.orphan_exchange_orders
            and not self.position_mismatches
        )


class MultiLegReconciler:
    """Compare strategy-owned state to exchange open orders and positions."""

    def __init__(self, policy: Optional[ReconciliationPolicy] = None) -> None:
        self.policy = policy or ReconciliationPolicy()

    def reconcile(
        self,
        *,
        local_orders: Iterable[LocalOrderSnapshot] = (),
        exchange_orders: Iterable[Mapping[str, Any]] = (),
        local_positions: Iterable[LocalPositionSnapshot] = (),
        exchange_positions: Iterable[Mapping[str, Any]] = (),
    ) -> ReconciliationReport:
        local_orders_list = list(local_orders)
        exchange_orders_list = [dict(o) for o in exchange_orders]
        missing = self._missing_exchange_orders(local_orders_list, exchange_orders_list)
        orphans = self._orphan_exchange_orders(local_orders_list, exchange_orders_list)
        mismatches = self._position_mismatches(local_positions, exchange_positions)
        suggested = self._suggest_actions(orphans)
        return ReconciliationReport(
            missing_exchange_orders=missing,
            orphan_exchange_orders=orphans,
            position_mismatches=mismatches,
            suggested_actions=suggested,
        )

    def _missing_exchange_orders(
        self,
        local_orders: List[LocalOrderSnapshot],
        exchange_orders: List[Dict[str, Any]],
    ) -> List[LocalOrderSnapshot]:
        exchange_keys: Set[str] = set()
        for order in exchange_orders:
            exchange_keys.update(_order_key(order))
        missing: List[LocalOrderSnapshot] = []
        for order in local_orders:
            keys = {
                str(order.exchange_order_id or ""),
                str(order.client_order_id or ""),
                str(order.order_id or ""),
            }
            if keys.isdisjoint(exchange_keys):
                missing.append(order)
        return missing

    def _orphan_exchange_orders(
        self,
        local_orders: List[LocalOrderSnapshot],
        exchange_orders: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        local_keys = set()
        for order in local_orders:
            local_keys.update(
                {
                    str(order.exchange_order_id or ""),
                    str(order.client_order_id or ""),
                    str(order.order_id or ""),
                }
            )
        local_keys.discard("")
        orphans: List[Dict[str, Any]] = []
        for order in exchange_orders:
            if not self._in_scope(order):
                continue
            if _order_key(order).isdisjoint(local_keys):
                orphans.append(dict(order))
        return orphans

    def _position_mismatches(
        self,
        local_positions: Iterable[LocalPositionSnapshot],
        exchange_positions: Iterable[Mapping[str, Any]],
    ) -> List[PositionMismatch]:
        if self.policy.skip_position_reconciliation:
            return []
        local = _position_quantities(local_positions)
        exchange = _exchange_position_quantities(exchange_positions)
        keys = set(local) | set(exchange)
        mismatches: List[PositionMismatch] = []
        for symbol, side in sorted(keys):
            local_qty = local.get((symbol, side), 0.0)
            exchange_qty = exchange.get((symbol, side), 0.0)
            if abs(local_qty - exchange_qty) > self.policy.quantity_tolerance:
                mismatches.append(
                    PositionMismatch(
                        symbol=symbol,
                        side=side,
                        local_quantity=local_qty,
                        exchange_quantity=exchange_qty,
                    )
                )
        return mismatches

    def _suggest_actions(self, orphans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.policy.cancel_orphan_exchange_orders:
            return []
        actions: List[Dict[str, Any]] = []
        for order in orphans:
            order_id = str(order.get("order_id") or order.get("orderId") or "")
            symbol = str(order.get("symbol", "") or "").upper()
            if order_id and symbol:
                actions.append(
                    {
                        "action": "cancel",
                        "symbol": symbol,
                        "exchange_order_id": order_id,
                        "reason": "orphan_exchange_order",
                        "is_algo_order": bool(order.get("_is_algo_order")),
                    }
                )
        return actions

    def _in_scope(self, order: Mapping[str, Any]) -> bool:
        if not self.policy.client_id_prefixes:
            return True
        cid = str(order.get("client_order_id") or order.get("clientOrderId") or "")
        return any(cid.startswith(prefix) for prefix in self.policy.client_id_prefixes)


def _order_key(order: Mapping[str, Any]) -> Set[str]:
    return {
        str(order.get("order_id") or order.get("orderId") or ""),
        str(order.get("client_order_id") or order.get("clientOrderId") or ""),
    } - {""}


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").upper().strip()
    if "/" in raw:
        base, rest = raw.split("/", 1)
        quote = rest.split(":", 1)[0]
        return f"{base}{quote}"
    return raw.split(":", 1)[0]


def _position_quantities(
    positions: Iterable[LocalPositionSnapshot],
) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for pos in positions:
        key = (_normalize_symbol(pos.symbol), str(pos.side).upper())
        out[key] = out.get(key, 0.0) + abs(float(pos.quantity))
    return out


def _exchange_position_quantities(
    positions: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for pos in positions:
        symbol = _normalize_symbol(pos.get("symbol", ""))
        side = str(pos.get("position_side") or pos.get("positionSide") or "").upper()
        if not side:
            amount = float(pos.get("position_amount") or pos.get("positionAmt") or 0.0)
            side = "LONG" if amount >= 0 else "SHORT"
        quantity = abs(float(pos.get("position_amount") or pos.get("positionAmt") or 0.0))
        if symbol and side and quantity > 0:
            key = (symbol, side)
            out[key] = out.get(key, 0.0) + quantity
    return out
