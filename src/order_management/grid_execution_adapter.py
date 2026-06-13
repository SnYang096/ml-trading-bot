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


def _cancel_reason_bucket(reason: str) -> str:
    """Low-cardinality bucket for Prometheus; see audit logs for raw reason."""
    r = (reason or "").strip().lower()
    if not r or r == "unspecified":
        return "unspecified"
    if "orphan" in r:
        return "orphan_exchange"
    if "reconcile" in r or "mismatch" in r:
        return "reconcile"
    return "other"


def _market_exit_reason_bucket(action: Dict[str, Any]) -> str:
    """Coarse bucket from engine ``reason=`` field (full text still in audit logs)."""
    r = str(action.get("reason") or "").strip().lower()
    if not r:
        return "unspecified"
    if "regime" in r:
        return "regime_exit"
    if "risk" in r or "catastrophic" in r or "stop" in r:
        return "risk_stop"
    if "flip" in r:
        return "trend_flip"
    if "orphan" in r or "reconcile" in r:
        return "reconcile"
    return "other"


def _exchange_place_reject_reason(
    api: Any,
    symbol: str,
    quantity: float,
    *,
    price: Optional[float] = None,
    skip_min_notional: bool = False,
) -> str:
    """Return a rejection reason when qty/notional is below exchange limits, else ''.

    ``skip_min_notional`` exempts the min-notional filter (reduce-only market
    exits are accepted by the exchange even below it) while still enforcing the
    hard min-quantity (LOT_SIZE) filter that no order can bypass.
    """
    getter = getattr(api, "get_symbol_info", None)
    if not callable(getter):
        return ""
    try:
        info = getter(symbol)
    except Exception:
        logger.debug("exchange min-order check: get_symbol_info failed", exc_info=True)
        return ""
    if not info:
        return ""
    limits = info.get("limits") if isinstance(info, dict) else None
    if not isinstance(limits, dict):
        return ""
    amount_limits = limits.get("amount") or {}
    min_amount = amount_limits.get("min")
    if min_amount is not None:
        try:
            min_qty = float(min_amount)
        except (TypeError, ValueError):
            min_qty = 0.0
        if min_qty > 0.0 and float(quantity) < min_qty:
            return (
                f"exchange_min_qty: quantity {float(quantity):.8f} < "
                f"min {min_qty:.8f} for {symbol}"
            )
    if skip_min_notional:
        return ""
    cost_limits = limits.get("cost") or {}
    min_cost = cost_limits.get("min")
    if min_cost is not None and price is not None and float(price) > 0:
        try:
            min_notional = float(min_cost)
        except (TypeError, ValueError):
            min_notional = 0.0
        notional = float(quantity) * float(price)
        if min_notional > 0.0 and notional < min_notional:
            return (
                f"exchange_min_notional: notional {notional:.4f} < "
                f"min {min_notional:.4f} for {symbol}"
            )
    return ""


def _is_invalid_order_size_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "invalidorder" in text
        or "minimum amount" in text
        or "must be greater than minimum" in text
        or "min notional" in text
    )


def _resolve_live_cancel_order_id(action: Dict[str, Any]) -> str:
    """Return an exchange order id suitable for cancel API calls, or empty."""
    exchange_id = str(action.get("exchange_order_id") or "").strip()
    if exchange_id:
        return exchange_id
    order_id = str(action.get("order_id") or "").strip()
    if order_id.isdigit():
        return order_id
    return ""


def _is_duplicate_client_order_id_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "-4116" in text or "clientorderid is duplicated" in text


def _is_reduce_only_rejected_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "-2022" in text or "reduceonly order is rejected" in text


def derive_multileg_client_order_id(
    action: Dict[str, Any], *, client_id_prefix: str = "cg"
) -> str:
    """Deterministic exchange client id (must match MultiLegExecutionAdapter)."""
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
    prefix = str(client_id_prefix or "cg")[:8]
    return f"{prefix}_{digest}"[:36]


def _is_live_order_status(status: Any) -> bool:
    return str(status or "").strip().lower() in {
        "open",
        "new",
        "submitted",
        "accepted",
        "partially_filled",
        "partial",
        "working",
        "pending",
        "triggered",
    }


def _find_protection_order_by_client_id(
    binance_api: Any, client_order_id: str, symbol: str
) -> Optional[Dict[str, Any]]:
    """Resolve an existing protection order (regular limit or algo conditional)."""
    cid = str(client_order_id or "").strip()
    if not cid:
        return None
    fetch = getattr(binance_api, "get_order_by_client_id", None)
    if callable(fetch):
        try:
            order = fetch(cid, symbol)
        except Exception as exc:
            logger.warning(
                "protection lookup failed client_order_id=%s symbol=%s: %s",
                cid,
                symbol,
                exc,
            )
            order = None
        if order and _is_live_order_status(order.get("status")):
            return order
    scan = getattr(binance_api, "get_open_orders_for_sl_cleanup", None)
    if callable(scan):
        for row in scan(symbol) or []:
            row_cid = str(
                row.get("client_order_id")
                or (row.get("info") or {}).get("clientAlgoId")
                or (row.get("info") or {}).get("clientOrderId")
                or ""
            ).strip()
            if row_cid == cid and _is_live_order_status(row.get("status")):
                return row
    return None


def _is_order_already_gone(exc: BaseException) -> bool:
    """True when cancel target is already filled/canceled (Binance -2011, etc.)."""
    name = type(exc).__name__.lower()
    if "ordernotfound" in name or "order not found" in name:
        return True
    msg = str(exc).lower()
    return "unknown order" in msg or "-2011" in msg or "order does not exist" in msg


def _is_binance_rate_limit_detail(text: str) -> bool:
    """Recognize Binance / ccxt wording for REST rate limits (-1003, HTTP 429)."""
    chunk = str(text or "")
    lowered = chunk.lower()
    return (
        "-1003" in chunk
        or "429" in chunk
        or "too many requests" in lowered
        or "ddosprotection" in lowered.replace(" ", "")
    )


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
        probe_err: Optional[str] = getattr(binance_api, "hedge_mode_probe_error", None)
        hm = bool(getattr(self.binance_api, "hedge_mode", False))
        if self.require_hedge_mode and not bool(self.shadow):
            rate_hit = probe_err and _is_binance_rate_limit_detail(probe_err)
            if probe_err and not rate_hit:
                raise MultiLegExecutionError(
                    "cannot verify Binance USDM hedge mode (/fapi/v1/positionSide/dual "
                    "failed). Check Futures API permission on this key, IP whitelist "
                    "for your server egress, or use MULTI_LEG_BINANCE_* keys if "
                    f"distinct from BINANCE_* . Detail: {probe_err}"
                )
            if rate_hit:
                logger.warning(
                    "multi-leg: Hedge Mode probe returned rate-limit error; "
                    "continuing assuming the account stays in Hedge Mode (verify "
                    "when Binance quota recovers): %s",
                    probe_err,
                )
            elif not hm:
                raise MultiLegExecutionError(
                    "multi-leg real execution requires Binance Hedge Mode "
                    "(dualSidePosition=true). Enable Hedge Mode under Binance Futures "
                    "preferences for this account."
                )

    def _record_cancel_reason_metric(self, symbol: str, reason: str) -> None:
        try:
            from src.time_series_model.live.metrics_exporter import METRICS

            METRICS.multi_leg_cancel_reason_bucket_total.labels(
                strategy=self.strategy_name,
                symbol=str(symbol or "").strip().upper(),
                reason_bucket=_cancel_reason_bucket(reason),
            ).inc(1)
        except Exception:
            logger.debug("multi-leg cancel reason metric skipped", exc_info=True)

    def _record_market_exit_metric(self, symbol: str) -> None:
        try:
            from src.time_series_model.live.metrics_exporter import METRICS

            METRICS.multi_leg_market_exit_total.labels(
                strategy=self.strategy_name,
                symbol=str(symbol or "").strip().upper(),
            ).inc(1)
        except Exception:
            logger.debug("multi-leg market_exit metric skipped", exc_info=True)

    def _record_market_exit_reason_metric(
        self, symbol: str, action: Dict[str, Any]
    ) -> None:
        try:
            from src.time_series_model.live.metrics_exporter import METRICS

            METRICS.multi_leg_market_exit_reason_bucket_total.labels(
                strategy=self.strategy_name,
                symbol=str(symbol or "").strip().upper(),
                reason_bucket=_market_exit_reason_bucket(action),
            ).inc(1)
        except Exception:
            logger.debug(
                "multi-leg market_exit reason metric skipped", exc_info=True
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
        """Fetch open exchange orders for reconcile/reporting (incl. algo/conditional)."""
        sym = symbol or self.default_symbol
        fetch = getattr(self.binance_api, "get_open_orders_for_sl_cleanup", None)
        if callable(fetch):
            return list(fetch(sym) or [])
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

        logger.info(
            "multi-leg place requested: strategy=%s symbol=%s side=%s type=%s qty=%s price=%s tif=%s local_order_id=%s",
            self.strategy_name,
            symbol,
            side.value,
            order_type.value,
            quantity,
            price,
            time_in_force,
            str(action.get("order_id") or ""),
        )
        reject = _exchange_place_reject_reason(
            self.binance_api, symbol, quantity, price=price
        )
        if reject:
            logger.warning(
                "multi-leg place rejected (%s): strategy=%s symbol=%s "
                "qty=%s price=%s local_order_id=%s",
                reject,
                self.strategy_name,
                symbol,
                quantity,
                price,
                str(action.get("order_id") or ""),
            )
            result = MultiLegExecutionResult(
                action="place",
                status="rejected",
                symbol=symbol,
                client_order_id=client_order_id,
                reason=reject,
                raw={**dict(action), "reject_reason": reject},
            )
            self._persist_order_result(action, result, purpose="entry")
            return result
        try:
            order = self.binance_api.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                client_order_id=client_order_id,
                time_in_force=time_in_force,
            )
        except Exception as exc:
            if _is_invalid_order_size_error(exc):
                reject = str(exc)
                logger.warning(
                    "multi-leg place rejected (exchange): strategy=%s symbol=%s "
                    "qty=%s reason=%s",
                    self.strategy_name,
                    symbol,
                    quantity,
                    reject,
                )
                result = MultiLegExecutionResult(
                    action="place",
                    status="rejected",
                    symbol=symbol,
                    client_order_id=client_order_id,
                    reason=reject,
                    raw={**dict(action), "reject_reason": reject},
                )
                self._persist_order_result(action, result, purpose="entry")
                return result
            raise
        result = MultiLegExecutionResult(
            action="place",
            status=str(order.get("status", "submitted")),
            symbol=symbol,
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id") or client_order_id),
            raw={
                **dict(order),
                "local_order_id": action.get("order_id"),
                "local_client_order_id": client_order_id,
            },
        )
        logger.info(
            "multi-leg place result: strategy=%s symbol=%s local_order_id=%s exchange_order_id=%s client_order_id=%s status=%s",
            self.strategy_name,
            symbol,
            str(action.get("order_id") or ""),
            result.order_id,
            result.client_order_id,
            result.status,
        )
        self._persist_order_result(action, result, purpose="entry")
        return result

    def _cancel(self, action: Dict[str, Any]) -> MultiLegExecutionResult:
        symbol = _required_str(action, "symbol", fallback="")
        local_order_id = str(action.get("order_id") or "").strip()
        order_id = (
            _resolve_live_cancel_order_id(action)
            if not self.shadow
            else str(action.get("exchange_order_id") or action.get("order_id") or "").strip()
        )
        if not order_id:
            if not self.shadow and local_order_id:
                reason = str(action.get("reason") or "").strip()
                logger.warning(
                    "multi-leg cancel skipped (local-only order id): strategy=%s "
                    "symbol=%s local_order_id=%s reason=%s",
                    self.strategy_name,
                    symbol,
                    local_order_id,
                    reason or "unspecified",
                )
                result = MultiLegExecutionResult(
                    action="cancel",
                    status="canceled",
                    symbol=symbol,
                    order_id=local_order_id,
                    raw=dict(action),
                )
                self._persist_order_result(action, result, purpose="cancel")
                return result
            raise MultiLegExecutionError(
                "cancel action requires order_id or exchange_order_id"
            )
        if not symbol:
            raise MultiLegExecutionError(
                "cancel action requires symbol for exchange cancel"
            )
        reason = str(action.get("reason") or "").strip()
        kind = "shadow" if self.shadow else "live"
        logger.info(
            "multi-leg cancel (%s): strategy=%s symbol=%s order_id=%s reason=%s",
            kind,
            self.strategy_name,
            symbol,
            order_id,
            reason or "unspecified",
        )
        self._record_cancel_reason_metric(symbol, reason)
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
        is_algo = bool(action.get("is_algo_order") or action.get("_is_algo_order"))
        try:
            if is_algo and hasattr(self.binance_api, "cancel_algo_order"):
                ok = bool(self.binance_api.cancel_algo_order(order_id, symbol))
            else:
                ok = self.binance_api.cancel_order(order_id, symbol)
        except Exception as exc:
            if _is_order_already_gone(exc):
                logger.info(
                    "multi-leg cancel: order already gone (%s): strategy=%s "
                    "symbol=%s order_id=%s reason=%s",
                    exc,
                    self.strategy_name,
                    symbol,
                    order_id,
                    reason or "unspecified",
                )
                ok = True
            else:
                raise
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
        logger.info(
            "multi-leg market_exit requested: strategy=%s symbol=%s position_side=%s qty=%s local_order_id=%s",
            self.strategy_name,
            symbol,
            pos_side,
            quantity,
            str(action.get("order_id") or ""),
        )
        self._record_market_exit_metric(symbol)
        self._record_market_exit_reason_metric(symbol, action)
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

        mark: Optional[float] = None
        for key in ("exit_price", "mark_price", "price"):
            raw = action.get(key)
            if raw is None:
                continue
            try:
                candidate = float(raw)
            except (TypeError, ValueError):
                continue
            if candidate > 0:
                mark = candidate
                break
        # Reduce-only market exits may close below min notional; the exchange
        # accepts them even when resting TP/SL would be rejected. Still enforce
        # the hard min-quantity (LOT_SIZE) filter to avoid an exchange exception.
        reject = _exchange_place_reject_reason(
            self.binance_api, symbol, quantity, price=mark, skip_min_notional=True
        )
        if reject:
            logger.warning(
                "multi-leg market_exit rejected (%s): strategy=%s symbol=%s qty=%s",
                reject,
                self.strategy_name,
                symbol,
                quantity,
            )
            result = MultiLegExecutionResult(
                action="market_exit",
                status="rejected",
                symbol=symbol,
                client_order_id=client_order_id,
                reason=reject,
                raw={**dict(action), "reject_reason": reject},
            )
            self._persist_order_result(action, result, purpose="market_exit")
            return result
        try:
            order = self.binance_api.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                reduce_only=True,
                client_order_id=client_order_id,
            )
        except Exception as exc:
            if _is_reduce_only_rejected_error(exc):
                # -2022: exchange has no (remaining) position to reduce. The local
                # inventory is stale; treat as already-flat instead of crashing the
                # whole multi-leg daemon (mirrors _place_protection handling).
                logger.warning(
                    "multi-leg market_exit skipped (no exchange position to reduce): "
                    "strategy=%s symbol=%s qty=%s local_order_id=%s error=%s",
                    self.strategy_name,
                    symbol,
                    quantity,
                    str(action.get("order_id") or ""),
                    exc,
                )
                result = MultiLegExecutionResult(
                    action="market_exit",
                    status="skipped_no_position",
                    symbol=symbol,
                    client_order_id=client_order_id,
                    reason=str(exc),
                    raw={
                        **dict(action),
                        "local_order_id": action.get("order_id"),
                        "error": str(exc),
                    },
                )
                self._persist_order_result(action, result, purpose="market_exit")
                return result
            if _is_invalid_order_size_error(exc):
                reject = str(exc)
                logger.warning(
                    "multi-leg market_exit rejected (exchange): strategy=%s "
                    "symbol=%s qty=%s reason=%s",
                    self.strategy_name,
                    symbol,
                    quantity,
                    reject,
                )
                result = MultiLegExecutionResult(
                    action="market_exit",
                    status="rejected",
                    symbol=symbol,
                    client_order_id=client_order_id,
                    reason=reject,
                    raw={**dict(action), "reject_reason": reject},
                )
                self._persist_order_result(action, result, purpose="market_exit")
                return result
            raise
        result = MultiLegExecutionResult(
            action="market_exit",
            status=str(order.get("status", "submitted")),
            symbol=symbol,
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id") or client_order_id),
            raw={
                **dict(order),
                "local_order_id": action.get("order_id"),
                "local_client_order_id": client_order_id,
            },
        )
        logger.info(
            "multi-leg market_exit result: strategy=%s symbol=%s local_order_id=%s exchange_order_id=%s client_order_id=%s status=%s",
            self.strategy_name,
            symbol,
            str(action.get("order_id") or ""),
            result.order_id,
            result.client_order_id,
            result.status,
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
                raw={
                    **dict(action),
                    "purpose": purpose,
                    "order_type": order_type.value,
                },
            )
            self._persist_order_result(action, result, purpose=purpose)
            return result

        existing = _find_protection_order_by_client_id(
            self.binance_api, client_order_id, symbol
        )
        if existing:
            logger.debug(
                "multi-leg protection already open: local_order_id=%s "
                "client_order_id=%s exchange_order_id=%s",
                action.get("order_id"),
                client_order_id,
                existing.get("order_id"),
            )
            result = MultiLegExecutionResult(
                action="place_protection",
                status=str(existing.get("status", "submitted")),
                symbol=symbol,
                order_id=str(existing.get("order_id", "")),
                client_order_id=str(existing.get("client_order_id") or client_order_id),
                raw={
                    **existing,
                    "purpose": purpose,
                    "local_order_id": action.get("order_id"),
                    "leg_id": action.get("leg_id") or action.get("order_id"),
                },
            )
            self._persist_order_result(action, result, purpose=purpose)
            return result

        prot_price = order_price if order_price is not None else protection_price
        reject = _exchange_place_reject_reason(
            self.binance_api, symbol, quantity, price=prot_price
        )
        if reject:
            logger.warning(
                "multi-leg place_protection rejected (%s): strategy=%s symbol=%s "
                "qty=%s leg_id=%s",
                reject,
                self.strategy_name,
                symbol,
                quantity,
                action.get("leg_id") or action.get("order_id"),
            )
            result = MultiLegExecutionResult(
                action="place_protection",
                status="rejected",
                symbol=symbol,
                client_order_id=client_order_id,
                reason=reject,
                raw={**dict(action), "reject_reason": reject, "purpose": purpose},
            )
            self._persist_order_result(action, result, purpose=purpose)
            return result

        try:
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
        except Exception as exc:
            if _is_reduce_only_rejected_error(exc):
                logger.warning(
                    "multi-leg protection skipped (no exchange position to reduce): "
                    "symbol=%s leg_id=%s client_order_id=%s error=%s",
                    symbol,
                    action.get("leg_id") or action.get("order_id"),
                    client_order_id,
                    exc,
                )
                result = MultiLegExecutionResult(
                    action="place_protection",
                    status="skipped_no_position",
                    symbol=symbol,
                    client_order_id=client_order_id,
                    raw={
                        **dict(action),
                        "purpose": purpose,
                        "local_order_id": action.get("order_id"),
                        "leg_id": action.get("leg_id") or action.get("order_id"),
                        "error": str(exc),
                    },
                )
                self._persist_order_result(action, result, purpose=purpose)
                return result
            if _is_invalid_order_size_error(exc):
                reject = str(exc)
                logger.warning(
                    "multi-leg place_protection rejected (exchange): strategy=%s "
                    "symbol=%s qty=%s reason=%s",
                    self.strategy_name,
                    symbol,
                    quantity,
                    reject,
                )
                result = MultiLegExecutionResult(
                    action="place_protection",
                    status="rejected",
                    symbol=symbol,
                    client_order_id=client_order_id,
                    reason=reject,
                    raw={**dict(action), "reject_reason": reject, "purpose": purpose},
                )
                self._persist_order_result(action, result, purpose=purpose)
                return result
            if not _is_duplicate_client_order_id_error(exc):
                raise
            order = _find_protection_order_by_client_id(
                self.binance_api, client_order_id, symbol
            )
            if not order:
                raise
            logger.warning(
                "multi-leg protection order already exists: local_order_id=%s "
                "client_order_id=%s exchange_order_id=%s",
                action.get("order_id"),
                client_order_id,
                order.get("order_id"),
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
        return derive_multileg_client_order_id(
            action, client_id_prefix=self.client_id_prefix
        )

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
            raw = result.raw or dict(action)
            info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
            filled_qty = 0.0
            for key in ("filled", "filled_quantity", "executedQty"):
                val = raw.get(key) if key != "executedQty" else info.get("executedQty")
                if val is None:
                    continue
                try:
                    filled_qty = float(val)
                except (TypeError, ValueError):
                    continue
                if filled_qty > 0:
                    break
            avg_px = None
            for key in ("average_price", "price", "avgPrice"):
                val = raw.get(key) if key != "avgPrice" else info.get("avgPrice")
                if val is None:
                    continue
                try:
                    px = float(val)
                except (TypeError, ValueError):
                    continue
                if px > 0:
                    avg_px = px
                    break
            if avg_px is None and purpose == "market_exit":
                for key in ("exit_price", "mark_price"):
                    val = raw.get(key)
                    if val is None:
                        continue
                    try:
                        px = float(val)
                    except (TypeError, ValueError):
                        continue
                    if px > 0:
                        avg_px = px
                        break
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
                    "filled_quantity": filled_qty,
                    "average_price": avg_px,
                    "raw": raw,
                }
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - persistence must not block trading
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
