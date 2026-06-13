"""
订单管理器
处理订单创建、修改、取消、状态同步
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from threading import Lock

from .models import Order, OrderSide, OrderType, OrderStatus
from .storage import Storage
from .binance_api import BinanceAPI

logger = logging.getLogger(__name__)


class OrderManager:
    """订单管理器"""

    def __init__(self, storage: Storage, binance_api: BinanceAPI, shadow: bool = False):
        """
        初始化订单管理器

        Args:
            storage: 存储层实例
            binance_api: Binance API实例
            shadow: Shadow 模式 - 只记录订单到数据库, 不实际下单
        """
        self.storage = storage
        self.binance_api = binance_api
        self.shadow = shadow
        self._lock = Lock()
        if shadow:
            logger.info("🔇 OrderManager: Shadow 模式启用 — 订单只记录不执行")

    @staticmethod
    def _binance_safe_client_order_id() -> str:
        """Binance Futures ``newClientOrderId``: short prefix + uuid, capped at 36 chars."""
        raw = (
            os.getenv("MLBOT_LIVE_CLIENT_ORDER_PREFIX", "tl").strip() or "tl"
        ).replace("-", "")
        slug = "".join(c for c in raw if str(c).isalnum())
        slug = slug[:12] if slug else "tl"
        return (f"{slug}_{uuid.uuid4().hex}")[:36]

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        position_id: Optional[str] = None,
        reduce_only: bool = False,
        close_position: bool = False,
    ) -> Order:
        """
        下单

        Args:
            symbol: 交易对符号
            side: 订单方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单需要）
            stop_price: 止损价格（止损单需要）
            position_id: 关联的仓位ID
            reduce_only: 是否只减仓
            close_position: 是否平仓

        Returns:
            订单对象
        """
        with self._lock:
            order_id = f"order_{uuid.uuid4().hex}"
            client_order_id = self._binance_safe_client_order_id()

            # Shadow 模式: 记录订单但不实际执行
            if self.shadow:
                order = Order(
                    order_id=order_id,
                    client_order_id=client_order_id,
                    position_id=position_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    status=OrderStatus.SHADOW,
                    created_at=datetime.now(),
                )
                self.storage.create_order(order)
                logger.info(
                    "🔇 trend/shadow order: internal_id=%s symbol=%s side=%s type=%s "
                    "qty=%s price=%s stop_price=%s client_order_id=%s position_id=%s "
                    "reduce_only=%s close_position=%s",
                    order_id,
                    symbol,
                    side.value,
                    order_type.value,
                    quantity,
                    price,
                    stop_price,
                    client_order_id,
                    position_id,
                    reduce_only,
                    close_position,
                )
                return order

            # 调用Binance API下单
            try:
                binance_order = self.binance_api.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    reduce_only=reduce_only,
                    close_position=close_position,
                    client_order_id=client_order_id,
                )
            except Exception as e:
                logger.error(f"Binance API下单失败: {e}")
                # 创建失败的订单记录
                order = Order(
                    order_id=order_id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    status=OrderStatus.REJECTED,
                    error_message=str(e),
                    position_id=position_id,
                    created_at=datetime.now(),
                )
                self.storage.create_order(order)
                raise

            # 创建订单对象
            # 解析订单方向
            binance_side = binance_order.get("side")
            if binance_side is None:
                # 如果ccxt没有返回side，根据我们的参数设置
                binance_side = side.value
            order_side = OrderSide(binance_side) if binance_side else side

            # 解析订单类型
            binance_type = binance_order.get("type")
            if binance_type is None:
                binance_type = order_type.value
            order_type_parsed = OrderType(binance_type) if binance_type else order_type

            order = Order(
                order_id=order_id,
                binance_order_id=str(
                    binance_order.get("order_id", binance_order.get("id", ""))
                ),
                client_order_id=binance_order.get("client_order_id", client_order_id),
                position_id=position_id,
                symbol=symbol,
                side=order_side,
                order_type=order_type_parsed,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=self._convert_order_status(
                    binance_order.get("status", "pending")
                ),
                filled_quantity=binance_order.get("filled", 0)
                or binance_order.get("filled_quantity", 0),
                created_at=datetime.now(),
            )
            # REST place 回报常不含或不完整；能拿到则一并落下，避免单靠 WS 回填
            _avg = binance_order.get("average_price")
            if _avg is not None:
                try:
                    order.average_price = float(_avg)
                except (TypeError, ValueError):
                    pass
            if order.status == OrderStatus.FILLED:
                _ts_ms = binance_order.get("timestamp")
                try:
                    if _ts_ms is not None:
                        ts = float(_ts_ms)
                        if ts > 1e12:
                            ts /= 1000.0
                        order.filled_at = datetime.fromtimestamp(ts)
                    else:
                        order.filled_at = datetime.now()
                except Exception:
                    order.filled_at = datetime.now()

            # 保存到数据库
            if self.storage.create_order(order):
                logger.info(
                    "trend/live place OK: internal_id=%s symbol=%s side=%s type=%s "
                    "qty=%s price=%s client_order_id=%s binance_order_id=%s "
                    "position_id=%s reduce_only=%s close_position=%s filled=%s",
                    order_id,
                    symbol,
                    side.value,
                    order_type.value,
                    quantity,
                    price,
                    order.client_order_id,
                    order.binance_order_id,
                    position_id,
                    reduce_only,
                    close_position,
                    order.filled_quantity,
                )
                return order
            else:
                raise Exception(f"保存订单失败: {order_id}")

    def cancel_order(self, order_id: str) -> bool:
        """
        撤单

        Args:
            order_id: 订单ID

        Returns:
            是否成功
        """
        with self._lock:
            order = self.storage.get_order(order_id)
            if not order:
                raise ValueError(f"订单不存在: {order_id}")

            logger.info(
                "trend/live cancel requested: internal_id=%s symbol=%s side=%s type=%s "
                "qty=%s client_order_id=%s binance_order_id=%s status=%s position_id=%s",
                order_id,
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.quantity,
                order.client_order_id,
                order.binance_order_id or "",
                order.status.value,
                getattr(order, "position_id", "") or "",
            )

            # Shadow 订单直接标记取消
            if order.status == OrderStatus.SHADOW or self.shadow:
                order.status = OrderStatus.CANCELED
                order.canceled_at = datetime.now()
                self.storage.update_order(order)
                logger.info(
                    "trend/live cancel shadow: internal_id=%s client_order_id=%s",
                    order_id,
                    order.client_order_id,
                )
                return True

            if order.status != OrderStatus.PENDING:
                logger.warning(f"订单状态不允许撤单: {order.status}")
                return False

            # 调用Binance API撤单
            try:
                success = self.binance_api.cancel_order(
                    order.binance_order_id or order_id, order.symbol
                )
            except Exception as e:
                logger.error(f"Binance API撤单失败: {e}")
                # 更新订单状态为拒绝
                order.status = OrderStatus.REJECTED
                order.error_message = str(e)
                order.canceled_at = datetime.now()
                self.storage.update_order(order)
                raise

            if success:
                order.status = OrderStatus.CANCELED
                order.canceled_at = datetime.now()
                self.storage.update_order(order)
                logger.info(
                    "trend/live cancel OK: internal_id=%s symbol=%s "
                    "client_order_id=%s binance_order_id=%s",
                    order_id,
                    order.symbol,
                    order.client_order_id,
                    order.binance_order_id or "",
                )
                return True
            else:
                return False

    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[str]:
        """
        撤销所有订单

        Args:
            symbol: 交易对符号，None表示撤销所有交易对的订单

        Returns:
            撤销的订单ID列表
        """
        with self._lock:
            open_orders = self.storage.get_open_orders(symbol)
            canceled_order_ids = []

            for order in open_orders:
                try:
                    if self.cancel_order(order.order_id):
                        canceled_order_ids.append(order.order_id)
                except Exception as e:
                    logger.warning(f"撤销订单 {order.order_id} 失败: {e}")

            return canceled_order_ids

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单信息"""
        return self.storage.get_order(order_id)

    def get_order_by_binance_id(self, binance_order_id: str) -> Optional[Order]:
        """通过Binance订单ID获取订单"""
        return self.storage.get_order_by_binance_id(binance_order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未完成订单"""
        return self.storage.get_open_orders(symbol)

    def sync_order_status(self, order_id: str) -> Order:
        """
        同步订单状态（从Binance API获取最新状态）

        Args:
            order_id: 订单ID

        Returns:
            更新后的订单对象
        """
        with self._lock:
            order = self.storage.get_order(order_id)
            if not order:
                raise ValueError(f"订单不存在: {order_id}")

            if not order.binance_order_id:
                logger.warning(f"订单没有Binance订单ID，无法同步: {order_id}")
                return order

            # 从Binance API获取订单状态
            try:
                binance_order = self.binance_api.get_order(
                    order.binance_order_id, order.symbol
                )
            except Exception as e:
                logger.error(f"获取Binance订单状态失败: {e}")
                return order

            if not binance_order:
                logger.warning(f"Binance订单不存在: {order.binance_order_id}")
                return order

            # 更新订单状态（REST 兜底：补齐 WS 可能漏掉的终态字段）
            self._apply_binance_order_snapshot(order, binance_order)
            order.updated_at = datetime.now()
            self.storage.update_order(order)

            logger.info(f"同步订单状态成功: {order_id}, status={order.status}")
            return order

    def sync_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        同步所有未完成订单的状态

        Args:
            symbol: 交易对符号，None表示同步所有交易对的订单

        Returns:
            更新后的订单列表
        """
        open_orders = self.storage.get_open_orders(symbol)
        updated_orders = []

        for order in open_orders:
            try:
                updated_order = self.sync_order_status(order.order_id)
                updated_orders.append(updated_order)
            except Exception as e:
                logger.warning(f"同步订单 {order.order_id} 失败: {e}")

        return updated_orders

    def reconcile_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        对账本地未完成订单与交易所未完成订单
        """
        with self._lock:
            local_open = self.storage.get_open_orders(symbol)
            exchange_open = self.binance_api.get_open_orders(symbol)

            exchange_by_id = {
                str(o.get("order_id")): o for o in exchange_open if o.get("order_id")
            }
            exchange_by_client = {
                str(o.get("client_order_id")): o
                for o in exchange_open
                if o.get("client_order_id")
            }

            updated_orders: List[Order] = []

            # 更新本地订单状态（本地存在，交易所不存在）
            for order in local_open:
                found = False
                if order.binance_order_id and order.binance_order_id in exchange_by_id:
                    found = True
                if (
                    not found
                    and order.client_order_id
                    and order.client_order_id in exchange_by_client
                ):
                    found = True

                if not found and order.binance_order_id:
                    # 获取交易所最终状态（含 client_order_id fallback 查 algo/条件单）
                    binance_order = self.binance_api.get_order(
                        order.binance_order_id,
                        order.symbol,
                        client_order_id=order.client_order_id,
                    )
                    if binance_order:
                        self._apply_binance_order_snapshot(order, binance_order)
                        order.updated_at = datetime.now()
                        self.storage.update_order(order)
                        updated_orders.append(order)

            # 交易所存在但本地不存在的订单
            for ex_order in exchange_open:
                ex_id = str(ex_order.get("order_id", ""))
                client_id = ex_order.get("client_order_id")
                existing = None
                if ex_id:
                    existing = self.storage.get_order_by_binance_id(ex_id)
                if not existing and client_id:
                    existing = self.storage.get_order_by_client_id(str(client_id))
                if existing:
                    continue

                # 创建本地记录
                new_order = Order(
                    order_id=f"binance_{ex_id}",
                    binance_order_id=ex_id or None,
                    client_order_id=str(client_id) if client_id else None,
                    symbol=ex_order.get("symbol", ""),
                    side=self._parse_order_side(ex_order.get("side")),
                    order_type=self._parse_order_type(ex_order.get("type")),
                    quantity=ex_order.get("quantity", 0),
                    price=ex_order.get("price"),
                    status=self._convert_order_status(ex_order.get("status")),
                    filled_quantity=ex_order.get("filled", 0),
                    average_price=ex_order.get("average_price"),
                    created_at=datetime.now(),
                )
                self.storage.create_order(new_order)
                updated_orders.append(new_order)

            return updated_orders

    def reconcile_recent_terminal_orders(
        self,
        *,
        symbol: Optional[str] = None,
        lookback_hours: int = 24,
        limit: int = 200,
    ) -> List[Order]:
        """Backfill terminal orders with missing fields via REST fetch_order."""
        candidates = self.storage.get_recent_orders_for_backfill(
            symbol=symbol,
            lookback_hours=int(lookback_hours),
            limit=int(limit),
        )
        stale_grace_seconds = float(
            max(
                0,
                int(os.getenv("MLBOT_TERMINAL_STALE_OPEN_GRACE_SECONDS", str(6 * 3600))),
            )
        )
        updated: List[Order] = []
        stale_marked = 0
        api_error_count = 0
        open_snapshot_cache: Dict[str, tuple[set[str], set[str]] | None] = {}

        def _open_ids_for_symbol(sym: str) -> tuple[set[str], set[str]] | None:
            nonlocal api_error_count
            key = str(sym or "").upper()
            if key in open_snapshot_cache:
                return open_snapshot_cache[key]
            by_id: set[str] = set()
            by_client: set[str] = set()
            try:
                rows = self.binance_api.get_open_orders_for_sl_cleanup(key) or []
                for row in rows:
                    oid = str(row.get("order_id") or "").strip()
                    cid = str(row.get("client_order_id") or "").strip()
                    if oid:
                        by_id.add(oid)
                    if cid:
                        by_client.add(cid)
            except Exception as exc:
                api_error_count += 1
                logger.debug("terminal backfill open-orders snapshot failed: %s", exc)
                open_snapshot_cache[key] = None
                return None
            snapshot = (by_id, by_client)
            open_snapshot_cache[key] = snapshot
            return snapshot

        for order in candidates:
            if not order.binance_order_id:
                continue
            try:
                snap = self.binance_api.get_order(order.binance_order_id, order.symbol)
            except Exception as exc:
                api_error_count += 1
                logger.debug(
                    "recent terminal backfill get_order failed: %s (%s)",
                    order.order_id,
                    exc,
                )
                continue
            if not snap:
                order_status = str(getattr(order.status, "value", order.status) or "").lower()
                if order_status in {"pending", "partially_filled"}:
                    age_seconds = (
                        (datetime.now() - (order.updated_at or order.created_at)).total_seconds()
                        if (order.updated_at or order.created_at) is not None
                        else 0.0
                    )
                    open_snapshot = _open_ids_for_symbol(order.symbol)
                    if open_snapshot is None:
                        continue
                    ex_open_ids, ex_open_clients = open_snapshot
                    in_exchange_open = (
                        str(order.binance_order_id or "").strip() in ex_open_ids
                        or str(order.client_order_id or "").strip() in ex_open_clients
                    )
                    if age_seconds >= stale_grace_seconds and not in_exchange_open:
                        order.status = OrderStatus.EXPIRED
                        order.canceled_at = order.canceled_at or datetime.now()
                        emsg = str(order.error_message or "").strip()
                        if "exchange_order_missing" not in emsg:
                            order.error_message = (
                                f"{emsg};exchange_order_missing".strip(";")
                                if emsg
                                else "exchange_order_missing"
                            )
                        order.updated_at = datetime.now()
                        self.storage.update_order(order)
                        updated.append(order)
                        stale_marked += 1
                        logger.warning(
                            "terminal backfill: mark stale local open as expired symbol=%s order_id=%s binance_order_id=%s",
                            order.symbol,
                            order.order_id,
                            order.binance_order_id,
                        )
                continue
            before = (
                order.status,
                order.filled_quantity,
                order.average_price,
                order.filled_at,
                order.canceled_at,
                order.error_message,
            )
            self._apply_binance_order_snapshot(order, snap)
            after = (
                order.status,
                order.filled_quantity,
                order.average_price,
                order.filled_at,
                order.canceled_at,
                order.error_message,
            )
            if after != before:
                order.updated_at = datetime.now()
                self.storage.update_order(order)
                updated.append(order)
        self._last_terminal_backfill_stats = {
            "updated_rows": len(updated),
            "stale_marked": stale_marked,
            "api_error": api_error_count,
        }
        return updated

    def _apply_binance_order_snapshot(
        self, order: Order, binance_order: Dict[str, Any]
    ) -> None:
        """Apply REST snapshot fields to a local order object."""
        order.status = self._convert_order_status(binance_order.get("status"))
        filled = binance_order.get("filled")
        if filled is not None:
            try:
                order.filled_quantity = float(filled)
            except (TypeError, ValueError):
                pass
        avg = binance_order.get("average_price")
        if avg is not None:
            try:
                v = float(avg)
                if v > 0:
                    order.average_price = v
            except (TypeError, ValueError):
                pass

        ts = (
            binance_order.get("update_time")
            or binance_order.get("timestamp")
            or binance_order.get("created_at")
        )
        dt: Optional[datetime] = None
        if ts is not None:
            try:
                raw = float(ts)
                if raw > 1e12:
                    raw /= 1000.0
                dt = datetime.fromtimestamp(raw)
            except (TypeError, ValueError, OSError):
                dt = None

        if order.status == OrderStatus.FILLED and order.filled_at is None:
            order.filled_at = dt or datetime.now()
        if order.status in (
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ):
            order.canceled_at = order.canceled_at or dt or datetime.now()
            err = binance_order.get("error_message") or binance_order.get("reject_reason")
            if err:
                order.error_message = str(err)

    def _convert_order_status(self, status: Optional[str]) -> OrderStatus:
        """转换订单状态"""
        if not status:
            return OrderStatus.PENDING
        status_lower = str(status).lower()
        if status_lower in ["new", "open", "pending"]:
            return OrderStatus.PENDING
        elif status_lower in ["partially_filled", "partial", "partiallyfilled"]:
            return OrderStatus.PARTIALLY_FILLED
        elif status_lower in ["filled", "closed"]:
            return OrderStatus.FILLED
        elif status_lower == "canceled":
            return OrderStatus.CANCELED
        elif status_lower == "rejected":
            return OrderStatus.REJECTED
        elif status_lower == "expired":
            return OrderStatus.EXPIRED
        else:
            logger.warning(f"未知的订单状态: {status}")
            return OrderStatus.PENDING

    def _parse_order_side(self, side: Optional[str]) -> OrderSide:
        if not side:
            return OrderSide.BUY
        side_lower = str(side).lower()
        return OrderSide.BUY if side_lower in ["buy", "long"] else OrderSide.SELL

    def _parse_order_type(self, order_type: Optional[str]) -> OrderType:
        if not order_type:
            return OrderType.MARKET
        ot = str(order_type).lower()
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_market": OrderType.STOP_MARKET,
            "take_profit": OrderType.TAKE_PROFIT,
            "take_profit_market": OrderType.TAKE_PROFIT_MARKET,
        }
        return mapping.get(ot, OrderType.MARKET)

    def handle_execution_report(self, report: Dict[str, Any]) -> Optional[Order]:
        """
        处理User Data Stream的订单回报（executionReport/ORDER_TRADE_UPDATE）
        """
        with self._lock:
            order = None
            order_id = report.get("order_id")
            client_order_id = report.get("client_order_id")

            if order_id:
                order = self.storage.get_order_by_binance_id(str(order_id))
            if not order and client_order_id:
                order = self.storage.get_order_by_client_id(str(client_order_id))

            if not order:
                # 创建本地缺失订单（对账场景）
                symbol = report.get("symbol") or ""
                _av_miss: Optional[float] = None
                for raw in (report.get("avg_price"), report.get("last_filled_price")):
                    if raw is None:
                        continue
                    try:
                        v = float(raw)
                        if v > 0:
                            _av_miss = v
                            break
                    except (TypeError, ValueError):
                        pass
                _st = self._convert_order_status(report.get("status"))
                _fa_miss: Optional[datetime] = None
                if _st == OrderStatus.FILLED:
                    ts_miss = report.get("trade_time") or report.get("event_time")
                    if ts_miss:
                        _fa_miss = datetime.fromtimestamp(int(ts_miss))
                    else:
                        _fa_miss = datetime.now()
                order = Order(
                    order_id=f"binance_{order_id}",
                    binance_order_id=str(order_id) if order_id else None,
                    client_order_id=str(client_order_id) if client_order_id else None,
                    symbol=symbol,
                    side=self._parse_order_side(report.get("side")),
                    order_type=self._parse_order_type(report.get("order_type")),
                    quantity=report.get("filled_qty", 0) or 0,
                    status=_st,
                    filled_quantity=report.get("filled_qty", 0) or 0,
                    average_price=_av_miss,
                    commission=_as_float(report.get("commission"), 0.0),
                    commission_asset=(
                        str(report["commission_asset"]).strip().upper()
                        if report.get("commission_asset")
                        else None
                    ),
                    filled_at=_fa_miss,
                    canceled_at=(
                        datetime.now()
                        if _st in (OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED)
                        else None
                    ),
                    error_message=(
                        str(report.get("error_message") or report.get("reject_reason") or "")
                        or None
                    ),
                    created_at=datetime.now(),
                )
                self.storage.create_order(order)
                return order

            # 更新订单字段
            if order_id and not order.binance_order_id:
                order.binance_order_id = str(order_id)
            if client_order_id and not order.client_order_id:
                order.client_order_id = str(client_order_id)

            order.status = self._convert_order_status(report.get("status"))
            filled_qty = report.get("filled_qty")
            if filled_qty is not None:
                order.filled_quantity = float(filled_qty)

            _avg_px: Optional[float] = None
            raw_avg = report.get("avg_price")
            if raw_avg is not None:
                try:
                    v = float(raw_avg)
                    if v > 0:
                        _avg_px = v
                except (TypeError, ValueError):
                    pass
            if _avg_px is None:
                lf = report.get("last_filled_price")
                if lf is not None:
                    try:
                        v = float(lf)
                        if v > 0 and order.status in (
                            OrderStatus.FILLED,
                            OrderStatus.PARTIALLY_FILLED,
                        ):
                            _avg_px = v
                    except (TypeError, ValueError):
                        pass
            if _avg_px is not None:
                order.average_price = _avg_px

            cass = report.get("commission_asset")
            if cass:
                order.commission_asset = str(cass).strip().upper()
            raw_comm = report.get("commission")
            if raw_comm is not None:
                try:
                    order.commission = float(order.commission or 0.0) + float(raw_comm)
                except (TypeError, ValueError):
                    pass

            rej = report.get("error_message") or report.get("reject_reason")
            if rej:
                order.error_message = str(rej)

            if order.status == OrderStatus.FILLED:
                ts = report.get("trade_time") or report.get("event_time")
                if ts:
                    order.filled_at = datetime.fromtimestamp(int(ts))
                elif order.filled_at is None:
                    order.filled_at = datetime.now()
            elif order.status in (
                OrderStatus.CANCELED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            ):
                order.canceled_at = datetime.now()

            order.updated_at = datetime.now()
            self.storage.update_order(order)
            return order


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
