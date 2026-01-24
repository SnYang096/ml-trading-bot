"""
订单管理器
处理订单创建、修改、取消、状态同步
"""

import logging
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

    def __init__(self, storage: Storage, binance_api: BinanceAPI):
        """
        初始化订单管理器

        Args:
            storage: 存储层实例
            binance_api: Binance API实例
        """
        self.storage = storage
        self.binance_api = binance_api
        self._lock = Lock()

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
                )
            except Exception as e:
                logger.error(f"Binance API下单失败: {e}")
                # 创建失败的订单记录
                order = Order(
                    order_id=order_id,
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

            # 保存到数据库
            if self.storage.create_order(order):
                logger.info(
                    f"下单成功: {order_id}, {symbol}, {side.value}, {order_type.value}"
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
                logger.info(f"撤单成功: {order_id}")
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

            # 更新订单状态
            order.status = self._convert_order_status(binance_order["status"])
            order.filled_quantity = binance_order.get("filled", 0)
            order.average_price = binance_order.get("average_price")

            if order.status == OrderStatus.FILLED:
                order.filled_at = datetime.fromtimestamp(
                    binance_order.get("created_at", datetime.now().timestamp())
                )

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

    def _convert_order_status(self, status: Optional[str]) -> OrderStatus:
        """转换订单状态"""
        if not status:
            return OrderStatus.PENDING
        status_lower = str(status).lower()
        if status_lower in ["new", "pending"]:
            return OrderStatus.PENDING
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
