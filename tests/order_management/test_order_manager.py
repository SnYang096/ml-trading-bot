"""
订单管理器测试
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.order_management.storage import Storage
from src.order_management.binance_api import BinanceAPI
from src.order_management.order_manager import OrderManager
from src.order_management.models import OrderSide, OrderType, OrderStatus


@pytest.fixture
def mock_binance_api():
    """创建模拟的Binance API"""
    api = Mock(spec=BinanceAPI)
    api.place_order.return_value = {
        "order_id": "binance_order_123",
        "client_order_id": "client_123",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "status": "new",
        "amount": 0.1,
        "filled": 0,
    }
    api.get_order.return_value = {
        "order_id": "binance_order_123",
        "client_order_id": "client_123",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "status": "filled",
        "amount": 0.1,
        "filled": 0.1,
        "average_price": 50000.0,
    }
    api.get_open_orders.return_value = []
    api.cancel_order.return_value = True
    return api


@pytest.fixture
def temp_storage():
    """创建临时存储"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = Storage(path)
    yield storage
    os.unlink(path)


@pytest.fixture
def order_manager(temp_storage, mock_binance_api):
    """创建订单管理器"""
    return OrderManager(temp_storage, mock_binance_api)


def test_place_order(order_manager):
    """测试下单"""
    order = order_manager.place_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1
    )

    assert order is not None
    assert order.symbol == "BTCUSDT"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.MARKET
    assert order.binance_order_id == "binance_order_123"
    assert order.client_order_id == "client_123"


def test_cancel_order(order_manager):
    """测试撤单"""
    # 先下单
    order = order_manager.place_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1
    )

    # 撤单
    success = order_manager.cancel_order(order.order_id)
    assert success == True

    # 验证订单状态
    updated = order_manager.get_order(order.order_id)
    assert updated.status == OrderStatus.CANCELED


def test_sync_order_status(order_manager):
    """测试同步订单状态"""
    # 先下单
    order = order_manager.place_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1
    )

    # 同步状态
    updated = order_manager.sync_order_status(order.order_id)
    assert updated.status == OrderStatus.FILLED
    assert updated.filled_quantity == 0.1


def test_handle_execution_report_updates_order(order_manager, mock_binance_api):
    """测试executionReport更新订单"""
    order = order_manager.place_order(
        symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1
    )
    report = {
        "order_id": "binance_order_123",
        "client_order_id": order.client_order_id,
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "MARKET",
        "status": "PARTIALLY_FILLED",
        "filled_qty": 0.05,
        "avg_price": 50010.0,
        "event_time": int(datetime.now().timestamp()),
    }
    updated = order_manager.handle_execution_report(report)
    assert updated is not None
    assert updated.status == OrderStatus.PARTIALLY_FILLED
    assert updated.filled_quantity == 0.05
    assert updated.average_price == 50010.0


def test_reconcile_open_orders_creates_missing(order_manager, mock_binance_api):
    """测试对账创建本地缺失订单"""
    mock_binance_api.get_open_orders.return_value = [
        {
            "order_id": "binance_order_missing",
            "client_order_id": "client_missing",
            "symbol": "BTCUSDT",
            "side": "buy",
            "type": "limit",
            "status": "new",
            "quantity": 0.1,
            "price": 50000.0,
            "filled": 0,
            "average_price": None,
        }
    ]
    updated = order_manager.reconcile_open_orders()
    assert len(updated) == 1
    created = order_manager.get_order(updated[0].order_id)
    assert created is not None
    assert created.binance_order_id == "binance_order_missing"


def test_reconcile_open_orders_syncs_canceled_pending(order_manager, mock_binance_api):
    """Local pending + exchange canceled (not in openOrders) must update SQLite."""
    from src.order_management.models import Order

    mock_binance_api.get_open_orders.return_value = []
    mock_binance_api.get_order.return_value = {
        "order_id": "4000001325799046",
        "client_order_id": "tl_test",
        "symbol": "ETHUSDT",
        "side": "buy",
        "type": "limit",
        "status": "canceled",
        "quantity": 0.05,
        "filled": 0,
        "average_price": None,
    }
    order = Order(
        order_id="local_stale_pending",
        binance_order_id="4000001325799046",
        client_order_id="tl_test",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.05,
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
    )
    order_manager.storage.create_order(order)

    updated = order_manager.reconcile_open_orders()
    assert len(updated) == 1
    got = order_manager.get_order("local_stale_pending")
    assert got is not None
    assert got.status == OrderStatus.CANCELED
