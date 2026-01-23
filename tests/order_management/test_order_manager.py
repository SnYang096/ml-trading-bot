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
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "status": "new",
        "amount": 0.1,
        "filled": 0,
    }
    api.get_order.return_value = {
        "order_id": "binance_order_123",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "status": "filled",
        "amount": 0.1,
        "filled": 0.1,
        "average_price": 50000.0,
    }
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
