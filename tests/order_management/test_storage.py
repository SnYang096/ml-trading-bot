"""
存储层测试
"""

import pytest
import tempfile
import os
from datetime import datetime

from src.order_management.storage import Storage
from src.order_management.models import (
    Position,
    PositionSide,
    PositionStatus,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    PositionOperation,
    OperationType,
)


@pytest.fixture
def temp_db():
    """创建临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = Storage(path)
    yield storage, path
    os.unlink(path)


def test_create_position(temp_db):
    """测试创建仓位"""
    storage, path = temp_db[0], temp_db[1]

    position = Position(
        position_id="test_pos_1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(),
        entry_price=50000.0,
        initial_size=0.1,
        current_size=0.1,
        total_cost=5000.0,
        status=PositionStatus.OPEN,
    )

    assert storage.create_position(position) == True
    retrieved = storage.get_position("test_pos_1")
    assert retrieved is not None
    assert retrieved.symbol == "BTCUSDT"
    assert retrieved.side == PositionSide.LONG


def test_create_order(temp_db):
    """测试创建订单"""
    storage, path = temp_db[0], temp_db[1]

    order = Order(
        order_id="test_order_1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
    )

    assert storage.create_order(order) == True
    retrieved = storage.get_order("test_order_1")
    assert retrieved is not None
    assert retrieved.symbol == "BTCUSDT"
    assert retrieved.status == OrderStatus.PENDING


def test_get_open_positions(temp_db):
    """测试获取开仓"""
    storage, path = temp_db[0], temp_db[1]

    # 创建多个仓位
    for i in range(3):
        position = Position(
            position_id=f"test_pos_{i}",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            entry_price=50000.0,
            initial_size=0.1,
            current_size=0.1,
            total_cost=5000.0,
            status=PositionStatus.OPEN,
        )
        storage.create_position(position)

    # 创建一个已平仓仓位
    closed_position = Position(
        position_id="test_pos_closed",
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(),
        exit_time=datetime.now(),
        entry_price=3000.0,
        exit_price=3100.0,
        initial_size=1.0,
        current_size=0.0,
        total_cost=3000.0,
        status=PositionStatus.CLOSED,
    )
    storage.create_position(closed_position)

    open_positions = storage.get_open_positions()
    assert len(open_positions) == 3

    # 测试按symbol过滤
    btc_positions = storage.get_open_positions("BTCUSDT")
    assert len(btc_positions) == 3


def test_update_position(temp_db):
    """测试更新仓位"""
    storage, path = temp_db[0], temp_db[1]

    position = Position(
        position_id="test_pos_1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(),
        entry_price=50000.0,
        initial_size=0.1,
        current_size=0.1,
        total_cost=5000.0,
        status=PositionStatus.OPEN,
    )
    storage.create_position(position)

    # 更新仓位
    position.current_size = 0.2
    position.unrealized_pnl = 100.0
    assert storage.update_position(position) == True

    retrieved = storage.get_position("test_pos_1")
    assert retrieved.current_size == 0.2
    assert retrieved.unrealized_pnl == 100.0
