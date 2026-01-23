"""
仓位管理器测试
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.order_management.storage import Storage
from src.order_management.binance_api import BinanceAPI
from src.order_management.position_manager import PositionManager
from src.order_management.models import PositionSide


@pytest.fixture
def mock_binance_api():
    """创建模拟的Binance API"""
    api = Mock(spec=BinanceAPI)
    api.get_position.return_value = {"mark_price": 51000.0, "size": 0.1, "leverage": 1}
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
def position_manager(temp_storage, mock_binance_api):
    """创建仓位管理器"""
    return PositionManager(temp_storage, mock_binance_api)


def test_create_position(position_manager):
    """测试创建仓位"""
    position = position_manager.create_position(
        symbol="BTCUSDT", side=PositionSide.LONG, entry_price=50000.0, size=0.1
    )

    assert position is not None
    assert position.symbol == "BTCUSDT"
    assert position.side == PositionSide.LONG
    assert position.entry_price == 50000.0
    assert position.current_size == 0.1


def test_add_to_position(position_manager):
    """测试加仓"""
    # 先创建仓位
    position = position_manager.create_position(
        symbol="BTCUSDT", side=PositionSide.LONG, entry_price=50000.0, size=0.1
    )

    # 加仓
    updated = position_manager.add_to_position(
        position.position_id, size=0.1, price=51000.0
    )

    assert updated.current_size == 0.2
    # 平均成本价应该更新
    assert updated.entry_price > 50000.0
    assert updated.entry_price < 51000.0


def test_reduce_position(position_manager):
    """测试减仓"""
    # 先创建仓位
    position = position_manager.create_position(
        symbol="BTCUSDT", side=PositionSide.LONG, entry_price=50000.0, size=0.2
    )

    # 减仓
    updated = position_manager.reduce_position(
        position.position_id, size=0.1, price=51000.0
    )

    assert updated.current_size == 0.1
    assert updated.realized_pnl > 0  # 应该有盈利


def test_close_position(position_manager):
    """测试平仓"""
    # 先创建仓位
    position = position_manager.create_position(
        symbol="BTCUSDT", side=PositionSide.LONG, entry_price=50000.0, size=0.1
    )

    # 平仓
    updated = position_manager.close_position(position.position_id, price=51000.0)

    assert updated.current_size == 0.0
    assert updated.status.value == "closed"
    assert updated.exit_time is not None


def test_update_stop_loss(position_manager):
    """测试更新止损"""
    # 先创建仓位
    position = position_manager.create_position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=50000.0,
        size=0.1,
        stop_loss_price=49000.0,
    )

    # 更新止损
    updated = position_manager.update_stop_loss(
        position.position_id, stop_loss_price=49500.0
    )

    assert updated.stop_loss_price == 49500.0
