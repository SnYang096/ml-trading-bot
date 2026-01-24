"""
数据模型测试
"""

import pytest
from datetime import datetime

from src.order_management.models import (
    PositionSide,
    PositionStatus,
    OrderSide,
    OrderType,
    OrderStatus,
    OperationType,
    Position,
    Order,
    PositionOperation,
    StopLossTrailing,
    PerformanceMetrics,
)


def test_position_side_enum():
    """测试PositionSide枚举"""
    assert PositionSide.LONG == "long"
    assert PositionSide.SHORT == "short"
    assert PositionSide.LONG.value == "long"
    assert PositionSide.SHORT.value == "short"


def test_position_status_enum():
    """测试PositionStatus枚举"""
    assert PositionStatus.OPEN == "open"
    assert PositionStatus.CLOSED == "closed"
    assert PositionStatus.PARTIAL == "partial"
    assert PositionStatus.OPEN.value == "open"
    assert PositionStatus.CLOSED.value == "closed"
    assert PositionStatus.PARTIAL.value == "partial"


def test_order_side_enum():
    """测试OrderSide枚举"""
    assert OrderSide.BUY == "buy"
    assert OrderSide.SELL == "sell"
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_order_type_enum():
    """测试OrderType枚举"""
    assert OrderType.MARKET == "market"
    assert OrderType.LIMIT == "limit"
    assert OrderType.STOP == "stop"
    assert OrderType.STOP_MARKET == "stop_market"
    assert OrderType.TAKE_PROFIT == "take_profit"
    assert OrderType.TAKE_PROFIT_MARKET == "take_profit_market"


def test_order_status_enum():
    """测试OrderStatus枚举"""
    assert OrderStatus.PENDING == "pending"
    assert OrderStatus.FILLED == "filled"
    assert OrderStatus.CANCELED == "canceled"
    assert OrderStatus.REJECTED == "rejected"
    assert OrderStatus.EXPIRED == "expired"


def test_operation_type_enum():
    """测试OperationType枚举"""
    assert OperationType.ADD == "add"
    assert OperationType.REDUCE == "reduce"
    assert OperationType.STOP_LOSS_MOVE == "stop_loss_move"
    assert OperationType.TAKE_PROFIT_MOVE == "take_profit_move"


def test_position_creation():
    """测试Position创建"""
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

    assert position.position_id == "test_pos_1"
    assert position.symbol == "BTCUSDT"
    assert position.side == PositionSide.LONG
    assert position.entry_price == 50000.0
    assert position.initial_size == 0.1
    assert position.current_size == 0.1
    assert position.status == PositionStatus.OPEN


def test_position_defaults():
    """测试Position默认值"""
    position = Position(
        position_id="test_pos_1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(),
    )

    assert position.exit_time is None
    assert position.entry_price == 0.0
    assert position.exit_price is None
    assert position.initial_size == 0.0
    assert position.current_size == 0.0
    assert position.total_cost == 0.0
    assert position.status == PositionStatus.OPEN
    assert position.stop_loss_price is None
    assert position.take_profit_price is None


def test_position_to_dict():
    """测试Position转换为字典"""
    position = Position(
        position_id="test_pos_1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(),
        entry_price=50000.0,
        status=PositionStatus.OPEN,
    )

    result = position.to_dict()

    assert isinstance(result, dict)
    assert result["position_id"] == "test_pos_1"
    assert result["symbol"] == "BTCUSDT"
    assert result["side"] == "long"  # 枚举值转换为字符串
    assert result["status"] == "open"
    assert "entry_time" in result
    assert isinstance(result["entry_time"], str)  # datetime转换为ISO格式字符串


def test_order_creation():
    """测试Order创建"""
    order = Order(
        order_id="test_order_1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        status=OrderStatus.PENDING,
    )

    assert order.order_id == "test_order_1"
    assert order.symbol == "BTCUSDT"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.MARKET
    assert order.quantity == 0.1
    assert order.status == OrderStatus.PENDING


def test_order_defaults():
    """测试Order默认值"""
    order = Order(
        order_id="test_order_1",
    )

    assert order.binance_order_id is None
    assert order.position_id is None
    assert order.symbol == ""
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.MARKET
    assert order.quantity == 0.0
    assert order.price is None
    assert order.stop_price is None
    assert order.status == OrderStatus.PENDING
    assert order.filled_quantity == 0.0
    assert order.commission == 0.0


def test_order_to_dict():
    """测试Order转换为字典"""
    order = Order(
        order_id="test_order_1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
    )

    result = order.to_dict()

    assert isinstance(result, dict)
    assert result["order_id"] == "test_order_1"
    assert result["side"] == "buy"
    assert result["order_type"] == "market"
    assert result["status"] == "pending"
    assert "created_at" in result
    assert isinstance(result["created_at"], str)


def test_position_operation_creation():
    """测试PositionOperation创建"""
    operation = PositionOperation(
        operation_id="op_1",
        position_id="pos_1",
        operation_type=OperationType.ADD,
        operation_time=datetime.now(),
        size=0.1,
        price=50000.0,
    )

    assert operation.operation_id == "op_1"
    assert operation.position_id == "pos_1"
    assert operation.operation_type == OperationType.ADD
    assert operation.size == 0.1
    assert operation.price == 50000.0


def test_position_operation_defaults():
    """测试PositionOperation默认值"""
    operation = PositionOperation(
        operation_id="op_1",
        position_id="pos_1",
        operation_type=OperationType.ADD,
        operation_time=datetime.now(),
    )

    assert operation.size == 0.0
    assert operation.price == 0.0
    assert operation.pnl is None
    assert operation.cumulative_pnl is None
    assert operation.stop_loss_price is None
    assert operation.take_profit_price is None
    assert operation.reason is None
    assert operation.order_id is None


def test_position_operation_to_dict():
    """测试PositionOperation转换为字典"""
    operation = PositionOperation(
        operation_id="op_1",
        position_id="pos_1",
        operation_type=OperationType.ADD,
        operation_time=datetime.now(),
    )

    result = operation.to_dict()

    assert isinstance(result, dict)
    assert result["operation_type"] == "add"
    assert "operation_time" in result
    assert isinstance(result["operation_time"], str)


def test_stop_loss_trailing_creation():
    """测试StopLossTrailing创建"""
    trailing = StopLossTrailing(
        record_id="trail_1",
        position_id="pos_1",
        old_stop_loss=49000.0,
        new_stop_loss=49500.0,
        move_time=datetime.now(),
        current_price=51000.0,
    )

    assert trailing.record_id == "trail_1"
    assert trailing.position_id == "pos_1"
    assert trailing.old_stop_loss == 49000.0
    assert trailing.new_stop_loss == 49500.0
    assert trailing.current_price == 51000.0


def test_stop_loss_trailing_defaults():
    """测试StopLossTrailing默认值"""
    trailing = StopLossTrailing(
        record_id="trail_1",
        position_id="pos_1",
        old_stop_loss=49000.0,
        new_stop_loss=49500.0,
        move_time=datetime.now(),
        current_price=51000.0,
    )

    assert trailing.profit_protected is None
    assert trailing.reason is None
    assert trailing.created_at is None


def test_stop_loss_trailing_to_dict():
    """测试StopLossTrailing转换为字典"""
    trailing = StopLossTrailing(
        record_id="trail_1",
        position_id="pos_1",
        old_stop_loss=49000.0,
        new_stop_loss=49500.0,
        move_time=datetime.now(),
        current_price=51000.0,
    )

    result = trailing.to_dict()

    assert isinstance(result, dict)
    assert result["record_id"] == "trail_1"
    assert "move_time" in result
    assert isinstance(result["move_time"], str)


def test_performance_metrics_creation():
    """测试PerformanceMetrics创建"""
    metrics = PerformanceMetrics(
        metric_id="metrics_1",
        date=datetime.now(),
        symbol="BTCUSDT",
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        total_pnl=500.0,
    )

    assert metrics.metric_id == "metrics_1"
    assert metrics.symbol == "BTCUSDT"
    assert metrics.total_trades == 10
    assert metrics.winning_trades == 6
    assert metrics.losing_trades == 4
    assert metrics.win_rate == 0.6
    assert metrics.total_pnl == 500.0


def test_performance_metrics_defaults():
    """测试PerformanceMetrics默认值"""
    metrics = PerformanceMetrics(
        metric_id="metrics_1",
        date=datetime.now(),
    )

    assert metrics.symbol is None
    assert metrics.total_trades == 0
    assert metrics.winning_trades == 0
    assert metrics.losing_trades == 0
    assert metrics.win_rate is None
    assert metrics.total_pnl == 0.0
    assert metrics.total_profit == 0.0
    assert metrics.total_loss == 0.0
    assert metrics.profit_factor is None
    assert metrics.max_drawdown is None
    assert metrics.sharpe_ratio is None


def test_performance_metrics_to_dict():
    """测试PerformanceMetrics转换为字典"""
    metrics = PerformanceMetrics(
        metric_id="metrics_1",
        date=datetime.now(),
        symbol="BTCUSDT",
    )

    result = metrics.to_dict()

    assert isinstance(result, dict)
    assert result["metric_id"] == "metrics_1"
    assert result["symbol"] == "BTCUSDT"
    assert "date" in result
    assert isinstance(result["date"], str)


def test_position_with_optional_fields():
    """测试Position可选字段"""
    position = Position(
        position_id="test_pos_1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(),
        exit_time=datetime.now(),
        entry_price=50000.0,
        exit_price=51000.0,
        stop_loss_price=49000.0,
        take_profit_price=52000.0,
        unrealized_pnl=100.0,
        realized_pnl=50.0,
        strategy_id="strategy_1",
        notes="Test notes",
    )

    assert position.exit_time is not None
    assert position.exit_price == 51000.0
    assert position.stop_loss_price == 49000.0
    assert position.take_profit_price == 52000.0
    assert position.unrealized_pnl == 100.0
    assert position.realized_pnl == 50.0
    assert position.strategy_id == "strategy_1"
    assert position.notes == "Test notes"


def test_order_with_optional_fields():
    """测试Order可选字段"""
    order = Order(
        order_id="test_order_1",
        binance_order_id="binance_123",
        position_id="pos_1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=50000.0,
        stop_price=49000.0,
        status=OrderStatus.FILLED,
        filled_quantity=0.1,
        average_price=50000.0,
        commission=0.001,
        commission_asset="USDT",
        created_at=datetime.now(),
        filled_at=datetime.now(),
    )

    assert order.binance_order_id == "binance_123"
    assert order.position_id == "pos_1"
    assert order.price == 50000.0
    assert order.stop_price == 49000.0
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 0.1
    assert order.average_price == 50000.0
    assert order.commission == 0.001
    assert order.commission_asset == "USDT"
    assert order.filled_at is not None


def test_enum_comparison():
    """测试枚举值比较"""
    # 字符串枚举可以直接与字符串比较
    assert PositionSide.LONG == "long"
    assert OrderSide.BUY == "buy"
    assert OrderType.MARKET == "market"
    assert OrderStatus.PENDING == "pending"

    # 也可以使用value属性
    assert PositionSide.LONG.value == "long"
    assert OrderSide.BUY.value == "buy"


def test_enum_from_string():
    """测试从字符串创建枚举"""
    # 可以直接使用字符串值
    position_side = PositionSide("long")
    assert position_side == PositionSide.LONG

    order_side = OrderSide("buy")
    assert order_side == OrderSide.BUY

    order_type = OrderType("market")
    assert order_type == OrderType.MARKET


def test_enum_invalid_value():
    """测试无效枚举值"""
    with pytest.raises(ValueError):
        PositionSide("invalid")

    with pytest.raises(ValueError):
        OrderSide("invalid")

    with pytest.raises(ValueError):
        OrderType("invalid")
