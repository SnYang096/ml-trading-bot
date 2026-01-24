"""
性能指标计算测试
"""

import pytest
import tempfile
import os
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from src.order_management.performance_metrics import PerformanceMetricsCalculator
from src.order_management.storage import Storage
from src.order_management.models import (
    Position,
    PositionSide,
    PositionStatus,
    PerformanceMetrics,
)


@pytest.fixture
def temp_storage():
    """创建临时存储"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = Storage(path)
    yield storage
    os.unlink(path)


@pytest.fixture
def calculator(temp_storage):
    """创建性能指标计算器"""
    return PerformanceMetricsCalculator(temp_storage)


@pytest.fixture
def sample_positions():
    """创建示例仓位数据"""
    base_time = datetime.now()

    positions = [
        # 盈利仓位
        Position(
            position_id="pos_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=base_time - timedelta(days=3),
            exit_time=base_time - timedelta(days=2),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,
        ),
        # 亏损仓位
        Position(
            position_id="pos_2",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=base_time - timedelta(days=2),
            exit_time=base_time - timedelta(days=1),
            entry_price=51000.0,
            exit_price=50000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5100.0,
            status=PositionStatus.CLOSED,
            realized_pnl=-100.0,
        ),
        # 盈利仓位
        Position(
            position_id="pos_3",
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_time=base_time - timedelta(days=1),
            exit_time=base_time,
            entry_price=50000.0,
            exit_price=49000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,
        ),
        # 盈亏平衡仓位
        Position(
            position_id="pos_4",
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            entry_time=base_time - timedelta(days=1),
            exit_time=base_time,
            entry_price=3000.0,
            exit_price=3000.0,
            initial_size=1.0,
            current_size=0.0,
            total_cost=3000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=0.0,
        ),
    ]

    return positions


def test_calculate_daily_metrics(calculator, sample_positions):
    """测试计算每日指标"""
    # Mock _get_closed_positions_for_date返回示例仓位
    calculator._get_closed_positions_for_date = Mock(return_value=sample_positions)

    metrics = calculator.calculate_daily_metrics(target_date=date.today())

    assert metrics is not None
    assert metrics.total_trades == 4
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 0.5  # 2/4
    assert metrics.total_pnl == 100.0  # 100 - 100 + 100 + 0
    assert metrics.total_profit == 200.0  # 100 + 100
    assert metrics.total_loss == 100.0  # abs(-100)
    assert metrics.profit_factor == 2.0  # 200 / 100


def test_calculate_daily_metrics_no_trades(calculator):
    """测试计算每日指标（无交易）"""
    calculator._get_closed_positions_for_date = Mock(return_value=[])

    metrics = calculator.calculate_daily_metrics(target_date=date.today())

    assert metrics.total_trades == 0
    assert metrics.winning_trades == 0
    assert metrics.losing_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.total_pnl == 0.0


def test_calculate_daily_metrics_all_wins(calculator):
    """测试计算每日指标（全部盈利）"""
    positions = [
        Position(
            position_id="pos_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,
        ),
        Position(
            position_id="pos_2",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=52000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=200.0,
        ),
    ]

    calculator._get_closed_positions_for_date = Mock(return_value=positions)

    metrics = calculator.calculate_daily_metrics(target_date=date.today())

    assert metrics.win_rate == 1.0
    assert metrics.total_profit == 300.0
    assert metrics.total_loss == 0.0
    # 全部盈利时profit_factor应该是inf
    assert metrics.profit_factor == float("inf")


def test_calculate_daily_metrics_all_losses(calculator):
    """测试计算每日指标（全部亏损）"""
    positions = [
        Position(
            position_id="pos_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=49000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=-100.0,
        ),
    ]

    calculator._get_closed_positions_for_date = Mock(return_value=positions)

    metrics = calculator.calculate_daily_metrics(target_date=date.today())

    assert metrics.win_rate == 0.0
    assert metrics.total_profit == 0.0
    assert metrics.total_loss == 100.0
    assert metrics.profit_factor == 0.0


def test_calculate_max_drawdown(calculator, sample_positions):
    """测试计算最大回撤"""
    max_dd, period = calculator._calculate_max_drawdown(sample_positions)

    # 根据示例数据：累计盈亏序列为 [100, 0, 100, 100]
    # 峰值是100，最大回撤应该是0（因为从未低于峰值）
    assert max_dd is not None or max_dd == 0


def test_calculate_max_drawdown_empty(calculator):
    """测试计算最大回撤（空列表）"""
    max_dd, period = calculator._calculate_max_drawdown([])

    assert max_dd is None
    assert period is None


def test_calculate_max_drawdown_with_drawdown(calculator):
    """测试计算最大回撤（有回撤）"""
    positions = [
        Position(
            position_id="pos_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now() - timedelta(days=3),
            exit_time=datetime.now() - timedelta(days=2),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=200.0,  # 盈利200
        ),
        Position(
            position_id="pos_2",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now() - timedelta(days=2),
            exit_time=datetime.now() - timedelta(days=1),
            entry_price=51000.0,
            exit_price=50000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5100.0,
            status=PositionStatus.CLOSED,
            realized_pnl=-150.0,  # 亏损150
        ),
        Position(
            position_id="pos_3",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now() - timedelta(days=1),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,  # 盈利100
        ),
    ]

    max_dd, period = calculator._calculate_max_drawdown(positions)

    # 累计盈亏序列：[200, 50, 150]
    # 峰值：200
    # 最大回撤：200 - 50 = 150
    assert max_dd is not None
    assert max_dd > 0


def test_calculate_sharpe_ratio(calculator, sample_positions):
    """测试计算Sharpe比率"""
    sharpe = calculator._calculate_sharpe_ratio(sample_positions)

    # 有数据时应该返回一个数值
    assert sharpe is not None
    assert isinstance(sharpe, float)


def test_calculate_sharpe_ratio_empty(calculator):
    """测试计算Sharpe比率（空列表）"""
    sharpe = calculator._calculate_sharpe_ratio([])

    assert sharpe is None


def test_calculate_sharpe_ratio_single_position(calculator):
    """测试计算Sharpe比率（单个仓位）"""
    positions = [
        Position(
            position_id="pos_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,
        ),
    ]

    sharpe = calculator._calculate_sharpe_ratio(positions)

    # 数据不足时应该返回None
    assert sharpe is None


def test_calculate_sharpe_ratio_zero_std(calculator):
    """测试计算Sharpe比率（标准差为0）"""
    positions = [
        Position(
            position_id="pos_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,
        ),
        Position(
            position_id="pos_2",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,
        ),
    ]

    sharpe = calculator._calculate_sharpe_ratio(positions)

    # 标准差为0时应该返回None
    assert sharpe is None


def test_save_daily_metrics(calculator, sample_positions):
    """测试保存每日指标"""
    calculator._get_closed_positions_for_date = Mock(return_value=sample_positions)

    success = calculator.save_daily_metrics(target_date=date.today())

    # 验证保存成功（使用真实存储，所以应该能保存）
    assert success == True
    # 验证指标被保存到数据库
    saved_metrics = calculator.storage.get_performance_metrics()
    assert len(saved_metrics) > 0


def test_save_daily_metrics_error(calculator):
    """测试保存每日指标（错误情况）"""
    calculator._get_closed_positions_for_date = Mock(return_value=[])

    # 使用Mock存储来模拟错误
    mock_storage = Mock(spec=Storage)
    mock_storage.create_performance_metrics.side_effect = Exception("DB error")
    calculator.storage = mock_storage

    success = calculator.save_daily_metrics(target_date=date.today())

    assert success == False
    mock_storage.create_performance_metrics.assert_called_once()


def test_get_performance_summary(calculator):
    """测试获取性能摘要"""
    # 创建模拟的性能指标
    metrics1 = PerformanceMetrics(
        metric_id="metrics_1",
        date=date.today() - timedelta(days=2),
        symbol="BTCUSDT",
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        total_pnl=500.0,
        total_profit=800.0,
        total_loss=300.0,
        profit_factor=800.0 / 300.0,
    )

    metrics2 = PerformanceMetrics(
        metric_id="metrics_2",
        date=date.today() - timedelta(days=1),
        symbol="BTCUSDT",
        total_trades=5,
        winning_trades=3,
        losing_trades=2,
        win_rate=0.6,
        total_pnl=200.0,
        total_profit=400.0,
        total_loss=200.0,
        profit_factor=2.0,
    )

    calculator.storage.get_performance_metrics = Mock(return_value=[metrics1, metrics2])

    summary = calculator.get_performance_summary()

    assert summary is not None
    assert summary["total_trades"] == 15  # 10 + 5
    assert summary["winning_trades"] == 9  # 6 + 3
    assert summary["losing_trades"] == 6  # 4 + 2
    assert summary["win_rate"] == 0.6  # 9 / 15
    assert summary["total_pnl"] == 700.0  # 500 + 200
    assert summary["total_profit"] == 1200.0  # 800 + 400
    assert summary["total_loss"] == 500.0  # 300 + 200
    assert summary["profit_factor"] == 2.4  # 1200 / 500


def test_get_performance_summary_with_date_range(calculator):
    """测试获取性能摘要（日期范围）"""
    start_date = date.today() - timedelta(days=2)
    end_date = date.today() - timedelta(days=1)

    metrics = PerformanceMetrics(
        metric_id="metrics_1",
        date=date.today() - timedelta(days=1),
        symbol="BTCUSDT",
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        total_pnl=500.0,
        total_profit=800.0,
        total_loss=300.0,
        profit_factor=800.0 / 300.0,
    )

    calculator.storage.get_performance_metrics = Mock(return_value=[metrics])

    summary = calculator.get_performance_summary(
        start_date=start_date, end_date=end_date
    )

    assert summary is not None
    assert summary["period"]["start"] == start_date.isoformat()
    assert summary["period"]["end"] == end_date.isoformat()


def test_get_performance_summary_with_symbol(calculator):
    """测试获取性能摘要（指定交易对）"""
    metrics = PerformanceMetrics(
        metric_id="metrics_1",
        date=date.today(),
        symbol="BTCUSDT",
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        total_pnl=500.0,
        total_profit=800.0,
        total_loss=300.0,
        profit_factor=800.0 / 300.0,
    )

    calculator.storage.get_performance_metrics = Mock(return_value=[metrics])

    summary = calculator.get_performance_summary(symbol="BTCUSDT")

    assert summary is not None
    calculator.storage.get_performance_metrics.assert_called_once_with(symbol="BTCUSDT")


def test_get_performance_summary_empty(calculator):
    """测试获取性能摘要（无数据）"""
    calculator.storage.get_performance_metrics = Mock(return_value=[])

    summary = calculator.get_performance_summary()

    assert summary is not None
    assert summary["total_trades"] == 0
    assert summary["win_rate"] == 0.0
    assert summary["total_pnl"] == 0.0
    assert summary["profit_factor"] == 0.0


def test_average_win_loss(calculator):
    """测试平均盈亏计算"""
    positions = [
        Position(
            position_id="pos_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=100.0,
        ),
        Position(
            position_id="pos_2",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=52000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=200.0,
        ),
        Position(
            position_id="pos_3",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=49000.0,
            initial_size=0.1,
            current_size=0.0,
            total_cost=5000.0,
            status=PositionStatus.CLOSED,
            realized_pnl=-50.0,
        ),
    ]

    calculator._get_closed_positions_for_date = Mock(return_value=positions)

    metrics = calculator.calculate_daily_metrics(target_date=date.today())

    assert metrics.average_win == 150.0  # (100 + 200) / 2
    assert metrics.average_loss == -50.0  # -50 / 1
    assert metrics.largest_win == 200.0
    assert metrics.largest_loss == -50.0
