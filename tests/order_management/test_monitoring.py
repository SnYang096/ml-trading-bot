"""
监控服务测试
"""

import pytest
import sys
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Mock prometheus_client before importing monitoring
mock_prometheus = MagicMock()
mock_prometheus.Counter = Mock
mock_prometheus.Gauge = Mock
mock_prometheus.Histogram = Mock
mock_prometheus.Summary = Mock
mock_prometheus.start_http_server = Mock

sys.modules["prometheus_client"] = mock_prometheus

from src.order_management.monitoring import MonitoringService
from src.order_management.storage import Storage
from src.order_management.position_manager import PositionManager
from src.order_management.order_manager import OrderManager
from src.order_management.binance_api import BinanceAPI
from src.order_management.models import (
    Position,
    PositionSide,
    PositionStatus,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
)


@pytest.fixture
def mock_storage():
    """创建模拟的存储层"""
    return Mock(spec=Storage)


@pytest.fixture
def mock_binance_api():
    """创建模拟的Binance API"""
    api = Mock(spec=BinanceAPI)
    api.get_account_info.return_value = {
        "total_balance": 10000.0,
        "free_balance": 5000.0,
        "used_balance": 5000.0,
    }
    api.get_position.return_value = {
        "symbol": "BTCUSDT",
        "mark_price": 51000.0,
        "size": 0.1,
    }
    return api


@pytest.fixture
def mock_position_manager():
    """创建模拟的仓位管理器"""
    manager = Mock(spec=PositionManager)

    # 创建模拟仓位
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
        unrealized_pnl=100.0,
        total_value=5100.0,
    )

    manager.get_open_positions.return_value = [position]
    return manager


@pytest.fixture
def mock_order_manager():
    """创建模拟的订单管理器"""
    manager = Mock(spec=OrderManager)

    # 创建模拟订单
    order = Order(
        order_id="test_order_1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        status=OrderStatus.PENDING,
    )

    manager.get_open_orders.return_value = [order]
    manager.sync_all_orders.return_value = [order]
    return manager


@pytest.fixture
def monitoring_service(
    mock_storage, mock_position_manager, mock_order_manager, mock_binance_api
):
    """创建监控服务实例"""
    return MonitoringService(
        storage=mock_storage,
        position_manager=mock_position_manager,
        order_manager=mock_order_manager,
        binance_api=mock_binance_api,
        update_interval=1,  # 使用较短的更新间隔以便测试
    )


def test_init(monitoring_service):
    """测试初始化"""
    assert monitoring_service.storage is not None
    assert monitoring_service.position_manager is not None
    assert monitoring_service.order_manager is not None
    assert monitoring_service.binance_api is not None
    assert monitoring_service.update_interval == 1
    assert monitoring_service._running == False


def test_start(monitoring_service):
    """测试启动监控服务"""
    monitoring_service.start()

    assert monitoring_service._running == True
    assert monitoring_service._monitor_thread is not None
    assert monitoring_service._monitor_thread.is_alive()

    # 清理
    monitoring_service.stop()


def test_start_already_running(monitoring_service):
    """测试重复启动"""
    monitoring_service.start()

    # 再次启动应该不会创建新线程
    initial_thread = monitoring_service._monitor_thread
    monitoring_service.start()

    assert monitoring_service._monitor_thread == initial_thread

    # 清理
    monitoring_service.stop()


def test_stop(monitoring_service):
    """测试停止监控服务"""
    monitoring_service.start()
    assert monitoring_service._running == True

    monitoring_service.stop()

    assert monitoring_service._running == False
    # 等待线程结束
    if monitoring_service._monitor_thread:
        monitoring_service._monitor_thread.join(timeout=2)


def test_register_alert_callback(monitoring_service):
    """测试注册告警回调"""
    callback = Mock()

    monitoring_service.register_alert_callback(callback)

    assert callback in monitoring_service._alert_callbacks


def test_trigger_alert(monitoring_service):
    """测试触发告警"""
    callback = Mock()
    monitoring_service.register_alert_callback(callback)

    monitoring_service._trigger_alert("test_alert", "Test message")

    callback.assert_called_once_with("test_alert", "Test message")


def test_trigger_alert_callback_error(monitoring_service):
    """测试告警回调错误处理"""
    callback = Mock(side_effect=Exception("Callback error"))
    monitoring_service.register_alert_callback(callback)

    # 不应该抛出异常
    monitoring_service._trigger_alert("test_alert", "Test message")

    callback.assert_called_once()


@patch("src.order_management.monitoring.metrics")
def test_update_metrics(mock_metrics, monitoring_service):
    """测试更新指标"""
    monitoring_service._update_metrics()

    # 验证仓位指标更新
    mock_metrics.position_count.labels.assert_called()
    mock_metrics.position_unrealized_pnl.labels.assert_called()
    mock_metrics.position_total_value.labels.assert_called()

    # 验证风险指标更新
    mock_metrics.margin_usage_ratio.set.assert_called()
    mock_metrics.daily_pnl.set.assert_called()


@patch("src.order_management.monitoring.metrics")
def test_update_order_metrics(mock_metrics, monitoring_service):
    """测试更新订单指标"""
    monitoring_service._update_order_metrics()

    # 验证订单同步被调用
    monitoring_service.order_manager.sync_all_orders.assert_called_once()

    # 验证订单指标更新
    mock_metrics.orders_total.labels.assert_called()


def test_check_stop_loss_alerts_long(monitoring_service):
    """测试检查止损告警（多仓）"""
    # 创建带止损的多仓
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
        stop_loss_price=49000.0,
    )

    monitoring_service.position_manager.get_open_positions.return_value = [position]

    # 当前价格低于止损价
    monitoring_service.binance_api.get_position.return_value = {
        "mark_price": 48000.0,
    }

    with patch.object(monitoring_service, "_trigger_alert") as mock_alert:
        monitoring_service._check_stop_loss_alerts()
        mock_alert.assert_called_once()


def test_check_stop_loss_alerts_short(monitoring_service):
    """测试检查止损告警（空仓）"""
    # 创建带止损的空仓
    position = Position(
        position_id="test_pos_1",
        symbol="BTCUSDT",
        side=PositionSide.SHORT,
        entry_time=datetime.now(),
        entry_price=50000.0,
        initial_size=0.1,
        current_size=0.1,
        total_cost=5000.0,
        status=PositionStatus.OPEN,
        stop_loss_price=51000.0,
    )

    monitoring_service.position_manager.get_open_positions.return_value = [position]

    # 当前价格高于止损价
    monitoring_service.binance_api.get_position.return_value = {
        "mark_price": 52000.0,
    }

    with patch.object(monitoring_service, "_trigger_alert") as mock_alert:
        monitoring_service._check_stop_loss_alerts()
        mock_alert.assert_called_once()


def test_check_stop_loss_alerts_no_trigger(monitoring_service):
    """测试检查止损告警（未触发）"""
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
        stop_loss_price=49000.0,
    )

    monitoring_service.position_manager.get_open_positions.return_value = [position]

    # 当前价格高于止损价
    monitoring_service.binance_api.get_position.return_value = {
        "mark_price": 50000.0,
    }

    with patch.object(monitoring_service, "_trigger_alert") as mock_alert:
        monitoring_service._check_stop_loss_alerts()
        mock_alert.assert_not_called()


def test_check_margin_alerts_high_usage(monitoring_service):
    """测试检查保证金告警（使用率过高）"""
    # 设置保证金使用率超过80%
    monitoring_service.binance_api.get_account_info.return_value = {
        "total_balance": 10000.0,
        "free_balance": 1000.0,
        "used_balance": 9000.0,
    }

    with patch.object(monitoring_service, "_trigger_alert") as mock_alert:
        monitoring_service._check_margin_alerts()
        mock_alert.assert_called()
        # 验证告警类型
        call_args = mock_alert.call_args[0]
        assert call_args[0] == "high_margin_usage"


def test_check_margin_alerts_low_balance(monitoring_service):
    """测试检查保证金告警（可用余额不足）"""
    # 设置可用余额小于100
    monitoring_service.binance_api.get_account_info.return_value = {
        "total_balance": 10000.0,
        "free_balance": 50.0,
        "used_balance": 9950.0,
    }

    with patch.object(monitoring_service, "_trigger_alert") as mock_alert:
        monitoring_service._check_margin_alerts()
        mock_alert.assert_called()
        # 验证告警类型
        call_args = mock_alert.call_args[0]
        assert call_args[0] == "low_available_balance"


def test_check_margin_alerts_normal(monitoring_service):
    """测试检查保证金告警（正常情况）"""
    # 设置正常的保证金使用率
    monitoring_service.binance_api.get_account_info.return_value = {
        "total_balance": 10000.0,
        "free_balance": 7000.0,
        "used_balance": 3000.0,
    }

    with patch.object(monitoring_service, "_trigger_alert") as mock_alert:
        monitoring_service._check_margin_alerts()
        # 不应该触发告警
        mock_alert.assert_not_called()


def test_check_alerts(monitoring_service):
    """测试检查告警（综合）"""
    with patch.object(monitoring_service, "_check_stop_loss_alerts") as mock_stop_loss:
        with patch.object(monitoring_service, "_check_margin_alerts") as mock_margin:
            monitoring_service._check_alerts()

            mock_stop_loss.assert_called_once()
            mock_margin.assert_called_once()


def test_get_monitoring_summary(monitoring_service):
    """测试获取监控摘要"""
    summary = monitoring_service.get_monitoring_summary()

    assert summary is not None
    assert "timestamp" in summary
    assert "positions" in summary
    assert "orders" in summary
    assert "account" in summary

    assert summary["positions"]["count"] == 1
    assert summary["account"]["total_balance"] == 10000.0


def test_get_monitoring_summary_error(monitoring_service):
    """测试获取监控摘要（错误情况）"""
    monitoring_service.binance_api.get_account_info.side_effect = Exception("API error")

    summary = monitoring_service.get_monitoring_summary()

    assert "error" in summary


@patch("time.sleep")
def test_monitor_loop(mock_sleep, monitoring_service):
    """测试监控循环"""
    monitoring_service._running = True

    # 设置sleep只执行一次就退出
    call_count = 0

    def sleep_side_effect(*args):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            monitoring_service._running = False

    mock_sleep.side_effect = sleep_side_effect

    with patch.object(monitoring_service, "_update_metrics") as mock_update:
        with patch.object(monitoring_service, "_check_alerts") as mock_check:
            monitoring_service._monitor_loop()

            # 验证update_metrics和check_alerts被调用
            assert mock_update.call_count >= 1
            assert mock_check.call_count >= 1


def test_monitor_loop_error_handling(monitoring_service):
    """测试监控循环错误处理"""
    monitoring_service._running = True

    # 模拟_update_metrics抛出异常
    with patch.object(
        monitoring_service, "_update_metrics", side_effect=Exception("Test error")
    ):
        with patch("time.sleep") as mock_sleep:

            def sleep_side_effect(*args):
                monitoring_service._running = False

            mock_sleep.side_effect = sleep_side_effect

            # 不应该抛出异常
            monitoring_service._monitor_loop()

            # 验证sleep被调用（说明循环继续运行）
            assert mock_sleep.called
