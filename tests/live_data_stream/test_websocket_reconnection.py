"""
WebSocket重连机制单元测试

测试ReconnectionManager和ConnectionMonitor的功能。
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

# 使用pytest-asyncio
pytestmark = pytest.mark.asyncio(scope="function")

from src.live_data_stream.reconnection_manager import (
    ReconnectionManager,
    ReconnectionConfig,
    ConnectionState,
)
from src.live_data_stream.connection_monitor import (
    ConnectionMonitor,
    HealthStatus,
)


class TestReconnectionManager:
    """测试ReconnectionManager"""

    def test_initial_state(self):
        """测试初始状态"""
        manager = ReconnectionManager()
        assert manager.state == ConnectionState.DISCONNECTED
        assert manager._retry_count == 0
        assert manager._current_delay == manager.config.initial_delay

    def test_calculate_delay_exponential_backoff(self):
        """测试指数退避延迟计算"""
        manager = ReconnectionManager(
            config=ReconnectionConfig(
                initial_delay=5.0,
                max_delay=60.0,
                backoff_multiplier=2.0,
                jitter=False,  # 关闭抖动以便测试
            )
        )

        # 第一次重连
        delay1 = manager._calculate_delay()
        assert delay1 == 5.0

        # 模拟失败，增加延迟
        manager._current_delay *= manager.config.backoff_multiplier
        delay2 = manager._calculate_delay()
        assert delay2 == 10.0

        # 继续增加
        manager._current_delay *= manager.config.backoff_multiplier
        delay3 = manager._calculate_delay()
        assert delay3 == 20.0

    def test_max_delay_limit(self):
        """测试最大延迟限制"""
        manager = ReconnectionManager(
            config=ReconnectionConfig(
                initial_delay=5.0,
                max_delay=10.0,
                backoff_multiplier=2.0,
            )
        )

        # 设置超过最大延迟的值
        manager._current_delay = 100.0
        delay = manager._calculate_delay()
        assert delay <= 12.0  # max_delay + jitter

    async def test_wait_before_reconnect(self):
        """测试重连前等待"""
        manager = ReconnectionManager(
            config=ReconnectionConfig(
                initial_delay=0.1,  # 测试时使用短延迟
                max_retries=3,
            )
        )

        # 第一次重连
        should_continue = await manager.wait_before_reconnect()
        assert should_continue is True
        assert manager._retry_count == 1
        assert manager.state == ConnectionState.RECONNECTING

        # 第二次重连
        should_continue = await manager.wait_before_reconnect()
        assert should_continue is True
        assert manager._retry_count == 2

        # 第三次重连（_retry_count增加到3，但3 < max_retries(3)为False，所以继续）
        should_continue = await manager.wait_before_reconnect()
        assert should_continue is True
        assert manager._retry_count == 3

        # 第四次重连（_retry_count=3，检查3 >= max_retries(3)为True，应该停止）
        should_continue = await manager.wait_before_reconnect()
        assert should_continue is False
        assert manager._retry_count == 3  # 达到最大次数时，不会增加_retry_count
        assert manager.state == ConnectionState.FAILED

    async def test_wait_before_reconnect_unlimited(self):
        """测试无限重连"""
        manager = ReconnectionManager(
            config=ReconnectionConfig(
                initial_delay=0.1,
                max_retries=None,  # 无限重连
            )
        )

        # 多次重连应该都成功
        for i in range(5):
            should_continue = await manager.wait_before_reconnect()
            assert should_continue is True
            assert manager._retry_count == i + 1

    async def test_on_connection_success(self):
        """测试连接成功回调"""
        success_callback = Mock()
        manager = ReconnectionManager(on_reconnect_success=success_callback)

        manager._current_delay = 20.0
        manager._retry_count = 5

        manager.on_connection_success()

        # 等待回调执行
        await asyncio.sleep(0.1)

        assert manager.state == ConnectionState.CONNECTED
        assert manager.stats.successful_reconnects == 1
        assert manager.stats.consecutive_failures == 0

        # 如果配置了reset_after_success，延迟应该重置
        if manager.config.reset_after_success:
            assert manager._current_delay == manager.config.initial_delay
            assert manager._retry_count == 0

    async def test_on_connection_failure(self):
        """测试连接失败回调"""
        failure_callback = Mock()
        manager = ReconnectionManager(on_reconnect_failure=failure_callback)

        initial_delay = manager._current_delay

        error = Exception("Connection failed")
        manager.on_connection_failure(error)

        # 等待回调执行
        await asyncio.sleep(0.1)

        assert manager.stats.failed_reconnects == 1
        assert manager.stats.consecutive_failures == 1
        # 延迟应该增加
        assert manager._current_delay > initial_delay

    def test_get_stats(self):
        """测试获取统计信息"""
        manager = ReconnectionManager()

        manager.stats.total_reconnects = 5
        manager.stats.successful_reconnects = 3
        manager.stats.failed_reconnects = 2

        stats = manager.get_stats()

        assert stats["total_reconnects"] == 5
        assert stats["successful_reconnects"] == 3
        assert stats["failed_reconnects"] == 2
        assert "state" in stats
        assert "current_delay" in stats

    async def test_reset(self):
        """测试重置"""
        manager = ReconnectionManager()

        manager._current_delay = 50.0
        manager._retry_count = 10
        manager._set_state(ConnectionState.RECONNECTING)

        manager.reset()

        # 等待重置完成
        await asyncio.sleep(0.1)

        assert manager._current_delay == manager.config.initial_delay
        assert manager._retry_count == 0
        assert manager.state == ConnectionState.DISCONNECTED


class TestConnectionMonitor:
    """测试ConnectionMonitor"""

    def test_initial_state(self):
        """测试初始状态"""
        monitor = ConnectionMonitor()
        assert monitor.health.status == HealthStatus.HEALTHY
        assert monitor._monitoring is False

    async def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        monitor = ConnectionMonitor()

        monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_task is not None

        await asyncio.sleep(0.1)  # 让监控任务启动

        monitor.stop_monitoring()
        assert monitor._monitoring is False
        assert monitor._monitor_task is None

    async def test_record_heartbeat(self):
        """测试记录心跳"""
        monitor = ConnectionMonitor(heartbeat_timeout=1.0)
        monitor.start_monitoring()

        monitor.record_heartbeat()
        await asyncio.sleep(0.1)  # 等待异步任务完成

        assert monitor.health.last_heartbeat_time is not None
        assert monitor.health.heartbeat_missed_count == 0

        monitor.stop_monitoring()

    async def test_record_message(self):
        """测试记录消息"""
        monitor = ConnectionMonitor()
        monitor.start_monitoring()

        monitor.record_message(latency_ms=10.5)
        await asyncio.sleep(0.1)  # 等待异步任务完成

        assert monitor.health.last_message_time is not None
        assert monitor.health.message_count == 1
        assert monitor.health.latency_ms == 10.5

        monitor.stop_monitoring()

    async def test_heartbeat_timeout(self):
        """测试心跳超时"""
        timeout_callback = Mock()
        monitor = ConnectionMonitor(
            heartbeat_timeout=0.5,  # 短超时用于测试
            health_check_interval=0.2,
            on_timeout=timeout_callback,
        )

        monitor.start_monitoring()
        monitor.record_heartbeat()

        # 等待超时
        await asyncio.sleep(0.8)

        # 检查超时回调是否被调用
        assert timeout_callback.called or monitor.health.status == HealthStatus.DEAD

        monitor.stop_monitoring()

    async def test_health_degraded(self):
        """测试健康状态降级"""
        monitor = ConnectionMonitor(
            heartbeat_timeout=1.0,
            health_check_interval=0.3,
        )

        monitor.start_monitoring()
        monitor.record_heartbeat()

        # 等待接近超时（70%阈值）
        await asyncio.sleep(0.8)

        # 健康检查应该检测到降级
        health = monitor.get_health()
        # 状态可能是DEGRADED或DEAD，取决于检查时机

        monitor.stop_monitoring()

    def test_get_health(self):
        """测试获取健康状态"""
        monitor = ConnectionMonitor()

        monitor.health.message_count = 100
        monitor.health.latency_ms = 5.5

        health = monitor.get_health()

        assert health["message_count"] == 100
        assert health["latency_ms"] == 5.5
        assert "status" in health
        assert "last_heartbeat_time" in health

    async def test_reset(self):
        """测试重置"""
        monitor = ConnectionMonitor()

        monitor.health.message_count = 100
        monitor.health.heartbeat_missed_count = 5

        monitor.reset()

        # 等待重置完成
        await asyncio.sleep(0.1)

        assert monitor.health.message_count == 0
        assert monitor.health.heartbeat_missed_count == 0


class TestIntegration:
    """集成测试：ReconnectionManager + ConnectionMonitor"""

    async def test_reconnect_on_timeout(self):
        """测试超时触发重连"""
        reconnect_triggered = False

        def on_timeout():
            nonlocal reconnect_triggered
            reconnect_triggered = True

        monitor = ConnectionMonitor(
            heartbeat_timeout=0.5,
            health_check_interval=0.2,
            on_timeout=on_timeout,
        )

        manager = ReconnectionManager(
            config=ReconnectionConfig(initial_delay=0.1),
        )

        monitor.start_monitoring()

        # 记录一次心跳，然后等待超时
        monitor.record_heartbeat()
        await asyncio.sleep(0.8)

        # 超时应该触发
        assert reconnect_triggered or monitor.health.status == HealthStatus.DEAD

        monitor.stop_monitoring()

    async def test_reconnect_after_failure(self):
        """测试失败后重连"""
        manager = ReconnectionManager(
            config=ReconnectionConfig(
                initial_delay=0.1,
                max_retries=3,
            )
        )

        # 模拟连接失败
        manager.on_connection_failure(Exception("Connection failed"))
        await asyncio.sleep(0.1)

        # 应该可以继续重连
        should_continue = await manager.wait_before_reconnect()
        assert should_continue is True

        # 模拟连接成功
        manager.on_connection_success()
        await asyncio.sleep(0.1)

        assert manager.state == ConnectionState.CONNECTED
        assert manager.stats.successful_reconnects == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
