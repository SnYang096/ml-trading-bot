#!/usr/bin/env python3
"""
简单的WebSocket重连机制测试

测试核心功能，不依赖网络连接。
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.live_data_stream.reconnection_manager import (
    ReconnectionManager,
    ReconnectionConfig,
    ConnectionState,
)
from src.live_data_stream.connection_monitor import (
    ConnectionMonitor,
    HealthStatus,
)


async def test_reconnection_manager():
    """测试ReconnectionManager"""
    print("=" * 60)
    print("测试 ReconnectionManager")
    print("=" * 60)

    # 测试1: 初始状态
    print("\n1. 测试初始状态")
    manager = ReconnectionManager()
    assert manager.state == ConnectionState.DISCONNECTED
    print("   ✅ 初始状态正确")

    # 测试2: 指数退避
    print("\n2. 测试指数退避")
    manager = ReconnectionManager(
        config=ReconnectionConfig(
            initial_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=False,  # 关闭抖动以便测试
        )
    )

    delay1 = manager._calculate_delay()
    print(f"   第一次延迟: {delay1:.2f}s")
    assert delay1 == 1.0

    # 模拟失败
    manager._current_delay *= 2.0
    delay2 = manager._calculate_delay()
    print(f"   第二次延迟: {delay2:.2f}s")
    assert delay2 == 2.0

    manager._current_delay *= 2.0
    delay3 = manager._calculate_delay()
    print(f"   第三次延迟: {delay3:.2f}s")
    assert delay3 == 4.0
    print("   ✅ 指数退避正常工作")

    # 测试3: 最大重连次数
    print("\n3. 测试最大重连次数限制")
    manager = ReconnectionManager(
        config=ReconnectionConfig(
            initial_delay=0.1,
            max_retries=3,
        )
    )

    for i in range(3):
        should_continue = await manager.wait_before_reconnect()
        print(f"   重连尝试 {i+1}: should_continue={should_continue}")
        assert should_continue is True

    # 第4次应该失败
    should_continue = await manager.wait_before_reconnect()
    print(f"   重连尝试 4: should_continue={should_continue}")
    assert should_continue is False
    assert manager.state == ConnectionState.FAILED
    print("   ✅ 最大重连次数限制正常工作")

    # 测试4: 连接成功/失败回调
    print("\n4. 测试连接成功/失败回调")
    success_count = [0]
    failure_count = [0]

    def on_success():
        success_count[0] += 1

    def on_failure(error):
        failure_count[0] += 1

    manager = ReconnectionManager(
        on_reconnect_success=on_success,
        on_reconnect_failure=on_failure,
    )

    manager.on_connection_success()
    await asyncio.sleep(0.1)
    assert success_count[0] == 1
    assert manager.state == ConnectionState.CONNECTED
    print("   ✅ 连接成功回调正常工作")

    manager.on_connection_failure(Exception("Test error"))
    await asyncio.sleep(0.1)
    assert failure_count[0] == 1
    print("   ✅ 连接失败回调正常工作")

    # 测试5: 统计信息
    print("\n5. 测试统计信息")
    stats = manager.get_stats()
    assert "total_reconnects" in stats
    assert "successful_reconnects" in stats
    assert "failed_reconnects" in stats
    print(f"   统计信息: {stats}")
    print("   ✅ 统计信息正常")

    print("\n" + "=" * 60)
    print("✅ ReconnectionManager 所有测试通过")
    print("=" * 60)


async def test_connection_monitor():
    """测试ConnectionMonitor"""
    print("\n" + "=" * 60)
    print("测试 ConnectionMonitor")
    print("=" * 60)

    # 测试1: 初始状态
    print("\n1. 测试初始状态")
    monitor = ConnectionMonitor()
    assert monitor.health.status == HealthStatus.HEALTHY
    print("   ✅ 初始状态正确")

    # 测试2: 启动/停止监控
    print("\n2. 测试启动/停止监控")
    monitor.start_monitoring()
    assert monitor._monitoring is True
    await asyncio.sleep(0.1)
    monitor.stop_monitoring()
    assert monitor._monitoring is False
    print("   ✅ 启动/停止监控正常工作")

    # 测试3: 记录心跳和消息
    print("\n3. 测试记录心跳和消息")
    monitor = ConnectionMonitor()
    monitor.start_monitoring()

    monitor.record_heartbeat()
    await asyncio.sleep(0.1)
    assert monitor.health.last_heartbeat_time is not None
    print("   ✅ 心跳记录正常")

    monitor.record_message(latency_ms=5.5)
    await asyncio.sleep(0.1)
    assert monitor.health.message_count == 1
    assert monitor.health.latency_ms == 5.5
    print("   ✅ 消息记录正常")

    monitor.stop_monitoring()

    # 测试4: 健康状态
    print("\n4. 测试健康状态")
    health = monitor.get_health()
    assert "status" in health
    assert "message_count" in health
    print(f"   健康状态: {health}")
    print("   ✅ 健康状态正常")

    # 测试5: 重置
    print("\n5. 测试重置")
    monitor.health.message_count = 100
    monitor.reset()
    await asyncio.sleep(0.1)
    assert monitor.health.message_count == 0
    print("   ✅ 重置功能正常")

    print("\n" + "=" * 60)
    print("✅ ConnectionMonitor 所有测试通过")
    print("=" * 60)


async def test_integration():
    """集成测试"""
    print("\n" + "=" * 60)
    print("集成测试：ReconnectionManager + ConnectionMonitor")
    print("=" * 60)

    timeout_triggered = [False]

    def on_timeout():
        timeout_triggered[0] = True

    monitor = ConnectionMonitor(
        heartbeat_timeout=0.5,
        health_check_interval=0.2,
        on_timeout=on_timeout,
    )

    manager = ReconnectionManager(
        config=ReconnectionConfig(initial_delay=0.1),
    )

    print("\n1. 测试超时触发")
    monitor.start_monitoring()
    monitor.record_heartbeat()

    # 等待超时
    await asyncio.sleep(0.8)

    # 检查超时是否触发
    health = monitor.get_health()
    print(f"   健康状态: {health['status']}")
    assert (
        health["status"] in (HealthStatus.DEAD, HealthStatus.UNHEALTHY)
        or timeout_triggered[0]
    )
    print("   ✅ 超时检测正常工作")

    monitor.stop_monitoring()

    print("\n2. 测试失败后重连")
    manager.on_connection_failure(Exception("Test failure"))
    await asyncio.sleep(0.1)

    should_continue = await manager.wait_before_reconnect()
    assert should_continue is True
    print("   ✅ 失败后重连正常工作")

    manager.on_connection_success()
    await asyncio.sleep(0.1)
    assert manager.state == ConnectionState.CONNECTED
    print("   ✅ 重连成功正常工作")

    print("\n" + "=" * 60)
    print("✅ 集成测试通过")
    print("=" * 60)


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("WebSocket重连机制测试")
    print("=" * 60)

    try:
        await test_reconnection_manager()
        await test_connection_monitor()
        await test_integration()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
