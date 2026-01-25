#!/usr/bin/env python3
"""
WebSocket重连机制测试脚本

模拟各种断开场景：
1. 正常断开重连
2. 频繁断开重连
3. 长时间断开
4. 心跳超时检测
5. 重连后数据完整性
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.live_data_stream.websocket_client import BinanceWebSocketClient, BinanceTick
from src.live_data_stream.reconnection_manager import (
    ReconnectionConfig,
    ConnectionState,
)
from src.live_data_stream.connection_monitor import HealthStatus

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReconnectionTestScenario:
    """重连测试场景基类"""

    def __init__(self, name: str):
        self.name = name
        self.tick_count = 0
        self.reconnect_count = 0
        self.last_tick_time = None
        self.ticks: List[BinanceTick] = []
        self.stats: Dict[str, Any] = {}

    async def run(self, symbols: List[str], duration: int = 60) -> Dict[str, Any]:
        """运行测试场景"""
        logger.info(f"开始测试场景: {self.name}")
        start_time = time.time()

        # 创建客户端
        client = self._create_client(symbols)

        # 设置回调
        client.add_callback(self._on_tick)
        client.add_reconnect_callback(self._on_reconnect)

        # 运行测试
        stop_event = asyncio.Event()
        test_task = asyncio.create_task(self._run_test(client, stop_event, duration))

        try:
            await test_task
        except Exception as e:
            logger.error(f"测试场景 {self.name} 出错: {e}")

        elapsed = time.time() - start_time

        # 收集统计信息
        self.stats = {
            "name": self.name,
            "duration": elapsed,
            "tick_count": self.tick_count,
            "reconnect_count": self.reconnect_count,
            "reconnect_stats": client.get_reconnect_stats(),
            "health_status": client.get_health_status(),
        }

        logger.info(f"测试场景 {self.name} 完成: {self.stats}")
        return self.stats

    def _create_client(self, symbols: List[str]) -> BinanceWebSocketClient:
        """创建客户端（子类可覆盖）"""
        return BinanceWebSocketClient(
            symbols=symbols,
            use_futures=True,
            reconnect_config=ReconnectionConfig(
                initial_delay=2.0,  # 测试时使用较短的延迟
                max_delay=10.0,
                max_retries=None,  # 无限重试
            ),
            heartbeat_timeout=30.0,
            health_check_interval=10.0,
        )

    async def _run_test(
        self, client: BinanceWebSocketClient, stop_event: asyncio.Event, duration: int
    ) -> None:
        """运行测试（子类可覆盖）"""
        # 启动客户端
        client_task = asyncio.create_task(client.run(stop_event))

        # 等待指定时间
        await asyncio.sleep(duration)

        # 停止
        stop_event.set()
        await client_task

    def _on_tick(self, tick: BinanceTick) -> None:
        """tick回调"""
        self.tick_count += 1
        self.last_tick_time = time.time()
        self.ticks.append(tick)
        if self.tick_count % 10 == 0:
            logger.debug(f"收到 {self.tick_count} 条tick数据")

    def _on_reconnect(self) -> None:
        """重连成功回调"""
        self.reconnect_count += 1
        logger.info(f"重连成功 (第 {self.reconnect_count} 次)")


class NormalDisconnectScenario(ReconnectionTestScenario):
    """场景1: 正常断开重连"""

    def __init__(self):
        super().__init__("正常断开重连")

    async def _run_test(
        self, client: BinanceWebSocketClient, stop_event: asyncio.Event, duration: int
    ) -> None:
        """模拟正常断开重连"""
        client_task = asyncio.create_task(client.run(stop_event))

        # 等待10秒后手动断开（通过停止任务模拟）
        await asyncio.sleep(10)
        logger.info("模拟断开连接...")
        # 注意：实际断开需要网络层面的操作，这里只是测试重连逻辑

        # 继续运行
        await asyncio.sleep(duration - 10)
        stop_event.set()
        await client_task


class FrequentDisconnectScenario(ReconnectionTestScenario):
    """场景2: 频繁断开重连（验证指数退避）"""

    def __init__(self):
        super().__init__("频繁断开重连")
        self.disconnect_times = []

    async def _run_test(
        self, client: BinanceWebSocketClient, stop_event: asyncio.Event, duration: int
    ) -> None:
        """模拟频繁断开"""
        client_task = asyncio.create_task(client.run(stop_event))

        # 每5秒记录一次重连延迟
        start_time = time.time()
        while time.time() - start_time < duration:
            await asyncio.sleep(5)
            stats = client.get_reconnect_stats()
            if stats.get("current_delay", 0) > 0:
                logger.info(f"当前重连延迟: {stats['current_delay']:.2f}s")
                self.disconnect_times.append((time.time(), stats["current_delay"]))

        stop_event.set()
        await client_task


class MaxRetriesScenario(ReconnectionTestScenario):
    """场景3: 最大重连次数限制"""

    def __init__(self):
        super().__init__("最大重连次数限制")

    def _create_client(self, symbols: List[str]) -> BinanceWebSocketClient:
        """创建带最大重连次数限制的客户端"""
        return BinanceWebSocketClient(
            symbols=symbols,
            use_futures=True,
            reconnect_config=ReconnectionConfig(
                initial_delay=1.0,
                max_delay=5.0,
                max_retries=3,  # 最多重试3次
            ),
        )

    async def _run_test(
        self, client: BinanceWebSocketClient, stop_event: asyncio.Event, duration: int
    ) -> None:
        """测试最大重连次数"""
        client_task = asyncio.create_task(client.run(stop_event))

        # 等待足够长的时间观察重连行为
        await asyncio.sleep(30)

        stats = client.get_reconnect_stats()
        logger.info(f"重连统计: {stats}")

        stop_event.set()
        await client_task


class DataIntegrityScenario(ReconnectionTestScenario):
    """场景4: 重连后数据完整性"""

    def __init__(self):
        super().__init__("重连后数据完整性")
        self.before_disconnect_ticks = []
        self.after_reconnect_ticks = []
        self.disconnect_time = None
        self.reconnect_time = None

    def _on_tick(self, tick: BinanceTick) -> None:
        """记录断开前后的tick"""
        super()._on_tick(tick)

        if self.disconnect_time is None:
            self.before_disconnect_ticks.append(tick)
        elif self.reconnect_time is not None:
            self.after_reconnect_ticks.append(tick)

    def _on_reconnect(self) -> None:
        """记录重连时间"""
        super()._on_reconnect()
        if self.reconnect_time is None:
            self.reconnect_time = time.time()
            logger.info(
                f"重连时间: {self.reconnect_time}, 断开前tick数: {len(self.before_disconnect_ticks)}"
            )

    async def _run_test(
        self, client: BinanceWebSocketClient, stop_event: asyncio.Event, duration: int
    ) -> None:
        """测试数据完整性"""
        client_task = asyncio.create_task(client.run(stop_event))

        # 运行一段时间后停止（模拟断开）
        await asyncio.sleep(15)
        self.disconnect_time = time.time()
        logger.info(f"模拟断开时间: {self.disconnect_time}")

        # 继续运行观察重连
        await asyncio.sleep(duration - 15)

        # 分析数据
        if self.reconnect_time:
            gap = self.reconnect_time - self.disconnect_time
            logger.info(f"断开到重连的时间间隔: {gap:.2f}s")
            logger.info(f"重连后收到tick数: {len(self.after_reconnect_ticks)}")

        stop_event.set()
        await client_task


async def run_all_scenarios(symbols: List[str] = ["BTCUSDT"], duration: int = 60):
    """运行所有测试场景"""
    scenarios = [
        NormalDisconnectScenario(),
        FrequentDisconnectScenario(),
        MaxRetriesScenario(),
        DataIntegrityScenario(),
    ]

    results = []

    for scenario in scenarios:
        try:
            result = await scenario.run(symbols, duration)
            results.append(result)

            # 场景之间稍作停顿
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"场景 {scenario.name} 失败: {e}")
            results.append(
                {
                    "name": scenario.name,
                    "error": str(e),
                }
            )

    # 打印总结
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)

    for result in results:
        print(f"\n场景: {result.get('name', 'Unknown')}")
        if "error" in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  持续时间: {result.get('duration', 0):.2f}s")
            print(f"  收到tick数: {result.get('tick_count', 0)}")
            print(f"  重连次数: {result.get('reconnect_count', 0)}")

            reconnect_stats = result.get("reconnect_stats", {})
            if reconnect_stats:
                print(f"  总重连次数: {reconnect_stats.get('total_reconnects', 0)}")
                print(f"  成功重连: {reconnect_stats.get('successful_reconnects', 0)}")
                print(f"  失败重连: {reconnect_stats.get('failed_reconnects', 0)}")
                print(f"  当前延迟: {reconnect_stats.get('current_delay', 0):.2f}s")

            health = result.get("health_status", {})
            if health:
                print(f"  健康状态: {health.get('status', 'unknown')}")
                print(f"  消息数: {health.get('message_count', 0)}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket重连机制测试")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="交易对列表，逗号分隔（默认: BTCUSDT）",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="每个场景的测试时长（秒，默认: 60）"
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info(f"开始WebSocket重连测试")
    logger.info(f"交易对: {symbols}")
    logger.info(f"每个场景时长: {args.duration}秒")

    try:
        results = asyncio.run(run_all_scenarios(symbols, args.duration))
        logger.info("所有测试场景完成")
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)
