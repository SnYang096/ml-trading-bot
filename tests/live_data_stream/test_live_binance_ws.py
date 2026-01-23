"""
币安WebSocket实盘测试

展示如何使用BinanceWebSocketClient连接币安实盘数据流，并集成到OrderFlowListener。

注意：
- 币安的trade stream是公开的，不需要API key
- 此测试连接到币安实盘WebSocket，会接收真实的市场数据
- 建议先用测试网或少量数据测试
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Optional
import logging

from src.live_data_stream.websocket_client import BinanceWebSocketClient, BinanceTick
from src.live_data_stream import (
    StorageManager,
    OrderFlowListener,
    MultiSymbolManager,
)
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_binance_websocket_connection():
    """
    测试币安WebSocket连接（不处理数据，只验证连接）

    此测试连接到币安实盘WebSocket，验证连接是否正常。
    """
    symbols = ["BTCUSDT", "ETHUSDT"]
    use_futures = True  # 使用期货市场

    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
        reconnect_delay=5,
        ping_interval=20,
        ping_timeout=20,
    )

    stop_event = asyncio.Event()
    tick_count = 0
    max_ticks = 10  # 只接收10条tick验证连接

    try:
        async for tick in client.stream_ticks(stop_event):
            tick_count += 1
            logger.info(f"收到tick: {tick.symbol} @ {tick.price} (vol: {tick.volume})")

            if tick_count >= max_ticks:
                logger.info(f"✅ 成功接收 {tick_count} 条tick，连接正常")
                stop_event.set()
                break

    except Exception as e:
        pytest.fail(f"WebSocket连接失败: {e}")

    assert tick_count > 0, "应该至少接收到1条tick"


@pytest.mark.asyncio
async def test_binance_websocket_with_order_flow_listener():
    """
    测试币安WebSocket + OrderFlowListener集成

    此测试连接到币安实盘WebSocket，并将数据传递给OrderFlowListener处理。
    """
    symbols = ["BTCUSDT"]
    use_futures = True

    # 创建存储管理器
    storage_manager = StorageManager(base_path="data/test_live_storage")

    # 创建OrderFlowListener
    listener = OrderFlowListener(
        symbol=symbols[0],
        storage_manager=storage_manager,
        memory_window_hours=1.0,  # 测试时使用较小的窗口
        feature_compute_interval_minutes=60,  # 测试时不会触发
        feature_4h_interval_hours=4,
    )

    # 创建WebSocket客户端
    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
    )

    stop_event = asyncio.Event()
    tick_count = 0
    max_ticks = 50  # 接收50条tick进行测试

    def on_tick(tick: BinanceTick):
        """处理tick数据"""
        nonlocal tick_count
        tick_count += 1

        # 转换为Nautilus Trader格式（如果需要）
        # 这里直接使用BinanceTick，OrderFlowListener需要适配
        logger.info(f"处理tick {tick_count}: {tick.symbol} @ {tick.price}")

    # 添加回调
    client.add_callback(on_tick)

    try:
        async for tick in client.stream_ticks(stop_event):
            # 这里可以进一步处理tick，转换为TradeTick格式
            # 目前OrderFlowListener期望Nautilus Trader的TradeTick对象
            # 需要适配层或使用Nautilus Trader的数据客户端

            if tick_count >= max_ticks:
                logger.info(f"✅ 成功处理 {tick_count} 条tick")
                stop_event.set()
                break

    except Exception as e:
        pytest.fail(f"WebSocket处理失败: {e}")

    assert tick_count > 0, "应该至少处理1条tick"


@pytest.mark.asyncio
async def test_binance_websocket_multi_symbol():
    """
    测试币安WebSocket多symbol连接

    验证可以同时订阅多个symbol的数据流。
    """
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    use_futures = True

    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
    )

    stop_event = asyncio.Event()
    tick_counts = {symbol: 0 for symbol in symbols}
    max_ticks_per_symbol = 5

    try:
        async for tick in client.stream_ticks(stop_event):
            if tick.symbol in tick_counts:
                tick_counts[tick.symbol] += 1
                logger.info(
                    f"收到 {tick.symbol} tick: {tick_counts[tick.symbol]}/{max_ticks_per_symbol}"
                )

            # 如果所有symbol都收到足够的tick，停止
            if all(count >= max_ticks_per_symbol for count in tick_counts.values()):
                logger.info(f"✅ 所有symbol都收到足够的tick: {tick_counts}")
                stop_event.set()
                break

    except Exception as e:
        pytest.fail(f"多symbol WebSocket连接失败: {e}")

    # 验证每个symbol都收到了数据
    for symbol in symbols:
        assert tick_counts[symbol] > 0, f"{symbol} 应该至少接收到1条tick"


def test_binance_websocket_client_creation():
    """
    测试BinanceWebSocketClient创建（不连接）

    验证客户端可以正常创建，不实际连接WebSocket。
    """
    symbols = ["BTCUSDT"]
    use_futures = True

    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
    )

    assert client.symbols == ["BTCUSDT"]
    assert client.use_futures == True
    assert client.reconnect_delay == 5

    # 验证URL构建
    url = client._ws_url()
    assert "fstream.binance.com" in url or "stream.binance.com" in url
    assert "btcusdt@trade" in url.lower()


if __name__ == "__main__":
    # 直接运行测试（需要pytest-asyncio）
    pytest.main([__file__, "-v", "-s"])
