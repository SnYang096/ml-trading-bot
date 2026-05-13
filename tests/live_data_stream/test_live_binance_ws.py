"""
币安 WebSocket 相关测试。

- **默认不连交易所**：连接实盘的 async 测试会阻塞很久（甚至数小时）若网络/对端
  无 tick 或半开连接；CI 与本地「Run all live_data_stream tests」应秒过。
- 需要真连时设置环境变量：``MLBOT_RUN_LIVE_WS_TESTS=1``（可选再调
  ``MLBOT_LIVE_WS_TEST_TIMEOUT_SEC``，默认 90 秒）。
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

from src.live_data_stream import OrderFlowListener, StorageManager
from src.live_data_stream.websocket_client import BinanceTick, BinanceWebSocketClient

logger = logging.getLogger(__name__)


def _live_ws_tests_enabled() -> bool:
    v = os.environ.get("MLBOT_RUN_LIVE_WS_TESTS", "").strip().lower()
    return v in ("1", "true", "yes")


def _live_ws_timeout_sec() -> float:
    raw = os.environ.get("MLBOT_LIVE_WS_TEST_TIMEOUT_SEC", "").strip()
    if not raw:
        return 90.0
    return max(5.0, float(raw))


live_ws = pytest.mark.live_binance_ws


def _require_live_ws() -> None:
    if not _live_ws_tests_enabled():
        pytest.skip(
            "Live Binance WebSocket tests are off by default (can hang without ticks). "
            "Set MLBOT_RUN_LIVE_WS_TESTS=1 to enable; optional MLBOT_LIVE_WS_TEST_TIMEOUT_SEC=90."
        )


async def _stream_ticks_until(
    client: BinanceWebSocketClient,
    stop_event: asyncio.Event,
    *,
    stop_after_n_ticks: int,
    timeout_sec: float,
) -> int:
    """Consume stream until ``stop_after_n_ticks`` or timeout; always sets ``stop_event``."""

    async def _consume() -> int:
        n = 0
        async for _tick in client.stream_ticks(stop_event):
            n += 1
            if n >= stop_after_n_ticks:
                break
        return n

    try:
        n = await asyncio.wait_for(_consume(), timeout=timeout_sec)
    finally:
        stop_event.set()
    return n


@live_ws
@pytest.mark.asyncio
async def test_binance_websocket_connection():
    """连接币安 futures trade 流，收到若干条 tick 即通过。"""
    _require_live_ws()
    symbols = ["BTCUSDT", "ETHUSDT"]
    use_futures = True

    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
        reconnect_delay=5,
        ping_interval=20,
        ping_timeout=20,
    )

    stop_event = asyncio.Event()
    max_ticks = 10
    timeout_sec = _live_ws_timeout_sec()

    try:
        tick_count = await _stream_ticks_until(
            client, stop_event, stop_after_n_ticks=max_ticks, timeout_sec=timeout_sec
        )
    except asyncio.TimeoutError as e:
        pytest.fail(
            f"WebSocket: no {max_ticks} ticks within {timeout_sec}s "
            f"(check network / firewall). {e}"
        )
    except Exception as e:
        pytest.fail(f"WebSocket连接失败: {e}")

    assert tick_count > 0, "应该至少接收到1条tick"


@live_ws
@pytest.mark.asyncio
async def test_binance_websocket_with_order_flow_listener():
    """WebSocket + OrderFlowListener：tick 计数在回调与流两侧一致。"""
    _require_live_ws()
    symbols = ["BTCUSDT"]
    use_futures = True

    storage_manager = StorageManager(base_path="data/test_live_storage")

    listener = OrderFlowListener(
        symbol=symbols[0],
        storage_manager=storage_manager,
        memory_window_hours=1.0,
        feature_compute_interval_minutes=60,
        feature_4h_interval_hours=4,
    )

    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
    )

    stop_event = asyncio.Event()
    cb_count = 0
    max_ticks = 50
    timeout_sec = max(_live_ws_timeout_sec(), 120.0)

    def on_tick(tick: BinanceTick):
        nonlocal cb_count
        cb_count += 1
        logger.info(f"callback tick {cb_count}: {tick.symbol} @ {tick.price}")

    client.add_callback(on_tick)

    stream_count = 0
    try:

        async def _both():
            nonlocal stream_count
            async for tick in client.stream_ticks(stop_event):
                stream_count += 1
                if cb_count >= max_ticks or stream_count >= max_ticks:
                    break

        await asyncio.wait_for(_both(), timeout=timeout_sec)
    except asyncio.TimeoutError as e:
        pytest.fail(
            f"WebSocket+listener: fewer than {max_ticks} ticks in {timeout_sec}s. {e}"
        )
    except Exception as e:
        pytest.fail(f"WebSocket处理失败: {e}")
    finally:
        stop_event.set()

    assert cb_count > 0 and stream_count > 0, "应收到 tick（回调与流均 >0）"


@live_ws
@pytest.mark.asyncio
async def test_binance_websocket_multi_symbol():
    """多 symbol：在超时内尽量收到各 symbol 的成交；不强求各 5 条以免久等。"""
    _require_live_ws()
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    use_futures = True

    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
    )

    stop_event = asyncio.Event()
    tick_counts = {symbol: 0 for symbol in symbols}
    timeout_sec = max(_live_ws_timeout_sec(), 120.0)
    min_per = 2

    try:

        async def _multi():
            async for tick in client.stream_ticks(stop_event):
                if tick.symbol in tick_counts:
                    tick_counts[tick.symbol] += 1
                if all(tick_counts[s] >= min_per for s in symbols):
                    break

        await asyncio.wait_for(_multi(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        pass
    finally:
        stop_event.set()

    for symbol in symbols:
        assert (
            tick_counts[symbol] > 0
        ), f"{symbol} 在 {timeout_sec}s 内应至少收到 1 条 tick"


def test_binance_websocket_client_creation():
    """仅创建客户端与 URL，不建立连接。"""
    symbols = ["BTCUSDT"]
    use_futures = True

    client = BinanceWebSocketClient(
        symbols=symbols,
        use_futures=use_futures,
    )

    assert client.symbols == ["BTCUSDT"]
    assert client.use_futures is True
    assert client.reconnect_manager.config.initial_delay == 5.0

    url = client._ws_url()
    assert "fstream.binance.com" in url
    assert "btcusdt@trade" in url.lower()

    spot_client = BinanceWebSocketClient(symbols=symbols, use_futures=False)
    spot_url = spot_client._ws_url()
    assert "stream.binance.com" in spot_url
    assert "btcusdt@trade" in spot_url.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
