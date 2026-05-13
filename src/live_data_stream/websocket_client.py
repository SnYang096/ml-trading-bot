"""Binance USD-M futures aggregate-trade WebSocket client.

支持：
1. Binance USD-M futures ``@aggTrade`` real-time stream
2. 连接监控（基于真实消息的心跳检测、健康状态评估）
3. 多币种订阅
4. 数据回调接口

底层使用 python-binance 的 ``AsyncClient`` + ``BinanceSocketManager``，不再维护自建
``aiohttp``/``@trade`` 直连监听。Binance ``aggTrade`` 会按约 100ms 聚合同价、
同 taker side 的成交，能显著减少逐笔成交流量。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
import logging

try:
    from binance import AsyncClient, BinanceSocketManager
    from binance.enums import FuturesType
except ImportError:
    AsyncClient = None  # type: ignore
    BinanceSocketManager = None  # type: ignore
    FuturesType = None  # type: ignore

from .reconnection_manager import (
    ReconnectionManager,
    ReconnectionConfig,
    ConnectionState,
)
from .connection_monitor import ConnectionMonitor, HealthStatus

logger = logging.getLogger(__name__)

FUTURES_WS_BASE = "wss://fstream.binance.com"


@dataclass
class BinanceTick:
    """Binance tick 数据"""

    symbol: str
    timestamp_ms: int
    price: float
    volume: float
    turnover: float
    side: int  # 1=BUY, -1=SELL
    trade_id: Optional[int] = None

    @classmethod
    def from_binance(cls, payload: Dict[str, Any]) -> "BinanceTick":
        """从 Binance 数据解析"""
        price = float(payload.get("p") or payload.get("price") or 0)
        qty = float(payload.get("q") or payload.get("qty") or 0)
        ts_ms = int(payload.get("T") or payload.get("E") or time.time() * 1000)
        trade_id = payload.get("t") or payload.get(
            "a"
        )  # trade 用 t；遗留 aggTrade 解析用 a

        # Binance: m=True 表示买方是 maker => 卖方是 taker（主动卖出）。
        # ``aggTrade`` 保持同一字段语义，只是 q 已是同价同方向聚合量。
        is_buyer_maker = payload.get("m", False)
        side = 1 if not is_buyer_maker else -1

        return cls(
            symbol=str(payload.get("s") or "").upper(),
            timestamp_ms=ts_ms,
            price=price,
            volume=qty,
            turnover=price * qty,
            side=side,
            trade_id=trade_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "price": self.price,
            "volume": self.volume,
            "turnover": self.turnover,
            "side": self.side,
            "trade_id": self.trade_id,
        }


class BinanceWebSocketClient:
    """
    Binance WebSocket 客户端（改进版）

    特性：
    - 自动重连（连接异常或长时间无真实消息时重建 async socket）
    - 连接监控（心跳检测、健康状态评估）
    - 多币种订阅
    - 数据回调
    - 重连统计和回调
    """

    def __init__(
        self,
        symbols: List[str],
        use_futures: bool = True,
        reconnect_delay: int = 5,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        reconnect_config: Optional[ReconnectionConfig] = None,
        max_reconnect_retries: Optional[int] = None,
        heartbeat_timeout: float = 60.0,
        health_check_interval: float = 30.0,
    ):
        """
        Args:
            symbols: 交易对列表
            use_futures: 是否使用期货市场
            reconnect_delay: 初始重连延迟（秒）（已废弃，使用reconnect_config）
            ping_interval: 兼容旧 aiohttp 参数；BSM 模式不直接使用
            ping_timeout: 兼容旧 aiohttp 参数；BSM 模式不直接使用
            reconnect_config: 重连配置（如果为None，使用默认配置）
            max_reconnect_retries: 最大重连次数（None=无限）
            heartbeat_timeout: 心跳超时时间（秒）
            health_check_interval: 健康检查间隔（秒）
        """
        if AsyncClient is None or BinanceSocketManager is None:
            raise ImportError(
                "python-binance 模块未安装，请安装: pip install python-binance"
            )

        if not symbols:
            raise ValueError("symbols must not be empty")

        self.symbols = [s.upper() for s in symbols]
        if not use_futures:
            logger.warning(
                "BinanceWebSocketClient now always uses USD-M futures aggTrade; "
                "use_futures=False is ignored."
            )
        self.use_futures = True
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # 创建重连管理器
        if reconnect_config is None:
            reconnect_config = ReconnectionConfig(
                initial_delay=float(reconnect_delay),
                max_retries=max_reconnect_retries,
            )
        else:
            if max_reconnect_retries is not None:
                reconnect_config.max_retries = max_reconnect_retries

        self.reconnect_manager = ReconnectionManager(
            config=reconnect_config,
            on_reconnect_success=self._on_reconnect_success,
            on_reconnect_failure=self._on_reconnect_failure,
            on_state_change=self._on_state_change,
        )

        # 创建连接监控器
        self.connection_monitor = ConnectionMonitor(
            heartbeat_timeout=heartbeat_timeout,
            health_check_interval=health_check_interval,
            on_health_change=self._on_health_change,
            on_timeout=self._on_heartbeat_timeout,
        )

        self._stop_event: Optional[asyncio.Event] = None
        self._callbacks: List[Callable[[BinanceTick], None]] = []
        self._reconnect_callbacks: List[Callable[[], None]] = []
        self._health_callbacks: List[Callable[[HealthStatus], None]] = []
        self._binance_client: Optional[AsyncClient] = None

    def _ws_url(self) -> str:
        """Diagnostic URL equivalent to the python-binance aggTrade subscriptions."""
        streams = "/".join(f"{sym.lower()}@aggTrade" for sym in self.symbols)
        return f"{FUTURES_WS_BASE}/stream?streams={streams}"

    def add_callback(self, callback: Callable[[BinanceTick], None]) -> None:
        """添加数据回调"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[BinanceTick], None]) -> None:
        """移除数据回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_reconnect_callback(self, callback: Callable[[], None]) -> None:
        """添加重连成功回调"""
        self._reconnect_callbacks.append(callback)

    def add_health_callback(self, callback: Callable[[HealthStatus], None]) -> None:
        """添加健康状态变化回调"""
        self._health_callbacks.append(callback)

    def _on_reconnect_success(self) -> None:
        """重连成功回调"""
        for callback in self._reconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in reconnect callback: {e}")

    def _on_reconnect_failure(self, error: Exception) -> None:
        """重连失败回调"""
        logger.debug(f"Reconnect failure callback: {error}")

    def _on_state_change(self, state: ConnectionState) -> None:
        """连接状态变化回调"""
        logger.debug(f"Connection state changed: {state.value}")

    def _on_health_change(self, status: HealthStatus) -> None:
        """健康状态变化回调"""
        logger.debug(f"Health status changed: {status.value}")
        for callback in self._health_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Error in health callback: {e}")

    def _on_heartbeat_timeout(self) -> None:
        """心跳超时回调"""
        logger.warning("Heartbeat timeout detected by connection monitor")

    async def stream_ticks(
        self, stop_event: asyncio.Event
    ) -> AsyncIterator[BinanceTick]:
        """
        流式获取 USD-M futures aggTrade tick 数据。

        ``python-binance`` 的 ``BinanceSocketManager`` 原生运行在当前 async loop。
        每条 Binance
        ``aggTrade`` 仍转换为 ``BinanceTick``，因此上层 order-flow 聚合逻辑
        不需要关心底层 stream 类型变化。

        Args:
            stop_event: 停止事件

        Yields:
            BinanceTick 对象
        """
        self._stop_event = stop_event
        while not stop_event.is_set():
            stream = self._stream_ticks_once(stop_event)
            try:
                async for tick in stream:
                    yield tick
                break
            except Exception as exc:
                if stop_event.is_set():
                    break
                logger.error("python-binance aggTrade stream failed: %s", exc)
                self.reconnect_manager.on_connection_failure(exc)
                # ReconnectionManager updates stats in an async task.
                await asyncio.sleep(0)
                should_continue = await self.reconnect_manager.wait_before_reconnect()
                if not should_continue:
                    raise
            finally:
                await stream.aclose()

    async def _stream_ticks_once(
        self, stop_event: asyncio.Event
    ) -> AsyncIterator[BinanceTick]:
        """Open one async Binance socket session and yield ticks until stale."""
        last_message_monotonic = time.monotonic()

        # 启动连接监控
        self.connection_monitor.start_monitoring()

        def _dispatch_tick(tick: BinanceTick) -> None:
            nonlocal last_message_monotonic
            if stop_event.is_set():
                return
            last_message_monotonic = time.monotonic()
            self.connection_monitor.record_message()
            self.connection_monitor.record_heartbeat()
            for callback in self._callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error("Callback error: %s", e)

        def _parse_message(message: Dict[str, Any]) -> Optional[BinanceTick]:
            payload = message.get("data", message) if isinstance(message, dict) else {}
            if not isinstance(payload, dict):
                return None
            if payload.get("e") == "error":
                raise ConnectionError(f"python-binance socket error: {payload}")
            if payload.get("e") != "aggTrade":
                logger.debug("Ignoring non-aggTrade payload: %s", str(message)[:200])
                return None

            try:
                return BinanceTick.from_binance(payload)
            except Exception as e:
                logger.error("Error parsing aggTrade payload: %s", e)
                return None

        try:
            self.reconnect_manager._set_state(ConnectionState.CONNECTING)
            client = await AsyncClient.create()
            self._binance_client = client
            bsm = BinanceSocketManager(client)
            streams = [f"{symbol.lower()}@aggTrade" for symbol in self.symbols]

            self.reconnect_manager.on_connection_success()
            self.connection_monitor.record_heartbeat()
            logger.info(
                "✅ python-binance USD-M aggTrade stream started: %s",
                ",".join(self.symbols),
            )

            async with bsm.futures_multiplex_socket(
                streams=streams, futures_type=FuturesType.USD_M
            ) as socket:
                while not stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(socket.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        elapsed = time.monotonic() - last_message_monotonic
                        if elapsed > self.connection_monitor.heartbeat_timeout:
                            raise TimeoutError(
                                "No aggTrade messages received for "
                                f"{elapsed:.1f}s; restarting websocket session"
                            )
                        continue

                    tick = _parse_message(message)
                    if tick is None:
                        continue
                    _dispatch_tick(tick)
                    yield tick

        finally:
            if self._binance_client is not None:
                try:
                    await self._binance_client.close_connection()
                except Exception as e:
                    logger.warning("Error closing Binance AsyncClient: %s", e)
                self._binance_client = None
            # 停止连接监控
            self.connection_monitor.stop_monitoring()
            self.reconnect_manager._set_state(ConnectionState.DISCONNECTED)
            logger.info("python-binance aggTrade stream stopped")

    def get_reconnect_stats(self) -> Dict[str, Any]:
        """获取重连统计信息"""
        return self.reconnect_manager.get_stats()

    def get_health_status(self) -> Dict[str, Any]:
        """获取连接健康状态"""
        return self.connection_monitor.get_health()

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        运行 WebSocket 客户端（阻塞）

        Args:
            stop_event: 停止事件
        """
        async for tick in self.stream_ticks(stop_event):
            # 数据已通过回调处理
            pass


# 便捷函数
async def create_and_run_websocket(
    symbols: List[str],
    callback: Callable[[BinanceTick], None],
    use_futures: bool = True,
) -> None:
    """
    创建并运行 WebSocket 客户端

    Args:
        symbols: 交易对列表
        callback: 数据回调函数
        use_futures: 是否使用期货市场
    """
    client = BinanceWebSocketClient(symbols=symbols, use_futures=use_futures)
    client.add_callback(callback)

    stop_event = asyncio.Event()

    try:
        await client.run(stop_event)
    except KeyboardInterrupt:
        logger.info("Stopping WebSocket client...")
        stop_event.set()


if __name__ == "__main__":
    # 示例用法
    async def example_callback(tick: BinanceTick) -> None:
        print(f"Tick: {tick.symbol} @ {tick.price} ({tick.volume})")

    async def main():
        await create_and_run_websocket(
            symbols=["BTCUSDT", "ETHUSDT"],
            callback=example_callback,
            use_futures=True,
        )

    asyncio.run(main())
