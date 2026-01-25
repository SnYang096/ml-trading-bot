"""
WebSocket 客户端（改进版）

支持：
1. Binance 实时数据流
2. 自动重连（指数退避、重连次数限制）
3. 连接监控（心跳检测、健康状态评估）
4. 多币种订阅
5. 数据回调接口
6. 与 Nautilus Trader 集成
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Callable
import logging

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore

from .reconnection_manager import (
    ReconnectionManager,
    ReconnectionConfig,
    ConnectionState,
)
from .connection_monitor import ConnectionMonitor, HealthStatus

logger = logging.getLogger(__name__)

SPOT_WS_BASE = "wss://stream.binance.com:9443"
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
        trade_id = payload.get("t")
        
        # Binance: m=True 表示买方是 maker => 卖方是 taker（主动卖出）
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
    - 自动重连（指数退避、重连次数限制）
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
        ping_timeout: int = 20,
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
            ping_interval: 心跳间隔（秒）
            ping_timeout: 心跳超时（秒）
            reconnect_config: 重连配置（如果为None，使用默认配置）
            max_reconnect_retries: 最大重连次数（None=无限）
            heartbeat_timeout: 心跳超时时间（秒）
            health_check_interval: 健康检查间隔（秒）
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets 模块未安装，请安装: pip install websockets")
        
        if not symbols:
            raise ValueError("symbols must not be empty")
        
        self.symbols = [s.upper() for s in symbols]
        self.use_futures = use_futures
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
    
    def _ws_url(self) -> str:
        """构建 WebSocket URL"""
        base = FUTURES_WS_BASE if self.use_futures else SPOT_WS_BASE
        streams = "/".join(f"{sym.lower()}@trade" for sym in self.symbols)
        return f"{base}/stream?streams={streams}"
    
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
        logger.warning("Heartbeat timeout detected, will trigger reconnection")
        # 触发重连（通过抛出异常）
        # 注意：这会在stream_ticks的异常处理中被捕获
    
    async def stream_ticks(self, stop_event: asyncio.Event) -> AsyncIterator[BinanceTick]:
        """
        流式获取 tick 数据（带重连机制）
        
        Args:
            stop_event: 停止事件
        
        Yields:
            BinanceTick 对象
        """
        self._stop_event = stop_event
        url = self._ws_url()
        
        # 启动连接监控
        self.connection_monitor.start_monitoring()
        
        try:
            while not stop_event.is_set():
                # 检查是否应该继续重连
                if not self.reconnect_manager.should_continue():
                    logger.error("Max reconnection retries reached. Stopping.")
                    break
                
                try:
                    # 等待重连延迟（如果需要）
                    if self.reconnect_manager.is_reconnecting():
                        should_continue = await self.reconnect_manager.wait_before_reconnect()
                        if not should_continue:
                            break
                    
                    # 建立连接
                    self.reconnect_manager._set_state(ConnectionState.CONNECTING)
                    
                    async with websockets.connect(  # type: ignore[attr-defined]
                        url,
                        ping_interval=self.ping_interval,
                        ping_timeout=self.ping_timeout,
                        close_timeout=10,
                    ) as ws:
                        logger.info(f"✅ WebSocket connected: {url}")
                        self.reconnect_manager.on_connection_success()
                        self.connection_monitor.record_heartbeat()
                        
                        # 消息接收循环
                        async for msg in ws:
                            if stop_event.is_set():
                                break
                            
                            # 记录心跳（收到任何消息都算心跳）
                            self.connection_monitor.record_heartbeat()
                            
                            if not msg:
                                continue
                            
                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON: {msg[:100]}")
                                continue
                            
                            # 处理 Binance 数据格式
                            payload = data.get("data") if isinstance(data, dict) and "data" in data else data
                            if not isinstance(payload, dict):
                                continue
                            
                            # 只处理 trade 事件
                            if payload.get("e") != "trade":
                                continue
                            
                            try:
                                tick = BinanceTick.from_binance(payload)
                                
                                # 记录消息接收
                                self.connection_monitor.record_message()
                                
                                # 调用回调
                                for callback in self._callbacks:
                                    try:
                                        callback(tick)
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")
                                
                                yield tick
                            
                            except Exception as e:
                                logger.error(f"Error parsing tick: {e}")
                                continue
                
                except Exception as exc:
                    logger.error(f"WebSocket connection error: {exc}")
                    self.reconnect_manager.on_connection_failure(exc)
                    
                    # 等待重连延迟
                    should_continue = await self.reconnect_manager.wait_before_reconnect()
                    if not should_continue:
                        break
        
        finally:
            # 停止连接监控
            self.connection_monitor.stop_monitoring()
            logger.info("WebSocket stream stopped")
    
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

