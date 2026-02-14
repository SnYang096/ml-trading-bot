"""WebSocket 客户端（改进版）

支持：
1. Binance 实时数据流
2. 自动重连（指数退避、重连次数限制）
3. 连接监控（心跳检测、健康状态评估）
4. 多币种订阅
5. 数据回调接口
6. 与 Nautilus Trader 集成

底层使用 aiohttp 纯异步 WebSocket。
连接策略：先直连探测，失败则自动 fallback 到 HTTP_PROXY 环境变量代理。
- 远程服务器：直连成功，不走代理
- 本地开发（Clash/TUN）：直连超时，自动走 HTTP CONNECT 代理
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Callable
import logging

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore

# 保留 websockets / websocket-client 检测用于向后兼容
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
        trade_id = payload.get("a") or payload.get("t")  # aggTrade用a, trade用t
        
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
            ping_interval: 心跳间隔（秒）
            ping_timeout: 心跳超时（秒）
            reconnect_config: 重连配置（如果为None，使用默认配置）
            max_reconnect_retries: 最大重连次数（None=无限）
            heartbeat_timeout: 心跳超时时间（秒）
            health_check_interval: 健康检查间隔（秒）
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp 模块未安装，请安装: pip install aiohttp")
        
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
        streams = "/".join(f"{sym.lower()}@aggTrade" for sym in self.symbols)
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
    
    @staticmethod
    def _detect_proxy() -> Optional[str]:
        """
        从环境变量检测 HTTP 代理地址。
        
        优先级：HTTPS_PROXY > https_proxy > HTTP_PROXY > http_proxy
        返回格式："http://127.0.0.1:7897" 或 None
        """
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            val = os.environ.get(key)
            if val:
                return val
        return None
    
    async def _resolve_proxy(
        self, session: aiohttp.ClientSession, probe_timeout: float = 5.0,
    ) -> Optional[str]:
        """
        自适应探测连接方式：先直连，失败则 fallback 到代理。
        
        1. 直连 Binance WebSocket (probe_timeout 秒)
           - 成功 → return None     （服务器环境，无需代理）
           - 超时/失败 → 进入步骤 2
        2. 通过 HTTP_PROXY 代理连接
           - 有代理环境变量且连接成功 → return proxy_url
           - 无代理或连接失败 → return None（报错留给上层重连逻辑处理）
        
        Returns:
            proxy URL 或 None
        """
        base = FUTURES_WS_BASE if self.use_futures else SPOT_WS_BASE
        # 用单个轻量流做探测，避免组合流 URL 太长
        probe_url = f"{base}/ws/btcusdt@aggTrade"
        
        async def _try_connect(px: Optional[str]) -> bool:
            """尝试连接并接收 1 条消息，成功返回 True。"""
            ct = aiohttp.ClientTimeout(total=probe_timeout, connect=probe_timeout)
            async with session.ws_connect(
                probe_url, proxy=px, timeout=ct,
            ) as ws:
                await ws.receive()
                await ws.close()
            return True
        
        # --- 步骤 1：直连探测 ---
        try:
            await asyncio.wait_for(_try_connect(None), timeout=probe_timeout)
            logger.info("✅ Direct connection OK — no proxy needed")
            return None
        except Exception as e:
            logger.info(f"⚠️  Direct probe failed ({type(e).__name__}), trying proxy...")
        
        # --- 步骤 2：代理探测 ---
        env_proxy = self._detect_proxy()
        if not env_proxy:
            logger.warning("⚠️  No proxy env var found (HTTP_PROXY/HTTPS_PROXY). "
                           "Will attempt direct connection anyway.")
            return None
        
        try:
            await asyncio.wait_for(_try_connect(env_proxy), timeout=probe_timeout)
            logger.info(f"✅ Proxy connection OK — using {env_proxy}")
            return env_proxy
        except Exception as e:
            logger.warning(f"⚠️  Proxy probe also failed ({type(e).__name__}: {e}). "
                           f"Will attempt direct connection as fallback.")
            return None
    
    async def stream_ticks(self, stop_event: asyncio.Event) -> AsyncIterator[BinanceTick]:
        """
        流式获取 tick 数据（带重连机制）
        
        底层使用 aiohttp 纯异步 WebSocket。
        启动时自动探测连接方式：直连 OK 则不走代理，失败则 fallback 到 HTTP_PROXY。
        
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
            async with aiohttp.ClientSession() as session:
                # 探测连接方式：直连 or 代理（只在首次启动时探测）
                proxy = await self._resolve_proxy(session)
                if proxy:
                    logger.info(f"🔌 WebSocket using proxy: {proxy}")
                else:
                    logger.info("🔌 WebSocket: direct connection (no proxy)")
                
                while not stop_event.is_set():
                    # 检查是否应该继续重连
                    if not self.reconnect_manager.should_continue():
                        logger.error("Max reconnection retries reached. Stopping.")
                        break
                    
                    # 等待重连延迟（如果需要）
                    if self.reconnect_manager.is_reconnecting():
                        should_continue = await self.reconnect_manager.wait_before_reconnect()
                        if not should_continue:
                            break
                    
                    # 建立连接
                    self.reconnect_manager._set_state(ConnectionState.CONNECTING)
                    
                    try:
                        async with session.ws_connect(
                            url,
                            proxy=proxy,
                            heartbeat=self.ping_interval,
                            timeout=aiohttp.ClientTimeout(
                                total=None,      # 无总超时（长连接）
                                connect=30.0,    # 连接超时 30秒
                                sock_connect=30.0,
                                sock_read=None,  # 无读超时（流式读取）
                            ),
                            max_msg_size=2**23,  # 8MB
                        ) as ws:
                            logger.info(f"✅ WebSocket connected: {url}")
                            self.reconnect_manager.on_connection_success()
                            self.connection_monitor.record_heartbeat()
                            
                            # 消息接收循环
                            async for msg in ws:
                                if stop_event.is_set():
                                    break
                                
                                # 记录心跳
                                self.connection_monitor.record_heartbeat()
                                
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    try:
                                        data = json.loads(msg.data)
                                    except json.JSONDecodeError:
                                        logger.warning(f"Invalid JSON: {str(msg.data)[:100]}")
                                        continue
                                    
                                    # 处理 Binance 数据格式
                                    payload = data.get("data") if isinstance(data, dict) and "data" in data else data
                                    if not isinstance(payload, dict):
                                        continue
                                    
                                    # 只处理 trade / aggTrade 事件
                                    event_type = payload.get("e", "")
                                    if event_type not in ("trade", "aggTrade"):
                                        continue
                                    
                                    try:
                                        tick = BinanceTick.from_binance(payload)
                                        self.connection_monitor.record_message()
                                        
                                        for callback in self._callbacks:
                                            try:
                                                callback(tick)
                                            except Exception as e:
                                                logger.error(f"Callback error: {e}")
                                        
                                        yield tick
                                    except Exception as e:
                                        logger.error(f"Error parsing tick: {e}")
                                        continue
                                
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logger.error(f"WebSocket error: {ws.exception()}")
                                    break
                                
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                                    logger.info("WebSocket closed by server")
                                    break
                    
                    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                        logger.error(f"WebSocket connection error: {exc}")
                        self.reconnect_manager.on_connection_failure(exc)
                    
                    except Exception as exc:
                        logger.error(f"WebSocket unexpected error: {exc}")
                        self.reconnect_manager.on_connection_failure(exc)
                    
                    # 如果不是因为 stop_event 退出，走重连流程
                    if not stop_event.is_set():
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

