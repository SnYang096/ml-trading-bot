from __future__ import annotations

import asyncio
import json
import time
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

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
class AlltickTick:
    code: str
    tick_time_ms: int
    price: float
    volume: float
    turnover: float
    trade_direction: int  # 1=BUY, 2=SELL

    @classmethod
    def from_binance(cls, payload: Dict[str, Any]) -> "AlltickTick":
        """
        Parse a Binance trade payload into the legacy tick structure.
        """
        price = float(payload.get("p") or payload.get("price") or 0)
        qty = float(payload.get("q") or payload.get("qty") or 0)
        ts_ms = int(payload.get("T") or payload.get("E") or time.time() * 1000)
        # Binance: m=True means buyer is maker => seller is taker (sell aggressive)
        trade_direction = 2 if payload.get("m", False) else 1
        return cls(
            code=str(payload.get("s") or "").upper(),
            tick_time_ms=ts_ms,
            price=price,
            volume=qty,
            turnover=price * qty,
            trade_direction=trade_direction,
        )


class AlltickWebsocketClient:
    """
    Minimal Binance trade websocket client (compatibility shim for previous Alltick client).

    Responsibilities:
    - manages connection & reconnection (with exponential backoff)
    - subscribes via combined stream to all symbols
    - yields parsed AlltickTick objects from Binance trade events
    - connection monitoring and health checks
    """

    def __init__(
        self,
        token: str,  # kept for compatibility; not used for Binance
        symbols: List[str],
        use_stock_ws: bool = True,
        heartbeat_interval: int = 10,  # unused but preserved
        reconnect_delay: int = 5,
        reconnect_config: Optional[ReconnectionConfig] = None,
        max_reconnect_retries: Optional[int] = None,
        heartbeat_timeout: float = 60.0,
        health_check_interval: float = 30.0,
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets 模块未安装，请安装: pip install websockets")
        if not symbols:
            raise ValueError("symbols must not be empty for Binance subscription")

        self.symbols = [s.upper() for s in symbols]
        self.use_stock_ws = use_stock_ws
        
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
        )
        
        # 创建连接监控器
        self.connection_monitor = ConnectionMonitor(
            heartbeat_timeout=heartbeat_timeout,
            health_check_interval=health_check_interval,
            on_timeout=self._on_heartbeat_timeout,
        )
    
    def _on_reconnect_success(self) -> None:
        """重连成功回调"""
        logger.debug("Reconnection successful")
    
    def _on_reconnect_failure(self, error: Exception) -> None:
        """重连失败回调"""
        logger.debug(f"Reconnection failure: {error}")
    
    def _on_heartbeat_timeout(self) -> None:
        """心跳超时回调"""
        logger.warning("Heartbeat timeout detected")

    def _ws_url(self) -> str:
        base = SPOT_WS_BASE if self.use_stock_ws else FUTURES_WS_BASE
        streams = "/".join(f"{sym.lower()}@trade" for sym in self.symbols)
        return f"{base}/stream?streams={streams}"

    async def stream_ticks(self, stop_event: asyncio.Event) -> AsyncIterator[AlltickTick]:
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
                        url, ping_interval=20, ping_timeout=20, close_timeout=10
                    ) as ws:
                        logger.info(f"✅ WebSocket connected: {url}")
                        self.reconnect_manager.on_connection_success()
                        self.connection_monitor.record_heartbeat()
                        
                        async for msg in ws:
                            if stop_event.is_set():
                                break
                            
                            # 记录心跳
                            self.connection_monitor.record_heartbeat()
                            
                            if not msg:
                                continue
                            
                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError:
                                continue

                            payload = data.get("data") if isinstance(data, dict) and "data" in data else data
                            if not isinstance(payload, dict):
                                continue

                            # Binance trade stream event key is "trade"
                            if payload.get("e") != "trade":
                                continue

                            tick = AlltickTick.from_binance(payload)
                            
                            # 记录消息接收
                            self.connection_monitor.record_message()
                            
                            yield tick

                except Exception as exc:  # reconnect on any error
                    logger.error(f"[binance] connection error: {exc}")
                    self.reconnect_manager.on_connection_failure(exc)
                    
                    # 等待重连延迟
                    should_continue = await self.reconnect_manager.wait_before_reconnect()
                    if not should_continue:
                        break
        
        finally:
            # 停止连接监控
            self.connection_monitor.stop_monitoring()
        
        return
    
    def get_reconnect_stats(self) -> Dict[str, Any]:
        """获取重连统计信息"""
        return self.reconnect_manager.get_stats()
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取连接健康状态"""
        return self.connection_monitor.get_health()
