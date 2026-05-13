from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from .reconnection_manager import (
    ReconnectionManager,
    ReconnectionConfig,
)
from .connection_monitor import ConnectionMonitor, HealthStatus
from .websocket_client import BinanceTick, BinanceWebSocketClient

logger = logging.getLogger(__name__)


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
        Parse a Binance aggTrade payload into the legacy tick structure.
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

    @classmethod
    def from_binance_tick(cls, tick: BinanceTick) -> "AlltickTick":
        return cls(
            code=tick.symbol,
            tick_time_ms=int(tick.timestamp_ms),
            price=float(tick.price),
            volume=float(tick.volume),
            turnover=float(tick.turnover),
            trade_direction=1 if int(tick.side) == 1 else 2,
        )


class AlltickWebsocketClient:
    """
    Compatibility shim for the previous Alltick client.

    Responsibilities:
    - uses ``BinanceWebSocketClient`` (python-binance USD-M futures aggTrade)
    - yields parsed AlltickTick objects from Binance aggTrade events
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
        streams = "/".join(f"{sym.lower()}@aggTrade" for sym in self.symbols)
        return f"{FUTURES_WS_BASE}/stream?streams={streams}"

    async def stream_ticks(
        self, stop_event: asyncio.Event
    ) -> AsyncIterator[AlltickTick]:
        # 启动连接监控
        self.connection_monitor.start_monitoring()
        client = BinanceWebSocketClient(self.symbols, use_futures=True)

        try:
            async for tick in client.stream_ticks(stop_event):
                self.connection_monitor.record_heartbeat()
                self.connection_monitor.record_message()
                yield AlltickTick.from_binance_tick(tick)

        except Exception as exc:
            logger.error("[binance] aggTrade connection error: %s", exc)
            self.reconnect_manager.on_connection_failure(exc)
            raise

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
