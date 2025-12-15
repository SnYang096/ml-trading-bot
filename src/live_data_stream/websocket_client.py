"""
WebSocket 客户端（改进版）

支持：
1. Binance 实时数据流
2. 自动重连
3. 多币种订阅
4. 数据回调接口
5. 与 Nautilus Trader 集成
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
    - 自动重连
    - 多币种订阅
    - 数据回调
    - 心跳检测
    """
    
    def __init__(
        self,
        symbols: List[str],
        use_futures: bool = True,
        reconnect_delay: int = 5,
        ping_interval: int = 20,
        ping_timeout: int = 20,
    ):
        """
        Args:
            symbols: 交易对列表
            use_futures: 是否使用期货市场
            reconnect_delay: 重连延迟（秒）
            ping_interval: 心跳间隔（秒）
            ping_timeout: 心跳超时（秒）
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets 模块未安装，请安装: pip install websockets")
        
        if not symbols:
            raise ValueError("symbols must not be empty")
        
        self.symbols = [s.upper() for s in symbols]
        self.use_futures = use_futures
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        self._stop_event: Optional[asyncio.Event] = None
        self._callbacks: List[Callable[[BinanceTick], None]] = []
    
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
    
    async def stream_ticks(self, stop_event: asyncio.Event) -> AsyncIterator[BinanceTick]:
        """
        流式获取 tick 数据
        
        Args:
            stop_event: 停止事件
        
        Yields:
            BinanceTick 对象
        """
        self._stop_event = stop_event
        url = self._ws_url()
        
        while not stop_event.is_set():
            try:
                async with websockets.connect(  # type: ignore[attr-defined]
                    url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=10,
                ) as ws:
                    logger.info(f"✅ WebSocket connected: {url}")
                    
                    async for msg in ws:
                        if stop_event.is_set():
                            break
                        
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
                logger.error(f"WebSocket connection error: {exc}, retrying in {self.reconnect_delay}s")
                await asyncio.sleep(self.reconnect_delay)
        
        logger.info("WebSocket stream stopped")
    
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

