from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore


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
    - manages connection & reconnection
    - subscribes via combined stream to all symbols
    - yields parsed AlltickTick objects from Binance trade events
    """

    def __init__(
        self,
        token: str,  # kept for compatibility; not used for Binance
        symbols: List[str],
        use_stock_ws: bool = True,
        heartbeat_interval: int = 10,  # unused but preserved
        reconnect_delay: int = 5,
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets 模块未安装，请安装: pip install websockets")
        if not symbols:
            raise ValueError("symbols must not be empty for Binance subscription")

        self.symbols = [s.upper() for s in symbols]
        self.use_stock_ws = use_stock_ws
        self.reconnect_delay = reconnect_delay

    def _ws_url(self) -> str:
        base = SPOT_WS_BASE if self.use_stock_ws else FUTURES_WS_BASE
        streams = "/".join(f"{sym.lower()}@trade" for sym in self.symbols)
        return f"{base}/stream?streams={streams}"

    async def stream_ticks(self, stop_event: asyncio.Event) -> AsyncIterator[AlltickTick]:
        url = self._ws_url()

        while not stop_event.is_set():
            try:
                async with websockets.connect(  # type: ignore[attr-defined]
                    url, ping_interval=20, ping_timeout=20, close_timeout=10
                ) as ws:
                    async for msg in ws:
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
                        yield tick

                        if stop_event.is_set():
                            break

            except Exception as exc:  # reconnect on any error
                print(f"[binance] connection error: {exc}, retrying in {self.reconnect_delay}s")
                await asyncio.sleep(self.reconnect_delay)

        return
