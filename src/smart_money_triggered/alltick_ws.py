from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore


STOCK_WS_URL = "wss://quote.alltick.co/quote-stock-b-ws-api"
GENERIC_WS_URL = "wss://quote.alltick.co/quote-b-ws-api"


@dataclass
class AlltickTick:
    code: str
    tick_time_ms: int
    price: float
    volume: float
    turnover: float
    trade_direction: int  # 1=BUY, 2=SELL

    @classmethod
    def from_push(cls, payload: Dict[str, Any]) -> "AlltickTick":
        return cls(
            code=str(payload.get("code") or ""),
            tick_time_ms=int(payload.get("tick_time") or 0),
            price=float(payload.get("price") or 0),
            volume=float(payload.get("volume") or 0),
            turnover=float(payload.get("turnover") or 0),
            trade_direction=int(payload.get("trade_direction") or 0),
        )


def _make_subscribe_body(symbols: Iterable[str], seq_id: int) -> Dict[str, Any]:
    return {
        "cmd_id": 22004,
        "seq_id": seq_id,
        "trace": str(uuid.uuid4()),
        "data": {"symbol_list": [{"code": code} for code in symbols]},
    }


def _make_heartbeat_body(seq_id: int) -> Dict[str, Any]:
    return {"cmd_id": 22000, "seq_id": seq_id, "trace": str(uuid.uuid4()), "data": {}}


class AlltickWebsocketClient:
    """
    Minimal Alltick websocket client.

    Responsibilities:
    - manages connection & heartbeat
    - sends subscription request with full symbol list
    - yields parsed AlltickTick objects from push cmd_id=22998
    """

    def __init__(
        self,
        token: str,
        symbols: List[str],
        use_stock_ws: bool = True,
        heartbeat_interval: int = 10,
        reconnect_delay: int = 5,
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets 模块未安装，请安装: pip install websockets")
        self.token = token
        self.symbols = symbols
        self.use_stock_ws = use_stock_ws
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_delay = reconnect_delay
        self._seq = 1

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _ws_url(self) -> str:
        base = STOCK_WS_URL if self.use_stock_ws else GENERIC_WS_URL
        return f"{base}?token={self.token}"

    async def _send_heartbeat(self, ws: websockets.WebSocketClientProtocol) -> None:  # type: ignore[attr-defined]
        payload = _make_heartbeat_body(self._next_seq())
        await ws.send(json.dumps(payload))

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol) -> None:  # type: ignore[attr-defined]
        payload = _make_subscribe_body(self.symbols, self._next_seq())
        await ws.send(json.dumps(payload))

    async def stream_ticks(self, stop_event: asyncio.Event) -> AsyncIterator[AlltickTick]:
        url = self._ws_url()
        heartbeat_task: Optional[asyncio.Task] = None

        while not stop_event.is_set():
            try:
                async with websockets.connect(  # type: ignore[attr-defined]
                    url, ping_interval=None, ping_timeout=None
                ) as ws:
                    await self._subscribe(ws)

                    async def _heartbeat_loop() -> None:
                        while not stop_event.is_set():
                            await asyncio.sleep(self.heartbeat_interval)
                            try:
                                await self._send_heartbeat(ws)
                            except Exception:
                                break

                    heartbeat_task = asyncio.create_task(_heartbeat_loop())

                    async for msg in ws:
                        if not msg:
                            continue
                        try:
                            data = json.loads(msg)
                        except json.JSONDecodeError:
                            continue

                        cmd_id = data.get("cmd_id")
                        if cmd_id == 22998 and isinstance(data.get("data"), dict):
                            tick = AlltickTick.from_push(data["data"])
                            yield tick

                        if stop_event.is_set():
                            break

            except Exception as exc:  # reconnect on any error
                print(f"[alltick] connection error: {exc}, retrying in {self.reconnect_delay}s")
                await asyncio.sleep(self.reconnect_delay)
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
                heartbeat_task = None

        return

