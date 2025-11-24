"""
Nautilus Trader live-trading stub integrating Binance & Hyperliquid websockets.

This module outlines how to bootstrap Nautilus Trader live feeds. It intentionally
keeps implementations lightweight and safe (no actual orders are sent). Use this
as a starting point when wiring real live-trading logic.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    websockets = None


HYPERLIQUID_WSS = "wss://api.hyperliquid.xyz/ws"


@dataclass
class LiveStreamConfig:
    symbol: str
    interval: str = "1m"
    market: Optional[str] = None


class NautilusLiveStub:
    """
    Minimal stub for bridging Nautilus Trader with popular exchanges via websockets.

    Usage:
        stub = NautilusLiveStub(strategy_name="sr_reversal")
        asyncio.run(stub.run_binance_feed(LiveStreamConfig(symbol="btcusdt")))
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    async def run_hyperliquid_feed(self, config: LiveStreamConfig) -> None:
        if websockets is None:
            print("⚠️  Install `websockets` package to enable Hyperliquid streaming.")
            return

        print(f"🔌 Connecting to Hyperliquid stream: {HYPERLIQUID_WSS}")
        async with websockets.connect(HYPERLIQUID_WSS, ping_interval=20) as ws:
            subscribe = {
                "method": "subscribe",
                "topic": "trades",
                "market": config.market or config.symbol.upper(),
            }
            await ws.send(json.dumps(subscribe))
            async for message in ws:
                payload = json.loads(message)
                self._handle_market_data("hyperliquid", payload)

    def _handle_market_data(self, venue: str, payload: Dict[str, Any]) -> None:
        """
        Placeholder for feeding data into Nautilus Trader.
        Replace prints with Nautilus Trader event ingestion in production.
        """
        print(f"[{self.strategy_name}] {venue}: {payload}")

    def run_example(self) -> None:
        """Helper to quickly test the stub."""

        async def runner():
            await self.run_hyperliquid_feed(
                LiveStreamConfig(symbol="BTC", market="BTC-USD-PERP")
            )

        asyncio.run(runner())
