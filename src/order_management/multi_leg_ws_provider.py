"""WebSocket-backed bar provider for multi-leg live strategies.

This provider reuses the classic live data stack for market data:

``BinanceWebSocketClient -> MultiSymbolManager -> OrderFlowListener ->
IncrementalFeatureComputer``

It deliberately does not use ``GenericLiveStrategy`` / ``LivePCM`` /
``OrderManager``. The resulting feature snapshots are translated into
``MultiLegBarEvent`` objects consumed by ``MultiLegLiveDaemon``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from types import SimpleNamespace
from typing import Any, Deque, Dict, Iterable, List, Optional

import pandas as pd

from src.live_data_stream.feature_storage import StorageManager
from src.live_data_stream.multi_symbol_manager import MultiSymbolManager
from src.live_data_stream.websocket_client import BinanceTick, BinanceWebSocketClient
from src.order_management.multi_leg_daemon import MultiLegBarEvent

logger = logging.getLogger(__name__)


class _DataOnlyOrderManager:
    """Sentinel so ``MultiSymbolManager`` does not initialize classic trading."""


class MultiLegWebSocketBarProvider:
    """Produce multi-leg feature bars from the classic WebSocket data stack."""

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        storage_base_path: str = "data/live_storage",
        feature_compute_interval_minutes: int = 15,
        memory_window_hours: float = 4.0,
        orderflow_window_minutes: Optional[int] = None,
        feature_4h_interval_hours: int = 4,
        warmup_days: int = 0,
        use_futures: bool = True,
    ) -> None:
        self.symbols = [str(s).upper() for s in symbols]
        self.storage_manager = StorageManager(storage_base_path)
        self.manager = MultiSymbolManager(
            symbols=self.symbols,
            storage_manager=self.storage_manager,
            memory_window_hours=memory_window_hours,
            feature_compute_interval_minutes=feature_compute_interval_minutes,
            orderflow_window_minutes=orderflow_window_minutes,
            feature_4h_interval_hours=feature_4h_interval_hours,
            # Prevent accidental initialization of the classic OrderManager.
            order_manager=_DataOnlyOrderManager(),
        )
        self.ws_client = BinanceWebSocketClient(self.symbols, use_futures=use_futures)
        self.warmup_days = int(warmup_days)
        self._queue: Deque[MultiLegBarEvent] = deque()
        self._latest_1m_bar: Dict[str, Dict[str, Any]] = {}
        self._seen_feature_keys: set[tuple[str, str]] = set()
        self._stop_event: Optional[asyncio.Event] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._started = False

        for symbol, listener in self.manager.listeners.items():
            listener.on_bar_callback = self._make_bar_callback(symbol)
            listener.on_feature_callback = self._make_feature_callback(symbol)
        self.ws_client.add_callback(self._on_tick)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self.warmup_days > 0:
            await self.manager.warmup_all(days=self.warmup_days)
        else:
            # Avoid an immediate batch feature computation against an empty or
            # partially prepared live_storage directory. The first slow signal
            # will be computed after feature_compute_interval_minutes.
            now = pd.Timestamp.now(tz="UTC")
            for listener in self.manager.listeners.values():
                listener.last_feature_compute_time = now
        await self.manager.start_all()
        self._stop_event = asyncio.Event()
        self._ws_task = asyncio.create_task(self.ws_client.run(self._stop_event))
        logger.info("multi-leg market WebSocket provider started")

    async def stop(self) -> None:
        if not self._started:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._ws_task is not None:
            try:
                await asyncio.wait_for(self._ws_task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._ws_task.cancel()
        await self.manager.stop_all()
        self._started = False
        logger.info("multi-leg market WebSocket provider stopped")

    def latest_closed_bars(self, symbols: Iterable[str]) -> List[MultiLegBarEvent]:
        allowed = {str(s).upper() for s in symbols}
        out: List[MultiLegBarEvent] = []
        remaining: Deque[MultiLegBarEvent] = deque()
        while self._queue:
            bar = self._queue.popleft()
            if bar.symbol in allowed:
                out.append(bar)
            else:
                remaining.append(bar)
        self._queue = remaining
        return out

    def _on_tick(self, tick: BinanceTick) -> None:
        listener_tick = SimpleNamespace(
            price=float(tick.price),
            size=float(tick.volume),
            side=int(tick.side),
            timestamp=pd.Timestamp(tick.timestamp_ms, unit="ms", tz="UTC"),
            trade_id=tick.trade_id,
        )
        self.manager.on_trade_tick(tick.symbol, listener_tick)

    def _make_bar_callback(self, symbol: str):
        def _callback(bar: Dict[str, Any]) -> None:
            self._latest_1m_bar[symbol.upper()] = dict(bar)

        return _callback

    def _make_feature_callback(self, symbol: str):
        def _callback(features: Dict[str, Any]) -> None:
            event = self._feature_event(symbol.upper(), features)
            key = (event.symbol, event.timestamp)
            if key in self._seen_feature_keys:
                return
            self._seen_feature_keys.add(key)
            self._queue.append(event)

        return _callback

    def _feature_event(self, symbol: str, features: Dict[str, Any]) -> MultiLegBarEvent:
        latest_bar = self._latest_1m_bar.get(symbol, {})
        ts = features.get("timestamp") or latest_bar.get("timestamp")
        if ts is None:
            ts = pd.Timestamp.utcnow()
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        else:
            ts_pd = ts_pd.tz_convert("UTC")
        ts_str = str(ts_pd)
        close = _first_float(features, latest_bar, ["close", "Close"], 0.0)
        high = _first_float(features, latest_bar, ["high", "High"], close)
        low = _first_float(features, latest_bar, ["low", "Low"], close)
        atr = _first_float(
            features,
            latest_bar,
            ["atr14", "atr", "ATR", "volatility_atr"],
            0.0,
        )
        return MultiLegBarEvent(
            symbol=symbol,
            timestamp=ts_str,
            high=high,
            low=low,
            close=close,
            atr=atr,
            features=dict(features),
        )


def _first_float(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    keys: List[str],
    default: float,
) -> float:
    for src in (primary, fallback):
        for key in keys:
            if key in src:
                try:
                    return float(src[key])
                except (TypeError, ValueError):
                    pass
    return float(default)
