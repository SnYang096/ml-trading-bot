from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from .websocket_client import BinanceWebSocketClient
from .config_loader import SmartMoneySettings, load_settings
from .signals import SignalResult, generate_signal
from .tick_store import TickStorage, aggregate_ticks_100ms


def _current_trading_date(now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now()
    return now.strftime("%Y-%m-%d")


@dataclass
class SignalEngineConfig:
    threshold: float = 0.0
    takebuy_min: float = 0.65
    cluster_min: float = 0.7
    vpin_max: float = 0.75
    vwap_discount: float = 0.98


class SmartMoneyEngine:
    """
    Event-driven + manual execution engine.
    """

    def __init__(
        self,
        settings: Optional[SmartMoneySettings] = None,
        engine_cfg: Optional[SignalEngineConfig] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.engine_cfg = engine_cfg or SignalEngineConfig()
        self.storage = TickStorage(root=self.settings.storage_dir)

    async def _collect_and_store(
        self, stop_event: asyncio.Event, symbols: List[str], use_stock_ws: bool
    ) -> None:
        client = BinanceWebSocketClient(symbols=symbols, use_futures=not use_stock_ws)

        buffer: Dict[str, List] = {sym: [] for sym in symbols}
        trading_date = _current_trading_date()

        async for tick in client.stream_ticks(stop_event):
            buffer.setdefault(tick.symbol, []).append(tick)

            # flush per symbol when buffer large
            if len(buffer[tick.symbol]) >= 200:  # ~20s of 100ms bins worst-case
                agg_df = aggregate_ticks_100ms(buffer[tick.symbol])
                if not agg_df.empty:
                    self.storage.append(tick.symbol, trading_date, agg_df)
                buffer[tick.symbol].clear()

            if stop_event.is_set():
                break

        # flush remaining
        for sym, items in buffer.items():
            if items:
                agg_df = aggregate_ticks_100ms(items)
                if not agg_df.empty:
                    self.storage.append(sym, trading_date, agg_df)

    async def start_realtime(self) -> asyncio.Task:
        """
        Start realtime subscriptions for stocks and cryptos in parallel.
        Returns a task that can be awaited or cancelled.
        """
        stop_event = asyncio.Event()

        async def _runner() -> None:
            tasks = []
            if self.settings.stock_symbols:
                tasks.append(
                    asyncio.create_task(
                        self._collect_and_store(stop_event, self.settings.stock_symbols, True)
                    )
                )
            if self.settings.crypto_symbols:
                tasks.append(
                    asyncio.create_task(
                        self._collect_and_store(stop_event, self.settings.crypto_symbols, False)
                    )
                )
            try:
                await asyncio.gather(*tasks)
            finally:
                stop_event.set()

        return asyncio.create_task(_runner())

    def compute_signal_for_day(self, symbol: str, trading_date: Optional[str] = None) -> SignalResult:
        """
        Manual run: load stored ticks (100ms agg) for a given date and compute signal at 14:50.
        """
        trading_date = trading_date or _current_trading_date()
        df = self.storage.load(symbol, trading_date)
        # 过滤到 14:50 之前
        cutoff = pd.Timestamp(f"{trading_date} 14:50", tz="Asia/Shanghai")
        df = df[df["ts"] <= cutoff.tz_convert("UTC")]
        result = generate_signal(
            df,
            threshold=self.engine_cfg.threshold,
            takebuy_min=self.engine_cfg.takebuy_min,
            cluster_min=self.engine_cfg.cluster_min,
            vpin_max=self.engine_cfg.vpin_max,
            vwap_discount=self.engine_cfg.vwap_discount,
        )
        return result

    def compute_signals_for_all(self, trading_date: Optional[str] = None) -> Dict[str, SignalResult]:
        trading_date = trading_date or _current_trading_date()
        results: Dict[str, SignalResult] = {}
        symbols = self.settings.unique_symbols()
        for sym in symbols:
            try:
                results[sym] = self.compute_signal_for_day(sym, trading_date)
            except FileNotFoundError:
                continue
        return results

    async def run_daily_signal_loop(
        self,
        time_str: str = "14:50",
        tz: str = "Asia/Shanghai",
        on_result=None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Simple scheduler: every day at `time_str` compute signals for all symbols.
        """
        stop_event = stop_event or asyncio.Event()
        zone = ZoneInfo(tz)
        hour, minute = map(int, time_str.split(":"))

        while not stop_event.is_set():
            now = dt.datetime.now(zone)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                target = target + dt.timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
                break
            except asyncio.TimeoutError:
                pass  # reach target time

            trading_date = target.strftime("%Y-%m-%d")
            results = self.compute_signals_for_all(trading_date)
            if on_result:
                on_result(trading_date, results)
            else:
                print(f"[smart-money] signals for {trading_date}: {results}")

