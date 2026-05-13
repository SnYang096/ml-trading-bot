#!/usr/bin/env python3
"""Publish shared live bars/features to a disk-backed feature bus.

This is the B-framework market-data process: it owns the Binance market
WebSocket, aggregates 1m bars, computes configured feature timeframes, and
atomically publishes rolling parquet snapshots for downstream consumers.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_data_stream.feature_publisher_stack import (  # noqa: E402
    build_feature_bus_manager,
)
from src.live_data_stream.feature_bus import FeatureBusWriter  # noqa: E402
from src.live_data_stream.websocket_client import (  # noqa: E402
    BinanceTick,
    BinanceWebSocketClient,
)
from live.scripts.prepare_warmup_ticks import prepare_warmup_dataset  # noqa: E402

logger = logging.getLogger(__name__)


class FastMoveBarEmitter:
    """Emit supplemental execution bars as soon as intraminute moves are too large."""

    def __init__(
        self,
        writer: FeatureBusWriter,
        *,
        threshold_pct: float = 0.03,
        bucket_seconds: int = 10,
    ) -> None:
        self.writer = writer
        self.threshold_pct = float(threshold_pct)
        self.bucket_seconds = max(1, int(bucket_seconds))
        self._state: Dict[str, Dict[str, Any]] = {}

    def on_tick(self, tick: BinanceTick) -> None:
        if self.threshold_pct <= 0:
            return
        symbol = str(tick.symbol).upper()
        ts = pd.Timestamp(tick.timestamp_ms, unit="ms", tz="UTC")
        bucket_ns = self.bucket_seconds * 1_000_000_000
        bucket = pd.Timestamp((ts.value // bucket_ns) * bucket_ns, tz="UTC")
        price = float(tick.price)
        size = float(tick.volume or 0.0)
        state = self._state.get(symbol)
        if state is None or state["timestamp"] != bucket:
            state = {
                "timestamp": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "trade_count": 0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "buy_count": 0,
                "sell_count": 0,
                "_emitted": False,
            }
            self._state[symbol] = state
        state["high"] = max(float(state["high"]), price)
        state["low"] = min(float(state["low"]), price)
        state["close"] = price
        state["volume"] += size
        state["trade_count"] += 1
        if int(tick.side) == 1:
            state["buy_volume"] += size
            state["buy_count"] += 1
        else:
            state["sell_volume"] += size
            state["sell_count"] += 1
        ref = max(abs(float(state["open"])), 1e-12)
        move_pct = max(
            abs(float(state["high"]) / ref - 1.0),
            abs(float(state["low"]) / ref - 1.0),
        )
        if state["_emitted"] or move_pct < self.threshold_pct:
            return
        row = dict(state)
        row["timestamp"] = ts
        row["_bucket_start"] = bucket.isoformat()
        row.pop("_emitted", None)
        total_volume = float(row.get("volume") or 0.0)
        row["buy_ratio"] = (
            float(row.get("buy_volume") or 0.0) / total_volume if total_volume else 0.0
        )
        row["sell_ratio"] = (
            float(row.get("sell_volume") or 0.0) / total_volume if total_volume else 0.0
        )
        row["delta"] = float(row.get("buy_volume") or 0.0) - float(
            row.get("sell_volume") or 0.0
        )
        row["_bar_kind"] = "fast_intraminute"
        row["_source_timeframe_seconds"] = self.bucket_seconds
        row["_trigger_move_pct"] = move_pct
        self.writer.append_bar_1m(symbol, row)
        state["_emitted"] = True
        logger.warning(
            "fast execution bar emitted: %s ts=%s move=%.2f%% threshold=%.2f%%",
            symbol,
            bucket,
            move_pct * 100,
            self.threshold_pct * 100,
        )


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _tf_to_minutes(tf: str) -> int:
    tf_norm = str(tf).strip().lower()
    if tf_norm.endswith("min"):
        return int(tf_norm[:-3])
    if tf_norm.endswith("t"):
        return int(tf_norm[:-1])
    if tf_norm.endswith("h"):
        return int(float(tf_norm[:-1]) * 60)
    return int(pd.Timedelta(tf_norm).total_seconds() // 60)


def _tick_to_listener_tick(tick: BinanceTick) -> SimpleNamespace:
    return SimpleNamespace(
        price=float(tick.price),
        size=float(tick.volume),
        side=int(tick.side),
        timestamp=pd.Timestamp(tick.timestamp_ms, unit="ms", tz="UTC"),
        trade_id=tick.trade_id,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    p.add_argument("--feature-bus-root", default="live/shared_feature_bus")
    p.add_argument("--live-storage-base", default="data/live_storage")
    p.add_argument("--strategies-root", default="live/highcap/config/strategies")
    p.add_argument(
        "--constitution-yaml",
        default="",
        help="Override constitution path (default: next to strategies_root / constitution/).",
    )
    p.add_argument("--warmup-days", type=int, default=0)
    p.add_argument(
        "--warmup-months",
        type=int,
        default=6,
        help=(
            "Prepare live warmup ticks/bars via prepare_warmup_ticks.py (default: "
            "daily Binance Vision aggTrades ZIPs for the warmup window). "
            "Set 0 to skip."
        ),
    )
    p.add_argument(
        "--warmup-raw-dir",
        default="data/warmup_raw/highcap",
        help="Directory for Binance Vision warmup ZIPs.",
    )
    p.add_argument(
        "--skip-warmup-prepare",
        action="store_true",
        help=(
            "Skip Vision warmup ZIP download/prepare and only load existing disk data."
        ),
    )
    p.add_argument("--memory-window-hours", type=float, default=4.0)
    p.add_argument("--feature-compute-interval-minutes", type=int, default=15)
    p.add_argument("--orderflow-window-minutes", type=int, default=None)
    p.add_argument("--feature-4h-interval-hours", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--fast-bar-threshold-pct", type=float, default=0.03)
    p.add_argument("--fast-bar-bucket-seconds", type=int, default=10)
    p.add_argument("--use-futures", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _prepare_live_warmup(args: argparse.Namespace) -> None:
    if args.skip_warmup_prepare or int(args.warmup_months) <= 0:
        logger.warning("quant-feature-bus: warmup prepare skipped by config")
        return

    live_storage_base = _resolve_project_path(args.live_storage_base)
    prepare_warmup_dataset(
        symbols=_parse_symbols(args.symbols),
        months=int(args.warmup_months),
        ticks_dir=live_storage_base / "ticks",
        bars_dir=live_storage_base / "bars",
        zip_dir=_resolve_project_path(args.warmup_raw_dir),
        force_full=False,
        skip_download=False,
    )


async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    _prepare_live_warmup(args)
    writer = FeatureBusWriter(args.feature_bus_root, max_rows=args.max_rows)
    manager = build_feature_bus_manager(args, writer)
    fast_emitter = FastMoveBarEmitter(
        writer,
        threshold_pct=args.fast_bar_threshold_pct,
        bucket_seconds=args.fast_bar_bucket_seconds,
    )
    if args.warmup_days > 0:
        await manager.warmup_all(days=args.warmup_days, use_gap_filler=True)
    else:
        now = pd.Timestamp.now(tz="UTC")
        for listener in manager.listeners.values():
            listener.last_feature_compute_time = now
    await manager.start_all()

    ws_client = BinanceWebSocketClient(
        symbols=_parse_symbols(args.symbols), use_futures=args.use_futures
    )

    def _handle_tick(tick: BinanceTick) -> None:
        fast_emitter.on_tick(tick)
        manager.on_trade_tick(tick.symbol, _tick_to_listener_tick(tick))

    ws_client.add_callback(_handle_tick)
    stop_event = asyncio.Event()
    try:
        await ws_client.run(stop_event)
    finally:
        stop_event.set()
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(async_main())
