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

from scripts.run_live import (
    _load_strategy_timeframe,
    _me_strategy_package_name,
)  # noqa: E402
from src.live_data_stream import StorageManager, MultiSymbolManager  # noqa: E402
from src.live_data_stream.feature_bus import FeatureBusWriter  # noqa: E402
from src.live_data_stream.websocket_client import (  # noqa: E402
    BinanceTick,
    BinanceWebSocketClient,
)
from src.time_series_model.live.incremental_feature_computer import (  # noqa: E402
    IncrementalFeatureComputer,
)
from src.time_series_model.live.live_feature_plan import (  # noqa: E402
    extract_features_from_archetypes,
)


logger = logging.getLogger(__name__)


class _DataOnlyOrderManager:
    """Sentinel so the publisher never initializes classic trading."""


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


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


def _extract_feature_plan(archetypes_dir: str) -> tuple[set, list]:
    if not os.path.isdir(archetypes_dir):
        return set(), []
    try:
        return extract_features_from_archetypes(archetypes_dir)
    except Exception as exc:
        logger.warning("feature extraction failed for %s: %s", archetypes_dir, exc)
        return set(), []


class FeatureBusDecisionSink:
    """DecisionHandler-compatible sink that publishes every computed timeframe."""

    def __init__(self, writer: FeatureBusWriter) -> None:
        self.writer = writer

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Any = None,
        features_by_timeframe: Dict[str, Dict[str, Any]] | None = None,
        decision_time: Any = None,
    ) -> list:
        ts = decision_time or features.get("timestamp") or pd.Timestamp.now(tz="UTC")
        by_tf = features_by_timeframe or {}
        if not by_tf:
            by_tf = {"primary": dict(features)}
        for tf, feat in by_tf.items():
            self.writer.append_features(
                symbol=symbol,
                timeframe=tf,
                features=feat,
                timestamp=ts,
            )
            # Multi-leg research uses pandas "2h"; classic live config often uses "120T".
            if str(tf).upper() == "120T":
                self.writer.append_features(
                    symbol=symbol,
                    timeframe="2h",
                    features=feat,
                    timestamp=ts,
                )
        return []


def _make_bar_callback(writer: FeatureBusWriter, symbol: str):
    def _callback(bar: Dict[str, Any]) -> None:
        try:
            writer.append_bar_1m(symbol, bar)
        except Exception:
            logger.exception("feature bus bar write failed: %s", symbol)

    return _callback


def _build_manager(
    args: argparse.Namespace, writer: FeatureBusWriter
) -> MultiSymbolManager:
    symbols = _parse_symbols(args.symbols)
    strategies_root = args.strategies_root
    me_pkg = _me_strategy_package_name(strategies_root)
    tf_bpc = _load_strategy_timeframe(strategies_root, "bpc")
    tf_me = _load_strategy_timeframe(strategies_root, me_pkg)
    tf_srb = _load_strategy_timeframe(strategies_root, "srb")
    tf_tpc = _load_strategy_timeframe(strategies_root, "tpc")

    bpc_dir = os.path.join(strategies_root, "bpc", "archetypes")
    me_dir = os.path.join(strategies_root, me_pkg, "archetypes")
    srb_dir = os.path.join(strategies_root, "srb", "archetypes")
    tpc_dir = os.path.join(strategies_root, "tpc", "archetypes")
    fer_dir = os.path.join(strategies_root, "fer", "archetypes")

    fer_set, fer_nodes = _extract_feature_plan(fer_dir)
    srb_set, srb_nodes = _extract_feature_plan(srb_dir)
    tpc_set, tpc_nodes = _extract_feature_plan(tpc_dir)

    def _base_fc(symbol: str) -> IncrementalFeatureComputer:
        bm = _tf_to_minutes(tf_bpc)
        fc = IncrementalFeatureComputer(
            tick_window_minutes=bm,
            bar_window_size=bm * 2,
            archetypes_dir=bpc_dir,
            primary_timeframe=tf_bpc,
        )
        if fer_set:
            fc.live_feature_set |= fer_set
            fc.live_feature_nodes = sorted(set(fc.live_feature_nodes) | set(fer_nodes))
        return fc

    storage = StorageManager(args.live_storage_base)
    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        feature_computer_factory=_base_fc,
        memory_window_hours=args.memory_window_hours,
        feature_compute_interval_minutes=args.feature_compute_interval_minutes,
        orderflow_window_minutes=args.orderflow_window_minutes,
        feature_4h_interval_hours=args.feature_4h_interval_hours,
        order_manager=_DataOnlyOrderManager(),
    )
    sink = FeatureBusDecisionSink(writer)

    for symbol, listener in manager.listeners.items():
        listener.on_bar_callback = _make_bar_callback(writer, symbol)
        listener.decision_handler = sink
        listener.order_manager = None
        extra: Dict[str, IncrementalFeatureComputer] = {}

        bm_me = _tf_to_minutes(tf_me)
        me_fc = IncrementalFeatureComputer(
            tick_window_minutes=bm_me,
            bar_window_size=bm_me * 2,
            archetypes_dir=me_dir,
            primary_timeframe=tf_me,
        )
        if fer_set:
            me_fc.live_feature_set |= fer_set
            me_fc.live_feature_nodes = sorted(
                set(me_fc.live_feature_nodes) | set(fer_nodes)
            )
        extra[tf_me] = me_fc

        for tf, base_dir, feat_set, nodes in [
            (tf_srb, srb_dir, srb_set, srb_nodes),
            (tf_tpc, tpc_dir, tpc_set, tpc_nodes),
        ]:
            if tf in extra:
                extra[tf].live_feature_set |= feat_set
                extra[tf].live_feature_nodes = sorted(
                    set(extra[tf].live_feature_nodes) | set(nodes)
                )
                continue
            bm = _tf_to_minutes(tf)
            fc = IncrementalFeatureComputer(
                tick_window_minutes=bm,
                bar_window_size=bm * 2,
                archetypes_dir=base_dir,
                primary_timeframe=tf,
            )
            fc.live_feature_set |= feat_set
            fc.live_feature_nodes = sorted(set(fc.live_feature_nodes) | set(nodes))
            extra[tf] = fc
        listener.extra_feature_computers = extra
    return manager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    p.add_argument("--feature-bus-root", default="live/shared_feature_bus")
    p.add_argument("--live-storage-base", default="data/live_storage")
    p.add_argument("--strategies-root", default="live/highcap/config/strategies")
    p.add_argument("--warmup-days", type=int, default=0)
    p.add_argument("--memory-window-hours", type=float, default=4.0)
    p.add_argument("--feature-compute-interval-minutes", type=int, default=15)
    p.add_argument("--orderflow-window-minutes", type=int, default=None)
    p.add_argument("--feature-4h-interval-hours", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--use-futures", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    writer = FeatureBusWriter(args.feature_bus_root, max_rows=args.max_rows)
    manager = _build_manager(args, writer)
    if args.warmup_days > 0:
        await manager.warmup_all(days=args.warmup_days, use_gap_filler=False)
    else:
        now = pd.Timestamp.now(tz="UTC")
        for listener in manager.listeners.values():
            listener.last_feature_compute_time = now
    await manager.start_all()

    ws_client = BinanceWebSocketClient(
        symbols=_parse_symbols(args.symbols), use_futures=args.use_futures
    )

    def _handle_tick(tick: BinanceTick) -> None:
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
