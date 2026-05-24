#!/usr/bin/env python3
"""Publish shared live bars/features to a disk-backed feature bus.

This is the B-framework market-data process: it owns the Binance market
WebSocket, aggregates 1m bars, computes configured feature timeframes, and
atomically publishes rolling parquet snapshots for downstream consumers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_data_stream.feature_publisher_stack import (  # noqa: E402
    build_feature_bus_manager,
)
from src.live_data_stream.feature_bus import FeatureBusWriter  # noqa: E402
from src.live_data_stream.auto_gap_fill import (  # noqa: E402
    auto_gap_fill_loop,
    run_auto_gap_fill_once,
)
from src.live_data_stream.websocket_client import (  # noqa: E402
    BinanceTick,
    BinanceWebSocketClient,
    configure_binance_ws_queue_size,
)
from src.time_series_model.live.metrics_exporter import (  # noqa: E402
    METRICS,
    start_metrics_server,
)
from live.scripts.prepare_warmup_ticks import prepare_warmup_dataset  # noqa: E402
from scripts.live_audit_file import configure_audit_from_env_defaults  # noqa: E402

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
    p.add_argument(
        "--auto-gap-fill-interval-minutes",
        type=float,
        default=float(os.getenv("MLBOT_AUTO_GAP_FILL_INTERVAL_MINUTES", "60")),
        help="Background large-gap repair interval. Set <=0 to disable.",
    )
    p.add_argument(
        "--auto-gap-fill-lookback-hours",
        type=float,
        default=float(os.getenv("MLBOT_AUTO_GAP_FILL_LOOKBACK_HOURS", "48")),
        help="How far back to scan live 1m bars for large gaps.",
    )
    p.add_argument(
        "--auto-gap-fill-startup-lookback-hours",
        type=float,
        default=float(os.getenv("MLBOT_AUTO_GAP_FILL_STARTUP_LOOKBACK_HOURS", "0")),
        help=(
            "Startup repair scan window. Set 0 to use warmup-days; "
            "keeps long feature windows clean before the first feature audit."
        ),
    )
    p.add_argument(
        "--auto-gap-fill-min-gap-minutes",
        type=float,
        default=float(os.getenv("MLBOT_AUTO_GAP_FILL_MIN_GAP_MINUTES", "60")),
        help="Minimum 1m bar gap size repaired by the background filler.",
    )
    p.add_argument(
        "--auto-gap-fill-max-gaps-per-run",
        type=int,
        default=int(os.getenv("MLBOT_AUTO_GAP_FILL_MAX_GAPS_PER_RUN", "24")),
        help="Maximum detected gaps repaired per auto-gap-fill pass.",
    )
    p.add_argument(
        "--auto-gap-fill-initial-delay-seconds",
        type=float,
        default=float(os.getenv("MLBOT_AUTO_GAP_FILL_INITIAL_DELAY_SECONDS", "300")),
        help="Delay after feature-bus startup before first background gap scan.",
    )
    p.add_argument(
        "--macro-kline-root",
        default="live/highcap/data/macro/spot_klines",
        help="Cache for Binance Vision spot 1d klines (weekly EMA macro lane).",
    )
    p.add_argument(
        "--weekly-ema-seed-root",
        default="live/highcap/data/macro/spot_weekly_ema200",
        help="Output/read path for weekly EMA200 seed parquets.",
    )
    p.add_argument(
        "--skip-macro-warmup",
        action="store_true",
        help="Skip Vision spot daily download and weekly EMA seed build.",
    )
    p.add_argument(
        "--macro-seed-start-date",
        default="2017-01-01",
        help="Start date for spot daily kline seed (YYYY-MM-DD).",
    )
    return p.parse_args()


def _refresh_funding_oi_on_startup(symbols: List[str]) -> None:
    """REST 增量补最近 funding/OI（长历史仍靠 Vision parquet + 挂卷）。"""
    flag = os.getenv("MLBOT_FUNDING_OI_REFRESH_ON_START", "1").strip().lower()
    if flag in ("0", "false", "off", "no"):
        logger.info("quant-feature-bus: funding/OI startup refresh disabled")
        return
    lookback = int(os.getenv("MLBOT_FUNDING_OI_LOOKBACK_DAYS", "60"))
    data_root = os.getenv("MLBOT_DATA_ROOT", "data")
    try:
        from scripts.refresh_funding_oi_data import refresh_all

        logger.info(
            "quant-feature-bus: refreshing funding/OI (lookback=%d days, root=%s)...",
            lookback,
            data_root,
        )
        result = refresh_all(symbols, data_root=data_root, lookback_days=lookback)
        logger.info(
            "quant-feature-bus: funding/OI refresh done FR=%d files, OI=%d files",
            result["funding_rate_files"],
            result["oi_files"],
        )
    except Exception as exc:
        logger.warning(
            "quant-feature-bus: funding/OI refresh failed (non-fatal): %s", exc
        )


def _prepare_macro_weekly_ema_seed(args: argparse.Namespace) -> None:
    symbols = _parse_symbols(args.symbols)
    seed_root = _resolve_project_path(args.weekly_ema_seed_root)
    os.environ["MLBOT_WEEKLY_EMA_SEED_ROOT"] = str(seed_root)

    skip_flag = os.getenv("MLBOT_SKIP_MACRO_WARMUP", "").strip().lower()
    skip_env = skip_flag in ("1", "true", "yes", "on")
    if getattr(args, "skip_macro_warmup", False) or skip_env:
        try:
            from src.live_data_stream.spot_weekly_ema_seed import macro_seeds_ready

            ready, missing = macro_seeds_ready(symbols, seed_root)
            if ready:
                logger.info(
                    "quant-feature-bus: macro seed download skipped; using existing seeds at %s",
                    seed_root,
                )
            else:
                logger.warning(
                    "quant-feature-bus: macro seed skipped but not ready for: %s "
                    "(run quant-macro-seed-prepare or scripts/prepare_spot_weekly_ema_seed.py)",
                    ",".join(missing),
                )
        except Exception:
            logger.warning(
                "quant-feature-bus: macro weekly EMA seed download skipped by config"
            )
        return

    kline_root = _resolve_project_path(args.macro_kline_root)
    try:
        from datetime import date

        from src.live_data_stream.spot_weekly_ema_seed import (
            prepare_spot_weekly_ema_seed,
        )

        start = date.fromisoformat(str(args.macro_seed_start_date))
        logger.info(
            "quant-feature-bus: preparing spot weekly EMA seed for %s (root=%s)...",
            ",".join(symbols),
            seed_root,
        )
        written = prepare_spot_weekly_ema_seed(
            symbols,
            kline_root=kline_root,
            seed_root=seed_root,
            start_date=start,
        )
        logger.info(
            "quant-feature-bus: macro seed done (%d symbols, seed_root=%s)",
            len(written),
            seed_root,
        )
    except Exception as exc:
        logger.warning(
            "quant-feature-bus: macro weekly EMA seed failed (non-fatal): %s", exc
        )


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


def _configure_feature_bus_audit(live_storage_base: Path) -> None:
    os.environ.setdefault("MLBOT_FEATURE_BUS_AUDIT", "1")
    audit_root = _resolve_project_path(
        os.getenv("MLBOT_FEATURE_BUS_AUDIT_BASE", str(live_storage_base))
    )
    configure_audit_from_env_defaults(
        default_log_file=audit_root / "logs" / "feature_bus_audit.log",
        disable_env="MLBOT_FEATURE_BUS_AUDIT_DISABLE",
        path_env="MLBOT_FEATURE_BUS_AUDIT_LOG",
        retention_env="MLBOT_FEATURE_BUS_AUDIT_RETENTION_DAYS",
        rotation_env="MLBOT_FEATURE_BUS_AUDIT_ROTATION",
        banner="quant-feature-bus audit file",
    )


async def _startup_feature_audit(manager: Any) -> None:
    """One-shot compute after warmup to catch OI/FR gaps before WS goes live."""
    if os.getenv("MLBOT_FEATURE_BUS_AUDIT_POST_WARMUP", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return
    logger.info(
        "quant-feature-bus: post-warmup feature audit (one compute per symbol)..."
    )
    loop = asyncio.get_running_loop()

    def _run_one(listener: Any) -> None:
        listener._compute_and_save_15min_features()

    for symbol, listener in manager.listeners.items():
        try:
            await loop.run_in_executor(None, _run_one, listener)
        except Exception as exc:
            from src.live_data_stream.feature_bus_audit import FeatureBusAuditError

            if isinstance(exc, FeatureBusAuditError):
                raise
            logger.warning(
                "quant-feature-bus: post-warmup audit failed for %s: %s", symbol, exc
            )
    logger.info("quant-feature-bus: post-warmup feature audit done")


async def _startup_gap_repair(
    args: argparse.Namespace, manager: Any, symbols: List[str], writer: Any
) -> None:
    """Repair persisted gaps before computing feature-bus snapshots."""
    if args.auto_gap_fill_interval_minutes <= 0 or manager.gap_filler is None:
        return

    startup_lookback = float(args.auto_gap_fill_startup_lookback_hours)
    if startup_lookback <= 0:
        startup_lookback = max(
            float(args.auto_gap_fill_lookback_hours),
            float(args.warmup_days) * 24.0,
        )

    logger.info(
        "auto-gap-fill: startup repair lookback=%.1fh min_gap=%.1fmin",
        startup_lookback,
        args.auto_gap_fill_min_gap_minutes,
    )
    loop = asyncio.get_running_loop()

    def _run_once() -> int:
        return run_auto_gap_fill_once(
            manager.storage_manager,
            manager.gap_filler,
            symbols,
            lookback_hours=startup_lookback,
            min_gap_minutes=args.auto_gap_fill_min_gap_minutes,
            max_gaps_per_run=args.auto_gap_fill_max_gaps_per_run,
            feature_bus_writer=writer,
        )

    try:
        written = await loop.run_in_executor(None, _run_once)
        logger.info("auto-gap-fill: startup repair done written_bars=%d", written)
    except Exception:
        logger.exception(
            "auto-gap-fill: startup repair failed; continuing to start_all"
        )


async def _periodic_process_metrics() -> None:
    interval = int(os.getenv("MLBOT_MARKET_DATA_INTERVAL", "30"))
    loop = asyncio.get_running_loop()
    while True:
        try:
            await loop.run_in_executor(None, METRICS.update_system_health)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("feature-bus process metrics update skipped: %s", exc)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    ws_queue = configure_binance_ws_queue_size()
    logger.info(
        "Binance WS MAX_QUEUE_SIZE=%d (override with MLBOT_BINANCE_WS_MAX_QUEUE)",
        ws_queue,
    )
    symbols = _parse_symbols(args.symbols)
    live_storage_base = _resolve_project_path(args.live_storage_base)
    _configure_feature_bus_audit(live_storage_base)
    os.environ.setdefault("MLBOT_LIVE_STORAGE_BASE", str(live_storage_base))
    os.environ.setdefault("MLBOT_FEATURE_BUS_ROOT", str(args.feature_bus_root))
    metrics_port = int(os.getenv("MLBOT_METRICS_PORT", "9090"))
    start_metrics_server(port=metrics_port)
    _refresh_funding_oi_on_startup(symbols)
    _prepare_macro_weekly_ema_seed(args)
    _prepare_live_warmup(args)
    writer = FeatureBusWriter(args.feature_bus_root, max_rows=int(args.max_rows))
    logger.info(
        "feature-bus rolling cap=%d rows (warmup-days=%s; warmup is restart-only and"
        " no longer dictates bus capacity)",
        args.max_rows,
        args.warmup_days,
    )
    manager = build_feature_bus_manager(args, writer)
    from src.live_data_stream.feature_storage import sanitize_dated_parquet_for_symbols

    sanitize_lookback = int(os.getenv("MLBOT_PARQUET_SANITIZE_LOOKBACK_DAYS", "3"))
    sanitize_dated_parquet_for_symbols(
        manager.storage_manager,
        symbols,
        lookback_days=sanitize_lookback,
    )
    fast_emitter = FastMoveBarEmitter(
        writer,
        threshold_pct=args.fast_bar_threshold_pct,
        bucket_seconds=args.fast_bar_bucket_seconds,
    )
    if args.warmup_days > 0:
        await manager.warmup_all(days=args.warmup_days, use_gap_filler=True)
        await _startup_gap_repair(args, manager, symbols, writer)
        await _startup_feature_audit(manager)
    else:
        now = pd.Timestamp.now(tz="UTC")
        for listener in manager.listeners.values():
            listener.last_feature_compute_time = now
    await manager.start_all()
    auto_gap_task: Optional[asyncio.Task] = None
    if args.auto_gap_fill_interval_minutes > 0 and manager.gap_filler is not None:
        logger.info(
            "auto-gap-fill: enabled interval=%.1fmin lookback=%.1fh min_gap=%.1fmin",
            args.auto_gap_fill_interval_minutes,
            args.auto_gap_fill_lookback_hours,
            args.auto_gap_fill_min_gap_minutes,
        )
        auto_gap_task = asyncio.create_task(
            auto_gap_fill_loop(
                manager.storage_manager,
                manager.gap_filler,
                symbols,
                interval_seconds=args.auto_gap_fill_interval_minutes * 60.0,
                lookback_hours=args.auto_gap_fill_lookback_hours,
                startup_lookback_hours=(
                    args.auto_gap_fill_startup_lookback_hours
                    if args.auto_gap_fill_startup_lookback_hours > 0
                    else None
                ),
                min_gap_minutes=args.auto_gap_fill_min_gap_minutes,
                max_gaps_per_run=args.auto_gap_fill_max_gaps_per_run,
                initial_delay_seconds=args.auto_gap_fill_initial_delay_seconds,
                feature_bus_writer=writer,
            )
        )
    else:
        logger.info("auto-gap-fill: disabled")

    ws_client = BinanceWebSocketClient(symbols=symbols, use_futures=args.use_futures)
    tick_queue: asyncio.Queue[BinanceTick] = asyncio.Queue(
        maxsize=max(1000, int(os.getenv("MLBOT_TICK_DISPATCH_QUEUE", "20000")))
    )

    def _set_ws_connected(value: int) -> None:
        for symbol in symbols:
            METRICS.ws_connected.labels(symbol=symbol).set(value)

    def _on_ws_health(status: Any) -> None:
        status_value = getattr(status, "value", str(status))
        _set_ws_connected(0 if status_value in {"unhealthy", "dead"} else 1)

    def _enqueue_tick(tick: BinanceTick) -> None:
        try:
            tick_queue.put_nowait(tick)
        except asyncio.QueueFull:
            logger.warning(
                "tick dispatch queue full (%d); drop %s",
                tick_queue.maxsize,
                tick.symbol,
            )

    def _process_tick(tick: BinanceTick) -> None:
        fast_emitter.on_tick(tick)
        manager.on_trade_tick(tick.symbol, _tick_to_listener_tick(tick))

    async def _tick_consumer() -> None:
        loop = asyncio.get_running_loop()
        while True:
            tick = await tick_queue.get()
            try:
                await loop.run_in_executor(None, _process_tick, tick)
            except Exception:
                logger.exception("tick consumer failed for %s", tick.symbol)
            finally:
                tick_queue.task_done()

    ws_client.add_callback(_enqueue_tick)
    ws_client.add_health_callback(_on_ws_health)
    ws_client.add_reconnect_callback(lambda: _set_ws_connected(1))
    stop_event = asyncio.Event()
    metrics_task = asyncio.create_task(_periodic_process_metrics())
    n_consumers = max(1, int(os.getenv("MLBOT_TICK_CONSUMER_WORKERS", "2")))
    tick_consumer_tasks = [
        asyncio.create_task(_tick_consumer(), name=f"tick-consumer-{i}")
        for i in range(n_consumers)
    ]
    logger.info(
        "tick dispatch: queue=%d workers=%d",
        tick_queue.maxsize,
        n_consumers,
    )
    try:
        _set_ws_connected(1)
        await ws_client.run(stop_event)
    finally:
        for task in tick_consumer_tasks:
            task.cancel()
        for task in tick_consumer_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if auto_gap_task is not None:
            auto_gap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await auto_gap_task
        metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await metrics_task
        _set_ws_connected(0)
        stop_event.set()
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(async_main())
