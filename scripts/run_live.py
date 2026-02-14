"""run_live.py — BPC Live 入口

BPCLiveStrategy → 纯规则决策引擎
无 ML 依赖；使用 Gate + Entry Filter + Evidence + Tier

数据管线:
  BinanceWS → MultiSymbolManager → OrderFlowListener → BPC decide → OrderManager
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd

from src.live_data_stream import StorageManager, GapFiller, MultiSymbolManager
from src.live_data_stream.websocket_client import BinanceWebSocketClient, BinanceTick
from src.live_data_stream.order_manager_factory import init_order_manager_from_env
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)

logger = logging.getLogger(__name__)


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in (raw or "").split(",") if s.strip()]


def _tick_to_listener_tick(tick: BinanceTick) -> Any:
    return SimpleNamespace(
        price=float(tick.price),
        size=float(tick.volume),
        side=int(tick.side),
        timestamp=pd.Timestamp(tick.timestamp_ms, unit="ms", tz="UTC"),
        trade_id=tick.trade_id,
    )


def _build_gap_filler(storage: StorageManager):
    """尝试创建 GapFiller（需要 ccxt）"""
    if os.getenv("MLBOT_LIVE_GAP_FILL", "true").lower() not in {"1", "true", "yes"}:
        return None
    try:
        import ccxt

        exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        return GapFiller(
            storage_manager=storage,
            exchange=exchange,
            feature_store_dir=os.getenv("MLBOT_FEATURE_STORE_DIR", "feature_store"),
            feature_store_layer=os.getenv("MLBOT_FEATURE_STORE_LAYER", ""),
        )
    except Exception:
        return None


def _setup_bpc(
    symbols: List[str],
    storage: StorageManager,
    gap_filler,
    trade_size: float,
):
    """BPCLiveStrategy 纯规则决策模式"""
    from src.time_series_model.live.bpc_live_strategy import BPCLiveStrategy

    strategies_root = os.getenv("MLBOT_STRATEGIES_ROOT", "config/strategies")
    bar_minutes = int(os.getenv("MLBOT_BPC_BAR_MINUTES", "240"))
    window_minutes = int(os.getenv("MLBOT_BPC_WINDOW_MINUTES", "15"))

    # Archetypes directory: auto-detect features from gate/evidence/entry_filters
    archetypes_dir = os.path.join(strategies_root, "bpc", "archetypes")

    # 创建 BPC 决策引擎
    bpc = BPCLiveStrategy(
        strategies_root=strategies_root,
        holding_yaml_path=os.getenv("MLBOT_BPC_HOLDING_YAML"),
        trade_size=trade_size,
        primary_timeframe=f"{bar_minutes}T",
        bar_minutes=bar_minutes,
    )
    bpc.load_configs()

    order_manager = init_order_manager_from_env()

    # 为每个 symbol 创建 IncrementalFeatureComputer (archetypes auto-detect)
    def _make_feature_computer(symbol: str) -> IncrementalFeatureComputer:
        return IncrementalFeatureComputer(
            tick_window_minutes=bar_minutes,
            bar_window_size=bar_minutes * 2,
            archetypes_dir=archetypes_dir,
            primary_timeframe=f"{bar_minutes}T",
        )

    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        feature_computer_factory=_make_feature_computer,
        gap_filler=gap_filler,
        feature_compute_interval_minutes=window_minutes,
        orderflow_window_minutes=window_minutes,
        order_manager=order_manager,
    )

    # 给每个 listener 注入 decision_handler
    for sym in symbols:
        listener = manager.get_listener(sym)
        if listener is None:
            continue
        listener.decision_handler = bpc
        listener.order_manager = order_manager
        if trade_size > 0:
            listener.trade_size = trade_size

    logger.info(
        f"[bpc] Initialized: {len(symbols)} symbols, "
        f"bar_minutes={bar_minutes}, window={window_minutes}min"
    )
    return manager


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols = _parse_symbols(os.getenv("MLBOT_LIVE_SYMBOLS", "BTCUSDT"))
    if not symbols:
        raise ValueError("No symbols provided. Set MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT")

    storage_base = os.getenv("MLBOT_LIVE_STORAGE_BASE", "data/live_storage")
    use_futures = os.getenv("MLBOT_LIVE_USE_FUTURES", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    # warmup 只需恢复 memory_window + 时间戳（7 天足够）
    # 特征计算通过 compute_features_batch() 从磁盘直接读取 90+ 天数据
    warmup_days = int(os.getenv("MLBOT_LIVE_WARMUP_DAYS", "7"))
    trade_size = float(os.getenv("MLBOT_LIVE_TRADE_SIZE", "0.0"))

    storage = StorageManager(base_path=storage_base)
    gap_filler = _build_gap_filler(storage)

    logger.info(f"🚀 Starting live trading: symbols={symbols}")

    manager = _setup_bpc(symbols, storage, gap_filler, trade_size)

    # Warmup 与启动质量闸门
    if warmup_days > 0:
        logger.info(f"🔄 Starting warmup: {warmup_days} days...")
        warmup_results = await manager.warmup_all(
            days=warmup_days, use_gap_filler=bool(gap_filler), max_retries=3
        )

        # 根据warmup结果决定启动模式
        decision = manager.decide_startup_mode(warmup_results)
        manager.mode_manager.set_mode(decision)

        logger.info(f"⚡ Startup mode: {decision.mode.value}")
        logger.info(f"   Reason: {decision.reason}")
        logger.info(
            f"   Data: {decision.bar_count} bars, {decision.data_coverage_hours:.2f}h coverage"
        )

        # 策略B：OFFLINE模式不再崩溃，而是继续运行等待实时数据累积
        if decision.mode.value == "OFFLINE":
            logger.warning("⚠️  System starting in OFFLINE mode (Strategy B)")
            logger.warning(
                f"   Got: {decision.bar_count} bars, need >= 120 (2h) for DEGRADED, >= 240 (4h) for NORMAL"
            )
            logger.warning(
                "   Trading is DISABLED. Waiting for real-time data accumulation..."
            )
            logger.warning(
                "   System will auto-upgrade: OFFLINE → DEGRADED (2h) → NORMAL (4h)"
            )

        # DEGRADED模式警告
        if decision.mode.value == "DEGRADED":
            logger.warning("⚠️  System starting in DEGRADED mode")
            logger.warning("   Trading is DISABLED. Observation only.")
            logger.warning(
                "   System will auto-upgrade to NORMAL when data is complete."
            )

    await manager.start_all()

    ws_client = BinanceWebSocketClient(symbols=symbols, use_futures=use_futures)

    def _handle_tick(tick: BinanceTick) -> None:
        listener_tick = _tick_to_listener_tick(tick)
        manager.on_trade_tick(tick.symbol, listener_tick)

    ws_client.add_callback(_handle_tick)
    stop_event = asyncio.Event()
    try:
        await ws_client.run(stop_event)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
