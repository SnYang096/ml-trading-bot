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
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

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
    """BPCLiveStrategy 纯规则决策模式（通过 LivePCM 包装）"""
    from src.time_series_model.live.bpc_live_strategy import BPCLiveStrategy
    from src.time_series_model.portfolio.live_pcm import LivePCM

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

    # 包装进 LivePCM（单策略时行为等价直接挂 BPC）
    pcm = LivePCM(
        max_slots=int(os.getenv("MLBOT_MAX_SLOTS", "2")),
    )
    pcm.register("bpc", bpc)

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

    # 给每个 listener 注入 LivePCM 作为 decision_handler
    for sym in symbols:
        listener = manager.get_listener(sym)
        if listener is None:
            continue
        listener.decision_handler = pcm
        listener.order_manager = order_manager
        if trade_size > 0:
            listener.trade_size = trade_size

    logger.info(
        f"[bpc] Initialized via LivePCM: {len(symbols)} symbols, "
        f"bar_minutes={bar_minutes}, window={window_minutes}min, "
        f"archetypes={pcm.registered_archetypes}"
    )
    return manager, pcm


def _setup_three_strategies(
    symbols: List[str],
    storage: StorageManager,
    gap_filler,
    trade_size: float,
):
    """三策略实盘启动 (BPC + ME + FER)"""
    from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
    from src.time_series_model.portfolio.live_pcm import LivePCM
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )

    strategies_root = os.getenv(
        "MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies"
    )
    bar_minutes = int(os.getenv("MLBOT_BPC_BAR_MINUTES", "240"))
    window_minutes = int(os.getenv("MLBOT_BPC_WINDOW_MINUTES", "15"))

    # 创建三个策略实例
    logger.info("🚀 初始化三策略...")

    bpc = GenericLiveStrategy(strategy_name="bpc", strategies_root=strategies_root)
    me = GenericLiveStrategy(strategy_name="me", strategies_root=strategies_root)
    fer = GenericLiveStrategy(strategy_name="fer", strategies_root=strategies_root)

    # 加载配置
    bpc.load_configs()
    me.load_configs()
    fer.load_configs()

    logger.info("✅ 三策略配置加载完成")

    # 创建 PCM 仲裁层 (FER > ME > BPC)
    pcm = LivePCM(
        archetype_priority=["fer", "me", "bpc"],
        max_slots=int(os.getenv("MLBOT_MAX_SLOTS", "2")),
    )
    pcm.register("bpc", bpc)
    pcm.register("me", me)
    pcm.register("fer", fer)

    logger.info(f"✅ PCM 仲裁层初始化: 优先级={pcm.archetype_priority}")

    order_manager = init_order_manager_from_env()

    # 为每个 symbol 创建 IncrementalFeatureComputer
    # 注意：这里需要支持多策略特征计算
    def _make_feature_computer(symbol: str) -> IncrementalFeatureComputer:
        # 使用 BPC 的 archetypes_dir 作为默认
        archetypes_dir = os.path.join(strategies_root, "bpc", "archetypes")
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

    # 给每个 listener 注入 LivePCM 作为 decision_handler
    for sym in symbols:
        listener = manager.get_listener(sym)
        if listener is None:
            continue
        listener.decision_handler = pcm
        listener.order_manager = order_manager
        if trade_size > 0:
            listener.trade_size = trade_size

    logger.info(
        f"✅ 三策略实盘启动完成: {len(symbols)} symbols, "
        f"bar_minutes={bar_minutes}, window={window_minutes}min, "
        f"archetypes={pcm.registered_archetypes}"
    )
    return manager, pcm


def _compute_initial_quantiles(
    decision_handler,
    manager: MultiSymbolManager,
    storage: StorageManager,
) -> None:
    """Warmup 后为 Evidence 模块计算分位数阈值。

    从磁盘加载每个 symbol 的历史 bars + ticks，用
    compute_features_dataframe() 得到完整 DataFrame，然后
    合并所有 symbol 的数据计算 quantiles。

    支持 BPCLiveStrategy 和 LivePCM（自动透传给内部策略）。
    """
    if not hasattr(decision_handler, "set_quantiles") and not hasattr(
        decision_handler, "set_quantiles_from_df"
    ):
        return

    # 使用全部可用历史数据（与 warmup 准备的数据一致）
    # 默认 180 天 = 6 个月，可通过环境变量覆盖
    quantile_lookback_days = int(os.getenv("MLBOT_QUANTILE_LOOKBACK_DAYS", "180"))

    all_dfs: List[pd.DataFrame] = []
    now = pd.Timestamp.now(tz="UTC")

    for symbol, listener in manager.listeners.items():
        try:
            # 加载全部可用 bars（默认 180 天，覆盖 atr_percentile 等长 lookback）
            bar_start = (now - timedelta(days=quantile_lookback_days)).strftime(
                "%Y-%m-%d"
            )
            bar_end = now.strftime("%Y-%m-%d")
            bars_disk = storage.bar_1min.load_range(symbol, bar_start, bar_end)
            if bars_disk.empty:
                logger.warning("[quantiles] %s: bars 为空，跳过", symbol)
                continue

            # 加载 ticks（VPIN 需要 7 天，用 8 天保险）
            tick_start = (now - timedelta(days=8)).strftime("%Y-%m-%d")
            ticks_disk = storage.ticks.load_range(symbol, tick_start, bar_end)

            # 计算完整特征 DataFrame
            fc = listener.feature_computer
            features_df = fc.compute_features_dataframe(
                bars_1min=bars_disk,
                ticks_1min=ticks_disk,
            )
            if features_df is not None and not features_df.empty:
                all_dfs.append(features_df)
                logger.info(
                    "[quantiles] %s: %d rows × %d cols",
                    symbol,
                    len(features_df),
                    len(features_df.columns),
                )
        except Exception as e:
            logger.warning("[quantiles] %s 失败: %s", symbol, e)

    if not all_dfs:
        logger.warning("[quantiles] 无可用数据，跳过 quantile 计算")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    # LivePCM.set_quantiles() 或 BPCLiveStrategy.set_quantiles_from_df()
    if hasattr(decision_handler, "set_quantiles_from_df"):
        decision_handler.set_quantiles_from_df(combined)
    elif hasattr(decision_handler, "set_quantiles"):
        decision_handler.set_quantiles(combined)
    logger.info(
        "[quantiles] 完成: %d symbols, %d 总行数",
        len(all_dfs),
        len(combined),
    )


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

    manager, pcm = _setup_bpc(symbols, storage, gap_filler, trade_size)

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

    # ── 计算 Evidence 分位数阈值（从历史数据）──
    _compute_initial_quantiles(pcm, manager, storage)

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
