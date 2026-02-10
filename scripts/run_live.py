"""run_live.py — 统一 Live 入口

支持两种模式（通过 MLBOT_LIVE_MODE 环境变量选择）：

  meta_router (默认):
    MetaRouterCore + ConstitutionExecutor → 多 archetype 决策
    需要 ML 模型预测 (pred_dir_prob 等)

  bpc:
    BPCLiveStrategy → 纯规则决策引擎
    无 ML 依赖；使用 Gate + Entry Filter + Evidence + Tier

两种模式共享同一个数据管线:
  BinanceWS → MultiSymbolManager → OrderFlowListener → 决策 → OrderManager
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


# ────────────────────────────────────────────────────
# Mode: meta_router
# ────────────────────────────────────────────────────


def _setup_meta_router_mode(
    symbols: List[str],
    storage: StorageManager,
    gap_filler,
    trade_size: float,
):
    """MetaRouterCore + ConstitutionExecutor 模式"""
    from src.time_series_model.core.meta_router_core import (
        MetaRouterCore,
        MetaRouterCoreConfig,
    )
    from src.time_series_model.core.constitution.constitution_executor import (
        ConstitutionExecutor,
    )
    from src.time_series_model.live.meta_router_config import (
        load_meta_router_live_config,
    )

    live_config_path = os.getenv(
        "MLBOT_LIVE_CONFIG_YAML",
        "config/live/live_config_defaults.yaml",
    )
    live_cfg = load_meta_router_live_config(config_path=live_config_path)
    window_minutes = live_cfg.window_minutes

    core_cfg = MetaRouterCoreConfig(
        strategies_root=os.getenv("MLBOT_STRATEGIES_ROOT", "config/strategies"),
        evidence_quantiles_path=os.getenv("MLBOT_EVIDENCE_QUANTILES_JSON"),
        live_config_path=live_config_path,
    )
    meta_router = MetaRouterCore(core_cfg)

    constitution_yaml = os.getenv(
        "MLBOT_CONSTITUTION_YAML", "config/constitution/constitution.yaml"
    )
    executor = ConstitutionExecutor(constitution_yaml=constitution_yaml)
    runtime_state = executor.load_runtime_state()
    order_manager = init_order_manager_from_env()

    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        gap_filler=gap_filler,
        feature_compute_interval_minutes=window_minutes,
        orderflow_window_minutes=window_minutes,
        order_manager=order_manager,
    )

    for sym in symbols:
        listener = manager.get_listener(sym)
        if listener is None:
            continue
        listener.meta_router_core = meta_router
        listener.constitution_executor = executor
        listener.runtime_state = runtime_state
        listener.order_manager = order_manager
        if trade_size > 0:
            listener.trade_size = trade_size

    logger.info(
        f"[meta_router] Initialized: {len(symbols)} symbols, "
        f"window={window_minutes}min, archetypes={live_cfg.enabled_archetypes}"
    )
    return manager


# ────────────────────────────────────────────────────
# Mode: bpc
# ────────────────────────────────────────────────────


def _setup_bpc_mode(
    symbols: List[str],
    storage: StorageManager,
    gap_filler,
    trade_size: float,
):
    """BPCLiveStrategy 纯规则决策模式"""
    from src.time_series_model.live.bpc_live_strategy import BPCLiveStrategy

    strategies_root = os.getenv("MLBOT_STRATEGIES_ROOT", "config/strategies")
    bpc_feature_plan = os.getenv(
        "MLBOT_BPC_FEATURE_PLAN_YAML",
        "config/live/live_feature_plan.yaml",
    )
    bar_minutes = int(os.getenv("MLBOT_BPC_BAR_MINUTES", "240"))
    window_minutes = int(os.getenv("MLBOT_BPC_WINDOW_MINUTES", "15"))

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

    # 为每个 symbol 创建带 BPC feature plan 的 IncrementalFeatureComputer
    def _make_feature_computer(symbol: str) -> IncrementalFeatureComputer:
        return IncrementalFeatureComputer(
            tick_window_minutes=bar_minutes,
            bar_window_size=bar_minutes * 2,
            live_feature_plan_path=bpc_feature_plan,
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


# ────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols = _parse_symbols(os.getenv("MLBOT_LIVE_SYMBOLS", "BTCUSDT"))
    if not symbols:
        raise ValueError("No symbols provided. Set MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT")

    mode = os.getenv("MLBOT_LIVE_MODE", "meta_router").strip().lower()
    storage_base = os.getenv("MLBOT_LIVE_STORAGE_BASE", "data/live_storage")
    use_futures = os.getenv("MLBOT_LIVE_USE_FUTURES", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    warmup_days = int(os.getenv("MLBOT_LIVE_WARMUP_DAYS", "30"))
    trade_size = float(os.getenv("MLBOT_LIVE_TRADE_SIZE", "0.0"))

    storage = StorageManager(base_path=storage_base)
    gap_filler = _build_gap_filler(storage)

    logger.info(f"🚀 Starting live trading: mode={mode}, symbols={symbols}")

    if mode == "bpc":
        manager = _setup_bpc_mode(symbols, storage, gap_filler, trade_size)
    elif mode in {"meta_router", "meta"}:
        manager = _setup_meta_router_mode(symbols, storage, gap_filler, trade_size)
    else:
        raise ValueError(
            f"Unknown MLBOT_LIVE_MODE={mode!r}. Use 'meta_router' or 'bpc'."
        )

    if warmup_days > 0:
        await manager.warmup_all(days=warmup_days, use_gap_filler=bool(gap_filler))

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
