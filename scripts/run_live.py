from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Dict, Any, List

import pandas as pd

from src.live_data_stream import StorageManager, GapFiller, MultiSymbolManager
from src.live_data_stream.websocket_client import BinanceWebSocketClient, BinanceTick
from src.live_data_stream.order_manager_factory import init_order_manager_from_env
from src.time_series_model.core.meta_router_core import (
    MetaRouterCore,
    MetaRouterCoreConfig,
)
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)


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


async def main() -> None:
    symbols = _parse_symbols(os.getenv("MLBOT_LIVE_SYMBOLS", "BTCUSDT"))
    if not symbols:
        raise ValueError("No symbols provided. Set MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT")

    storage_base = os.getenv("MLBOT_LIVE_STORAGE_BASE", "data/live_storage")
    use_futures = os.getenv("MLBOT_LIVE_USE_FUTURES", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    warmup_days = int(os.getenv("MLBOT_LIVE_WARMUP_DAYS", "30"))
    trade_size = float(os.getenv("MLBOT_LIVE_TRADE_SIZE", "0.0"))

    storage = StorageManager(base_path=storage_base)
    gap_filler = None
    if os.getenv("MLBOT_LIVE_GAP_FILL", "true").lower() in {"1", "true", "yes"}:
        try:
            import ccxt

            exchange = ccxt.binance(
                {"enableRateLimit": True, "options": {"defaultType": "future"}}
            )
            gap_filler = GapFiller(
                storage_manager=storage,
                exchange=exchange,
                feature_store_dir=os.getenv("MLBOT_FEATURE_STORE_DIR", "feature_store"),
                feature_store_layer=os.getenv("MLBOT_FEATURE_STORE_LAYER", ""),
            )
        except Exception:
            gap_filler = None

    core_cfg = MetaRouterCoreConfig(
        live_config_path=os.getenv(
            "MLBOT_LIVE_CONFIG", "config/nnmultihead/live/meta_router_live_config.yaml"
        ),
        archetype_registry_path=os.getenv(
            "MLBOT_ARCHETYPE_REGISTRY",
            "config/nnmultihead/execution_archetypes.yaml",
        ),
        evidence_quantiles_path=os.getenv("MLBOT_EVIDENCE_QUANTILES_JSON"),
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
        order_manager=order_manager,
    )

    # wire pipeline dependencies
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
