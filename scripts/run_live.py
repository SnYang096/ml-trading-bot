"""run_live.py — Live 实盘入口

GenericLiveStrategy → 配置驱动通用决策引擎
支持 BPC/ME/FER 任意策略，通过 YAML 配置驱动。

数据管线:
  BinanceWS → MultiSymbolManager → OrderFlowListener → GenericLiveStrategy decide → OrderManager
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
from src.time_series_model.live.stats_collector import StatsCollector
from src.time_series_model.live.metrics_exporter import start_metrics_server, METRICS
from pathlib import Path as _Path
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)

logger = logging.getLogger(__name__)


def _load_strategy_timeframe(strategies_root: str, strategy_name: str) -> str:
    """从 meta.yaml 读取策略的 timeframe，缺失时 fallback 到 240T。"""
    import yaml

    meta_path = os.path.join(strategies_root, strategy_name, "meta.yaml")
    try:
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        tf = (meta.get("strategy") or {}).get("timeframe")
        if tf:
            return str(tf)
    except FileNotFoundError:
        logger.warning("meta.yaml 不存在: %s，使用默认 240T", meta_path)
    except Exception as e:
        logger.warning("读取 meta.yaml 失败: %s — %s，使用默认 240T", meta_path, e)
    return "240T"


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
    """尝试创建 GapFiller（需要 ccxt）

    代理兼容：与 WebSocket 一致，自动检测 HTTPS_PROXY/HTTP_PROXY 环境变量。
    TUN 模式下无需设置（透明代理），HTTP 代理模式需设环境变量。
    """
    if os.getenv("MLBOT_LIVE_GAP_FILL", "true").lower() not in {"1", "true", "yes"}:
        return None
    try:
        import ccxt

        # 检测代理（与 websocket_client._detect_proxy 一致）
        proxy_url = None
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            val = os.environ.get(key)
            if val:
                proxy_url = val
                break

        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
            "timeout": 30000,  # 30s timeout
        }
        if proxy_url:
            # ccxt 使用 proxies 字典或 aiohttp_proxy
            config["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
            logger.info(f"📡 GapFiller using proxy: {proxy_url}")

        exchange = ccxt.binance(config)
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
    risk_per_trade: float = 0.0,
):
    """BPC 单策略模式（通过 GenericLiveStrategy + LivePCM 包装）"""
    from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
    from src.time_series_model.portfolio.live_pcm import LivePCM

    strategies_root = os.getenv("MLBOT_STRATEGIES_ROOT", "config/strategies")
    bar_minutes = int(os.getenv("MLBOT_BPC_BAR_MINUTES", "240"))
    window_minutes = int(os.getenv("MLBOT_BPC_WINDOW_MINUTES", "15"))

    # Archetypes directory: auto-detect features from gate/evidence/entry_filters
    archetypes_dir = os.path.join(strategies_root, "bpc", "archetypes")

    # 创建 BPC 决策引擎（使用通用 GenericLiveStrategy）
    bpc = GenericLiveStrategy(
        strategy_name="bpc",
        strategies_root=strategies_root,
        trade_size=trade_size,
        primary_timeframe=f"{bar_minutes}T",
        bar_minutes=bar_minutes,
    )

    # 包装进 LivePCM
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
    risk_per_slot = pcm.constitution.get("risk_per_slot", 0.01)
    per_strategy_limits = pcm.constitution.get("per_strategy_limits", {})
    for sym in symbols:
        listener = manager.get_listener(sym)
        if listener is None:
            continue
        listener.decision_handler = pcm
        listener.order_manager = order_manager
        listener.risk_per_slot = risk_per_slot
        listener.per_strategy_limits = per_strategy_limits
        if trade_size > 0:
            listener.trade_size = trade_size
        if risk_per_trade > 0:
            listener.risk_per_trade = risk_per_trade

    logger.info(
        f"[bpc] Initialized via LivePCM: {len(symbols)} symbols, "
        f"bar_minutes={bar_minutes}, window={window_minutes}min, "
        f"risk_per_slot={risk_per_slot:.2%}, "
        f"archetypes={pcm.registered_archetypes}"
    )
    return manager, pcm


def _ccxt_symbol_to_raw(sym: str) -> str:
    """ccxt symbol → 原始 Binance symbol:  'BTC/USDT:USDT' → 'BTCUSDT'"""
    return sym.replace("/", "").split(":")[0]


def _sync_slots_with_exchange(
    order_manager,
    constitution_exec,
    runtime_st,
    symbols,
) -> None:
    """Slot 与交易所持仓同步: 释放服务端无对应持仓的 stale slot。"""
    active_slots = dict(runtime_st.slots.active)  # copy

    # order_manager 为 None 时无法查询交易所，强制清空所有 slot
    # （没有 order_manager 也无法下单，slot 残留毫无意义只会阻塞信号）
    if order_manager is None:
        if active_slots:
            logger.warning(
                "⚠️ Slot 同步: order_manager=None, 强制清空 %d 个残留 slot: %s",
                len(active_slots),
                list(active_slots.keys()),
            )
            for pid in list(runtime_st.slots.active.keys()):
                constitution_exec.release_slot(
                    st=runtime_st, position_id=pid, reason="stale_sync"
                )
            constitution_exec.save_runtime_state(runtime_st)
        return

    api = getattr(order_manager, "binance_api", None)
    if api is None:
        if active_slots:
            logger.warning(
                "⚠️ Slot 同步: binance_api=None, 强制清空 %d 个残留 slot",
                len(active_slots),
            )
            for pid in list(runtime_st.slots.active.keys()):
                constitution_exec.release_slot(
                    st=runtime_st, position_id=pid, reason="stale_sync"
                )
            constitution_exec.save_runtime_state(runtime_st)
        return

    if not active_slots:
        return

    try:
        exchange_positions = api.get_positions()
    except Exception as e:
        logger.warning("Slot 同步: 查询 Binance 持仓失败: %s", e)
        return

    # get_positions() 已经只返回 contracts!=0 的持仓
    # symbol 是 ccxt 格式 (BTC/USDT:USDT)，需转换为原始格式 (BTCUSDT)
    live_symbols = set()
    for p in exchange_positions:
        raw_sym = _ccxt_symbol_to_raw(p.get("symbol", ""))
        if raw_sym:
            live_symbols.add(raw_sym)

    logger.info(
        "Slot 同步: 服务端持仓 symbols=%s, 本地 active slots=%d",
        live_symbols or "{}",
        len(active_slots),
    )

    # 服务端无持仓的 slot → 释放
    stale_count = 0
    for pid, rec in active_slots.items():
        slot_symbol = getattr(rec, "symbol", None) or ""
        if slot_symbol and slot_symbol not in live_symbols:
            constitution_exec.release_slot(
                st=runtime_st, position_id=pid, reason="stale_sync"
            )
            stale_count += 1
            logger.warning(
                "🗑️ 释放 stale slot: %s (%s) — 服务端无持仓", pid, slot_symbol
            )

    if stale_count > 0:
        constitution_exec.save_runtime_state(runtime_st)
        logger.info("✅ Slot 同步: 释放 %d 个 stale slot", stale_count)
    else:
        logger.info(
            "✅ Slot 同步: 无 stale (%d 个 slot 均有服务端持仓)", len(active_slots)
        )


def _setup_three_strategies(
    symbols: List[str],
    storage: StorageManager,
    gap_filler,
    trade_size: float,
    risk_per_trade: float = 0.0,
):
    """三策略实盘启动 (BPC + ME + FER) — 多时间框架

    时间框架:
      BPC: 4H (240T)
      FER: 4H (240T)
      ME:  1H (60T)

    数据管线:
      同一组 1min bars/ticks → 分别重采样为 240T/60T → 各策略拿对应 timeframe 的特征
    """
    from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
    from src.time_series_model.portfolio.live_pcm import LivePCM
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )
    from src.time_series_model.live.live_feature_plan import (
        extract_features_from_archetypes,
    )

    strategies_root = os.getenv(
        "MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies"
    )
    window_minutes = int(os.getenv("MLBOT_BPC_WINDOW_MINUTES", "15"))

    # ── 0. 从 pcm_regime.yaml 读取 enabled_archetypes ──
    import yaml as _yaml

    config_root = os.path.join(strategies_root, "..")
    pcm_regime_path = os.getenv(
        "MLBOT_PCM_REGIME_CONFIG",
        os.path.join(config_root, "pcm_regime.yaml"),
    )
    _ALL_ARCHETYPES = ["bpc", "me-long", "fer", "lv"]
    try:
        with open(pcm_regime_path, "r", encoding="utf-8") as _f:
            _pcm_cfg = _yaml.safe_load(_f)
        enabled_archetypes = [
            a.lower() for a in (_pcm_cfg.get("enabled_archetypes") or _ALL_ARCHETYPES)
        ]
    except Exception as _e:
        logger.warning(
            "读取 pcm_regime.yaml enabled_archetypes 失败: %s, 默认全部启用", _e
        )
        enabled_archetypes = _ALL_ARCHETYPES
    logger.info("📋 enabled_archetypes (来自 pcm_regime.yaml): %s", enabled_archetypes)

    # ── 1. 从 meta.yaml 读取各策略 timeframe (不再硬编码) ──
    tf_bpc = _load_strategy_timeframe(strategies_root, "bpc")  # 默认 240T
    tf_me = _load_strategy_timeframe(strategies_root, "me-long")  # 默认 60T
    tf_fer = _load_strategy_timeframe(strategies_root, "fer")  # 默认 240T
    tf_lv = _load_strategy_timeframe(strategies_root, "lv")  # 默认 15T

    def _tf_to_bar_minutes(tf: str) -> int:
        """'240T' → 240, '60T' → 60"""
        return int(tf.replace("T", ""))

    bar_minutes_bpc = _tf_to_bar_minutes(tf_bpc)
    bar_minutes_me = _tf_to_bar_minutes(tf_me)
    bar_minutes_fer = _tf_to_bar_minutes(tf_fer)
    bar_minutes_lv = _tf_to_bar_minutes(tf_lv)

    # ── 初始化已启用的策略对象 ──
    _strategy_map = {}
    _tf_map = {}
    if "bpc" in enabled_archetypes:
        _strategy_map["bpc"] = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=strategies_root,
            trade_size=trade_size,
            primary_timeframe=tf_bpc,
            bar_minutes=bar_minutes_bpc,
        )
        _tf_map["bpc"] = tf_bpc
    if "me-long" in enabled_archetypes:
        _strategy_map["me-long"] = GenericLiveStrategy(
            strategy_name="me-long",
            strategies_root=strategies_root,
            trade_size=trade_size,
            primary_timeframe=tf_me,
            bar_minutes=bar_minutes_me,
        )
        _tf_map["me-long"] = tf_me
    if "fer" in enabled_archetypes:
        _strategy_map["fer"] = GenericLiveStrategy(
            strategy_name="fer",
            strategies_root=strategies_root,
            trade_size=trade_size,
            primary_timeframe=tf_fer,
            bar_minutes=bar_minutes_fer,
        )
        _tf_map["fer"] = tf_fer
    if "lv" in enabled_archetypes:
        _strategy_map["lv"] = GenericLiveStrategy(
            strategy_name="lv",
            strategies_root=strategies_root,
            trade_size=trade_size,
            primary_timeframe=tf_lv,
            bar_minutes=bar_minutes_lv,
        )
        _tf_map["lv"] = tf_lv

    # 将1个变量方便后续使用
    bpc = _strategy_map.get("bpc")
    me = _strategy_map.get("me-long")
    fer = _strategy_map.get("fer")
    lv = _strategy_map.get("lv")

    logger.info("✅ 策略初始化完成: %s", list(_strategy_map.keys()))

    # ── 2. 创建 PCM 仲裁层 (注册策略 + timeframe 绑定) ──
    pcm = LivePCM(
        archetype_priority=["LV", "FER", "ME-LONG", "BPC"],
        regime_config_path=pcm_regime_path,
        constitution_yaml=os.getenv(
            "MLBOT_CONSTITUTION_YAML",
            os.path.join(config_root, "constitution", "constitution.yaml"),
        ),
    )
    for _name, _strat in _strategy_map.items():
        pcm.register(_name, _strat, timeframe=_tf_map[_name])

    logger.info(f"✅ PCM 仲裁层初始化: 优先级={pcm.archetype_priority}")

    order_manager = init_order_manager_from_env()

    # ── 3. 创建特征计算器 (per-symbol, per-timeframe) ──
    bpc_archetypes = os.path.join(strategies_root, "bpc", "archetypes")
    fer_archetypes = os.path.join(strategies_root, "fer", "archetypes")
    me_archetypes = os.path.join(strategies_root, "me-long", "archetypes")
    lv_archetypes = os.path.join(strategies_root, "lv", "archetypes")

    # 预提取 FER 特征集 (用于合并到 4H FC)
    fer_extra_feat_set = set()
    fer_extra_feat_nodes = []
    try:
        fer_extra_feat_set, fer_extra_feat_nodes = extract_features_from_archetypes(
            fer_archetypes
        )
        logger.info(
            "  FER features: %d columns, %d nodes",
            len(fer_extra_feat_set),
            len(fer_extra_feat_nodes),
        )
    except Exception as e:
        logger.warning("  FER feature extraction failed: %s", e)

    def _make_feature_computer_4h(symbol: str) -> IncrementalFeatureComputer:
        """4H FC: BPC + FER 合并特征集 (timeframe 从 meta.yaml 读取)"""
        fc = IncrementalFeatureComputer(
            tick_window_minutes=bar_minutes_bpc,
            bar_window_size=bar_minutes_bpc * 2,
            archetypes_dir=bpc_archetypes,
            primary_timeframe=tf_bpc,
        )
        # 合并 FER 特征到 4H FC
        if fer_extra_feat_set:
            fc.live_feature_set |= fer_extra_feat_set
            merged_nodes = sorted(
                set(fc.live_feature_nodes) | set(fer_extra_feat_nodes)
            )
            fc.live_feature_nodes = merged_nodes
        return fc

    def _make_feature_computer_me(symbol: str) -> IncrementalFeatureComputer:
        """ME FC: timeframe 从 meta.yaml 读取"""
        return IncrementalFeatureComputer(
            tick_window_minutes=bar_minutes_me,
            bar_window_size=bar_minutes_me * 2,
            archetypes_dir=me_archetypes,
            primary_timeframe=tf_me,
        )

    def _make_feature_computer_lv(symbol: str) -> IncrementalFeatureComputer:
        """LV FC: 15T timeframe 从 meta.yaml 读取"""
        return IncrementalFeatureComputer(
            tick_window_minutes=bar_minutes_lv,
            bar_window_size=bar_minutes_lv * 2,
            archetypes_dir=lv_archetypes,
            primary_timeframe=tf_lv,
        )

    # ── 4. MultiSymbolManager (primary FC = 4H) ──
    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        feature_computer_factory=_make_feature_computer_4h,
        gap_filler=gap_filler,
        feature_compute_interval_minutes=window_minutes,
        orderflow_window_minutes=window_minutes,
        order_manager=order_manager,
    )

    # ── 5. 注入 decision_handler + 额外 FC + stats_collector ──
    # 监控统计收集器 (始终启用，自动清理默认关闭)
    stats_db_path = os.path.join(
        os.getenv("MLBOT_LIVE_BASE", "live/highcap"),
        "data",
        "db",
        "live_monitor.db",
    )
    auto_cleanup = os.getenv("MLBOT_STATS_AUTO_CLEANUP", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    stats_collector = StatsCollector(db_path=stats_db_path, auto_cleanup=auto_cleanup)
    pcm.stats_collector = stats_collector
    logger.info(
        "✅ StatsCollector 启用: %s (auto_cleanup=%s)", stats_db_path, auto_cleanup
    )

    # 创建 ConstitutionExecutor + RuntimeState
    constitution_yaml_path = os.getenv(
        "MLBOT_CONSTITUTION_YAML",
        os.path.join(config_root, "constitution", "constitution.yaml"),
    )
    constitution_exec = ConstitutionExecutor(constitution_yaml=constitution_yaml_path)
    runtime_st = constitution_exec.load_runtime_state()
    logger.info("✅ ConstitutionExecutor 初始化: %s", constitution_yaml_path)

    # ── 启动时 slot 同步: 从 Binance 查真实持仓，清理残留 stale slot ──
    _sync_slots_with_exchange(order_manager, constitution_exec, runtime_st, symbols)

    for sym in symbols:
        listener = manager.get_listener(sym)
        if listener is None:
            continue
        listener.decision_handler = pcm
        listener.order_manager = order_manager
        listener.constitution_executor = constitution_exec
        listener.runtime_state = runtime_st
        # 从宪法注入 risk_per_slot + per_strategy_limits
        risk_per_slot = pcm.constitution.get("risk_per_slot", 0.01)
        per_strategy_limits = pcm.constitution.get("per_strategy_limits", {})
        listener.risk_per_slot = risk_per_slot
        listener.per_strategy_limits = per_strategy_limits
        if trade_size > 0:
            listener.trade_size = trade_size
        if risk_per_trade > 0:
            listener.risk_per_trade = risk_per_trade
        # 注入监控统计收集器
        listener.stats_collector = stats_collector
        # 注入 ME FC + LV FC (各自独立 timeframe)
        _extra_fcs = {}
        if me is not None:
            _extra_fcs[tf_me] = _make_feature_computer_me(sym)
        if lv is not None:
            _extra_fcs[tf_lv] = _make_feature_computer_lv(sym)
        listener.extra_feature_computers = _extra_fcs

    logger.info(
        f"✅ 三策略实盘启动完成: {len(symbols)} symbols, "
        f"BPC={tf_bpc}, FER={tf_fer}, ME={tf_me}, "
        f"window={window_minutes}min, archetypes={pcm.registered_archetypes}"
    )
    return manager, pcm


def _set_quantiles_per_symbol(
    strategy,
    per_symbol_dfs: List[pd.DataFrame],
    arch_name: str,
    tf: str | None,
) -> None:
    """Per-symbol quantile 计算 + 中位数融合。

    对每个 symbol 的 DataFrame 独立计算 quantile 阈值，
    然后取所有 symbol 的中位数作为最终阈值。
    避免跨 symbol 分布污染（BTC VPIN >> ADA 等）。
    """
    import numpy as np

    if not per_symbol_dfs or not hasattr(strategy, "set_quantiles"):
        return

    # 1. 先用第一个 DataFrame 计算一次（保证 _quantiles 返回结构一致）
    strategy.set_quantiles(per_symbol_dfs[0])

    if len(per_symbol_dfs) <= 1:
        logger.info(
            "[quantiles] %s (timeframe=%s): 1 symbol, %d 行",
            arch_name,
            tf or "default",
            len(per_symbol_dfs[0]),
        )
        return

    # 2. 为每个 symbol 独立计算 quantile 阈值
    all_quantiles: List[Dict[str, Dict[str, float]]] = []
    for df in per_symbol_dfs:
        strategy.set_quantiles(df)
        if hasattr(strategy, "_quantiles") and strategy._quantiles:
            # deep copy
            all_quantiles.append({k: dict(v) for k, v in strategy._quantiles.items()})

    if not all_quantiles:
        logger.warning("[quantiles] %s: 无有效 quantiles", arch_name)
        return

    # 3. 取中位数融合
    merged: Dict[str, Dict[str, float]] = {}
    # 收集所有出现过的 feature keys
    all_feat_keys: set = set()
    for q in all_quantiles:
        all_feat_keys |= q.keys()

    for feat_key in all_feat_keys:
        merged[feat_key] = {}
        # 收集该特征所有分位点 keys
        q_keys: set = set()
        for q in all_quantiles:
            if feat_key in q:
                q_keys |= q[feat_key].keys()
        for q_key in q_keys:
            vals = []
            for q in all_quantiles:
                if feat_key in q and q_key in q[feat_key]:
                    vals.append(q[feat_key][q_key])
            if vals:
                merged[feat_key][q_key] = float(np.median(vals))

    strategy._quantiles = merged
    n_feats = len(merged)
    total_rows = sum(len(df) for df in per_symbol_dfs)
    logger.info(
        "[quantiles] %s (timeframe=%s): %d symbols, %d features, %d 总行数 (按 symbol 中位数融合)",
        arch_name,
        tf or "default",
        len(per_symbol_dfs),
        n_feats,
        total_rows,
    )


def _compute_initial_quantiles(
    decision_handler,
    manager: MultiSymbolManager,
    storage: StorageManager,
) -> None:
    """Warmup 后为 Evidence 模块计算分位数阈值。

    从磁盘加载每个 symbol 的历史 bars + ticks，用
    compute_features_dataframe() 得到完整 DataFrame，然后
    合并所有 symbol 的数据计算 quantiles。

    支持多时间框架:
      - 主 FC 计算 primary_timeframe quantiles (4H) → BPC/FER
      - 额外 FC 计算 extra_timeframe quantiles (1H) → ME
      - 按 timeframe 分别设置给对应策略
    """
    if not hasattr(decision_handler, "set_quantiles") and not hasattr(
        decision_handler, "set_quantiles_from_df"
    ):
        return

    quantile_lookback_days = int(os.getenv("MLBOT_QUANTILE_LOOKBACK_DAYS", "180"))

    # 按 timeframe 收集 feature DataFrames
    tf_dfs: Dict[str, List[pd.DataFrame]] = {}  # timeframe → [df, ...]
    now = pd.Timestamp.now(tz="UTC")

    for symbol, listener in manager.listeners.items():
        try:
            bar_start = (now - timedelta(days=quantile_lookback_days)).strftime(
                "%Y-%m-%d"
            )
            bar_end = now.strftime("%Y-%m-%d")
            bars_disk = storage.bar_1min.load_range(symbol, bar_start, bar_end)
            if bars_disk.empty:
                logger.warning("[quantiles] %s: bars 为空，跳过", symbol)
                continue

            # 注入 _symbol 列 — funding_rate / OI join 等特征需要
            if "_symbol" not in bars_disk.columns:
                bars_disk["_symbol"] = symbol

            tick_start = (now - timedelta(days=8)).strftime("%Y-%m-%d")
            ticks_disk = storage.ticks.load_range(symbol, tick_start, bar_end)

            # Primary timeframe
            fc = listener.feature_computer
            primary_tf = fc.primary_timeframe or "240T"
            features_df = fc.compute_features_dataframe(
                bars_1min=bars_disk,
                ticks_1min=ticks_disk,
            )
            if features_df is not None and not features_df.empty:
                tf_dfs.setdefault(primary_tf, []).append(features_df)
                logger.info(
                    "[quantiles] %s/%s: %d rows × %d cols",
                    symbol,
                    primary_tf,
                    len(features_df),
                    len(features_df.columns),
                )

            # Extra timeframes (e.g., 1H for ME)
            for tf, extra_fc in getattr(
                listener, "extra_feature_computers", {}
            ).items():
                try:
                    extra_df = extra_fc.compute_features_dataframe(
                        bars_1min=bars_disk,
                        ticks_1min=ticks_disk,
                        primary_timeframe=tf,
                    )
                    if extra_df is not None and not extra_df.empty:
                        tf_dfs.setdefault(tf, []).append(extra_df)
                        logger.info(
                            "[quantiles] %s/%s: %d rows × %d cols",
                            symbol,
                            tf,
                            len(extra_df),
                            len(extra_df.columns),
                        )
                except Exception as e:
                    logger.warning("[quantiles] %s/%s 失败: %s", symbol, tf, e)

        except Exception as e:
            logger.warning("[quantiles] %s 失败: %s", symbol, e)

    if not tf_dfs:
        logger.warning("[quantiles] 无可用数据，跳过 quantile 计算")
        return

    # 按 timeframe 分别设置 quantiles
    # 🐛 Fix: 按 symbol 分别计算 quantile 阈值再取中位数
    #   之前 concat 所有 symbol 导致跨 symbol 分布污染（BTC 的 VPIN >> ADA）
    #   某些 symbol 的 evidence 特征值始终落在极端分位数，evidence 永远 1.0
    if hasattr(decision_handler, "_strategy_timeframes") and hasattr(
        decision_handler, "_strategies"
    ):
        # LivePCM: 按策略的 timeframe 分别设置
        for arch_name, strategy in decision_handler._strategies.items():
            tf = decision_handler._strategy_timeframes.get(arch_name)
            dfs = tf_dfs.get(tf, [])
            if not dfs:
                # fallback: 使用任意可用数据
                for v in tf_dfs.values():
                    if v:
                        dfs = v
                        break
            if dfs and hasattr(strategy, "set_quantiles"):
                _set_quantiles_per_symbol(strategy, dfs, arch_name, tf)
    else:
        # 单策略: 使用所有可用数据
        all_dfs: List[pd.DataFrame] = []
        for dfs in tf_dfs.values():
            all_dfs.extend(dfs)
        if all_dfs:
            handler = decision_handler
            if hasattr(handler, "set_quantiles_from_df"):
                _set_quantiles_per_symbol(handler, all_dfs, "single", None)
            elif hasattr(handler, "set_quantiles"):
                _set_quantiles_per_symbol(handler, all_dfs, "single", None)
            logger.info(
                "[quantiles] 完成: %d symbols",
                len(all_dfs),
            )


# ====================================================================
# Retrain trigger check (runs every 6h inside live process)
# ====================================================================


def _run_retrain_check() -> None:
    """Synchronous function executed in thread pool by _periodic_retrain_check.

    Reuses monitor_retrain.check_triggers() to evaluate 5 retrain conditions,
    then updates Prometheus Gauges so Grafana can display retrain signals.
    """
    import yaml as _yaml

    from scripts.monitor_retrain import (
        check_triggers,
        compute_consecutive_losses,
        compute_live_sharpe,
        days_since_last_train,
        get_baseline_sharpe,
        get_last_research,
        load_live_trades,
    )

    project_root = _Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "research_pipeline.yaml"
    if not config_path.exists():
        logger.warning("[retrain-check] config not found: %s", config_path)
        return

    cfg = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    triggers_cfg = cfg.get("retrain_triggers", {})
    history_dir = project_root / cfg.get("output", {}).get("history_dir", "results")

    # DB path: live deployment or dev
    db_path = project_root / "data" / "order_management.db"
    if not db_path.exists():
        alt = project_root / "live" / "highcap" / "data" / "order_management.db"
        if alt.exists():
            db_path = alt
    log_dir = project_root / "data"

    strategy_names = list(cfg.get("strategies", {}).keys())
    if not strategy_names:
        strategy_names = ["bpc", "fer", "me-long"]

    for strat in strategy_names:
        try:
            report = get_last_research(history_dir, strat)
            trades = load_live_trades(db_path, log_dir, strategy=strat, days=90)
            result = check_triggers(strat, trades, report, triggers_cfg)

            # Update Prometheus Gauges
            triggered = 1 if result.get("triggered") else 0
            METRICS.retrain_triggered.labels(strategy=strat).set(triggered)
            METRICS.retrain_trigger_count.labels(strategy=strat).set(
                result.get("trigger_count", 0)
            )

            details = result.get("details", {})
            live_sharpe = details.get("live_sharpe_30d", 0.0)
            METRICS.sharpe_live_30d.labels(strategy=strat).set(
                live_sharpe if live_sharpe is not None else 0.0
            )

            sharpe_ratio = details.get("sharpe_ratio")
            METRICS.sharpe_decay_ratio.labels(strategy=strat).set(
                sharpe_ratio if sharpe_ratio is not None else 0.0
            )

            consec = details.get("consecutive_losses", 0)
            METRICS.consecutive_losses.labels(strategy=strat).set(
                consec if consec is not None else 0
            )

            days_train = details.get("days_since_last_train", 9999)
            METRICS.days_since_last_train.labels(strategy=strat).set(
                days_train if days_train is not None else 9999
            )

            # Alpha decay: from leading_indicator_decay sub-result
            leading = details.get("leading_indicator_decay", {})
            max_decay = (
                leading.get("max_decay", 0.0) if isinstance(leading, dict) else 0.0
            )
            METRICS.alpha_decay_max.labels(strategy=strat).set(
                max_decay if max_decay is not None else 0.0
            )

            status = "TRIGGERED" if triggered else "OK"
            logger.info(
                "[retrain-check] %s: %s (triggers=%d, sharpe_30d=%.3f, consec=%d, days=%d, decay=%.2f)",
                strat,
                status,
                result.get("trigger_count", 0),
                live_sharpe or 0.0,
                consec or 0,
                days_train or 0,
                max_decay or 0.0,
            )
        except Exception as exc:
            logger.warning("[retrain-check] %s failed: %s", strat, exc)


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
    risk_per_trade = float(os.getenv("MLBOT_RISK_PER_TRADE", "0.0"))
    if risk_per_trade > 0:
        logger.info(f"💰 风险仓位模式: 每笔风险=${risk_per_trade}")
    elif trade_size > 0:
        logger.info(
            f"⚠️  固定数量模式: trade_size={trade_size} (建议改用 MLBOT_RISK_PER_TRADE)"
        )

    storage = StorageManager(base_path=storage_base)
    gap_filler = _build_gap_filler(storage)

    logger.info(f"🚀 Starting live trading: symbols={symbols}")

    # ── Prometheus metrics server ──
    metrics_port = int(os.getenv("MLBOT_METRICS_PORT", "9090"))
    start_metrics_server(port=metrics_port)

    # 选择启动模式: bpc (单策略) 或 three_strategies (三策略多时间框架)
    live_mode = os.getenv("MLBOT_LIVE_MODE", "bpc")
    if live_mode == "three_strategies":
        manager, pcm = _setup_three_strategies(
            symbols, storage, gap_filler, trade_size, risk_per_trade
        )
    else:
        manager, pcm = _setup_bpc(
            symbols, storage, gap_filler, trade_size, risk_per_trade
        )

    # Warmup 与启动质量闸门
    if warmup_days > 0:
        logger.info(f"🔄 Starting warmup: {warmup_days} days...")
        warmup_results = await manager.warmup_all(
            days=warmup_days, use_gap_filler=bool(gap_filler), max_retries=3
        )

        # 根据warmup结果决定启动模式
        decision = manager.decide_startup_mode(warmup_results)
        manager.mode_manager.set_mode(decision)

        # 更新 Prometheus 系统模式指标
        _MODE_VALUES = {"OFFLINE": 0, "DEGRADED": 1, "NORMAL": 2}
        METRICS.system_mode.set(_MODE_VALUES.get(decision.mode.value, 0))

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
                "   Auto-upgrade enabled: OFFLINE → DEGRADED (2h) → NORMAL (4h)"
            )

        # DEGRADED模式警告
        if decision.mode.value == "DEGRADED":
            remaining = max(0, 240 - decision.bar_count)
            logger.warning("⚠️  System starting in DEGRADED mode")
            logger.warning("   Trading is DISABLED. Observation only.")
            logger.warning(
                f"   Auto-upgrade enabled: will upgrade to NORMAL after {remaining} realtime 1min bars (~{remaining}min)"
            )

    # ── 后台补数据 task (Vision 重试) ──
    bg_gap_task = None
    if gap_filler and gap_filler._pending_vision_gaps:
        pending_count = len(gap_filler._pending_vision_gaps)
        logger.info(f"📦 {pending_count} 个 gap 待后台 Vision 补齐，每 15min 重试一次")

        async def _background_vision_retry() -> None:
            """后台定期重试 Binance Vision 下载，直到所有 gap 补齐"""
            interval = 15 * 60  # 15min
            while True:
                try:
                    await asyncio.sleep(interval)
                    remaining = len(gap_filler._pending_vision_gaps)
                    if remaining == 0:
                        logger.info("✅ 后台补数据完成: 所有 Vision gap 已填充")
                        break
                    logger.info(f"📦 后台 Vision 重试: {remaining} 个 gap 待处理...")
                    loop = asyncio.get_running_loop()
                    all_done = await loop.run_in_executor(
                        None, gap_filler.retry_pending_gaps
                    )
                    if all_done:
                        logger.info("✅ 后台补数据完成: 所有 Vision gap 已填充")
                        break
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning("后台 Vision 重试异常: %s", exc)
                    await asyncio.sleep(60)

        bg_gap_task = asyncio.create_task(_background_vision_retry())

    # ── 计算 Evidence 分位数阈值（从历史数据）──
    _compute_initial_quantiles(pcm, manager, storage)

    await manager.start_all()

    ws_client = BinanceWebSocketClient(symbols=symbols, use_futures=use_futures)

    def _handle_tick(tick: BinanceTick) -> None:
        listener_tick = _tick_to_listener_tick(tick)
        manager.on_trade_tick(tick.symbol, listener_tick)
        # 更新 WebSocket 连接状态指标
        METRICS.ws_connected.labels(symbol=tick.symbol).set(1)

    ws_client.add_callback(_handle_tick)

    # ── 定期获取市场数据 & 账户数据 & 连接状态 ──
    async def _periodic_market_update() -> None:
        """30s 一次获取 Binance 市场数据 (funding rate / mark price / OI / account)"""
        interval = int(os.getenv("MLBOT_MARKET_DATA_INTERVAL", "30"))
        _mode_map = {"OFFLINE": 0, "DEGRADED": 1, "NORMAL": 2}
        while True:
            try:
                await asyncio.sleep(interval)
                # 在线程池中执行同步 HTTP 请求，避免阻塞事件循环
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, METRICS.update_market_data, symbols)
                await loop.run_in_executor(None, METRICS.update_account_data)
                # 更新系统模式指标
                cur_mode = manager.mode_manager.get_current_mode()
                METRICS.system_mode.set(_mode_map.get(cur_mode.value, 0))
                # 更新 WebSocket 连接状态
                ws_health = ws_client.get_health_status()
                ws_ok = 1 if ws_health.get("status") in ("healthy", "degraded") else 0
                for sym in symbols:
                    METRICS.ws_connected.labels(symbol=sym).set(ws_ok)
                # 更新数据新鲜度（距上次特征计算的秒数）
                now = pd.Timestamp.now(tz="UTC")
                for sym in symbols:
                    listener = manager.get_listener(sym)
                    if listener and listener.last_feature_compute_time:
                        age = (now - listener.last_feature_compute_time).total_seconds()
                        METRICS.last_bar_age.labels(symbol=sym).set(age)
                # 更新 per-strategy slot 指标
                _first_listener = manager.get_listener(symbols[0]) if symbols else None
                if _first_listener:
                    _rs = getattr(_first_listener, "runtime_state", None)
                    _psl = getattr(_first_listener, "per_strategy_limits", {}) or {}
                    _slots_cfg = (pcm.constitution or {}).get("slots", {})
                    _global_max = int(_slots_cfg.get("slot_count", 2))
                    METRICS.update_slot_metrics(_rs, _psl, _global_max)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("市场数据更新异常: %s", exc)
                await asyncio.sleep(60)  # 出错后等 60s 再重试

    market_task = asyncio.create_task(_periodic_market_update())

    # ── 每日刷新 Funding Rate / OI parquet 数据 ──
    async def _daily_funding_oi_refresh() -> None:
        """12h 一次增量刷新 funding_rate / OI parquet (Binance 公开 API)"""
        interval = 12 * 3600  # 12 小时
        await asyncio.sleep(interval)  # 首次延迟: 启动时 start_live.sh 已刷新
        while True:
            try:
                from scripts.refresh_funding_oi_data import refresh_all

                logger.info("📊 定时刷新 Funding/OI 数据...")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, refresh_all, symbols, "data", 30
                )
                logger.info(
                    "✅ Funding/OI 刷新完成: FR=%d files, OI=%d files",
                    result["funding_rate_files"],
                    result["oi_files"],
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("定时 Funding/OI 刷新失败: %s", exc)
            await asyncio.sleep(interval)

    funding_oi_task = asyncio.create_task(_daily_funding_oi_refresh())

    # ── 定期重训触发检测 (6h) ──
    async def _periodic_retrain_check() -> None:
        """6h 一次检查重训触发条件, 更新 Prometheus"""
        await asyncio.sleep(300)  # 首次延迟 5min 等系统稳定
        while True:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _run_retrain_check)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("重训检测异常: %s", exc)
            await asyncio.sleep(6 * 3600)

    retrain_task = asyncio.create_task(_periodic_retrain_check())

    stop_event = asyncio.Event()
    try:
        await ws_client.run(stop_event)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        market_task.cancel()
        funding_oi_task.cancel()
        retrain_task.cancel()
        if bg_gap_task:
            bg_gap_task.cancel()
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
