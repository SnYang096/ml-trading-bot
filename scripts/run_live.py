"""run_live.py — Directional trend/fat-tail live consumer

GenericLiveStrategy → 配置驱动通用决策引擎
支持 BPC/ME/SRB/TPC（及可选 LV）等 TradeIntent 策略；``resource_allocation.enabled_archetypes`` 决定参与集合。

数据管线（唯一支持）:
  quant-feature-bus: BinanceWS → 特征 → 磁盘 shared_feature_bus
  quant-trend-fattail: 磁盘 Feature Bus → MultiSymbolManager → PCM → OrderManager（+ 可选 User Stream）
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.live_data_stream import StorageManager, GapFiller, MultiSymbolManager
from src.live_data_stream.classic_feature_bus_provider import ClassicFeatureBusProvider
from src.live_data_stream.constitution_config import (
    enabled_archetypes_from_constitution,
    load_constitution_dict,
    pcm_archetype_priority_for_registry,
    pcm_resolve_registry_key,
    resolve_constitution_yaml,
    validate_classic_slot_capacity,
)
from src.live_data_stream.order_manager_factory import init_order_manager_from_env
from src.live_data_stream.classic_listener_feature_stack import (
    build_extra_feature_computers_for_symbol,
    make_primary_feature_computer_factory,
)
from src.live_data_stream.strategy_runtime_config import (
    load_strategy_timeframe,
    me_enabled_in_allowlist,
    me_strategy_package_name,
)
from src.time_series_model.live.stats_collector import StatsCollector
from src.time_series_model.live.metrics_exporter import start_metrics_server, METRICS
from pathlib import Path as _Path
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)

logger = logging.getLogger(__name__)


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in (raw or "").split(",") if s.strip()]


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
    """多策略实盘启动 (BPC + ME + SRB + TPC + 可选 LV) — 多时间框架

    时间框架 (默认来自各策略 meta.yaml):
      BPC: 通常 240T
      SRB / TPC: 通常 120T（若相同则共用一个增量 FC；不同则各 FC）
      ME: 通常 60T

    数据管线（本进程不连行情 WS）:
      Feature Bus 磁盘 → listener 按各 timeframe 重放特征 → PCM
    """
    from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
    from src.time_series_model.portfolio.live_pcm import LivePCM
    from src.time_series_model.live.live_feature_plan import (
        extract_features_from_archetypes,
    )

    strategies_root = os.getenv(
        "MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies"
    )
    window_minutes = int(os.getenv("MLBOT_BPC_WINDOW_MINUTES", "15"))

    # ── 0. 从 constitution 读取 enabled_archetypes（与 quant-feature-bus 同源）──
    constitution_yaml_path = resolve_constitution_yaml(strategies_root, override=None)
    _const_cfg = load_constitution_dict(constitution_yaml_path)
    _from_const = (
        (_const_cfg.get("resource_allocation") or {}).get("enabled_archetypes")
        or _const_cfg.get("enabled_archetypes")
        or []
    )
    validate_classic_slot_capacity(
        constitution_cfg=_const_cfg,
        symbols=symbols,
    )
    enabled_archetypes = enabled_archetypes_from_constitution(_const_cfg)
    logger.info(
        "📋 enabled_archetypes=%s (source=%s)",
        enabled_archetypes,
        "constitution" if _from_const else "default_all",
    )

    # ── 1. BPC 主周期（主 FC 时钟）；其余 timeframe 来自各已注册策略 meta.yaml ──
    me_pkg = me_strategy_package_name(strategies_root)
    tf_bpc = load_strategy_timeframe(strategies_root, "bpc")  # 默认 240T

    def _tf_to_bar_minutes(tf: str) -> int:
        """'240T' → 240, '60T' → 60"""
        return int(tf.replace("T", ""))

    bar_minutes_bpc = _tf_to_bar_minutes(tf_bpc)

    # ── 初始化 PCM 注册策略（与 Step 9.5 / feature-bus 同源：仅 ``enabled_archetypes``）──
    logger.info(
        "📋 LivePCM register candidates (enabled_archetypes)=%s", enabled_archetypes
    )

    _strategy_map: Dict[str, Any] = {}
    _tf_map: Dict[str, str] = {}
    for arch in enabled_archetypes:
        rk = pcm_resolve_registry_key(str(arch), me_pkg, me_enabled_in_allowlist)
        if not rk or rk in _strategy_map:
            continue
        disk = me_pkg if rk == me_pkg else rk
        strat_dir = os.path.join(strategies_root, disk)
        if not os.path.isdir(strat_dir):
            logger.warning("PCM: skip %s (missing directory %s)", arch, strat_dir)
            continue
        tf = load_strategy_timeframe(strategies_root, disk)
        bm = _tf_to_bar_minutes(tf)
        _strategy_map[rk] = GenericLiveStrategy(
            strategy_name=disk,
            strategies_root=strategies_root,
            trade_size=trade_size,
            primary_timeframe=tf,
            bar_minutes=bm,
        )
        _tf_map[rk] = tf

    if "bpc" not in _strategy_map:
        raise ValueError(
            "constitution resource_allocation.enabled_archetypes must include `bpc` "
            "and strategies/bpc must exist (primary feature computer uses "
            "strategies/bpc/archetypes)."
        )

    logger.info("✅ PCM 策略注册完成: %s", list(_strategy_map.keys()))

    # 创建 ConstitutionExecutor + RuntimeState（与 enabled_archetypes 同源文件）
    constitution_exec = ConstitutionExecutor(constitution_yaml=constitution_yaml_path)
    runtime_st = constitution_exec.load_runtime_state()
    logger.info("✅ ConstitutionExecutor 初始化: %s", constitution_yaml_path)

    # ── 2. 创建 PCM 仲裁层 (注册策略 + timeframe 绑定) ──
    pcm_priority = pcm_archetype_priority_for_registry(
        _const_cfg,
        registry_keys=set(_strategy_map.keys()),
        me_pkg=me_pkg,
        me_enabled_in_allowlist_fn=me_enabled_in_allowlist,
    )
    pcm = LivePCM(
        archetype_priority=pcm_priority,
        constitution_yaml=constitution_yaml_path,
        get_open_slot_count=lambda: runtime_st.slots.active_count(),
    )
    for _name, _strat in _strategy_map.items():
        pcm.register(_name, _strat, timeframe=_tf_map[_name])

    logger.info(f"✅ PCM 仲裁层初始化: 优先级={pcm.archetype_priority}")

    order_manager = init_order_manager_from_env()

    # ── 3. 特征计算器：主 BPC 时钟 + 与 BPC 同周期的已注册策略列合并进主 FC ──
    bpc_archetypes = os.path.join(strategies_root, "bpc", "archetypes")
    fer_archetypes = os.path.join(strategies_root, "fer", "archetypes")
    fer_extra_feat_set: set = set()
    fer_extra_feat_nodes: list = []
    if os.path.isdir(fer_archetypes):
        try:
            fer_extra_feat_set, fer_extra_feat_nodes = extract_features_from_archetypes(
                fer_archetypes
            )
            logger.info(
                "  FER-side nodes merged into primary FC (no FER strategy): %d cols",
                len(fer_extra_feat_set),
            )
        except Exception as e:
            logger.warning("  FER archetype extraction for primary FC failed: %s", e)

    same_tf_other_dirs: List[str] = []
    for rk, tf in _tf_map.items():
        if rk == "bpc" or tf != tf_bpc:
            continue
        disk = me_pkg if rk == me_pkg else rk
        ad = os.path.join(strategies_root, disk, "archetypes")
        if os.path.isdir(ad):
            same_tf_other_dirs.append(ad)
    if same_tf_other_dirs:
        logger.info(
            "  primary FC also merges archetypes dirs (same tf as BPC): %s",
            same_tf_other_dirs,
        )

    primary_fc_factory = make_primary_feature_computer_factory(
        strategies_root=strategies_root,
        tf_bpc=tf_bpc,
        bar_minutes_bpc=bar_minutes_bpc,
        bpc_archetypes_dir=bpc_archetypes,
        fer_feat=fer_extra_feat_set,
        fer_nodes=fer_extra_feat_nodes,
        same_tf_other_dirs=same_tf_other_dirs,
    )

    # ── 4. MultiSymbolManager (primary FC = BPC 周期) ──
    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        feature_computer_factory=primary_fc_factory,
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

    # ── 启动时 slot 同步: 从 Binance 查真实持仓，清理残留 stale slot ──
    _sync_slots_with_exchange(order_manager, constitution_exec, runtime_st, symbols)

    logged_extra_timeframes: Optional[List[str]] = None
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
        _xf = build_extra_feature_computers_for_symbol(
            strategies_root=strategies_root,
            registry_tf_map=_tf_map,
            me_pkg=me_pkg,
            tf_bpc=tf_bpc,
            fer_feat=fer_extra_feat_set,
            fer_nodes=fer_extra_feat_nodes,
        )
        listener.extra_feature_computers = _xf
        if logged_extra_timeframes is None:
            logged_extra_timeframes = list(_xf.keys())

    logger.info(
        "✅ 多策略实盘启动完成: %s symbols primary=%s extras_tf=%s window=%smin pcm=%s",
        len(symbols),
        tf_bpc,
        logged_extra_timeframes or [],
        window_minutes,
        pcm.registered_archetypes,
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
        logger.info(
            "[quantiles] %s: 跳过多 symbol 分位数融合（各 symbol 均未产生 quantile 表："
            "gate 无 quantile_* 规则，或特征列不足/有效点<10）；多数策略可忽略。",
            arch_name,
        )
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
      - 主 FC 计算 primary_timeframe quantiles → BPC
      - 额外 FC 计算各策略 timeframe → ME / SRB / TPC / LV
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
    config_path = project_root / "config" / "pipelines" / "pcm_orchestrate_2h.yaml"
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
        strategy_names = ["bpc", "me", "tpc"]

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


def _manager_primary_timeframe(manager: MultiSymbolManager) -> str:
    for listener in manager.listeners.values():
        tf = getattr(
            getattr(listener, "feature_computer", None), "primary_timeframe", None
        )
        if tf:
            return str(tf)
    return "240T"


def _manager_feature_timeframes(
    manager: MultiSymbolManager, pcm: Any, primary_timeframe: str
) -> List[str]:
    tfs: List[str] = [primary_timeframe]
    for listener in manager.listeners.values():
        for tf in getattr(listener, "extra_feature_computers", {}).keys():
            if str(tf) not in tfs:
                tfs.append(str(tf))
    for tf in getattr(pcm, "_strategy_timeframes", {}).values():
        if str(tf) not in tfs:
            tfs.append(str(tf))
    return tfs


def _pcm_strategy_timeframes(pcm: Any, primary_timeframe: str) -> Dict[str, str]:
    """Return strategy -> feature timeframe labels for dashboard metrics."""
    strategies = getattr(pcm, "_strategies", {}) or {}
    raw_tfs = getattr(pcm, "_strategy_timeframes", {}) or {}
    out: Dict[str, str] = {}
    for name in strategies.keys():
        out[str(name)] = str(raw_tfs.get(name) or primary_timeframe)
    return out


def _add_bus_bars_to_listener(listener: Any, bars: List[Dict[str, Any]]) -> None:
    if not bars:
        return
    for bar in bars:
        try:
            listener.memory_window.add(dict(bar))
        except Exception:
            logger.debug("[%s] feature-bus bar memory update skipped", listener.symbol)
    try:
        listener.current_1min_bar = dict(bars[-1])
        listener._bar_count += len(bars)
    except Exception:
        pass


def _enforce_bus_execution_bars(
    listener: Any,
    bars: List[Dict[str, Any]],
    feature_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Run software exits on 1m/fast bars without triggering new entries."""
    if not bars:
        return
    _add_bus_bars_to_listener(listener, bars)
    tracker = getattr(listener, "_position_tracker", None)
    order_manager = getattr(listener, "order_manager", None)
    if tracker is None or order_manager is None:
        return
    for bar in bars:
        features = dict(feature_context or {})
        features.update(
            {
                "timestamp": bar.get("timestamp"),
                "open": bar.get("open"),
                "high": bar.get("high"),
                "low": bar.get("low"),
                "close": bar.get("close"),
                "price": bar.get("close"),
            }
        )
        closed = tracker.enforce_all(features=features)
        if closed:
            logger.info(
                "[%s] 1m bus execution closed positions: %s", listener.symbol, closed
            )


async def _run_external_feature_bus_mode(
    *,
    manager: MultiSymbolManager,
    pcm: Any,
    symbols: List[str],
    bg_gap_task: Optional[asyncio.Task],
) -> None:
    """Run trend/fat-tail decisions from disk Feature Bus instead of market WS."""

    feature_bus_root = os.getenv("MLBOT_FEATURE_BUS_ROOT", "live/shared_feature_bus")
    poll_seconds = float(os.getenv("MLBOT_FEATURE_BUS_POLL_SECONDS", "5"))
    max_stale = float(os.getenv("MLBOT_FEATURE_BUS_MAX_STALENESS_SECONDS", "1800"))
    primary_tf = _manager_primary_timeframe(manager)
    timeframes = _manager_feature_timeframes(manager, pcm, primary_tf)
    strategy_timeframes = _pcm_strategy_timeframes(pcm, primary_tf)
    provider = ClassicFeatureBusProvider(
        feature_bus_root=feature_bus_root,
        symbols=symbols,
        primary_timeframe=primary_tf,
        timeframes=timeframes,
        max_staleness_seconds=max_stale,
        bars_lookback=int(os.getenv("MLBOT_FEATURE_BUS_BARS_LOOKBACK", "240")),
        initial_bars_lookback=int(
            os.getenv("MLBOT_FEATURE_BUS_INITIAL_BARS_LOOKBACK", "1")
        ),
    )
    latest_features: Dict[str, Dict[str, Any]] = {}
    logger.info(
        "🚌 External Feature Bus mode: root=%s primary=%s timeframes=%s poll=%.1fs",
        feature_bus_root,
        primary_tf,
        timeframes,
        poll_seconds,
    )

    if manager.user_stream is not None:
        await manager.user_stream.start()

    async def _periodic_market_update() -> None:
        interval = int(os.getenv("MLBOT_MARKET_DATA_INTERVAL", "30"))
        _mode_map = {"OFFLINE": 0, "DEGRADED": 1, "NORMAL": 2}
        loop = asyncio.get_running_loop()
        while True:
            # Account gauges use separate try so a failing public market fetch
            # cannot starve BINANCE-signed `update_account_data` for a whole backoff window.
            try:
                await loop.run_in_executor(None, METRICS.update_account_data)
            except Exception:
                logger.warning(
                    "账户指标更新异常（仍为上次成功的余额或默认值）",
                    exc_info=True,
                )

            try:
                await loop.run_in_executor(None, METRICS.update_market_data, symbols)
            except Exception:
                logger.warning(
                    "市场公开数据指标更新异常（premiumIndex/openInterest）",
                    exc_info=True,
                )

            try:
                cur_mode = manager.mode_manager.get_current_mode()
                METRICS.system_mode.set(_mode_map.get(cur_mode.value, 0))
                for sym in symbols:
                    listener = manager.get_listener(sym)
                    if listener and listener.last_feature_compute_time:
                        now = pd.Timestamp.now(tz="UTC")
                        age = (now - listener.last_feature_compute_time).total_seconds()
                        METRICS.last_bar_age.labels(symbol=sym).set(age)
                _first_listener = manager.get_listener(symbols[0]) if symbols else None
                if _first_listener:
                    _rs = getattr(_first_listener, "runtime_state", None)
                    _psl = getattr(_first_listener, "per_strategy_limits", {}) or {}
                    _slots_cfg = (pcm.constitution or {}).get("slots", {})
                    _global_max = int(_slots_cfg.get("slot_count", 2))
                    METRICS.update_slot_metrics(_rs, _psl, _global_max)
                try:
                    _pcm_stats = pcm.get_stats() if pcm is not None else {}
                    METRICS.update_pcm_notional_metrics(
                        runtime=_pcm_stats.get("notional_runtime") or {},
                        policy=(
                            (_pcm_stats.get("constitution") or {}).get(
                                "notional_policy"
                            )
                        )
                        or {},
                    )
                except Exception:
                    logger.debug("PCM notional metrics 更新异常", exc_info=True)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("运行时指标更新异常（模式/slot/bar age 等）: %s", exc)
                await asyncio.sleep(60)
                continue
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def _daily_funding_oi_refresh() -> None:
        interval = 12 * 3600
        await asyncio.sleep(interval)
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

    async def _periodic_retrain_check() -> None:
        await asyncio.sleep(300)
        while True:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _run_retrain_check)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("重训检测异常: %s", exc)
            await asyncio.sleep(6 * 3600)

    market_task = asyncio.create_task(_periodic_market_update())
    funding_oi_task = asyncio.create_task(_daily_funding_oi_refresh())
    retrain_task = asyncio.create_task(_periodic_retrain_check())

    try:
        while True:
            for sym in symbols:
                snap_age = provider.reader.latest_snapshot_age_seconds(
                    symbol=sym, timeframe=primary_tf
                )
                if snap_age is not None:
                    METRICS.feature_bus_snapshot_age_seconds.labels(symbol=sym).set(
                        snap_age
                    )
            for symbol, bars in provider.poll_bars().items():
                listener = manager.get_listener(symbol)
                if listener is None:
                    continue
                context = latest_features.get(symbol)
                if context is None:
                    bundle = provider.latest_feature_bundle(symbol)
                    context = bundle.get(primary_tf, {})
                    if context:
                        latest_features[symbol] = context
                _enforce_bus_execution_bars(listener, bars, context)

            events = provider.poll()
            for event in events:
                listener = manager.get_listener(event.symbol)
                if listener is None:
                    continue
                _enforce_bus_execution_bars(listener, event.bars, event.features)
                latest_features[event.symbol] = dict(event.features)
                listener.last_feature_compute_time = event.timestamp
                for strategy, tf in strategy_timeframes.items():
                    row = event.features_by_timeframe.get(tf) or event.features
                    METRICS.update_strategy_symbol_ohlc(
                        strategy=strategy,
                        symbol=event.symbol,
                        timeframe=tf,
                        values=row,
                    )
                    METRICS.update_strategy_feature_values(
                        strategy=strategy,
                        symbol=event.symbol,
                        timeframe=tf,
                        values=row,
                        layer="trend",
                    )
                listener._handle_features(
                    dict(event.features),
                    features_by_timeframe=(
                        event.features_by_timeframe
                        if len(event.features_by_timeframe) > 1
                        else None
                    ),
                )
                METRICS.last_bar_age.labels(symbol=event.symbol).set(0)
            await asyncio.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("External Feature Bus mode interrupted")
    finally:
        market_task.cancel()
        funding_oi_task.cancel()
        retrain_task.cancel()
        if bg_gap_task:
            bg_gap_task.cancel()
        if manager.user_stream is not None:
            await manager.user_stream.stop()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols = _parse_symbols(os.getenv("MLBOT_LIVE_SYMBOLS", "BTCUSDT"))
    if not symbols:
        raise ValueError("No symbols provided. Set MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT")

    storage_base = os.getenv("MLBOT_LIVE_STORAGE_BASE", "data/live_storage")
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

    manager, pcm = _setup_three_strategies(
        symbols, storage, gap_filler, trade_size, risk_per_trade
    )
    try:
        METRICS.publish_dashboard_catalog(
            strategies=pcm.registered_archetypes,
            symbols=symbols,
        )
    except Exception:
        logger.debug("dashboard catalog publish skipped", exc_info=True)
    _METRIC_MODE_MAP = {"OFFLINE": 0, "DEGRADED": 1, "NORMAL": 2, "ABNORMAL": 0}
    try:
        METRICS.system_mode.set(
            _METRIC_MODE_MAP.get(manager.mode_manager.get_current_mode().value, 0)
        )
    except Exception:
        logger.debug("initial system_mode gauge skipped", exc_info=True)

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

    feature_source = os.getenv("MLBOT_FEATURE_SOURCE", "bus").strip().lower()
    if feature_source not in {"bus", "feature-bus", "feature_store", "feature-store"}:
        raise SystemExit(
            "MLBOT_FEATURE_SOURCE must be bus / feature-bus / feature-store. "
            "Classic live only consumes the disk feature bus; run quant-feature-bus "
            "(scripts/run_market_feature_publisher.py) for ticks → features → disk."
        )
    await _run_external_feature_bus_mode(
        manager=manager,
        pcm=pcm,
        symbols=symbols,
        bg_gap_task=bg_gap_task,
    )


if __name__ == "__main__":
    asyncio.run(main())
