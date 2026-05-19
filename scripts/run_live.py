"""run_live.py — Directional trend/fat-tail live consumer

GenericLiveStrategy → 配置驱动通用决策引擎
支持 TPC/ME/SRB（及可选 LV）等 TradeIntent 策略；``resource_allocation.enabled_archetypes`` 决定参与集合。

数据管线（唯一支持）:
  quant-feature-bus: BinanceWS → 特征 → 磁盘 shared_feature_bus
  quant-trend-fattail: 磁盘 Feature Bus → MultiSymbolManager → PCM → OrderManager（+ 可选 User Stream）

审计日志文件（可选，默认开启）：
  ``{MLBOT_LIVE_STORAGE_BASE}/logs/trend_live_audit.log`` — 默认**按小时**切分并保留约
  ``MLBOT_LIVE_AUDIT_RETENTION_DAYS``（默认 30 天，约 720 个按小时归档）。环境变量：
  ``MLBOT_LIVE_AUDIT_LOG``（路径或 ``default``）、``MLBOT_LIVE_AUDIT_DISABLE``、
  ``MLBOT_LIVE_AUDIT_RETENTION_DAYS``、``MLBOT_LIVE_AUDIT_ROTATION``（或共享 ``MLBOT_AUDIT_ROTATION``：
  ``hour``/``day``）。趋势单 ``clientOrderId`` 前缀：
  ``MLBOT_LIVE_CLIENT_ORDER_PREFIX``（默认 ``tl``，需满足 Binance 长度限制）。

终态订单 REST 回填（缺 ``average_price`` / ``filled_at`` / 驳回原因等，见
``src/live_data_stream/terminal_order_backfill.py``）：

  ``MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS``（未设置时默认 ``60``；``0``/``false``/``off`` 关闭）；
  ``MLBOT_TERMINAL_ORDER_BACKFILL_LOOKBACK_HOURS``（默认 ``168``）；
  ``MLBOT_TERMINAL_ORDER_BACKFILL_LIMIT``（默认 ``200``）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config.strategy_layout import resolve_strategy_package_under_root
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
)
from src.time_series_model.live.stats_collector import StatsCollector
from src.time_series_model.live.metrics_exporter import start_metrics_server, METRICS
from scripts.live_audit_file import configure_audit_from_env_defaults
from src.live_data_stream.terminal_order_backfill import (
    periodic_terminal_order_backfill,
    terminal_order_backfill_enabled_interval_seconds,
    terminal_order_backfill_should_run,
)

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


def _exchange_live_symbols(order_manager) -> Optional[set[str]]:
    """Return exchange symbols with non-zero positions, or None if unavailable."""
    if order_manager is None:
        return None
    api = getattr(order_manager, "binance_api", None)
    if api is None:
        return None
    try:
        exchange_positions = api.get_positions()
    except Exception as e:
        logger.warning("PositionTracker 恢复: 查询 Binance 持仓失败: %s", e)
        return None
    live_symbols: set[str] = set()
    for p in exchange_positions:
        raw_sym = _ccxt_symbol_to_raw(p.get("symbol", ""))
        if raw_sym:
            live_symbols.add(raw_sym)
    return live_symbols


def _restore_position_trackers_from_disk(
    *,
    manager,
    order_manager,
    symbols: List[str],
) -> None:
    """Restore per-symbol PositionTracker state after restart.

    Only restores symbols that still have an exchange-backed position. This
    avoids adopting orphan JSON state after a manual close or exchange SL fill.
    """
    live_symbols = _exchange_live_symbols(order_manager)
    if live_symbols is None:
        logger.warning(
            "PositionTracker 恢复: 跳过（无法确认交易所持仓，避免误接管孤儿仓）"
        )
        return
    restored_total = 0
    restored_by_symbol: dict[str, int] = {}
    for sym in symbols:
        try:
            listener = manager.get_listener(sym)
        except Exception:
            continue
        if listener is None:
            continue
        tracker = getattr(listener, "_position_tracker", None)
        if tracker is None or not hasattr(tracker, "restore_from_disk"):
            continue
        n = int(tracker.restore_from_disk(live_symbols=live_symbols) or 0)
        restored_by_symbol[str(sym).upper().strip()] = n
        restored_total += n
    missing_snapshots = sorted(
        s
        for s in live_symbols
        if s in {str(x).upper().strip() for x in symbols}
        and int(restored_by_symbol.get(s, 0) or 0) <= 0
    )
    if missing_snapshots:
        logger.warning(
            "PositionTracker 恢复: 交易所有仓但无本地执行快照，软件 trailing/结构出场"
            "无法完整恢复: %s",
            missing_snapshots,
        )
    logger.info(
        "PositionTracker 恢复完成: restored=%d live_symbols=%s",
        restored_total,
        sorted(live_symbols),
    )


def _open_trend_positions_snapshot_from_manager(
    manager: Optional[Any], symbols: List[str]
) -> List[Dict[str, Any]]:
    """Build open trend slot snapshot for LivePCM trend_pool_guard."""
    if manager is None:
        return []
    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            listener = manager.get_listener(sym)
        except Exception:
            continue
        if listener is None:
            continue
        tracker = getattr(listener, "_position_tracker", None)
        if tracker is None:
            continue
        try:
            pos_map = tracker.all_positions() or {}
        except Exception:
            continue
        for pos in pos_map.values():
            if not isinstance(pos, dict):
                continue
            archetype = str(pos.get("archetype", "") or "").strip().lower()
            if not archetype:
                continue
            side = str(pos.get("side", "") or "").strip().lower()
            entry_price = float(pos.get("entry_price") or 0.0)
            stop_price = pos.get("stop_loss_price")
            stop_nonnegative = False
            if stop_price is not None and entry_price > 0:
                try:
                    stop_v = float(stop_price)
                    if side == "long":
                        stop_nonnegative = stop_v >= entry_price
                    elif side == "short":
                        stop_nonnegative = stop_v <= entry_price
                except Exception:
                    stop_nonnegative = False
            rows.append(
                {
                    "symbol": str(pos.get("symbol", sym) or sym).upper().strip(),
                    "archetype": archetype,
                    "side": side,
                    "breakeven_locked": bool(pos.get("breakeven_locked", False)),
                    "stop_risk_nonnegative": bool(stop_nonnegative),
                }
            )
    return rows


def _setup_three_strategies(
    symbols: List[str],
    storage: StorageManager,
    gap_filler,
    trade_size: float,
    risk_per_trade: float = 0.0,
):
    """多策略实盘启动 (BPC + ME + SRB + TPC + 可选 LV) — 多时间框架

    时间框架 (默认来自各策略 meta.yaml):
      TPC: 通常 120T；ME: 通常 60T（若宪法启用且 live 目录存在则参与 PCM）

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

    def _tf_to_bar_minutes(tf: str) -> int:
        """'240T' → 240, '60T' → 60"""
        return int(tf.replace("T", ""))

    # ── 1. PCM 策略注册（与 Step 9.5 / feature-bus 同源：仅 enabled_archetypes）──
    logger.info(
        "📋 LivePCM register candidates (enabled_archetypes)=%s", enabled_archetypes
    )

    _strategy_map: Dict[str, Any] = {}
    _tf_map: Dict[str, str] = {}
    for arch in enabled_archetypes:
        rk = pcm_resolve_registry_key(str(arch), "me", me_enabled_in_allowlist)
        if not rk or rk in _strategy_map:
            continue
        pkg = resolve_strategy_package_under_root(
            Path(strategies_root), rk, allow_bad_candidates=False
        )
        if not pkg.is_dir():
            logger.warning("PCM: skip %s (missing directory %s)", arch, pkg)
            continue
        tf = load_strategy_timeframe(strategies_root, rk)
        bm = _tf_to_bar_minutes(tf)
        _strategy_map[rk] = GenericLiveStrategy(
            strategy_name=rk,
            strategies_root=strategies_root,
            trade_size=trade_size,
            primary_timeframe=tf,
            bar_minutes=bm,
        )
        _tf_map[rk] = tf

    if not _strategy_map:
        raise ValueError(
            "constitution enabled_archetypes produced no disk-backed strategies "
            f"under strategies_root={strategies_root!r}"
        )

    pcm_priority_preview = pcm_archetype_priority_for_registry(
        _const_cfg,
        registry_keys=set(_strategy_map.keys()),
        me_pkg="me",
        me_enabled_in_allowlist_fn=me_enabled_in_allowlist,
    )
    if not pcm_priority_preview:
        raise ValueError(
            "cannot derive PCM archetype ordering (empty registry ∩ priority ordering)"
        )
    primary_registry_key = pcm_priority_preview[0]
    tf_primary = _tf_map[primary_registry_key]
    bar_minutes_primary = _tf_to_bar_minutes(tf_primary)

    logger.info(
        "✅ PCM 策略注册完成: %s ; primary_fc=%s tf=%s",
        list(_strategy_map.keys()),
        primary_registry_key,
        tf_primary,
    )

    # 创建 ConstitutionExecutor + RuntimeState（与 enabled_archetypes 同源文件）
    constitution_exec = ConstitutionExecutor(constitution_yaml=constitution_yaml_path)
    runtime_st = constitution_exec.load_runtime_state()
    logger.info("✅ ConstitutionExecutor 初始化: %s", constitution_yaml_path)

    # ── 2. 创建 PCM 仲裁层 (注册策略 + timeframe 绑定) ──
    pcm_priority = pcm_archetype_priority_for_registry(
        _const_cfg,
        registry_keys=set(_strategy_map.keys()),
        me_pkg="me",
        me_enabled_in_allowlist_fn=me_enabled_in_allowlist,
    )
    # 实盘 trend_pool_guard 需要读取「当前开仓是否已 breakeven 锁盈」；
    # 通过 manager/listener 的 PositionTracker 快照回传给 LivePCM。
    _manager_ref: Dict[str, Any] = {"manager": None}

    def _open_trend_positions_snapshot() -> List[Dict[str, Any]]:
        return _open_trend_positions_snapshot_from_manager(
            _manager_ref.get("manager"), symbols
        )

    pcm = LivePCM(
        archetype_priority=pcm_priority,
        constitution_yaml=constitution_yaml_path,
        get_open_slot_count=lambda: runtime_st.slots.active_count(),
        get_open_trend_positions=_open_trend_positions_snapshot,
    )
    for _name, _strat in _strategy_map.items():
        pcm.register(_name, _strat, timeframe=_tf_map[_name])

    logger.info(f"✅ PCM 仲裁层初始化: 优先级={pcm.archetype_priority}")

    order_manager = init_order_manager_from_env()

    # ── 3. 特征计算器：PCM 顺位第一的 archetype 为主时钟；同周期的其它注册策略并入主 FC ──
    _primary_pkg = resolve_strategy_package_under_root(
        Path(strategies_root), primary_registry_key, allow_bad_candidates=False
    )
    primary_archetypes_dir = str(_primary_pkg / "archetypes")

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
        if rk == primary_registry_key or tf != tf_primary:
            continue
        ad = (
            resolve_strategy_package_under_root(
                Path(strategies_root), rk, allow_bad_candidates=False
            )
            / "archetypes"
        )
        if ad.is_dir():
            same_tf_other_dirs.append(str(ad))
    if same_tf_other_dirs:
        logger.info(
            "  primary FC also merges archetypes dirs (same tf as %s): %s",
            primary_registry_key,
            same_tf_other_dirs,
        )

    primary_fc_factory = make_primary_feature_computer_factory(
        strategies_root=strategies_root,
        tf_bpc=tf_primary,
        bar_minutes_bpc=bar_minutes_primary,
        bpc_archetypes_dir=primary_archetypes_dir,
        fer_feat=fer_extra_feat_set,
        fer_nodes=fer_extra_feat_nodes,
        same_tf_other_dirs=same_tf_other_dirs,
    )

    # ── 4. MultiSymbolManager (primary FC = PCM 首选 archetype 周期) ──
    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        feature_computer_factory=primary_fc_factory,
        gap_filler=gap_filler,
        feature_compute_interval_minutes=window_minutes,
        orderflow_window_minutes=window_minutes,
        order_manager=order_manager,
    )
    _manager_ref["manager"] = manager

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
    # 恢复重启前的完整持仓执行状态（trailing/breakeven/结构出场等）。
    _restore_position_trackers_from_disk(
        manager=manager,
        order_manager=order_manager,
        symbols=symbols,
    )
    # 进程重启后 PCM 内存槽位为空；用持久化 constitution slot 回填，避免下一信号误当「新开」
    pcm.hydrate_slot_evidence_from_constitution_slots(runtime_st)

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
            tf_bpc=tf_primary,
            fer_feat=fer_extra_feat_set,
            fer_nodes=fer_extra_feat_nodes,
            primary_registry_key=primary_registry_key,
        )
        listener.extra_feature_computers = _xf
        if logged_extra_timeframes is None:
            logged_extra_timeframes = list(_xf.keys())

    logger.info(
        "✅ 多策略实盘启动完成: %s symbols primary=%s/%s extras_tf=%s window=%smin pcm=%s",
        len(symbols),
        primary_registry_key,
        tf_primary,
        logged_extra_timeframes or [],
        window_minutes,
        pcm.registered_archetypes,
    )
    return manager, pcm


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

    project_root = Path(__file__).resolve().parents[1]
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
        strategy_names = ["bpc", "tpc"]

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


def _resolve_feature_bus_timeframes_for_disk(
    *,
    feature_bus_root: str,
    manager_primary: str,
    manager: MultiSymbolManager,
    pcm: Any,
) -> Tuple[str, List[str]]:
    """Pick ``features/<tf>/`` keys present on disk (legacy ``primary`` vs meta tf)."""

    from src.live_data_stream.feature_bus import normalize_timeframe

    feat_root = Path(feature_bus_root) / "features"
    mp_n = normalize_timeframe(manager_primary)
    preferred = feat_root / mp_n
    legacy_primary = feat_root / "primary"
    primary_tf: str
    if preferred.is_dir() and any(preferred.glob("*.parquet")):
        primary_tf = manager_primary
    elif legacy_primary.is_dir() and any(legacy_primary.glob("*.parquet")):
        primary_tf = "primary"
        logger.warning(
            "🚌 Reading Feature Bus primary rows from legacy features/primary/ "
            "(strategy metadata timeframe is %s). Align publisher timeframe keys.",
            manager_primary,
        )
    else:
        primary_tf = manager_primary

    requested = _manager_feature_timeframes(manager, pcm, manager_primary)
    out: List[str] = []
    seen: set[str] = set()

    def _maybe_add_tf(tf_key: str) -> None:
        kn = normalize_timeframe(tf_key)
        if kn in seen:
            return
        dir_path = feat_root / kn
        if dir_path.is_dir() and any(dir_path.glob("*.parquet")):
            seen.add(kn)
            out.append(tf_key)

    for tf in requested:
        if normalize_timeframe(tf) == mp_n:
            _maybe_add_tf(primary_tf)
        else:
            _maybe_add_tf(tf)

    if not out:
        _maybe_add_tf(primary_tf)
    if not out:
        logger.warning(
            "🚌 No feature-bus parquet snapshots under %s for requested keys %s; "
            "provider references %s until data appears.",
            feat_root,
            requested,
            primary_tf,
        )
        out = [primary_tf]

    return primary_tf, out


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
    manager_primary = _manager_primary_timeframe(manager)
    primary_tf, timeframes = _resolve_feature_bus_timeframes_for_disk(
        feature_bus_root=feature_bus_root,
        manager_primary=manager_primary,
        manager=manager,
        pcm=pcm,
    )
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

    terminal_backfill_task: Optional[asyncio.Task] = None
    om = getattr(manager, "order_manager", None)
    if (
        terminal_order_backfill_enabled_interval_seconds() > 0
        and terminal_order_backfill_should_run(om)
    ):
        terminal_backfill_task = asyncio.create_task(
            periodic_terminal_order_backfill(om, startup_delay_seconds=20.0)
        )

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
        if terminal_backfill_task is not None:
            terminal_backfill_task.cancel()
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
    configure_audit_from_env_defaults(
        default_log_file=Path(storage_base) / "logs" / "trend_live_audit.log",
        disable_env="MLBOT_LIVE_AUDIT_DISABLE",
        path_env="MLBOT_LIVE_AUDIT_LOG",
        retention_env="MLBOT_LIVE_AUDIT_RETENTION_DAYS",
        rotation_env="MLBOT_LIVE_AUDIT_ROTATION",
        banner="trend/live audit file",
    )
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
