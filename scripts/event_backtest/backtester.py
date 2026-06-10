from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from scripts.account_ledger import AccountLedger
from scripts.capital_report import write_capital_report_from_trades
from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.features.timeline import (
    _align_feature_index_to_bar_close,
    _feature_asof_from_sym_tf_features,
    _feature_row_asof_from_sym_tf_features,
    _get_bar_minutes,
    _get_timeframe,
    _iter_update_bars_1min,
    _iter_update_bars_primary_tf,
    _ohlc_dict_from_bar_row,
    _sync_ema_1200_from_feature_row,
    _sync_macro_tp_vwap_from_feature_row,
    _timeframe_from_strategy_meta,
    _timeframe_to_timedelta,
    row_to_features,
)
from scripts.event_backtest.modes import BacktestMode, resolve_backtest_mode
from scripts.event_backtest.reporting.audit import (
    _add_attempt_snapshot,
    _apply_pcm_direction_ffill,
    _er_pct_attempt_stats,
    _extract_path_efficiency_pct,
    _trade_audit_row_from_fill,
)
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.sizing import sync_event_backtest_sizing_equity
from scripts.event_backtest.simulator.om_bridge import OMBridge, OM_AVAILABLE
from scripts.event_backtest.simulator.position import (
    PositionSimulator,
    _collect_open_parent_pids,
    _load_add_position_runtime_from_resume,
    _merge_add_position_runtime_with_open_legs,
    _prune_stale_add_position_records,
)
from scripts.event_backtest.spot.budget import _build_spot_capital_budget_or_none
from scripts.event_backtest.spot.metrics import (
    _compute_deploy_quote_pct_series,
    _compute_spot_accum_accumulation_audit,
    _compute_spot_buy_hold_benchmarks,
    _compute_spot_inventory_metrics,
)
from scripts.event_backtest.types.trade import ClosedTrade
from src.data_tools.data_handler import DataHandler
from src.feature_store import FeatureStore, FeatureStoreSpec
from src.feature_store.layer_naming import detect_layer_for_strategy
from src.live_data_stream.feature_storage import StorageManager
from src.time_series_model.core.constitution.add_position_rules import (
    resolve_float_r_ladder_only as _shared_resolve_float_r_ladder_only,
)
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.event_backtest_srb_hooks import SrbEventBacktestHooks
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.portfolio.live_pcm import LivePCM
from src.features.cross_symbol.macro_tp_vwap_anchor import (
    ANCHOR_COLUMN,
    parse_macro_tp_vwap_anchor_config,
)


def _inject_scores_has_usable_columns(df: pd.DataFrame) -> bool:
    cols = {str(c).lower() for c in df.columns}
    if "score" in cols or "add_ml_score" in cols:
        return True
    return "score_long" in cols and "score_short" in cols


class EventBacktester:
    """
    事件驱动回测主类 — 完全模拟实盘多策略环境

    与实盘一致的架构:
      1. LivePCM 仲裁 (全局 slot 控制, 优先级排序, Regime 感知)
      2. 多策略 GenericLiveStrategy.decide() 信号生成 (BPC + FER + ME)
      3. 多 timeframe 特征计算（timeframe 优先来自各策略 meta.yaml）
      4. PositionSimulator: 1min bar 持仓管理
      5. 跨 symbol 时间线交叉处理 (同实盘顺序)

    用法:
        bt = EventBacktester(strategies=["bpc", "fer", "me"], live_root="live/highcap")
        result = bt.run(symbols=["BTCUSDT", ...], days=180)
        result.print_report()
    """

    def __init__(
        self,
        strategies: List[str],
        live_root: str = "live/highcap",
        strategies_root: Optional[str] = None,
        constitution_yaml: Optional[str] = None,
        db_path: Optional[str] = None,
        data_path: Optional[str] = None,
        fee_rate: float = 0.0,
    ):
        # Keep original strategy casing (e.g. "bpc-short-120T"), because
        # config paths are case-sensitive on Linux.
        self.strategy_names = [s.strip() for s in strategies]
        self.backtest_mode = resolve_backtest_mode(self.strategy_names)
        logger.info(
            "Event backtest mode: %s (strategies=%s)",
            self.backtest_mode.value,
            self.strategy_names,
        )
        self.live_root = live_root
        self.data_path = data_path  # 研究数据目录 (e.g. data/parquet_data)
        self.strategies_root = strategies_root or "config/strategies"
        self.constitution_yaml = constitution_yaml or str(
            Path("config") / "constitution" / "constitution.yaml"
        )
        self.fee_rate = fee_rate  # 单边手续费率

        # Per-strategy timeframe 映射
        self._tf_map: Dict[str, str] = {}  # {strategy: "240T"}
        self._bm_map: Dict[str, int] = {}  # {strategy: 240}
        for s in self.strategy_names:
            self._tf_map[s] = _get_timeframe(s, strategies_root=self.strategies_root)
            self._bm_map[s] = _get_bar_minutes(s, strategies_root=self.strategies_root)

        # 主 bar 分钟数 (position simulator default) + 对应 timeframe token
        self._primary_bar_minutes = max(self._bm_map.values())
        self._primary_timeframe = next(
            self._tf_map[s]
            for s in self.strategy_names
            if self._bm_map[s] == self._primary_bar_minutes
        )

        # order_management 集成 (可选)
        self._om_bridge: Optional[OMBridge] = None
        if db_path and OM_AVAILABLE:
            self._om_bridge = OMBridge(db_path)

        # 初始化 GenericLiveStrategy — 每策略一个
        self._strats: Dict[str, GenericLiveStrategy] = {}
        for s in self.strategy_names:
            self._strats[s] = GenericLiveStrategy(
                strategy_name=s,
                strategies_root=self.strategies_root,
                primary_timeframe=self._tf_map[s],
                bar_minutes=self._bm_map[s],
            )

        # LivePCM 仲裁器 (同实盘: 读取 constitution slot 配置)
        pcm_regime_yaml = str(Path("config") / "pcm_regime.yaml")
        self._simulators: Dict[str, PositionSimulator] = {}
        self.pcm = LivePCM(
            constitution_yaml=self.constitution_yaml,
            regime_config_path=(
                pcm_regime_yaml if Path(pcm_regime_yaml).exists() else None
            ),
            get_open_slot_count=self._global_open_count,
            get_open_trend_positions=self._open_trend_positions_snapshot,
        )
        for s in self.strategy_names:
            self.pcm.register(s, self._strats[s], timeframe=self._tf_map[s])

        # 特征计算器 — 按 unique timeframe 分组 (同 run_live.py)
        # BPC+FER 共享 240T FC，ME 独立 60T FC
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )

        unique_tfs = sorted(set(self._tf_map.values()))
        self._feature_computers: Dict[str, IncrementalFeatureComputer] = {}

        for tf in unique_tfs:
            tf_strats = [s for s in self.strategy_names if self._tf_map[s] == tf]
            first = tf_strats[0]
            archetypes_dir = str(Path(self.strategies_root) / first / "archetypes")
            fc = IncrementalFeatureComputer(
                primary_timeframe=tf,
                archetypes_dir=archetypes_dir,
            )
            # 合并同 timeframe 其他策略的特征集 (同 run_live.py 4H FC)
            for extra in tf_strats[1:]:
                extra_dir = str(Path(self.strategies_root) / extra / "archetypes")
                try:
                    extra_feat_set, extra_feat_nodes = extract_features_from_archetypes(
                        extra_dir
                    )
                    if fc.live_feature_set:
                        fc.live_feature_set |= extra_feat_set
                    fc.live_feature_nodes = sorted(
                        set(fc.live_feature_nodes) | set(extra_feat_nodes)
                    )
                except Exception as e:
                    logger.warning(f"  Feature merge for {extra} failed: {e}")
            # 禁用 live_feature_set 过滤 — 保留所有计算出的特征
            fc.live_feature_set = None
            self._feature_computers[tf] = fc

    def _load_research_data(
        self, sym: str, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """从研究数据目录 (data/parquet_data) 加载 1min bars + ticks

        与 compare_same_data.py 的实盘路径加载逻辑一致:
          DataHandler.load_ohlcv(timeframe="1T") → 1min bars
          glob {SYMBOL}_*.parquet → ticks
        """
        data_root = Path(self.data_path)

        # 1. 加载 1min bars (resample from tick data via DataHandler)
        dh = DataHandler(str(data_root))
        bars_1min = dh.load_ohlcv(
            symbol=sym, timeframe="1T", start_date=start_date, end_date=end_date
        )
        if not bars_1min.empty:
            bars_1min.index = pd.to_datetime(bars_1min.index, utc=True)
            # 列名适配: buy_qty → buy_volume, sell_qty → sell_volume
            col_rename = {"buy_qty": "buy_volume", "sell_qty": "sell_volume"}
            bars_1min = bars_1min.rename(
                columns={k: v for k, v in col_rename.items() if k in bars_1min.columns}
            )
            if "timestamp" not in bars_1min.columns:
                bars_1min["timestamp"] = bars_1min.index

        # 2. 加载 ticks (直接读 parquet 原始数据)
        tick_frames = []
        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)
        for fp in sorted(data_root.glob(f"{sym}_*.parquet")):
            try:
                df_tick = pd.read_parquet(fp)
                if "price" in df_tick.columns and "volume" in df_tick.columns:
                    tick_frames.append(df_tick)
            except Exception:
                pass
        if tick_frames:
            ticks_1min = pd.concat(tick_frames, ignore_index=True)
            ticks_1min["timestamp"] = pd.to_datetime(ticks_1min["timestamp"], utc=True)
            ticks_1min = ticks_1min[
                (ticks_1min["timestamp"] >= start_ts)
                & (ticks_1min["timestamp"] <= end_ts)
            ]
            # 设置 DatetimeIndex — footprint 计算需要
            ticks_1min = ticks_1min.set_index("timestamp", drop=False).sort_index()
        else:
            ticks_1min = pd.DataFrame()

        return bars_1min, ticks_1min

    def _preload_anchor_macro_cache(
        self,
        *,
        anchor_sym: str,
        warmup_start: str,
        end_date_str: str,
        use_research: bool,
        storage: Optional[Any],
    ) -> Dict[str, pd.Series]:
        """Compute anchor symbol macro_tp_vwap series per timeframe (when anchor ∉ backtest universe)."""
        out: Dict[str, pd.Series] = {}
        if use_research:
            bars_1min, ticks_1min = self._load_research_data(
                anchor_sym, warmup_start, end_date_str
            )
        else:
            if storage is None:
                return out
            bars_1min = storage.bar_1min.load_range(
                anchor_sym, warmup_start, end_date_str
            )
            ticks_1min = storage.ticks.load_range(
                anchor_sym, warmup_start, end_date_str
            )
        if len(bars_1min) < 100:
            logger.warning(
                "macro_tp_vwap_anchor: %s insufficient bars for preload", anchor_sym
            )
            return out
        if "_symbol" not in bars_1min.columns:
            bars_1min = bars_1min.copy()
            bars_1min["_symbol"] = anchor_sym
        for tf, fc in self._feature_computers.items():
            fc._current_symbol = anchor_sym
            features_df = fc.compute_features_dataframe(
                bars_1min=bars_1min,
                ticks_1min=ticks_1min,
                primary_timeframe=tf,
            )
            if features_df.empty or ANCHOR_COLUMN not in features_df.columns:
                continue
            features_df.index = pd.to_datetime(features_df.index, utc=True)
            features_df = _align_feature_index_to_bar_close(features_df, tf)
            out[tf] = features_df[ANCHOR_COLUMN].copy()
        logger.info(
            "macro_tp_vwap_anchor: preloaded %s for %d timeframes",
            anchor_sym,
            len(out),
        )
        return out

    def _global_open_count(self) -> int:
        """跨所有 symbol 的全局 slot 数（仅母仓，加仓不占全局 slot）。"""
        return sum(sim.slot_position_count for sim in self._simulators.values())

    def _open_trend_positions_snapshot(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sim in self._simulators.values():
            for pos in sim._positions.values():
                if not isinstance(pos, dict) or bool(
                    pos.get("_is_add_position", False)
                ):
                    continue
                side = str(pos.get("side", "")).upper()
                entry_price = float(pos.get("entry_price", 0.0) or 0.0)
                stop_price = pos.get("stop_loss_price")
                stop_nonnegative = False
                if stop_price is not None and entry_price > 0:
                    try:
                        stop_f = float(stop_price)
                        if side in {"LONG", "BUY"}:
                            stop_nonnegative = stop_f >= entry_price
                        elif side in {"SHORT", "SELL"}:
                            stop_nonnegative = stop_f <= entry_price
                    except (TypeError, ValueError):
                        stop_nonnegative = False
                rows.append(
                    {
                        "symbol": str(pos.get("symbol", "")).upper().strip(),
                        "archetype": str(pos.get("archetype", "")).lower().strip(),
                        "side": side.lower(),
                        "breakeven_locked": bool(pos.get("breakeven_locked", False)),
                        "stop_risk_nonnegative": bool(stop_nonnegative),
                    }
                )
        return rows

    def run(
        self,
        symbols: List[str],
        days: int = 180,
        warmup_days: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fast_mode: bool = False,
        resume_state: Optional[Dict[str, Any]] = None,
        force_close_end: bool = True,
        no_kill_switch: bool = False,
        inject_add_ml_scores_path: Optional[str] = None,
        equity_anchor_usdt: Optional[float] = None,
        compound_sizing: bool = True,
    ) -> BacktestResult:
        """
        运行事件驱动回测 — 多策略 + 多 timeframe + 跨 symbol 时间线交叉处理

        时间范围:
          - 默认: end_date=now(), test_start=end_date - days
          - 指定 --start-date / --end-date: 精确控制, 用于与向量回测对齐

        equity_anchor_usdt:
          若为 None则内部权益锚点仍为 1000；CLI 传入与 --initial-capital（及 constitution
          spot.account.equity_usdt 默认覆盖）对齐，并与 spot_accum 名义 deploy 上限共用。

        compound_sizing:
          True（默认）: 每笔新开/加仓按当前权益 × risk_per_slot 反算名义（与宪法一致）。
          False: 冻结 initial 权益 sizing（legacy 对照）。
        """
        result = BacktestResult(strategy="+".join(self.strategy_names))
        funnel = defaultdict(int)

        # ── FeatureStore 补充: 检测可用的 FeatureStore layer 用于补充 IFC 缺失的特征 ──
        _fs_layers: Dict[str, str] = {}  # {strategy: layer_name}
        _fs = None
        for s in self.strategy_names:
            _det = detect_layer_for_strategy(
                strategy=s,
                features_store_root="feature_store",
                timeframe=self._tf_map.get(s),
            )
            if _det:
                _fs_layers[s] = _det
        if _fs_layers:
            _fs = FeatureStore("feature_store")
            logger.info(f"FeatureStore layers detected: {_fs_layers}")

        if end_date:
            _end = pd.Timestamp(end_date, tz="UTC")
        else:
            _end = pd.Timestamp(datetime.now(), tz="UTC")
        if start_date:
            _start = pd.Timestamp(start_date, tz="UTC")
        else:
            _start = _end - timedelta(days=days)

        end_date_str = _end.strftime("%Y-%m-%d")
        warmup_start = (_start - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        test_start_ts = _start
        logger.info(f"Time range: test={_start} → {_end}, warmup_start={warmup_start}")

        inject_scores_df: Optional[pd.DataFrame] = None
        if inject_add_ml_scores_path:
            _ip = Path(inject_add_ml_scores_path)
            if _ip.is_file():
                inject_scores_df = pd.read_parquet(_ip)
                inject_scores_df = inject_scores_df.rename(
                    columns={c: str(c).lower() for c in inject_scores_df.columns}
                )
                if "symbol" not in inject_scores_df.columns:
                    logger.warning(
                        "inject add_ml_scores parquet missing 'symbol' column: %s", _ip
                    )
                    inject_scores_df = None
                elif "timestamp" not in inject_scores_df.columns:
                    logger.warning(
                        "inject add_ml_scores parquet missing 'timestamp' column: %s",
                        _ip,
                    )
                    inject_scores_df = None
                elif not _inject_scores_has_usable_columns(inject_scores_df):
                    logger.warning(
                        "inject scores parquet missing score columns "
                        "(need score, add_ml_score, or score_long+score_short): %s",
                        _ip,
                    )
                    inject_scores_df = None
                else:
                    inject_scores_df["symbol"] = (
                        inject_scores_df["symbol"].astype(str).str.strip().str.upper()
                    )
                    inject_scores_df["timestamp"] = pd.to_datetime(
                        inject_scores_df["timestamp"], utc=True
                    )
                    logger.info(
                        "Loaded score injections: %d rows from %s",
                        len(inject_scores_df),
                        _ip,
                    )
            else:
                logger.warning("inject add_ml_scores path not found: %s", _ip)

        # 数据源: --data-path (研究数据) 或 StorageManager (实盘数据)
        use_research = self.data_path is not None
        storage = None
        if not use_research:
            storage = StorageManager(f"{self.live_root}/data")

        # ── Phase 1: 加载数据 + 按 timeframe 计算特征 ──
        sym_data: Dict[str, Dict[str, Any]] = {}
        quantile_dfs_by_tf: Dict[str, List[pd.DataFrame]] = defaultdict(list)

        _meta_full_ev: Dict[str, Any] = {}
        _pri_strat = self.strategy_names[0]
        _meta_path_ev = Path(self.strategies_root) / _pri_strat / "meta.yaml"
        try:
            if _meta_path_ev.exists():
                _meta_full_ev = (
                    yaml.safe_load(_meta_path_ev.read_text(encoding="utf-8")) or {}
                )
        except Exception as _eme:
            logger.warning("macro_tp_vwap_anchor: meta read failed: %s", _eme)
        _meta_strat_ev = _meta_full_ev.get("strategy")
        if not isinstance(_meta_strat_ev, dict):
            _meta_strat_ev = _meta_full_ev if isinstance(_meta_full_ev, dict) else {}
        _anchor_en_ev, _anchor_sym_ev = parse_macro_tp_vwap_anchor_config(
            meta_strategy=_meta_strat_ev,
            meta_yaml_full=_meta_full_ev,
        )
        _anchor_macro_cache: Dict[str, pd.Series] = {}
        _anchor_u = str(_anchor_sym_ev).strip().upper()
        _syms_iter = list(symbols)
        if _anchor_en_ev:
            _syms_iter = sorted(
                _syms_iter,
                key=lambda x: 0 if str(x).strip().upper() == _anchor_u else 1,
            )
            if not any(str(s).strip().upper() == _anchor_u for s in symbols):
                _anchor_macro_cache = self._preload_anchor_macro_cache(
                    anchor_sym=_anchor_sym_ev,
                    warmup_start=warmup_start,
                    end_date_str=end_date_str,
                    use_research=use_research,
                    storage=storage,
                )

        for sym in _syms_iter:
            logger.info(f"{'='*60}")
            logger.info(f"Loading {sym}")
            t0 = time.time()

            if use_research:
                # ── 研究数据路径: DataHandler → 1min bars + ticks ──
                bars_1min, ticks_1min = self._load_research_data(
                    sym, warmup_start, end_date_str
                )
            else:
                # ── 实盘数据路径: StorageManager ──
                bars_1min = storage.bar_1min.load_range(sym, warmup_start, end_date_str)
                ticks_1min = storage.ticks.load_range(sym, warmup_start, end_date_str)

            logger.info(
                f"  Data: {len(bars_1min)} 1min bars, {len(ticks_1min)} ticks "
                f"({time.time()-t0:.1f}s)"
            )
            if len(bars_1min) < 100:
                logger.warning(f"  {sym}: bars 不足, 跳过")
                continue

            # 注入 _symbol 列 — OI join 等特征需要识别 symbol
            if "_symbol" not in bars_1min.columns:
                bars_1min["_symbol"] = sym

            tf_features: Dict[str, pd.DataFrame] = {}
            for tf, fc in self._feature_computers.items():
                t0 = time.time()
                fc._current_symbol = sym  # for health report
                features_df = fc.compute_features_dataframe(
                    bars_1min=bars_1min,
                    ticks_1min=ticks_1min,
                    primary_timeframe=tf,
                )
                logger.info(
                    f"  Features [{tf}]: {len(features_df)} rows × "
                    f"{len(features_df.columns)} cols ({time.time()-t0:.1f}s)"
                )
                if features_df.empty:
                    continue

                # 特征健康报告
                fc.report_feature_health_df(features_df, symbol=sym, timeframe=tf)

                # ── FeatureStore 补充: 合并 IFC 缺失的特征列 ──
                if _fs and _fs_layers:
                    # 按 timeframe 匹配对应的 layer (e.g., 60T → features_me-long_60T_xxx)
                    _layer = None
                    for _s, _ln in _fs_layers.items():
                        _ln_parts = _ln.split("_")
                        if tf in _ln_parts:
                            _layer = _ln
                            break
                    if _layer is None:
                        _layer = next(iter(_fs_layers.values()))  # fallback
                    try:
                        _spec = FeatureStoreSpec(layer=_layer, symbol=sym, timeframe=tf)
                        _fs_start = features_df.index.min()
                        _fs_end = features_df.index.max()
                        if hasattr(_fs_start, "tz") and _fs_start.tz is not None:
                            _fs_start = _fs_start.tz_convert(None)
                            _fs_end = _fs_end.tz_convert(None)
                        _fs_df = _fs.read_range(_spec, start=_fs_start, end=_fs_end)
                        if not _fs_df.empty:
                            _fs_df.index = pd.to_datetime(_fs_df.index, utc=True)
                            _missing = [
                                c
                                for c in _fs_df.columns
                                if c not in features_df.columns
                            ]
                            if _missing:
                                features_df = features_df.join(
                                    _fs_df[_missing], how="left"
                                )
                                # ffill 填充 join 时间戳未对齐产生的 NaN
                                features_df[_missing] = features_df[_missing].ffill()
                                logger.info(
                                    f"  FeatureStore merged {len(_missing)} cols for {sym}/{tf}"
                                )
                            # 用 FeatureStore 填充已有列中的 NaN (e.g., IFC 无 funding_rate 数据)
                            _nan_fill = [
                                c
                                for c in _fs_df.columns
                                if c in features_df.columns
                                and features_df[c].isna().any()
                            ]
                            if _nan_fill:
                                _fs_aligned = _fs_df[_nan_fill].reindex(
                                    features_df.index, method="ffill"
                                )
                                features_df[_nan_fill] = features_df[_nan_fill].fillna(
                                    _fs_aligned
                                )
                    except Exception as e:
                        logger.warning(
                            f"  FeatureStore merge failed for {sym}/{tf}: {e}"
                        )

                features_df.index = pd.to_datetime(features_df.index, utc=True)
                # Keep decision timestamp at bar close to avoid look-ahead leakage.
                features_df = _align_feature_index_to_bar_close(features_df, tf)

                if inject_scores_df is not None:
                    sym_u = str(sym).strip().upper()
                    sub = inject_scores_df[
                        inject_scores_df["symbol"].astype(str).str.upper() == sym_u
                    ]
                    if not sub.empty:
                        sub2 = sub.sort_values("timestamp").drop_duplicates(
                            "timestamp", keep="last"
                        )
                        skip_cols = {"symbol", "timestamp"}
                        for inj_col in sub2.columns:
                            if inj_col in skip_cols:
                                continue
                            ser = pd.Series(
                                pd.to_numeric(sub2[inj_col], errors="coerce").values,
                                index=pd.DatetimeIndex(sub2["timestamp"], tz="UTC"),
                            ).sort_index()
                            aligned = ser.reindex(features_df.index).ffill()
                            features_df[inj_col] = aligned.to_numpy(
                                dtype=float, copy=False
                            )

                if _anchor_en_ev and ANCHOR_COLUMN in features_df.columns:
                    if str(sym).strip().upper() == _anchor_u:
                        _anchor_macro_cache[tf] = features_df[ANCHOR_COLUMN].copy()
                    else:
                        ser = _anchor_macro_cache.get(tf)
                        if ser is not None and len(ser) > 0:
                            fill = ser.reindex(features_df.index).ffill()
                            features_df[ANCHOR_COLUMN] = fill.to_numpy(
                                dtype=float, copy=False
                            )
                        else:
                            logger.warning(
                                "macro_tp_vwap_anchor: no anchor series for tf=%s sym=%s",
                                tf,
                                sym,
                            )

                quantile_dfs_by_tf[tf].append(features_df)

                test_df = features_df[
                    (features_df.index >= test_start_ts) & (features_df.index <= _end)
                ]
                if not test_df.empty:
                    tf_features[tf] = test_df

            if not tf_features:
                continue

            # 准备 1min bars 索引
            bars_1min_idx = bars_1min.copy()
            if not isinstance(bars_1min_idx.index, pd.DatetimeIndex):
                if "timestamp" in bars_1min_idx.columns:
                    bars_1min_idx.index = pd.to_datetime(
                        bars_1min_idx["timestamp"], utc=True
                    )
            if bars_1min_idx.index.tz is None:
                bars_1min_idx.index = bars_1min_idx.index.tz_localize("UTC")
            bars_1min_test = bars_1min_idx[
                (bars_1min_idx.index >= test_start_ts) & (bars_1min_idx.index <= _end)
            ]

            sym_data[sym] = {
                "tf_features": tf_features,
                "bars_1min_test": bars_1min_test,
            }
            for tf, tdf in tf_features.items():
                logger.info(
                    f"  Test [{tf}]: {tdf.index.min()} → {tdf.index.max()}, "
                    f"{len(tdf)} bars"
                )

        if not sym_data:
            logger.warning("No valid symbols")
            return result

        # 设置 per-strategy Evidence 分位数 (从对应 timeframe 特征)
        # Runtime quantile calibration was removed; strategies consume precomputed
        # quantile/rank features directly from the feature pipeline.

        # ── Phase 2: 构建统一时间线 (多 timeframe union) ──
        timeline_events: List[Tuple[pd.Timestamp, str, Dict[str, pd.Series]]] = []
        for sym, data in sym_data.items():
            tf_features = data["tf_features"]
            ts_to_tfs: Dict[pd.Timestamp, set] = defaultdict(set)
            for tf, test_df in tf_features.items():
                for ts in test_df.index:
                    ts_to_tfs[ts].add(tf)

            for ts in sorted(ts_to_tfs.keys()):
                tf_rows = {}
                for tf in ts_to_tfs[ts]:
                    tf_rows[tf] = tf_features[tf].loc[ts]
                timeline_events.append((ts, sym, tf_rows))

        timeline_events.sort(key=lambda x: x[0])

        # 初始化 per-symbol simulators
        for sym in sym_data:
            sim = PositionSimulator(
                default_bar_minutes=self._primary_bar_minutes,
                max_positions=len(self.strategy_names),  # 每策略独占 1 slot
                fee_rate=self.fee_rate,
            )
            if self._om_bridge:
                sim._om_bridge = self._om_bridge
            self._simulators[sym] = sim

        # spot_accum constitution: 同一批 symbol 仿真器互为 peer（共享全局/日.deploy 账本）
        _plist_sims = list(self._simulators.values())
        _spot_daily_deploy_totals: defaultdict[str, float] = defaultdict(float)
        _spot_symbol_daily_leg_counts: defaultdict[str, int] = defaultdict(int)
        for _sim in _plist_sims:
            _sim._spot_peer_sims = _plist_sims
            _sim._spot_daily_deploy_totals = _spot_daily_deploy_totals
            _sim._spot_symbol_daily_leg_counts = _spot_symbol_daily_leg_counts

        _srb_hooks = SrbEventBacktestHooks.try_from_strategies(
            self.strategy_names, self._strats
        )
        if _srb_hooks is not None:
            _srb_hooks.attach_to_simulators(self._simulators)
        else:
            for _sim in self._simulators.values():
                _sim._srb_add_policy = None
                _sim._srb_wide_entry_guard = None

        # 可选: 加载跨月续跑状态
        if resume_state:
            resume_symbols = resume_state.get("symbols", {}) or {}
            loaded_total = 0
            for sym, sym_obj in resume_symbols.items():
                sim = self._simulators.get(str(sym))
                if sim is None or not isinstance(sym_obj, dict):
                    continue
                rows = sym_obj.get("open_positions", []) or []
                loaded = sim.restore_open_positions(rows)
                loaded_total += loaded
                if loaded > 0 and hasattr(self.pcm, "_record_slot"):
                    for row in rows:
                        pos = (row or {}).get("position", {}) or {}
                        if bool(pos.get("_is_add_position", False)):
                            continue
                        arch = str(pos.get("archetype", "") or "").strip()
                        if not arch:
                            continue
                        try:
                            self.pcm._record_slot(str(sym), arch, 0.5)
                        except Exception:
                            pass
            if loaded_total > 0:
                logger.info("Resumed open positions: %d", loaded_total)

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Timeline: {len(timeline_events)} events across {len(sym_data)} symbols"
        )
        logger.info(f"Strategies: {', '.join(self.strategy_names)}")
        logger.info(f"PCM max_slots={self.pcm._max_slots}")

        # ── Phase 3: 遍历统一时间线 ──
        prev_ts: Dict[str, pd.Timestamp] = {}

        # ── Constitution Executor (复用实盘同一份代码) ──
        _executor: Optional[ConstitutionExecutor] = None
        _runtime_state = ConstitutionRuntimeState()
        _safety_state = SafetyRuntimeState()
        if resume_state:
            ap_root = resume_state.get("add_position_state")
            if isinstance(ap_root, dict) and ap_root.get("positions"):
                _load_add_position_runtime_from_resume(resume_state, _runtime_state)
            open_parent_ids = _collect_open_parent_pids(self._simulators)
            _prune_stale_add_position_records(_runtime_state, open_parent_ids)
            for _sim in self._simulators.values():
                _merge_add_position_runtime_with_open_legs(_sim, _runtime_state)
        constitution_path = str(self.constitution_yaml)
        try:
            _executor = ConstitutionExecutor(constitution_yaml=constitution_path)
            if no_kill_switch:
                import dataclasses as _dc

                object.__setattr__(
                    _executor, "cfg", _dc.replace(_executor.cfg, kill_enabled=False)
                )
                logger.info("Kill Switch 已通过 --no-kill-switch 禁用")
            elif _executor.cfg.kill_enabled:
                logger.info(
                    f"Kill Switch (共享 evaluate_safety_state): "
                    f"max_dd={_executor.cfg.max_dd:.0%}, "
                    f"daily={_executor.cfg.daily_loss_limit:.0%}, "
                    f"cooldown={_executor.cfg.cooldown_minutes}min"
                )
        except Exception as e:
            logger.warning(f"Constitution 加载失败, kill switch/加仓禁用: {e}")

        # 加仓启用检查 (从 executor 读取, 与实盘同一份 resolve 逻辑)
        _add_pos_enabled = False
        _add_pos_count = 0
        _add_pos_rejected = 0
        if _executor:
            try:
                _psl = _executor._resolve_per_strategy_limits()
                _add_pos_enabled = any(
                    isinstance(v, dict) and v.get("allow_add_position", False)
                    for v in _psl.values()
                )
                if _add_pos_enabled:
                    _max_add_vals = [
                        int(v.get("max_add_times", 1) or 1)
                        for v in _psl.values()
                        if isinstance(v, dict) and v.get("allow_add_position", False)
                    ]
                    logger.info(
                        f"加仓模拟 (共享 validate_add_position): "
                        f"max_add={max(_max_add_vals) if _max_add_vals else 1}, "
                        "trigger_r=execution.add_position"
                    )
            except Exception:
                pass
        # execution.add_position.trigger.type=float_r_ladder_only — 浮盈阶梯加仓（事件回测，不依赖 PCM 再次发信号）
        _strats_float_ladder_meta: Dict[str, Dict[str, Any]] = {}
        _add_trigger_types: Dict[str, str] = {}
        for s in self.strategy_names:
            raw = self._strats[s].archetype.execution.raw or {}
            ap = raw.get("add_position")
            _tt = ""
            if isinstance(ap, dict):
                _trg = ap.get("trigger") or {}
                if isinstance(_trg, dict):
                    _tt = str(_trg.get("type", "") or "").strip()
            _add_trigger_types[s.lower()] = _tt or "(missing trigger.type)"
            if isinstance(ap, dict) and _shared_resolve_float_r_ladder_only(ap):
                _strats_float_ladder_meta[s.lower()] = {
                    "strategy": s,
                    "add_position": dict(ap),
                    "execution_constraints": dict(
                        raw.get("execution_constraints") or {}
                    ),
                }
        result.add_trigger_types = dict(_add_trigger_types)
        _risk_per_slot = float(
            self.pcm._constitution.get("risk_per_slot", 0.01)
            if hasattr(self.pcm, "_constitution") and self.pcm._constitution
            else 0.01
        )
        _account_risk_limits = dict(
            self.pcm._constitution.get("account_risk_limits", {})
            if hasattr(self.pcm, "_constitution") and self.pcm._constitution
            else {}
        )
        try:
            _initial_cash = (
                float(equity_anchor_usdt) if equity_anchor_usdt is not None else 1000.0
            )
        except (TypeError, ValueError):
            _initial_cash = 1000.0
        if bool(_account_risk_limits.get("enabled", False)):
            _peer_sims = list(self._simulators.values())
            for _sim_ar in _peer_sims:
                _sim_ar._account_risk_limits = dict(_account_risk_limits)
                _sim_ar._account_risk_peer_sims = _peer_sims
                _sim_ar._account_risk_equity = float(_initial_cash)
        _shared_account_ledger = AccountLedger(
            account="event_backtest",
            initial_cash_usdt=_initial_cash,
        )
        _risk_usdt_per_unit = float(_initial_cash) * float(_risk_per_slot)
        for _sim_acc in self._simulators.values():
            _sim_acc._risk_per_slot_usdt = _risk_usdt_per_unit
            _sim_acc._account_ledger = _shared_account_ledger

        def _trade_realized_usdt(ct: ClosedTrade) -> float:
            rv = float(getattr(ct, "pnl_usd_realized", 0.0) or 0.0)
            has_real = (
                float(getattr(ct, "notional_usdt", 0.0) or 0.0) > 0.0
                or float(getattr(ct, "qty_base", 0.0) or 0.0) > 0.0
                or float(getattr(ct, "entry_fee_usdt", 0.0) or 0.0) > 0.0
                or float(getattr(ct, "exit_fee_usdt", 0.0) or 0.0) > 0.0
            )
            if rv == rv and has_real:
                return rv
            _rb = float(_initial_cash) * float(_risk_per_slot)
            if compound_sizing:
                _rb = max(0.0, float(_equity)) * float(_risk_per_slot)
            return (
                _rb
                * float(getattr(ct, "size_multiplier", 1.0) or 1.0)
                * float(ct.pnl_r)
            )

        _equity = _initial_cash
        _equity_peak = _equity
        _spot_cap_budget: Optional[Dict[str, Any]] = None

        def _sync_sizing_equity() -> None:
            sync_event_backtest_sizing_equity(
                simulators=self._simulators,
                equity_usdt=_equity,
                risk_per_slot=_risk_per_slot,
                compound_sizing=compound_sizing,
                initial_cash_usdt=_initial_cash,
                spot_capital_budget=_spot_cap_budget,
            )

        def _record_closed_trade_pnl(pnl_usd: float, mark_ts: pd.Timestamp) -> None:
            nonlocal _equity, _equity_peak
            _equity = max(0.0, _equity + float(pnl_usd))
            _equity_curve.append(_equity)
            _equity_curve_ts.append(mark_ts)
            if _equity > _equity_peak:
                _equity_peak = _equity
            _sync_sizing_equity()

        _equity_curve = [_equity]
        _t0_line = timeline_events[0][0] if timeline_events else None
        _equity_curve_ts: List[pd.Timestamp] = (
            [pd.Timestamp(_t0_line)] if _t0_line is not None else []
        )
        if _t0_line is None:
            logger.warning("Empty timeline: equity curve timestamps disabled")
        if fast_mode:
            logger.info(
                "Fast mode: position updates on primary TF %s bars only "
                "(not 1min-exact; SL/trailing may differ from prod).",
                self._primary_timeframe,
            )
        _ks_triggers: list = []
        _ks_skipped = 0
        _ks_executed = 0
        _period_equity_daily = _equity
        _period_equity_weekly = _equity
        _period_equity_monthly = _equity
        _prev_day = None
        _prev_week = None
        _prev_month = None

        _spot_raw_for_budget: Dict[str, Any] = {}
        try:
            _spot_raw_for_budget = (
                yaml.safe_load(Path(constitution_path).read_text(encoding="utf-8"))
                or {}
            )
        except Exception as _e_spbud:
            logger.warning(
                "Spot capital budget: constitution YAML read failed: %s", _e_spbud
            )
        _spot_cap_budget = _build_spot_capital_budget_or_none(
            constitution_raw=_spot_raw_for_budget,
            strategy_names=list(self.strategy_names),
            equity_anchor_usdt=_initial_cash,
        )
        if _spot_cap_budget is not None:
            _sb = _spot_cap_budget.get("symbol_budgets_usdt") or {}
            _su = _spot_cap_budget.get("symbol_unit_notional_usdt") or {}
            logger.info(
                "Spot capital budget (spot_accum): equity_usdt=%.2f deploy_pct=%.4f "
                "tranches_per_symbol=%d unit_usdt=%.2f symbol_budgets=%s symbol_units=%s "
                "caps gross/daily pct=(%.4f,%.4f)",
                float(_spot_cap_budget["equity_usdt"]),
                float(_spot_cap_budget["target_deploy_pct"]),
                int(_spot_cap_budget.get("tranches_per_symbol") or 0),
                float(_spot_cap_budget["unit_notional_usdt"]),
                dict(_sb) if isinstance(_sb, dict) else {},
                dict(_su) if isinstance(_su, dict) else {},
                float(_spot_cap_budget["max_gross_notional_pct"]),
                float(_spot_cap_budget["max_daily_deploy_pct"]),
            )
            for _sim_bb in self._simulators.values():
                _sim_bb._spot_capital_budget = _spot_cap_budget
        else:
            for _sim_bb in self._simulators.values():
                _sim_bb._spot_capital_budget = None

        _sync_sizing_equity()
        if compound_sizing:
            logger.info(
                "Position sizing: compound (risk_per_slot=%.4f × current equity)",
                float(_risk_per_slot),
            )
        else:
            logger.info(
                "Position sizing: fixed-base (risk_per_slot=%.4f × initial=%.2f)",
                float(_risk_per_slot),
                float(_initial_cash),
            )

        # ── 每日入场节流 (max_new_entries_per_day) ──
        _daily_entry_limits: Dict[str, Optional[int]] = {}
        if _executor:
            for s in self.strategy_names:
                _daily_entry_limits[s.lower()] = (
                    _executor.resolve_max_new_entries_per_day(s)
                )
        _daily_entry_counts: Dict[tuple, int] = {}  # (strategy, date) -> count
        _daily_entry_limit_log = False
        for _s, _lim in _daily_entry_limits.items():
            if _lim is not None:
                if not _daily_entry_limit_log:
                    logger.info("每日入场节流:")
                    _daily_entry_limit_log = True
                logger.info(f"  {_s}: max_new_entries_per_day={_lim}")

        # _pos_last_ts: 独立跟踪每个 symbol 持仓上次被处理到的时间点
        # 与 prev_ts (信号时间) 分离, 确保跨 symbol slot 释放不延迟
        _pos_last_ts: Dict[str, pd.Timestamp] = {}
        # path_efficiency_pct（类 ER 分位）在每次加仓尝试时的快照，供 er_gated_float_ladder 设计
        _er_rows_signal_add: List[Dict[str, Any]] = []
        _er_rows_float_ladder: List[Dict[str, Any]] = []
        _add_attempt_rows: List[Dict[str, Any]] = []
        _trade_map_audit_rows: List[Dict[str, Any]] = []
        _funnel_per_bar_rows: List[Dict[str, Any]] = []
        _pcm_direction_ffill: Dict[Tuple[str, str], Dict[str, float]] = {}

        for ts, sym, tf_rows in timeline_events:
            simulator = self._simulators[sym]
            bars_1min_test = sym_data[sym]["bars_1min_test"]
            funnel["total_signals_checked"] += 1

            # ── 更新所有 symbol 的持仓到当前 ts (模拟实盘实时 bar 处理) ──
            # fast_mode=True: 用当前 timeframe bar 的 OHLC 直接更新持仓 (60x faster)
            # fast_mode=False: 用 1min bars 逐分钟更新 (精确但慢)
            for upd_sym, upd_sim in self._simulators.items():
                if not upd_sim.has_positions:
                    continue
                upd_prev = _pos_last_ts.get(upd_sym)
                if upd_prev is None or upd_prev >= ts:
                    continue

                _sym_bundle = sym_data.get(upd_sym) or {}
                if fast_mode:
                    bar_iter = _iter_update_bars_primary_tf(
                        _sym_bundle,
                        upd_prev,
                        ts,
                        self._primary_timeframe,
                    )
                else:
                    upd_bars = sym_data[upd_sym]["bars_1min_test"]
                    bar_iter = _iter_update_bars_1min(upd_bars, upd_prev, ts)
                for bar_ts, bar_row in bar_iter:
                    if fast_mode:
                        _frow = bar_row
                    else:
                        _frow = _feature_row_asof_from_sym_tf_features(
                            _sym_bundle,
                            bar_ts,
                            require_macro=True,
                        )
                    _sync_macro_tp_vwap_from_feature_row(upd_sim, _frow)
                    _sync_ema_1200_from_feature_row(upd_sim, _frow)
                    if _frow is not None and "ema_200" in _frow.index:
                        try:
                            _e = float(_frow["ema_200"])
                            if _e == _e and _e > 0.0:
                                upd_sim._structural_price = _e
                        except (TypeError, ValueError):
                            pass
                    elif not fast_mode:
                        _ema_upd = _feature_asof_from_sym_tf_features(
                            _sym_bundle,
                            bar_ts,
                            "ema_200",
                        )
                        if _ema_upd is not None:
                            upd_sim._structural_price = _ema_upd
                    if fast_mode:
                        bar_dict = _ohlc_dict_from_bar_row(bar_ts, bar_row)
                    else:
                        bar_dict = {
                            "timestamp": bar_ts,
                            "open": float(bar_row.get("open", 0)),
                            "high": float(bar_row.get("high", 0)),
                            "low": float(bar_row.get("low", 0)),
                            "close": float(bar_row.get("close", 0)),
                        }
                    closed = upd_sim.update(bar_dict)
                    for ct in closed:
                        self.pcm.notify_position_closed(upd_sym, ct.archetype)
                    for ct in closed:
                        pnl_usd = _trade_realized_usdt(ct)
                        _record_closed_trade_pnl(pnl_usd, pd.Timestamp(bar_ts))
                _pos_last_ts[upd_sym] = ts

            # ── Kill switch 检查 (复用实盘 evaluate_safety_state) ──
            _ks_blocked = False
            if _executor and _executor.cfg.kill_enabled:
                # 日/周/月 边界重置
                ts_date = ts.date() if hasattr(ts, "date") else None
                ts_week = ts.isocalendar()[1] if hasattr(ts, "isocalendar") else None
                ts_month = ts.month if hasattr(ts, "month") else None
                if ts_date and ts_date != _prev_day:
                    _period_equity_daily = _equity
                    _prev_day = ts_date
                if ts_week and ts_week != _prev_week:
                    _period_equity_weekly = _equity
                    _prev_week = ts_week
                if ts_month and ts_month != _prev_month:
                    _period_equity_monthly = _equity
                    _prev_month = ts_month

                dd = (
                    (_equity_peak - _equity) / _equity_peak if _equity_peak > 0 else 0.0
                )
                d_loss = (
                    max(0.0, (_period_equity_daily - _equity) / _period_equity_daily)
                    if _period_equity_daily > 0
                    else 0.0
                )
                w_loss = (
                    max(0.0, (_period_equity_weekly - _equity) / _period_equity_weekly)
                    if _period_equity_weekly > 0
                    else 0.0
                )
                m_loss = (
                    max(
                        0.0, (_period_equity_monthly - _equity) / _period_equity_monthly
                    )
                    if _period_equity_monthly > 0
                    else 0.0
                )

                # 调用实盘同一份 evaluate_safety_state (来自 safety_runtime.py)
                now_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                if hasattr(now_dt, "tzinfo") and now_dt.tzinfo is None:
                    now_dt = now_dt.replace(tzinfo=timezone.utc)
                _safety_decision = evaluate_safety_state(
                    state=_safety_state,
                    now=now_dt,
                    cooldown_minutes=int(_executor.cfg.cooldown_minutes),
                    daily_reset_tz=_executor.cfg.daily_reset_timezone,
                    daily_loss=d_loss,
                    weekly_loss=w_loss,
                    monthly_loss=m_loss,
                    drawdown=dd,
                    hard_violation=False,
                    data_bad=False,
                    daily_cost_mean=None,
                    daily_turnover_mean=None,
                    limits={
                        "max_dd": float(_executor.cfg.max_dd),
                        "daily_loss_limit": float(_executor.cfg.daily_loss_limit),
                        "weekly_loss_limit": float(_executor.cfg.weekly_loss_limit),
                        "monthly_loss_limit": float(_executor.cfg.monthly_loss_limit),
                        "max_turnover_mean": float(_executor.cfg.max_turnover_mean),
                        "max_cost_mean": float(_executor.cfg.max_cost_mean),
                    },
                )
                if not _safety_decision.ok:
                    _ks_blocked = True
                    _ks_triggers.append(
                        {
                            "timestamp": str(ts),
                            "reasons": list(_safety_decision.reasons),
                            "equity": _equity,
                            "dd": dd,
                        }
                    )

            # 构建 features_by_timeframe 供 PCM 路由
            features_by_tf: Dict[str, Dict[str, float]] = {}
            for tf, row in tf_rows.items():
                _fd = row_to_features(row)
                _apply_pcm_direction_ffill(sym, tf, _fd, _pcm_direction_ffill)
                features_by_tf[tf] = _fd

            # 主特征 = 第一个可用 timeframe 的特征 (PCM 回退用)
            primary_features = next(iter(features_by_tf.values()))

            if _srb_hooks is not None:
                _srb_hooks.inject_regime_features(
                    sym=sym,
                    ts=ts,
                    sym_bundle=sym_data[sym],
                    tf_srb=self._tf_map.get("srb"),
                    features_by_tf=features_by_tf,
                    primary_features=primary_features,
                )

            try:
                _pat = float(primary_features.get("atr") or 0)
                if _pat > 0:
                    simulator._primary_tf_atr = _pat
            except (TypeError, ValueError):
                pass

            SrbEventBacktestHooks.sync_wide_sr_levels_on_simulator(
                simulator, primary_features
            )

            # Phase D: 维护近 N primary close 的滚动缓存用于 recent_net_move_atr
            try:
                _pc_val = primary_features.get("close")
                if _pc_val is not None:
                    _pc_f = float(_pc_val)
                    if _pc_f == _pc_f:
                        simulator._primary_close_buffer.append(_pc_f)
                        if (
                            len(simulator._primary_close_buffer)
                            > simulator._primary_close_buffer_max
                        ):
                            simulator._primary_close_buffer.pop(0)
            except (TypeError, ValueError):
                pass

            simulator._primary_bar_count += 1

            # 更新 structural_price (EMA200) 用于 BPC trend_hold 结构性退出
            _ema_200_val = primary_features.get("ema_200")
            if _ema_200_val is not None:
                try:
                    simulator._structural_price = float(_ema_200_val)
                except (TypeError, ValueError):
                    pass
            _mv = primary_features.get("macro_tp_vwap_1200_position")
            if _mv is not None:
                try:
                    simulator._macro_tp_vwap_position = float(_mv)
                except (TypeError, ValueError):
                    pass
            else:
                simulator._macro_tp_vwap_level = None
            try:
                _pcl = primary_features.get("close")
                if _mv is not None and _pcl is not None:
                    _c = float(_pcl)
                    _m = float(_mv)
                    if _c > 0.0 and _m == _m:
                        simulator._macro_tp_vwap_level = _c * (1.0 - _m)
            except (TypeError, ValueError):
                pass

            # EMA1200 structural exit: 同步 ema_1200_position 和冻结的 EMA1200 水平
            _ev = primary_features.get("ema_1200_position")
            if _ev is not None:
                try:
                    simulator._ema_1200_position = float(_ev)
                except (TypeError, ValueError):
                    pass
            else:
                simulator._ema_1200_level = None
            try:
                if _ev is not None and _pcl is not None:
                    _c = float(_pcl)
                    _e = float(_ev)
                    if _c > 0.0 and _e == _e:
                        simulator._ema_1200_level = _c * (1.0 - _e)
            except (TypeError, ValueError):
                pass

            # weekly macro-cycle structural exit signal（投影到 primary-TF）
            _mcs = primary_features.get("weekly_macro_cycle_exit_signal")
            if _mcs is not None:
                try:
                    simulator._macro_cycle_exit_signal = float(_mcs)
                except (TypeError, ValueError):
                    pass
            else:
                simulator._macro_cycle_exit_signal = None

            _mrs = primary_features.get("abc_macro_regime_score")
            if _mrs is not None:
                try:
                    simulator._macro_regime_score = float(_mrs)
                except (TypeError, ValueError):
                    simulator._macro_regime_score = None
            else:
                simulator._macro_regime_score = None

            # LivePCM.decide() — 多策略仲裁 + 全局 slot 控制
            intents = self.pcm.decide(
                features=primary_features,
                symbol=sym,
                features_by_timeframe=features_by_tf,
                decision_time=ts,
            )
            _pcm_tr = dict(getattr(self.pcm, "_last_decide_trace", None) or {})

            for s_name, s_obj in self._strats.items():
                lf = getattr(s_obj, "_last_funnel", None) or {}
                if not lf:
                    continue
                _frow = {
                    "timestamp": ts,
                    "symbol": sym,
                    "strategy": s_name,
                    "pcm_direction_filter": lf.get("pcm_direction_filter"),
                    "prefilter": lf.get("prefilter"),
                    "gate": lf.get("gate"),
                    "entry_filter": lf.get("entry_filter"),
                    "direction": lf.get("direction"),
                    "direction_value": lf.get("direction_value"),
                    "pcm_n_candidates": int(_pcm_tr.get("all_intents", 0) or 0),
                    "pcm_n_accepted": int(len(intents)),
                    "pcm_drop_direction_policy": int(
                        _pcm_tr.get("drop_direction_policy", 0) or 0
                    ),
                    "pcm_drop_family_conflict": int(
                        _pcm_tr.get("drop_family_conflict", 0) or 0
                    ),
                    "pcm_drop_daily_limit": int(
                        _pcm_tr.get("drop_daily_limit", 0) or 0
                    ),
                    "pcm_drop_slot": int(_pcm_tr.get("drop_slot", 0) or 0),
                    "pcm_drop_trend_pool_anchor_first": int(
                        _pcm_tr.get("drop_trend_pool_anchor_first", 0) or 0
                    ),
                }
                # 供离线统计「是 gate 挡还是 prefilter 挡、触发了哪条规则」
                if lf.get("gate_reasons") is not None:
                    _frow["gate_reasons"] = lf.get("gate_reasons")
                if lf.get("prefilter_reason") is not None:
                    _frow["prefilter_reason"] = lf.get("prefilter_reason")
                if lf.get("direction_rule") is not None:
                    _frow["direction_rule"] = lf.get("direction_rule")
                if lf.get("direction_reason") is not None:
                    _frow["direction_reason"] = lf.get("direction_reason")
                if lf.get("entry_filter_reason") is not None:
                    _frow["entry_filter_reason"] = lf.get("entry_filter_reason")
                for _ak in (
                    "accumulation_policy",
                    "accumulation_score",
                    "accumulation_transition_override",
                    "prefilter_alignment_override",
                    "alignment_used",
                    "prefilter_recent_pass",
                ):
                    if _ak in lf:
                        _frow[_ak] = lf[_ak]
                _frow["kill_switch_blocked"] = bool(
                    _ks_blocked
                    and _executor is not None
                    and bool(_executor.cfg.kill_enabled)
                )
                _funnel_per_bar_rows.append(_frow)

            # NOTE: Evidence slot 竞争已移除 (改为入场门槛 + 仓位缩放)
            # _last_evictions 始终为空, 此块保留为 no-op 以保持兼容
            for evicted_sym, evicted_arch in getattr(self.pcm, "_last_evictions", []):
                ev_sim = self._simulators.get(evicted_sym)
                if ev_sim and ev_sim.has_positions:
                    ev_bars = sym_data[evicted_sym]["bars_1min_test"]
                    ev_close_price = 0.0
                    _ev_mask = ev_bars.index <= ts
                    if _ev_mask.any():
                        ev_close_price = float(ev_bars.loc[_ev_mask, "close"].iloc[-1])
                    ev_close_time = (
                        ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                    )
                    if (
                        hasattr(ev_close_time, "tzinfo")
                        and ev_close_time.tzinfo is None
                    ):
                        ev_close_time = ev_close_time.replace(tzinfo=timezone.utc)
                    ev_closed = ev_sim.close_by_archetype(
                        evicted_arch, ev_close_price, ev_close_time
                    )
                    for ct in ev_closed:
                        self.pcm.notify_position_closed(evicted_sym, ct.archetype)
                        pnl_usd = _trade_realized_usdt(ct)
                        _record_closed_trade_pnl(pnl_usd, pd.Timestamp(ts))
                    funnel.setdefault("evicted_by_evidence", 0)
                    funnel["evicted_by_evidence"] += len(ev_closed)

            if intents:
                funnel["signals_generated"] += len(intents)

                for intent in intents:
                    # Kill switch 模拟: 被暂停时拒绝新入场
                    if _ks_blocked:
                        _ks_skipped += 1
                        funnel.setdefault("reject_kill_switch", 0)
                        funnel["reject_kill_switch"] += 1
                        continue
                    _ks_executed += 1
                    # 用获胜 archetype 对应 timeframe 的特征构建入场 bar
                    winning_arch = getattr(intent, "archetype", "")
                    winning_tf = self._tf_map.get(winning_arch, "")
                    entry_feats = features_by_tf.get(winning_tf, primary_features)
                    entry_bar = {
                        "close": entry_feats.get("close", 0),
                        "high": entry_feats.get("high", 0),
                        "low": entry_feats.get("low", 0),
                        "open": entry_feats.get("open", 0),
                        "timestamp": ts,
                        "atr": entry_feats.get("atr", 0),
                    }
                    winning_bm = self._bm_map.get(
                        winning_arch, self._primary_bar_minutes
                    )

                    # ── 每日入场节流: 仅限新开仓（非加仓），检查日内上限 ──
                    _arch_lc = str(winning_arch or "").strip().lower()
                    _entry_limit = _daily_entry_limits.get(_arch_lc)
                    _ts_date = ts.date() if hasattr(ts, "date") else None
                    _is_new_entry = not any(
                        p.get("symbol") == sym
                        and str(p.get("archetype", "")).lower().strip() == _arch_lc
                        for p in simulator._positions.values()
                    )
                    if (
                        _entry_limit is not None
                        and _ts_date is not None
                        and _is_new_entry
                    ):
                        _dk = (_arch_lc, _ts_date)
                        if _daily_entry_counts.get(_dk, 0) >= _entry_limit:
                            funnel.setdefault("reject_daily_entry_limit", 0)
                            funnel["reject_daily_entry_limit"] += 1
                            continue

                    if (
                        _srb_hooks is not None
                        and _srb_hooks.reject_new_entry_wide_sr_guard(
                            arch_lc=_arch_lc,
                            is_new_entry=_is_new_entry,
                            simulator=simulator,
                            entry_feats=entry_feats,
                            intent=intent,
                            funnel=funnel,
                        )
                    ):
                        continue

                    # 2026-06-10: PCM 已将重复信号转为 add_position=True。
                    # 此时跳过 open_position()，直接走 try_add_position()。
                    _pcm_add_intent = bool(getattr(intent, "add_position", False))
                    if _pcm_add_intent and _add_pos_enabled and _executor:
                        _win_lc = str(winning_arch or "").strip().lower()
                        _tried_signal_add = _win_lc not in _strats_float_ladder_meta
                        added = None
                        if _tried_signal_add:
                            added = simulator.try_add_position(
                                intent,
                                entry_bar,
                                entry_feats,
                                executor=_executor,
                                runtime_state=_runtime_state,
                                bar_minutes=winning_bm,
                            )
                            if added:
                                _add_pos_count += 1
                                funnel.setdefault("add_position_pcm_ok", 0)
                                funnel["add_position_pcm_ok"] += 1
                            else:
                                _add_pos_rejected += 1
                                funnel.setdefault("add_position_pcm_rejected", 0)
                                funnel["add_position_pcm_rejected"] += 1
                        continue

                    opened = simulator.open_position(
                        intent, entry_bar, entry_feats, bar_minutes=winning_bm
                    )
                    if _srb_hooks is not None:
                        _srb_hooks.annotate_mother_on_open(
                            opened=opened,
                            arch_lc=_arch_lc,
                            is_new_entry=_is_new_entry,
                            simulator=simulator,
                            entry_feats=entry_feats,
                            entry_bar=entry_bar,
                        )
                    if opened is not None and _entry_limit is not None and _ts_date:
                        _op = simulator._positions.get(opened) or {}
                        _scale_in_evt = int(_op.get("_accumulate_deploys", 0) or 0) > 0
                        if not _scale_in_evt:
                            _dk2 = (_arch_lc, _ts_date)
                            _daily_entry_counts[_dk2] = (
                                _daily_entry_counts.get(_dk2, 0) + 1
                            )
                    if opened is not None:
                        # 同步 PCM slot evidence（确保下次 decide() 能识别已有持仓 → add_position）
                        if hasattr(self.pcm, "_record_slot"):
                            self.pcm._record_slot(
                                str(sym), str(winning_arch or ""), 0.5
                            )
                        _opened_pos = simulator._positions.get(opened) or {}
                        _entry_src = (
                            "pcm_scale_in"
                            if int(_opened_pos.get("_accumulate_deploys", 0) or 0) > 0
                            else "pcm_new"
                        )
                        _trade_map_audit_rows.append(
                            _trade_audit_row_from_fill(
                                strats=self._strats,
                                symbol=str(sym),
                                archetype=str(winning_arch or ""),
                                ts=ts,
                                is_add_position=False,
                                entry_source=_entry_src,
                                features=entry_feats,
                                kill_switch_blocked_at_eval=bool(_ks_blocked),
                                intent_action=str(getattr(intent, "action", "") or ""),
                            )
                        )
                    if opened is None:
                        _lor_open = str(
                            getattr(simulator, "last_open_reject_reason", "") or ""
                        )
                        _spot_budget_rej = _lor_open.startswith("spot_budget")
                        _atr_open = float(entry_feats.get("atr", 0) or 0.0)
                        _dup_open = any(
                            p.get("symbol") == sym
                            and str(p.get("archetype", "")).lower().strip() == _arch_lc
                            for p in simulator._positions.values()
                        )
                        if _spot_budget_rej:
                            _rej_key = (
                                "reject_spot_tranches_full"
                                if "tranches" in _lor_open
                                else (
                                    "reject_spot_min_interval"
                                    if "min_interval" in _lor_open
                                    else "reject_spot_capital_budget"
                                )
                            )
                            funnel.setdefault(_rej_key, 0)
                            funnel[_rej_key] += 1
                        elif _atr_open <= 0.0:
                            funnel.setdefault("reject_open_atr_nonpositive", 0)
                            funnel["reject_open_atr_nonpositive"] += 1
                        elif _dup_open and not _lor_open:
                            # 2026-06-10: 同 symbol 同 archetype 已有持仓时不应拒单，
                            # 应落入下方 _add_pos_enabled 分支尝试 signal_add（PCM 再信号加仓）。
                            # 之前此处计数 reject_open_duplicate_archetype 导致 signal_add 永为 0。
                            pass
                        elif _lor_open.startswith("account_risk"):
                            funnel.setdefault("reject_account_risk_limit", 0)
                            funnel["reject_account_risk_limit"] += 1
                        # 已有持仓，尝试加仓（trigger.type=float_r_ladder_only 时仅走下方阶梯逻辑）
                        elif _add_pos_enabled and _executor:
                            _win_lc = str(winning_arch or "").strip().lower()
                            added = None
                            _tried_signal_add = _win_lc not in _strats_float_ladder_meta
                            if _tried_signal_add:
                                added = simulator.try_add_position(
                                    intent,
                                    entry_bar,
                                    entry_feats,
                                    executor=_executor,
                                    runtime_state=_runtime_state,
                                    bar_minutes=winning_bm,
                                )
                                _add_outcome = (
                                    "ok"
                                    if added
                                    else str(
                                        getattr(
                                            simulator,
                                            "last_add_reject_reason",
                                            "",
                                        )
                                    )
                                    or "other"
                                )
                                _er_rows_signal_add.append(
                                    {
                                        "pct": _extract_path_efficiency_pct(
                                            entry_feats
                                        ),
                                        "outcome": _add_outcome,
                                    }
                                )
                                _add_attempt_rows.append(
                                    _add_attempt_snapshot(
                                        timestamp=ts,
                                        symbol=sym,
                                        archetype=winning_arch,
                                        side=getattr(intent, "action", ""),
                                        path_type="signal_add",
                                        features=entry_feats,
                                        signal=getattr(
                                            simulator,
                                            "last_add_attempt_signal",
                                            {},
                                        ),
                                        outcome=_add_outcome,
                                    )
                                )
                            if added:
                                _trade_map_audit_rows.append(
                                    _trade_audit_row_from_fill(
                                        strats=self._strats,
                                        symbol=str(sym),
                                        archetype=str(winning_arch or ""),
                                        ts=ts,
                                        is_add_position=True,
                                        entry_source="pcm_signal_add",
                                        features=entry_feats,
                                        kill_switch_blocked_at_eval=bool(_ks_blocked),
                                        intent_action=str(
                                            getattr(intent, "action", "") or ""
                                        ),
                                    )
                                )
                                _add_pos_count += 1
                                funnel.setdefault("add_position_ok", 0)
                                funnel["add_position_ok"] += 1
                            elif _tried_signal_add:
                                _add_pos_rejected += 1
                                funnel.setdefault("add_position_rejected", 0)
                                funnel["add_position_rejected"] += 1
                                _why = (
                                    str(
                                        getattr(
                                            simulator,
                                            "last_add_reject_reason",
                                            "",
                                        )
                                    )
                                    or "other"
                                )
                                if _why == "max_add_times":
                                    funnel.setdefault("reject_add_max_times", 0)
                                    funnel["reject_add_max_times"] += 1
                                elif _why == "locked_profit_required":
                                    funnel.setdefault(
                                        "reject_add_locked_profit_required", 0
                                    )
                                    funnel["reject_add_locked_profit_required"] += 1
                                elif _why == "constitution_reject":
                                    funnel.setdefault("reject_add_constitution", 0)
                                    funnel["reject_add_constitution"] += 1
                                elif str(_why).startswith("account_risk"):
                                    funnel.setdefault(
                                        "reject_add_account_risk_limit", 0
                                    )
                                    funnel["reject_add_account_risk_limit"] += 1
                                elif _why in (
                                    "trigger_not_met",
                                    "add_min_current_r",
                                    "add_bpc_breakout_mismatch",
                                    "add_trigger_feature_rules",
                                ):
                                    funnel.setdefault("reject_add_trigger", 0)
                                    funnel["reject_add_trigger"] += 1
                                    if _why == "add_min_current_r":
                                        funnel.setdefault("reject_add_detail_min_r", 0)
                                        funnel["reject_add_detail_min_r"] += 1
                                    elif _why == "add_bpc_breakout_mismatch":
                                        funnel.setdefault(
                                            "reject_add_detail_bpc_breakout", 0
                                        )
                                        funnel["reject_add_detail_bpc_breakout"] += 1
                                    elif _why == "add_trigger_feature_rules":
                                        funnel.setdefault(
                                            "reject_add_detail_me_features",
                                            0,
                                        )
                                        funnel["reject_add_detail_me_features"] += 1
                                elif _why == "no_parent_position":
                                    funnel.setdefault("reject_add_no_parent", 0)
                                    funnel["reject_add_no_parent"] += 1
                                elif _why == "srb_policy_regime_bucket":
                                    funnel.setdefault("reject_add_srb_regime_bucket", 0)
                                    funnel["reject_add_srb_regime_bucket"] += 1
                                elif _why == "srb_policy_volume_compression":
                                    funnel.setdefault(
                                        "reject_add_srb_volume_compression",
                                        0,
                                    )
                                    funnel["reject_add_srb_volume_compression"] += 1
                                elif _why.startswith("shape_gate_"):
                                    _key = f"reject_add_{_why}"
                                    funnel.setdefault(_key, 0)
                                    funnel[_key] += 1
                                else:
                                    funnel.setdefault("reject_add_other", 0)
                                    funnel["reject_add_other"] += 1
                        else:
                            funnel.setdefault("reject_max_positions", 0)
                            funnel["reject_max_positions"] += 1

            else:
                _pcm_cand = int(_pcm_tr.get("all_intents", 0) or 0)
                if _pcm_cand > 0:
                    funnel["reject_pcm_direction_policy"] = int(
                        funnel.get("reject_pcm_direction_policy", 0) or 0
                    ) + int(_pcm_tr.get("drop_direction_policy", 0) or 0)
                    funnel["reject_pcm_family_conflict"] = int(
                        funnel.get("reject_pcm_family_conflict", 0) or 0
                    ) + int(_pcm_tr.get("drop_family_conflict", 0) or 0)
                    funnel["reject_pcm_daily_throttle"] = int(
                        funnel.get("reject_pcm_daily_throttle", 0) or 0
                    ) + int(_pcm_tr.get("drop_daily_limit", 0) or 0)
                    funnel["reject_pcm_slot_full"] = int(
                        funnel.get("reject_pcm_slot_full", 0) or 0
                    ) + int(_pcm_tr.get("drop_slot", 0) or 0)
                    funnel["reject_pcm_trend_symbol_conflict"] = int(
                        funnel.get("reject_pcm_trend_symbol_conflict", 0) or 0
                    ) + int(_pcm_tr.get("drop_trend_symbol_slot_conflict", 0) or 0)
                    funnel["reject_pcm_trend_pool_anchor_first"] = int(
                        funnel.get("reject_pcm_trend_pool_anchor_first", 0) or 0
                    ) + int(_pcm_tr.get("drop_trend_pool_anchor_first", 0) or 0)
                    funnel["reject_pcm_trend_pool_unprotected_cap"] = int(
                        funnel.get("reject_pcm_trend_pool_unprotected_cap", 0) or 0
                    ) + int(_pcm_tr.get("drop_trend_pool_unprotected_cap", 0) or 0)
                    funnel["reject_pcm_trend_pool_post_unlock_cap"] = int(
                        funnel.get("reject_pcm_trend_pool_post_unlock_cap", 0) or 0
                    ) + int(_pcm_tr.get("drop_trend_pool_post_unlock_cap", 0) or 0)
                # 诊断拒绝原因: 逐策略检查 _last_funnel 确定最深到达阶段
                _had_signal = False
                _deepest = "no_evaluable_signal"  # 最浅
                for s_name, s_obj in self._strats.items():
                    lf = getattr(s_obj, "_last_funnel", {})
                    if not lf:
                        continue  # 未评估 (timeframe 不匹配 / 空特征)
                    if lf.get("pcm_direction_filter") is False:
                        if _deepest == "no_evaluable_signal":
                            _deepest = "pcm_direction_filter"
                        continue
                    if lf.get("regime") is False:
                        if _deepest in (
                            "no_evaluable_signal",
                            "pcm_direction_filter",
                        ):
                            _deepest = "regime_deny"
                        continue
                    if lf.get("regime_side_block"):
                        if _deepest in (
                            "no_evaluable_signal",
                            "pcm_direction_filter",
                            "regime_deny",
                        ):
                            _deepest = "regime_side_deny"
                        continue
                    if lf.get("prefilter") is False:
                        if _deepest in ("no_evaluable_signal", "pcm_direction_filter"):
                            _deepest = "prefilter_deny"
                        continue
                    if not lf.get("direction", False):
                        continue  # direction=0, 无信号
                    # direction != 0
                    if lf.get("gate") is False:
                        if _deepest in (
                            "no_evaluable_signal",
                            "pcm_direction_filter",
                            "prefilter_deny",
                        ):
                            _deepest = "gate_deny"
                        continue
                    # gate passed (or no gate)
                    if lf.get("entry_filter") is False:
                        if _deepest in (
                            "no_evaluable_signal",
                            "pcm_direction_filter",
                            "prefilter_deny",
                            "gate_deny",
                        ):
                            _deepest = "entry_filter_deny"
                        continue
                    # 全部通过 → 策略层已产 intent，但 pcm.decide 返回空（见 _pcm_tr）
                    _had_signal = True
                    break
                if _pcm_cand > 0:
                    # 已在上方按 _last_decide_trace 细分，不再用 _deepest 重复归因
                    pass
                elif _had_signal:
                    funnel.setdefault("reject_pcm_struct_pass_no_intent", 0)
                    funnel["reject_pcm_struct_pass_no_intent"] += 1
                elif _deepest == "pcm_direction_filter":
                    funnel["reject_pcm_direction_filter"] += 1
                elif _deepest == "regime_deny":
                    funnel.setdefault("reject_regime", 0)
                    funnel["reject_regime"] += 1
                elif _deepest == "regime_side_deny":
                    funnel.setdefault("reject_regime_side", 0)
                    funnel["reject_regime_side"] += 1
                elif _deepest == "prefilter_deny":
                    funnel["reject_prefilter_deny"] += 1
                elif _deepest == "gate_deny":
                    funnel["reject_gate_deny"] += 1
                elif _deepest == "entry_filter_deny":
                    funnel["reject_entry_filter_deny"] += 1
                else:
                    funnel["reject_no_direction"] += 1

            # 浮盈阶梯加仓: 不依赖 PCM 再次发信号，按 min_current_r_by_add 逐档检查
            if (
                _strats_float_ladder_meta
                and _add_pos_enabled
                and _executor
                and not _ks_blocked
            ):
                entry_bar_primary = {
                    "close": float(primary_features.get("close", 0) or 0),
                    "high": float(primary_features.get("high", 0) or 0),
                    "low": float(primary_features.get("low", 0) or 0),
                    "open": float(primary_features.get("open", 0) or 0),
                    "timestamp": ts,
                    "atr": float(primary_features.get("atr", 0) or 0),
                }
                pf = dict(primary_features)
                pf["equity"] = float(_equity)
                for arch_lc, meta in _strats_float_ladder_meta.items():
                    ladder_done = False
                    for _pid, pos in list(simulator._positions.items()):
                        if ladder_done:
                            break
                        if pos.get("symbol") != sym:
                            continue
                        if bool(pos.get("_is_add_position", False)):
                            continue
                        if str(pos.get("archetype", "")).strip().lower() != arch_lc:
                            continue
                        min_gap_m = float(
                            (meta.get("execution_constraints") or {}).get(
                                "min_order_interval_minutes", 0
                            )
                            or 0
                        )
                        last_add = pos.get("_last_float_ladder_add_ts")
                        if min_gap_m > 0 and last_add is not None:
                            try:
                                gap_min = (
                                    pd.Timestamp(ts) - pd.Timestamp(last_add)
                                ).total_seconds() / 60.0
                            except Exception:
                                gap_min = 1.0e9
                            if gap_min < min_gap_m:
                                continue
                        arch_disp = str(pos.get("archetype", "")).strip()
                        side = pos.get("side", "LONG")
                        action = (
                            "LONG" if str(side).upper() in ("LONG", "BUY") else "SHORT"
                        )
                        ladder_intent = TradeIntent(
                            action=action,
                            symbol=sym,
                            archetype=arch_disp,
                            execution_strategy=str(meta.get("strategy", arch_disp)),
                            add_position=True,
                            size_multiplier=float(
                                pos.get("_size_multiplier", 1.0) or 1.0
                            ),
                            execution_profile={
                                "add_position": meta["add_position"],
                            },
                        )
                        ladder_bm = self._bm_map.get(
                            arch_disp, self._primary_bar_minutes
                        )
                        added_fl = simulator.try_add_position(
                            ladder_intent,
                            entry_bar_primary,
                            pf,
                            executor=_executor,
                            runtime_state=_runtime_state,
                            bar_minutes=ladder_bm,
                            skip_signal_trigger=True,
                        )
                        _add_fl_outcome = (
                            "ok"
                            if added_fl
                            else str(
                                getattr(
                                    simulator,
                                    "last_add_reject_reason",
                                    "",
                                )
                            )
                            or "other"
                        )
                        _er_rows_float_ladder.append(
                            {
                                "pct": _extract_path_efficiency_pct(pf),
                                "outcome": _add_fl_outcome,
                            }
                        )
                        _add_attempt_rows.append(
                            _add_attempt_snapshot(
                                timestamp=ts,
                                symbol=sym,
                                archetype=arch_disp,
                                side=action,
                                path_type="float_ladder",
                                features=pf,
                                signal=getattr(
                                    simulator, "last_add_attempt_signal", {}
                                ),
                                outcome=_add_fl_outcome,
                            )
                        )
                        if added_fl:
                            _trade_map_audit_rows.append(
                                _trade_audit_row_from_fill(
                                    strats=self._strats,
                                    symbol=str(sym),
                                    archetype=str(arch_disp or ""),
                                    ts=ts,
                                    is_add_position=True,
                                    entry_source="float_ladder_add",
                                    features=dict(pf),
                                    kill_switch_blocked_at_eval=bool(_ks_blocked),
                                    intent_action=str(action),
                                )
                            )
                            pos["_last_float_ladder_add_ts"] = ts
                            _add_pos_count += 1
                            funnel.setdefault("add_position_ok", 0)
                            funnel["add_position_ok"] = (
                                int(funnel.get("add_position_ok", 0)) + 1
                            )
                            funnel.setdefault("float_ladder_add_ok", 0)
                            funnel["float_ladder_add_ok"] = (
                                int(funnel.get("float_ladder_add_ok", 0)) + 1
                            )
                            ladder_done = True

            # 更新 _pos_last_ts 确保当前 symbol 也被跟踪
            if sym not in _pos_last_ts or ts > _pos_last_ts[sym]:
                _pos_last_ts[sym] = ts
            prev_ts[sym] = ts

        # ── Phase 4: 处理最后一个信号后的 bars + 关闭残留持仓 ──
        for sym, simulator in self._simulators.items():
            data = sym_data[sym]

            # 最后一个信号后的 bars (用 _pos_last_ts 避免重复处理)
            last_update = _pos_last_ts.get(sym)
            if last_update is not None and simulator.has_positions:
                if fast_mode:
                    bar_tail = _iter_update_bars_primary_tf(
                        data,
                        last_update,
                        _end,
                        self._primary_timeframe,
                    )
                else:
                    bars_1min_test = data["bars_1min_test"]
                    bar_tail = (
                        (bar_ts, bar_row)
                        for bar_ts, bar_row in bars_1min_test[
                            bars_1min_test.index > last_update
                        ].iterrows()
                    )
                for bar_ts, bar_row in bar_tail:
                    if fast_mode:
                        bar_dict = _ohlc_dict_from_bar_row(bar_ts, bar_row)
                    else:
                        bar_dict = {
                            "timestamp": bar_ts,
                            "open": float(bar_row.get("open", 0)),
                            "high": float(bar_row.get("high", 0)),
                            "low": float(bar_row.get("low", 0)),
                            "close": float(bar_row.get("close", 0)),
                        }
                    closed = simulator.update(bar_dict)
                    for ct in closed:
                        self.pcm.notify_position_closed(sym, ct.archetype)
                        pnl_usd = _trade_realized_usdt(ct)
                        _record_closed_trade_pnl(pnl_usd, pd.Timestamp(bar_ts))

            # 关闭残留持仓
            if simulator.has_positions:
                if force_close_end:
                    # 从任一可用 timeframe 取最后收盘价
                    last_close = 0.0
                    last_time = datetime.now(timezone.utc)
                    tf_features = data["tf_features"]
                    for tf in sorted(tf_features.keys(), reverse=True):
                        tdf = tf_features[tf]
                        if not tdf.empty:
                            last_close = float(tdf.iloc[-1].get("close", 0))
                            last_time = tdf.index[-1].to_pydatetime()
                            break
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=timezone.utc)
                    _fc_closed = simulator.force_close_all(last_close, last_time)
                    _fc_ts = pd.Timestamp(last_time)
                    for ct in _fc_closed:
                        pnl_usd = _trade_realized_usdt(ct)
                        _record_closed_trade_pnl(pnl_usd, _fc_ts)
                else:
                    logger.info(
                        "%s: keep %d open positions for next run",
                        sym,
                        simulator.position_count,
                    )

            if simulator.has_positions:
                for row in simulator.snapshot_open_positions():
                    result.open_positions_end.append(
                        {
                            "symbol": sym,
                            "pid": row.get("pid"),
                            "position": row.get("position", {}),
                        }
                    )

            sym_trades = simulator.closed_trades
            result.trades.extend(sym_trades)
            result.per_symbol[sym] = sym_trades
            result.bars_1min[sym] = data["bars_1min_test"]
            logger.info(f"  {sym}: {len(sym_trades)} trades")

        result.trades.sort(key=lambda t: t.entry_time)
        result.funnel = dict(funnel)
        result.funnel_per_bar = _funnel_per_bar_rows
        result.add_attempt_rows = _add_attempt_rows
        result.trade_map_audit_rows = _trade_map_audit_rows
        result.spot_inventory_metrics = _compute_spot_inventory_metrics(
            result.trades, _spot_cap_budget
        )

        # 保存 equity curve 和 kill switch 统计
        result.equity_curve = _equity_curve
        _ts_iso: List[str] = []
        for t in _equity_curve_ts:
            ct = pd.Timestamp(t)
            if ct.tzinfo is None:
                ct = ct.tz_localize("UTC")
            _ts_iso.append(ct.isoformat())
        if _ts_iso and len(_ts_iso) == len(_equity_curve):
            result.equity_curve_ts = _ts_iso
        elif _ts_iso:
            logger.warning(
                "equity_curve_ts length %d != equity_curve %d; map will omit time axis",
                len(_ts_iso),
                len(_equity_curve),
            )
            result.equity_curve_ts = None
        else:
            result.equity_curve_ts = None

        # spot_accum：BTC/EW BH 基准 + 吸筹漏斗占比 + 已开仓 quote vs 预算曲线（对齐 equity_curve_ts）
        if isinstance(_spot_cap_budget, dict):
            _inv_agg = dict(result.spot_inventory_metrics or {})
            _inv_agg["accumulation_audit"] = _compute_spot_accum_accumulation_audit(
                _funnel_per_bar_rows,
            )
            if result.equity_curve_ts:
                _inv_agg["deploy_quote_pct_series"] = _compute_deploy_quote_pct_series(
                    result.trades,
                    result.open_positions_end,
                    _spot_cap_budget,
                    result.equity_curve_ts,
                )
            result.spot_inventory_metrics = _inv_agg
            result.spot_benchmarks = _compute_spot_buy_hold_benchmarks(
                equity_ts_iso=result.equity_curve_ts,
                bars_by_sym=result.bars_1min,
                spot_budget=_spot_cap_budget,
            )
        else:
            result.spot_benchmarks = None

        if _executor:
            try:
                _cfg = _executor.cfg
                result.constitution_execution_summary = {
                    "constitution_yaml": constitution_path,
                    "risk_per_slot": _risk_per_slot,
                    "kill_switch_enabled": bool(_cfg.kill_enabled),
                    "max_dd_limit": float(_cfg.max_dd),
                    "daily_loss_limit": float(_cfg.daily_loss_limit),
                    "weekly_loss_limit": float(_cfg.weekly_loss_limit),
                    "monthly_loss_limit": float(_cfg.monthly_loss_limit),
                    "cooldown_minutes": int(_cfg.cooldown_minutes),
                    "daily_reset_timezone": str(_cfg.daily_reset_timezone or "UTC"),
                }
            except Exception:
                pass

        if _executor and _executor.cfg.kill_enabled:
            result.kill_switch_stats = {
                "trigger_count": len(_ks_triggers),
                "trades_skipped": _ks_skipped,
                "trades_executed": _ks_executed,
                "triggers": _ks_triggers,
            }

        # 保存加仓统计
        if _add_pos_enabled:
            add_trades = [t for t in result.trades if t.is_add_position]
            add_pnl = [t.pnl_r for t in add_trades]
            add_mult = [t.size_multiplier for t in add_trades]
            max_observed_lev = max(
                (
                    float(sim.max_observed_leverage or 0.0)
                    for sim in self._simulators.values()
                ),
                default=0.0,
            )
            max_observed_notional = max(
                (
                    float(sim.max_observed_notional_frac or 0.0)
                    for sim in self._simulators.values()
                ),
                default=0.0,
            )
            result.add_position_stats = {
                "enabled": True,
                "add_count": _add_pos_count,
                "rejected_count": _add_pos_rejected,
                "add_trades": len(add_trades),
                "add_mean_r": float(np.mean(add_pnl)) if add_pnl else 0.0,
                "add_mean_size": float(np.mean(add_mult)) if add_mult else 0.0,
                "add_win_rate": (
                    float(np.mean([p > 0 for p in add_pnl])) if add_pnl else 0.0
                ),
                "reject_locked_profit_required": int(
                    funnel.get("reject_add_locked_profit_required", 0) or 0
                ),
                "max_observed_leverage": float(max_observed_lev),
                "max_observed_notional_frac": float(max_observed_notional),
                "path_efficiency_pct_at_add": {
                    "note": (
                        "path_efficiency_pct = path_efficiency_pct_f 输出的历史分位 [0,1] "
                        "(净位移/路径长度，因果 shift)；用于对照 ER-gated 加仓门槛"
                    ),
                    "signal_add_attempts": _er_pct_attempt_stats(_er_rows_signal_add),
                    "float_ladder_attempts": _er_pct_attempt_stats(
                        _er_rows_float_ladder
                    ),
                },
            }

        return result
