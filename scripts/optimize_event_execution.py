#!/usr/bin/env python3
"""
事件回测 Execution 参数 Grid Search — 找平坦高原

数据只加载一次, 每组参数重跑 simulation loop (~5-15s/combo)。
复用 backtest_execution_layer._identify_plateau() 的 CV 高原检测。

用法:
    # 单策略 (推荐: 先跑单策略确定参数, 再用联合回测验证)
    python scripts/optimize_event_execution.py --strategy bpc \\
        --start-date 2025-09-01 --end-date 2026-03-01

    # 指定 symbol 加速 (只用 BTCUSDT 快速扫描)
    python scripts/optimize_event_execution.py --strategy bpc \\
        --symbols BTCUSDT --start-date 2025-09-01 --end-date 2026-03-01

    # 自定义 grid (覆盖 execution.yaml 中的 optimization 段)
    python scripts/optimize_event_execution.py --strategy me \\
        --initial-r 1.0:0.5:4.0 --activation-r 0.5:0.5:3.0 --trail-r 0.5:0.5:3.0
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.event_backtest import (
    EventBacktester,
    BacktestResult,
    ClosedTrade,
    PositionSimulator,
)

# ═══════════════════════════════════════════════════════════════════════════
# Grid & Plateau (复用 backtest_execution_layer 的逻辑)
# ═══════════════════════════════════════════════════════════════════════════


def _parse_range_str(s: str) -> List[float]:
    """解析 'start:step:end' → [start, start+step, ..., end]"""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"格式必须是 start:step:end, 得到 '{s}'")
    lo, step, hi = float(parts[0]), float(parts[1]), float(parts[2])
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


# ═══════════════════════════════════════════════════════════════════════════
# Execution 参数合理性约束
# ═══════════════════════════════════════════════════════════════════════════
# 问题: initial_r/activation_r 比率过高时, 随机入场的理论 WR 也很高:
#   P(random_win) = initial_r / (activation_r + initial_r)
# 例如 initial_r=3, activation_r=0.5 → P(random)=85.7%, 任何信号看起来都"有效"
# 约束:
#   1. initial_r / activation_r <= MAX_SL_ACT_RATIO  (防止 WR 膨胀)
#   2. activation_r - trail_r >= MIN_WIN_R            (保证最小盈利)
#   3. trail_r <= activation_r                        (trailing stop 不能比激活距离更宽)

MAX_SL_ACT_RATIO = 3.0  # initial_r / activation_r <= 3 → 随机 WR <= 75%
MIN_WIN_R = 0.0  # activation_r - trail_r >= 0 → 最小盈利 >= 0R (不亏)


def _filter_degenerate_combos(
    param_names: List[str],
    combos: List[Tuple],
    max_ratio: float = MAX_SL_ACT_RATIO,
    min_win_r: float = MIN_WIN_R,
) -> Tuple[List[Tuple], int]:
    """过滤退化参数组合 (高 WR 但无 edge).

    Returns:
        (valid_combos, n_filtered)
    """
    # 定位参数在 combo 中的 index
    ir_idx = act_idx = trail_idx = None
    for i, name in enumerate(param_names):
        if name == "stop_loss.initial_r":
            ir_idx = i
        elif name == "stop_loss.trailing.activation_r":
            act_idx = i
        elif name == "stop_loss.trailing.trail_r":
            trail_idx = i

    if ir_idx is None and act_idx is None:
        return combos, 0  # 没有 SL/activation 参数, 不过滤

    valid = []
    for combo in combos:
        ir = combo[ir_idx] if ir_idx is not None else None
        act = combo[act_idx] if act_idx is not None else None
        trail = combo[trail_idx] if trail_idx is not None else None

        # 约束1: initial_r / activation_r <= max_ratio
        if ir is not None and act is not None:
            if act > 0 and ir / act > max_ratio:
                continue

        # 约束2: activation_r - trail_r >= min_win_r
        if act is not None and trail is not None:
            if act - trail < min_win_r - 1e-9:
                continue

        # 约束3: trail_r <= activation_r
        if act is not None and trail is not None:
            if trail > act + 1e-9:
                continue

        valid.append(combo)

    return valid, len(combos) - len(valid)


def _parse_optimization_grid(
    opt_cfg: Dict[str, Any],
) -> Tuple[List[str], List[List[float]]]:
    """从 execution.yaml optimization 段解析 grid"""
    params_cfg = opt_cfg.get("params", {})
    names, values = [], []
    for pname, cfg in params_cfg.items():
        rng = cfg.get("range", [0, 1])
        step = cfg.get("step", 0.5)
        vals = []
        v = rng[0]
        while v <= rng[1] + 1e-9:
            vals.append(round(v, 4))
            v += step
        names.append(pname)
        values.append(vals)
    return names, values


def _set_nested(d: dict, dotted_key: str, value: float) -> None:
    parts = dotted_key.split(".")
    target = d
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _identify_plateau(
    results: List[Dict[str, Any]],
    param_names: List[str],
    param_values: List[List[float]],
    top_frac: float = 0.25,
    cv_threshold: float = 0.15,
    rank_key: str = "sharpe",
) -> Dict[str, Any]:
    """复用 backtest_execution_layer._identify_plateau 的核心逻辑.

    rank_key:
      - "sharpe": legacy behavior
      - "objective_score": risk-aware objective
    """
    sorted_results = sorted(
        results, key=lambda r: r.get(rank_key, float("-inf")), reverse=True
    )
    top_n = max(3, int(len(sorted_results) * top_frac))
    top = sorted_results[:top_n]

    sharpe_values = [r["sharpe"] for r in top]
    mean_sharpe = float(np.mean(sharpe_values))
    std_sharpe = float(np.std(sharpe_values))
    cv = std_sharpe / mean_sharpe if mean_sharpe > 1e-8 else float("inf")
    is_plateau = cv < cv_threshold

    recommended = sorted_results[0]
    param_analysis = {}
    sufficient_values = {}

    for pi, pname in enumerate(param_names):
        vals = sorted(set(param_values[pi]))
        val_mean_sharpe = {}
        for v in vals:
            matching = [
                r["sharpe"] for r in results if abs(r.get(pname, -999) - v) < 1e-6
            ]
            if matching:
                val_mean_sharpe[v] = float(np.mean(matching))
        if not val_mean_sharpe:
            continue

        sorted_vals = sorted(val_mean_sharpe.keys())
        max_mean = max(val_mean_sharpe.values())
        suff_threshold = max_mean * 0.95

        sufficient_val = sorted_vals[-1]
        for v in sorted_vals:
            if val_mean_sharpe[v] >= suff_threshold:
                sufficient_val = v
                break

        at_boundary = abs(sorted_vals[-1] - sufficient_val) < 1e-6
        param_analysis[pname] = {
            "values": sorted_vals,
            "mean_sharpes": [val_mean_sharpe[v] for v in sorted_vals],
            "max_mean_sharpe": max_mean,
            "sufficient_value": sufficient_val,
            "at_boundary": at_boundary,
            "best_value": max(val_mean_sharpe, key=val_mean_sharpe.get),
        }
        sufficient_values[pname] = sufficient_val

    if sufficient_values:
        best_rank_score = float(sorted_results[0].get(rank_key, 0.0))
        if best_rank_score == 0.0:
            threshold = -0.05
        elif best_rank_score > 0.0:
            threshold = best_rank_score * 0.85
        else:
            threshold = best_rank_score * 1.15
        eligible = [
            r for r in sorted_results if r.get(rank_key, float("-inf")) >= threshold
        ]
        if eligible:

            def _deviation(r):
                return sum(
                    abs(r.get(p, 0) - sufficient_values.get(p, 0)) for p in param_names
                )

            recommended = min(eligible, key=_deviation)

    return {
        "is_plateau": is_plateau,
        "top_n": top_n,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "cv": cv,
        "best": sorted_results[0],
        "recommended": recommended,
        "param_analysis": param_analysis,
        "top_results": top,
        "all_sorted": sorted_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EventBacktester 数据缓存 + 多次模拟
# ═══════════════════════════════════════════════════════════════════════════


def _prepare_data(
    bt: EventBacktester,
    symbols: List[str],
    start_date: str,
    end_date: str,
    warmup_days: int = 100,
) -> Dict[str, Any]:
    """
    Phase 1+2 of EventBacktester.run() — 加载数据 + 计算特征 + 构建时间线.
    返回缓存 dict, 供多次 _run_simulation() 复用.
    对齐 EventBacktester.run(): 包含 FeatureStore 补充步骤 (atr_percentile 等 IFC 缺失特征).
    """
    import logging

    logger = logging.getLogger("event_backtest")

    _end = pd.Timestamp(end_date, tz="UTC")
    _start = pd.Timestamp(start_date, tz="UTC")
    warmup_start = (_start - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    end_date_str = _end.strftime("%Y-%m-%d")
    test_start_ts = _start

    use_research = bt.data_path is not None
    storage = None
    if not use_research:
        from src.live_data_stream.feature_storage import StorageManager

        storage = StorageManager(f"{bt.live_root}/data")

    # ── FeatureStore 补充: 检测可用 layer, 用于补充 IFC 缺失的特征 (如 atr_percentile) ──
    # 必须与 EventBacktester.run() 对齐, 否则依赖 FeatureStore 特征的 prefilter 全部失败
    _fs_layers: Dict[str, str] = {}
    _fs = None
    try:
        from src.feature_store import FeatureStore, FeatureStoreSpec
        from src.feature_store.layer_naming import detect_layer_for_strategy

        for s in bt.strategy_names:
            _det = detect_layer_for_strategy(
                strategy=s,
                features_store_root="feature_store",
                timeframe=bt._tf_map.get(s),
            )
            if _det:
                _fs_layers[s] = _det
        if _fs_layers:
            _fs = FeatureStore("feature_store")
            logger.info(f"FeatureStore layers detected: {_fs_layers}")
    except Exception as _fs_e:
        logger.warning(f"FeatureStore 初始化失败, 跳过补充: {_fs_e}")

    sym_data: Dict[str, Dict[str, Any]] = {}
    quantile_dfs_by_tf: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    for sym in symbols:
        logger.info(f"{'='*60}")
        logger.info(f"Loading {sym}")
        t0 = time.time()

        if use_research:
            bars_1min, ticks_1min = bt._load_research_data(
                sym, warmup_start, end_date_str
            )
        else:
            bars_1min = storage.bar_1min.load_range(sym, warmup_start, end_date_str)
            ticks_1min = storage.ticks.load_range(sym, warmup_start, end_date_str)

        logger.info(
            f"  Data: {len(bars_1min)} 1min bars, {len(ticks_1min)} ticks "
            f"({time.time()-t0:.1f}s)"
        )
        if len(bars_1min) < 100:
            logger.warning(f"  {sym}: bars 不足, 跳过")
            continue

        if "_symbol" not in bars_1min.columns:
            bars_1min["_symbol"] = sym

        tf_features: Dict[str, pd.DataFrame] = {}
        for tf, fc in bt._feature_computers.items():
            t0 = time.time()
            fc._current_symbol = sym
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
            fc.report_feature_health_df(features_df, symbol=sym, timeframe=tf)

            # ── FeatureStore 补充: 合并 IFC 缺失的特征列 (同 EventBacktester.run()) ──
            if _fs and _fs_layers:
                _layer = None
                for _s, _ln in _fs_layers.items():
                    if tf in _ln.split("_"):
                        _layer = _ln
                        break
                if _layer is None:
                    _layer = next(iter(_fs_layers.values()))
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
                            c for c in _fs_df.columns if c not in features_df.columns
                        ]
                        if _missing:
                            features_df.index = pd.to_datetime(
                                features_df.index, utc=True
                            )
                            features_df = features_df.join(_fs_df[_missing], how="left")
                            features_df[_missing] = features_df[_missing].ffill()
                            logger.info(
                                f"  FeatureStore merged {len(_missing)} cols for {sym}/{tf}"
                            )
                        # 填充已有列中的 NaN
                        _nan_fill = [
                            c
                            for c in _fs_df.columns
                            if c in features_df.columns and features_df[c].isna().any()
                        ]
                        if _nan_fill:
                            _fs_aligned = _fs_df[_nan_fill].reindex(
                                features_df.index, method="ffill"
                            )
                            features_df[_nan_fill] = features_df[_nan_fill].fillna(
                                _fs_aligned
                            )
                except Exception as _e:
                    logger.warning(f"  FeatureStore merge failed for {sym}/{tf}: {_e}")

            features_df.index = pd.to_datetime(features_df.index, utc=True)
            quantile_dfs_by_tf[tf].append(features_df)

            test_df = features_df[
                (features_df.index >= test_start_ts) & (features_df.index <= _end)
            ]
            if not test_df.empty:
                tf_features[tf] = test_df

        if not tf_features:
            continue

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
                f"  Test [{tf}]: {tdf.index.min()} → {tdf.index.max()}, {len(tdf)} bars"
            )

    # Runtime quantile calibration was removed; precomputed quantile/rank
    # features from the feature pipeline are used directly.

    # 构建时间线
    timeline_events = []
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

    return {
        "sym_data": sym_data,
        "timeline_events": timeline_events,
        "test_start_ts": test_start_ts,
        "end_ts": _end,
    }


def _run_simulation(
    bt: EventBacktester,
    cache: Dict[str, Any],
) -> BacktestResult:
    """
    Phase 3+4: 用缓存的数据跑一次完整的 simulation.
    ExecutionParamGenerator.config 应在调用前已被修改.
    """
    from scripts.event_backtest import row_to_features

    import logging

    logger = logging.getLogger("event_backtest")

    sym_data = cache["sym_data"]
    timeline_events = cache["timeline_events"]

    result = BacktestResult(strategy="+".join(bt.strategy_names))
    funnel = defaultdict(int)

    # 重置 simulators
    bt._simulators = {}
    for sym in sym_data:
        sim = PositionSimulator(
            default_bar_minutes=bt._primary_bar_minutes,
            max_positions=len(bt.strategy_names),
        )
        bt._simulators[sym] = sim

    # 重置 PCM slot 状态
    bt.pcm._slot_evidence = {}

    # Constitution
    try:
        from src.time_series_model.portfolio.constitution_executor import (
            ConstitutionExecutor,
            ConstitutionRuntimeState,
        )
        from src.time_series_model.portfolio.safety_runtime import (
            SafetyRuntimeState,
            evaluate_safety_state,
        )

        constitution_path = str(Path("config") / "constitution" / "constitution.yaml")
        _executor = ConstitutionExecutor(constitution_yaml=constitution_path)
    except Exception:
        _executor = None

    _risk_per_slot = float(
        bt.pcm._constitution.get("risk_per_slot", 0.01)
        if hasattr(bt.pcm, "_constitution") and bt.pcm._constitution
        else 0.01
    )
    _initial_cash = 1000.0
    _equity = _initial_cash
    _equity_curve = [_equity]
    _equity_peak = _equity

    _pos_last_ts: Dict[str, pd.Timestamp] = {}
    prev_ts: Dict[str, pd.Timestamp] = {}

    for ts, sym, tf_rows in timeline_events:
        simulator = bt._simulators[sym]
        bars_1min_test = sym_data[sym]["bars_1min_test"]
        funnel["total_signals_checked"] += 1

        # 更新所有 symbol 持仓到当前 ts
        for upd_sym, upd_sim in bt._simulators.items():
            if not upd_sim.has_positions:
                continue
            upd_prev = _pos_last_ts.get(upd_sym)
            if upd_prev is None or upd_prev >= ts:
                continue
            upd_bars = sym_data[upd_sym]["bars_1min_test"]
            upd_mask = (upd_bars.index > upd_prev) & (upd_bars.index <= ts)
            for bar_ts_i, bar_row in upd_bars[upd_mask].iterrows():
                bar_dict = {
                    "timestamp": bar_ts_i,
                    "open": float(bar_row.get("open", 0)),
                    "high": float(bar_row.get("high", 0)),
                    "low": float(bar_row.get("low", 0)),
                    "close": float(bar_row.get("close", 0)),
                }
                closed = upd_sim.update(bar_dict)
                for ct in closed:
                    bt.pcm.notify_position_closed(upd_sym, ct.archetype)
                for ct in closed:
                    pnl_usd = _initial_cash * _risk_per_slot * ct.pnl_r
                    _equity += pnl_usd
                    _equity = max(_equity, 0.0)
                    _equity_curve.append(_equity)
                    if _equity > _equity_peak:
                        _equity_peak = _equity
            _pos_last_ts[upd_sym] = ts

        # 构建 features
        features_by_tf: Dict[str, Dict[str, float]] = {}
        for tf, row in tf_rows.items():
            features_by_tf[tf] = row_to_features(row)
        primary_features = next(iter(features_by_tf.values()))

        _ema_200_val = primary_features.get("ema_200")
        if _ema_200_val is not None:
            try:
                simulator._structural_price = float(_ema_200_val)
            except (TypeError, ValueError):
                pass

        intents = bt.pcm.decide(
            features=primary_features,
            symbol=sym,
            features_by_timeframe=features_by_tf,
            decision_time=ts,
        )
        _pcm_tr = dict(getattr(bt.pcm, "_last_decide_trace", None) or {})

        if intents:
            funnel["signals_generated"] += len(intents)
            for intent in intents:
                winning_arch = getattr(intent, "archetype", "")
                winning_tf = bt._tf_map.get(winning_arch, "")
                entry_feats = features_by_tf.get(winning_tf, primary_features)
                entry_bar = {
                    "close": entry_feats.get("close", 0),
                    "high": entry_feats.get("high", 0),
                    "low": entry_feats.get("low", 0),
                    "open": entry_feats.get("open", 0),
                    "timestamp": ts,
                    "atr": entry_feats.get("atr", 0),
                }
                winning_bm = bt._bm_map.get(winning_arch, bt._primary_bar_minutes)
                opened = simulator.open_position(
                    intent, entry_bar, entry_feats, bar_minutes=winning_bm
                )
                if opened is None:
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
            _had_signal = False
            _deepest = "no_direction"
            for s_name, s_obj in bt._strats.items():
                lf = getattr(s_obj, "_last_funnel", {})
                if not lf:
                    continue
                if not lf.get("direction", False):
                    continue
                if lf.get("gate") is False:
                    if _deepest == "no_direction":
                        _deepest = "gate_deny"
                    continue
                if lf.get("entry_filter") is False:
                    if _deepest in ("no_direction", "gate_deny"):
                        _deepest = "entry_filter_deny"
                    continue
                _had_signal = True
                break
            if _pcm_cand > 0:
                pass
            elif _had_signal:
                funnel.setdefault("reject_pcm_struct_pass_no_intent", 0)
                funnel["reject_pcm_struct_pass_no_intent"] += 1
            elif _deepest == "gate_deny":
                funnel["reject_gate_deny"] += 1
            elif _deepest == "entry_filter_deny":
                funnel["reject_entry_filter_deny"] += 1
            else:
                funnel["reject_no_direction"] += 1

        if sym not in _pos_last_ts or ts > _pos_last_ts[sym]:
            _pos_last_ts[sym] = ts
        prev_ts[sym] = ts

    # Phase 4: 关闭残留
    for sym, simulator in bt._simulators.items():
        data = sym_data[sym]
        bars_1min_test = data["bars_1min_test"]
        last_update = _pos_last_ts.get(sym)
        if last_update is not None and simulator.has_positions:
            remaining = bars_1min_test[bars_1min_test.index > last_update]
            for bar_ts_i, bar_row in remaining.iterrows():
                bar_dict = {
                    "timestamp": bar_ts_i,
                    "open": float(bar_row.get("open", 0)),
                    "high": float(bar_row.get("high", 0)),
                    "low": float(bar_row.get("low", 0)),
                    "close": float(bar_row.get("close", 0)),
                }
                closed = simulator.update(bar_dict)
                for ct in closed:
                    bt.pcm.notify_position_closed(sym, ct.archetype)
                    pnl_usd = _initial_cash * _risk_per_slot * ct.pnl_r
                    _equity += pnl_usd
                    _equity = max(_equity, 0.0)
                    _equity_curve.append(_equity)
                    if _equity > _equity_peak:
                        _equity_peak = _equity

        if simulator.has_positions:
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
            simulator.force_close_all(last_close, last_time)

        sym_trades = simulator.closed_trades
        result.trades.extend(sym_trades)
        result.per_symbol[sym] = sym_trades

    result.trades.sort(key=lambda t: t.entry_time)
    result.funnel = dict(funnel)
    result.equity_curve = _equity_curve
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="事件回测 Execution 参数 Grid Search + 平坦高原检测",
    )
    parser.add_argument("--strategy", "-s", required=True, help="策略名 (单个)")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT",
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--strategies-root", default=None)
    parser.add_argument(
        "--initial-r",
        default=None,
        help="initial_r grid: start:step:end (e.g. 1.0:0.5:4.0)",
    )
    parser.add_argument(
        "--activation-r",
        default=None,
        help="activation_r grid: start:step:end",
    )
    parser.add_argument(
        "--trail-r",
        default=None,
        help="trail_r grid: start:step:end",
    )
    parser.add_argument(
        "--sym-r",
        default=None,
        help="对称模式: initial_r=activation_r=trail_r 三者联动, start:step:end (e.g. 1.0:0.5:4.0)",
    )
    parser.add_argument(
        "--trailing",
        action="store_true",
        help="开启 trailing stop (--sym-r 时默认已开启，也可单独用于当前 yaml 参数的开关验证)",
    )
    parser.add_argument(
        "--tp-r",
        default=None,
        help="take_profit target_r grid: start:step:end (仅 TP enabled 策略)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON 输出路径 (默认 /tmp/{strategy}_event_grid.json)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="将 recommended 参数写入 execution.yaml",
    )
    parser.add_argument(
        "--objective",
        choices=["sharpe", "risk_aware", "total_r_over_dd"],
        default="sharpe",
        help=(
            "参数选择目标: sharpe(默认) / risk_aware(sharpe-惩罚项) / "
            "total_r_over_dd( total_r/max(max_dd_r,eps) - max_dd_penalty*max_dd_r "
            "- near_stop_penalty*near_stop_rate，偏总R与控回撤)"
        ),
    )
    parser.add_argument(
        "--near-stop-threshold-r",
        type=float,
        default=-0.9,
        help="定义'近-1R亏损'阈值 (默认 -0.9)",
    )
    parser.add_argument(
        "--near-stop-penalty",
        type=float,
        default=0.0,
        help="近-1R亏损比例惩罚系数 (risk_aware / total_r_over_dd 生效)",
    )
    parser.add_argument(
        "--max-dd-penalty",
        type=float,
        default=0.0,
        help="max_drawdown_r 线性惩罚系数 (risk_aware / total_r_over_dd 生效)",
    )
    parser.add_argument(
        "--min-trades-soft",
        type=int,
        default=0,
        help="软最小交易数, 低于该值按 undertrade_penalty 惩罚",
    )
    parser.add_argument(
        "--undertrade-penalty",
        type=float,
        default=0.0,
        help="交易数不足惩罚系数 (risk_aware 生效)",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0004,
        help="单边手续费率 (默认 0.0004 = 0.04%% Binance taker, 设 0 关闭)",
    )
    args = parser.parse_args()

    # Keep original strategy casing (Linux paths are case-sensitive).
    strategy = args.strategy.strip()
    symbols = [s.strip() for s in args.symbols.split(",")]
    output_path = args.output or f"/tmp/{strategy}_event_grid.json"

    print("=" * 72)
    print(f"  🔬 事件回测 Grid Search: {strategy.upper()}")
    print("=" * 72)
    print(f"  Symbols: {symbols}")
    print(f"  Period:  {args.start_date} → {args.end_date}")
    print(f"  Objective: {args.objective}")

    # 创建 EventBacktester
    bt = EventBacktester(
        strategies=[strategy],
        strategies_root=args.strategies_root,
        data_path=args.data_path if args.data_path.lower() != "none" else None,
        fee_rate=args.fee_rate,
    )

    # 解析参数 grid
    strat_obj = bt._strats[strategy]
    exec_cfg = strat_obj.execution_generator.config
    opt_cfg = exec_cfg.get("optimization", {})

    if args.initial_r or args.activation_r or args.trail_r or args.tp_r or args.sym_r:
        # CLI 自定义 grid
        param_names = []
        param_values = []
        if args.sym_r:
            # 对称模式: 三者联动扫描，用自定义对象封装
            _vals = _parse_range_str(args.sym_r)
            param_names.append("__sym_r__")
            param_values.append(_vals)
        else:
            if args.initial_r:
                param_names.append("stop_loss.initial_r")
                param_values.append(_parse_range_str(args.initial_r))
            if args.activation_r:
                param_names.append("stop_loss.trailing.activation_r")
                param_values.append(_parse_range_str(args.activation_r))
            if args.trail_r:
                param_names.append("stop_loss.trailing.trail_r")
                param_values.append(_parse_range_str(args.trail_r))
        if args.tp_r:
            param_names.append("take_profit.target_r")
            param_values.append(_parse_range_str(args.tp_r))
    elif opt_cfg.get("enabled"):
        param_names, param_values = _parse_optimization_grid(opt_cfg)
    else:
        # 默认 grid
        param_names = [
            "stop_loss.initial_r",
            "stop_loss.trailing.activation_r",
            "stop_loss.trailing.trail_r",
        ]
        param_values = [
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        ]

    all_combos = list(itertools.product(*param_values))
    raw_total = len(all_combos)

    # ── 过滤退化参数组合 (防止 WR 膨胀) ──
    # 优先从 execution.yaml constraints 读取, 否则用全局默认值
    _constraints = opt_cfg.get("constraints", {})
    _max_ratio = _constraints.get("max_sl_activation_ratio", MAX_SL_ACT_RATIO)
    _min_win = _constraints.get("min_win_r", MIN_WIN_R)
    # 对称模式三者相等，必然满足约束，跳过过滤
    if "__sym_r__" not in param_names:
        all_combos, n_filtered = _filter_degenerate_combos(
            param_names, all_combos, max_ratio=_max_ratio, min_win_r=_min_win
        )
    else:
        n_filtered = 0
    total = len(all_combos)

    _display_names = [
        "sym_r (initial=activation=trail)" if n == "__sym_r__" else n
        for n in param_names
    ]

    print(
        f"  Grid:    {' × '.join(str(len(v)) for v in param_values)} = {raw_total} combos"
    )
    if n_filtered > 0:
        print(
            f"  ⚠️  过滤退化组合: {n_filtered} 个 "
            f"(initial_r/activation_r > {MAX_SL_ACT_RATIO} 或 trail_r > activation_r)"
        )
        print(f"  有效组合: {total}")
    print(f"  Params:  {_display_names}")
    print("=" * 72)

    # Phase 1+2: 加载数据 (只做一次)
    print("\n📦 加载数据 + 计算特征...")
    t0 = time.time()
    cache = _prepare_data(
        bt, symbols, start_date=args.start_date, end_date=args.end_date
    )
    data_time = time.time() - t0
    print(f"   ✅ 数据准备完成 ({data_time:.1f}s)")
    print(f"   Symbols: {list(cache['sym_data'].keys())}")
    print(f"   Timeline: {len(cache['timeline_events'])} events")

    # 保存原始 config
    original_config = copy.deepcopy(exec_cfg)

    # Phase 3: Grid Search
    print(f"\n🔍 Grid Search ({total} combos)...")
    results: List[Dict[str, Any]] = []
    t0_grid = time.time()

    for idx, combo in enumerate(all_combos, 1):
        # 修改策略 execution config
        modified = copy.deepcopy(original_config)
        for name, val in zip(param_names, combo):
            if name == "__sym_r__":
                # 对称模式: 三者同时设置
                _set_nested(modified, "stop_loss.initial_r", val)
                _set_nested(modified, "stop_loss.trailing.activation_r", val)
                _set_nested(modified, "stop_loss.trailing.trail_r", val)
            else:
                _set_nested(modified, name, val)
        # --trailing 开关: 强制启用 trailing
        if getattr(args, "trailing", False) or getattr(args, "sym_r", None):
            _set_nested(modified, "stop_loss.type", "trailing")
        # 确保优化 tp 时 take_profit.enabled=true
        if "take_profit.target_r" in param_names:
            _set_nested(modified, "take_profit.enabled", True)
        strat_obj.execution_generator.config = modified

        t0_sim = time.time()
        res = _run_simulation(bt, cache)
        sim_time = time.time() - t0_sim

        r = {
            "combo_idx": idx,
            "sharpe": res.sharpe,
            "mean_r": res.mean_r,
            "total_r": res.total_r,
            "win_rate": res.win_rate,
            "trades": res.n_trades,
            "max_dd_r": res.max_drawdown_r,
            "sim_time": round(sim_time, 1),
        }
        pnl_arr = np.array([t.pnl_r for t in res.trades], dtype=float)
        if len(pnl_arr) > 0:
            near_stop_rate = float(
                np.mean(pnl_arr <= float(args.near_stop_threshold_r))
            )
            loss_rate = float(np.mean(pnl_arr < 0))
        else:
            near_stop_rate = 0.0
            loss_rate = 0.0
        r["near_stop_rate"] = near_stop_rate
        r["loss_rate"] = loss_rate
        undertrade_gap = (
            max(0.0, float(args.min_trades_soft) - float(r["trades"]))
            if int(args.min_trades_soft) > 0
            else 0.0
        )
        if args.objective == "risk_aware":
            r["objective_score"] = (
                float(r["sharpe"])
                - float(args.near_stop_penalty) * near_stop_rate
                - float(args.max_dd_penalty) * float(r["max_dd_r"])
                - float(args.undertrade_penalty) * undertrade_gap
            )
        elif args.objective == "total_r_over_dd":
            max_dd = float(r["max_dd_r"])
            dd_floor = max(max_dd, 1e-6)
            r["objective_score"] = (
                float(r["total_r"]) / dd_floor
                - float(args.max_dd_penalty) * max_dd
                - float(args.near_stop_penalty) * near_stop_rate
                - float(args.undertrade_penalty) * undertrade_gap
            )
        else:
            r["objective_score"] = float(r["sharpe"])
        # Equity
        if res.equity_curve and len(res.equity_curve) > 1:
            r["equity_final"] = res.equity_curve[-1]
            peak = max(res.equity_curve)
            r["equity_max_dd_pct"] = round(
                (
                    (peak - min(res.equity_curve[res.equity_curve.index(peak) :]))
                    / peak
                    * 100
                    if peak > 0
                    else 0.0
                ),
                2,
            )
        for name, val in zip(param_names, combo):
            if name == "__sym_r__":
                r["stop_loss.initial_r"] = val
                r["stop_loss.trailing.activation_r"] = val
                r["stop_loss.trailing.trail_r"] = val
                r["sym_r"] = val  # 方便读表
            else:
                r[name] = val
        results.append(r)

        # 进度
        if idx % 5 == 0 or idx == total or idx == 1:
            elapsed = time.time() - t0_grid
            eta = elapsed / idx * (total - idx) if idx > 0 else 0
            print(
                f"   [{idx:3d}/{total}] "
                f"Score={r['objective_score']:.4f} Sharpe={r['sharpe']:.4f} Trades={r['trades']:4d} "
                f"({sim_time:.1f}s/combo, ETA {eta:.0f}s)"
            )

    total_time = time.time() - t0_grid
    print(
        f"\n   ✅ Grid Search 完成 ({total_time:.1f}s, avg {total_time/total:.1f}s/combo)"
    )

    # 恢复原始 config
    strat_obj.execution_generator.config = original_config

    # Phase 4: Plateau 分析
    print("\n📊 Plateau 分析...")
    rank_key = "objective_score"
    plateau = _identify_plateau(results, param_names, param_values, rank_key=rank_key)

    best = plateau["best"]
    rec = plateau["recommended"]

    # __sym_r__ 在结果字典里存为 sym_r，打印/保存时用展示名
    def _result_key(pn: str) -> str:
        return "sym_r" if pn == "__sym_r__" else pn

    print(f"\n{'='*72}")
    print(f"  📊 Grid Search 结果: {strategy.upper()}")
    print(f"{'='*72}")
    print(f"  Total combos:  {total}")
    print(
        f"  Plateau:       {'✅ stable' if plateau['is_plateau'] else '⚠️  unstable'} (CV={plateau['cv']:.3f})"
    )
    print()
    print(f"  🏆 Best:")
    for pn in param_names:
        print(f"     {_display_names[param_names.index(pn)]}: {best[_result_key(pn)]}")
    print(
        f"     Score={best['objective_score']:.4f}  Sharpe={best['sharpe']:.4f}  "
        f"Trades={best['trades']}  WinRate={best['win_rate']:.1%}  MeanR={best['mean_r']:.4f}  "
        f"NearStopRate={best.get('near_stop_rate', 0.0):.1%}"
    )
    print()
    print(f"  🎯 Recommended (conservative elbow):")
    for pn in param_names:
        print(f"     {_display_names[param_names.index(pn)]}: {rec[_result_key(pn)]}")
    print(
        f"     Score={rec['objective_score']:.4f}  Sharpe={rec['sharpe']:.4f}  "
        f"Trades={rec['trades']}  WinRate={rec['win_rate']:.1%}  MeanR={rec['mean_r']:.4f}  "
        f"NearStopRate={rec.get('near_stop_rate', 0.0):.1%}"
    )

    # Per-param marginal analysis
    if plateau["param_analysis"]:
        print(f"\n  📈 Per-parameter marginal analysis:")
        for pn, pa in plateau["param_analysis"].items():
            suffix = " ⚠️ at boundary" if pa["at_boundary"] else ""
            print(
                f"     {pn}: elbow={pa['sufficient_value']}, "
                f"best={pa['best_value']}, "
                f"max_mean_sharpe={pa['max_mean_sharpe']:.4f}{suffix}"
            )
            # 显示每个值的 mean sharpe
            for v, ms in zip(pa["values"], pa["mean_sharpes"]):
                marker = " ◀ elbow" if abs(v - pa["sufficient_value"]) < 1e-6 else ""
                marker += " ★ best" if abs(v - pa["best_value"]) < 1e-6 else ""
                print(f"       {v:5.1f} → {ms:.4f}{marker}")

    # Top 10
    print(f"\n  📋 Top 10:")
    header = "  Rank  " + "  ".join(
        f"{_display_names[i].split('.')[-1]:>10s}" for i, pn in enumerate(param_names)
    )
    header += "   Score  Sharpe  Trades  WinRate  MeanR  Equity"
    print(header)
    for i, r in enumerate(plateau["all_sorted"][:10], 1):
        row = f"  {i:4d}  "
        row += "  ".join(f"{r[_result_key(pn)]:10.1f}" for pn in param_names)
        eq_str = f"${r.get('equity_final', 0):.0f}" if "equity_final" in r else "-"
        row += (
            f"  {r.get('objective_score', r['sharpe']):.4f}  {r['sharpe']:.4f}  "
            f"{r['trades']:6d}  {r['win_rate']:6.1%}  {r['mean_r']:.4f}  {eq_str}"
        )
        print(row)

    print(f"{'='*72}")

    # 保存结果
    output = {
        "strategy": strategy,
        "symbols": symbols,
        "period": f"{args.start_date} → {args.end_date}",
        "total_combos": total,
        "data_load_time_s": round(data_time, 1),
        "grid_search_time_s": round(total_time, 1),
        "plateau": {
            "is_plateau": plateau["is_plateau"],
            "cv": plateau["cv"],
            "mean_sharpe": plateau["mean_sharpe"],
        },
        "best": {_result_key(pn): best[_result_key(pn)] for pn in param_names},
        "objective": args.objective,
        "objective_params": {
            "near_stop_threshold_r": args.near_stop_threshold_r,
            "near_stop_penalty": args.near_stop_penalty,
            "max_dd_penalty": args.max_dd_penalty,
            "min_trades_soft": args.min_trades_soft,
            "undertrade_penalty": args.undertrade_penalty,
        },
        "best_score": best.get("objective_score", best["sharpe"]),
        "best_sharpe": best["sharpe"],
        "best_trades": best["trades"],
        "recommended": {_result_key(pn): rec[_result_key(pn)] for pn in param_names},
        "recommended_score": rec.get("objective_score", rec["sharpe"]),
        "recommended_sharpe": rec["sharpe"],
        "recommended_trades": rec["trades"],
        "param_analysis": plateau["param_analysis"],
        "all_results": plateau["all_sorted"],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  📄 Results saved → {output_path}")

    # --promote
    if args.promote:
        import yaml

        exec_yaml_path = (
            Path(args.strategies_root or "config/strategies")
            / strategy
            / "archetypes"
            / "execution.yaml"
        )
        if exec_yaml_path.exists():
            with open(exec_yaml_path) as f:
                exec_doc = yaml.safe_load(f)
            # 更新参数
            for pn in param_names:
                if pn == "__sym_r__":
                    val = rec["sym_r"]
                    _set_nested(exec_doc, "stop_loss.initial_r", val)
                    _set_nested(exec_doc, "stop_loss.trailing.activation_r", val)
                    _set_nested(exec_doc, "stop_loss.trailing.trail_r", val)
                else:
                    _set_nested(exec_doc, pn, rec[_result_key(pn)])
            # 更新版本号
            ver = exec_doc.get("version", 0)
            exec_doc["version"] = ver + 1
            with open(exec_yaml_path, "w") as f:
                yaml.dump(exec_doc, f, default_flow_style=False, allow_unicode=True)
            print(f"  ✅ Promoted recommended params → {exec_yaml_path}")
        else:
            print(f"  ❌ execution.yaml not found: {exec_yaml_path}")

    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # 减少噪音
    logging.getLogger("src.features.registry").setLevel(logging.WARNING)
    logging.getLogger(
        "src.time_series_model.live.incremental_feature_computer"
    ).setLevel(logging.WARNING)
    sys.exit(main())
