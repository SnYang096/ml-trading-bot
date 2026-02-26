#!/usr/bin/env python3
"""
Execution Layer Backtest - 逐K线路径模拟 (Bar-by-Bar Execution Simulation)

使用 archetypes/execution.yaml 的止损/移动止损参数，对每个入场信号进行逐 bar 前向模拟。

自动行为:
    - Gate 过滤: 自动检测 gate_decision 列，无需手动指定
    - Entry Filter: 自动读取 entry_filters.yaml 中所有 enabled=true 的 filter，OR 组合
    - Grid Search: 已移至 optimize_execution_grid.py

用法:
    # 单 archetype 回测 (自动应用 gate + entry filter)
    python scripts/backtest_execution_layer.py \\
        --logs results/train_final_xxx/bpc/predictions.parquet \\
        --strategy bpc

    # 启用 Tiers + Noise Penalty
    python scripts/backtest_execution_layer.py \\
        --logs results/train_final_xxx/bpc/predictions.parquet \\
        --strategy bpc --tiers --noise-penalty \\
        --quantile-train-start 2025-02-01 --quantile-train-end 2025-08-01

    # 多 archetype PCM 仲裁回测
    python scripts/backtest_execution_layer.py \\
        --pcm bpc:results/bpc/predictions.parquet \\
             me:results/me/predictions.parquet \\
        --quantile-train-start 2025-02-01 --quantile-train-end 2025-08-01

输出:
    - Sharpe Ratio (bar-by-bar 模拟)
    - Per-symbol 交易地图 HTML 报告
    - PCM 模式: Per-archetype 统计 + 反事实分析
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import ColumnDataSource, HoverTool, Span, Range1d, Div
    from bokeh.layouts import column as bk_column
    from bokeh.resources import INLINE as BK_RESOURCES
    from bokeh.embed import file_html as bk_file_html

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

# Archetype 颜色方案 (用于交易地图)
_ARCH_PALETTE = {
    "bpc": "#2196F3",  # Blue
    "me": "#FF9800",  # Orange
    "fer": "#AB47BC",  # Purple
    "lv": "#66BB6A",  # Green
    "reversal": "#EC407A",  # Pink
}
_DEFAULT_ARCH_COLOR = "#00d4aa"  # Teal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec


# ================================================================
# Evidence Scoring + Tier Assignment
# ================================================================


def load_evidence_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/evidence.yaml 配置"""
    path = Path(strategies_root) / strategy / "archetypes" / "evidence.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_evidence_quantiles(
    df: pd.DataFrame,
    evidence_cfg: Dict[str, Any],
    silent: bool = False,
) -> Dict[str, List[float]]:
    """从 DataFrame 计算 evidence 分位数阈值。

    与实盘 GenericLiveStrategy.set_quantiles_from_df() 完全一致的逻辑。
    返回 {feature_name: [threshold_at_bin0, threshold_at_bin1, ...]}。

    用于在 backtest 中从训练/校准数据预计算阈值，
    避免用整个 OOS 数据计算导致 look-ahead bias。
    """
    evidence_list = evidence_cfg.get("evidence", [])
    quantiles: Dict[str, List[float]] = {}
    n_computed = 0

    for ev in evidence_list:
        feat = ev.get("feature", "")
        if feat not in df.columns:
            continue

        qmap = ev.get("quantile_mapping", {})
        bins = qmap.get("bins", [])
        if not bins:
            continue

        values = pd.to_numeric(df[feat], errors="coerce").dropna()
        if len(values) < 10:
            continue

        thresholds = [float(values.quantile(b)) for b in bins]
        quantiles[feat] = thresholds
        n_computed += 1

    if not silent:
        print(
            f"   📊 Evidence quantiles 已计算: {n_computed}/{len(evidence_list)} features, "
            f"基于 {len(df)} 行校准数据"
        )
        for feat, thresholds in quantiles.items():
            bins = []
            for ev in evidence_list:
                if ev.get("feature") == feat:
                    bins = ev.get("quantile_mapping", {}).get("bins", [])
                    break
            pairs = [f"{b:.2f}→{t:.4f}" for b, t in zip(bins, thresholds)]
            print(f"      {feat}: [{', '.join(pairs)}]")

    return quantiles


def compute_evidence_scores(
    df: pd.DataFrame,
    evidence_cfg: Dict[str, Any],
    silent: bool = False,
    precomputed_quantiles: Optional[Dict[str, List[float]]] = None,
) -> pd.Series:
    """
    计算 Evidence 综合评分（向量化版本）

    evidence.yaml 每条特征有:
      - feature: 特征列名
      - rank: 越低越重要（weight = 1/rank）
      - direction: positive/negative
      - quantile_mapping: { bins, labels }

    对每个 bar，按加权平均计算 composite evidence_score ∈ [0,1]。

    Args:
        precomputed_quantiles: 预计算的分位数阈值（必须）。
            格式: {feature_name: [threshold_at_bin0, ...]}。
            由 compute_evidence_quantiles() 从校准数据预先计算。
            与实盘 GenericLiveStrategy.set_quantiles_from_df() 对齐。

    Raises:
        ValueError: 如果 precomputed_quantiles 为 None（禁止 look-ahead）。
    """
    if precomputed_quantiles is None:
        raise ValueError(
            "precomputed_quantiles 不能为 None。"
            "必须先用 compute_evidence_quantiles() 从校准数据预计算分位数阈值，"
            "避免 look-ahead bias。参见 --quantile-train-start / --quantile-train-end 参数。"
        )

    evidence_list = evidence_cfg.get("evidence", [])
    if not evidence_list:
        return pd.Series(0.5, index=df.index)  # 无 evidence → 中性

    label_score_map = {
        "suppress": 0.0,
        "downweight": 0.25,
        "neutral": 0.5,
        "favor": 0.75,
        "amplify": 1.0,
    }

    weighted_sum = np.zeros(len(df))
    total_weight = 0.0
    used_features = []

    for ev in evidence_list:
        feat = ev.get("feature", "")
        if feat not in df.columns:
            continue

        rank = int(ev.get("rank", 1))
        weight = 1.0 / max(1, rank)
        direction = ev.get("direction", "positive").lower()

        qmap = ev.get("quantile_mapping", {})
        bins = qmap.get("bins", [])
        labels = qmap.get("labels", [])

        if not bins or not labels:
            continue

        # 使用预计算的分位数阈值（与实盘一致，无 look-ahead）
        values = df[feat].astype(float)
        if feat in precomputed_quantiles:
            thresholds = precomputed_quantiles[feat]
        else:
            # 该特征不在校准数据中（可能校准数据缺列），跳过
            continue

        # 向量化分箱
        scores_arr = np.full(len(df), 0.5)  # 默认中性
        for i in range(len(df)):
            v = values.iloc[i]
            if np.isnan(v):
                continue
            assigned = False
            for ti, thresh in enumerate(thresholds):
                if v <= thresh:
                    lbl = labels[ti] if ti < len(labels) else "neutral"
                    scores_arr[i] = label_score_map.get(lbl, 0.5)
                    assigned = True
                    break
            if not assigned:
                lbl = labels[-1] if labels else "neutral"
                scores_arr[i] = label_score_map.get(lbl, 0.5)

        weighted_sum += scores_arr * weight
        total_weight += weight
        used_features.append(feat)

    if total_weight > 0:
        composite = weighted_sum / total_weight
    else:
        composite = np.full(len(df), 0.5)

    composite = np.clip(composite, 0.0, 1.0)

    if not silent:
        print(
            f"   📊 Evidence score: {len(used_features)}/{len(evidence_list)} features"
        )
        cs = pd.Series(composite)
        print(
            f"      mean={cs.mean():.3f}  std={cs.std():.3f}  "
            f"min={cs.min():.3f}  p25={cs.quantile(0.25):.3f}  "
            f"p50={cs.quantile(0.5):.3f}  p75={cs.quantile(0.75):.3f}  max={cs.max():.3f}"
        )

    return pd.Series(composite, index=df.index)


def assign_tiers(
    df: pd.DataFrame,
    tiers_cfg: Dict[str, Any],
    evidence_scores: pd.Series,
    exec_config: Dict[str, Any],
    silent: bool = False,
) -> None:
    """
    根据 evidence_score 分配 tier，在 df 中写入 per-entry 执行参数列：
      _tier_name, _tier_initial_r, _tier_activation_r, _tier_trail_r, _tier_timeout, _tier_size

    score < 最低 tier 的 evidence_min → 使用全局默认参数
    """
    levels = tiers_cfg.get("levels", [])
    # 按 evidence_min 从高到低排序
    levels = sorted(levels, key=lambda x: x.get("evidence_min", 0), reverse=True)

    # 全局默认参数（fallback for score below any tier）
    sl = exec_config.get("stop_loss", {})
    trail = sl.get("trailing", {})
    holding = exec_config.get("holding", {})
    default_initial_r = float(sl.get("initial_r", 2.0))
    default_activation_r = float(trail.get("activation_r", 1.0))
    default_trail_r = float(trail.get("trail_r", 1.5))
    default_timeout = int(holding.get("time_stop_bars", 50) or 50)
    default_size = 1.0

    # 初始化为默认
    df["_tier_name"] = "default"
    df["_tier_initial_r"] = default_initial_r
    df["_tier_activation_r"] = default_activation_r
    df["_tier_trail_r"] = default_trail_r
    df["_tier_timeout"] = default_timeout
    df["_tier_size"] = default_size

    tier_counts = {}
    for level in reversed(levels):  # 从低到高赋值，高的覆盖低的
        emin = float(level.get("evidence_min", 0))
        mask = evidence_scores >= emin
        name = level.get("name", f"tier_{emin}")

        lsl = level.get("stop_loss", {})
        lt = lsl.get("trailing", {})

        df.loc[mask, "_tier_name"] = name
        df.loc[mask, "_tier_initial_r"] = float(lsl.get("initial_r", default_initial_r))
        df.loc[mask, "_tier_activation_r"] = float(
            lt.get("activation_r", default_activation_r)
        )
        df.loc[mask, "_tier_trail_r"] = float(lt.get("trail_r", default_trail_r))
        df.loc[mask, "_tier_timeout"] = int(
            level.get("time_stop_bars", default_timeout)
        )
        df.loc[mask, "_tier_size"] = float(level.get("size_multiplier", default_size))

    # 统计
    entry_mask = df["entry_direction"] != 0
    for level in levels:
        name = level.get("name", "")
        cnt = int((df.loc[entry_mask, "_tier_name"] == name).sum())
        tier_counts[name] = cnt
    cnt_default = int((df.loc[entry_mask, "_tier_name"] == "default").sum())
    if cnt_default > 0:
        tier_counts["default"] = cnt_default

    if not silent:
        total = int(entry_mask.sum())
        print(f"   🏷️  Tier assignment ({total} entries):")
        for name, cnt in tier_counts.items():
            pct = cnt / total * 100 if total > 0 else 0
            print(f"      {name}: {cnt} ({pct:.1f}%)")


# ================================================================
# Entry Filter: 入场时机过滤 (Config-Driven)
# 实现已抽取到 src/time_series_model/execution/entry_filter.py
# 此处 re-export 保持向后兼容
# ================================================================

from src.time_series_model.execution.entry_filter import (  # noqa: F401
    _OP_MAP,
    _build_mask_from_conditions,
    apply_entry_filter,
    apply_entry_filters_or,
    check_conditions_single,
    check_entry_filters_or_single,
    compute_derived_entry_features,
    get_available_filters,
    load_entry_filters_config,
    DerivedEntryFeatureState,
)


def load_execution_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/execution.yaml 配置"""
    exec_path = Path(strategies_root) / strategy / "archetypes" / "execution.yaml"
    if not exec_path.exists():
        raise FileNotFoundError(f"execution.yaml not found: {exec_path}")

    with open(exec_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gate_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/gate.yaml 配置"""
    gate_path = Path(strategies_root) / strategy / "archetypes" / "gate.yaml"
    if not gate_path.exists():
        return {}

    with open(gate_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_risk_equity_curve(
    r_returns: pd.Series,
    initial_cash: float = 1000.0,
    risk_per_slot: float = 0.01,
    stop_loss_r: float = 1.0,
    risk_per_trade_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """R-multiples 转换为基于风险的美元权益曲线

    每笔交易:
      risk_frac = risk_per_trade_series[i] if provided, else risk_per_slot
      risk_usd = equity × risk_frac
      dollar_pnl = risk_usd × realized_rr / stop_loss_r

    当 realized_rr = -stop_loss_r 时，dollar_pnl = -risk_usd（正好亏损风险预算）

    Args:
        r_returns: 每笔交易的 R-multiple (from simulate_rr_execution)
        initial_cash: 初始资金
        risk_per_slot: 每笔风险占 equity 的比例 (0.01 = 1%), 作为默认值
        stop_loss_r: 止损 R 值 (1.0 = 1×ATR)
        risk_per_trade_series: 可选, 每笔交易的风险比例 (索引对齐 r_returns)
            用于每策略不同的 risk cap (e.g. LV=0.005, BPC=0.01)

    Returns:
        dict with equity_curve, max_dd, final_equity, total_return_pct
    """
    valid = r_returns.dropna()
    if len(valid) == 0:
        return {
            "equity_curve": [],
            "max_dd": 0.0,
            "final_equity": initial_cash,
            "total_return_pct": 0.0,
        }

    # Align risk_per_trade_series with valid index
    risk_arr = None
    if risk_per_trade_series is not None:
        aligned = risk_per_trade_series.reindex(valid.index)
        risk_arr = aligned.values

    equity = initial_cash
    curve = [equity]
    peak = equity
    max_dd = 0.0

    for i, rr in enumerate(valid.values):
        risk_frac = risk_per_slot
        if risk_arr is not None and i < len(risk_arr):
            v = risk_arr[i]
            if v is not None and not (isinstance(v, float) and v != v):  # not NaN
                risk_frac = float(v)
        risk_usd = equity * risk_frac
        pnl = risk_usd * float(rr) / stop_loss_r
        equity += pnl
        equity = max(equity, 0.0)  # 不允许负 equity
        curve.append(equity)

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {
        "equity_curve": curve,
        "max_dd": max_dd,
        "final_equity": equity,
        "total_return_pct": (equity - initial_cash) / initial_cash * 100,
    }


def simulate_rr_execution(
    df: pd.DataFrame,
    exec_config: Dict[str, Any],
    atr_col: str = "atr",
    direction_col: str = "entry_direction",
    silent: bool = False,
    use_tier_params: bool = False,
    breakeven_lock_r: float = 0.0,
) -> pd.Series:
    """
    逐K线路径模拟 (Bar-by-Bar Execution Simulation)

    对每一行（视为入场点），从下一根 bar 开始逐 bar 检查：
    1. 止损 (SL)：long → low ≤ stop_price; short → high ≥ stop_price
    2. 保本锁定：MFE ≥ breakeven_lock_r × ATR → SL 移至入场价
    3. 移动止损激活：MFE ≥ activation_r × ATR
    4. 移动止损触发：激活后，价格回撤 trail_r × ATR
    5. 超时平仓：持仓超过 time_stop_bars 根 K 线

    数据要求：df 须包含同一 symbol 的连续 OHLC 时间序列（按时间排序）。
    每个 symbol 内需要有足够的后续 bar 供前向模拟。

    Args:
        use_tier_params: 是否使用 per-entry tier 参数列
                         (需要 _tier_initial_r, _tier_activation_r, _tier_trail_r, _tier_timeout)
        breakeven_lock_r: 保本锁定触发 R 值 (0 = 禁用)。浮盈达到此 R 后，
                          SL 移至入场价，不允许再亏。

    Returns:
        pd.Series: 每行对应的 realized R/R (方向=0 或 ATR 无效为 NaN)
    """
    stop_loss_cfg = exec_config.get("stop_loss", {})
    take_profit_cfg = exec_config.get("take_profit", {})
    holding_cfg = exec_config.get("holding", {})

    # 解析全局参数（tier 模式下作为 fallback）
    stop_type = stop_loss_cfg.get("type", "fixed")
    g_initial_r = float(stop_loss_cfg.get("initial_r", 2.0))

    trailing_cfg = stop_loss_cfg.get("trailing", {})
    g_activation_r = float(trailing_cfg.get("activation_r", 1.0))
    g_trail_r = float(trailing_cfg.get("trail_r", 1.5))

    tp_enabled = take_profit_cfg.get("enabled", False)
    tp_r = float(take_profit_cfg.get("target_r", 2.0)) if tp_enabled else float("inf")
    g_time_stop_bars = int(
        holding_cfg.get("time_stop_bars") or holding_cfg.get("max_holding_bars") or 50
    )

    # 检测方向列
    dir_col = None
    for col in [
        direction_col,
        "entry_direction",
        "bpc_breakout_direction",
        "direction",
    ]:
        if col in df.columns:
            dir_col = col
            break

    if dir_col is None:
        if not silent:
            print(
                f"⚠️  No direction column found. Available: {list(df.columns)[:20]}..."
            )
        return pd.Series(np.nan, index=df.index)

    # 检查 OHLC
    required = ["high", "low", "close", atr_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        if not silent:
            print(f"⚠️  Missing columns for bar-by-bar simulation: {missing}")
        return pd.Series(np.nan, index=df.index)

    # 检查 tier 参数列
    has_tier_cols = all(
        c in df.columns
        for c in [
            "_tier_initial_r",
            "_tier_activation_r",
            "_tier_trail_r",
            "_tier_timeout",
        ]
    )
    tier_mode = use_tier_params and has_tier_cols

    if not silent:
        mode_str = "per-entry tier" if tier_mode else "global"
        print(
            f"   🎯 Bar-by-bar simulation: direction={dir_col}, stop={stop_type}, params={mode_str}"
        )
        if not tier_mode:
            print(
                f"   📈 Params: initial_r={g_initial_r}, activation_r={g_activation_r}, trail_r={g_trail_r}, timeout={g_time_stop_bars}"
            )

    # 按 symbol 分组处理
    sym_col = "symbol" if "symbol" in df.columns else "_symbol"
    results = pd.Series(np.nan, index=df.index, dtype=float)
    trade_details: List[Dict[str, Any]] = []  # 交易详情 (for chart)

    total_entries = 0
    breakeven_lock_count = 0
    exit_stats = {"sl": 0, "trailing_sl": 0, "tp": 0, "timeout": 0, "no_data": 0}

    for sym, group in df.groupby(sym_col, sort=False):
        group = group.sort_index()
        idx_arr = group.index.values
        highs = group["high"].values.astype(float)
        lows = group["low"].values.astype(float)
        closes = group["close"].values.astype(float)
        atrs = group[atr_col].values.astype(float)
        directions = group[dir_col].values.astype(float)
        n = len(group)

        # Per-entry 参数数组（tier 模式）
        if tier_mode:
            t_initial_r = group["_tier_initial_r"].values.astype(float)
            t_activation_r = group["_tier_activation_r"].values.astype(float)
            t_trail_r = group["_tier_trail_r"].values.astype(float)
            t_timeout = group["_tier_timeout"].values.astype(int)

        for i in range(n):
            d = directions[i]
            a = atrs[i]

            # 跳过无效入场
            if d != 1.0 and d != -1.0:
                continue
            if np.isnan(a) or a <= 0:
                continue

            # 选择参数
            if tier_mode:
                initial_r = t_initial_r[i]
                activation_r = t_activation_r[i]
                trail_r = t_trail_r[i]
                time_stop_bars = t_timeout[i]
            else:
                initial_r = g_initial_r
                activation_r = g_activation_r
                trail_r = g_trail_r
                time_stop_bars = g_time_stop_bars

            direction = int(d)
            entry_price = closes[i]
            entry_atr = a

            # 初始止损价
            if direction == 1:
                sl_price = entry_price - initial_r * entry_atr
            else:
                sl_price = entry_price + initial_r * entry_atr

            trailing_active = False
            breakeven_locked = False
            best_price = entry_price
            exit_price = None
            exit_reason = None

            # 逐 bar 前向模拟（从 i+1 开始）
            max_j = min(i + 1 + time_stop_bars, n)
            for j in range(i + 1, max_j):
                h = highs[j]
                l = lows[j]
                if np.isnan(h) or np.isnan(l):
                    continue

                # 1. 检查止损
                if direction == 1 and l <= sl_price:
                    exit_price = sl_price
                    exit_reason = "trailing_sl" if trailing_active else "sl"
                    break
                elif direction == -1 and h >= sl_price:
                    exit_price = sl_price
                    exit_reason = "trailing_sl" if trailing_active else "sl"
                    break

                # 2. 检查止盈
                if tp_enabled:
                    if direction == 1 and h >= entry_price + tp_r * entry_atr:
                        exit_price = entry_price + tp_r * entry_atr
                        exit_reason = "tp"
                        break
                    elif direction == -1 and l <= entry_price - tp_r * entry_atr:
                        exit_price = entry_price - tp_r * entry_atr
                        exit_reason = "tp"
                        break

                # 3. 更新最优价
                if direction == 1:
                    if h > best_price:
                        best_price = h
                else:
                    if l < best_price:
                        best_price = l

                # 4. 保本锁定: MFE >= breakeven_lock_r → SL 移至入场价
                if breakeven_lock_r > 0 and not breakeven_locked:
                    mfe_r_be = abs(best_price - entry_price) / entry_atr
                    if mfe_r_be >= breakeven_lock_r:
                        breakeven_locked = True
                        if direction == 1:
                            if entry_price > sl_price:
                                sl_price = entry_price
                        else:
                            if entry_price < sl_price:
                                sl_price = entry_price

                # 5. 移动止损
                if stop_type == "trailing":
                    mfe_r = abs(best_price - entry_price) / entry_atr
                    if not trailing_active and mfe_r >= activation_r:
                        trailing_active = True
                    if trailing_active:
                        if direction == 1:
                            new_sl = best_price - trail_r * entry_atr
                            if new_sl > sl_price:
                                sl_price = new_sl
                        else:
                            new_sl = best_price + trail_r * entry_atr
                            if new_sl < sl_price:
                                sl_price = new_sl

            # 超时或数据不足
            if exit_price is None:
                if max_j > i + 1:
                    exit_price = closes[max_j - 1]
                    exit_reason = "timeout"
                else:
                    exit_stats["no_data"] += 1
                    continue

            # 计算 realized R/R
            if direction == 1:
                realized_rr = (exit_price - entry_price) / entry_atr
            else:
                realized_rr = (entry_price - exit_price) / entry_atr

            results.iloc[results.index.get_loc(idx_arr[i])] = realized_rr
            total_entries += 1
            exit_stats[exit_reason] += 1
            if breakeven_locked:
                breakeven_lock_count += 1

            # 记录交易详情
            exit_bar_idx = j if exit_reason != "timeout" else max_j - 1
            trade_details.append(
                {
                    "symbol": sym,
                    "entry_idx": int(idx_arr[i]),
                    "exit_idx": int(idx_arr[min(exit_bar_idx, n - 1)]),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "direction": direction,
                    "realized_rr": float(realized_rr),
                    "exit_reason": exit_reason,
                    "archetype": (
                        str(df.iloc[i].get("_pcm_archetype", ""))
                        if "_pcm_archetype" in df.columns
                        else ""
                    ),
                }
            )

    if not silent:
        print(
            f"   📊 Simulated {total_entries} trades: "
            f"SL={exit_stats['sl']}, TrailSL={exit_stats['trailing_sl']}, "
            f"TP={exit_stats['tp']}, Timeout={exit_stats['timeout']}, "
            f"NoData={exit_stats['no_data']}"
        )
        if breakeven_lock_r > 0:
            pct = breakeven_lock_count / total_entries * 100 if total_entries > 0 else 0
            print(
                f"   🔒 Breakeven lock: {breakeven_lock_count}/{total_entries} trades ({pct:.1f}%) reached {breakeven_lock_r}R"
            )

    return results, trade_details


def compute_sharpe(
    returns: pd.Series,
    annualize: bool = False,
    span_years: float = 0.0,
    n_symbols: int = 1,
) -> float:
    """计算 Sharpe Ratio

    Args:
        returns: per-trade R 倍数序列
        annualize: 是否年化
        span_years: 数据跨度(年)。年化公式: per_trade_sharpe × √(trades_per_symbol_per_year)
        n_symbols: symbol 数量，年化时用 per-symbol 交易频率
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    mean_r = returns.mean()
    std_r = returns.std(ddof=1)

    if std_r < 1e-8:
        return 0.0

    sharpe = mean_r / std_r

    if annualize and span_years > 0:
        n_sym = max(1, n_symbols)
        trades_per_year = len(returns) / n_sym / span_years
        sharpe *= np.sqrt(trades_per_year)

    return float(sharpe)


def _estimate_span_years(df: pd.DataFrame, bars_per_year: float = 2190.0) -> float:
    """从 DataFrame 估算数据跨度(年)。按每个 symbol 的 bar 数推算。"""
    sym_col = "symbol" if "symbol" in df.columns else "_symbol"
    if sym_col not in df.columns:
        return 0.0
    bars_per_symbol = df.groupby(sym_col).size().median()
    return float(bars_per_symbol / bars_per_year)


def compute_daily_sharpe(
    df: pd.DataFrame,
    exec_returns: pd.Series,
) -> float:
    """计算日收益 Sharpe Ratio = daily_mean / daily_std × √252

    将 per-trade R-multiples 按入场日期汇总为日收益，无交易日填 0。
    这是与业界可比的 Sharpe（2~3 为优秀）。

    Args:
        df: 包含 timestamp 和 symbol 的 DataFrame
        exec_returns: 与 df 行对齐的 per-trade R-multiples (NaN = 无交易)

    Returns:
        年化日收益 Sharpe，无法计算时返回 0.0
    """
    # 找 timestamp 列
    ts_col = None
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_col = "__index_ts"
        df = df.copy()
        df[ts_col] = df.index

    if ts_col is None:
        return 0.0

    # 构建日期 + R 序列
    trade_df = pd.DataFrame(
        {
            "date": pd.to_datetime(df[ts_col]).dt.date,
            "r": exec_returns.values,
        }
    )
    # 只保留有交易的行
    trade_df = trade_df.dropna(subset=["r"])
    if len(trade_df) < 2:
        return 0.0

    # 每日 R 汇总
    daily_r = trade_df.groupby("date")["r"].sum()

    # 填充无交易日为 0
    full_range = pd.date_range(
        start=daily_r.index.min(), end=daily_r.index.max(), freq="D"
    )
    daily_r = daily_r.reindex(full_range, fill_value=0.0)

    if len(daily_r) < 10 or daily_r.std() < 1e-8:
        return 0.0

    return float(daily_r.mean() / daily_r.std() * np.sqrt(252))


def load_meta_timeframe(
    strategy: str, strategies_root: str = "config/strategies"
) -> Optional[str]:
    """从 meta.yaml 读取 archetype 的 timeframe (如 '240T', '60T')。"""
    meta_path = Path(strategies_root) / strategy / "meta.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    return meta.get("timeframe")


def _load_full_ohlc_for_map(
    features_store_root: str,
    features_store_layer: str,
    symbols: List[str],
    timeframe: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Optional[pd.DataFrame]:
    """从 FeatureStore 加载全量连续 OHLC (用于 trading map K线背景).

    Returns None if FeatureStore unavailable or load fails.
    """
    try:
        from src.feature_store import FeatureStore, FeatureStoreSpec
    except ImportError:
        return None

    if not features_store_layer:
        return None

    store = FeatureStore(str(features_store_root))
    parts: List[pd.DataFrame] = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(features_store_layer), symbol=str(sym), timeframe=str(timeframe)
        )
        try:
            df_sym = store.read_range(
                spec,
                start=start or pd.Timestamp("1970-01-01"),
                end=end or pd.Timestamp("2100-01-01"),
            )
            if df_sym.empty:
                continue
            df_sym = df_sym.copy()
            if "symbol" not in df_sym.columns:
                df_sym["symbol"] = sym
            if df_sym.index.name == "timestamp":
                df_sym = df_sym.reset_index()
            elif isinstance(df_sym.index, pd.DatetimeIndex):
                df_sym["timestamp"] = df_sym.index
                df_sym = df_sym.reset_index(drop=True)
            # 只保留 K线必需列
            keep = ["timestamp", "symbol"]
            for c in ["open", "high", "low", "close"]:
                if c in df_sym.columns:
                    keep.append(c)
            parts.append(df_sym[keep])
        except Exception as e:
            print(f"   \u26a0\ufe0f  Full OHLC load failed for {sym}: {e}")

    if not parts:
        return None

    full = pd.concat(parts, axis=0, ignore_index=True)
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    n_bars = len(full)
    n_syms = full["symbol"].nunique()
    print(f"   \U0001f5fa  Full OHLC for map: {n_bars} bars ({n_syms} symbols)")
    return full


def _generate_trading_map_html(
    df: pd.DataFrame,
    trade_details: List[Dict[str, Any]],
    title: str = "Trading Map",
    timeframe: str = None,
    arch_timeframes: Optional[Dict[str, str]] = None,
    full_ohlc: Optional[pd.DataFrame] = None,
) -> str:
    """生成 per-symbol K线 + 交易标记的 HTML 图表 (Bokeh)。

    Args:
        df: 包含 OHLC + timestamp + symbol 的 DataFrame
        trade_details: simulate_rr_execution 返回的交易详情列表
        title: 页面标题
        timeframe: K线聚合周期 (如 '240T'), None 则不聚合
        arch_timeframes: per-archetype 时间粒度 (如 {'bpc':'240T','me':'60T'})
                         提供时每个 archetype 独立分区显示

    Returns:
        完整 HTML 字符串
    """
    if not BOKEH_AVAILABLE:
        return "<p>⚠️ Bokeh not installed. pip install bokeh</p>"

    if not trade_details:
        return "<p>No trades to visualize</p>"

    # 准备 timestamp
    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["timestamp"] = df.index
        ts_col = "timestamp"
    if ts_col is None:
        return "<p>⚠️ No timestamp column for chart</p>"

    sym_col = "symbol" if "symbol" in df.columns else "_symbol"
    symbols = sorted(df[sym_col].unique())

    all_layouts = []

    # “橡皮人”: 确定分区列表 (archetype, timeframe, trades)
    if arch_timeframes and len(arch_timeframes) > 1:
        # PCM 多 archetype → 每个 archetype 独立一组图
        sections = []
        for arch_name, tf in arch_timeframes.items():
            arch_trades = [
                t
                for t in trade_details
                if t.get("archetype", "").lower() == arch_name.lower()
            ]
            if not arch_trades:
                continue
            color = _ARCH_PALETTE.get(arch_name.lower(), _DEFAULT_ARCH_COLOR)
            sections.append((arch_name, tf, arch_trades, color))
    else:
        # 单 archetype → 全部交易一组
        single_tf = timeframe
        if not single_tf and arch_timeframes:
            single_tf = list(arch_timeframes.values())[0]
        sections = [(None, single_tf, trade_details, None)]

    for section_arch, section_tf, section_trades, section_color in sections:

        for sym in symbols:
            sym_trades = [t for t in section_trades if t["symbol"] == sym]
            if not sym_trades:
                continue

            # ---- 选择 K线数据源: 优先 full_ohlc (连续), fallback df (稀疏) ----
            _use_full = False
            if full_ohlc is not None:
                _ohlc_sc = "symbol" if "symbol" in full_ohlc.columns else "_symbol"
                _sym_ohlc = full_ohlc[full_ohlc[_ohlc_sc] == sym].copy()
                if (
                    not _sym_ohlc.empty
                    and "timestamp" in _sym_ohlc.columns
                    and all(
                        c in _sym_ohlc.columns for c in ["open", "high", "low", "close"]
                    )
                ):
                    # 截取交易时间范围 ± buffer
                    _trade_ts = []
                    for _t in sym_trades:
                        try:
                            _trade_ts.append(
                                pd.Timestamp(df.loc[_t["entry_idx"], ts_col])
                            )
                            _trade_ts.append(
                                pd.Timestamp(df.loc[_t["exit_idx"], ts_col])
                            )
                        except (KeyError, ValueError):
                            pass
                    if _trade_ts:
                        _min_ts = min(_trade_ts)
                        _max_ts = max(_trade_ts)
                        _buf = max((_max_ts - _min_ts) * 0.03, pd.Timedelta(hours=96))
                        _sym_ohlc = _sym_ohlc[
                            (_sym_ohlc["timestamp"] >= _min_ts - _buf)
                            & (_sym_ohlc["timestamp"] <= _max_ts + _buf)
                        ]
                    sym_df = (
                        _sym_ohlc.rename(columns={"timestamp": ts_col})
                        if ts_col != "timestamp"
                        else _sym_ohlc
                    )
                    _use_full = True

            if not _use_full:
                sym_df = df[df[sym_col] == sym].copy()

            sym_df = sym_df.sort_values(ts_col)

            # ---- OHLC 聚合 (使用当前 section 的 timeframe) ----
            tf_for_chart = section_tf
            if tf_for_chart and not _use_full:
                # 只对稀疏数据做 resample; full_ohlc 已按正确 timeframe 存储
                if "open" not in sym_df.columns:
                    sym_df["open"] = sym_df["close"].shift(1).fillna(sym_df["close"])
                sym_df = sym_df.set_index(ts_col)
                ohlc = (
                    sym_df[["open", "high", "low", "close"]]
                    .resample(tf_for_chart)
                    .agg(
                        {"open": "first", "high": "max", "low": "min", "close": "last"}
                    )
                    .dropna()
                )
                ohlc = ohlc.reset_index()
                sym_df = ohlc
                ts_col_local = sym_df.columns[0]  # resample 后第一列是时间
            else:
                ts_col_local = ts_col

            sym_df = sym_df.reset_index(drop=True)
            if "open" not in sym_df.columns:
                sym_df["open"] = sym_df["close"].shift(1).fillna(sym_df["close"])

            # ---- 序号 x 轴 (消除时间间隙) ----
            sym_df["_seq"] = range(len(sym_df))
            x_labels = sym_df[ts_col_local].dt.strftime("%Y-%m-%d %H:%M").tolist()
            seq_to_label = {i: lbl for i, lbl in enumerate(x_labels)}
            # timestamp(str) → seq 快查
            ts_str_to_seq = dict(
                zip(
                    sym_df[ts_col_local].astype(str).values,
                    sym_df["_seq"].values,
                )
            )

            inc = sym_df["close"] >= sym_df["open"]
            dec = ~inc

            # ---- 价格图 ----
            arch_label = section_arch.upper() if section_arch else ""
            tf_label = f" ({section_tf})" if section_tf else ""
            chart_title = (
                f"{arch_label} {sym}{tf_label}" if arch_label else f"{sym}{tf_label}"
            )

            p = bk_figure(
                title=chart_title,
                width=1600,
                height=450,
                tools="pan,wheel_zoom,box_zoom,reset,save,crosshair",
                active_drag="pan",
                active_scroll="wheel_zoom",
            )
            _apply_dark_theme(p, title_size="14px")

            # Candlestick: wicks + bodies
            p.segment(
                sym_df["_seq"].values,
                sym_df["high"].values,
                sym_df["_seq"].values,
                sym_df["low"].values,
                color="#78909C",
                line_width=0.7,
            )
            w = 0.45
            if inc.any():
                p.vbar(
                    sym_df.loc[inc, "_seq"].values,
                    w,
                    sym_df.loc[inc, "open"].values,
                    sym_df.loc[inc, "close"].values,
                    fill_color="#26a69a",
                    line_color="#26a69a",
                )
            if dec.any():
                p.vbar(
                    sym_df.loc[dec, "_seq"].values,
                    w,
                    sym_df.loc[dec, "open"].values,
                    sym_df.loc[dec, "close"].values,
                    fill_color="#ef5350",
                    line_color="#ef5350",
                )

            # ---- 交易标记 ----
            # 箭头方向 = 交易方向 (long=▲, short=▼)
            # 颜色 = 盈亏 (绿=盈利, 红=亏损)
            _WIN_COLOR = "#26a69a"  # 绿 (盈利)
            _LOSS_COLOR = "#ef5350"  # 红 (亏损)

            # 按 (direction, win/loss) 分 4 组
            groups: Dict[str, Dict[str, list]] = {
                "long_win": {"x": [], "y": [], "info": []},
                "long_loss": {"x": [], "y": [], "info": []},
                "short_win": {"x": [], "y": [], "info": []},
                "short_loss": {"x": [], "y": [], "info": []},
            }
            all_rr = []

            for t in sym_trades:
                try:
                    entry_ts_str = str(df.loc[t["entry_idx"], ts_col])
                    exit_ts_str = str(df.loc[t["exit_idx"], ts_col])
                except KeyError:
                    continue
                entry_seq = ts_str_to_seq.get(entry_ts_str)
                exit_seq = ts_str_to_seq.get(exit_ts_str)
                if entry_seq is None or exit_seq is None:
                    continue

                rr = t["realized_rr"]
                all_rr.append(rr)
                is_long = t["direction"] == 1
                is_win = rr >= 0
                d_str = "L" if is_long else "S"
                arch = t.get("archetype", "")
                arch_tag = f" [{arch.upper()}]" if arch else ""
                hover_text = f"{d_str} R={rr:+.2f} ({t['exit_reason']}){arch_tag}"

                line_color = _WIN_COLOR if is_win else _LOSS_COLOR
                p.line(
                    [entry_seq, exit_seq],
                    [t["entry_price"], t["exit_price"]],
                    line_dash="dotted",
                    line_color=line_color,
                    line_width=1,
                    line_alpha=0.5,
                )

                key = f"{'long' if is_long else 'short'}_{'win' if is_win else 'loss'}"
                groups[key]["x"].append(entry_seq)
                groups[key]["y"].append(t["entry_price"])
                groups[key]["info"].append(hover_text)

            n_total = sum(len(g["x"]) for g in groups.values())
            n_w = len(groups["long_win"]["x"]) + len(groups["short_win"]["x"])
            n_l = n_total - n_w
            mean_r = sum(all_rr) / len(all_rr) if all_rr else 0

            # Long Win: ▲ 绿
            if groups["long_win"]["x"]:
                src = ColumnDataSource(groups["long_win"])
                p.scatter(
                    "x",
                    "y",
                    source=src,
                    marker="triangle",
                    size=10,
                    fill_color=_WIN_COLOR,
                    line_color="white",
                    line_width=0.5,
                    legend_label=f"Long Win ({len(groups['long_win']['x'])})",
                )
            # Long Loss: ▲ 红
            if groups["long_loss"]["x"]:
                src = ColumnDataSource(groups["long_loss"])
                p.scatter(
                    "x",
                    "y",
                    source=src,
                    marker="triangle",
                    size=10,
                    fill_color=_LOSS_COLOR,
                    line_color="white",
                    line_width=0.5,
                    alpha=0.7,
                    legend_label=f"Long Loss ({len(groups['long_loss']['x'])})",
                )
            # Short Win: ▼ 绿
            if groups["short_win"]["x"]:
                src = ColumnDataSource(groups["short_win"])
                p.scatter(
                    "x",
                    "y",
                    source=src,
                    marker="inverted_triangle",
                    size=10,
                    fill_color=_WIN_COLOR,
                    line_color="white",
                    line_width=0.5,
                    legend_label=f"Short Win ({len(groups['short_win']['x'])})",
                )
            # Short Loss: ▼ 红
            if groups["short_loss"]["x"]:
                src = ColumnDataSource(groups["short_loss"])
                p.scatter(
                    "x",
                    "y",
                    source=src,
                    marker="inverted_triangle",
                    size=10,
                    fill_color=_LOSS_COLOR,
                    line_color="white",
                    line_width=0.5,
                    alpha=0.7,
                    legend_label=f"Short Loss ({len(groups['short_loss']['x'])})",
                )

            # 总计 legend
            if n_total > 0:
                p.scatter(
                    [],
                    [],
                    marker="circle",
                    size=0,
                    legend_label=f"Total: {n_w}W/{n_l}L avg={mean_r:+.2f}R",
                )

            # 添加统计摘要信息
            stats_div = Div(
                text=f"""
            <div style="padding: 10px; background-color: #16213e; border-radius: 5px; margin-bottom: 10px;">
                <h3 style="color: #00d4aa; margin-top: 0;">交易统计摘要 - {sym}</h3>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="margin-right: 20px;">
                        <strong style="color: #e0e0e0;">总交易数:</strong> <span style="color: #4fc3f7;">{n_total}</span><br>
                        <strong style="color: #e0e0e0;">胜率:</strong> <span style="color: #4fc3f7;">{(n_w/n_total*100 if n_total > 0 else 0):.1f}%</span><br>
                        <strong style="color: #e0e0e0;">平均R回报:</strong> <span style="color: #{'26a69a' if mean_r >= 0 else 'ef5350'};">{mean_r:+.3f}R</span><br>
                    </div>
                    <div>
                        <strong style="color: #e0e0e0;">盈利交易:</strong> <span style="color: #26a69a;">{n_w}</span><br>
                        <strong style="color: #e0e0e0;">亏损交易:</strong> <span style="color: #ef5350;">{n_l}</span><br>
                        <strong style="color: #e0e0e0;">盈亏比:</strong> <span style="color: #4fc3f7;">{(n_w/n_l if n_l > 0 else float('inf')):.2f} (W/L)</span><br>
                    </div>
                </div>
            </div>
            """
            )

            # HoverTool (仅交易标记)
            hover = HoverTool(tooltips=[("Trade", "@info")], mode="mouse")
            p.add_tools(hover)

            # x 轴标签
            n_ticks = min(30, len(sym_df))
            tick_step = max(1, len(sym_df) // n_ticks)
            p.xaxis.ticker = list(range(0, len(sym_df), tick_step))
            p.xaxis.major_label_overrides = seq_to_label
            p.yaxis.axis_label = "Price"

            # 图例
            if p.legend:
                p.legend.location = "top_left"
                p.legend.background_fill_alpha = 0.6
                p.legend.background_fill_color = "#16213e"
                p.legend.label_text_color = "#e0e0e0"
                p.legend.label_text_font_size = "11px"
                p.legend.border_line_color = "#333"
                p.legend.click_policy = "hide"

            # ---- R-Multiples 柱状图 (联动 x 轴) ----
            r_fig = bk_figure(
                title="R-Multiples",
                width=1600,
                height=170,
                x_range=p.x_range,
                tools="pan,wheel_zoom,reset",
                active_drag="pan",
                active_scroll="wheel_zoom",
            )
            _apply_dark_theme(r_fig, title_size="11px")
            r_fig.add_layout(
                Span(location=0, dimension="width", line_color="#555", line_width=0.5)
            )

            for t in sym_trades:
                try:
                    t_ts_str = str(df.loc[t["entry_idx"], ts_col])
                    t_seq = ts_str_to_seq.get(t_ts_str)
                except KeyError:
                    continue
                if t_seq is None:
                    continue
                rr = t["realized_rr"]
                bar_color = "#26a69a" if rr >= 0 else "#ef5350"
                r_fig.vbar(
                    x=[t_seq],
                    top=[rr],
                    width=0.4,
                    fill_color=bar_color,
                    line_color=bar_color,
                    fill_alpha=0.7,
                )

            r_fig.xaxis.ticker = p.xaxis.ticker
            r_fig.xaxis.major_label_overrides = seq_to_label
            r_fig.yaxis.axis_label = "R"

            all_layouts.append(bk_column(stats_div, p, r_fig))

    if not all_layouts:
        return "<p>No charts generated</p>"

    full_layout = bk_column(*all_layouts)
    html = bk_file_html(full_layout, BK_RESOURCES, title=title)
    return html


def _apply_dark_theme(fig, title_size="14px"):
    """给 Bokeh figure 应用深色主题。"""
    fig.background_fill_color = "#16213e"
    fig.border_fill_color = "#0f3460"
    fig.title.text_color = "#00d4aa"
    fig.title.text_font_size = title_size
    fig.outline_line_color = None
    fig.grid.grid_line_color = "#1a3a5c"
    fig.grid.grid_line_alpha = 0.5
    fig.xaxis.axis_line_color = "#555"
    fig.xaxis.major_tick_line_color = "#555"
    fig.xaxis.major_label_text_color = "#aaa"
    fig.xaxis.major_label_orientation = 0.7
    fig.yaxis.axis_line_color = "#555"
    fig.yaxis.major_tick_line_color = "#555"
    fig.yaxis.major_label_text_color = "#aaa"
    fig.yaxis.axis_label_text_color = "#aaa"


# ================================================================
# Multi-Archetype PCM Mode
# ================================================================

# 默认优先级（与 live_pcm.py / pcm_regime.yaml v3 NORMAL 一致）
_PCM_DEFAULT_PRIORITY = ["LV", "FER", "ME", "BPC"]


def load_direction_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/direction.yaml 配置"""
    path = Path(strategies_root) / strategy / "archetypes" / "direction.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_direction_rules(
    df: pd.DataFrame,
    archetype: str,
    direction_cfg: Dict[str, Any],
) -> str:
    """根据 direction.yaml 规则确定方向列。

    按 direction_rules 优先级从高到低尝试，命中第一个可用的 method 即返回。
    如果所有规则都不匹配，返回 None（调用方应报错终止）。

    Returns:
        使用的方向列名（已写入 df['entry_direction']）或 None
    """
    rules = direction_cfg.get("direction_rules", [])

    for rule in rules:
        feature = rule.get("feature", "")
        transform = rule.get("transform", "raw")

        if feature not in df.columns:
            continue

        series = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)

        if transform == "raw":
            df["entry_direction"] = series.values
        elif transform == "sign":
            df["entry_direction"] = np.sign(series).values
        elif transform == "negate_sign":
            df["entry_direction"] = (-np.sign(series)).values
        elif transform == "center_sign":
            df["entry_direction"] = np.sign(series - 0.5).values
        else:
            df["entry_direction"] = series.values

        # replace 0 with 0 (no direction)
        desc = rule.get("description", feature)
        print(f"   Direction: {feature} (transform={transform}) | {desc}")
        return feature

    # 所有规则都不匹配
    return None


def _pcm_get_priority_rank(archetype: str, priority: List[str]) -> int:
    """获取 archetype 的优先级排名（越小越优先，与 live_pcm.py 一致）"""
    arch_lower = archetype.lower()
    for i, a in enumerate(priority):
        if a.lower() == arch_lower:
            return i
    return len(priority)  # 未知 archetype 排最后


def _compute_evidence_for_archetype(
    df: pd.DataFrame,
    arch_name: str,
    strategies_root: str,
    quantile_train_start: Optional[str],
    quantile_train_end: Optional[str],
) -> pd.Series:
    """为单个 archetype 计算 evidence scores。

    如果提供了 quantile-train-start/end，从校准数据预计算 quantiles。
    否则返回 0.5（中性）。
    """
    evidence_cfg = load_evidence_config(arch_name, strategies_root)
    if not evidence_cfg or not evidence_cfg.get("evidence"):
        return pd.Series(0.5, index=df.index)

    if not quantile_train_start or not quantile_train_end:
        return pd.Series(0.5, index=df.index)

    # 找 timestamp 列
    ts_col = None
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    elif isinstance(df.index, pd.DatetimeIndex):
        df["_ts_tmp"] = df.index
        ts_col = "_ts_tmp"

    if ts_col is None:
        print(f"   ⚠️  {arch_name}: 无 timestamp 列，evidence 默认 0.5")
        return pd.Series(0.5, index=df.index)

    train_start = pd.Timestamp(quantile_train_start)
    train_end = pd.Timestamp(quantile_train_end)
    if (train_end - train_start).days < 180:
        print(
            f"❌ 校准数据时间范围不足 6 个月: "
            f"{train_start.date()} ~ {train_end.date()}"
        )
        if "_ts_tmp" in df.columns:
            df.drop(columns=["_ts_tmp"], inplace=True)
        return pd.Series(0.5, index=df.index)

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    calib_mask = (df[ts_col] >= train_start.tz_localize("UTC")) & (
        df[ts_col] < train_end.tz_localize("UTC")
    )
    calib_df = df[calib_mask]

    if len(calib_df) < 50:
        print(
            f"   ⚠️  {arch_name}: 校准数据不足 ({len(calib_df)} 行), evidence 默认 0.5"
        )
        if "_ts_tmp" in df.columns:
            df.drop(columns=["_ts_tmp"], inplace=True)
        return pd.Series(0.5, index=df.index)

    precomputed_quantiles = compute_evidence_quantiles(
        calib_df, evidence_cfg, silent=True
    )
    scores = compute_evidence_scores(
        df, evidence_cfg, precomputed_quantiles=precomputed_quantiles, silent=True
    )
    print(
        f"   📊 {arch_name} evidence: mean={scores.mean():.3f}, "
        f"calibration={len(calib_df)} rows ({train_start.date()}~{train_end.date()})"
    )
    if "_ts_tmp" in df.columns:
        df.drop(columns=["_ts_tmp"], inplace=True)
    return scores


def _run_pcm_mode(args) -> int:  # noqa: C901
    """Multi-archetype PCM arbitration backtest mode (v3: regime-aware).

    用法:
        python scripts/backtest_execution_layer.py \\
            --pcm bpc:results/bpc/predictions.parquet \\
                  me:results/me/predictions.parquet \\
            --quantile-train-start 2025-02-01 --quantile-train-end 2025-08-01

    v3 新增: Regime 检测 + 仓位缩放，与 live_pcm.py 一致。
    """
    print("=" * 80)
    print("🎯 PCM Multi-Archetype Backtest (Regime-Aware v3)")
    print("=" * 80)

    strategies_root = args.strategies_root
    priority = _PCM_DEFAULT_PRIORITY

    # 加载 Regime 配置 (与 live 一致)
    from src.time_series_model.portfolio.live_pcm import (
        RegimeDetector,
        load_regime_config,
        _build_regime_detector,
        _load_constitution_constraints,
    )

    regime_config_path = getattr(args, "regime_config", "config/pcm_regime.yaml")
    regime_cfg = load_regime_config(regime_config_path)
    regime_detector = _build_regime_detector(regime_cfg)

    # 从 constitution 读取 max_slots
    constitution_yaml = getattr(args, "constitution", None)
    if not constitution_yaml:
        constitution_yaml = regime_cfg.get("constitution_ref")
    const = _load_constitution_constraints(constitution_yaml)
    max_slots = getattr(args, "max_slots", None) or const["slot_count"]

    print(f"   📄 Regime config: {regime_config_path}")
    print(f"   📄 Constitution: {constitution_yaml or 'defaults'}")
    print(
        f"   🔒 max_slots={max_slots} (from {'args' if getattr(args, 'max_slots', None) else 'constitution'})"
    )

    # ── 1. 解析 --pcm 参数 ──
    arch_specs: Dict[str, str] = {}  # {archetype: logs_path}
    for spec in args.pcm:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            print(f"❌ Invalid --pcm format: {spec}. Expected archetype:path")
            return 1
        arch_name, logs_path = parts
        arch_specs[arch_name] = logs_path

    arch_names = list(arch_specs.keys())
    print(f"\n📋 Archetypes: {arch_names}")
    print(f"   Priority: {' > '.join(priority)}")
    print(f"   决策依据: 按语义要求的条件严格性划分（越严格越优先）")

    # ── 2. 加载各 archetype 配置 + 处理信号 ──
    arch_exec_configs: Dict[str, Dict] = {}
    arch_processed: Dict[str, pd.DataFrame] = {}  # direction + evidence
    base_df = None

    for arch_name, logs_path in arch_specs.items():
        path = Path(logs_path)
        if not path.exists():
            print(f"❌ {arch_name}: file not found: {path}")
            return 1

        # 加载配置
        try:
            exec_cfg = load_execution_config(arch_name, strategies_root)
            arch_exec_configs[arch_name] = exec_cfg
        except FileNotFoundError as e:
            print(f"❌ {arch_name}: {e}")
            return 1

        # 加载数据
        df = pd.read_parquet(path)
        if "_symbol" in df.columns and "symbol" not in df.columns:
            df["symbol"] = df["_symbol"]
        print(f"\n📂 {arch_name}: {len(df)} rows from {path}")

        # 检测方向列 —— 严格使用 direction.yaml
        dir_cfg = load_direction_config(arch_name, strategies_root)
        if not dir_cfg:
            print(
                f"❌ {arch_name}: direction.yaml 不存在 ({strategies_root}/{arch_name}/archetypes/direction.yaml)"
            )
            return 1
        applied = apply_direction_rules(df, arch_name, dir_cfg)
        if applied is None:
            available_cols = [
                c for c in df.columns if "dir" in c.lower() or "breakout" in c.lower()
            ]
            print(
                f"❌ {arch_name}: direction.yaml 规则无一命中。"
                f"可用方向相关列: {available_cols}"
            )
            return 1

        # Gate 过滤
        if "gate_decision" in df.columns:
            veto_mask = df["gate_decision"] != "allow"
            df.loc[veto_mask, "entry_direction"] = 0.0
            n_allowed = int((~veto_mask).sum())
            print(f"   🚪 Gate: {n_allowed} allow / {len(df)} total")
        elif "gate_ok" in df.columns:
            veto_mask = df["gate_ok"] != True  # noqa: E712
            df.loc[veto_mask, "entry_direction"] = 0.0
            n_allowed = int((~veto_mask).sum())
            print(f"   🚪 Gate: {n_allowed} allow / {len(df)} total")

        # Entry Filter
        if not args.no_entry_filter:
            ef_cfg = load_entry_filters_config(arch_name, strategies_root)
            if ef_cfg:
                compute_derived_entry_features(df)
                n_entries = apply_entry_filters_or(df, ef_cfg)
            else:
                print(f"   ℹ️  {arch_name}: entry_filters.yaml not found, skipping")
        else:
            print(f"   ℹ️  Entry filter disabled")

        # Evidence 计算
        evidence_scores = _compute_evidence_for_archetype(
            df,
            arch_name,
            strategies_root,
            args.quantile_train_start,
            args.quantile_train_end,
        )
        df["evidence_score"] = evidence_scores.values

        n_entries = int((df["entry_direction"] != 0).sum())
        print(f"   📊 Active entries: {n_entries}")

        arch_processed[arch_name] = df

    # ── 3. 构建合并 DataFrame ──
    # 用所有 archetype 的 OHLC 行取 union 作为统一时间线
    ohlc_cols = ["symbol", "high", "low", "close", "atr"]
    sample_df = next(iter(arch_processed.values()))
    if "timestamp" in sample_df.columns:
        ohlc_cols.insert(1, "timestamp")
    missing_ohlc = [c for c in ohlc_cols if c not in sample_df.columns]
    if missing_ohlc:
        print(f"❌ Data missing OHLC columns: {missing_ohlc}")
        return 1

    # 合并所有 archetype 的 OHLC 行，取 union (去重)
    ohlc_frames = []
    for arch_name, df in arch_processed.items():
        ohlc_frames.append(df[ohlc_cols].copy())
    merged = pd.concat(ohlc_frames, ignore_index=True)
    merge_key = ["symbol", "timestamp"] if "timestamp" in merged.columns else ["symbol"]
    merged = merged.drop_duplicates(subset=merge_key, keep="first")
    merged = merged.sort_values(merge_key).reset_index(drop=True)
    print(
        f"   📐 Unified timeline: {len(merged)} rows (union of {len(arch_processed)} archetypes)"
    )

    # 初始化结果列
    merged["entry_direction"] = 0.0
    merged["evidence_score"] = 0.5
    merged["_pcm_archetype"] = ""

    # 为每个 archetype 添加 direction + evidence 列
    for arch_name, df in arch_processed.items():
        df_sorted = df.sort_values(merge_key).reset_index(drop=True)

        if len(df_sorted) == len(merged):
            # 同长度 → 直接对齐
            merged[f"_{arch_name}_dir"] = df_sorted["entry_direction"].values
            merged[f"_{arch_name}_ev"] = df_sorted["evidence_score"].values
        elif "timestamp" in merged.columns and "timestamp" in df_sorted.columns:
            # 不同长度 → 按 (symbol, timestamp) left merge
            tmp = df_sorted[[*merge_key, "entry_direction", "evidence_score"]].rename(
                columns={
                    "entry_direction": f"_{arch_name}_dir",
                    "evidence_score": f"_{arch_name}_ev",
                }
            )
            merged = merged.merge(tmp, on=merge_key, how="left")
            merged[f"_{arch_name}_dir"] = merged[f"_{arch_name}_dir"].fillna(0.0)
            merged[f"_{arch_name}_ev"] = merged[f"_{arch_name}_ev"].fillna(0.5)
        else:
            print(
                f"⚠️  {arch_name}: 行数不一致 ({len(df_sorted)} vs {len(merged)}) 且无 timestamp，跳过"
            )
            merged[f"_{arch_name}_dir"] = 0.0
            merged[f"_{arch_name}_ev"] = 0.5

    # ── 4. PCM 仲裁 (Regime-Aware) ──
    print(f"\n🏗️  PCM arbitration (Regime-Aware)...")
    dir_cols = [f"_{a}_dir" for a in arch_names]
    has_any_signal = (merged[dir_cols].abs() > 0).any(axis=1)
    signal_indices = merged.index[has_any_signal]
    n_conflicts = 0
    arch_win_counts: Dict[str, int] = {a: 0 for a in arch_names}

    # Regime 统计
    regime_bar_counts: Dict[str, int] = {}
    regime_entry_counts: Dict[str, int] = {}
    regime_scales_applied: List[float] = []
    merged["_regime"] = ""
    merged["_position_scale"] = 1.0

    # 对每个信号 bar 进行仲裁
    for idx in signal_indices:
        # Regime 检测: 使用当前 bar 的特征
        row_features = {}
        for col in ["atr_percentile", "oi_zscore", "funding_rate_abs_zscore"]:
            if col in merged.columns:
                val = merged.at[idx, col]
                if pd.notna(val):
                    row_features[col] = float(val)
        current_regime = regime_detector.detect(row_features)
        current_priority = regime_detector.current_priority

        merged.at[idx, "_regime"] = current_regime
        regime_bar_counts[current_regime] = regime_bar_counts.get(current_regime, 0) + 1

        active: List[Tuple[str, float, float]] = []  # (arch, direction, evidence)
        for arch_name in arch_names:
            d = float(merged.at[idx, f"_{arch_name}_dir"])
            if d != 0.0:
                ev = float(merged.at[idx, f"_{arch_name}_ev"])
                active.append((arch_name, d, ev))

        if not active:
            continue

        if len(active) == 1:
            winner_arch, winner_dir, winner_ev = active[0]
        else:
            # 多个 archetype 冲突 → Regime 动态优先级 + Evidence
            n_conflicts += 1

            def _sort_key(x):
                arch, d, ev = x
                rank = _pcm_get_priority_rank(arch, current_priority)
                return (rank, -(ev if ev is not None else 0.5))

            winner_arch, winner_dir, winner_ev = min(active, key=_sort_key)

        # Regime 仓位缩放
        scale = regime_detector.get_archetype_scale(winner_arch)
        regime_scales_applied.append(scale)

        merged.at[idx, "entry_direction"] = winner_dir
        merged.at[idx, "evidence_score"] = winner_ev
        merged.at[idx, "_pcm_archetype"] = winner_arch
        merged.at[idx, "_position_scale"] = scale
        arch_win_counts[winner_arch] = arch_win_counts.get(winner_arch, 0) + 1
        regime_entry_counts[current_regime] = (
            regime_entry_counts.get(current_regime, 0) + 1
        )

    n_total_entries = int((merged["entry_direction"] != 0).sum())
    print(f"   Total entries: {n_total_entries}")
    print(f"   Conflicts resolved: {n_conflicts}")
    conflict_rate = n_conflicts / max(1, n_total_entries)
    print(f"   Conflict rate: {conflict_rate:.2%}")
    for arch_name in arch_names:
        cnt = arch_win_counts.get(arch_name, 0)
        print(f"   {arch_name}: {cnt} entries")

    # Regime 统计
    print(f"\n   🌍 Regime Distribution:")
    print(f"   {'Regime':<18} {'Bars':>7} {'Entries':>9} {'Avg Scale':>10}")
    print(f"   {'-' * 48}")
    for regime_name in ["NORMAL", "HIGH_VOL", "HIGH_LEVERAGE"]:
        bars = regime_bar_counts.get(regime_name, 0)
        entries = regime_entry_counts.get(regime_name, 0)
        regime_mask = merged["_regime"] == regime_name
        scales = merged.loc[
            regime_mask & (merged["_position_scale"] > 0), "_position_scale"
        ]
        avg_scale = scales.mean() if len(scales) > 0 else 1.0
        print(f"   {regime_name:<18} {bars:>7} {entries:>9} {avg_scale:>9.2f}")
    print(f"   Regime switches: {regime_detector.switch_count}")
    if regime_scales_applied:
        avg_drag = 1.0 - sum(regime_scales_applied) / len(regime_scales_applied)
        print(f"   Scale drag: {avg_drag:.2%} (avg reduction from regime scaling)")

    if n_total_entries == 0:
        print("❌ No entry signals after PCM arbitration")
        return 1

    # ── 5. Per-entry 执行参数（来自 winning archetype 的 execution.yaml）──
    # 初始化 per-entry 参数列
    first_exec = list(arch_exec_configs.values())[0]
    first_sl = first_exec.get("stop_loss", {})
    first_trail = first_sl.get("trailing", {})
    first_holding = first_exec.get("holding", {})

    merged["_tier_initial_r"] = float(first_sl.get("initial_r", 2.0))
    merged["_tier_activation_r"] = float(first_trail.get("activation_r", 1.0))
    merged["_tier_trail_r"] = float(first_trail.get("trail_r", 1.5))
    merged["_tier_timeout"] = int(first_holding.get("time_stop_bars", 50) or 50)
    merged["_tier_size"] = 1.0
    merged["_tier_name"] = "default"

    # 按 winning archetype 覆盖执行参数
    for arch_name in arch_names:
        if arch_name not in arch_exec_configs:
            continue
        ec = arch_exec_configs[arch_name]
        sl = ec.get("stop_loss", {})
        trail = sl.get("trailing", {})
        holding = ec.get("holding", {})

        mask = merged["_pcm_archetype"] == arch_name
        if mask.sum() == 0:
            continue

        merged.loc[mask, "_tier_initial_r"] = float(sl.get("initial_r", 2.0))
        merged.loc[mask, "_tier_activation_r"] = float(trail.get("activation_r", 1.0))
        merged.loc[mask, "_tier_trail_r"] = float(trail.get("trail_r", 1.5))
        merged.loc[mask, "_tier_timeout"] = int(holding.get("time_stop_bars", 50) or 50)
        merged.loc[mask, "_tier_name"] = arch_name

    # 应用 Regime 仓位缩放到 _tier_size
    entry_with_scale = merged["_position_scale"] < 1.0
    if entry_with_scale.any():
        merged.loc[entry_with_scale, "_tier_size"] *= merged.loc[
            entry_with_scale, "_position_scale"
        ]
        print(
            f"   📉 Regime scale applied to {entry_with_scale.sum()} entries "
            f"(avg size={merged.loc[entry_with_scale, '_tier_size'].mean():.2f})"
        )

    # ── 6. Bar-by-bar 执行模拟 ──
    breakeven_lock_r = args.breakeven if args.breakeven is not None else 0.0
    print(f"\n📈 Simulating bar-by-bar with per-archetype execution params...")
    exec_returns, trade_details = simulate_rr_execution(
        merged,
        first_exec,  # 全局 fallback config
        atr_col="atr",
        use_tier_params=True,
        breakeven_lock_r=breakeven_lock_r,
    )

    valid_returns = exec_returns.dropna()
    if len(valid_returns) == 0:
        print("❌ No valid returns computed")
        return 1

    # ── 7. 结果报告 ──
    span_years = _estimate_span_years(merged)
    sym_col_pcm = "symbol" if "symbol" in merged.columns else "_symbol"
    n_symbols_pcm = (
        merged[sym_col_pcm].nunique() if sym_col_pcm in merged.columns else 1
    )
    exec_sharpe = compute_sharpe(valid_returns, annualize=False)
    exec_sharpe_ann = compute_sharpe(
        valid_returns, annualize=True, span_years=span_years, n_symbols=n_symbols_pcm
    )
    trades_per_year = (
        len(valid_returns) / max(1, n_symbols_pcm) / span_years if span_years > 0 else 0
    )

    print("\n" + "=" * 80)
    print("📊 PCM MULTI-ARCHETYPE BACKTEST RESULTS")
    print("=" * 80)
    print(
        f"\n   Trades: {len(valid_returns)}  "
        f"({trades_per_year:.0f}/year, span={span_years:.2f}yr)"
    )
    print(f"   Mean R: {valid_returns.mean():.4f}")
    print(f"   Std R:  {valid_returns.std():.4f}")
    print(f"   Win Rate: {(valid_returns > 0).mean():.2%}")
    print(f"\n   Sharpe (per-trade): {exec_sharpe:.4f}")
    print(
        f"   Sharpe (annualized): {exec_sharpe_ann:.2f}  "
        f"= {exec_sharpe:.4f} \u00d7 \u221a{trades_per_year:.0f}"
    )
    daily_sharpe_pcm = compute_daily_sharpe(merged, exec_returns)
    print(f"   Sharpe (daily, ×√252): {daily_sharpe_pcm:.2f}  ← 业界可比指标")

    # ── 风险仓位 Equity Curve (每策略风险 cap) ──
    pcm_sl_r = float(first_exec.get("stop_loss", {}).get("initial_r", 1.0))
    risk_per_slot = float(const.get("risk_per_slot", 0.01))
    strategy_limits = const.get("per_strategy_limits") or {}
    # Build per-trade risk series based on archetype
    pcm_arch_col = (
        merged["_pcm_archetype"]
        if "_pcm_archetype" in merged.columns
        else pd.Series("", index=merged.index)
    )
    risk_series = exec_returns.copy()
    for idx_val in risk_series.index:
        arch = (
            str(pcm_arch_col.iloc[idx_val] if idx_val < len(pcm_arch_col) else "")
            .strip()
            .lower()
        )
        strat = strategy_limits.get(arch) or {}
        strat_risk = strat.get("max_risk_per_trade")
        if strat_risk is not None:
            risk_series.iloc[idx_val] = min(risk_per_slot, float(strat_risk))
        else:
            risk_series.iloc[idx_val] = risk_per_slot
    risk_eq_pcm = compute_risk_equity_curve(
        exec_returns,
        initial_cash=1000.0,
        risk_per_slot=risk_per_slot,
        stop_loss_r=pcm_sl_r,
        risk_per_trade_series=risk_series,
    )
    print(f"\n   💰 Risk-Based Equity ($1000, per-strategy risk, SL={pcm_sl_r}R):")
    print(
        f"      Final: ${risk_eq_pcm['final_equity']:.0f}  ({risk_eq_pcm['total_return_pct']:+.1f}%)"
    )
    print(f"      Max DD: {risk_eq_pcm['max_dd']:.1%}")

    # Per-symbol breakdown
    sym_col = "symbol" if "symbol" in merged.columns else "_symbol"
    print(f"\n   📋 Per-Symbol Breakdown:")
    print(f"   {'Symbol':<12} {'Trades':>7} {'Mean R':>8} {'Sharpe':>8} {'Win%':>7}")
    print(f"   {'-' * 46}")
    for sym in sorted(merged[sym_col].unique()):
        mask = merged[sym_col] == sym
        rr = exec_returns.loc[mask].dropna()
        if len(rr) > 1:
            sh = rr.mean() / rr.std() if rr.std() > 1e-8 else 0
            print(
                f"   {sym:<12} {len(rr):>7} {rr.mean():>8.4f} "
                f"{sh:>8.4f} {(rr > 0).mean() * 100:>6.1f}%"
            )

    # Per-archetype breakdown
    merged["_exec_rr"] = exec_returns.values
    print(f"\n   🏷️  Per-Archetype Breakdown:")
    print(
        f"   {'Archetype':<14} {'Trades':>7} {'Mean R':>8} "
        f"{'Sharpe':>8} {'Win%':>7} {'Conflicts':>10}"
    )
    print(f"   {'-' * 58}")
    entry_mask = merged["entry_direction"] != 0
    for arch_name in arch_names:
        arch_mask = entry_mask & (merged["_pcm_archetype"] == arch_name)
        rr = merged.loc[arch_mask, "_exec_rr"].dropna()
        if len(rr) > 0:
            sh = rr.mean() / rr.std() if len(rr) > 1 and rr.std() > 1e-8 else 0
            # 该 archetype 参与冲突但被选中的次数 vs 被其他 archetype 挤掉的次数
            print(
                f"   {arch_name:<14} {len(rr):>7} {rr.mean():>8.4f} "
                f"{sh:>8.4f} {(rr > 0).mean() * 100:>6.1f}%"
            )
    merged.drop(columns=["_exec_rr"], inplace=True)

    # 反事实分析: 被 PCM 丢弃的信号表现
    print(f"\n   🔍 Counterfactual (被丢弃信号的后续 R):")
    for arch_name in arch_names:
        # 该 archetype 有信号但被其他 archetype 抢走的 bar
        has_signal = merged[f"_{arch_name}_dir"] != 0
        was_rejected = merged["_pcm_archetype"] != arch_name
        rejected_mask = has_signal & was_rejected & entry_mask
        n_rejected = int(rejected_mask.sum())
        if n_rejected > 0:
            # 模拟这些被丢弃信号的 R
            tmp = merged.copy()
            tmp["entry_direction"] = 0.0
            tmp.loc[rejected_mask, "entry_direction"] = tmp.loc[
                rejected_mask, f"_{arch_name}_dir"
            ]
            ec = arch_exec_configs.get(arch_name, first_exec)
            cf_returns, _ = simulate_rr_execution(
                tmp,
                ec,
                atr_col="atr",
                silent=True,
            )
            cf_valid = cf_returns.dropna()
            if len(cf_valid) > 0:
                print(
                    f"   {arch_name} rejected: {len(cf_valid)} trades, "
                    f"mean_R={cf_valid.mean():.4f}, win={( cf_valid > 0).mean():.2%}"
                )

    # ── 8. 可选输出 ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _generate_per_symbol_html(
            df=merged,
            exec_returns=exec_returns,
            exec_config=first_exec,
            strategy="pcm_" + "_".join(arch_names),
            span_years=span_years,
        )
        html_path = output_path.with_suffix(".html")
        Path(html_path).write_text(html, encoding="utf-8")
        print(f"\n   📊 Per-Symbol HTML Report: {html_path}")

    # 生成交易地图 (K线 + 入场/出场标记)
    if trade_details:
        # PCM 模式: 输出到第一个 archetype 的目录
        first_parquet = list(arch_specs.values())[0]
        map_dir = Path(first_parquet).parent
        map_path = map_dir / f"trading_map_pcm_{'_'.join(arch_names)}.html"
        # 每个 archetype 用自己 meta.yaml 的 timeframe
        arch_tfs: Dict[str, str] = {}
        for a in arch_names:
            tf = load_meta_timeframe(a, strategies_root)
            if tf:
                arch_tfs[a] = tf
        # 尝试加载全量连续 OHLC (消除 prefilter 导致的 K 线跳空)
        _pcm_map_ohlc = None
        _fs_root = getattr(args, "features_store_root", "feature_store")
        _ts_col = "timestamp" if "timestamp" in merged.columns else None
        _start = pd.Timestamp(merged[_ts_col].min()) if _ts_col else None
        _end = pd.Timestamp(merged[_ts_col].max()) if _ts_col else None
        _syms = merged["symbol"].unique().tolist() if "symbol" in merged.columns else []
        if _syms and arch_tfs:
            # 用第一个 archetype 的 timeframe 和 layer 加载
            _first_arch = list(arch_tfs.keys())[0]
            _first_tf = arch_tfs[_first_arch]
            from src.feature_store.layer_naming import detect_layer_for_strategy

            _fs_layer = detect_layer_for_strategy(_first_arch, _fs_root)
            if _fs_layer:
                _pcm_map_ohlc = _load_full_ohlc_for_map(
                    features_store_root=_fs_root,
                    features_store_layer=_fs_layer,
                    symbols=_syms,
                    timeframe=_first_tf,
                    start=_start,
                    end=_end,
                )
        map_html = _generate_trading_map_html(
            merged,
            trade_details,
            title=f"PCM {'_'.join(arch_names)} Trading Map",
            arch_timeframes=arch_tfs if arch_tfs else None,
            timeframe=getattr(args, "timeframe", None),
            full_ohlc=_pcm_map_ohlc,
        )
        map_path.write_text(map_html, encoding="utf-8")
        print(f"   🗺️  Trading Map: {map_path}")

    if args.export_signals:
        export_path = Path(args.export_signals)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        has_dir = merged["entry_direction"] != 0
        export_data = {
            "symbol": merged.loc[has_dir, sym_col].values,
            "archetype": merged.loc[has_dir, "_pcm_archetype"].values,
            "direction": merged.loc[has_dir, "entry_direction"].values,
            "evidence_score": merged.loc[has_dir, "evidence_score"].values,
        }
        if "timestamp" in merged.columns:
            export_data["timestamp"] = merged.loc[has_dir, "timestamp"].values
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_path, index=False)
        print(f"\n   📤 Signals exported: {len(export_df)} rows → {export_path}")

    print("\n" + "=" * 80)
    return 0


# ================================================================
# Grid Search (imported by optimize_execution_grid.py)
# ================================================================


def _parse_optimization_grid(
    optimization_cfg: Dict[str, Any],
) -> Tuple[List[str], List[List[float]]]:
    """
    解析 execution.yaml 的 optimization 段，生成参数网格

    Returns:
        (param_names, param_value_lists)
    """
    params_cfg = optimization_cfg.get("params", {})
    param_names: List[str] = []
    param_values: List[List[float]] = []

    for param_path, cfg in params_cfg.items():
        rng = cfg.get("range", [0, 1])
        step = cfg.get("step", 0.5)
        values = []
        val = rng[0]
        while val <= rng[1] + 1e-9:
            values.append(round(val, 4))
            val += step
        param_names.append(param_path)
        param_values.append(values)

    return param_names, param_values


def _set_nested(d: dict, dotted_key: str, value: float) -> None:
    """设置嵌套字典值, e.g. 'stop_loss.initial_r' -> d['stop_loss']['initial_r']"""
    parts = dotted_key.split(".")
    target = d
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def run_grid_search(
    df: pd.DataFrame,
    exec_config: Dict[str, Any],
    param_names: List[str],
    param_values: List[List[float]],
    atr_col: str = "atr",
    span_years: float = 1.0,
    n_symbols: int = 1,
) -> List[Dict[str, Any]]:
    """
    执行全量网格搜索

    Args:
        n_symbols: symbol 数量，年化时用 per-symbol 交易频率
                   trades_per_year = trades / n_symbols / span_years

    Returns:
        每组参数的回测结果列表
    """
    import io
    import contextlib

    all_combos = list(itertools.product(*param_values))
    total = len(all_combos)
    results: List[Dict[str, Any]] = []

    for idx, combo in enumerate(all_combos, 1):
        # 构造修改后的配置
        modified = copy.deepcopy(exec_config)
        for name, val in zip(param_names, combo):
            _set_nested(modified, name, val)

        # 静默运行模拟（抑制 print 输出）
        with contextlib.redirect_stdout(io.StringIO()):
            returns, _ = simulate_rr_execution(df, modified, atr_col, silent=True)
        valid = returns.dropna()

        if len(valid) >= 2 and valid.std() > 1e-8:
            sharpe = float(valid.mean() / valid.std())
            # per-symbol 交易频率年化，避免多 symbol 合计膨胀
            n_sym = max(1, n_symbols)
            trades_per_year = len(valid) / n_sym / span_years if span_years > 0 else 0
            sharpe_ann = (
                sharpe * np.sqrt(trades_per_year) if trades_per_year > 0 else 0.0
            )
        else:
            sharpe = 0.0
            sharpe_ann = 0.0

        result = {
            "combo_idx": idx,
            "sharpe": sharpe,
            "sharpe_ann": sharpe_ann,
            "mean_r": float(valid.mean()) if len(valid) > 0 else 0.0,
            "std_r": float(valid.std()) if len(valid) > 1 else 0.0,
            "win_rate": float((valid > 0).mean()) if len(valid) > 0 else 0.0,
            "trades": len(valid),
        }
        for name, val in zip(param_names, combo):
            result[name] = val

        results.append(result)

        if idx % 10 == 0 or idx == total:
            print(f"   [{idx}/{total}] ...", end="\r")

    print()  # newline after progress
    return results


def _identify_plateau(
    results: List[Dict[str, Any]],
    top_frac: float = 0.25,
    cv_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    识别参数平坦高原区域

    Returns:
        plateau 分析结果
    """
    sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    top_n = max(3, int(len(sorted_results) * top_frac))
    top = sorted_results[:top_n]

    sharpe_values = [r["sharpe"] for r in top]
    mean_sharpe = np.mean(sharpe_values)
    std_sharpe = np.std(sharpe_values)
    cv = std_sharpe / mean_sharpe if mean_sharpe > 1e-8 else float("inf")

    is_plateau = cv < cv_threshold

    return {
        "is_plateau": is_plateau,
        "top_n": top_n,
        "mean_sharpe": float(mean_sharpe),
        "std_sharpe": float(std_sharpe),
        "cv": float(cv),
        "best": sorted_results[0],
        "top_results": top,
        "all_sorted": sorted_results,
    }


def _generate_grid_search_html(
    results: List[Dict[str, Any]],
    param_names: List[str],
    param_values: List[List[float]],
    plateau: Dict[str, Any],
    exec_config: Dict[str, Any],
    strategy: str,
    n_trades_total: int,
) -> str:
    """
    生成美化的 Grid Search HTML 报告
    """
    best = plateau["best"]
    all_sorted = plateau["all_sorted"]

    # 当前配置的 Sharpe（第一行匹配当前参数的结果）
    current_params = {}
    for name in param_names:
        parts = name.split(".")
        val = exec_config
        for p in parts:
            val = val.get(p, {})
        current_params[name] = float(val) if not isinstance(val, dict) else 0.0

    current_result = None
    for r in results:
        match = all(
            abs(r.get(n, -999) - current_params.get(n, -1)) < 1e-6 for n in param_names
        )
        if match:
            current_result = r
            break

    current_sharpe = current_result["sharpe"] if current_result else 0.0
    delta = best["sharpe"] - current_sharpe

    # 短参数名（用于显示）
    short_names = [n.split(".")[-1] for n in param_names]

    # ---- Heatmap 生成（base64 内嵌 matplotlib） ----
    heatmap_html = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        if len(param_names) >= 2:
            # 对每一对参数生成 heatmap（第三参数取最佳值的切片）
            pairs = []
            for i in range(len(param_names)):
                for j in range(i + 1, len(param_names)):
                    pairs.append((i, j))

            for pi, pj in pairs:
                # 找第三参数的最佳值
                other_indices = [
                    k for k in range(len(param_names)) if k != pi and k != pj
                ]
                best_other_vals = {
                    param_names[k]: best[param_names[k]] for k in other_indices
                }

                # 筛选匹配第三参数的结果
                filtered = []
                for r in results:
                    match = True
                    for k in other_indices:
                        if (
                            abs(
                                r.get(param_names[k], -999)
                                - best_other_vals[param_names[k]]
                            )
                            > 1e-6
                        ):
                            match = False
                            break
                    if match:
                        filtered.append(r)

                if not filtered:
                    continue

                # 构建 2D 数组
                x_vals = sorted(set(r[param_names[pi]] for r in filtered))
                y_vals = sorted(set(r[param_names[pj]] for r in filtered))
                grid = np.full((len(y_vals), len(x_vals)), np.nan)

                for r in filtered:
                    xi = x_vals.index(r[param_names[pi]])
                    yi = y_vals.index(r[param_names[pj]])
                    grid[yi, xi] = r["sharpe"]

                # 绘制
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                im = ax.imshow(
                    grid,
                    cmap="RdYlGn",
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                )
                ax.set_xticks(range(len(x_vals)))
                ax.set_xticklabels([f"{v:.1f}" for v in x_vals])
                ax.set_yticks(range(len(y_vals)))
                ax.set_yticklabels([f"{v:.1f}" for v in y_vals])
                ax.set_xlabel(short_names[pi])
                ax.set_ylabel(short_names[pj])

                # 固定参数信息
                fixed_info = ", ".join(
                    f"{short_names[k]}={best_other_vals[param_names[k]]:.1f}"
                    for k in other_indices
                )
                title = f"Sharpe Heatmap: {short_names[pi]} × {short_names[pj]}"
                if fixed_info:
                    title += f"  (fixed: {fixed_info})"
                ax.set_title(title, fontsize=12)

                # 标注数值
                for yi_idx in range(len(y_vals)):
                    for xi_idx in range(len(x_vals)):
                        val = grid[yi_idx, xi_idx]
                        if not np.isnan(val):
                            color = "white" if val < np.nanmean(grid) else "black"
                            ax.text(
                                xi_idx,
                                yi_idx,
                                f"{val:.3f}",
                                ha="center",
                                va="center",
                                fontsize=9,
                                color=color,
                                fontweight="bold",
                            )

                # 标记当前配置位置
                if (
                    param_names[pi] in current_params
                    and param_names[pj] in current_params
                ):
                    cx = current_params[param_names[pi]]
                    cy = current_params[param_names[pj]]
                    if cx in x_vals and cy in y_vals:
                        ax.plot(
                            x_vals.index(cx),
                            y_vals.index(cy),
                            "s",
                            color="blue",
                            markersize=18,
                            markerfacecolor="none",
                            markeredgewidth=2.5,
                            label="current",
                        )

                # 标记最佳位置
                bx = best[param_names[pi]]
                by = best[param_names[pj]]
                if bx in x_vals and by in y_vals:
                    ax.plot(
                        x_vals.index(bx),
                        y_vals.index(by),
                        "*",
                        color="gold",
                        markersize=20,
                        markeredgecolor="black",
                        markeredgewidth=1,
                        label="best",
                    )

                ax.legend(loc="upper right", fontsize=9)
                plt.colorbar(im, ax=ax, label="Sharpe", shrink=0.8)
                plt.tight_layout()

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                heatmap_html += f'<div style="text-align:center;margin:20px 0;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);"/></div>\n'

    except ImportError:
        heatmap_html = (
            '<p style="color:#e74c3c;">matplotlib not available — heatmap skipped</p>'
        )

    # ---- 结果表格 ----
    rows_html = ""
    for rank, r in enumerate(all_sorted[:30], 1):  # Top 30
        params_str = " / ".join(f"{r.get(n, 0):.1f}" for n in param_names)
        is_best = rank == 1
        is_current = current_result and all(
            abs(r.get(n, -999) - current_params.get(n, -1)) < 1e-6 for n in param_names
        )
        row_class = (
            ' style="background:#d4edda;font-weight:bold;"'
            if is_best
            else (' style="background:#cce5ff;"' if is_current else "")
        )
        badge = " ⭐" if is_best else (" 📌" if is_current else "")

        rows_html += f"""<tr{row_class}>
            <td>{rank}{badge}</td>
            <td><code>{params_str}</code></td>
            <td><strong>{r['sharpe']:.4f}</strong></td>
            <td>{r['sharpe_ann']:.1f}</td>
            <td>{r['mean_r']:.4f}</td>
            <td>{r['std_r']:.4f}</td>
            <td>{r['win_rate']:.1%}</td>
            <td>{r['trades']}</td>
        </tr>\n"""

    # ---- Plateau 分析 ----
    plateau_status = "✅ 平坦高原" if plateau["is_plateau"] else "⚠️ 未形成高原"
    plateau_color = "#27ae60" if plateau["is_plateau"] else "#f39c12"

    # ---- 完整 HTML ----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Execution Grid Search Report - {strategy.upper()}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #e67e22; margin-bottom: 30px; font-size: 28px; }}
        h2 {{ color: #34495e; border-bottom: 3px solid #e67e22; padding-bottom: 10px; margin: 30px 0 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; }}
        .kpi-item {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .kpi-item.primary {{ background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%); }}
        .kpi-item.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .kpi-item.warning {{ background: linear-gradient(135deg, #ee5a24 0%, #f39c12 100%); }}
        .kpi-item.info {{ background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }}
        .kpi-value {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
        .kpi-label {{ font-size: 13px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; font-size: 13px; }}
        th {{ background: #f8f9fa; color: #2c3e50; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #f8f9fa; }}
        .plateau-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; }}
        .secondary {{ color: #7f8c8d; font-size: 14px; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .timestamp {{ text-align: center; color: #95a5a6; font-size: 12px; margin-top: 30px; }}
        .hint {{ background: #fff3cd; border-left: 4px solid #f39c12; padding: 15px; margin-top: 20px; border-radius: 4px; }}
        .param-header {{ font-size: 12px; color: #95a5a6; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📈 Execution Layer Grid Search 报告</h1>
    <p style="text-align:center;color:#7f8c8d;margin-bottom:30px;">Strategy: {strategy.upper()} | 参数网格搜索 + Sharpe 目标</p>

    <h2>🎯 核心 KPI</h2>
    <div class="card">
        <div class="kpi-grid">
            <div class="kpi-item success">
                <div class="kpi-label">最佳 Sharpe</div>
                <div class="kpi-value">{best['sharpe']:.4f}</div>
                <div class="kpi-label">{' / '.join(f'{short_names[i]}={best[param_names[i]]:.1f}' for i in range(len(param_names)))}</div>
            </div>
            <div class="kpi-item info">
                <div class="kpi-label">当前 Sharpe</div>
                <div class="kpi-value">{current_sharpe:.4f}</div>
                <div class="kpi-label">{' / '.join(f'{short_names[i]}={current_params.get(param_names[i], 0):.1f}' for i in range(len(param_names)))}</div>
            </div>
            <div class="kpi-item {'success' if delta > 0 else 'warning'}">
                <div class="kpi-label">Delta</div>
                <div class="kpi-value">{'+' if delta > 0 else ''}{delta:.4f}</div>
                <div class="kpi-label">{'可提升' if delta > 0.01 else '当前已接近最优'}</div>
            </div>
            <div class="kpi-item primary">
                <div class="kpi-label">Annualized Sharpe</div>
                <div class="kpi-value">{best['sharpe_ann']:.1f}</div>
                <div class="kpi-label">最佳参数组</div>
            </div>
        </div>
        <div class="secondary">
            <strong>搜索空间:</strong> {len(results)} 组参数 | 
            <strong>样本:</strong> {n_trades_total} trades |
            <strong>Win Rate (best):</strong> {best['win_rate']:.1%} |
            <strong>Mean R (best):</strong> {best['mean_r']:.4f}
        </div>
    </div>

    <h2>📊 平坦高原分析</h2>
    <div class="card">
        <div class="kpi-grid">
            <div class="kpi-item" style="background:{'linear-gradient(135deg, #27ae60, #2ecc71)' if plateau['is_plateau'] else 'linear-gradient(135deg, #f39c12, #e67e22)'}">
                <div class="kpi-label">Plateau 状态</div>
                <div class="kpi-value" style="font-size:22px;">{plateau_status}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Top {plateau['top_n']} 平均 Sharpe</div>
                <div class="kpi-value">{plateau['mean_sharpe']:.4f}</div>
                <div class="kpi-label">± {plateau['std_sharpe']:.4f}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">CV (变异系数)</div>
                <div class="kpi-value">{plateau['cv']:.3f}</div>
                <div class="kpi-label">{'< 0.15 ✅' if plateau['cv'] < 0.15 else '>= 0.15 ⚠️'}</div>
            </div>
        </div>
        <div class="hint">
            <strong>💡 平坦高原解读:</strong>
            多种参数组合达到相似 Sharpe = 参数鲁棒，不会因小调整而大幅波动。
            CV (变异系数) < 0.15 表明高原区域稳定。建议选择高原中点作为最终参数。
        </div>
    </div>

    <h2>🗺️ Sharpe Heatmap</h2>
    <div class="card">
        {heatmap_html if heatmap_html else '<p>无 heatmap 数据</p>'}
        <div class="secondary">
            <strong>⭐</strong> = 最佳参数 | <strong>🔷</strong> = 当前配置 (蓝色方框)
        </div>
    </div>

    <h2>📋 完整排名 (Top 30)</h2>
    <div class="card" style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>参数 <span class="param-header">({' / '.join(short_names)})</span></th>
                    <th>Sharpe</th>
                    <th>Ann. Sharpe</th>
                    <th>Mean R</th>
                    <th>Std R</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody>
{rows_html}
            </tbody>
        </table>
    </div>

    <h2>🔧 推荐配置</h2>
    <div class="card">
        <pre style="background:#2c3e50;color:#ecf0f1;padding:20px;border-radius:8px;font-size:14px;overflow-x:auto;">
# execution.yaml 推荐更新 (基于 Grid Search)
stop_loss:
  type: trailing
  initial_r: {best[param_names[0]] if len(param_names) > 0 else 2.0}
  trailing:
    activation_r: {best[param_names[1]] if len(param_names) > 1 else 1.0}
    trail_r: {best[param_names[2]] if len(param_names) > 2 else 1.5}
        </pre>
        <div class="hint">
            <strong>⚠️ 注意:</strong>
            {'当前参数已在平坦高原内，无需调整。' if plateau['is_plateau'] and abs(delta) < 0.02 else
             '建议更新至最佳参数，但先在 holdout 期验证。' if delta > 0.02 else
             '当前参数已接近最优。'}
        </div>
    </div>

    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>"""
    return html


def _generate_per_symbol_html(
    df: pd.DataFrame,
    exec_returns: pd.Series,
    exec_config: Dict[str, Any],
    strategy: str,
    span_years: float = 1.0,
) -> str:
    """
    生成 per-symbol 交易地图 HTML 报告

    包含:
    - Overall KPI
    - Per-symbol KPI 表
    - Per-symbol 月度收益 heatmap
    - 交易散点图（RR 随时间变化）
    """
    sym_col = "symbol" if "symbol" in df.columns else "_symbol"
    df = df.copy()
    df["exec_rr"] = exec_returns.values

    # 检测可用的 RR 列
    rr_cols = []
    for col in ["ret_mean", "ret_trend"]:
        if col in df.columns:
            rr_cols.append(col)

    # Overall stats
    valid = df["exec_rr"].dropna()
    overall_sharpe = (
        valid.mean() / valid.std() if len(valid) > 1 and valid.std() > 1e-8 else 0.0
    )
    overall_sharpe_ann = (
        overall_sharpe * np.sqrt(len(valid) / span_years) if span_years > 0 else 0.0
    )

    # Per-symbol stats
    sym_stats = []
    for sym in sorted(df[sym_col].unique()):
        mask = df[sym_col] == sym
        rr = df.loc[mask, "exec_rr"].dropna()
        if len(rr) < 2:
            continue
        sh = rr.mean() / rr.std() if rr.std() > 1e-8 else 0.0
        row = {
            "symbol": sym,
            "trades": len(rr),
            "mean_r": rr.mean(),
            "std_r": rr.std(),
            "sharpe": sh,
            "sharpe_ann": sh * np.sqrt(len(rr) / span_years) if span_years > 0 else 0.0,
            "win_rate": (rr > 0).mean(),
            "pf": (
                rr[rr > 0].sum() / abs(rr[rr < 0].sum())
                if (rr < 0).any()
                else float("inf")
            ),
        }
        # 每个 RR 列的 Sharpe
        for rc in rr_cols:
            rv = df.loc[mask, rc].dropna()
            if len(rv) > 1 and rv.std() > 1e-8:
                row[f"sharpe_{rc}"] = rv.mean() / rv.std()
            else:
                row[f"sharpe_{rc}"] = 0.0
        sym_stats.append(row)

    # Per-symbol 表格行
    sym_rows_html = ""
    for s in sorted(sym_stats, key=lambda x: x["sharpe"], reverse=True):
        color = "#27ae60" if s["sharpe"] > 0 else "#e74c3c"
        extra_cols = ""
        for rc in rr_cols:
            v = s.get(f"sharpe_{rc}", 0)
            c2 = "#27ae60" if v > 0 else "#e74c3c"
            extra_cols += f'<td style="color:{c2};font-weight:bold;">{v:.4f}</td>'
        pf_str = f"{s['pf']:.2f}" if s["pf"] < 100 else "∞"
        sym_rows_html += f"""<tr>
            <td><strong>{s['symbol']}</strong></td>
            <td>{s['trades']}</td>
            <td style="color:{color};font-weight:bold;">{s['sharpe']:.4f}</td>
            <td>{s['sharpe_ann']:.1f}</td>
            <td>{s['mean_r']:.4f}</td>
            <td>{s['std_r']:.4f}</td>
            <td>{s['win_rate']:.1%}</td>
            <td>{pf_str}</td>
            {extra_cols}
        </tr>\n"""

    # 额外 RR 列 header
    extra_headers = ""
    for rc in rr_cols:
        extra_headers += f"<th>Sharpe ({rc})</th>"

    # ---- 图表生成 ----
    chart_images = []  # list of base64 img html strings
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        def _fig_to_img(fig):
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            return f'<div style="text-align:center;margin:20px 0;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);"/></div>'

        # 1. Per-symbol Sharpe 柱状图
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        syms = [s["symbol"] for s in sym_stats]
        sharpes = [s["sharpe"] for s in sym_stats]
        colors = ["#27ae60" if s > 0 else "#e74c3c" for s in sharpes]
        ax.bar(syms, sharpes, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="#7f8c8d", linewidth=1, linestyle="--")
        ax.axhline(
            overall_sharpe,
            color="#3498db",
            linewidth=1.5,
            linestyle="-.",
            label=f"Overall: {overall_sharpe:.4f}",
        )
        ax.set_ylabel("Sharpe (raw)")
        ax.set_title("Per-Symbol Sharpe Ratio")
        ax.legend()
        plt.tight_layout()
        chart_images.append(_fig_to_img(fig))

        # 2. 交易散点图（每个 symbol 一个子图）
        n_syms = len(syms)
        fig, axes = plt.subplots(n_syms, 1, figsize=(14, 3 * n_syms), sharex=True)
        if n_syms == 1:
            axes = [axes]

        for ax, sym in zip(axes, sorted(df[sym_col].unique())):
            mask = df[sym_col] == sym
            sub = df.loc[mask].dropna(subset=["exec_rr"])
            if "timestamp" in sub.columns:
                x = pd.to_datetime(sub["timestamp"])
            else:
                x = range(len(sub))
            y = sub["exec_rr"]
            colors_scatter = ["#27ae60" if v > 0 else "#e74c3c" for v in y]
            ax.scatter(x, y, c=colors_scatter, s=12, alpha=0.6, edgecolors="none")
            ax.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
            sh = y.mean() / y.std() if y.std() > 1e-8 else 0
            ax.set_ylabel("R")
            ax.set_title(
                f"{sym}  (Sharpe={sh:.4f}, n={len(y)}, Win={( y>0).mean():.0%})",
                fontsize=11,
                loc="left",
            )
            # 添加月度均线
            if "timestamp" in sub.columns:
                monthly = sub.set_index("timestamp")["exec_rr"].resample("M").mean()
                if len(monthly) > 1:
                    ax.plot(
                        monthly.index,
                        monthly.values,
                        color="#3498db",
                        linewidth=2,
                        label="Monthly mean",
                        alpha=0.8,
                    )
                    ax.legend(fontsize=8)

        plt.xlabel("Time")
        plt.tight_layout()
        chart_images.append(_fig_to_img(fig))

        # 3. 月度 heatmap（symbol × month）
        if "timestamp" in df.columns:
            df_ts = df.dropna(subset=["exec_rr"]).copy()
            df_ts["month"] = (
                pd.to_datetime(df_ts["timestamp"]).dt.to_period("M").astype(str)
            )
            pivot = df_ts.pivot_table(
                values="exec_rr", index=sym_col, columns="month", aggfunc="mean"
            )
            if not pivot.empty:
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(
                        max(10, len(pivot.columns) * 0.8),
                        max(3, len(pivot) * 0.6),
                    ),
                )
                im = ax.imshow(
                    pivot.values, cmap="RdYlGn", aspect="auto", interpolation="nearest"
                )
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=10)
                ax.set_title("Monthly Mean R by Symbol")
                for yi in range(len(pivot.index)):
                    for xi in range(len(pivot.columns)):
                        v = pivot.values[yi, xi]
                        if not np.isnan(v):
                            ax.text(
                                xi,
                                yi,
                                f"{v:.2f}",
                                ha="center",
                                va="center",
                                fontsize=7,
                                fontweight="bold",
                                color="white" if abs(v) > 2 else "black",
                            )
                plt.colorbar(im, ax=ax, label="Mean R", shrink=0.8)
                plt.tight_layout()
                chart_images.append(_fig_to_img(fig))

    except ImportError:
        pass  # matplotlib not available

    # ---- 止损配置 ----
    sl = exec_config.get("stop_loss", {})
    config_info = f"""<pre style="background:#2c3e50;color:#ecf0f1;padding:20px;border-radius:8px;font-size:14px;">
stop_loss:
  type: {sl.get('type', 'fixed')}
  initial_r: {sl.get('initial_r', 2.0)}
  trailing:
    activation_r: {sl.get('trailing', {}).get('activation_r', 1.0)}
    trail_r: {sl.get('trailing', {}).get('trail_r', 1.5)}
</pre>"""

    # ---- 完整 HTML ----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Execution Backtest Report - {strategy.upper()}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #e67e22; margin-bottom: 10px; font-size: 28px; }}
        h2 {{ color: #34495e; border-bottom: 3px solid #e67e22; padding-bottom: 10px; margin: 30px 0 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; }}
        .kpi-item {{ padding: 20px; border-radius: 10px; text-align: center; color: white; }}
        .kpi-item.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .kpi-item.warning {{ background: linear-gradient(135deg, #ee5a24 0%, #f39c12 100%); }}
        .kpi-item.info {{ background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }}
        .kpi-item.primary {{ background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%); }}
        .kpi-value {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
        .kpi-label {{ font-size: 13px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; font-size: 13px; }}
        th {{ background: #f8f9fa; color: #2c3e50; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #f8f9fa; }}
        .timestamp {{ text-align: center; color: #95a5a6; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Execution Layer Backtest 报告</h1>
    <p style="text-align:center;color:#7f8c8d;margin-bottom:30px;">Strategy: {strategy.upper()} | Per-Symbol 交易地图</p>

    <h2>🎯 Overall KPI</h2>
    <div class="card">
        <div class="kpi-grid">
            <div class="kpi-item {'success' if overall_sharpe > 0.1 else 'warning'}">
                <div class="kpi-label">Sharpe (per-trade)</div>
                <div class="kpi-value">{overall_sharpe:.4f}</div>
            </div>
            <div class="kpi-item {'success' if overall_sharpe_ann > 2 else 'info'}">
                <div class="kpi-label">Sharpe (annualized)</div>
                <div class="kpi-value">{overall_sharpe_ann:.1f}</div>
            </div>
            <div class="kpi-item primary">
                <div class="kpi-label">Trades</div>
                <div class="kpi-value">{len(valid):,}</div>
            </div>
            <div class="kpi-item {'success' if valid.mean() > 0 else 'warning'}">
                <div class="kpi-label">Mean R</div>
                <div class="kpi-value">{valid.mean():.4f}</div>
            </div>
            <div class="kpi-item info">
                <div class="kpi-label">Win Rate</div>
                <div class="kpi-value">{(valid > 0).mean():.1%}</div>
            </div>
        </div>
    </div>

    <h2>📋 Per-Symbol Breakdown</h2>
    <div class="card" style="overflow-x:auto;">
        <table>
            <thead><tr>
                <th>Symbol</th><th>Trades</th><th>Sharpe</th><th>Ann. Sharpe</th>
                <th>Mean R</th><th>Std R</th><th>Win Rate</th><th>PF</th>
                {extra_headers}
            </tr></thead>
            <tbody>
{sym_rows_html}
            </tbody>
        </table>
    </div>

    <h2>📊 Per-Symbol Sharpe 柱状图</h2>
    <div class="card">
        {chart_images[0] if len(chart_images) > 0 else '<p>No charts</p>'}
    </div>

    <h2>🗺️ 交易散点图 (RR over time)</h2>
    <div class="card">
        {chart_images[1] if len(chart_images) > 1 else '<p>No scatter data</p>'}
    </div>

    <h2>📅 月度收益 Heatmap</h2>
    <div class="card">
        {chart_images[2] if len(chart_images) > 2 else '<p>No monthly data</p>'}
    </div>

    <h2>⚙️ 执行配置</h2>
    <div class="card">
        {config_info}
    </div>

    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>"""
    return html


def main() -> int:
    p = argparse.ArgumentParser(
        description="Execution Layer Backtest with archetypes/execution.yaml"
    )
    p.add_argument(
        "--logs",
        required=False,
        default=None,
        help="Input logs file (predictions.parquet or logs_gated.parquet)",
    )
    p.add_argument(
        "--strategy", required=False, default=None, help="Strategy name (e.g., bpc)"
    )
    p.add_argument("--strategies-root", default="config/strategies")
    p.add_argument(
        "--pcm",
        nargs="+",
        default=None,
        help="Multi-archetype PCM mode: archetype:path pairs. "
        "Example: --pcm bpc:results/bpc/predictions.parquet me:results/me/predictions.parquet",
    )
    p.add_argument(
        "--max-slots",
        type=int,
        default=2,
        help="Max concurrent slots for PCM mode (default: 2)",
    )
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument(
        "--features-store-layer",
        default=None,
        help="FeatureStore layer (auto-detect from strategy if omitted)",
    )
    p.add_argument("--timeframe", default="240T")
    p.add_argument(
        "--tiers",
        action="store_true",
        help="Enable tier-based execution: per-entry params from evidence_score + tiers config",
    )
    p.add_argument(
        "--noise-penalty",
        action="store_true",
        help="Apply noise penalty adjustments (requires FeatureStore math features)",
    )
    p.add_argument(
        "--no-entry-filter",
        action="store_true",
        help="Disable automatic entry filter (skip entry_filters.yaml)",
    )
    p.add_argument(
        "--quantile-train-start",
        type=str,
        default=None,
        help="Evidence quantile 校准数据开始日期 (YYYY-MM-DD)。"
        "与 --quantile-train-end 配合，范围须 >= 6 个月，否则报错退出。",
    )
    p.add_argument(
        "--quantile-train-end",
        type=str,
        default=None,
        help="Evidence quantile 校准数据截止日期 (YYYY-MM-DD)。"
        "用 start~end 范围的数据计算分位数阈值，避免 look-ahead。"
        "--tiers 模式下必须与 --quantile-train-start 一起指定。",
    )
    p.add_argument(
        "--export-signals",
        type=str,
        default=None,
        help="导出逐 bar 信号决策 CSV，用于与 simulate_bpc_e2e.py 对比验证信号对齐。",
    )
    p.add_argument(
        "--breakeven",
        nargs="?",
        const=1.0,
        type=float,
        default=None,
        help="Enable breakeven lock. MFE >= R triggers SL move to entry price. "
        "Default trigger: 1.0R (from holding.yaml). Example: --breakeven or --breakeven 1.5",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path for results (HTML report).",
    )
    args = p.parse_args()

    # ── PCM mode: multi-archetype ──
    if args.pcm:
        return _run_pcm_mode(args)

    # ── Single-archetype mode: validate required args ──
    if not args.logs:
        p.error("--logs is required (or use --pcm for multi-archetype mode)")
    if not args.strategy:
        p.error("--strategy is required (or use --pcm for multi-archetype mode)")

    # Auto-detect feature store layer if not specified (may not be needed if logs have OHLC)
    if not args.features_store_layer:
        from src.feature_store.layer_naming import detect_layer_for_strategy

        detected = detect_layer_for_strategy(
            strategy=args.strategy,
            features_store_root=args.features_store_root,
        )
        if detected:
            args.features_store_layer = detected
            print(
                f"\u2139\ufe0f Auto-detected feature store layer for {args.strategy}: {detected}"
            )
        # If detection fails, we'll handle it later if FeatureStore is actually needed

    print("=" * 80)
    print("🎯 Execution Layer Backtest")
    print("=" * 80)

    # 加载 execution.yaml 配置
    try:
        exec_config = load_execution_config(args.strategy, args.strategies_root)
        print(f"\n📋 Loaded execution.yaml for '{args.strategy}':")
        stop_loss = exec_config.get("stop_loss", {})
        print(f"   Stop Loss Type: {stop_loss.get('type', 'fixed')}")
        print(f"   Initial R: {stop_loss.get('initial_r', 2.0)}")
        if stop_loss.get("type") == "trailing":
            trailing = stop_loss.get("trailing", {})
            print(f"   Trailing Activation: {trailing.get('activation_r', 1.0)}R")
            print(f"   Trail Distance: {trailing.get('trail_r', 1.5)}R")
    except Exception as e:
        print(f"❌ Failed to load execution.yaml: {e}")
        return 1

    # 读取 logs 文件
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"\n📂 Loaded logs: {len(df)} rows")

    # 处理列名兼容
    if "_symbol" in df.columns and "symbol" not in df.columns:
        df["symbol"] = df["_symbol"]

    # 创建 entry_direction 列：标记入场信号
    # 默认：每个有方向的 bar 都是入场信号
    if "entry_direction" in df.columns:
        print(f"   📍 Using existing entry_direction column")
    elif "bpc_breakout_direction" in df.columns:
        df["entry_direction"] = df["bpc_breakout_direction"].astype(float).copy()
    else:
        # 尝试 direction.yaml 规则
        dir_cfg = load_direction_config(args.strategy, args.strategies_root)
        if dir_cfg:
            applied = apply_direction_rules(df, args.strategy, dir_cfg)
            if applied:
                print(f"   📍 Direction: {applied} (from direction.yaml)")
            elif "entry_direction" not in df.columns:
                print("❌ direction.yaml 规则无一命中，且无 entry_direction 列")
                return 1
        else:
            df["entry_direction"] = 0.0

    # 保存原始方向（gate/entry_filter 前），用于 --export-signals
    df["_orig_direction"] = df["entry_direction"].copy()

    # Gate 过滤（自动检测）：不删除行（保持 OHLC 连续性），而是将非 allow 行的方向设为 0
    if "gate_decision" in df.columns:
        veto_mask = df["gate_decision"] != "allow"
        n_allowed = int((~veto_mask).sum())
        df.loc[veto_mask, "entry_direction"] = 0.0
        print(
            f"   🚪 Gate filter (auto): {n_allowed} allow entries / {len(df)} total bars"
        )
    elif "gate_ok" in df.columns:
        veto_mask = df["gate_ok"] != True
        n_allowed = int((~veto_mask).sum())
        df.loc[veto_mask, "entry_direction"] = 0.0
        print(
            f"   🚪 Gate filter (auto): {n_allowed} allow entries / {len(df)} total bars"
        )

    n_entries = int((df["entry_direction"] != 0).sum())
    if n_entries == 0:
        print("❌ No entry signals")
        return 1
    print(
        f"   Entry signals: {n_entries} / {len(df)} bars ({n_entries/len(df)*100:.1f}%)"
    )

    # ---- 构建连续 OHLC 模拟数据 ----
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
    if not symbols:
        print("❌ No symbols found in logs")
        return 1

    print(f"\n📊 Symbols: {', '.join(symbols)}")

    has_ohlc = all(c in df.columns for c in ["high", "low", "close", "atr"])

    if has_ohlc:
        # 日志已包含连续 OHLC → 直接使用（常见路径: predictions.parquet）
        merged = df.copy()
        # 按 (symbol, timestamp) 排序，保证每个 symbol 内部按时间连续
        sort_cols = ["symbol"]
        if "timestamp" in merged.columns:
            sort_cols.append("timestamp")
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
        print(
            f"\n🔄 Using OHLC from logs: {len(merged)} continuous bars, {n_entries} entries"
        )
    else:
        # 从 FeatureStore 获取连续 OHLC
        if not args.features_store_layer:
            print("❌ Logs don't have OHLC and no FeatureStore layer specified.")
            print("   Use --features-store-layer explicitly.")
            return 1

        print(f"\n📂 Loading continuous OHLC from FeatureStore...")
        store = FeatureStore(args.features_store_root)
        parts = []
        for sym in symbols:
            spec = FeatureStoreSpec(
                layer=args.features_store_layer, symbol=sym, timeframe=args.timeframe
            )
            try:
                df_sym = store.read_range(
                    spec,
                    start=pd.Timestamp("1970-01-01"),
                    end=pd.Timestamp("2100-01-01"),
                )
                if not df_sym.empty:
                    df_sym = df_sym.copy()
                    if "symbol" not in df_sym.columns:
                        df_sym["symbol"] = sym
                    if df_sym.index.name == "timestamp":
                        df_sym = df_sym.reset_index()
                    elif isinstance(df_sym.index, pd.DatetimeIndex):
                        df_sym["timestamp"] = df_sym.index
                        df_sym = df_sym.reset_index(drop=True)
                    parts.append(df_sym)
            except Exception as e:
                print(f"   ⚠️  Failed to read {sym}: {e}")

        if not parts:
            print("❌ No FeatureStore data loaded")
            return 1

        merged = pd.concat(parts, axis=0, ignore_index=True)
        merged["symbol"] = merged["symbol"].astype(str)

        # FeatureStore 的 bpc_breakout_direction 作为入场方向
        if "bpc_breakout_direction" in merged.columns:
            merged["entry_direction"] = merged["bpc_breakout_direction"].astype(float)
            merged["_orig_direction"] = merged["entry_direction"].copy()
        else:
            print("❌ No bpc_breakout_direction in FeatureStore")
            return 1

        # 按 symbol + timestamp 排序
        if "timestamp" in merged.columns:
            merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        n_entries = int((merged["entry_direction"] != 0).sum())
        print(
            f"   Loaded FeatureStore: {len(merged)} continuous bars, {n_entries} entries"
        )

    # ================================================================
    # Entry Filter: 入场时机过滤 (自动读取 entry_filters.yaml, OR 组合)
    # ================================================================
    if not args.no_entry_filter:
        entry_filters_cfg = load_entry_filters_config(
            args.strategy, args.strategies_root
        )
        if entry_filters_cfg:
            compute_derived_entry_features(merged)
            n_entries = apply_entry_filters_or(merged, entry_filters_cfg)
            if n_entries == 0:
                print("❌ No entry signals after entry filter")
                return 1
        else:
            print("   ℹ️  entry_filters.yaml not found, skipping entry filter")
    else:
        print("   ℹ️  Entry filter disabled (--no-entry-filter)")

    # ================================================================
    # Tiers 模式: 计算 evidence_score + 分配 per-entry 参数
    # ================================================================
    use_tier_params = False
    if args.tiers:
        tiers_cfg = exec_config.get("tiers", {})
        if not tiers_cfg.get("enabled"):
            print("⚠️  tiers.enabled=false in execution.yaml, running without tiers")
        elif not tiers_cfg.get("levels"):
            print("⚠️  No tiers.levels in execution.yaml")
        else:
            print("\n🏷️  Tier Mode: computing evidence scores...")
            evidence_cfg = load_evidence_config(args.strategy, args.strategies_root)

            # ── 预计算 quantiles（避免 look-ahead，与实盘对齐）──
            if not args.quantile_train_start or not args.quantile_train_end:
                print(
                    "❌ --tiers 模式需要 --quantile-train-start DATE --quantile-train-end DATE"
                )
                print(
                    "   示例: --quantile-train-start 2025-02-01 --quantile-train-end 2025-08-01"
                )
                return 1

            train_start = pd.Timestamp(args.quantile_train_start)
            train_end = pd.Timestamp(args.quantile_train_end)
            if (train_end - train_start).days < 180:
                print(
                    f"❌ 校准数据时间范围不足 6 个月: "
                    f"{train_start.date()} ~ {train_end.date()} "
                    f"({(train_end - train_start).days} 天)"
                )
                return 1

            # 确保 merged 有 timestamp 列
            ts_col = None
            if "timestamp" in merged.columns:
                ts_col = "timestamp"
            elif isinstance(merged.index, pd.DatetimeIndex):
                merged["_ts_tmp"] = merged.index
                ts_col = "_ts_tmp"

            if ts_col is None:
                print("❌ 数据中没有 timestamp 列，无法按日期切分校准数据")
                return 1

            merged[ts_col] = pd.to_datetime(merged[ts_col], utc=True)
            calib_mask = (merged[ts_col] >= train_start.tz_localize("UTC")) & (
                merged[ts_col] < train_end.tz_localize("UTC")
            )
            calib_df = merged[calib_mask]

            if len(calib_df) < 50:
                print(
                    f"❌ 校准数据不足: {len(calib_df)} 行 "
                    f"(需要 {train_start.date()} ~ {train_end.date()} 至少 6 个月数据)"
                )
                return 1

            print(
                f"   📐 Quantile calibration: {len(calib_df)} rows "
                f"({train_start.date()} ~ {train_end.date()}, no look-ahead)"
            )
            precomputed_quantiles = compute_evidence_quantiles(calib_df, evidence_cfg)

            # 清理临时列
            if "_ts_tmp" in merged.columns:
                merged.drop(columns=["_ts_tmp"], inplace=True)

            evidence_scores = compute_evidence_scores(
                merged,
                evidence_cfg,
                precomputed_quantiles=precomputed_quantiles,
            )
            merged["evidence_score"] = evidence_scores.values

            assign_tiers(merged, tiers_cfg, evidence_scores, exec_config)
            use_tier_params = True

    # ================================================================
    # Noise Penalty: 调整 per-entry 参数
    # ================================================================
    if args.noise_penalty:
        print("\n🔇 Noise Penalty: loading math features...")
        noise_features = [
            "wpt_price_fluctuation",
            "spectrum_price_entropy",
            "hilbert_price_env",
            "hurst_price_rolling",
        ]
        missing_nf = [f for f in noise_features if f not in merged.columns]

        if missing_nf:
            # 尝试从 FeatureStore 加载缺失的噪声特征
            print(f"   Loading {len(missing_nf)} noise features from FeatureStore...")
            try:
                store = FeatureStore(args.features_store_root)
                # 使用 auto-detected layer
                fs_layer = args.features_store_layer
                if not fs_layer:
                    from src.feature_store.layer_naming import detect_layer_for_strategy

                    fs_layer = detect_layer_for_strategy(
                        args.strategy, args.features_store_root
                    )

                if fs_layer:
                    sym_col_n = "symbol" if "symbol" in merged.columns else "_symbol"
                    for sym in merged[sym_col_n].unique():
                        spec = FeatureStoreSpec(
                            layer=fs_layer, symbol=sym, timeframe=args.timeframe
                        )
                        fs_df = store.read_range(
                            spec,
                            start=pd.Timestamp("1970-01-01"),
                            end=pd.Timestamp("2100-01-01"),
                        )
                        if fs_df.empty:
                            continue
                        # 仅合并缺失的列
                        avail = [f for f in missing_nf if f in fs_df.columns]
                        if not avail:
                            continue
                        # 用 timestamp 匹配（如果有）
                        if (
                            "timestamp" in merged.columns
                            and hasattr(fs_df.index, "name")
                            and fs_df.index.name == "timestamp"
                        ):
                            sym_mask = merged[sym_col_n] == sym
                            ts_merged = merged.loc[sym_mask, "timestamp"]
                            fs_sub = fs_df.loc[fs_df.index.isin(ts_merged), avail]
                            for f in avail:
                                if f in fs_sub.columns:
                                    ts_to_val = fs_sub[f].to_dict()
                                    merged.loc[sym_mask, f] = merged.loc[
                                        sym_mask, "timestamp"
                                    ].map(ts_to_val)
                        else:
                            # 无 timestamp，按顺序对齐（截取相同长度）
                            sym_mask = merged[sym_col_n] == sym
                            n_sym = int(sym_mask.sum())
                            for f in avail:
                                if f in fs_df.columns:
                                    vals = (
                                        fs_df[f].values[-n_sym:]
                                        if len(fs_df) >= n_sym
                                        else np.pad(
                                            fs_df[f].values,
                                            (n_sym - len(fs_df), 0),
                                            constant_values=np.nan,
                                        )
                                    )
                                    merged.loc[sym_mask, f] = vals
                else:
                    print("   ⚠️  Cannot detect FeatureStore layer for noise features")
            except Exception as e:
                print(f"   ⚠️  Failed to load noise features: {e}")

        # 检查是否全部可用
        still_missing = [f for f in noise_features if f not in merged.columns]
        if still_missing:
            print(f"   ⚠️  Noise penalty disabled: missing {still_missing}")
        else:
            from src.time_series_model.execution.noise_penalty import (
                ExecutionNoisePenalty,
                NoisePenaltyConfig,
            )

            np_calculator = ExecutionNoisePenalty(NoisePenaltyConfig())
            # 按 symbol 分别计算（避免跨 symbol 污染）
            merged["noise_penalty"] = 0.0
            sym_col_n = "symbol" if "symbol" in merged.columns else "_symbol"
            for sym in merged[sym_col_n].unique():
                sym_mask = merged[sym_col_n] == sym
                sym_df = merged.loc[sym_mask, noise_features].copy()
                try:
                    np_series = np_calculator.compute(sym_df)
                    merged.loc[sym_mask, "noise_penalty"] = np_series.values
                except Exception as e:
                    print(f"   ⚠️  Noise penalty failed for {sym}: {e}")

            np_vals = merged["noise_penalty"]
            print(
                f"   🔇 Noise penalty: mean={np_vals.mean():.3f}, "
                f"std={np_vals.std():.3f}, max={np_vals.max():.3f}"
            )

            # 应用噪声惩罚调整 per-entry 参数
            if use_tier_params:
                # Tier 模式: 调整 tier 参数
                np_arr = merged["noise_penalty"].values
                merged["_tier_initial_r"] = merged["_tier_initial_r"] * (
                    1 + 0.5 * np_arr
                )
                merged["_tier_trail_r"] = merged["_tier_trail_r"] * (1 + 0.3 * np_arr)
                merged["_tier_size"] = (merged["_tier_size"] * (1 - 0.7 * np_arr)).clip(
                    lower=0.1
                )
                print("   ✅ Noise penalty applied to tier params")
            else:
                # 非 Tier 模式: 先创建 per-entry 参数列，再调整
                sl = exec_config.get("stop_loss", {})
                trail = sl.get("trailing", {})
                holding = exec_config.get("holding", {})
                merged["_tier_initial_r"] = float(sl.get("initial_r", 2.0))
                merged["_tier_activation_r"] = float(trail.get("activation_r", 1.0))
                merged["_tier_trail_r"] = float(trail.get("trail_r", 1.5))
                merged["_tier_timeout"] = int(holding.get("time_stop_bars", 50) or 50)
                merged["_tier_size"] = 1.0
                merged["_tier_name"] = "default"

                np_arr = merged["noise_penalty"].values
                merged["_tier_initial_r"] = merged["_tier_initial_r"] * (
                    1 + 0.5 * np_arr
                )
                merged["_tier_trail_r"] = merged["_tier_trail_r"] * (1 + 0.3 * np_arr)
                merged["_tier_size"] = (merged["_tier_size"] * (1 - 0.7 * np_arr)).clip(
                    lower=0.1
                )
                use_tier_params = True  # 启用 per-entry 参数
                print("   ✅ Noise penalty applied to global params (per-entry)")

    # ================================================================
    # Breakeven Lock: 保本锁定
    # ================================================================
    breakeven_lock_r = 0.0
    if args.breakeven is not None:
        breakeven_lock_r = args.breakeven
        print(f"\n🔒 Breakeven lock enabled: trigger at {breakeven_lock_r}R")

    # ================================================================
    # 单次回测模式
    # ================================================================

    # 使用 execution.yaml 配置模拟 RR
    print("\n📈 Simulating with execution.yaml config...")
    exec_returns, trade_details = simulate_rr_execution(
        merged,
        exec_config,
        atr_col="atr",
        use_tier_params=use_tier_params,
        breakeven_lock_r=breakeven_lock_r,
    )

    valid_returns = exec_returns.dropna()
    if len(valid_returns) == 0:
        print("❌ No valid returns computed")
        return 1

    # 计算 Sharpe
    span_years = _estimate_span_years(merged)
    sym_col = "symbol" if "symbol" in merged.columns else "_symbol"
    n_symbols = merged[sym_col].nunique() if sym_col in merged.columns else 1
    exec_sharpe = compute_sharpe(valid_returns, annualize=False)
    exec_sharpe_ann = compute_sharpe(
        valid_returns, annualize=True, span_years=span_years, n_symbols=n_symbols
    )
    trades_per_year = (
        len(valid_returns) / max(1, n_symbols) / span_years if span_years > 0 else 0
    )

    print("\n" + "=" * 80)
    print("📊 EXECUTION LAYER BACKTEST RESULTS")
    print("=" * 80)
    print(
        f"\n   Trades: {len(valid_returns)}  ({trades_per_year:.0f}/year, span={span_years:.2f}yr)"
    )
    print(f"   Mean R: {valid_returns.mean():.4f}")
    print(f"   Std R:  {valid_returns.std():.4f}")
    print(f"   Win Rate: {(valid_returns > 0).mean():.2%}")
    print(f"\n   Sharpe (per-trade): {exec_sharpe:.4f}")
    print(
        f"   Sharpe (annualized): {exec_sharpe_ann:.2f}  = {exec_sharpe:.4f} \u00d7 \u221a{trades_per_year:.0f}"
    )
    daily_sharpe = compute_daily_sharpe(merged, exec_returns)
    print(f"   Sharpe (daily, ×√252): {daily_sharpe:.2f}  ← 业界可比指标")

    # ── 风险仓位 Equity Curve (每策略风险 cap) ──
    sl_r = float(exec_config.get("stop_loss", {}).get("initial_r", 1.0))
    # Resolve per-strategy risk from constitution
    from src.time_series_model.portfolio.live_pcm import (
        _load_constitution_constraints as _load_const_fn,
    )

    _const_yaml = getattr(args, "constitution", None)
    _const_single = _load_const_fn(_const_yaml)
    _risk_slot = float(_const_single.get("risk_per_slot", 0.01))
    _strategy_limits = _const_single.get("per_strategy_limits") or {}
    _strat = _strategy_limits.get(str(args.strategy).lower()) or {}
    _strat_risk = _strat.get("max_risk_per_trade")
    effective_risk = (
        min(_risk_slot, float(_strat_risk)) if _strat_risk is not None else _risk_slot
    )
    risk_eq = compute_risk_equity_curve(
        exec_returns,
        initial_cash=1000.0,
        risk_per_slot=effective_risk,
        stop_loss_r=sl_r,
    )
    print(
        f"\n   💰 Risk-Based Equity ($1000, {effective_risk:.1%}/trade [{args.strategy}], SL={sl_r}R):"
    )
    print(
        f"      Final: ${risk_eq['final_equity']:.0f}  ({risk_eq['total_return_pct']:+.1f}%)"
    )
    print(f"      Max DD: {risk_eq['max_dd']:.1%}")

    # Per-symbol breakdown
    sym_col = "symbol" if "symbol" in merged.columns else "_symbol"
    print(f"\n   📋 Per-Symbol Breakdown:")
    print(f"   {'Symbol':<12} {'Trades':>7} {'Mean R':>8} {'Sharpe':>8} {'Win%':>7}")
    print(f"   {'-'*46}")
    for sym in sorted(merged[sym_col].unique()):
        mask = merged[sym_col] == sym
        rr = exec_returns.loc[mask].dropna()
        if len(rr) > 1:
            sh = rr.mean() / rr.std() if rr.std() > 1e-8 else 0
            print(
                f"   {sym:<12} {len(rr):>7} {rr.mean():>8.4f} {sh:>8.4f} {(rr>0).mean()*100:>6.1f}%"
            )

    # Per-tier 统计（如果启用了 tiers）
    if use_tier_params and "_tier_name" in merged.columns:
        merged["exec_rr_temp"] = exec_returns.values
        print(f"\n   🏷️  Per-Tier Breakdown:")
        print(
            f"   {'Tier':<14} {'Trades':>7} {'Mean R':>8} {'Sharpe':>8} {'Win%':>7} {'Size':>6}"
        )
        print(f"   {'-'*54}")
        entry_mask = merged["entry_direction"] != 0
        for tier_name in merged.loc[entry_mask, "_tier_name"].unique():
            tier_mask = entry_mask & (merged["_tier_name"] == tier_name)
            rr = merged.loc[tier_mask, "exec_rr_temp"].dropna()
            if len(rr) > 1:
                sh = rr.mean() / rr.std() if rr.std() > 1e-8 else 0
                avg_size = merged.loc[tier_mask, "_tier_size"].mean()
                print(
                    f"   {tier_name:<14} {len(rr):>7} {rr.mean():>8.4f} {sh:>8.4f} {(rr>0).mean()*100:>6.1f}% {avg_size:>5.2f}x"
                )
        merged.drop(columns=["exec_rr_temp"], inplace=True)

    # 对比其他 RR 列（如果存在）— forward_rr 已移除
    for rr_col in ["ret_mean", "ret_trend"]:
        if rr_col in merged.columns:
            orig_returns = merged[rr_col].dropna()
            if len(orig_returns) > 0:
                orig_sharpe = compute_sharpe(orig_returns, annualize=False)
                orig_mean = orig_returns.mean()
                print(f"\n   📌 Reference {rr_col}:")
                print(f"      Mean: {orig_mean:.4f}")
                print(f"      Sharpe (raw): {orig_sharpe:.4f}")
                print(
                    f"      → Delta: {'+' if exec_sharpe > orig_sharpe else ''}{exec_sharpe - orig_sharpe:.4f}"
                )
                break

    # 生成 per-symbol HTML 报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = _generate_per_symbol_html(
            df=merged,
            exec_returns=exec_returns,
            exec_config=exec_config,
            strategy=args.strategy,
            span_years=span_years,
        )
        html_path = output_path.with_suffix(".html")
        Path(html_path).write_text(html, encoding="utf-8")
        print(f"\n   📊 Per-Symbol HTML Report: {html_path}")

    # 生成交易地图 (K线 + 入场/出场标记)
    if trade_details:
        logs_path = Path(args.logs)
        # 如果提供了 --output 参数，则使用该路径，否则使用默认路径
        if args.output:
            map_path = Path(args.output)
        else:
            map_path = (
                logs_path.parent / f"trading_map_{args.strategy or 'backtest'}.html"
            )
        # 从 meta.yaml 读取 timeframe
        auto_tf = (
            load_meta_timeframe(
                args.strategy, getattr(args, "strategies_root", "config/strategies")
            )
            if args.strategy
            else None
        )
        map_tf = getattr(args, "timeframe", None) or auto_tf
        # 尝试加载全量连续 OHLC (消除 prefilter 导致的 K 线跳空)
        _map_ohlc = None
        if getattr(args, "features_store_layer", None):
            _ts_col = "timestamp" if "timestamp" in merged.columns else None
            _start = pd.Timestamp(merged[_ts_col].min()) if _ts_col else None
            _end = pd.Timestamp(merged[_ts_col].max()) if _ts_col else None
            _syms = (
                merged["symbol"].unique().tolist() if "symbol" in merged.columns else []
            )
            if _syms:
                _map_ohlc = _load_full_ohlc_for_map(
                    features_store_root=args.features_store_root,
                    features_store_layer=args.features_store_layer,
                    symbols=_syms,
                    timeframe=map_tf or args.timeframe or "240T",
                    start=_start,
                    end=_end,
                )
        map_html = _generate_trading_map_html(
            merged,
            trade_details,
            title=f"{args.strategy or 'Backtest'} Trading Map",
            timeframe=map_tf,
            full_ohlc=_map_ohlc,
        )
        map_path.write_text(map_html, encoding="utf-8")
        print(f"   🗺️  Trading Map: {map_path}")

    # ── Export signals CSV（信号对齐验证）──
    if args.export_signals:
        export_path = Path(args.export_signals)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        sym_col = "symbol" if "symbol" in merged.columns else "_symbol"
        has_dir = merged["_orig_direction"] != 0
        export_data = {
            "symbol": merged.loc[has_dir, sym_col].values,
        }
        if "timestamp" in merged.columns:
            export_data["timestamp"] = merged.loc[has_dir, "timestamp"].values
        elif isinstance(merged.index, pd.DatetimeIndex):
            export_data["timestamp"] = merged.index[has_dir]
        export_data["direction"] = merged.loc[has_dir, "_orig_direction"].values
        export_data["entry_direction"] = merged.loc[has_dir, "entry_direction"].values
        if "evidence_score" in merged.columns:
            export_data["evidence_score"] = merged.loc[has_dir, "evidence_score"].values
        if "_tier_name" in merged.columns:
            export_data["tier"] = merged.loc[has_dir, "_tier_name"].values
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_path, index=False)
        print(f"\n   📤 Signals exported: {len(export_df)} rows → {export_path}")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
