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
        --strategy bpc --noise-penalty \\
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
    "me-long": "#FF9800",  # Orange
    "fer": "#AB47BC",  # Purple
    "lv": "#66BB6A",  # Green
    "reversal": "#EC407A",  # Pink
}
_DEFAULT_ARCH_COLOR = "#00d4aa"  # Teal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec
from src.time_series_model.archetype import load_strategy_archetype


# ================================================================
# 1min Bar Loading (研究数据 / 实盘数据 双路径)
# ================================================================


def _load_1min_bars(
    merged: pd.DataFrame,
    *,
    data_path: Optional[str] = None,
    live_root: str = "live/highcap",
) -> Optional[Dict[str, pd.DataFrame]]:
    """加载 1min bar 数据, 支持两种来源:

    1. --data-path (推荐用于向量回测): 从研究数据 data/parquet_data
       通过 DataHandler.load_ohlcv(timeframe="1T") 加载
    2. --live-root (用于事件回测/实盘): 从 StorageManager 加载
    """
    # 确定时间范围
    ts_col = "timestamp"
    if ts_col in merged.columns:
        ts_min = pd.Timestamp(merged[ts_col].min())
        ts_max = pd.Timestamp(merged[ts_col].max())
        start_str = (ts_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end_str = (ts_max + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        start_str = "2025-01-01"
        end_str = "2026-12-31"

    sym_col = "symbol" if "symbol" in merged.columns else "_symbol"
    symbols = merged[sym_col].unique()

    bars_1min_dict: Dict[str, pd.DataFrame] = {}

    if data_path:
        # ── 研究数据路径: DataHandler → 1min resample ──
        from src.data_tools.data_handler import DataHandler

        dh = DataHandler(data_path)
        print(f"   📂 1min 数据来源: {data_path} (研究数据)")
        for sym in symbols:
            try:
                b1m = dh.load_ohlcv(
                    symbol=sym,
                    timeframe="1T",
                    start_date=start_str,
                    end_date=end_str,
                )
                if not b1m.empty:
                    # 统一列名: 确保有 timestamp 列
                    if "timestamp" not in b1m.columns:
                        b1m = b1m.reset_index()  # index → column
                    elif b1m.index.name == "timestamp":
                        b1m = b1m.reset_index(drop=True)  # 去掉重复 index
                    b1m["timestamp"] = pd.to_datetime(b1m["timestamp"], utc=True)
                    bars_1min_dict[sym] = b1m
                    print(f"   🔬 {sym}: {len(b1m)} 1min bars loaded")
            except Exception as e:
                print(f"   ⚠️  {sym}: DataHandler 加载失败 ({e})")
    else:
        # ── 实盘数据路径: StorageManager ──
        from src.live_data_stream.feature_storage import StorageManager

        storage = StorageManager(f"{live_root}/data")
        print(f"   📂 1min 数据来源: {live_root}/data (实盘数据)")
        for sym in symbols:
            b1m = storage.bar_1min.load_range(sym, start_str, end_str)
            if not b1m.empty:
                bars_1min_dict[sym] = b1m
                print(f"   🔬 {sym}: {len(b1m)} 1min bars loaded")

    if bars_1min_dict:
        print(
            f"   ✅ 1min mode: {sum(len(v) for v in bars_1min_dict.values())} total bars"
        )
        return bars_1min_dict
    else:
        print("   ⚠️  No 1min bars found, falling back to 4H mode")
        return None


# ================================================================
# Evidence Scoring — 在 gate 放行子集上计算 evidence composite score
# 基于 archetype evidence.yaml 中的 quantile_mapping
# ================================================================


def _compute_evidence_for_archetype(
    df: pd.DataFrame,
    arch_name: str,
    archetype,
    precomputed_quantiles: Dict[str, Any],
) -> pd.Series:
    """计算每行的 evidence composite score.

    Args:
        df: 已过滤 gate 的 DataFrame
        arch_name: archetype 名称
        archetype: StrategyArchetype 实例
        precomputed_quantiles: {feature: {quantile: value}} 分位数查找表

    Returns:
        Series of evidence_score (0-1), 默认 0.5
    """
    ev_cfg = archetype.evidence
    if not ev_cfg or not ev_cfg.features:
        return pd.Series(0.5, index=df.index)

    scores = pd.Series(0.5, index=df.index)
    for idx in df.index:
        row = df.loc[idx]
        feature_values = {}
        for feat in ev_cfg.features:
            if feat.feature in row.index:
                val = row[feat.feature]
                if pd.notna(val):
                    feature_values[feat.feature] = float(val)
        if feature_values:
            composite, _ = ev_cfg.compute_composite_score(
                feature_values, precomputed_quantiles
            )
            scores.at[idx] = composite
    return scores


def _report_evidence_monotonicity(
    trade_details: List[Dict[str, Any]],
    label: str = "",
) -> None:
    """报告 evidence_score 与 trade outcome 的单调性 (Spearman)."""
    try:
        from scipy.stats import spearmanr

        ev_scores = []
        rr_vals = []
        for t in trade_details:
            es = t.get("evidence_score", 0.5)
            rr = t.get("rr")
            if rr is not None and not (isinstance(rr, float) and rr != rr):
                ev_scores.append(es)
                rr_vals.append(rr)

        if len(ev_scores) < 20:
            return

        r, p = spearmanr(ev_scores, rr_vals)
        tag = f" [{label}]" if label else ""
        print(
            f"   📊 Evidence monotonicity{tag}: "
            f"Spearman r={r:+.3f}, p={p:.4f}, n={len(ev_scores)}"
        )
    except Exception:
        pass


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
    kill_switch: Optional[Dict[str, float]] = None,
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
        kill_switch: 可选, 宪法 kill switch 模拟参数:
            - max_dd: 最大回撤限制 (0.20 = 20%)
            - daily_loss_limit: 日亏损限制 (0.04 = 4%)
            - weekly_loss_limit: 周亏损限制 (0.08 = 8%)
            - monthly_loss_limit: 月亏损限制 (0.12 = 12%)
            - cooldown_bars: kill switch 触发后冷却 bar 数 (默认 60 = ~10天4H)
            当任一限制被突破时，后续新入场被跳过直到冷却期结束

    Returns:
        dict with equity_curve, max_dd, final_equity, total_return_pct,
        以及 kill_switch 模拟统计 (如果启用)
    """
    valid = r_returns.dropna()
    if len(valid) == 0:
        base = {
            "equity_curve": [],
            "max_dd": 0.0,
            "final_equity": initial_cash,
            "total_return_pct": 0.0,
        }
        if kill_switch:
            base["kill_switch_stats"] = {
                "trades_skipped": 0,
                "trades_executed": 0,
                "triggers": [],
            }
        return base

    # Align risk_per_trade_series with valid index
    risk_arr = None
    if risk_per_trade_series is not None:
        aligned = risk_per_trade_series.reindex(valid.index)
        risk_arr = aligned.values

    # Kill switch 配置
    ks_enabled = kill_switch is not None and len(kill_switch) > 0
    ks_max_dd = float(kill_switch.get("max_dd", 1.0)) if ks_enabled else 1.0
    ks_daily = float(kill_switch.get("daily_loss_limit", 1.0)) if ks_enabled else 1.0
    ks_weekly = float(kill_switch.get("weekly_loss_limit", 1.0)) if ks_enabled else 1.0
    ks_monthly = (
        float(kill_switch.get("monthly_loss_limit", 1.0)) if ks_enabled else 1.0
    )
    ks_cooldown = int(kill_switch.get("cooldown_bars", 60)) if ks_enabled else 0

    # Kill switch 状态跟踪
    ks_halted_until = -1  # bar index until which we're halted
    ks_triggers: list = []
    ks_skipped = 0
    ks_executed = 0

    # Period loss 跟踪 (基于 index 的日期)
    has_datetime_idx = hasattr(valid.index, "date") or hasattr(
        valid.index, "to_pydatetime"
    )
    period_equity_start_daily = initial_cash
    period_equity_start_weekly = initial_cash
    period_equity_start_monthly = initial_cash
    prev_day = None
    prev_week = None
    prev_month = None

    equity = initial_cash
    curve = [equity]
    peak = equity
    max_dd = 0.0

    for i, rr in enumerate(valid.values):
        # ── Kill switch 检查 ──
        if ks_enabled and i <= ks_halted_until:
            # 在冷却期内，跳过入场
            ks_skipped += 1
            curve.append(equity)  # equity 不变
            continue

        if ks_enabled and has_datetime_idx:
            try:
                ts = valid.index[i]
                ts_date = ts.date() if hasattr(ts, "date") else None
                ts_week = ts.isocalendar()[1] if hasattr(ts, "isocalendar") else None
                ts_month = ts.month if hasattr(ts, "month") else None

                # 日/周/月 边界重置
                if ts_date and ts_date != prev_day:
                    period_equity_start_daily = equity
                    prev_day = ts_date
                if ts_week and ts_week != prev_week:
                    period_equity_start_weekly = equity
                    prev_week = ts_week
                if ts_month and ts_month != prev_month:
                    period_equity_start_monthly = equity
                    prev_month = ts_month
            except Exception:
                pass

        # ── 执行交易 ──
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
        ks_executed += 1

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        # ── Kill switch 触发检查 (交易执行后) ──
        if ks_enabled:
            trigger_reasons = []
            if dd >= ks_max_dd:
                trigger_reasons.append(f"max_dd={dd:.2%}>{ks_max_dd:.0%}")
            if has_datetime_idx and period_equity_start_daily > 0:
                daily_loss = (
                    period_equity_start_daily - equity
                ) / period_equity_start_daily
                if daily_loss >= ks_daily:
                    trigger_reasons.append(
                        f"daily_loss={daily_loss:.2%}>{ks_daily:.0%}"
                    )
            if has_datetime_idx and period_equity_start_weekly > 0:
                weekly_loss = (
                    period_equity_start_weekly - equity
                ) / period_equity_start_weekly
                if weekly_loss >= ks_weekly:
                    trigger_reasons.append(
                        f"weekly_loss={weekly_loss:.2%}>{ks_weekly:.0%}"
                    )
            if has_datetime_idx and period_equity_start_monthly > 0:
                monthly_loss = (
                    period_equity_start_monthly - equity
                ) / period_equity_start_monthly
                if monthly_loss >= ks_monthly:
                    trigger_reasons.append(
                        f"monthly_loss={monthly_loss:.2%}>{ks_monthly:.0%}"
                    )

            if trigger_reasons:
                ks_halted_until = i + ks_cooldown
                ts_str = ""
                try:
                    ts_str = str(valid.index[i])
                except Exception:
                    ts_str = f"bar_{i}"
                ks_triggers.append(
                    {
                        "bar_idx": i,
                        "timestamp": ts_str,
                        "reasons": trigger_reasons,
                        "equity": equity,
                        "dd": dd,
                        "halted_until_bar": ks_halted_until,
                    }
                )

    result = {
        "equity_curve": curve,
        "max_dd": max_dd,
        "final_equity": equity,
        "total_return_pct": (equity - initial_cash) / initial_cash * 100,
    }
    if ks_enabled:
        result["kill_switch_stats"] = {
            "trades_skipped": ks_skipped,
            "trades_executed": ks_executed,
            "triggers": ks_triggers,
            "trigger_count": len(ks_triggers),
        }
    return result


def simulate_rr_execution(
    df: pd.DataFrame,
    exec_config: Dict[str, Any],
    atr_col: str = "atr",
    direction_col: str = "entry_direction",
    silent: bool = False,
    use_tier_params: bool = False,
    breakeven_lock_r: float = 0.0,
    max_slots: int = 0,
    bars_1min_dict: Optional[Dict[str, pd.DataFrame]] = None,
    add_position_cfg: Optional[Dict[str, Any]] = None,
    per_strategy_limits: Optional[Dict[str, Any]] = None,
    evidence_min_score: float = 0.0,  # [DEPRECATED] 不再使用, 保留参数避免调用方报错
    per_strategy_ev_min: Optional[Dict[str, float]] = None,  # [DEPRECATED]
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
    # time_stop_bars: 0 或 None 表示禁用时间止损 (fat tail 模式)
    _raw_tsb = holding_cfg.get("time_stop_bars")
    if _raw_tsb is None:
        _raw_tsb = holding_cfg.get("max_holding_bars")
    g_time_stop_bars = int(_raw_tsb) if _raw_tsb and int(_raw_tsb) > 0 else 0

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
    exit_stats = {
        "sl": 0,
        "trailing_sl": 0,
        "tp": 0,
        "timeout": 0,
        "no_data": 0,
        "structural_exit_ema200": 0,
    }

    # 如果提供了 1min bar 数据，预处理为 numpy 数组以加速查找
    _1min_cache: Dict[str, Dict[str, np.ndarray]] = {}
    if bars_1min_dict:
        for _sym, _mdf in bars_1min_dict.items():
            if _mdf.empty:
                continue
            _mdf = _mdf.sort_values("timestamp")
            _ts = pd.to_datetime(_mdf["timestamp"]).values.astype("int64")
            _1min_cache[_sym] = {
                "ts": _ts,
                "high": _mdf["high"].values.astype(float),
                "low": _mdf["low"].values.astype(float),
                "close": _mdf["close"].values.astype(float),
            }
        if not silent:
            print(f"   🔬 1min bar mode: {len(_1min_cache)} symbols loaded")

    # 获取 entry timestamp——1min 模式需要
    has_ts = "timestamp" in df.columns
    use_1min = bool(_1min_cache)
    # slot 判定用时间戳 (nanoseconds int64) 代替 bar index，与事件侧 1min 精度对齐
    _slot_use_ts = use_1min and has_ts

    for sym, group in df.groupby(sym_col, sort=False):
        group = group.sort_index()
        idx_arr = group.index.values
        highs = group["high"].values.astype(float)
        lows = group["low"].values.astype(float)
        closes = group["close"].values.astype(float)
        atrs = group[atr_col].values.astype(float)
        directions = group[dir_col].values.astype(float)
        n = len(group)

        # 1min bar 数据查找表
        sym_1min = _1min_cache.get(sym)
        if use_1min and has_ts:
            entry_timestamps = pd.to_datetime(group["timestamp"]).values.astype("int64")

        # Per-entry 参数数组（tier 模式）
        if tier_mode:
            t_initial_r = group["_tier_initial_r"].values.astype(float)
            t_activation_r = group["_tier_activation_r"].values.astype(float)
            t_trail_r = group["_tier_trail_r"].values.astype(float)
            t_timeout = group["_tier_timeout"].values.astype(int)

        # structural exit 数组 (BPC trend_hold: ema200)
        _has_structural = "_structural_exit" in group.columns
        _has_ema200 = "ema_200" in group.columns
        if _has_structural:
            t_structural_exit = group["_structural_exit"].values
        if _has_ema200:
            t_ema200 = group["ema_200"].values.astype(float)

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

            # structural exit 参数 (BPC trend_hold: ema200)
            structural_exit_type = ""
            structural_ema200 = 0.0
            if _has_structural:
                structural_exit_type = str(t_structural_exit[i] or "")
            if structural_exit_type == "ema200" and _has_ema200:
                structural_ema200 = t_ema200[i] if not np.isnan(t_ema200[i]) else 0.0

            # ====== 1min bar 精细模拟 ======
            if use_1min and sym_1min is not None and has_ts:
                entry_ts_ns = entry_timestamps[i]
                m_ts = sym_1min["ts"]
                m_h = sym_1min["high"]
                m_l = sym_1min["low"]
                m_c = sym_1min["close"]
                # time_stop_bars 按 archetype bar 数，换算成 1min
                bar_minutes = (
                    int(group.iloc[i].get("_bar_minutes", 240))
                    if "_bar_minutes" in group.columns
                    else 240
                )
                # 找 entry bar CLOSE 之后的 1min bar 起始位置
                # entry_ts_ns 是 bar OPEN 时间，但入场价 = bar CLOSE 价格，
                # 事件回测的 enforce_position 从 bar CLOSE 后的第一根 1min bar 开始。
                # 如果用 bar OPEN，会多模拟 bar_minutes 的 1min bars（交易还没开仓！）
                _entry_close_ns = int(entry_ts_ns) + int(bar_minutes) * 60 * 10**9
                start_m = int(np.searchsorted(m_ts, _entry_close_ns, side="right"))
                # time_stop_bars=0 → 使用全部可用 1min bars (fat tail: 无限持仓)
                if time_stop_bars > 0:
                    max_m = min(start_m + time_stop_bars * bar_minutes, len(m_ts))
                else:
                    max_m = len(m_ts)
                _1min_exit_mi = None  # 精确退出的 1min bar index

                for mi in range(start_m, max_m):
                    h = m_h[mi]
                    l = m_l[mi]
                    if np.isnan(h) or np.isnan(l):
                        continue

                    # ── 顺序对齐 enforce_position (先更新再检查) ──

                    # 1. 保本锁定
                    if breakeven_lock_r > 0 and not breakeven_locked:
                        check_bp = h if direction == 1 else l
                        mfe_r_be = abs(check_bp - entry_price) / entry_atr
                        if mfe_r_be >= breakeven_lock_r:
                            breakeven_locked = True
                            if direction == 1:
                                if entry_price > sl_price:
                                    sl_price = entry_price
                            else:
                                if entry_price < sl_price:
                                    sl_price = entry_price

                    # 2. 更新最优价 (HWM/LWM)
                    if direction == 1:
                        if h > best_price:
                            best_price = h
                    else:
                        if l < best_price:
                            best_price = l

                    # 2b. Structural exit (EMA200) — BPC trend_hold
                    # 仅在 breakeven locked 后才检查 (避免入场即退出)
                    if (
                        structural_exit_type == "ema200"
                        and breakeven_locked
                        and structural_ema200 > 0
                    ):
                        _mc = m_c[mi]  # 1min close
                        if not np.isnan(_mc):
                            if direction == 1 and _mc < structural_ema200:
                                exit_price = _mc
                                exit_reason = "structural_exit_ema200"
                                _1min_exit_mi = mi
                                break
                            elif direction == -1 and _mc > structural_ema200:
                                exit_price = _mc
                                exit_reason = "structural_exit_ema200"
                                _1min_exit_mi = mi
                                break

                    # 3. 移动止损 (trailing activation + SL update)
                    _just_activated = False
                    if stop_type == "trailing":
                        mfe_r = abs(best_price - entry_price) / entry_atr
                        if not trailing_active and mfe_r >= activation_r:
                            trailing_active = True
                            _just_activated = True  # 首次激活，本 bar 不检查 SL
                        if trailing_active:
                            if direction == 1:
                                new_sl = best_price - trail_r * entry_atr
                                if new_sl > sl_price:
                                    sl_price = new_sl
                            else:
                                new_sl = best_price + trail_r * entry_atr
                                if new_sl < sl_price:
                                    sl_price = new_sl

                    # 4. SL 检查 (用刚更新的 sl_price)
                    # 首次 trailing 激活的 bar 跳过 SL 检查，避免同 bar 激活+触发
                    if _just_activated:
                        pass
                    elif direction == 1 and l <= sl_price:
                        exit_price = sl_price
                        exit_reason = "trailing_sl" if trailing_active else "sl"
                        _1min_exit_mi = mi
                        break
                    elif direction == -1 and h >= sl_price:
                        exit_price = sl_price
                        exit_reason = "trailing_sl" if trailing_active else "sl"
                        _1min_exit_mi = mi
                        break

                    # 5. TP 检查
                    if tp_enabled:
                        if direction == 1 and h >= entry_price + tp_r * entry_atr:
                            exit_price = entry_price + tp_r * entry_atr
                            exit_reason = "tp"
                            _1min_exit_mi = mi
                            break
                        elif direction == -1 and l <= entry_price - tp_r * entry_atr:
                            exit_price = entry_price - tp_r * entry_atr
                            exit_reason = "tp"
                            _1min_exit_mi = mi
                            break

                # 超时或数据不足
                if exit_price is None:
                    if max_m > start_m:
                        exit_price = float(m_c[max_m - 1])
                        exit_reason = "timeout"
                        _1min_exit_mi = max_m - 1
                    else:
                        exit_stats["no_data"] += 1
                        continue

            else:
                # ====== 原始 4H bar 模式 ======
                # time_stop_bars=0 → 使用全部可用 4H bars (fat tail: 无限持仓)
                if time_stop_bars > 0:
                    max_j = min(i + 1 + time_stop_bars, n)
                else:
                    max_j = n
                for j in range(i + 1, max_j):
                    h = highs[j]
                    l = lows[j]
                    if np.isnan(h) or np.isnan(l):
                        continue

                    # ── 顺序对齐 enforce_position (先更新再检查) ──

                    # 1. 保本锁定: MFE >= breakeven_lock_r → SL 移至入场价
                    if breakeven_lock_r > 0 and not breakeven_locked:
                        check_bp = h if direction == 1 else l
                        mfe_r_be = abs(check_bp - entry_price) / entry_atr
                        if mfe_r_be >= breakeven_lock_r:
                            breakeven_locked = True
                            if direction == 1:
                                if entry_price > sl_price:
                                    sl_price = entry_price
                            else:
                                if entry_price < sl_price:
                                    sl_price = entry_price

                    # 2. 更新最优价 (HWM/LWM)
                    if direction == 1:
                        if h > best_price:
                            best_price = h
                    else:
                        if l < best_price:
                            best_price = l

                    # 2b. Structural exit (EMA200) — BPC trend_hold
                    # 4H bar 模式: 用当前 bar 的 ema_200 (动态更新)
                    if (
                        structural_exit_type == "ema200"
                        and breakeven_locked
                        and _has_ema200
                    ):
                        _ema_j = t_ema200[j] if j < len(t_ema200) else 0.0
                        if _ema_j > 0 and not np.isnan(_ema_j):
                            if direction == 1 and closes[j] < _ema_j:
                                exit_price = closes[j]
                                exit_reason = "structural_exit_ema200"
                                break
                            elif direction == -1 and closes[j] > _ema_j:
                                exit_price = closes[j]
                                exit_reason = "structural_exit_ema200"
                                break

                    # 3. 移动止损 (trailing activation + SL update)
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

                    # 4. SL 检查 (用刚更新的 sl_price)
                    if direction == 1 and l <= sl_price:
                        exit_price = sl_price
                        exit_reason = "trailing_sl" if trailing_active else "sl"
                        break
                    elif direction == -1 and h >= sl_price:
                        exit_price = sl_price
                        exit_reason = "trailing_sl" if trailing_active else "sl"
                        break

                    # 5. TP 检查
                    if tp_enabled:
                        if direction == 1 and h >= entry_price + tp_r * entry_atr:
                            exit_price = entry_price + tp_r * entry_atr
                            exit_reason = "tp"
                            break
                        elif direction == -1 and l <= entry_price - tp_r * entry_atr:
                            exit_price = entry_price - tp_r * entry_atr
                            exit_reason = "tp"
                            break

                # 超时或数据不足
                if exit_price is None:
                    if max_j > i + 1:
                        exit_price = closes[max_j - 1]
                        exit_reason = "timeout"
                    else:
                        exit_stats["no_data"] += 1
                        continue

            # 计算 realized R/R (除以 initial_r × ATR，反映仓位大小)
            # 1R = initial_r × ATR = 止损距离 = position sizing 基准
            # 这样 wider stop → 更大的 1R → 同样价格变动产生更小的 R
            risk_distance = initial_r * entry_atr
            if direction == 1:
                realized_rr = (exit_price - entry_price) / risk_distance
            else:
                realized_rr = (entry_price - exit_price) / risk_distance

            results.iloc[results.index.get_loc(idx_arr[i])] = realized_rr
            total_entries += 1
            exit_stats[exit_reason] += 1
            if breakeven_locked:
                breakeven_lock_count += 1

            # 记录交易详情
            if use_1min and sym_1min is not None and has_ts:
                # 1min path: 估算退出对应的 4H bar 偏移
                if exit_reason == "timeout":
                    exit_bar_idx = min(i + time_stop_bars, n - 1)
                else:
                    bars_elapsed = (
                        max(1, (mi - start_m) // bar_minutes + 1) if mi > start_m else 1
                    )
                    exit_bar_idx = min(i + bars_elapsed, n - 1)
            else:
                # 4H path
                exit_bar_idx = j if exit_reason != "timeout" else (max_j - 1)
            # 精确退出时间戳 (1min 模式用实际 1min bar ts; 4H 模式用 bar timestamp)
            # entry_ts_ns 用 bar CLOSE 时间 (= open + bar_minutes)，与事件回测 decide() 调用时机对齐:
            #   事件回测: bar 收盘 → enforce_position 处理完 → decide() 检查 slot
            #   向量回测: slot 判定应在 bar 收盘时刻，已退出的仓位不再占用 slot
            _exit_ts_ns = 0
            _entry_ts_ns = 0
            _bm_for_ts = (
                int(group.iloc[i].get("_bar_minutes", 240))
                if "_bar_minutes" in group.columns
                else 240
            )
            _bar_close_offset_ns = int(_bm_for_ts) * 60 * 10**9
            if _slot_use_ts and sym_1min is not None and _1min_exit_mi is not None:
                _exit_ts_ns = int(m_ts[_1min_exit_mi])
                _entry_ts_ns = int(entry_ts_ns) + _bar_close_offset_ns
            elif _slot_use_ts and sym_1min is not None and exit_reason == "timeout":
                # timeout: _1min_exit_mi is None (没触发 SL/trailing)
                # 直接用时间计算，不用 mixed group index (混合了 1H+4H bars 会算错)
                _entry_ts_ns = int(entry_ts_ns) + _bar_close_offset_ns
                _exit_ts_ns = (
                    _entry_ts_ns + int(time_stop_bars) * int(bar_minutes) * 60 * 10**9
                )
            elif has_ts:
                try:
                    _entry_ts_ns = int(entry_timestamps[i]) + _bar_close_offset_ns
                    _exit_ts_ns = int(
                        pd.to_datetime(
                            group.iloc[min(exit_bar_idx, n - 1)]["timestamp"]
                        ).value
                    )
                except Exception:
                    pass
            trade_details.append(
                {
                    "symbol": sym,
                    "entry_idx": int(idx_arr[i]),
                    "exit_idx": int(idx_arr[min(exit_bar_idx, n - 1)]),
                    "entry_ts_ns": _entry_ts_ns,
                    "exit_ts_ns": _exit_ts_ns,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "direction": direction,
                    "realized_rr": float(realized_rr),
                    "exit_reason": exit_reason,
                    "evidence_score": float(
                        group.iloc[i].get("evidence_score", 0.5)
                        if "evidence_score" in group.columns
                        else 0.5
                    ),
                    "archetype": (
                        str(group.iloc[i].get("_pcm_archetype", ""))
                        if "_pcm_archetype" in group.columns
                        else ""
                    ),
                    "_bar_minutes": (
                        int(group.iloc[i].get("_bar_minutes", 240))
                        if "_bar_minutes" in group.columns
                        else 240
                    ),
                    "breakeven_locked": breakeven_locked,
                    "is_add_position": False,
                }
            )

    # ── slot 限制：per-strategy 独立 slot ──
    _add_pos_count = 0
    removed_indices: set = set()  # 初始化以避免 slot 过滤未执行时 NameError
    if max_slots and max_slots > 0 and trade_details:
        # 加仓配置
        _ap_enabled = add_position_cfg is not None
        _ap_rules = (
            add_position_cfg.get("add_position_rules", {}) if _ap_enabled else {}
        )
        _ap_per_strat = (
            add_position_cfg.get("per_strategy_limits", {}) if _ap_enabled else {}
        )
        _ap_max_add = int(_ap_rules.get("max_add_times", 1))

        # per-strategy slot 限制 — 始终从 constitution per_strategy_limits 读取
        # (不依赖 add_position_cfg 是否存在, 与事件侧 LivePCM._max_slots_for_strategy 对齐)
        _per_strat_max = {}  # archetype → max_slots
        # 优先从 per_strategy_limits 参数读取 (直接传入, 独立于 add_position)
        if per_strategy_limits:
            for arch_key, cfg in per_strategy_limits.items():
                if isinstance(cfg, dict) and "max_slots" in cfg:
                    _per_strat_max[arch_key.lower()] = int(cfg["max_slots"])
        elif _ap_per_strat:
            # 兼容旧调用: 从 add_position_cfg 回退读取
            for arch_key, cfg in _ap_per_strat.items():
                if isinstance(cfg, dict) and "max_slots" in cfg:
                    _per_strat_max[arch_key.lower()] = int(cfg["max_slots"])

        # 排序: 用时间戳 (跨 symbol 正确) 而非 dataframe index
        # merged 按 (symbol, timestamp) 排序 → 每 symbol 独占一段 index → exit_idx vs eidx 跨 symbol 永远 False
        # 必须用 entry_ts_ns 排序 + exit_ts_ns 对比, 才能正确实现跨 symbol slot 阻塞
        _arch_priority_map = {
            "lv": 0,
            "fer": 1,
            "me-long": 2,
            "bpc": 3,
        }  # 同时间戳: 高优先级先处理
        if _slot_use_ts:
            trade_details.sort(
                key=lambda t: (
                    t.get("entry_ts_ns", 0),
                    _arch_priority_map.get(t.get("archetype", "").lower().strip(), 99),
                )
            )
        else:
            trade_details.sort(key=lambda t: t["entry_idx"])
        accepted = []  # 当前 active trades
        rejected_indices = set()
        evicted_indices = set()
        _slot_diag = {"per_strat": 0, "global": 0, "both": 0}
        for trade in trade_details:
            eidx = trade["entry_idx"]
            trade_arch = trade.get("archetype", "").lower().strip()

            # 移除已平仓的 active trades
            if _slot_use_ts and trade.get("entry_ts_ns", 0) > 0:
                _entry_ns = trade["entry_ts_ns"]
                active = [t for t in accepted if t.get("exit_ts_ns", 0) > _entry_ns]
            else:
                active = [t for t in accepted if t["exit_idx"] > eidx]

            # 检查 per-strategy slot 限制
            arch_max = _per_strat_max.get(trade_arch, max_slots)  # 缺省回退全局
            arch_active = [
                t
                for t in active
                if t.get("archetype", "").lower().strip() == trade_arch
            ]
            per_strat_full = len(arch_active) >= arch_max
            global_full = len(active) >= max_slots

            if per_strat_full or global_full:
                # slot 满 — 先检查是否可以加仓
                can_add = False
                if _ap_enabled:
                    strat_cfg = _ap_per_strat.get(trade_arch, {})
                    if strat_cfg.get("allow_add_position", False):
                        # 找出同 symbol + 同 direction 的所有活跃仓位
                        _same_sym_dir = [
                            t
                            for t in active
                            if t["symbol"] == trade["symbol"]
                            and t["direction"] == trade["direction"]
                        ]
                        # 无风险加仓: 所有现有仓位都必须 breakeven locked
                        _all_locked = _same_sym_dir and all(
                            t.get("breakeven_locked", False) for t in _same_sym_dir
                        )
                        if _all_locked:
                            # 找父单 (跟踪 add_count 的那个)
                            for at in _same_sym_dir:
                                if at.get("_add_count", 0) < _ap_max_add:
                                    can_add = True
                                    at["_add_count"] = at.get("_add_count", 0) + 1
                                    trade["is_add_position"] = True
                                    _add_pos_count += 1
                                    break

                if can_add:
                    accepted = active + [trade]
                else:
                    # Slot 满 → 直接拒绝
                    rejected_indices.add(eidx)
                    if per_strat_full and global_full:
                        _slot_diag["both"] += 1
                    elif per_strat_full:
                        _slot_diag["per_strat"] += 1
                    else:
                        _slot_diag["global"] += 1
            else:
                accepted = active + [trade]

        # 清理被驱逐 + 被拒绝的 trades
        removed_indices = rejected_indices | evicted_indices
        if removed_indices:
            for ridx in removed_indices:
                loc = results.index.get_loc(ridx)
                results.iloc[loc] = np.nan
            trade_details = [
                t for t in trade_details if t["entry_idx"] not in removed_indices
            ]
            total_entries -= len(removed_indices)
            if not silent:
                print(
                    f"   🔒 Slot limit (max={max_slots}): "
                    f"rejected {len(rejected_indices)}, "
                    f"kept {total_entries}"
                )
                print(
                    f"   🔍 Reject reasons: "
                    f"per_strat={_slot_diag['per_strat']}, "
                    f"global={_slot_diag['global']}, "
                    f"both={_slot_diag['both']}"
                )

    if not silent:
        # exit_stats 是 slot 过滤前的全量统计; 有 slot 过滤时重算保留交易的统计
        _display_stats = exit_stats
        if max_slots and max_slots > 0 and removed_indices:
            _display_stats = {
                "sl": 0,
                "trailing_sl": 0,
                "tp": 0,
                "timeout": 0,
                "no_data": 0,
                "structural_exit_ema200": 0,
            }
            for t in trade_details:
                er = t.get("exit_reason", "")
                if er in _display_stats:
                    _display_stats[er] += 1
        print(
            f"   📊 Simulated {total_entries} trades: "
            f"SL={_display_stats['sl']}, TrailSL={_display_stats['trailing_sl']}, "
            f"TP={_display_stats['tp']}, Timeout={_display_stats['timeout']}, "
            f"NoData={_display_stats['no_data']}, EMA200Exit={_display_stats['structural_exit_ema200']}"
        )
        if breakeven_lock_r > 0:
            pct = breakeven_lock_count / total_entries * 100 if total_entries > 0 else 0
            print(
                f"   🔒 Breakeven lock: {breakeven_lock_count}/{total_entries} trades ({pct:.1f}%) reached {breakeven_lock_r}R"
            )

    return results, trade_details


def _export_trade_details_csv(
    trade_details: List[Dict[str, Any]],
    output_path: str,
    df: pd.DataFrame,
) -> None:
    """导出 trade_details 为 CSV，与事件回测 export_trades_csv 格式对齐。

    列: symbol, side, entry_price, exit_price, entry_time, exit_time,
         pnl_r, exit_reason, archetype, evidence, bars_held
    """
    has_ts = "timestamp" in df.columns
    rows = []
    for t in trade_details:
        entry_idx = t.get("entry_idx", 0)
        exit_idx = t.get("exit_idx", 0)
        direction = t.get("direction", 0)
        side = "LONG" if direction > 0 else ("SHORT" if direction < 0 else "UNKNOWN")
        # 优先用精确时间戳 (entry_ts_ns / exit_ts_ns)，避免 mixed-timeframe group index 错误
        entry_time = ""
        exit_time = ""
        _ets = t.get("entry_ts_ns", 0)
        _xts = t.get("exit_ts_ns", 0)
        if _ets > 0 and _xts > 0:
            entry_time = str(pd.Timestamp(_ets, tz="UTC"))
            exit_time = str(pd.Timestamp(_xts, tz="UTC"))
        elif has_ts:
            try:
                entry_time = str(df.iloc[entry_idx]["timestamp"])
                exit_time = str(df.iloc[exit_idx]["timestamp"])
            except (IndexError, KeyError):
                pass
        elif isinstance(df.index, pd.DatetimeIndex):
            try:
                entry_time = str(df.index[entry_idx])
                exit_time = str(df.index[exit_idx])
            except (IndexError, KeyError):
                pass
        # bars_held: 用精确时间戳算 archetype bars，避免 mixed-frame index 差值
        if _ets > 0 and _xts > 0:
            _bm = max(1, t.get("_bar_minutes", 240))
            bars_held = max(1, int((_xts - _ets) / (_bm * 60 * 10**9)))
        else:
            bars_held = max(1, int(exit_idx - entry_idx))
        rows.append(
            {
                "symbol": t.get("symbol", ""),
                "side": side,
                "entry_price": round(float(t.get("entry_price", 0)), 6),
                "exit_price": round(float(t.get("exit_price", 0)), 6),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "pnl_r": round(float(t.get("realized_rr", 0)), 4),
                "exit_reason": t.get("exit_reason", ""),
                "archetype": t.get("archetype", ""),
                "evidence": round(float(t.get("evidence_score", 0.5)), 4),
                "bars_held": bars_held,
            }
        )
    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\n   📤 Trades exported: {len(out)} rows → {output_path}")


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
    """从 DataFrame 估算数据跨度(年)。优先使用 timestamp 列精确计算。"""
    # 优先用 timestamp 直接计算 (多 timeframe merged 时 bar 数会膨胀)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        span_days = (ts.max() - ts.min()).total_seconds() / 86400
        return max(span_days / 365.25, 0.01)
    # fallback: 按 bar 数估算 (仅适用于单 timeframe)
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
    tf = meta.get("timeframe")
    if not tf:
        # 尝试 strategy.timeframe
        s = meta.get("strategy") or {}
        tf = s.get("timeframe")
    return tf


def _eval_when_vectorized(
    when: Dict[str, Any],
    df: pd.DataFrame,
) -> pd.Series:
    """向量化评估一个 when 子句, 返回 bool Series (True=条件命中)。

    与事件侧 _evaluate_when_clause (loader.py) 完全对齐:
    - all_of: AND (所有子条件均命中)
    - any_of: OR  (任一子条件命中)
    - 单 feature: AND (多个 feature 全部命中)
    """
    if not when:
        return pd.Series(False, index=df.index)

    # all_of: 递归 AND
    if "all_of" in when:
        conditions = when["all_of"]
        min_matches = int(when.get("min_matches", len(conditions)))
        if not conditions:
            return pd.Series(False, index=df.index)
        match_count = pd.Series(0, index=df.index, dtype=int)
        for sub in conditions:
            match_count += _eval_when_vectorized(sub, df).astype(int)
        return match_count >= min_matches

    # any_of: 递归 OR
    if "any_of" in when:
        conditions = when["any_of"]
        min_matches = int(when.get("min_matches", 1))
        if not conditions:
            return pd.Series(False, index=df.index)
        match_count = pd.Series(0, index=df.index, dtype=int)
        for sub in conditions:
            match_count += _eval_when_vectorized(sub, df).astype(int)
        return match_count >= min_matches

    # 单条件 / 多 feature: AND 逻辑 (与事件侧一致)
    result = pd.Series(True, index=df.index)
    has_any_feature = False
    for feature, conditions in when.items():
        if feature in ("all_of", "any_of", "min_matches"):
            continue
        if feature not in df.columns:
            # 特征缺失 → 不匹配 (与事件侧 on_missing=false 一致)
            return pd.Series(False, index=df.index)
        if not isinstance(conditions, dict):
            continue
        has_any_feature = True
        col = pd.to_numeric(df[feature], errors="coerce")
        for op, threshold in conditions.items():
            if op == "on_missing":
                continue
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                continue
            if op == "value_gt":
                result &= col > threshold
            elif op in ("value_gte", "value_ge"):
                result &= col >= threshold
            elif op == "value_lt":
                result &= col < threshold
            elif op in ("value_lte", "value_le"):
                result &= col <= threshold
    if not has_any_feature:
        return pd.Series(False, index=df.index)
    return result


def _apply_prefilter_vectorized(
    df: pd.DataFrame,
    arch_name: str,
    strategies_root: str,
) -> int:
    """从 prefilter.yaml 评估 prefilter (向量化), 将不满足的 bar 的 entry_direction 设为 0。

    语义: archetype 成立的前置环境条件, 不满足时不应产生信号。
    在 Direction 之后、Gate 之前执行。

    Returns:
        通过 prefilter 的行数
    """
    import operator as _op

    _PF_OPS = {
        ">=": _op.ge,
        ">": _op.gt,
        "<=": _op.le,
        "<": _op.lt,
        "==": _op.eq,
        "!=": _op.ne,
    }

    pf_path = Path(strategies_root) / arch_name / "archetypes" / "prefilter.yaml"
    if not pf_path.exists():
        return len(df)

    raw = yaml.safe_load(pf_path.read_text(encoding="utf-8")) or {}
    rules = raw.get("rules", [])
    if not rules:
        return len(df)

    reject_mask = pd.Series(False, index=df.index)
    reject_detail: Dict[str, int] = {}

    for rule in rules:
        if "any_of" in rule:
            # any_of OR 组: 全部子规则都不满足才 reject
            sub_rules = rule["any_of"]
            all_fail = pd.Series(True, index=df.index)
            descs = []
            for sub in sub_rules:
                sf, sop, sv = sub.get("feature"), sub.get("operator"), sub.get("value")
                if sf not in df.columns:
                    print(
                        f"   ⚠️  Prefilter: feature '{sf}' not in columns, treated as fail"
                    )
                    descs.append(f"{sf}{sop}{sv}[MISSING]")
                    continue
                op_func = _PF_OPS.get(sop)
                if op_func is None:
                    continue
                sub_pass = op_func(df[sf], sv)
                all_fail &= ~sub_pass
                descs.append(f"{sf}{sop}{sv}")
            n_rej = int(all_fail.sum())
            if n_rej > 0:
                reject_mask |= all_fail
                reject_detail[f"any_of({','.join(descs)})"] = n_rej
        else:
            feat = rule.get("feature")
            op_str = rule.get("operator")
            val = rule.get("value")
            if not feat or not op_str:
                continue
            if feat not in df.columns:
                print(
                    f"   ⚠️  Prefilter: feature '{feat}' not in columns — ALL bars rejected!"
                )
                reject_mask[:] = True
                reject_detail[f"{feat}(MISSING)"] = len(df)
                continue
            op_func = _PF_OPS.get(op_str)
            if op_func is None:
                continue
            fail_mask = ~op_func(df[feat], val)
            n_rej = int(fail_mask.sum())
            if n_rej > 0:
                reject_mask |= fail_mask
                reject_detail[f"{feat}{op_str}{val}"] = n_rej

    df.loc[reject_mask, "entry_direction"] = 0.0
    n_pass = int((~reject_mask).sum())
    print(f"   🛡️  Prefilter: {n_pass} pass / {len(df)} total")
    for desc, cnt in reject_detail.items():
        print(f"      {desc}: {cnt} reject")
    return n_pass


def _apply_gate_from_yaml_vectorized(
    df: pd.DataFrame,
    arch_name: str,
    strategies_root: str,
) -> int:
    """从 gate.yaml 重新评估 gate (向量化), 写入 gate_decision 列。

    支持 hard_gates 中的 value_gt/value_lt/value_le/value_ge 条件,
    以及 all_of/any_of 复合条件 (与事件侧 _evaluate_when_clause 完全对齐)。
    每条规则: when 条件命中 + action=deny → 该 bar 被否决。
    所有规则取 OR (任一 deny → 最终 deny)。

    Returns:
        allow 的行数
    """
    gate_cfg = load_gate_config(arch_name, strategies_root)
    hard_gates = gate_cfg.get("hard_gates", [])

    deny_mask = pd.Series(False, index=df.index)
    deny_detail: Dict[str, int] = {}  # rule_id → deny count

    for rule in hard_gates:
        when = rule.get("when", {})
        action = rule.get("then", {}).get("action", "deny")
        if action != "deny":
            continue

        rule_id = rule.get("id", "unknown")
        rule_deny = _eval_when_vectorized(when, df)

        n_deny = int(rule_deny.sum())
        if n_deny > 0:
            deny_detail[rule_id] = n_deny
        deny_mask |= rule_deny

    df["gate_decision"] = deny_mask.map({True: "deny", False: "allow"})
    n_allow = int((~deny_mask).sum())
    print(f"   🚪 Gate (from yaml): {n_allow} allow / {len(df)} total")
    for rid, cnt in deny_detail.items():
        print(f"      {rid}: {cnt} deny")
    return n_allow


def _load_raw_features_for_archetype(
    arch_name: str,
    strategies_root: str,
    symbols: List[str],
    data_path: str,
    test_start: str,
    test_end: str,
    warmup_days: int = 150,
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """从原始 1min 数据计算全量特征, 返回测试期 DataFrame。

    流程:
      1. 从 meta.yaml 读取 timeframe
      2. 初始化 IncrementalFeatureComputer (同事件回测)
      3. 对每个 symbol: 加载 1min bars → 计算特征 → 截取测试期
      4. 合并返回 (test_df, {})

    返回的 DataFrame 包含 OHLC + 所有策略特征, 不含 gate_decision/pred 列。
    precomputed_quantiles: 从 evidence.yaml 特征的分位数。
    """
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )
    from src.time_series_model.live.live_feature_plan import (
        extract_features_from_archetypes,
    )
    from src.data_tools.data_handler import DataHandler

    # 1. Timeframe
    tf = load_meta_timeframe(arch_name, strategies_root)
    if not tf:
        raise ValueError(f"{arch_name}: meta.yaml 中无 timeframe")
    print(f"\n📂 {arch_name}: from-raw mode, tf={tf}, {len(symbols)} symbols")

    # 2. FeatureComputer
    archetypes_dir = str(Path(strategies_root) / arch_name / "archetypes")
    fc = IncrementalFeatureComputer(
        primary_timeframe=tf,
        archetypes_dir=archetypes_dir,
    )
    # 禁用 live_feature_set 过滤 — 保留所有计算出的特征 (同事件回测)
    fc.live_feature_set = None

    # 3. 加载并计算
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    test_end_ts = pd.Timestamp(test_end, tz="UTC")
    warmup_start = (test_start_ts - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    end_str = test_end_ts.strftime("%Y-%m-%d")

    dh = DataHandler(data_path)
    parts: List[pd.DataFrame] = []
    calib_parts: List[pd.DataFrame] = []  # warmup 特征 (保留以备将来使用)

    import time as _time

    for sym in symbols:
        t0 = _time.time()
        try:
            bars_1min = dh.load_ohlcv(
                symbol=sym, timeframe="1T", start_date=warmup_start, end_date=end_str
            )
        except Exception as e:
            # 缓存文件可能损坏, 删除后重试一次
            cache_file = Path("cache/timeframes") / f"{sym}_1T.parquet"
            if cache_file.exists():
                print(
                    f"   ⚠️  {sym}: 缓存读取失败, 清除缓存重试 ({e.__class__.__name__})"
                )
                cache_file.unlink()
                try:
                    bars_1min = dh.load_ohlcv(
                        symbol=sym,
                        timeframe="1T",
                        start_date=warmup_start,
                        end_date=end_str,
                    )
                except Exception as e2:
                    print(f"   ⚠️  {sym}: 重试仍失败, 跳过 ({e2})")
                    continue
            else:
                print(f"   ⚠️  {sym}: 数据加载失败, 跳过 ({e})")
                continue

        if bars_1min is None or len(bars_1min) < 100:
            print(
                f"   ⚠️  {sym}: bars 不足 ({len(bars_1min) if bars_1min is not None else 0}), 跳过"
            )
            continue

        bars_1min.index = pd.to_datetime(bars_1min.index, utc=True)
        col_rename = {"buy_qty": "buy_volume", "sell_qty": "sell_volume"}
        bars_1min = bars_1min.rename(
            columns={k: v for k, v in col_rename.items() if k in bars_1min.columns}
        )
        if "timestamp" not in bars_1min.columns:
            bars_1min["timestamp"] = bars_1min.index

        # 加载 ticks (同事件回测)
        tick_frames = []
        data_root = Path(data_path)
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
            _ws = pd.Timestamp(warmup_start, tz="UTC")
            ticks_1min = ticks_1min[
                (ticks_1min["timestamp"] >= _ws)
                & (ticks_1min["timestamp"] <= test_end_ts)
            ]
        else:
            ticks_1min = pd.DataFrame()

        # 重新初始化 FC 状态 (每个 symbol 独立计算)
        fc.reset()

        # 注入 _symbol 列 — OI join 等特征需要识别 symbol
        bars_1min["_symbol"] = sym

        try:
            features_df = fc.compute_features_dataframe(
                bars_1min=bars_1min,
                ticks_1min=ticks_1min,
                primary_timeframe=tf,
            )
        except Exception as e:
            print(f"   ⚠️  {sym}: 特征计算异常, 跳过 ({e})")
            continue

        if features_df.empty:
            print(f"   ⚠️  {sym}: 特征计算为空, 跳过")
            continue

        features_df.index = pd.to_datetime(features_df.index, utc=True)

        # 收集 warmup 特征 (test_start 之前) — 保留以备将来使用
        warmup_df = features_df[features_df.index < test_start_ts]
        if not warmup_df.empty:
            calib_parts.append(warmup_df)

        test_df = features_df[
            (features_df.index >= test_start_ts) & (features_df.index <= test_end_ts)
        ]
        if test_df.empty:
            print(f"   ⚠️  {sym}: 测试期无数据, 跳过")
            continue

        test_df = test_df.copy()
        test_df["symbol"] = sym
        test_df["timestamp"] = test_df.index
        elapsed = _time.time() - t0
        print(
            f"   {sym}: {len(bars_1min)} 1min bars → {len(test_df)} {tf} bars "
            f"({elapsed:.1f}s)"
        )
        parts.append(test_df)

    if not parts:
        raise ValueError(f"{arch_name}: 所有 symbol 均无有效数据")

    df = pd.concat(parts, ignore_index=True)
    print(f"   📊 {arch_name}: {len(df)} total bars ({len(parts)} symbols)")

    # Evidence quantiles: 从全量数据计算每个 evidence 特征的分位数
    precomputed_quantiles: Dict[str, Any] = {}
    if archetype and archetype.evidence and archetype.evidence.features:
        for feat in archetype.evidence.features:
            col = feat.feature
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) < 10:
                continue
            feat_q = {}
            for b in feat.quantile_bins:
                q_val = float(vals.quantile(b))
                q_key = f"{b:.2f}".rstrip("0").rstrip(".")
                feat_q[q_key] = q_val
            precomputed_quantiles[col] = feat_q
        if precomputed_quantiles:
            print(
                f"   📊 Evidence quantiles computed for "
                f"{len(precomputed_quantiles)} features"
            )

    return df, precomputed_quantiles


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
        arch_timeframes: per-archetype 时间粒度 (如 {'bpc':'240T','me-long':'60T'})
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
                        # 统一 tz: 确保 timestamp 列和比较值都是 tz-aware
                        _ts_col = pd.to_datetime(_sym_ohlc["timestamp"], utc=True)
                        _min_ts = (
                            pd.Timestamp(_min_ts, tz="UTC")
                            if _min_ts.tzinfo is None
                            else _min_ts
                        )
                        _max_ts = (
                            pd.Timestamp(_max_ts, tz="UTC")
                            if _max_ts.tzinfo is None
                            else _max_ts
                        )
                        _sym_ohlc = _sym_ohlc[
                            (_ts_col >= _min_ts - _buf) & (_ts_col <= _max_ts + _buf)
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
            if len(sym_df) == 0:
                continue  # 该 (archetype, symbol) 组合无 K 线数据，跳过
            if "open" not in sym_df.columns:
                sym_df["open"] = sym_df["close"].shift(1).fillna(sym_df["close"])

            # ---- 序号 x 轴 (消除时间间隙) ----
            sym_df["_seq"] = range(len(sym_df))
            x_labels = sym_df[ts_col_local].dt.strftime("%Y-%m-%d %H:%M").tolist()
            seq_to_label = {i: lbl for i, lbl in enumerate(x_labels)}

            # timestamp(str) → seq 快查 (tz 归一化: 去除 +00:00 避免匹配失败)
            def _norm_ts_str(ts_val) -> str:
                """统一为 tz-naive 字符串，消除 '+00:00' 与无 tz 的格式差异。"""
                try:
                    t = pd.Timestamp(ts_val)
                    if t.tzinfo is not None:
                        t = t.tz_localize(None)
                    return str(t)
                except Exception:
                    return str(ts_val)

            ts_str_to_seq = {
                _norm_ts_str(ts): seq
                for ts, seq in zip(sym_df[ts_col_local].values, sym_df["_seq"].values)
            }

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
            # 入场: 箭头方向 = 交易方向 (long=▲, short=▼)
            # 出场: ⭕ 空心圆
            # 颜色 = 盈亏 (绿=盈利, 红=亏损)
            _WIN_COLOR = "#26a69a"  # 绿 (盈利)
            _LOSS_COLOR = "#ef5350"  # 红 (亏损)

            # 按 (direction, win/loss) 分 4 组 (入场)
            groups: Dict[str, Dict[str, list]] = {
                "long_win": {"x": [], "y": [], "info": []},
                "long_loss": {"x": [], "y": [], "info": []},
                "short_win": {"x": [], "y": [], "info": []},
                "short_loss": {"x": [], "y": [], "info": []},
            }
            # 出场分 2 组 (按盈亏着色)
            exit_groups: Dict[str, Dict[str, list]] = {
                "win": {"x": [], "y": [], "info": []},
                "loss": {"x": [], "y": [], "info": []},
            }
            all_rr = []

            for t in sym_trades:
                # 优先使用精确 ts_ns（回测执行层产出的真实入/出场时刻）
                entry_seq = None
                exit_seq = None
                if t.get("entry_ts_ns"):
                    _eq = pd.Timestamp(t["entry_ts_ns"], unit="ns").tz_localize(None)
                    entry_seq = ts_str_to_seq.get(str(_eq))
                if t.get("exit_ts_ns"):
                    _xq = pd.Timestamp(t["exit_ts_ns"], unit="ns").tz_localize(None)
                    exit_seq = ts_str_to_seq.get(str(_xq))
                # fallback: 用 entry_idx/exit_idx 回查时间（兼容旧 trade_details）
                if entry_seq is None or exit_seq is None:
                    try:
                        entry_ts_str = _norm_ts_str(df.loc[t["entry_idx"], ts_col])
                        exit_ts_str = _norm_ts_str(df.loc[t["exit_idx"], ts_col])
                    except KeyError:
                        continue
                    entry_seq = (
                        entry_seq
                        if entry_seq is not None
                        else ts_str_to_seq.get(entry_ts_str)
                    )
                    exit_seq = (
                        exit_seq
                        if exit_seq is not None
                        else ts_str_to_seq.get(exit_ts_str)
                    )
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

                # 出场圆 (exit_seq, exit_price)
                exit_key = "win" if is_win else "loss"
                exit_groups[exit_key]["x"].append(exit_seq)
                exit_groups[exit_key]["y"].append(t["exit_price"])
                exit_groups[exit_key]["info"].append(hover_text)

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

            # 出场圆: ⭕ 空心圆 (盈利=绿, 亏损=红)
            if exit_groups["win"]["x"]:
                src_ew = ColumnDataSource(exit_groups["win"])
                p.scatter(
                    "x",
                    "y",
                    source=src_ew,
                    marker="circle",
                    size=8,
                    fill_color=None,
                    line_color=_WIN_COLOR,
                    line_width=2,
                    legend_label=f"⭕ Exit Win ({len(exit_groups['win']['x'])})",
                )
            if exit_groups["loss"]["x"]:
                src_el = ColumnDataSource(exit_groups["loss"])
                p.scatter(
                    "x",
                    "y",
                    source=src_el,
                    marker="circle",
                    size=8,
                    fill_color=None,
                    line_color=_LOSS_COLOR,
                    line_width=2,
                    legend_label=f"⭕ Exit Loss ({len(exit_groups['loss']['x'])})",
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
                t_seq = None
                if t.get("entry_ts_ns"):
                    _eq = pd.Timestamp(t["entry_ts_ns"], unit="ns").tz_localize(None)
                    t_seq = ts_str_to_seq.get(str(_eq))
                if t_seq is None:
                    try:
                        t_ts_str = _norm_ts_str(df.loc[t["entry_idx"], ts_col])
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
_PCM_DEFAULT_PRIORITY = ["LV", "FER", "ME-LONG", "BPC"]


def load_direction_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/direction.yaml 配置"""
    path = Path(strategies_root) / strategy / "archetypes" / "direction.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_transform(series: pd.Series, transform: str) -> np.ndarray:
    """Apply direction transform to a numeric series."""
    if transform == "raw":
        return series.values
    elif transform == "sign":
        return np.sign(series).values
    elif transform == "negate_sign":
        return (-np.sign(series)).values
    elif transform == "center_sign":
        return np.sign(series - 0.5).values
    else:
        return series.values


def apply_direction_rules(
    df: pd.DataFrame,
    archetype: str,
    direction_cfg: Dict[str, Any],
) -> str:
    """根据 direction.yaml 规则确定方向列。

    按 direction_rules 优先级从高到低尝试：
    - 命中第一个可用规则后设置 entry_direction
    - 对 direction=0 的行，继续用后续规则 fallback 填充
    - 直到所有行都有方向或规则用尽

    特殊字段:
    - fixed_direction: long/short → 忽略 direction_rules，全部 bar 固定方向
    - direction_filter: long/short → 方向模型正常运行，但只保留指定方向

    Returns:
        使用的方向列名（已写入 df['entry_direction']）或 None
    """
    # ── fixed_direction: 跳过 direction_rules，所有 bar 固定方向 ──
    _fd = direction_cfg.get("fixed_direction", None)
    if _fd == "long":
        df["entry_direction"] = 1.0
        print(f"   Direction: fixed_direction=long → ALL {len(df)} bars = LONG")
        return "fixed_direction"
    elif _fd == "short":
        df["entry_direction"] = -1.0
        print(f"   Direction: fixed_direction=short → ALL {len(df)} bars = SHORT")
        return "fixed_direction"

    rules = direction_cfg.get("direction_rules", [])

    first_feature = None
    applied_rules = []

    for rule in rules:
        feature = rule.get("feature", "")
        transform = rule.get("transform", "raw")

        if feature not in df.columns:
            continue

        series = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        direction_vals = _apply_transform(series, transform)

        if first_feature is None:
            # 第一条规则：初始化 entry_direction
            df["entry_direction"] = direction_vals
            first_feature = feature
            desc = rule.get("description", feature)
            n_nonzero = int((df["entry_direction"] != 0).sum())
            n_zero = len(df) - n_nonzero
            print(f"   Direction: {feature} (transform={transform}) | {desc}")
            print(f"     → {n_nonzero} have direction, {n_zero} remain undecided")
            applied_rules.append(feature)
            if n_zero == 0:
                return first_feature
        else:
            # 后续规则：只填充 direction=0 的行
            zero_mask = df["entry_direction"] == 0
            n_before = int(zero_mask.sum())
            if n_before == 0:
                break
            df.loc[zero_mask, "entry_direction"] = direction_vals[zero_mask.values]
            n_filled = n_before - int((df["entry_direction"] == 0).sum())
            if n_filled > 0:
                desc = rule.get("description", feature)
                print(
                    f"     fallback: {feature} (transform={transform}) filled {n_filled} rows"
                )
                applied_rules.append(feature)

    if first_feature:
        n_final = int((df["entry_direction"] != 0).sum())
        print(
            f"     → final: {n_final}/{len(df)} have direction ({len(applied_rules)} rules used)"
        )

    # ── direction_filter: 方向模型结果只保留指定方向，不匹配的置 0 ──
    _df = direction_cfg.get("direction_filter", None)
    if _df == "long":
        n_before = int((df["entry_direction"] != 0).sum())
        df.loc[df["entry_direction"] < 0, "entry_direction"] = 0.0
        n_after = int((df["entry_direction"] != 0).sum())
        print(
            f"     direction_filter=long: {n_before} → {n_after} (过滤掉 {n_before - n_after} SHORT)"
        )
    elif _df == "short":
        n_before = int((df["entry_direction"] != 0).sum())
        df.loc[df["entry_direction"] > 0, "entry_direction"] = 0.0
        n_after = int((df["entry_direction"] != 0).sum())
        print(
            f"     direction_filter=short: {n_before} → {n_after} (过滤掉 {n_before - n_after} LONG)"
        )

    return first_feature


def _pcm_get_priority_rank(archetype: str, priority: List[str]) -> int:
    """获取 archetype 的优先级排名（越小越优先，与 live_pcm.py 一致）"""
    arch_lower = archetype.lower()
    for i, a in enumerate(priority):
        if a.lower() == arch_lower:
            return i
    return len(priority)  # 未知 archetype 排最后


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
    evidence_min_score = const.get("evidence_min_score", 0.0)
    evidence_position_scale = const.get("evidence_position_scale", False)

    print(f"   📄 Regime config: {regime_config_path}")
    print(f"   📄 Constitution: {constitution_yaml or 'defaults'}")
    print(
        f"   🔒 max_slots={max_slots} (from {'args' if getattr(args, 'max_slots', None) else 'constitution'})"
    )

    # ── 1. 解析 --pcm 参数 ──
    arch_specs: Dict[str, str] = {}  # {archetype: logs_path}
    from_raw_mode = getattr(args, "from_raw", False)
    for spec in args.pcm:
        if ":" in spec:
            parts = spec.split(":", 1)
            arch_name, logs_path = parts
            arch_specs[arch_name] = logs_path
        elif from_raw_mode:
            # --from-raw 模式: 纯名称即可, 无需路径
            arch_specs[spec] = ""
        else:
            print(
                f"❌ Invalid --pcm format: {spec}. Expected archetype:path (or use --from-raw for name-only)"
            )
            return 1

    arch_names = list(arch_specs.keys())
    print(f"\n📋 Archetypes: {arch_names}")
    print(f"   Priority: {' > '.join(priority)}")
    print(f"   决策依据: 按语义要求的条件严格性划分（越严格越优先）")

    # ── 2. 加载各 archetype 配置 + 处理信号 ──
    arch_exec_configs: Dict[str, Dict] = {}
    arch_processed: Dict[str, pd.DataFrame] = {}  # direction + evidence
    arch_precomputed_quantiles: Dict[str, Dict[str, List[float]]] = (
        {}
    )  # from-raw warmup
    base_df = None

    # --from-raw 模式: 从原始数据计算特征
    from_raw = getattr(args, "from_raw", False)
    recompute_gate = getattr(args, "recompute_gate", False) or from_raw
    raw_symbols = getattr(args, "symbols", None)
    if raw_symbols and isinstance(raw_symbols, str):
        raw_symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]
    elif not raw_symbols:
        raw_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

    raw_test_start = getattr(args, "test_start", None)
    raw_test_end = getattr(args, "test_end", None)
    raw_data_path = (
        getattr(args, "data_path", "data/parquet_data") or "data/parquet_data"
    )

    if from_raw:
        if not raw_test_start or not raw_test_end:
            print("❌ --from-raw 需要 --test-start 和 --test-end")
            return 1
        print(f"\n🔧 FROM-RAW 模式: 从原始数据计算全量特征")
        print(f"   test_start={raw_test_start}, test_end={raw_test_end}")
        print(f"   symbols={raw_symbols}")
        print(f"   data_path={raw_data_path}")

    # ── 漏斗统计初始化 (与 event_backtest.py 对齐) ──
    _funnel = {
        "total_signals_checked": 0,  # 所有 archetype × bar 的评估次数
        "reject_no_direction": 0,  # 无方向
        "reject_gate_deny": 0,  # gate 拒绝
        "reject_entry_filter_deny": 0,  # entry_filter 拒绝
        "signals_generated": 0,  # 有方向 + gate通过 + ef通过
    }

    for arch_name in list(arch_specs.keys()):
        logs_path = arch_specs[arch_name]

        # 加载配置
        try:
            exec_cfg = load_execution_config(arch_name, strategies_root)
            arch_exec_configs[arch_name] = exec_cfg
        except FileNotFoundError as e:
            print(f"❌ {arch_name}: {e}")
            return 1

        # ── 数据加载: from-raw vs 传统 ──
        if from_raw:
            # 从原始 1min 数据计算全量特征
            try:
                df, _raw_quantiles = _load_raw_features_for_archetype(
                    arch_name=arch_name,
                    strategies_root=strategies_root,
                    symbols=raw_symbols,
                    data_path=raw_data_path,
                    test_start=raw_test_start,
                    test_end=raw_test_end,
                )
                arch_precomputed_quantiles[arch_name] = _raw_quantiles
            except Exception as e:
                print(f"❌ {arch_name}: from-raw 加载失败: {e}")
                import traceback

                traceback.print_exc()
                return 1
        else:
            # 传统模式: 从 parquet 加载
            path = Path(logs_path)
            if not path.exists():
                print(f"❌ {arch_name}: file not found: {path}")
                return 1
            df = pd.read_parquet(path)
            if "_symbol" in df.columns and "symbol" not in df.columns:
                df["symbol"] = df["_symbol"]
            print(f"\n📂 {arch_name}: {len(df)} rows from {path}")

            # ── Holdout 时间过滤 (--logs 模式, 避免 in-sample 过拟合) ──
            if raw_test_start or raw_test_end:
                ts_col = None
                if "timestamp" in df.columns:
                    ts_col = "timestamp"
                elif df.index.name == "timestamp" or hasattr(df.index, "tz"):
                    df = df.reset_index()
                    ts_col = "timestamp" if "timestamp" in df.columns else None
                if ts_col:
                    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                    n_before = len(df)
                    if raw_test_start:
                        df = df[df[ts_col] >= pd.Timestamp(raw_test_start, tz="UTC")]
                    if raw_test_end:
                        df = df[df[ts_col] <= pd.Timestamp(raw_test_end, tz="UTC")]
                    print(
                        f"   🕐 Holdout filter: {n_before} → {len(df)} rows"
                        f" ({raw_test_start} ~ {raw_test_end})"
                    )

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

        # Prefilter 过滤 (在 gate 之前)
        _n_before_pf = int((df["entry_direction"] != 0).sum())
        _apply_prefilter_vectorized(df, arch_name, strategies_root)
        _n_after_pf = int((df["entry_direction"] != 0).sum())
        _funnel["reject_prefilter"] = (
            _funnel.get("reject_prefilter", 0) + _n_before_pf - _n_after_pf
        )

        # Gate 过滤
        # 收集 gate 前的统计
        _n_bars = len(df)
        _n_has_dir_before_gate = int((df["entry_direction"] != 0).sum())
        _funnel[
            "total_signals_checked"
        ] += _n_bars  # 每个 bar 对每个 archetype 评估一次
        _funnel["reject_no_direction"] += _n_bars - _n_has_dir_before_gate

        if recompute_gate:
            # 从 gate.yaml 重新评估 gate (不依赖 gate_decision 列)
            _apply_gate_from_yaml_vectorized(df, arch_name, strategies_root)
            veto_mask = df["gate_decision"] != "allow"
            df.loc[veto_mask, "entry_direction"] = 0.0
        elif "gate_decision" in df.columns:
            veto_mask = df["gate_decision"] != "allow"
            df.loc[veto_mask, "entry_direction"] = 0.0
            n_allowed = int((~veto_mask).sum())
            print(f"   🚪 Gate: {n_allowed} allow / {len(df)} total")
        elif "gate_ok" in df.columns:
            veto_mask = df["gate_ok"] != True  # noqa: E712
            df.loc[veto_mask, "entry_direction"] = 0.0
            n_allowed = int((~veto_mask).sum())
            print(f"   🚪 Gate: {n_allowed} allow / {len(df)} total")

        # 收集 gate 后的统计
        _n_has_dir_after_gate = int((df["entry_direction"] != 0).sum())
        _funnel["reject_gate_deny"] += _n_has_dir_before_gate - _n_has_dir_after_gate

        # Entry Filter
        _n_before_ef = _n_has_dir_after_gate
        if not args.no_entry_filter:
            ef_cfg = load_entry_filters_config(arch_name, strategies_root)
            if ef_cfg:
                compute_derived_entry_features(df)
                n_entries = apply_entry_filters_or(df, ef_cfg)
            else:
                print(f"   ℹ️  {arch_name}: entry_filters.yaml not found, skipping")
        else:
            print(f"   ℹ️  Entry filter disabled")

        # 收集 entry_filter 后的统计
        _n_after_ef = int((df["entry_direction"] != 0).sum())
        _funnel["reject_entry_filter_deny"] += _n_before_ef - _n_after_ef
        _funnel["signals_generated"] += _n_after_ef

        # Evidence 计算 — 使用 archetype evidence.yaml
        try:
            _archetype = load_strategy_archetype(arch_name, strategies_root)
            if _archetype.evidence and _archetype.evidence.features:
                df["evidence_score"] = _compute_evidence_for_archetype(
                    df, arch_name, _archetype, precomputed_quantiles
                )
                _ev_mean = df["evidence_score"].mean()
                print(
                    f"   📊 Evidence: mean={_ev_mean:.3f} ({len(_archetype.evidence.features)} features)"
                )
            else:
                df["evidence_score"] = 0.5
        except Exception as _ev_err:
            print(f"   ⚠️  Evidence fallback to 0.5: {_ev_err}")
            df["evidence_score"] = 0.5

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

    # ── 4. PCM 仲裁 (Regime-Aware, per-strategy 独立式 — 与事件回测 LivePCM 对齐) ──
    # 事件回测 LivePCM.decide() 对每个策略独立检查 slot，同 bar 可产生多个 intent。
    # 向量回测也保留所有 archetype 的信号（不再只选一个赢家），slot 竞争
    # 交给 simulate_rr_execution 后处理，与事件侧 per-strategy 独占 slot 一致。
    print(f"\n🏗️  PCM arbitration (Regime-Aware, per-strategy independent)...")
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

    # 收集需要扩展的行 (同 bar 多 archetype 信号)
    _expansion_rows: List[Dict[str, Any]] = (
        []
    )  # 额外行 (第一个 archetype 直接写入 merged)

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

        if len(active) > 1:
            n_conflicts += 1

        # 按优先级排序 (高优先级先处理, 与事件侧 PCM 注册顺序一致)
        def _sort_key(x):
            arch, d, ev = x
            rank = _pcm_get_priority_rank(arch, current_priority)
            return (rank, -(ev if ev is not None else 0.5))

        active.sort(key=_sort_key)

        # 第一个 archetype: 直接写入 merged 现有行
        first_arch, first_dir, first_ev = active[0]
        scale = regime_detector.get_archetype_scale(first_arch)
        regime_scales_applied.append(scale)

        merged.at[idx, "entry_direction"] = first_dir
        merged.at[idx, "evidence_score"] = first_ev if first_ev is not None else 0.5
        merged.at[idx, "_pcm_archetype"] = first_arch
        # Position scale = regime_scale × evidence_score
        _ev_scale = first_ev if first_ev is not None else 0.5
        merged.at[idx, "_position_scale"] = scale * (
            0.5 + _ev_scale
        )  # evidence 0→0.5x, 0.5→0.75x, 1→1.0x
        arch_win_counts[first_arch] = arch_win_counts.get(first_arch, 0) + 1
        regime_entry_counts[current_regime] = (
            regime_entry_counts.get(current_regime, 0) + 1
        )

        # 后续 archetype: 复制行作为新 entry
        for arch_name, d, ev in active[1:]:
            row_copy = merged.loc[idx].copy()
            sc = regime_detector.get_archetype_scale(arch_name)
            # Position scale = regime_scale × evidence_score
            row_copy["entry_direction"] = d
            row_copy["evidence_score"] = ev if ev is not None else 0.5
            row_copy["_pcm_archetype"] = arch_name
            _ev_sc = ev if ev is not None else 0.5
            row_copy["_position_scale"] = sc * (0.5 + _ev_sc)
            regime_scales_applied.append(sc)
            _expansion_rows.append(row_copy)
            arch_win_counts[arch_name] = arch_win_counts.get(arch_name, 0) + 1
            regime_entry_counts[current_regime] = (
                regime_entry_counts.get(current_regime, 0) + 1
            )

    # 扩展 merged DataFrame: 添加冲突 bar 的额外 archetype 行
    if _expansion_rows:
        extra_df = pd.DataFrame(_expansion_rows)
        merged = pd.concat([merged, extra_df], ignore_index=True)
        # 重新按 (symbol, timestamp) 排序确保时间线连续
        sort_key = (
            ["symbol", "timestamp"] if "timestamp" in merged.columns else ["symbol"]
        )
        merged = merged.sort_values(sort_key).reset_index(drop=True)
        print(f"   📐 Expanded: +{len(_expansion_rows)} rows for multi-archetype bars")

    n_total_entries = int((merged["entry_direction"] != 0).sum())
    print(f"   Total entries: {n_total_entries}")
    print(f"   Multi-archetype bars: {n_conflicts}")
    conflict_rate = n_conflicts / max(1, n_total_entries)
    print(f"   Multi-archetype rate: {conflict_rate:.2%}")
    for arch_name in arch_names:
        cnt = arch_win_counts.get(arch_name, 0)
        print(f"   {arch_name}: {cnt} entries")

    # ── 统一漏斗统计格式 (与 event_backtest.py 对齐) ──
    print(f"\n  信号漏斗:")
    print(f"    {'total_signals_checked':<30s}: {_funnel['total_signals_checked']}")
    print(f"    {'reject_no_direction':<30s}: {_funnel['reject_no_direction']}")
    print(f"    {'reject_gate_deny':<30s}: {_funnel['reject_gate_deny']}")
    print(
        f"    {'reject_entry_filter_deny':<30s}: {_funnel['reject_entry_filter_deny']}"
    )
    print(f"    {'signals_generated':<30s}: {_funnel['signals_generated']}")
    # Slot 过滤在执行层后处理, 这里先占位, 后面更新
    _funnel_pcm_before_slot = n_total_entries

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
    merged["_structural_exit"] = ""  # 空 = 无结构性退出, "ema200" = BPC trend_hold

    # 每行的 bar_minutes (用于 1min 模拟的 timeout 换算 + slot 时间戳比较)
    merged["_bar_minutes"] = 240  # 默认 4H
    for arch_name in arch_names:
        tf = load_meta_timeframe(arch_name, strategies_root)
        if tf:
            bm = int(tf.replace("T", ""))
            mask = merged["_pcm_archetype"] == arch_name
            merged.loc[mask, "_bar_minutes"] = bm

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
        # structural_exit: BPC trend_hold 用 ema200 结构性退出
        _se = sl.get("structural_exit", "")
        if _se:
            merged.loc[mask, "_structural_exit"] = str(_se)

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

    # 🔍 诊断: 检查 _bar_minutes per archetype
    if "_bar_minutes" in merged.columns and "_pcm_archetype" in merged.columns:
        _diag_entries = merged[merged["entry_direction"] != 0]
        for _da in sorted(_diag_entries["_pcm_archetype"].unique()):
            _sub = _diag_entries[_diag_entries["_pcm_archetype"] == _da]
            _bm_vals = _sub["_bar_minutes"].unique()
            _to_vals = (
                _sub["_tier_timeout"].unique()
                if "_tier_timeout" in _sub.columns
                else []
            )
            print(
                f"   🔍 DIAG {_da}: _bar_minutes={list(_bm_vals)}, _tier_timeout={list(_to_vals)}, entries={len(_sub)}"
            )

    # 加载 1min bar 数据（如果指定 --use-1min）
    bars_1min_dict = None
    if getattr(args, "use_1min", False):
        bars_1min_dict = _load_1min_bars(
            merged,
            data_path=getattr(args, "data_path", None),
            live_root=getattr(args, "live_root", "live/highcap"),
        )

    # ── 加仓配置 + per_strategy_limits (从 constitution.yaml 加载) ──
    _add_position_cfg = None
    _per_strategy_limits = None  # 独立于 add_position, 始终传入 slot 过滤
    if constitution_yaml:
        try:
            import yaml as _yaml_ap

            _c_ap = (
                _yaml_ap.safe_load(Path(constitution_yaml).read_text(encoding="utf-8"))
                or {}
            )
            _ra_ap = _c_ap.get("resource_allocation") or {}
            _ap_rules = _ra_ap.get("add_position_rules") or {}
            _ap_per_strat = _ra_ap.get("per_strategy_limits") or {}
            # per_strategy_limits 始终读取 (与事件侧 LivePCM._max_slots_for_strategy 对齐)
            if _ap_per_strat:
                _per_strategy_limits = _ap_per_strat
            if any(
                v.get("allow_add_position", False)
                for v in _ap_per_strat.values()
                if isinstance(v, dict)
            ):
                _add_position_cfg = {
                    "add_position_rules": _ap_rules,
                    "per_strategy_limits": _ap_per_strat,
                }
                _ap_strats = [
                    k
                    for k, v in _ap_per_strat.items()
                    if isinstance(v, dict) and v.get("allow_add_position", False)
                ]
                print(f"   📈 Add-position enabled for: {_ap_strats}")
        except Exception:
            pass

    print(f"\n📈 Simulating bar-by-bar with per-archetype execution params...")
    exec_returns, trade_details = simulate_rr_execution(
        merged,
        first_exec,  # 全局 fallback config
        atr_col="atr",
        use_tier_params=True,
        breakeven_lock_r=breakeven_lock_r,
        max_slots=max_slots,
        bars_1min_dict=bars_1min_dict,
        add_position_cfg=_add_position_cfg,
        per_strategy_limits=_per_strategy_limits,
    )

    valid_returns = exec_returns.dropna()
    if len(valid_returns) == 0:
        print("❌ No valid returns computed")
        return 1

    # Evidence 单调性验证
    _report_evidence_monotonicity(trade_details, label="PCM")

    # ── 应用 position_scale 仓位缩放 (regime × evidence) 到 R-multiples ──
    # _position_scale 在 PCM 仲裁时计算: regime_scale × evidence_factor
    # 与事件回测/实盘 LivePCM._apply_regime_scale() 一致
    if "_position_scale" in merged.columns:
        _has_scale = (merged["_position_scale"] < 1.0 - 1e-9).any()
        if _has_scale:
            exec_returns = exec_returns * merged["_position_scale"]
            valid_returns = exec_returns.dropna()
            _avg_scale = merged.loc[valid_returns.index, "_position_scale"].mean()
            print(f"   📐 Position scale applied to returns (avg={_avg_scale:.3f})")

    # ── 更新漏斗统计: slot 过滤后的数据 ──
    _n_slot_rejected = _funnel_pcm_before_slot - len(valid_returns)
    print(f"    {'reject_pcm_slot_full':<30s}: {_n_slot_rejected}")
    print(f"    {'trades_executed':<30s}: {len(valid_returns)}")

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
    # Kill switch 模拟参数 (从 constitution.yaml 加载)
    _ks_cfg = None
    if constitution_yaml:
        try:
            import yaml as _yaml

            _c_raw = (
                _yaml.safe_load(Path(constitution_yaml).read_text(encoding="utf-8"))
                or {}
            )
            _ks_raw = _c_raw.get("kill_switch") or {}
            if _ks_raw.get("enabled", False):
                _ks_cfg = {
                    "max_dd": float(_ks_raw.get("max_dd", 0.20)),
                    "daily_loss_limit": float(_ks_raw.get("daily_loss_limit", 0.04)),
                    "weekly_loss_limit": float(_ks_raw.get("weekly_loss_limit", 0.08)),
                    "monthly_loss_limit": float(
                        _ks_raw.get("monthly_loss_limit", 0.12)
                    ),
                    "cooldown_bars": int(_ks_raw.get("cooldown_minutes", 240))
                    // 240,  # 4H bars
                }
        except Exception:
            pass

    risk_eq_pcm = compute_risk_equity_curve(
        exec_returns,
        initial_cash=1000.0,
        risk_per_slot=risk_per_slot,
        stop_loss_r=pcm_sl_r,
        risk_per_trade_series=risk_series,
        kill_switch=_ks_cfg,
    )
    print(f"\n   💰 Risk-Based Equity ($1000, per-strategy risk, SL={pcm_sl_r}R):")
    print(
        f"      Final: ${risk_eq_pcm['final_equity']:.0f}  ({risk_eq_pcm['total_return_pct']:+.1f}%)"
    )
    print(f"      Max DD: {risk_eq_pcm['max_dd']:.1%}")

    # Kill switch 模拟统计
    if _ks_cfg and "kill_switch_stats" in risk_eq_pcm:
        ks_stats = risk_eq_pcm["kill_switch_stats"]
        print(f"\n   🚨 Kill Switch 模拟 (constitution.yaml):")
        print(f"      触发次数: {ks_stats['trigger_count']}")
        print(f"      跳过交易: {ks_stats['trades_skipped']}")
        print(f"      实际执行: {ks_stats['trades_executed']}")
        for trig in ks_stats["triggers"][:5]:  # 最多显示前5次
            print(
                f"      │ {trig['timestamp']}: {', '.join(trig['reasons'])} (eq=${trig['equity']:.0f}, dd={trig['dd']:.1%})"
            )
        if ks_stats["trigger_count"] > 5:
            print(f"      │ ... 另有 {ks_stats['trigger_count']-5} 次触发")

    # 加仓统计
    if _add_position_cfg and trade_details:
        ap_trades = [t for t in trade_details if t.get("is_add_position", False)]
        print(f"\n   📈 Add-Position 统计:")
        print(f"      加仓交易: {len(ap_trades)}")
        if ap_trades:
            ap_pnl = [t["realized_rr"] for t in ap_trades]
            print(f"      加仓平均R: {np.mean(ap_pnl):.4f}")
            print(f"      加仓胜率: {np.mean([p > 0 for p in ap_pnl]):.2%}")

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
        try:
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
        except Exception as e:
            print(f"   ⚠️  Trading Map 生成失败: {e}")

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

    # ── Export trades CSV (一致性对比用) ──
    if getattr(args, "export_trades", None) and trade_details:
        _export_trade_details_csv(trade_details, args.export_trades, merged)

    print("\n" + "=" * 80)
    return 0


# ================================================================
# Grid Search (imported by optimize_execution_grid.py)
# ================================================================


def _parse_range_str_vec(s: str) -> List[float]:
    """Parse 'start:step:end' → [start, start+step, ..., end] (for vector backtest CLI)"""
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


# KPI 验收门槛 (对称调优模式)
SYM_R_KPI = {
    "mean_r_min": 0.18,  # 覆盖实盐排耗：手续皅0.08R+滑点0.05R+延迟0.05R
    "sharpe_min": 0.05,  # 最低信号强度 (per-trade Sharpe)
    "trades_min": 100,  # 样本量门槛
}


def _print_sym_r_kpi_gate(
    plateau: Dict[str, Any],
    strategy: str,
) -> bool:
    """
    打印 KPI 验收结果，返回是否通过。

    Returns:
        True = 通过所有 KPI 门槛
        False = 未通过（不应更新配置）
    """
    rec = plateau.get("recommended", plateau.get("best", {}))
    best = plateau.get("best", {})
    param_analysis = plateau.get("param_analysis", {})

    mean_r = rec.get("mean_r", 0.0)
    sharpe = rec.get("sharpe", 0.0)
    trades = rec.get("trades", 0)
    sym_r_val = rec.get("sym_r", best.get("sym_r", None))

    # at_boundary 检查：任一参数卡在边界
    at_boundary = any(v.get("at_boundary", False) for v in param_analysis.values())

    print("\n" + "=" * 70)
    print(f"  KPI 验收门槛 [{strategy}] (向量回测 Sym-R Grid Search)")
    print("=" * 70)
    print(f"  {'KPI':<22} {'Value':>10}  {'Gate':>10}  {'Pass?':>6}")
    print(f"  {'-'*52}")

    all_pass = True
    checks = [
        ("mean_r", mean_r, SYM_R_KPI["mean_r_min"], ">=", "覆盖实盐排耗"),
        ("sharpe", sharpe, SYM_R_KPI["sharpe_min"], ">=", "per-trade"),
        ("n_trades", trades, SYM_R_KPI["trades_min"], ">=", "样本量"),
    ]
    for name, val, gate, op, note in checks:
        if op == ">=":
            ok = val >= gate
        else:
            ok = val <= gate
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:<22} {val:>10.4f}  {gate:>10.4f}  {status}  ({note})")
        if not ok:
            all_pass = False

    # 边界检查
    bd_status = "❌ FAIL (at boundary)" if at_boundary else "✅ PASS"
    print(f"  {'not_at_boundary':<22} {'---':>10}  {'---':>10}  {bd_status}")
    if at_boundary:
        all_pass = False

    print(f"\n  推荐参数: sym_r = {sym_r_val}")
    print(
        f"  is_plateau: {plateau.get('is_plateau', False)},  cv: {plateau.get('cv', 0):.4f}"
    )

    if all_pass:
        print("\n  ✅ 全部 KPI 通过 —— 可更新 execution.yaml 并进入事件回测验证。")
    else:
        print("\n  ❌ KPI 未全部通过 —— 信号质量不足，不建议更新配置。")
    print("=" * 70)
    return all_pass


def _run_sym_r_grid_search(
    args: Any,
    merged: pd.DataFrame,
    exec_config: Dict[str, Any],
    sym_r_str: str,
    span_years: float,
    n_symbols: int,
    bars_1min_dict: Optional[Dict[str, pd.DataFrame]],
) -> int:
    """
    对称 SL Grid Search 主流程:
      1. 解析 sym_r_str 为参数列表
      2. 将 stop_loss.type 设为 trailing
      3. 调用 run_grid_search，注入 __sym_r__ 三联动
      4. 打印 plateau 分析 + KPI 验收结果
      5. (可选) 导出 HTML 报告
    """
    print("\n" + "=" * 70)
    print("  对称 SL Grid Search (--sym-r 模式)")
    print("  设计: initial_r = activation_r = trail_r (三者相等)")
    print("=" * 70)

    # 解析参数范围
    try:
        sym_r_vals = _parse_range_str_vec(sym_r_str)
    except ValueError as e:
        print(f"❌ --sym-r 解析失败: {e}")
        return 1

    print(f"  参数范围: {sym_r_vals}")
    print(f"  组合数: {len(sym_r_vals)} combos")
    print(f"  (每组操作: initial_r=activation_r=trail_r=<value>)")
    print()

    # 导入 exec_config 并覆盖 trailing 模式
    import copy as _copy

    base_cfg = _copy.deepcopy(exec_config)
    _set_nested(base_cfg, "stop_loss.type", "trailing")

    # 构造参数网格: __sym_r__ 是内部标识，在 run_grid_search 中识别为三联动
    param_names = ["__sym_r__"]
    param_values = [sym_r_vals]

    # 运行 Grid Search
    t0 = time.time()
    grid_results = run_grid_search(
        df=merged,
        exec_config=base_cfg,
        param_names=param_names,
        param_values=param_values,
        atr_col="atr",
        span_years=span_years,
        n_symbols=n_symbols,
        bars_1min_dict=bars_1min_dict,
        sym_r_mode=True,
    )
    elapsed = time.time() - t0
    print(f"  完成 {len(grid_results)} combos, 耗时 {elapsed:.1f}s")

    if not grid_results:
        print("❌ Grid Search 无结果")
        return 1

    # Plateau 分析
    plateau = _identify_plateau(
        grid_results,
        param_names=["sym_r"],
        param_values=[sym_r_vals],
    )

    # 打印结果表
    print(
        f"\n  {'Rank':>5} {'sym_r':>8} {'Sharpe':>9} {'MeanR':>8} {'WinRate':>9} {'Trades':>8}"
    )
    print(f"  {'-'*52}")
    for i, r in enumerate(plateau["all_sorted"][:10], 1):
        sr = r.get("sym_r", "?")
        marker = " <-- best" if i == 1 else ""
        rec_val = plateau["recommended"].get("sym_r", None)
        if rec_val is not None and abs(float(sr) - float(rec_val)) < 1e-6 and i != 1:
            marker = " <-- recommended"
        print(
            f"  {i:>5} {sr:>8} {r['sharpe']:>9.4f} {r['mean_r']:>8.4f} "
            f"{r['win_rate']:>8.1%} {r['trades']:>8}"
            f"{marker}"
        )

    # KPI 验收
    passed = _print_sym_r_kpi_gate(
        plateau, strategy=str(getattr(args, "strategy", "?"))
    )

    # 导出 HTML
    html_out = getattr(args, "export_grid_html", None)
    if html_out:
        _param_names_display = ["sym_r"]
        _param_values_display = [sym_r_vals]
        html = _generate_grid_search_html(
            results=grid_results,
            param_names=_param_names_display,
            param_values=_param_values_display,
            plateau=plateau,
            exec_config=base_cfg,
            strategy=str(getattr(args, "strategy", "backtest")),
        )
        Path(html_out).parent.mkdir(parents=True, exist_ok=True)
        Path(html_out).write_text(html, encoding="utf-8")
        print(f"\n  📊 Grid Search HTML: {html_out}")

    return 0 if passed else 2


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
    bars_1min_dict: Optional[Dict[str, pd.DataFrame]] = None,
    sym_r_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    执行全量网格搜索

    Args:
        n_symbols: symbol 数量，年化时用 per-symbol 交易频率
                   trades_per_year = trades / n_symbols / span_years
        sym_r_mode: 对称模式，__sym_r__ 参数名会同时设置
                    initial_r / activation_r / trail_r 三者，结果字典中存为 'sym_r'

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
            if name == "__sym_r__":
                # 对称模式: 三者同时设置
                _set_nested(modified, "stop_loss.initial_r", val)
                _set_nested(modified, "stop_loss.trailing.activation_r", val)
                _set_nested(modified, "stop_loss.trailing.trail_r", val)
            else:
                _set_nested(modified, name, val)

        # 静默运行模拟（抑制 print 输出）
        with contextlib.redirect_stdout(io.StringIO()):
            returns, _ = simulate_rr_execution(
                df,
                modified,
                atr_col,
                silent=True,
                bars_1min_dict=bars_1min_dict,
            )
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
            # __sym_r__ 内部标识 → 结果字典中存为 'sym_r'
            result_key = "sym_r" if name == "__sym_r__" else name
            result[result_key] = val

        results.append(result)

        if idx % 10 == 0 or idx == total:
            print(f"   [{idx}/{total}] ...", end="\r")

    print()  # newline after progress
    return results


def _identify_plateau(
    results: List[Dict[str, Any]],
    top_frac: float = 0.25,
    cv_threshold: float = 0.15,
    param_names: Optional[List[str]] = None,
    param_values: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    识别参数平坦高原区域 + 逐参数边际分析选择保守参数。

    核心思路：对每个参数独立计算其“足够好”的最小值（elbow），
    然后从实际组合中找最接近这些 elbow 值的好组合。

    这解决了“参数和最小”方法的缺陷：
    当 initial_r 在所有好组合中都卡上限时，旧方法无法区分；
    新方法通过边际分析找到 initial_r 的 elbow，就能正确地选择更保守的值。

    Returns:
        plateau 分析结果，包含:
        - best: Sharpe 绝对最优组合
        - recommended: 逐参数 elbow 分析后的保守选择
        - param_analysis: 每个参数的边际分析结果
    """
    sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    top_n = max(3, int(len(sorted_results) * top_frac))
    top = sorted_results[:top_n]

    sharpe_values = [r["sharpe"] for r in top]
    mean_sharpe = np.mean(sharpe_values)
    std_sharpe = np.std(sharpe_values)
    cv = std_sharpe / mean_sharpe if mean_sharpe > 1e-8 else float("inf")

    is_plateau = cv < cv_threshold

    # ── 逐参数边际分析 (per-parameter marginal analysis) ──
    recommended = sorted_results[0]  # 默认 = best
    param_analysis = {}  # 每个参数的分析结果
    sufficient_values = {}  # 每个参数的 "elbow" 值

    if param_names and param_values:
        for pi, pname in enumerate(param_names):
            vals = sorted(set(param_values[pi]))
            # 计算每个值的平均 Sharpe（边际化其他参数）
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
            # “足够好”阈值 = 95% 的该参数维度最优均值
            suff_threshold = max_mean * 0.95

            # 找最小的值使得 mean_sharpe >= 95% max
            sufficient_val = sorted_vals[-1]  # 默认取最大
            for v in sorted_vals:
                if val_mean_sharpe[v] >= suff_threshold:
                    sufficient_val = v
                    break

            # 检测是否卡在搜索上限
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

        # 从实际组合中找最接近 sufficient_values 的好组合
        if sufficient_values:
            best_sharpe = sorted_results[0]["sharpe"]
            # 候选：Sharpe >= 85% best（稍宽松一点，因为 elbow 可能比 best 低）
            threshold = best_sharpe * 0.85
            eligible = [r for r in sorted_results if r["sharpe"] >= threshold]
            if eligible:
                # 按各参数与 sufficient_value 的偏差排序，选偏差最小的
                def _deviation(r):
                    return sum(
                        abs(r.get(p, 0) - sufficient_values.get(p, 0))
                        for p in param_names
                    )

                recommended = min(eligible, key=_deviation)

    return {
        "is_plateau": is_plateau,
        "top_n": top_n,
        "mean_sharpe": float(mean_sharpe),
        "std_sharpe": float(std_sharpe),
        "cv": float(cv),
        "best": sorted_results[0],
        "recommended": recommended,
        "param_analysis": param_analysis,
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
        default=None,
        help="Max concurrent slots for PCM mode (default: from constitution.yaml)",
    )
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument(
        "--features-store-layer",
        default=None,
        help="FeatureStore layer (auto-detect from strategy if omitted)",
    )
    p.add_argument("--timeframe", default="240T")
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
        "--export-trades",
        type=str,
        default=None,
        help="导出交易明细 CSV (symbol/side/entry_time/exit_time/pnl_r/exit_reason/archetype)，"
        "用于 compare_vector_event_consistency.py 与事件回测对比。",
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
    p.add_argument(
        "--use-1min",
        action="store_true",
        help="Use 1min bar data for precise SL/trailing simulation (matches live trading)",
    )
    p.add_argument(
        "--live-root",
        default="live/highcap",
        help="Live data root for 1min bars (fallback when --data-path=none)",
    )
    p.add_argument(
        "--data-path",
        default="data/parquet_data",
        help="研究数据目录 (默认 data/parquet_data, 设为 none 使用实盘数据验证)",
    )
    p.add_argument(
        "--from-raw",
        action="store_true",
        dest="from_raw",
        help="从原始 1min 数据计算全量特征 (不依赖 logs_gated.parquet)。"
        "此模式需要 --test-start 和 --test-end。--pcm 可用纯名称: --pcm bpc fer me",
    )
    p.add_argument(
        "--test-start",
        type=str,
        default=None,
        dest="test_start",
        help="回测开始日期 (YYYY-MM-DD)，--from-raw 模式必填",
    )
    p.add_argument(
        "--test-end",
        type=str,
        default=None,
        dest="test_end",
        help="回测结束日期 (YYYY-MM-DD)，--from-raw 模式必填",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="交易标的 (逗号分隔, 默认: BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT)",
    )
    p.add_argument(
        "--recompute-gate",
        action="store_true",
        dest="recompute_gate",
        help="从 gate.yaml 重新评估 gate (不使用 logs_gated 中的 gate_decision 列)",
    )
    p.add_argument(
        "--simple-execution",
        action="store_true",
        dest="simple_execution",
        help="使用中性简单执行模式 (SL=1.5R, TP=3R, 50bar timeout, 无 trailing/structural)。"
        "用于研究管线评估信号质量，不受 execution 参数影响。",
    )
    p.add_argument(
        "--simple-sl",
        type=float,
        default=None,
        dest="simple_sl",
        help="--simple-execution 的止损 R 倍数 (默认 1.5)。可在 research_pipeline.yaml simple_execution.sl_r 配置。",
    )
    p.add_argument(
        "--simple-tp",
        type=float,
        default=None,
        dest="simple_tp",
        help="--simple-execution 的止盈 R 倍数 (默认 3.0)。",
    )
    p.add_argument(
        "--simple-timeout",
        type=int,
        default=None,
        dest="simple_timeout",
        help="--simple-execution 的超时 bar 数 (默认 50)。",
    )
    p.add_argument(
        "--sym-r",
        default=None,
        dest="sym_r",
        help="对称 SL Grid Search: initial_r=activation_r=trail_r 三者联动。"
        "格式: start:step:end (e.g. 1.0:0.5:4.0)。"
        "自动启用 trailing 模式，跑完后打印 plateau 分析与 KPI 验收结果。",
    )
    p.add_argument(
        "--export-grid-html",
        default=None,
        dest="export_grid_html",
        help="Grid Search 结果导出为 HTML 报告路径 (e.g. /tmp/grid_me.html)。",
    )
    args = p.parse_args()

    # --data-path none → 显式使用实盘数据
    if args.data_path and args.data_path.lower() == "none":
        args.data_path = None

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
    except Exception as e:
        print(f"❌ Failed to load execution.yaml: {e}")
        return 1

    # --simple-execution: 覆盖为中性简单执行配置
    # 目的: 研究管线评估 Gate/Evidence/Entry Filter 信号质量
    #       不受 execution 参数 (trailing/structural/fat-tail) 影响
    if getattr(args, "simple_execution", False):
        _sl_r = getattr(args, "simple_sl", None) or 1.5
        _tp_r = getattr(args, "simple_tp", None) or 3.0
        _timeout = getattr(args, "simple_timeout", None) or 50
        exec_config = {
            "stop_loss": {
                "type": "fixed",
                "initial_r": _sl_r,
            },
            "take_profit": {
                "enabled": True,
                "target_r": _tp_r,
            },
            "holding": {
                "max_holding_bars": _timeout,
                "time_stop_bars": _timeout,
            },
        }
        print("\n📋 Simple execution mode (signal quality evaluation):")
        print(
            f"   Stop Loss: fixed {_sl_r}R | Take Profit: {_tp_r}R | Timeout: {_timeout} bars"
        )
    else:
        print(f"\n📋 Loaded execution.yaml for '{args.strategy}':")
        stop_loss = exec_config.get("stop_loss", {})
        print(f"   Stop Loss Type: {stop_loss.get('type', 'fixed')}")
        print(f"   Initial R: {stop_loss.get('initial_r', 2.0)}")
        if stop_loss.get("type") == "trailing":
            trailing = stop_loss.get("trailing", {})
            print(f"   Trailing Activation: {trailing.get('activation_r', 1.0)}R")
            print(f"   Trail Distance: {trailing.get('trail_r', 1.5)}R")

    # 读取 logs 文件
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"\n📂 Loaded logs: {len(df)} rows")

    # ── Holdout 时间过滤 (--test-start / --test-end, 与 PCM 模式一致) ──
    _ts_start = getattr(args, "test_start", None)
    _ts_end = getattr(args, "test_end", None)
    if _ts_start or _ts_end:
        _ts_col = None
        if "timestamp" in df.columns:
            _ts_col = "timestamp"
        elif df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            _ts_col = "timestamp" if "timestamp" in df.columns else None
        if _ts_col:
            df[_ts_col] = pd.to_datetime(df[_ts_col], utc=True)
            _n_before = len(df)
            if _ts_start:
                df = df[df[_ts_col] >= pd.Timestamp(_ts_start, tz="UTC")]
            if _ts_end:
                df = df[df[_ts_col] <= pd.Timestamp(_ts_end, tz="UTC")]
            print(
                f"   🕐 Time filter: {_n_before} → {len(df)} rows  "
                f"(start={_ts_start}, end={_ts_end})"
            )
        else:
            print(f"   ⚠️  --test-start/--test-end 指定但无 timestamp 列, 跳过过滤")

    # 处理列名兼容
    if "_symbol" in df.columns and "symbol" not in df.columns:
        df["symbol"] = df["_symbol"]

    # ── 设置 _pcm_archetype 列 (单策略模式, 使 slot per-strategy 匹配正确) ──
    if "_pcm_archetype" not in df.columns:
        df["_pcm_archetype"] = args.strategy.lower()
        print(f"   🏷️  Set _pcm_archetype='{args.strategy.lower()}' for slot matching")

    # 创建 entry_direction 列：标记入场信号
    # fixed_direction 优先：无论是否已有 entry_direction 列，都强制覆盖
    dir_cfg = load_direction_config(args.strategy, args.strategies_root)
    _fd = dir_cfg.get("fixed_direction", None) if dir_cfg else None
    if _fd in ("long", "short"):
        _dir_val = 1.0 if _fd == "long" else -1.0
        df["entry_direction"] = _dir_val
        print(
            f"   📍 Direction: fixed_direction={_fd} → ALL {len(df)} bars = {_fd.upper()}"
        )
    elif "entry_direction" in df.columns:
        print(f"   📍 Using existing entry_direction column")
        # 仍然检查 direction_filter
        if dir_cfg:
            _df_filter = dir_cfg.get("direction_filter", None)
            if _df_filter == "long":
                n_before = int((df["entry_direction"] != 0).sum())
                df.loc[df["entry_direction"] < 0, "entry_direction"] = 0.0
                n_after = int((df["entry_direction"] != 0).sum())
                print(
                    f"     direction_filter=long: {n_before} → {n_after} "
                    f"(过滤掉 {n_before - n_after} SHORT)"
                )
            elif _df_filter == "short":
                n_before = int((df["entry_direction"] != 0).sum())
                df.loc[df["entry_direction"] > 0, "entry_direction"] = 0.0
                n_after = int((df["entry_direction"] != 0).sum())
                print(
                    f"     direction_filter=short: {n_before} → {n_after} "
                    f"(过滤掉 {n_before - n_after} LONG)"
                )
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

    # Prefilter 过滤 (在 gate 之前)
    _n_before_pf = int((df["entry_direction"] != 0).sum())
    _apply_prefilter_vectorized(df, args.strategy, args.strategies_root)
    _n_after_pf = int((df["entry_direction"] != 0).sum())
    if _n_before_pf > _n_after_pf:
        print(
            f"   🛡️  Prefilter: {_n_before_pf} → {_n_after_pf} "
            f"(rejected {_n_before_pf - _n_after_pf})"
        )

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
    else:
        # 无 gate 列 → 可能用了错误的输入文件 (predictions.parquet 而非 logs_gated.parquet)
        print(
            "   ⚠️  无 gate_decision/gate_ok 列 — Gate 未生效!"
            " 应使用 logs_gated.parquet (见 research_pipeline.yaml data_flow)"
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
    # Noise Penalty: 调整 per-entry 参数
    # ================================================================
    use_tier_params = False
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
            # 先创建 per-entry 参数列，再调整
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
            merged["_tier_initial_r"] = merged["_tier_initial_r"] * (1 + 0.5 * np_arr)
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

    # 加载 1min bar 数据（如果指定 --use-1min）
    bars_1min_dict = None
    if getattr(args, "use_1min", False):
        bars_1min_dict = _load_1min_bars(
            merged,
            data_path=getattr(args, "data_path", None),
            live_root=getattr(args, "live_root", "live/highcap"),
        )

    # ── 从 constitution.yaml 读取 per_strategy max_slots ──
    from src.time_series_model.portfolio.live_pcm import (
        _load_constitution_constraints as _load_const_pre,
    )

    _const_yaml_single = getattr(args, "constitution", None)
    if not _const_yaml_single:
        # 自动发现 constitution.yaml（与 PCM 模式一致）
        _const_yaml_single = "config/constitution/constitution.yaml"
        if not Path(_const_yaml_single).exists():
            _const_yaml_single = None
    _const_pre = _load_const_pre(_const_yaml_single)
    _per_strat_limits = _const_pre.get("per_strategy_limits") or {}
    _strat_cfg = _per_strat_limits.get(str(args.strategy).lower()) or {}
    _max_slots_single = int(_strat_cfg.get("max_slots", 1))
    # --simple-execution: 无槽位限制 — 研究管线评估纯信号质量，不做容量管理
    if getattr(args, "simple_execution", False):
        _max_slots_single = 0
        print("   ℹ️  --simple-execution: max_slots=0 (unlimited, 评估纯信号质量)")
    else:
        print(
            f"   🔒 Single-strategy max_slots={_max_slots_single} (from constitution: {_const_yaml_single or 'defaults'})"
        )

    # ── 加仓配置 (从 constitution.yaml 加载, 与事件回测一致) ──
    _add_pos_cfg_single = None
    _per_strategy_limits_single = None
    if _const_yaml_single:
        try:
            import yaml as _yaml_const

            _c_const = (
                _yaml_const.safe_load(
                    Path(_const_yaml_single).read_text(encoding="utf-8")
                )
                or {}
            )
            _ra_const = _c_const.get("resource_allocation") or {}
            _ap_rules_const = _ra_const.get("add_position_rules") or {}
            _ap_per_strat_const = _ra_const.get("per_strategy_limits") or {}
            # per_strategy_limits 始终传入 (slot 过滤需要)
            _per_strategy_limits_single = _ap_per_strat_const
            if any(
                v.get("allow_add_position", False)
                for v in _ap_per_strat_const.values()
                if isinstance(v, dict)
            ):
                _add_pos_cfg_single = {
                    "add_position_rules": _ap_rules_const,
                    "per_strategy_limits": _ap_per_strat_const,
                }
        except Exception:
            pass

    # ================================================================
    # --sym-r Grid Search 模式：对称 SL 参数优化
    # ================================================================
    _sym_r_str = getattr(args, "sym_r", None)
    if _sym_r_str:
        return _run_sym_r_grid_search(
            args=args,
            merged=merged,
            exec_config=exec_config,
            sym_r_str=_sym_r_str,
            span_years=_estimate_span_years(merged),
            n_symbols=merged["symbol"].nunique() if "symbol" in merged.columns else 1,
            bars_1min_dict=bars_1min_dict,
        )

    # 使用 execution.yaml 配置模拟 RR
    print("\n📈 Simulating with execution.yaml config...")
    exec_returns, trade_details = simulate_rr_execution(
        merged,
        exec_config,
        atr_col="atr",
        use_tier_params=use_tier_params,
        breakeven_lock_r=breakeven_lock_r,
        max_slots=_max_slots_single,
        bars_1min_dict=bars_1min_dict,
        add_position_cfg=_add_pos_cfg_single,
        per_strategy_limits=_per_strategy_limits_single,
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
    # Kill switch 模拟 (单策略模式)
    _ks_single = None
    if _const_yaml:
        try:
            import yaml as _y

            _c_r = _y.safe_load(Path(_const_yaml).read_text(encoding="utf-8")) or {}
            _ks_r = _c_r.get("kill_switch") or {}
            if _ks_r.get("enabled", False):
                _ks_single = {
                    "max_dd": float(_ks_r.get("max_dd", 0.20)),
                    "daily_loss_limit": float(_ks_r.get("daily_loss_limit", 0.04)),
                    "weekly_loss_limit": float(_ks_r.get("weekly_loss_limit", 0.08)),
                    "monthly_loss_limit": float(_ks_r.get("monthly_loss_limit", 0.12)),
                    "cooldown_bars": int(_ks_r.get("cooldown_minutes", 240)) // 240,
                }
        except Exception:
            pass
    risk_eq = compute_risk_equity_curve(
        exec_returns,
        initial_cash=1000.0,
        risk_per_slot=effective_risk,
        stop_loss_r=sl_r,
        kill_switch=_ks_single,
    )
    print(
        f"\n   💰 Risk-Based Equity ($1000, {effective_risk:.1%}/trade [{args.strategy}], SL={sl_r}R):"
    )
    print(
        f"      Final: ${risk_eq['final_equity']:.0f}  ({risk_eq['total_return_pct']:+.1f}%)"
    )
    print(f"      Max DD: {risk_eq['max_dd']:.1%}")
    if _ks_single and "kill_switch_stats" in risk_eq:
        ks = risk_eq["kill_switch_stats"]
        print(
            f"\n   🚨 Kill Switch: {ks['trigger_count']} triggers, {ks['trades_skipped']} skipped, {ks['trades_executed']} executed"
        )
        for trig in ks["triggers"][:3]:
            print(f"      │ {trig['timestamp']}: {', '.join(trig['reasons'])}")

    # 加仓统计 (单策略模式)
    if _add_pos_cfg_single and trade_details:
        ap_trades = [t for t in trade_details if t.get("is_add_position", False)]
        print(f"\n   📈 Add-Position 统计:")
        print(f"      加仓交易: {len(ap_trades)}")
        if ap_trades:
            ap_pnl = [t["realized_rr"] for t in ap_trades]
            print(f"      加仓平均R: {np.mean(ap_pnl):.4f}")
            print(f"      加仓胜率: {np.mean([p > 0 for p in ap_pnl]):.2%}")

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
        # 优先用 meta.yaml 的 timeframe (auto_tf), 避免 CLI 默认 "240T" 覆盖 me/lv 等 60T/15T 策略
        map_tf = auto_tf or getattr(args, "timeframe", None) or "240T"
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

    # ── Export trades CSV (一致性对比用) ──
    if getattr(args, "export_trades", None) and trade_details:
        _export_trade_details_csv(trade_details, args.export_trades, merged)

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
