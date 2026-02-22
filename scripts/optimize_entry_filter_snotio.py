#!/usr/bin/env python3
"""
Entry Filter 组合搜索 — 以 snotio 为主 KPI

snotio (Signal-to-Noise of Trade I/O):
  = mean(R-multiples)
  = 平均每笔交易的风险调整收益

为什么用 snotio 而不是 Sharpe?
  1. Sharpe 被 sqrt(trades) 人为抬高, 在 trades 极多 + execution 很强的系统中失真
  2. Entry Filter 的目标是"避免低性价比交易", 不是"提高长期收益波动比"
  3. snotio 不受 trade count 影响, 只有 per-trade 质量提升才会改善

层级 KPI 定位:
  Entry Filter → snotio / worst-10%
  Evidence     → failure rate / drawdown contribution
  Execution    → Sharpe / Calmar / MDD
  全系统        → OOS Sharpe (最后看)

用法:
  python scripts/optimize_entry_filter_snotio.py \\
    --logs results/.../predictions.parquet \\
    --strategy bpc

输出:
  - 终端: Top 30 组合 + Best-per-N 表
  - HTML: entry_filter_snotio_combo.html (含 snotio vs N 图、worst-10% 分析)
"""

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.time_series_model.execution.entry_filter import (
    _build_mask_from_conditions,
    compute_derived_entry_features,
    load_entry_filters_config,
)
from scripts.backtest_execution_layer import (
    _estimate_span_years,
    apply_direction_rules,
    compute_sharpe,
    load_direction_config,
    load_execution_config,
    simulate_rr_execution,
)


# ================================================================
# snotio metrics
# ================================================================


def compute_snotio(returns: pd.Series) -> float:
    """snotio = mean(R-multiples), 每笔交易的平均风险调整收益"""
    valid = returns.dropna()
    if len(valid) < 1:
        return 0.0
    return float(valid.mean())


def compute_worst_pct(returns: pd.Series, pct: float = 10.0) -> float:
    """worst N% trades 的平均 R-multiple (越不负越好)"""
    valid = returns.dropna()
    if len(valid) < 2:
        return 0.0
    n = max(1, int(len(valid) * pct / 100.0))
    worst = valid.nsmallest(n)
    return float(worst.mean())


def compute_mae_per_risk(returns: pd.Series) -> float:
    """负收益交易的平均损失 (衡量每笔 bad trade 的伤害)"""
    valid = returns.dropna()
    losses = valid[valid < 0]
    if len(losses) < 1:
        return 0.0
    return float(losses.mean())


def compute_loss_rate(returns: pd.Series) -> float:
    """loss_rate = 亏损交易占比 (R < 0)"""
    valid = returns.dropna()
    if len(valid) < 1:
        return 0.0
    return float((valid < 0).mean())


def compute_stop_rate(
    returns: pd.Series, sl_r: float = 2.0, eps: float = 0.01
) -> float:
    """stop_rate = 触发止损的交易占比 (R <= -SL + eps)

    当 SL=2.0R 时, 止损交易的 R ≈ -2.0。
    用 eps=0.01 容忍浮点误差。

    高 stop_rate 意味着 filter 未能过滤掉"注定被止损"的交易。
    好的 Entry Filter 应该同时降低 stop_rate (少踩雷) 和提升 snotio (多挑好的)。
    """
    valid = returns.dropna()
    if len(valid) < 1:
        return 0.0
    return float((valid <= -(sl_r - eps)).mean())


# ================================================================
# Feature Scan (--scan mode)
# ================================================================

# 系统列 / 标签列 / 方向列 — 不应作为 entry filter 候选
_SCAN_EXCLUDE_PREFIXES = ("ef_",)  # 衍生 entry filter 特征
_SCAN_EXCLUDE_COLS = {
    # index / meta
    "symbol",
    "_symbol",
    "timestamp",
    "date",
    "datetime",
    "index",
    # target / label
    "forward_rr",
    "label",
    "target",
    "forward_return",
    # direction
    "entry_direction",
    "bpc_breakout_direction",
    "me_delta_net_flow",
    # gate
    "gate_decision",
    "gate_score",
    "gate_allow",
    # OHLCV / price
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr",
    # evidence
    "evidence_score",
    "position_size",
}


def _load_raw_scale_columns(
    config_path: str = "config/feature_dependencies.yaml",
) -> Set[str]:
    """Load raw_scale_columns from feature_dependencies.yaml.

    Returns flat set of column names that should be excluded from
    cross-asset scans (unnormalized price/flow/energy columns).
    """
    p = Path(config_path)
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("raw_scale_columns", {})
    cols: Set[str] = set()
    for category_list in raw.values():
        if isinstance(category_list, list):
            cols.update(str(c) for c in category_list)
    return cols


def run_feature_scan(
    merged: pd.DataFrame,
    orig_dir: pd.Series,
    exec_config: Dict[str, Any],
    span_years: float,
    min_samples: int = 200,
    silent: bool = False,
) -> List[Dict[str, Any]]:
    """全特征自动扫描: 每个数值列 × 多阈值 → execution 模拟 → snotio 排序

    对每个数值特征:
      - 高阈值: P70, P80, P90 (>=)
      - 低阈值: P10, P20, P30 (<=)
    共 6 个切点, 过滤后运行 simulate_rr_execution 计算 snotio。
    """
    # 筛选数值列
    raw_scale = _load_raw_scale_columns()
    if not silent and raw_scale:
        print(
            f"   Excluding {len(raw_scale)} raw-scale columns from feature_dependencies.yaml"
        )
    numeric_cols = []
    for c in merged.columns:
        if c in _SCAN_EXCLUDE_COLS:
            continue
        if c in raw_scale:
            continue
        if any(c.startswith(p) for p in _SCAN_EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(merged[c]):
            vals = pd.to_numeric(merged[c], errors="coerce")
            if vals.isna().sum() > len(merged) * 0.5:
                continue
            # 跳过常量列
            if vals.nunique() < 3:
                continue
            numeric_cols.append(c)

    pct_high = [0.70, 0.80, 0.90]
    pct_low = [0.10, 0.20, 0.30]
    total_evals = len(numeric_cols) * (len(pct_high) + len(pct_low))

    if not silent:
        print(
            f"   Numeric features: {len(numeric_cols)}, thresholds: {len(pct_high)+len(pct_low)} per feature"
        )
        print(f"   Total evaluations: {total_evals} (est. ~{total_evals * 0.15:.0f}s)")

    orig_merged = merged.copy()
    results = []
    done = 0

    for col in numeric_cols:
        vals = pd.to_numeric(merged[col], errors="coerce")

        # --- high thresholds (>=) ---
        for pct in pct_high:
            done += 1
            threshold = float(vals.quantile(pct))
            mask = vals >= threshold
            n_pass = int(mask.sum())
            if n_pass < min_samples or n_pass > len(merged) * 0.90:
                continue

            test_df = orig_merged.copy()
            test_df["entry_direction"] = orig_dir.copy()
            test_df.loc[~mask, "entry_direction"] = 0.0
            if int((test_df["entry_direction"] != 0).sum()) < 20:
                continue

            rr, _ = simulate_rr_execution(
                test_df, exec_config, atr_col="atr", silent=True
            )
            valid = rr.dropna()
            if len(valid) < 10:
                continue

            results.append(
                {
                    "feature": col,
                    "operator": ">=",
                    "threshold": round(threshold, 6),
                    "percentile": f"P{int(pct*100)}",
                    "n": 1,
                    "filters": f"{col}>={threshold:.4f}(P{int(pct*100)})",
                    "trades": len(valid),
                    "snotio": round(compute_snotio(valid), 4),
                    "worst_10": round(compute_worst_pct(valid, 10.0), 4),
                    "worst_5": round(compute_worst_pct(valid, 5.0), 4),
                    "mae_risk": round(compute_mae_per_risk(valid), 4),
                    "loss_rate": round(compute_loss_rate(valid), 4),
                    "stop_rate": round(compute_stop_rate(valid), 4),
                    "sharpe_pt": round(compute_sharpe(valid, annualize=False), 4),
                    "win_rate": round(float((valid > 0).mean()), 4),
                }
            )

        # --- low thresholds (<=) ---
        for pct in pct_low:
            done += 1
            threshold = float(vals.quantile(pct))
            mask = vals <= threshold
            n_pass = int(mask.sum())
            if n_pass < min_samples or n_pass > len(merged) * 0.90:
                continue

            test_df = orig_merged.copy()
            test_df["entry_direction"] = orig_dir.copy()
            test_df.loc[~mask, "entry_direction"] = 0.0
            if int((test_df["entry_direction"] != 0).sum()) < 20:
                continue

            rr, _ = simulate_rr_execution(
                test_df, exec_config, atr_col="atr", silent=True
            )
            valid = rr.dropna()
            if len(valid) < 10:
                continue

            results.append(
                {
                    "feature": col,
                    "operator": "<=",
                    "threshold": round(threshold, 6),
                    "percentile": f"P{int(pct*100)}",
                    "n": 1,
                    "filters": f"{col}<={threshold:.4f}(P{int(pct*100)})",
                    "trades": len(valid),
                    "snotio": round(compute_snotio(valid), 4),
                    "worst_10": round(compute_worst_pct(valid, 10.0), 4),
                    "worst_5": round(compute_worst_pct(valid, 5.0), 4),
                    "mae_risk": round(compute_mae_per_risk(valid), 4),
                    "loss_rate": round(compute_loss_rate(valid), 4),
                    "stop_rate": round(compute_stop_rate(valid), 4),
                    "sharpe_pt": round(compute_sharpe(valid, annualize=False), 4),
                    "win_rate": round(float((valid > 0).mean()), 4),
                }
            )

        # progress
        if not silent and done % 300 == 0:
            print(f"   ... {done}/{total_evals} evaluated, {len(results)} valid so far")

    if not silent:
        print(
            f"   Scan complete: {done}/{total_evals} evaluated, {len(results)} valid results"
        )

    results.sort(key=lambda x: -x["snotio"])
    return results


# ================================================================
# Main logic
# ================================================================


def run_combo_search(
    merged: pd.DataFrame,
    orig_dir: pd.Series,
    exec_config: Dict[str, Any],
    entry_cfg: Dict[str, Any],
    span_years: float,
    silent: bool = False,
    max_n: int = 0,
) -> List[Dict[str, Any]]:
    """穷举 2^N - 1 种 filter 子集组合

    max_n: 最大组合大小 (0 = 不限制)。--all 模式下建议设为 1 避免 2^16 爆炸。
    """

    filters_list = entry_cfg.get("filters", [])
    enabled = [f for f in filters_list if f.get("enabled", False)]
    filter_ids = [f["id"] for f in enabled]

    if not silent:
        print(f"   Enabled filters: {len(enabled)}, span={span_years:.2f}yr")

    # 预计算每个 filter 的 mask
    filter_masks = {}
    for filt in enabled:
        conditions = filt.get("conditions", [])
        if conditions:
            filter_masks[filt["id"]] = _build_mask_from_conditions(
                merged, conditions, silent=True
            )

    results = []
    orig_merged = merged.copy()
    total = 2 ** len(enabled) - 1
    max_r = max_n if max_n > 0 else len(enabled)

    for r in range(1, max_r + 1):
        for combo in itertools.combinations(range(len(enabled)), r):
            combo_ids = [filter_ids[i] for i in combo]

            # OR mask
            or_mask = pd.Series(False, index=merged.index)
            for idx in combo:
                fid = filter_ids[idx]
                if fid in filter_masks:
                    or_mask = or_mask | filter_masks[fid]

            # apply filter
            test_df = orig_merged.copy()
            test_df["entry_direction"] = orig_dir.copy()
            test_df.loc[~or_mask, "entry_direction"] = 0.0
            n_entries = int((test_df["entry_direction"] != 0).sum())

            if n_entries < 20:
                continue

            rr, _ = simulate_rr_execution(
                test_df, exec_config, atr_col="atr", silent=True
            )
            valid = rr.dropna()
            if len(valid) < 10:
                continue

            snotio = compute_snotio(valid)
            worst10 = compute_worst_pct(valid, 10.0)
            worst5 = compute_worst_pct(valid, 5.0)
            mae_risk = compute_mae_per_risk(valid)
            loss_rate = compute_loss_rate(valid)
            stop_rate = compute_stop_rate(valid)
            sh_pt = compute_sharpe(valid, annualize=False)
            win_rate = float((valid > 0).mean())

            results.append(
                {
                    "n": r,
                    "filters": ",".join(combo_ids),
                    "trades": len(valid),
                    "snotio": round(snotio, 4),
                    "worst_10": round(worst10, 4),
                    "worst_5": round(worst5, 4),
                    "mae_risk": round(mae_risk, 4),
                    "loss_rate": round(loss_rate, 4),
                    "stop_rate": round(stop_rate, 4),
                    "sharpe_pt": round(sh_pt, 4),
                    "win_rate": round(win_rate, 4),
                }
            )

    if not silent:
        total_actual = sum(
            1
            for r in range(1, max_r + 1)
            for _ in itertools.combinations(range(len(enabled)), r)
        )
        print(
            f"   Evaluated {len(results)}/{total_actual} valid combos (max_n={max_r})"
        )

    return results


def compute_baseline(
    merged: pd.DataFrame,
    orig_dir: pd.Series,
    exec_config: Dict[str, Any],
) -> Dict[str, Any]:
    """baseline: 无 entry filter"""
    test_df = merged.copy()
    test_df["entry_direction"] = orig_dir.copy()
    rr, _ = simulate_rr_execution(test_df, exec_config, atr_col="atr", silent=True)
    valid = rr.dropna()
    return {
        "n": 0,
        "filters": "none",
        "trades": len(valid),
        "snotio": round(compute_snotio(valid), 4),
        "worst_10": round(compute_worst_pct(valid, 10.0), 4),
        "worst_5": round(compute_worst_pct(valid, 5.0), 4),
        "mae_risk": round(compute_mae_per_risk(valid), 4),
        "loss_rate": round(compute_loss_rate(valid), 4),
        "stop_rate": round(compute_stop_rate(valid), 4),
        "sharpe_pt": round(compute_sharpe(valid, annualize=False), 4),
        "win_rate": round(float((valid > 0).mean()), 4),
    }


def get_best_per_n(results: List[Dict], key: str = "snotio") -> List[Dict]:
    """每个 N 取最优"""
    best = {}
    for r in sorted(results, key=lambda x: -x[key]):
        if r["n"] not in best:
            best[r["n"]] = r
    return [best[k] for k in sorted(best.keys())]


def run_marginal_analysis(
    merged: pd.DataFrame,
    orig_dir: pd.Series,
    exec_config: Dict[str, Any],
    entry_cfg: Dict[str, Any],
    baseline: Dict[str, Any],
    silent: bool = False,
) -> List[Dict[str, Any]]:
    """边际贡献分析: 每个 filter 独有的交易 (其他所有 filter 都没触发的)

    对每个 filter A:
      marginal_A = A 触发 AND NOT (其他任一 filter 触发)
      计算 marginal_A 的 snotio/loss_rate/stop_rate

    用途:
      - 如果 marginal snotio ≈ baseline → 该 filter 没有独立过滤能力
      - 如果 marginal stop_rate ≈ baseline → 该 filter 的独有交易和随机交易无异
      → 只有 marginal 明显优于 baseline 的 filter 才值得启用
    """
    filters_list = entry_cfg.get("filters", [])
    all_filters = [f for f in filters_list if f.get("conditions")]

    # 预计算所有 filter 的 mask (不限 enabled)
    filter_masks = {}
    for filt in all_filters:
        conditions = filt.get("conditions", [])
        if conditions:
            filter_masks[filt["id"]] = _build_mask_from_conditions(
                merged, conditions, silent=True
            )

    filter_ids = list(filter_masks.keys())
    if not silent:
        print(f"   Analyzing {len(filter_ids)} filters for marginal contribution")

    orig_merged = merged.copy()
    results = []

    for fid in filter_ids:
        this_mask = filter_masks[fid]

        # OR of all OTHER filters
        other_mask = pd.Series(False, index=merged.index)
        for other_id in filter_ids:
            if other_id != fid and other_id in filter_masks:
                other_mask = other_mask | filter_masks[other_id]

        # 这个 filter 触发的交易
        test_this = orig_merged.copy()
        test_this["entry_direction"] = orig_dir.copy()
        test_this.loc[~this_mask, "entry_direction"] = 0.0
        rr_this, _ = simulate_rr_execution(
            test_this, exec_config, atr_col="atr", silent=True
        )
        valid_this = rr_this.dropna()

        # 其他所有 filter 触发的交易
        test_other = orig_merged.copy()
        test_other["entry_direction"] = orig_dir.copy()
        test_other.loc[~other_mask, "entry_direction"] = 0.0
        rr_other, _ = simulate_rr_execution(
            test_other, exec_config, atr_col="atr", silent=True
        )
        valid_other = rr_other.dropna()

        # marginal = 这个 filter 触发但其他 filter 都没触发的交易
        # 用 mask 差集找 marginal 交易的位置
        marginal_mask = this_mask & ~other_mask
        test_marginal = orig_merged.copy()
        test_marginal["entry_direction"] = orig_dir.copy()
        test_marginal.loc[~marginal_mask, "entry_direction"] = 0.0
        rr_marginal, _ = simulate_rr_execution(
            test_marginal, exec_config, atr_col="atr", silent=True
        )
        valid_marginal = rr_marginal.dropna()

        n_this = len(valid_this)
        n_marginal = len(valid_marginal)
        n_overlap = n_this - n_marginal

        # 计算 this filter 整体指标
        if n_this >= 5:
            this_stats = {
                "snotio": round(compute_snotio(valid_this), 4),
                "loss_rate": round(compute_loss_rate(valid_this), 4),
                "stop_rate": round(compute_stop_rate(valid_this), 4),
                "win_rate": round(float((valid_this > 0).mean()), 4),
                "trades": n_this,
            }
        else:
            this_stats = {
                "snotio": 0,
                "loss_rate": 0,
                "stop_rate": 0,
                "win_rate": 0,
                "trades": n_this,
            }

        # 计算 marginal 指标
        if n_marginal >= 5:
            marginal_stats = {
                "snotio": round(compute_snotio(valid_marginal), 4),
                "loss_rate": round(compute_loss_rate(valid_marginal), 4),
                "stop_rate": round(compute_stop_rate(valid_marginal), 4),
                "win_rate": round(float((valid_marginal > 0).mean()), 4),
                "trades": n_marginal,
            }
        else:
            marginal_stats = {
                "snotio": 0,
                "loss_rate": 0,
                "stop_rate": 0,
                "win_rate": 0,
                "trades": n_marginal,
            }

        # 判定
        bl_snotio = baseline["snotio"]
        bl_stop = baseline["stop_rate"]
        m_snotio = marginal_stats["snotio"]
        m_stop = marginal_stats["stop_rate"]

        if n_marginal < 5:
            verdict = "样本不足"
        elif m_snotio > bl_snotio * 1.05 and m_stop < bl_stop - 0.03:
            verdict = "✅ 真正好 Entry"
        elif m_snotio > bl_snotio and m_stop < bl_stop:
            verdict = "✅ 较好"
        elif abs(m_stop - bl_stop) < 0.03:
            verdict = "❌ ≈ baseline, 无过滤效果"
        elif m_snotio < bl_snotio * 0.9:
            verdict = "❌ snotio 差"
        else:
            verdict = "⚠️ 边际"

        results.append(
            {
                "filter_id": fid,
                "this": this_stats,
                "marginal": marginal_stats,
                "overlap": n_overlap,
                "verdict": verdict,
            }
        )

    # 按 marginal snotio 降序
    results.sort(key=lambda x: -x["marginal"]["snotio"])
    return results


# ================================================================
# HTML report
# ================================================================


def generate_html_report(
    results: List[Dict],
    baseline: Dict,
    best_per_n: List[Dict],
    span_years: float,
    strategy: str,
    output_path: str,
    marginal: List[Dict] = None,
):
    """生成 snotio combo search HTML 报告"""

    # Sort by snotio
    sorted_results = sorted(results, key=lambda x: -x["snotio"])
    best = sorted_results[0] if sorted_results else baseline

    # snotio vs N chart data
    chart_ns = [r["n"] for r in best_per_n]
    chart_snotio = [r["snotio"] for r in best_per_n]
    chart_worst10 = [r["worst_10"] for r in best_per_n]
    chart_trades = [r["trades"] for r in best_per_n]
    chart_loss_rate = [r["loss_rate"] for r in best_per_n]
    chart_stop_rate = [r["stop_rate"] for r in best_per_n]

    # Top 30 table rows
    rows_html = ""
    for rank, r in enumerate(sorted_results[:30], 1):
        is_best = rank == 1
        row_class = ' style="background:#d4edda;font-weight:bold;"' if is_best else ""
        badge = " ⭐" if is_best else ""
        # Short filter names
        short = (
            r["filters"]
            .replace("deep_pullback_", "dp_")
            .replace("deep_pullback", "dp_base")
        )
        rows_html += f"""<tr{row_class}>
            <td>{rank}{badge}</td>
            <td>{r['n']}</td>
            <td><strong>{r['snotio']:.4f}</strong></td>
            <td>{r['loss_rate']:.1%}</td>
            <td>{r['stop_rate']:.1%}</td>
            <td>{r['worst_10']:.4f}</td>
            <td>{r['worst_5']:.4f}</td>
            <td>{r['mae_risk']:.4f}</td>
            <td>{r['sharpe_pt']:.4f}</td>
            <td>{r['win_rate']:.1%}</td>
            <td>{r['trades']}</td>
            <td title="{r['filters']}"><code>{short}</code></td>
        </tr>\n"""

    # Best-per-N table
    bpn_rows = ""
    for r in best_per_n:
        short = (
            r["filters"]
            .replace("deep_pullback_", "dp_")
            .replace("deep_pullback", "dp_base")
        )
        is_best_n = r["snotio"] == best["snotio"]
        rc = ' style="background:#d4edda;font-weight:bold;"' if is_best_n else ""
        bpn_rows += f"""<tr{rc}>
            <td>N={r['n']}</td>
            <td><strong>{r['snotio']:.4f}</strong></td>
            <td>{r['loss_rate']:.1%}</td>
            <td>{r['stop_rate']:.1%}</td>
            <td>{r['worst_10']:.4f}</td>
            <td>{r['worst_5']:.4f}</td>
            <td>{r['mae_risk']:.4f}</td>
            <td>{r['sharpe_pt']:.4f}</td>
            <td>{r['win_rate']:.1%}</td>
            <td>{r['trades']}</td>
            <td title="{r['filters']}"><code>{short}</code></td>
        </tr>\n"""

    # Baseline row
    bl_row = f"""<tr style="background:#fff3cd;">
        <td>N=0 (none)</td>
        <td>{baseline['snotio']:.4f}</td>
        <td>{baseline['loss_rate']:.1%}</td>
        <td>{baseline['stop_rate']:.1%}</td>
        <td>{baseline['worst_10']:.4f}</td>
        <td>{baseline['worst_5']:.4f}</td>
        <td>{baseline['mae_risk']:.4f}</td>
        <td>{baseline['sharpe_pt']:.4f}</td>
        <td>{baseline['win_rate']:.1%}</td>
        <td>{baseline['trades']}</td>
        <td><code>none</code></td>
    </tr>"""

    # Delta vs baseline
    delta_snotio = best["snotio"] - baseline["snotio"]
    delta_worst10 = best["worst_10"] - baseline["worst_10"]

    # Marginal contribution table rows
    marginal_rows_html = ""
    bl_snotio = baseline["snotio"]
    bl_stop = baseline["stop_rate"]
    bl_trades = baseline["trades"]
    if marginal:
        for m in marginal:
            fid_short = m["filter_id"].replace("deep_pullback_", "dp_")
            t = m["this"]
            mg = m["marginal"]
            # color the verdict
            v = m["verdict"]
            if "✅" in v:
                v_style = "color:#27ae60;font-weight:bold;"
            elif "❌" in v:
                v_style = "color:#e74c3c;font-weight:bold;"
            else:
                v_style = "color:#f39c12;"
            # color marginal snotio vs baseline
            ms = mg["snotio"]
            ms_color = (
                "#27ae60"
                if ms > bl_snotio * 1.05
                else ("#e74c3c" if ms < bl_snotio * 0.95 else "#7f8c8d")
            )
            mstop = mg["stop_rate"]
            mstop_color = (
                "#27ae60"
                if mstop < bl_stop - 0.03
                else ("#e74c3c" if mstop > bl_stop + 0.03 else "#7f8c8d")
            )
            marginal_rows_html += f"""<tr>
                <td><code>{fid_short}</code></td>
                <td>{t['snotio']:.4f}</td><td>{t['stop_rate']:.1%}</td><td>{t['trades']}</td>
                <td style="border-left:2px solid #3498db;color:{ms_color};font-weight:bold;">{mg['snotio']:.4f}</td>
                <td>{mg['loss_rate']:.1%}</td>
                <td style="color:{mstop_color};font-weight:bold;">{mg['stop_rate']:.1%}</td>
                <td>{mg['trades']}</td>
                <td>{m['overlap']}</td>
                <td style="{v_style}">{v}</td>
            </tr>\n"""

    # Chart JS data
    chart_ns_with_baseline = [0] + chart_ns
    chart_snotio_with_baseline = [baseline["snotio"]] + chart_snotio
    chart_worst10_with_baseline = [baseline["worst_10"]] + chart_worst10
    chart_trades_with_baseline = [baseline["trades"]] + chart_trades
    chart_loss_rate_with_baseline = [baseline["loss_rate"]] + chart_loss_rate
    chart_stop_rate_with_baseline = [baseline["stop_rate"]] + chart_stop_rate

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Entry Filter snotio Combo Search — {strategy}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 20px; background: #f5f6fa; color: #2c3e50; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 15px; margin: 15px 0; }}
  .kpi-item {{ background: white; border-radius: 8px; padding: 18px; text-align: center;
               box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  .kpi-item.primary {{ border-left: 4px solid #3498db; }}
  .kpi-item.success {{ border-left: 4px solid #27ae60; }}
  .kpi-item.warning {{ border-left: 4px solid #f39c12; }}
  .kpi-item.danger  {{ border-left: 4px solid #e74c3c; }}
  .kpi-item.info    {{ border-left: 4px solid #9b59b6; }}
  .kpi-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 28px; font-weight: bold; margin: 5px 0; }}
  .kpi-sub   {{ font-size: 11px; color: #95a5a6; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ padding: 8px 12px; text-align: right; border: 1px solid #ddd; font-size: 13px; }}
  th {{ background: #34495e; color: white; }}
  td:last-child {{ text-align: left; max-width: 300px; overflow: hidden; text-overflow: ellipsis; }}
  tr:hover {{ background: #ecf0f1; }}
  .info-box {{ background: #eaf2f8; border-left: 4px solid #3498db; padding: 12px 16px;
               margin: 15px 0; border-radius: 4px; font-size: 13px; }}
  .warn-box {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 12px 16px;
               margin: 15px 0; border-radius: 4px; font-size: 13px; }}
  canvas {{ max-width: 100%; }}
  code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>

<h1>🎯 Entry Filter snotio Combo Search — {strategy}</h1>

<div class="info-box">
  <strong>snotio</strong> = mean(R-multiples) = 平均每笔交易的风险调整收益。<br>
  Entry Filter 主 KPI。不受 trade count 影响，只有 per-trade 质量提升才会改善。<br>
  Sharpe 仅作辅助参考（全系统毕业指标），不参与 Entry 层排序。
</div>

<div class="kpi-grid">
  <div class="kpi-item primary">
    <div class="kpi-label">Best snotio</div>
    <div class="kpi-value">{best['snotio']:.4f}</div>
    <div class="kpi-sub">N={best['n']}, {best['trades']} trades</div>
  </div>
  <div class="kpi-item {'success' if delta_snotio > 0 else 'warning'}">
    <div class="kpi-label">vs Baseline</div>
    <div class="kpi-value">{'+' if delta_snotio > 0 else ''}{delta_snotio:.4f}</div>
    <div class="kpi-sub">baseline snotio={baseline['snotio']:.4f}</div>
  </div>
  <div class="kpi-item {'success' if delta_worst10 > 0 else 'danger'}">
    <div class="kpi-label">Worst 10% Δ</div>
    <div class="kpi-value">{'+' if delta_worst10 > 0 else ''}{delta_worst10:.4f}</div>
    <div class="kpi-sub">best={best['worst_10']:.4f} vs bl={baseline['worst_10']:.4f}</div>
  </div>
  <div class="kpi-item info">
    <div class="kpi-label">Execution Load</div>
    <div class="kpi-value">{best['trades']}</div>
    <div class="kpi-sub">vs baseline {baseline['trades']} ({best['trades']/baseline['trades']:.1%})</div>
  </div>
  <div class="kpi-item {'success' if best.get('loss_rate',0) < baseline.get('loss_rate',1) else 'danger'}">
    <div class="kpi-label">Loss Rate</div>
    <div class="kpi-value">{best.get('loss_rate',0):.1%}</div>
    <div class="kpi-sub">baseline {baseline.get('loss_rate',0):.1%}</div>
  </div>
  <div class="kpi-item {'success' if best.get('stop_rate',0) < baseline.get('stop_rate',1) else 'danger'}">
    <div class="kpi-label">Stop Rate</div>
    <div class="kpi-value">{best.get('stop_rate',0):.1%}</div>
    <div class="kpi-sub">baseline {baseline.get('stop_rate',0):.1%}</div>
  </div>
</div>

<h2>📈 snotio vs Filter 数量 (Best-per-N)</h2>
<div class="card">
  <div class="warn-box">
    理想曲线: N=0→k <strong>快速上升</strong>（删垃圾）→ N=k→m <strong>平台</strong>（够用了）→ N&gt;m <strong>下降</strong>（过拟合）<br>
    <strong>选平台中段，不选峰值。</strong>
  </div>
  <canvas id="snotioChart" height="100"></canvas>
</div>

<h2>📊 Best per N (by snotio)</h2>
<div class="card">
  <table>
    <thead>
      <tr><th>N</th><th>snotio</th><th>Loss%</th><th>Stop%</th><th>Worst10%</th><th>Worst5%</th><th>MAE/risk</th>
          <th>Sharpe_pt</th><th>Win%</th><th>Trades</th><th>Filters</th></tr>
    </thead>
    <tbody>
      {bl_row}
      {bpn_rows}
    </tbody>
  </table>
</div>

<h2>🏆 Top 30 Combos (by snotio)</h2>
<div class="card">
  <table>
    <thead>
      <tr><th>Rank</th><th>N</th><th>snotio</th><th>Loss%</th><th>Stop%</th><th>Worst10%</th><th>Worst5%</th>
          <th>MAE/risk</th><th>Sharpe_pt</th><th>Win%</th><th>Trades</th><th>Filters</th></tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>

<h2>🔬 边际贡献分析 (Marginal Contribution)</h2>
<div class="card">
  <div class="info-box" style="margin-bottom:15px;">
    <strong>边际交易</strong> = 该 filter 触发但其他所有 filter 都未触发的交易。<br>
    如果边际 snotio ≈ baseline 且 stop_rate ≈ baseline → 该 filter 没有独立过滤能力，不值得启用。
  </div>
  <table>
    <thead>
      <tr>
        <th>Filter</th>
        <th>整体 snotio</th><th>整体 Stop%</th><th>整体 Trades</th>
        <th style="border-left:2px solid #3498db;">边际 snotio</th><th>边际 Loss%</th><th>边际 Stop%</th><th>边际 Trades</th>
        <th>重叠</th><th>判定</th>
      </tr>
    </thead>
    <tbody>
      {marginal_rows_html}
      <tr style="background:#f8f9fa;font-style:italic;">
        <td>Baseline (none)</td>
        <td>{bl_snotio:.4f}</td><td>{bl_stop:.1%}</td><td>{bl_trades}</td>
        <td style="border-left:2px solid #3498db;">—</td><td>—</td><td>—</td><td>—</td>
        <td>—</td><td>对照</td>
      </tr>
    </tbody>
  </table>
</div>

<h2>🔍 Loss Rate / Stop Rate 分析</h2>
<div class="card">
  <canvas id="rateChart" height="80"></canvas>
  <div class="info-box" style="margin-top:15px;">
    <strong>Loss Rate</strong> = 亏损交易占比 (R &lt; 0)。<strong>Stop Rate</strong> = 触发止损占比 (R ≈ -SL)。<br>
    好的 Entry Filter 应同时降低两者。如果 snotio 提升但 stop_rate 不变 → filter 只是挑 easy trade（伪改进）。
  </div>
</div>

<h2>🔍 Worst-10% Trade 分析</h2>
<div class="card">
  <canvas id="worstChart" height="80"></canvas>
  <div class="info-box" style="margin-top:15px;">
    Entry Filter 的真正功劳 = <strong>把最烂的交易赶出去</strong>。<br>
    如果 worst-10% 明显改善但 snotio 不动 → filter 在精准狙击坏 trade。<br>
    如果 worst-10% 不动但 snotio 变好 → filter 可能在挑 easy trade（警惕过拟合）。
  </div>
</div>

<script>
const ns = {chart_ns_with_baseline};
const snotioData = {chart_snotio_with_baseline};
const worst10Data = {chart_worst10_with_baseline};
const tradesData = {chart_trades_with_baseline};
const lossRateData = {chart_loss_rate_with_baseline};
const stopRateData = {chart_stop_rate_with_baseline};

new Chart(document.getElementById('snotioChart'), {{
  type: 'line',
  data: {{
    labels: ns.map(n => n === 0 ? 'none' : 'N=' + n),
    datasets: [
      {{
        label: 'snotio (mean R)',
        data: snotioData,
        borderColor: '#3498db',
        backgroundColor: 'rgba(52,152,219,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 6,
        pointBackgroundColor: '#3498db',
        yAxisID: 'y',
      }},
      {{
        label: 'Trades',
        data: tradesData,
        borderColor: '#95a5a6',
        borderDash: [5, 5],
        pointRadius: 4,
        pointBackgroundColor: '#95a5a6',
        yAxisID: 'y1',
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'snotio vs Entry Filter 数量 (higher = better per-trade quality)' }} }},
    scales: {{
      y: {{ position: 'left', title: {{ display: true, text: 'snotio (mean R)' }} }},
      y1: {{ position: 'right', title: {{ display: true, text: 'Trades' }}, grid: {{ drawOnChartArea: false }} }}
    }}
  }}
}});

new Chart(document.getElementById('worstChart'), {{
  type: 'bar',
  data: {{
    labels: ns.map(n => n === 0 ? 'none' : 'N=' + n),
    datasets: [{{
      label: 'Worst 10% mean R',
      data: worst10Data,
      backgroundColor: worst10Data.map(v => v < -1.5 ? '#e74c3c' : v < -1.0 ? '#f39c12' : '#27ae60'),
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Worst 10% Trade 平均 R (越接近 0 越好 = 坏 trade 被过滤)' }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Worst 10% mean R' }} }} }}
  }}
}});

new Chart(document.getElementById('rateChart'), {{
  type: 'bar',
  data: {{
    labels: ns.map(n => n === 0 ? 'none' : 'N=' + n),
    datasets: [
      {{
        label: 'Loss Rate',
        data: lossRateData.map(v => v * 100),
        backgroundColor: '#e74c3c',
        borderRadius: 4,
      }},
      {{
        label: 'Stop Rate',
        data: stopRateData.map(v => v * 100),
        backgroundColor: '#e67e22',
        borderRadius: 4,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Loss Rate & Stop Rate vs N (低 = Entry Filter 有效过滤坑交易)' }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Rate (%)' }} }} }}
  }}
}});
</script>

<div class="card" style="margin-top:30px; background:#ecf0f1; font-size:12px; color:#7f8c8d;">
  <strong>层级 KPI 定位:</strong>
  Entry Filter → <strong>snotio / loss_rate / stop_rate</strong> |
  Evidence → failure rate / drawdown |
  Execution → Sharpe / Calmar / MDD |
  全系统 → OOS Sharpe<br>
  <strong>数据:</strong> {strategy} predictions.parquet, span={span_years:.2f}yr, {len(results)} combos evaluated
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ================================================================
# CLI
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Entry Filter 组合搜索 — snotio KPI (2^N-1 子集穷举)"
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="predictions.parquet 路径",
    )
    parser.add_argument("--strategy", required=True, help="策略名 (e.g. bpc)")
    parser.add_argument(
        "--output",
        default=None,
        help="HTML 输出路径 (默认: 同目录 entry_filter_snotio_combo.html)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="全量评估: 包含 filters + disabled_filters 中所有有 conditions 的 filter",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        default=False,
        help="全特征扫描: 自动测试所有数值列 × 多阈值 (P10-P90), execution 模拟排序",
    )
    parser.add_argument(
        "--scan-top",
        type=int,
        default=40,
        help="--scan 模式下终端显示前 N 条 (默认 40)",
    )
    args = parser.parse_args()

    # Load data
    print("📂 Loading data...")
    merged = pd.read_parquet(args.logs)
    if "_symbol" in merged.columns and "symbol" not in merged.columns:
        merged = merged.rename(columns={"_symbol": "symbol"})

    exec_config = load_execution_config(args.strategy)
    entry_cfg = load_entry_filters_config(args.strategy, research=True)

    # --all: 合并 filters + disabled_filters 为一个列表，全部当作 enabled 评估
    if args.all and not args.scan:
        all_filters = list(entry_cfg.get("filters", []))
        disabled = entry_cfg.get("disabled_filters", [])
        if disabled:
            all_filters.extend(disabled)
        # 去重 (by id)
        seen_ids = set()
        deduped = []
        for f in all_filters:
            fid = f.get("id", "")
            if fid and fid not in seen_ids:
                seen_ids.add(fid)
                deduped.append(f)
        # 全部设为 enabled
        for f in deduped:
            f["enabled"] = True
        entry_cfg["filters"] = deduped
        print(f"   --all mode: {len(deduped)} filters (filters + disabled_filters)")

    # 策略感知的方向确定 (direction.yaml)
    dir_cfg = load_direction_config(args.strategy)
    if dir_cfg and dir_cfg.get("direction_rules"):
        used = apply_direction_rules(merged, args.strategy, dir_cfg)
        if used is None:
            print("❌ No direction rule matched, check direction.yaml")
            return
    elif "bpc_breakout_direction" in merged.columns:
        merged["entry_direction"] = merged["bpc_breakout_direction"].astype(float)
    else:
        print(
            "❌ No direction column found (no direction.yaml and no bpc_breakout_direction)"
        )
        return
    merged = merged.sort_values(["symbol"]).reset_index(drop=True)

    # 计算衍生 entry filter 特征（正交维度）
    compute_derived_entry_features(merged)
    print(f"   Derived features: {[c for c in merged.columns if c.startswith('ef_')]}")

    if "gate_decision" in merged.columns:
        merged.loc[merged["gate_decision"] != "allow", "entry_direction"] = 0.0

    orig_dir = merged["entry_direction"].copy()
    span_years = _estimate_span_years(merged)

    # ============================================================
    # --scan 模式: 全特征自动扫描
    # ============================================================
    if args.scan:
        print(f"\n🔍 Running full feature scan (execution simulation)...")
        baseline = compute_baseline(merged, orig_dir, exec_config)
        scan_results = run_feature_scan(
            merged, orig_dir, exec_config, span_years, min_samples=200
        )

        # Terminal output
        top_n = args.scan_top
        print()
        print("=" * 140)
        print(
            f"FEATURE SCAN — snotio KPI ({len(scan_results)} valid, span={span_years:.2f}yr)"
        )
        print("=" * 140)
        print(
            f"{'Rank':>4} {'snotio':>8} {'Loss%':>6} {'Stop%':>6} {'Sh_pt':>7} {'Win%':>6} {'Trades':>7}  "
            f"{'Feature':<40} {'Op':>3} {'Thresh':>10} {'Pct':>4}"
        )
        print("-" * 140)
        for i, r in enumerate(scan_results[:top_n], 1):
            delta = r["snotio"] - baseline["snotio"]
            marker = " ★" if delta > baseline["snotio"] * 0.1 else ""
            print(
                f"{i:>4} {r['snotio']:>8.4f} {r['loss_rate']*100:>5.1f}% {r['stop_rate']*100:>5.1f}% "
                f"{r['sharpe_pt']:>7.4f} {r['win_rate']*100:>5.1f}% {r['trades']:>7}  "
                f"{r['feature']:<40} {r['operator']:>3} {r['threshold']:>10.4f} {r['percentile']:>4}{marker}"
            )

        print(
            f"\n   Baseline (none): snotio={baseline['snotio']:.4f}, Loss={baseline['loss_rate']:.1%}, "
            f"Stop={baseline['stop_rate']:.1%}, Sh_pt={baseline['sharpe_pt']:.4f}, Trades={baseline['trades']}"
        )

        if scan_results:
            best = scan_results[0]
            delta = best["snotio"] - baseline["snotio"]
            print(
                f"\n   🏆 BEST: {best['feature']} {best['operator']} {best['threshold']:.4f} ({best['percentile']})"
            )
            print(
                f"      snotio={best['snotio']:.4f}, Trades={best['trades']}, "
                f"Δ snotio vs baseline: {'+' if delta > 0 else ''}{delta:.4f} ({delta/baseline['snotio']*100:+.1f}%)"
            )

        # 输出 YAML 片段 (top 5)
        print(f"\n   📋 Top 5 YAML 片段 (可直接复制到 entry_filters.yaml):")
        for i, r in enumerate(scan_results[:5], 1):
            fid = r["feature"].replace(".", "_")
            op_str = "high" if r["operator"] == ">=" else "low"
            print(
                f"      # {i}. {r['feature']} {r['operator']} {r['threshold']:.4f} | snotio={r['snotio']:.4f}"
            )
            print(f"      - id: scan_{fid}_{op_str}")
            print(f"        enabled: true")
            print(f"        conditions:")
            print(f'          - feature: {r["feature"]}')
            print(f'            operator: "{r["operator"]}"')
            print(f"            value: {r['threshold']}")
            print()

        # HTML
        if args.output:
            out_path = args.output
        else:
            out_dir = os.path.dirname(args.logs)
            out_path = os.path.join(out_dir, "entry_filter_scan.html")
        generate_html_report(
            scan_results, baseline, [], span_years, args.strategy, out_path
        )
        print(f"   📄 HTML report: {out_path}")
        return

    # ============================================================
    # 正常模式: combo search
    # ============================================================
    max_n = 1 if args.all else 0  # --all: 只跑单 filter (N=1)，避免 2^16 组合爆炸
    print(
        f"\n🔍 Running combo search (snotio KPI, max_n={'all' if max_n == 0 else max_n})..."
    )
    results = run_combo_search(
        merged, orig_dir, exec_config, entry_cfg, span_years, max_n=max_n
    )

    # Baseline
    print("\n📊 Computing baseline (no entry filter)...")
    baseline = compute_baseline(merged, orig_dir, exec_config)

    # Sort by snotio
    results.sort(key=lambda x: -x["snotio"])
    best_per_n = get_best_per_n(results, "snotio")

    # Terminal output
    print()
    print("=" * 120)
    print(
        f"ENTRY FILTER COMBO SEARCH — snotio KPI ({len(results)} combos, span={span_years:.2f}yr)"
    )
    print("=" * 120)
    print(
        f"{'Rank':>4} {'N':>2} {'snotio':>8} {'Loss%':>6} {'Stop%':>6} {'W10%':>7} {'W5%':>7} {'MAE/r':>7} {'Sh_pt':>7} {'Win%':>6} {'Trades':>7}  Filters"
    )
    print("-" * 130)
    for i, r in enumerate(results[:20], 1):
        print(
            f"{i:>4} {r['n']:>2} {r['snotio']:>8.4f} {r['loss_rate']*100:>5.1f}% {r['stop_rate']*100:>5.1f}% {r['worst_10']:>7.4f} "
            f"{r['worst_5']:>7.4f} {r['mae_risk']:>7.4f} {r['sharpe_pt']:>7.4f} "
            f"{r['win_rate']*100:>5.1f}% {r['trades']:>7}  {r['filters']}"
        )

    print(
        f"\n   Baseline (none): snotio={baseline['snotio']:.4f}, Loss={baseline['loss_rate']:.1%}, Stop={baseline['stop_rate']:.1%}, "
        f"Sh_pt={baseline['sharpe_pt']:.4f}, Trades={baseline['trades']}"
    )

    if results:
        best = results[0]
        print(
            f"\n   🏆 BEST: N={best['n']}, snotio={best['snotio']:.4f}, "
            f"Loss={best['loss_rate']:.1%}, Stop={best['stop_rate']:.1%}, Trades={best['trades']}"
        )
        print(f"      Filters: {best['filters']}")
        delta = best["snotio"] - baseline["snotio"]
        print(f"      Δ snotio vs baseline: {'+' if delta > 0 else ''}{delta:.4f}")

    print(f"\n   📊 Best per N (by snotio):")
    for r in best_per_n:
        print(
            f"      N={r['n']}: snotio={r['snotio']:.4f}, Loss={r['loss_rate']:.1%}, Stop={r['stop_rate']:.1%}, "
            f"Sh_pt={r['sharpe_pt']:.4f}, Trades={r['trades']}, Win={r['win_rate']*100:.1f}%"
        )

    # Marginal contribution analysis
    print("\n🔬 Running marginal contribution analysis...")
    marginal = run_marginal_analysis(merged, orig_dir, exec_config, entry_cfg, baseline)

    if marginal:
        print()
        print("=" * 120)
        print("MARGINAL CONTRIBUTION ANALYSIS (边际贡献分析)")
        print("=" * 120)
        print(
            f"{'Filter':<28} {'整体snotio':>9} {'整体Stop':>8} {'整体N':>5}  │ {'边际snotio':>9} {'边际Loss':>9} {'边际Stop':>9} {'边际N':>5}  判定"
        )
        print("-" * 120)
        for m in marginal:
            t = m["this"]
            mg = m["marginal"]
            print(
                f"{m['filter_id']:<28} {t['snotio']:>9.4f} {t['stop_rate']*100:>6.1f}% {t['trades']:>5}  │ "
                f"{mg['snotio']:>9.4f} {mg['loss_rate']*100:>7.1f}% {mg['stop_rate']*100:>7.1f}% {mg['trades']:>5}  {m['verdict']}"
            )
        print(
            f"{'Baseline':<28} {baseline['snotio']:>9.4f} {baseline['stop_rate']*100:>6.1f}% {baseline['trades']:>5}  │ {'—':>9} {'—':>9} {'—':>9} {'—':>5}  对照"
        )

    # HTML output
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.dirname(args.logs)
        out_path = os.path.join(out_dir, "entry_filter_snotio_combo.html")

    generate_html_report(
        results,
        baseline,
        best_per_n,
        span_years,
        args.strategy,
        out_path,
        marginal=marginal,
    )
    print(f"\n   📄 HTML report: {out_path}")


if __name__ == "__main__":
    main()
