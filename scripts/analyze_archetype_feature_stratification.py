#!/usr/bin/env python3
"""分位数分层分析：验证 archetype 语义特征的预测力。

对指定策略的候选特征，按百分位阈值切分数据，
对比「有语义信号」vs「无语义信号」的 bad rate 和 median RR。

用法:
    # 默认: 自动读取 config/strategies/{strategy}/prefilter.yaml
    python scripts/analyze_archetype_feature_stratification.py \
        --logs results/train_final_xxx/bpc/predictions.parquet \
        --strategy bpc

    # 指定阈值百分位
    python scripts/analyze_archetype_feature_stratification.py \
        --logs results/train_final_xxx/fer/predictions.parquet \
        --strategy fer --percentiles 5,10,80,90,95

    # 输出 JSON 报告
    python scripts/analyze_archetype_feature_stratification.py \
        --logs results/train_final_xxx/bpc/predictions.parquet \
        --strategy bpc --output results/bpc_stratification.json

算法:
    对每个候选特征:
      1. 取全量数据 (predictions.parquet)
      2. 按 P5 / P10 / P80 / P90 / P95 阈值切分数据为两组
      3. 计算两组的:
         - bad rate = failure_rr_extreme 占比 (forward_rr < -0.8R)
         - median forward_rr
      4. 差异越大 → 该特征在此阈值处有区分力

特征来源:
    config/strategies/{strategy}/prefilter.yaml 的 candidates 列表
    → 查 feature_dependencies.yaml 的 output_columns 解析实际列名
    → 匹配 parquet 中存在的列
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd

# ── 默认百分位 ────────────────────────────────────────────────
DEFAULT_PERCENTILES = [5, 10, 20, 80, 90, 95]

# ── 默认最小样本量 ─────────────────────────────────────────
DEFAULT_MIN_SAMPLES = 30

# ── 默认 feature_dependencies.yaml 路径 ───────────────────────
DEFAULT_DEPS_PATH = "config/feature_dependencies.yaml"

# ── Temporal analysis constants ─────────────────────────────
TEMPORAL_WINDOW_MONTHS = [2, 3, 4, 6]
TEMPORAL_MIN_SAMPLES_PER_WINDOW = 1080  # 统计可信最小样本量


def _resolve_features_from_config(
    config_path: str,
    deps_path: str,
    available_columns: List[str],
) -> List[str]:
    """从 prefilter.yaml + feature_dependencies.yaml 解析候选特征列名。

    链路: prefilter.yaml (candidates) → feature_dependencies.yaml (output_columns)
          → 匹配 parquet 中存在的列
    """
    # 1. 读 prefilter.yaml
    with open(config_path, "r", encoding="utf-8") as f:
        prefilter_cfg = yaml.safe_load(f)

    candidates = prefilter_cfg.get("candidates", [])
    if not candidates:
        print(f"❌ prefilter.yaml 中没有 candidates: {config_path}")
        return []

    # 2. 读 feature_dependencies.yaml
    with open(deps_path, "r", encoding="utf-8") as f:
        deps_cfg = yaml.safe_load(f)

    all_features_def = deps_cfg.get("features", {})

    # 3. 解析 _f → output_columns
    resolved_columns = []
    available_set = set(available_columns)

    for feat_f in candidates:
        if feat_f not in all_features_def:
            print(f"⚠️  _f '{feat_f}' 在 feature_dependencies.yaml 中未找到, 跳过")
            continue

        output_cols = all_features_def[feat_f].get("output_columns", [])
        matched = [c for c in output_cols if c in available_set]
        skipped = [c for c in output_cols if c not in available_set]

        if matched:
            resolved_columns.extend(matched)
            suffix = "..." if len(matched) > 5 else ""
            print(
                f"  \u2705 {feat_f}: {len(matched)} \u5217\u5339\u914d ({', '.join(matched[:5])}{suffix})"
            )
        if skipped:
            print(f"  ℹ️  {feat_f}: {len(skipped)} 列不在 parquet 中: {skipped[:5]}")

    return sorted(set(resolved_columns))


def _compute_stratification(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    operator: str,  # "high" (>= threshold) or "low" (<= threshold)
    rr_col: str,
    label_col: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Optional[Dict[str, Any]]:
    """
    按阈值分层，计算两组的 bad rate 和 median RR。

    operator="high": 分析「高端信号」→ df[feature >= threshold] vs 其余
    operator="low":  分析「低端信号」→ df[feature <= threshold] vs 其余
    """
    # 避免重复列名导致 DataFrame 而非 Series
    cols_needed = [feature, label_col]
    if rr_col in df.columns and df[rr_col].notna().any():
        cols_needed.append(rr_col)
        has_rr = True
    else:
        has_rr = False
    # 去重列名
    cols_needed = list(dict.fromkeys(cols_needed))
    valid = df[cols_needed].copy()
    if valid.columns.duplicated().any():
        valid = valid.loc[:, ~valid.columns.duplicated()]
    valid = valid.dropna().reset_index(drop=True)
    if len(valid) < min_samples * 2:
        return None

    if operator == "high":
        signal_mask = (valid[feature] >= threshold).values
    else:
        signal_mask = (valid[feature] <= threshold).values

    signal_df = valid.loc[signal_mask]
    rest_df = valid.loc[~signal_mask]

    if len(signal_df) < min_samples or len(rest_df) < min_samples:
        return None

    # bad rate = 标签为 0 的比例 (label_col = success_no_rr_extreme, 0 = 踩坑)
    signal_bad_rate = (signal_df[label_col] == 0).mean()
    rest_bad_rate = (rest_df[label_col] == 0).mean()

    # median forward_rr
    signal_med_rr = (
        float(signal_df[rr_col].median())
        if rr_col in signal_df.columns
        else float("nan")
    )
    rest_med_rr = (
        float(rest_df[rr_col].median()) if rr_col in rest_df.columns else float("nan")
    )

    return {
        "n_signal": len(signal_df),
        "n_rest": len(rest_df),
        "bad_rate_signal": round(signal_bad_rate, 4),
        "bad_rate_rest": round(rest_bad_rate, 4),
        "bad_rate_diff": round(signal_bad_rate - rest_bad_rate, 4),
        "bad_rate_diff_abs": round(abs(signal_bad_rate - rest_bad_rate), 4),
        "median_rr_signal": round(signal_med_rr, 2),
        "median_rr_rest": round(rest_med_rr, 2),
        "threshold": round(threshold, 4),
    }


def analyze_feature(
    df: pd.DataFrame,
    feature: str,
    percentiles: List[int],
    rr_col: str,
    label_col: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[Dict[str, Any]]:
    """分析单个特征在多个百分位阈值下的分层效果。"""
    valid = df[feature].dropna()
    if len(valid) < min_samples * 2:
        return []

    results = []

    for pct in percentiles:
        threshold = float(np.percentile(valid, pct))

        # 高端分析 (P80, P90, P95): feature >= threshold → "有信号"
        if pct >= 50:
            r = _compute_stratification(
                df, feature, threshold, "high", rr_col, label_col, min_samples
            )
            if r:
                r["percentile"] = f"P{pct}"
                r["direction"] = "high"
                r["feature"] = feature
                results.append(r)

        # 低端分析 (P5, P10, P20): feature <= threshold → "低端/缺失"
        if pct <= 50:
            r = _compute_stratification(
                df, feature, threshold, "low", rr_col, label_col, min_samples
            )
            if r:
                r["percentile"] = f"P{pct}"
                r["direction"] = "low"
                r["feature"] = feature
                results.append(r)

    return results


def _format_table(results: List[Dict[str, Any]], title: str) -> str:
    """格式化输出表格。"""
    if not results:
        return f"\n{title}\n  (无有效结果)\n"

    lines = [
        f"\n{title}",
        f"{'特征':<35s} {'阈值':<12s} {'方向':>4s} {'n':>6s} {'bad_rate':>10s} {'vs其余':>10s} {'差异':>8s} {'medRR':>8s} {'vs其余':>8s}",
        "-" * 110,
    ]

    for r in results:
        op_str = ">=" if r["direction"] == "high" else "<="
        threshold_str = f"{r['percentile']}({op_str}{r['threshold']:.3f})"
        diff_str = f"{r['bad_rate_diff']:+.1%}"
        lines.append(
            f"{r['feature']:<35s} {threshold_str:<12s} {r['direction']:>4s} "
            f"{r['n_signal']:>6d} {r['bad_rate_signal']:>9.1%} {r['bad_rate_rest']:>9.1%} "
            f"{diff_str:>8s} {r['median_rr_signal']:>+7.2f} {r['median_rr_rest']:>+7.2f}"
        )

    return "\n".join(lines)


def _classify_results(
    results: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分类结果为三组:
    - 正信号 (high=good): 高端有信号 + bad_rate 降低
    - 反信号 (high=bad): 高端有信号 + bad_rate 升高
    - 低端信号 (absence=bad): 低端缺失信号 + bad_rate 升高
    """
    positive = []  # high=good (bad_rate_signal < bad_rate_rest)
    anti = []  # high=bad (bad_rate_signal > bad_rate_rest)
    absence = []  # low, absence=bad (bad_rate_signal > bad_rate_rest)

    for r in results:
        if r["direction"] == "high":
            if r["bad_rate_diff"] < -0.02:  # 信号组 bad rate 低 2% 以上 → 正信号
                positive.append(r)
            elif r["bad_rate_diff"] > 0.02:  # 信号组 bad rate 高 2% 以上 → 反信号
                anti.append(r)
        elif r["direction"] == "low":
            if r["bad_rate_diff"] > 0.02:  # 低端 bad rate 高 → absence=bad
                absence.append(r)
            elif r["bad_rate_diff"] < -0.02:
                positive.append(r)

    # 按差异绝对值排序
    positive.sort(key=lambda x: x["bad_rate_diff_abs"], reverse=True)
    anti.sort(key=lambda x: x["bad_rate_diff_abs"], reverse=True)
    absence.sort(key=lambda x: x["bad_rate_diff_abs"], reverse=True)

    return positive, anti, absence


# ── Prefilter AND 组合样本量测算 + Jaccard 冗余矩阵 ──────────────────────


def _build_prefilter_mask(
    df: pd.DataFrame,
    result: Dict[str, Any],
    category: str,
) -> pd.Series:
    """为单条 prefilter 候选构建 pass mask (True = 保留的行)。

    - positive (high=good): 保留高端 → feature >= threshold
    - anti (high=bad):      拒绝高端 → feature < threshold
    - absence (low=bad):    拒绝低端 → feature > threshold
    """
    feat = result["feature"]
    thr = result["threshold"]
    col = df[feat]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]

    if category == "positive":
        if result["direction"] == "high":
            return col >= thr
        else:  # positive from low direction (low端 bad_rate < rest)
            return col <= thr
    elif category == "anti":
        return col < thr  # 拒绝高端
    elif category == "absence":
        return col > thr  # 拒绝低端
    return pd.Series(True, index=df.index)


def _best_per_feature(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """每个 feature 只保留 bad_rate_diff_abs 最大的一条。"""
    best: Dict[str, Dict[str, Any]] = {}
    for r in results:
        feat = r["feature"]
        if feat not in best or r["bad_rate_diff_abs"] > best[feat]["bad_rate_diff_abs"]:
            best[feat] = r
    return list(best.values())


def _prefilter_combination_analysis(
    df: pd.DataFrame,
    positive: List[Dict[str, Any]],
    anti: List[Dict[str, Any]],
    absence: List[Dict[str, Any]],
    label_col: str,
    rr_col: str,
    n_total: int,
) -> Optional[Dict[str, Any]]:
    """Prefilter AND 组合分析：Jaccard 冗余 + 贪心样本量测算。

    输出:
    1. 每条候选的 pass mask 统计
    2. Jaccard 对称矩阵 (高 Jaccard = 高冗余)
    3. 贪心前向选择: 每步加 AND 后 bad_rate 降幅最大的 + 样本量
    """
    # ── 1. 收集候选 (每 feature 取最强信号) ──
    candidates: List[Tuple[Dict[str, Any], str]] = []  # (result, category)
    for r in _best_per_feature(positive):
        candidates.append((r, "positive"))
    for r in _best_per_feature(anti):
        candidates.append((r, "anti"))
    for r in _best_per_feature(absence):
        candidates.append((r, "absence"))

    if len(candidates) < 2:
        return None

    # 按 bad_rate_diff_abs 排序
    candidates.sort(key=lambda x: x[0]["bad_rate_diff_abs"], reverse=True)

    # ── 2. 构建 pass masks ──
    names: List[str] = []
    masks: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    baseline_bad_rate = float((df[label_col] == 0).mean())

    for r, cat in candidates:
        mask = _build_prefilter_mask(df, r, cat).fillna(False).values
        feat = r["feature"]
        pct = r["percentile"]
        direction = r["direction"]
        thr = r["threshold"]

        # 标签: 方向+特征名+百分位
        if cat == "positive":
            if direction == "high":
                op_str = f">={thr:.3f}"
            else:
                op_str = f"<={thr:.3f}"
        elif cat == "anti":
            op_str = f"<{thr:.3f}"
        else:  # absence
            op_str = f">{thr:.3f}"

        label = f"{feat} {op_str} ({pct})"
        pass_count = int(mask.sum())
        pass_bad = (
            float((df.loc[mask, label_col] == 0).mean())
            if pass_count > 0
            else float("nan")
        )

        names.append(label)
        masks.append(mask)
        meta.append(
            {
                "feature": feat,
                "percentile": pct,
                "category": cat,
                "operator": op_str,
                "threshold": thr,
                "pass_count": pass_count,
                "pass_pct": round(pass_count / n_total * 100, 1),
                "pass_bad_rate": round(pass_bad, 4),
                "bad_rate_diff": round(r["bad_rate_diff"], 4),
            }
        )

    # ── 3. Jaccard 矩阵 ──
    n = len(masks)
    jaccard_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            inter = np.sum(masks[i] & masks[j])
            union = np.sum(masks[i] | masks[j])
            jac = inter / union if union > 0 else 0.0
            jaccard_matrix[i, j] = jac
            jaccard_matrix[j, i] = jac

    # ── 4. 贪心前向选择 (AND, bad_rate 最低优先) ──
    remaining = set(range(n))
    selected_steps: List[Dict[str, Any]] = []
    combined_mask = np.ones(len(df), dtype=bool)

    for step in range(min(n, 8)):  # 最多 8 步
        best_idx = -1
        best_bad_rate = float("inf")
        best_combined = None

        for idx in remaining:
            trial = combined_mask & masks[idx]
            trial_count = int(trial.sum())
            if trial_count < TEMPORAL_MIN_SAMPLES_PER_WINDOW:  # 样本量保底
                continue
            trial_bad = float((df.loc[trial, label_col] == 0).mean())
            if trial_bad < best_bad_rate:
                best_bad_rate = trial_bad
                best_idx = idx
                best_combined = trial

        if best_idx < 0:
            break

        combined_mask = best_combined
        remaining.discard(best_idx)
        combo_count = int(combined_mask.sum())
        combo_bad = float((df.loc[combined_mask, label_col] == 0).mean())
        combo_med_rr = (
            float(df.loc[combined_mask, rr_col].median())
            if rr_col in df.columns
            else float("nan")
        )

        selected_steps.append(
            {
                "step": step + 1,
                "name": names[best_idx],
                "feature": meta[best_idx]["feature"],
                "pass_count_solo": meta[best_idx]["pass_count"],
                "combined_count": combo_count,
                "combined_pct": round(combo_count / n_total * 100, 1),
                "combined_bad_rate": round(combo_bad, 4),
                "combined_med_rr": (
                    round(combo_med_rr, 2) if not np.isnan(combo_med_rr) else None
                ),
                "delta_bad_rate": round(combo_bad - baseline_bad_rate, 4),
            }
        )

    # ── 5. 打印 ──
    print(f"\n{'='*110}")
    print("🧩 Prefilter AND 组合分析 (冗余检测 + 贪心前向选择)")
    print(f"{'='*110}")

    # 5a. 单条候选概览 (显示 top 20，其余折叠)
    DISPLAY_CAP = 20
    display_n = min(len(names), DISPLAY_CAP)
    print(
        f"\n  📊 候选 Prefilter ({len(names)} 条, 每特征取最强信号, 显示 top {display_n}):"
    )
    max_name_len = max(len(nm) for nm in names[:display_n])
    print(
        f"  {'条件':<{max_name_len+2}s} {'通过':>7s} {'占比':>6s} {'bad_rate':>9s} {'vs基线':>8s} {'类型':>8s}"
    )
    print(f"  {'-'*(max_name_len+2+7+6+9+8+8+5)}")
    for i in range(display_n):
        nm = names[i]
        m = meta[i]
        delta = m["pass_bad_rate"] - baseline_bad_rate
        cat_label = {"positive": "正信号", "anti": "反信号", "absence": "低端"}[
            m["category"]
        ]
        print(
            f"  {nm:<{max_name_len+2}s} {m['pass_count']:>7,d} {m['pass_pct']:>5.1f}% "
            f"{m['pass_bad_rate']:>8.1%} {delta:>+7.1%} {cat_label:>8s}"
        )
    if len(names) > DISPLAY_CAP:
        print(
            f"  ... 还有 {len(names) - DISPLAY_CAP} 条未显示 (保存至 JSON --output 查看全部)"
        )

    # 5b. Jaccard 矩阵 (仅 top 20 候选, 避免 all-features 模式下矩阵爆炸)
    JACCARD_CAP = 20
    if n >= 2:
        jac_n = min(n, JACCARD_CAP)
        # 缩短名称用于矩阵显示
        short_names = []
        for i in range(jac_n):
            feat = meta[i]["feature"]
            pct = meta[i]["percentile"]
            short = f"{feat[:25]}_{pct}" if len(feat) > 25 else f"{feat}_{pct}"
            short_names.append(short)

        max_sn = max(len(s) for s in short_names)
        extra_note = f" (top {jac_n}/{n})" if n > JACCARD_CAP else ""
        print(f"\n  📐 Jaccard 重叠矩阵{extra_note} (高值=高冗余, >=0.5 标 ⚠️):")
        header = "  " + " " * (max_sn + 5)
        for j in range(jac_n):
            header += f" {j:>5d}"
        print(header)
        for i in range(jac_n):
            row = f"  {i:>3d}: {short_names[i]:<{max_sn}s}"
            for j in range(jac_n):
                if j < i:
                    row += "      "
                elif j == i:
                    row += "   1.0"
                else:
                    jv = jaccard_matrix[i, j]
                    if jv < 0.05:
                        row += "    . "
                    else:
                        warn = "⚠" if jv >= 0.5 else " "
                        row += f" {jv:.2f}{warn}"
            print(row)
        if n > JACCARD_CAP:
            print(f"  ... 完整 {n}×{n} 矩阵保存至 JSON --output")

    # 5c. 贪心 AND 组合
    if selected_steps:
        print(f"\n  🔗 贪心前向选择 (AND 组合, 每步选 bad_rate 最低):")
        print(
            f"  {'Step':>4s}  {'+ Prefilter':<{max_name_len+2}s} {'样本量':>8s} {'占比':>6s} {'bad_rate':>9s} {'vs基线':>8s} {'medRR':>7s}"
        )
        print(f"  {'-'*(4+2+max_name_len+2+8+6+9+8+7+6)}")

        # Baseline row
        baseline_med_rr = (
            float(df[rr_col].median()) if rr_col in df.columns else float("nan")
        )
        print(
            f"  {'base':>4s}  {'(无 prefilter)':<{max_name_len+2}s} "
            f"{n_total:>8,d} {100.0:>5.1f}% {baseline_bad_rate:>8.1%} {0:>+7.1%} "
            f"{baseline_med_rr:>+6.2f}"
        )

        for s in selected_steps:
            rr_str = (
                f"{s['combined_med_rr']:>+6.2f}"
                if s["combined_med_rr"] is not None
                else "   N/A"
            )
            print(
                f"  {s['step']:>4d}  +{s['name']:<{max_name_len+1}s} "
                f"{s['combined_count']:>8,d} {s['combined_pct']:>5.1f}% "
                f"{s['combined_bad_rate']:>8.1%} {s['delta_bad_rate']:>+7.1%} "
                f"{rr_str}"
            )

        final = selected_steps[-1]
        print(
            f"\n  ✅ {len(selected_steps)} 条 AND 组合: "
            f"样本 {n_total:,d} → {final['combined_count']:,d} ({final['combined_pct']:.1f}%), "
            f"bad_rate {baseline_bad_rate:.1%} → {final['combined_bad_rate']:.1%} "
            f"({final['delta_bad_rate']:+.1%})"
        )
        if final["combined_count"] < TEMPORAL_MIN_SAMPLES_PER_WINDOW:
            print(
                f"  ⚠️  最终样本量 {final['combined_count']} < {TEMPORAL_MIN_SAMPLES_PER_WINDOW}, 统计不可信!"
            )
    else:
        print(
            f"\n  ⚠️  无法构建有效 AND 组合 (所有组合样本量 < {TEMPORAL_MIN_SAMPLES_PER_WINDOW})"
        )

    return {
        "candidates": meta,
        "jaccard_matrix": jaccard_matrix.tolist(),
        "candidate_names": names,
        "greedy_and_steps": selected_steps,
        "baseline_bad_rate": round(baseline_bad_rate, 4),
        "n_total": n_total,
    }


# ── 综合评分精选推荐 + rules: YAML 生成 ────────────────────────────


def _compute_robustness(
    feature: str,
    percentile: str,
    direction: str,
    bad_rate_diff: float,
    all_results: List[Dict[str, Any]],
) -> Tuple[float, int]:
    """Check adjacent percentile consistency for robustness.

    Only compares with same-feature, same-direction results at other percentiles.
    E.g. sma_200_position P5 low vs P10 low, P20 low (not P80 high).

    Returns (robustness_score, n_adjacent_agree).
    """
    same_feat_dir = [
        r
        for r in all_results
        if r["feature"] == feature
        and r.get("direction") == direction
        and r.get("percentile") != percentile
    ]
    if not same_feat_dir:
        return 0.5, 0  # no adjacent to compare, neutral

    # Reference sign: does this candidate improve (diff < 0) or worsen (diff > 0)?
    ref_negative = bad_rate_diff < 0
    agree = 0
    for r in same_feat_dir:
        other_diff = r["bad_rate_diff"]
        if ref_negative and other_diff < -0.01:
            agree += 1
        elif not ref_negative and other_diff > 0.01:
            agree += 1

    return (agree / len(same_feat_dir)), agree


def _prefilter_recommendation(
    positive: List[Dict[str, Any]],
    anti: List[Dict[str, Any]],
    absence: List[Dict[str, Any]],
    all_results: List[Dict[str, Any]],
    temporal_report: Optional[Dict] = None,
    strategy: str = "",
    df: Optional[pd.DataFrame] = None,
    label_col: str = "success_no_rr_extreme",
    n_gate_features: Optional[int] = None,
    min_prefilter_pass_rate: Optional[float] = None,
    min_prefilter_rows: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Generate composite-scored recommendation table + rules: YAML.

    Scoring: robustness (40%) > CV (30%) > bad_rate_diff (30%)
    Signal routing:
      - positive + absence -> prefilter rules
      - anti -> leave to Gate

    Includes:
      - 覆盖率列 (% of total)
      - Tier 自动分级 (T1 >= 0.80, T2 >= 0.70, T3 >= 0.50)
      - AND 累积模拟: 逐条 AND 后剩余样本量 + bad_rate
      - 实用方案推荐: 自动找出满足最低训练量的 AND 组合
      - 方案 E: 语义 OR 组合 (自动检测 longs/shorts 互补对)
    """
    import operator as op_module

    # Operator map for AND simulation
    _SIM_OPS = {
        ">=": op_module.ge,
        "<=": op_module.le,
        ">": op_module.gt,
        "<": op_module.lt,
    }

    n_dataset = len(df) if df is not None else 0
    MIN_TRAIN_SAMPLES = 1080  # train_strategy_pipeline.py hard floor

    # Build temporal CV map
    cv_map: Dict[Tuple[str, str, str], float] = {}
    if temporal_report and "feature_stability" in temporal_report:
        for tr in temporal_report["feature_stability"]:
            key = (tr["feature"], tr.get("percentile", ""), tr.get("direction", ""))
            cv_map[key] = tr.get("cv", None)

    # Collect prefilter-eligible candidates (positive + absence, per-feature best)
    prefilter_pool: List[Dict[str, Any]] = []
    seen_feat: set = set()

    # positive signals -> prefilter candidates
    for r in positive:
        feat = r["feature"]
        if feat in seen_feat:
            continue
        seen_feat.add(feat)
        prefilter_pool.append({**r, "_category": "positive"})

    # absence signals -> prefilter candidates (reject low end)
    for r in absence:
        feat = r["feature"]
        if feat in seen_feat:
            continue
        seen_feat.add(feat)
        prefilter_pool.append({**r, "_category": "absence"})

    if not prefilter_pool:
        return None

    # Score each candidate
    scored: List[Dict[str, Any]] = []
    for r in prefilter_pool:
        feat = r["feature"]
        pct = r["percentile"]
        direction = r["direction"]
        bad_diff = abs(r["bad_rate_diff"])

        # Robustness: adjacent percentile consistency
        rob_score, n_adj = _compute_robustness(
            feat, pct, direction, r["bad_rate_diff"], all_results
        )

        # Temporal CV
        cv_key = (feat, pct, direction)
        cv = cv_map.get(cv_key)
        if cv is None:
            cv_weight = 0.6  # unknown CV, penalize moderately
        elif cv < 0.3:
            cv_weight = 1.0
        elif cv < 0.5:
            cv_weight = 0.8
        elif cv < 1.0:
            cv_weight = 0.5
        else:
            cv_weight = 0.3

        # Robustness weight
        if rob_score >= 0.8:
            rob_weight = 1.0
        elif rob_score >= 0.5:
            rob_weight = 0.7
        else:
            rob_weight = 0.4

        # Composite: robustness (40%) + CV (30%) + bad_rate_diff (30%)
        # Normalize bad_diff to [0, 1] range (cap at 20%)
        bad_norm = min(bad_diff / 0.20, 1.0)
        composite = rob_weight * 0.40 + cv_weight * 0.30 + bad_norm * 0.30

        # Build operator for rules: YAML
        cat = r["_category"]
        if cat == "positive":
            if direction == "high":
                op = ">="
                val = r["threshold"]
            else:
                op = "<="
                val = r["threshold"]
        else:  # absence: reject low end -> pass = feature > threshold
            op = ">"
            val = r["threshold"]

        # Tier classification
        if composite >= 0.80:
            tier = "T1"
            verdict = "✅ T1"
        elif composite >= 0.70:
            tier = "T2"
            verdict = "✅ T2"
        elif composite >= 0.50:
            tier = "T3"
            verdict = "❓ T3"
        else:
            tier = "-"
            verdict = "❌ 弱"

        # Coverage (single rule)
        coverage_pct = r["n_signal"] / n_dataset * 100 if n_dataset > 0 else 0.0

        scored.append(
            {
                "feature": feat,
                "percentile": pct,
                "direction": direction,
                "category": cat,
                "bad_rate_diff": r["bad_rate_diff"],
                "cv": cv,
                "robustness": round(rob_score, 2),
                "n_adjacent": n_adj,
                "cv_weight": round(cv_weight, 2),
                "rob_weight": round(rob_weight, 2),
                "composite": round(composite, 3),
                "tier": tier,
                "verdict": verdict,
                "operator": op,
                "value": round(val, 4),
                "threshold": r["threshold"],
                "n_signal": r["n_signal"],
                "coverage_pct": round(coverage_pct, 1),
            }
        )

    scored.sort(key=lambda x: x["composite"], reverse=True)

    # ── Print recommendation table ──
    print(f"\n{'='*130}")
    print("🏆 Prefilter 精选推荐 (综合评分 = 鲁棒性 40% + CV 30% + 信号强度 30%)")
    print(f"{'='*130}")
    print(f"  • 此表只含 prefilter 候选 (正信号 + 低端信号); 反信号留给 Gate 学习")
    print(f"  • 鲁棒性 = 相邻分位数方向一致性 (0~1); CV = 时间窗口稳定性")
    print(f"  • Tier: T1(>=0.80) T2(>=0.70) T3(>=0.50)")
    if n_dataset > 0:
        ratio_info = ""
        if n_gate_features and n_gate_features > 0:
            ratio_info = (
                f" | Gate 特征数: {n_gate_features}, 最低要求 sample:feat >= 20:1"
            )
        print(
            f"  • 数据集: {n_dataset:,} 行 | 训练最低要求: {MIN_TRAIN_SAMPLES:,} 行{ratio_info}"
        )
    print()

    print(
        f"  {'#':>3s} {'tier':>5s}  {'feature':<30s} {'cond':<20s} {'bad_diff':>8s} "
        f"{'CV':>6s} {'robust':>6s} {'评分':>6s} {'n':>6s} {'覆盖率':>6s}"
    )
    print(f"  {'-'*120}")

    for i, s in enumerate(scored[:30]):
        cv_str = f"{s['cv']:.2f}" if s["cv"] is not None else "  N/A"
        cond_str = f"{s['operator']} {s['value']:.4g}"
        cov_str = f"{s['coverage_pct']:5.1f}%"
        print(
            f"  {i+1:>3d} {s['verdict']:>5s}  {s['feature']:<30s} {cond_str:<20s} "
            f"{s['bad_rate_diff']:>+7.1%} {cv_str:>6s} {s['robustness']:>5.2f} "
            f"{s['composite']:>6.3f} {s['n_signal']:>6,d} {cov_str:>6s}"
        )
    if len(scored) > 30:
        print(f"  ... 剩余 {len(scored) - 30} 条省略")

    # ── AND 累积模拟 ──
    recommended = [s for s in scored if s["composite"] >= 0.5]
    if recommended and df is not None and n_dataset > 0:
        # Estimate train ratio from data (assume temporal split around ~44%)
        _time_col = _find_time_column(df)
        _train_ratio = 0.44  # default estimate
        if _time_col is not None:
            if _time_col == "__index__":
                _ts = df.index
            else:
                _ts = pd.to_datetime(df[_time_col], errors="coerce")
            if hasattr(_ts, "year"):
                # Use 2024-05-01 as typical split point
                _split_ts = pd.Timestamp("2024-05-01")
                if hasattr(_ts, "tz") and _ts.tz is not None:
                    _split_ts = _split_ts.tz_localize(_ts.tz)
                _n_before = int((_ts < _split_ts).sum())
                if _n_before > 0:
                    _train_ratio = _n_before / n_dataset
        _TRAIN_RATIO = _train_ratio

        print(f"\n{'='*130}")
        print("📊 AND 累积覆盖率模拟 (训练管线逐条 AND 过滤, 按评分排序)")
        print(f"{'='*130}")
        print(f"  ℹ️  模拟基于全量数据集 ({n_dataset:,} 行), 与训练管线一致")
        print(f"  ⚠️  P5 规则单条仅保留 ~5%, 多条 AND 会指数级衰减!")
        print(
            f"  📐 训练最低要求: {MIN_TRAIN_SAMPLES:,} 行 (Train 估算比例: {_TRAIN_RATIO:.0%})\n"
        )

        # Tier summary counts
        tier_counts = {}
        for s in recommended:
            tier_counts[s["tier"]] = tier_counts.get(s["tier"], 0) + 1
        tier_summary = ", ".join(f"{t}: {c}条" for t, c in sorted(tier_counts.items()))
        print(f"  推荐规则统计: {len(recommended)}条 ({tier_summary})")
        print()

        # Simulate AND
        df_sim = df.copy()
        baseline_bad = float((df_sim[label_col] == 0).mean())
        print(
            f"  {'#':>3s} {'tier':>4s} {'feature':<30s} {'cond':<20s} "
            f"{'\u5269\u4f59':>8s} {'\u8986\u76d6\u7387':>7s} {'bad_rate':>9s} {'vs\u57fa\u7ebf':>8s} {'train_est':>10s}  \u72b6\u6001"
        )
        print(f"  {'-'*130}")
        print(
            f"  {'':>3s} {'':>4s} {'(\u57fa\u7ebf)':<30s} {'':20s} "
            f"{n_dataset:>8,d} {'100.0%':>7s} {baseline_bad:>8.1%} {'':>8s} {int(n_dataset * _TRAIN_RATIO):>10,d}  \u2500"
        )

        hit_floor = False
        last_viable_idx = -1
        and_results = []  # track for practical recommendation
        n_skipped_redundant = 0
        for i, s in enumerate(recommended):
            feat = s["feature"]
            op_str = s["operator"]
            val = s["value"]
            op_func = _SIM_OPS.get(op_str)
            if op_func is None or feat not in df_sim.columns:
                continue
            n_before = len(df_sim)
            df_sim = df_sim[op_func(df_sim[feat], val)].copy()
            n_remain = len(df_sim)

            # Skip redundant rule (no change = fully subsumed by previous rules)
            if n_remain == n_before and i > 0:
                n_skipped_redundant += 1
                cond_str = f"{op_str} {val:.4g}"
                print(
                    f"  {i+1:>3d} {s['tier']:>4s} {feat:<30s} {cond_str:<20s} "
                    f"{'':>8s} {'':>7s} {'':>9s} {'':>8s}  ⚙️ 冗余(跳过)"
                )
                continue

            pct_remain = n_remain / n_dataset * 100
            bad_now = float((df_sim[label_col] == 0).mean()) if n_remain > 0 else 0.0
            bad_delta = bad_now - baseline_bad

            if n_remain >= MIN_TRAIN_SAMPLES:
                train_est = int(n_remain * _TRAIN_RATIO)
                if train_est < MIN_TRAIN_SAMPLES:
                    status = "⚠️ train不足"
                    # total OK but train split too small
                else:
                    status = "✅"
                    last_viable_idx = i
            elif n_remain > 0:
                status = "⚠️ 不足" if not hit_floor else "❌ 危险"
                hit_floor = True
            else:
                status = "❌ 清零"
                hit_floor = True

            cond_str = f"{op_str} {val:.4g}"
            train_est = int(n_remain * _TRAIN_RATIO)
            and_results.append(
                {
                    "idx": i + 1,
                    "feature": feat,
                    "tier": s["tier"],
                    "n_remain": n_remain,
                    "pct_remain": pct_remain,
                    "bad_rate": bad_now,
                    "bad_delta": bad_delta,
                    "status": status,
                    "cond_str": cond_str,
                    "train_est": train_est,
                }
            )
            print(
                f"  {i+1:>3d} {s['tier']:>4s} {feat:<30s} {cond_str:<20s} "
                f"{n_remain:>8,d} {pct_remain:>6.1f}% {bad_now:>8.1%} {bad_delta:>+7.1%} {train_est:>10,d}  {status}"
            )
            if n_remain == 0:
                print(f"  ... 后续规则跳过 (已清零)")
                break

        if n_skipped_redundant > 0:
            print(
                f"\n  ℹ️  {n_skipped_redundant} 条冗余规则已跳过 (AND后样本量不变 = 完全被前序规则包含)"
            )

        # ── 实用方案推荐 ──
        print(f"\n{'='*130}")
        print("💡 实用方案推荐")
        print(f"{'='*130}")

        if last_viable_idx >= 0:
            viable = and_results[last_viable_idx]
            viable_rules = recommended[: last_viable_idx + 1]
            print(f"\n  方案 A: 前 {last_viable_idx + 1} 条规则 AND (评分排序截断)")
            print(f"  ────────────────────────────────")
            print(f"    剩余样本: {viable['n_remain']:,} ({viable['pct_remain']:.1f}%)")
            print(
                f"    bad_rate: {viable['bad_rate']:.1%} (基线 {baseline_bad:.1%}, "
                f"差异 {viable['bad_delta']:+.1%})"
            )
            print(f"    规则:")
            for s in viable_rules:
                print(
                    f"      - {s['feature']} {s['operator']} {s['value']} [{s['tier']}, 评分={s['composite']:.3f}]"
                )
        else:
            print(
                f"\n  ⚠️  按评分排序逐条 AND, 任何单条规则都低于训练最低 {MIN_TRAIN_SAMPLES:,}"
            )
            print(f"  ⚠️  可能需要放宽阈值 (P5→P10/P20)")

        # 方案 B: 只用 T1 rules (if available)
        t1_rules = [s for s in recommended if s["tier"] == "T1"]
        if t1_rules and len(t1_rules) != (
            last_viable_idx + 1 if last_viable_idx >= 0 else 0
        ):
            df_t1 = df.copy()
            for s in t1_rules:
                op_func = _SIM_OPS.get(s["operator"])
                if op_func and s["feature"] in df_t1.columns:
                    df_t1 = df_t1[op_func(df_t1[s["feature"]], s["value"])].copy()
            n_t1 = len(df_t1)
            pct_t1 = n_t1 / n_dataset * 100
            bad_t1 = float((df_t1[label_col] == 0).mean()) if n_t1 > 0 else 0.0
            status_t1 = "✅" if n_t1 >= MIN_TRAIN_SAMPLES else "⚠️ 不足"
            print(f"\n  方案 B: 仅 T1 规则 AND ({len(t1_rules)} 条, composite >= 0.80)")
            print(f"  ────────────────────────────────")
            print(f"    剩余样本: {n_t1:,} ({pct_t1:.1f}%) {status_t1}")
            print(
                f"    bad_rate: {bad_t1:.1%} (基线 {baseline_bad:.1%}, "
                f"差异 {bad_t1 - baseline_bad:+.1%})"
            )
            print(f"    规则:")
            for s in t1_rules:
                print(
                    f"      - {s['feature']} {s['operator']} {s['value']} [{s['tier']}, 评分={s['composite']:.3f}]"
                )

        # 方案 C: 单条最强
        best = recommended[0]
        print(f"\n  方案 C: 单条最强规则")
        print(f"  ────────────────────────────────")
        print(f"    {best['feature']} {best['operator']} {best['value']}")
        print(
            f"    样本: {best['n_signal']:,} ({best['coverage_pct']:.1f}%), "
            f"bad_rate 差异: {best['bad_rate_diff']:+.1%}, 评分: {best['composite']:.3f}"
        )

        # 方案 D: 宽松阈值探索 (找同特征 P20 替代 P5)
        _relaxed_candidates = []
        _relaxed_feats_seen = set()  # dedup for _left/_right variants
        for s in recommended[:5]:  # top 5 by composite
            feat = s["feature"]
            pct_num = int(s["percentile"].replace("P", ""))
            if pct_num > 20:
                continue  # already wide enough
            # Skip redundant _left/_right variant if base feature already included
            base_feat = feat.replace("_left", "").replace("_right", "")
            if base_feat in _relaxed_feats_seen:
                continue
            _relaxed_feats_seen.add(base_feat)
            # Find P20 for same feature, same direction
            for r in all_results:
                if r["feature"] == feat and r.get("direction") == s["direction"]:
                    r_pct_num = int(r["percentile"].replace("P", ""))
                    if r_pct_num == 20:
                        _relaxed_candidates.append(
                            {
                                "feature": feat,
                                "original_pct": s["percentile"],
                                "relaxed_pct": "P20",
                                "operator": s["operator"],
                                "value": round(r["threshold"], 4),
                                "n_signal": r["n_signal"],
                                "bad_rate_diff": r["bad_rate_diff"],
                            }
                        )
                        break
        if _relaxed_candidates:
            # Simulate AND with relaxed
            df_relax = df.copy()
            for rc in _relaxed_candidates:
                op_func = _SIM_OPS.get(rc["operator"])
                if op_func and rc["feature"] in df_relax.columns:
                    df_relax = df_relax[
                        op_func(df_relax[rc["feature"]], rc["value"])
                    ].copy()
            n_relax = len(df_relax)
            pct_relax = n_relax / n_dataset * 100
            bad_relax = float((df_relax[label_col] == 0).mean()) if n_relax > 0 else 0.0
            status_relax = "✅" if n_relax >= MIN_TRAIN_SAMPLES else "⚠️ 不足"
            print(f"\n  方案 D: Top-{len(_relaxed_candidates)} 放宽至 P20 AND")
            print(f"  ────────────────────────────────")
            print(f"    剩余样本: {n_relax:,} ({pct_relax:.1f}%) {status_relax}")
            print(
                f"    bad_rate: {bad_relax:.1%} (基线 {baseline_bad:.1%}, "
                f"差异 {bad_relax - baseline_bad:+.1%})"
            )
            print(f"    规则:")
            for rc in _relaxed_candidates:
                print(
                    f"      - {rc['feature']} {rc['operator']} {rc['value']} "
                    f"[{rc['original_pct']}→{rc['relaxed_pct']}, bad_diff={rc['bad_rate_diff']:+.1%}]"
                )

        print()

        # ── 方案 E: 语义 OR 组合 (自动检测 longs/shorts 互补对) ──
        import re as _re

        _OR_PAIR_PATTERNS = [
            (_re.compile(r"^(.+)_longs_(.+)$"), "_shorts_"),
            (_re.compile(r"^(.+)_long_(.+)$"), "_short_"),
        ]
        # Build map: feature -> signals from ALL categories (positive + anti + absence)
        # trapped scores may be absence/anti signals, not positive
        _sig_map: Dict[str, List[Dict]] = {}
        for r in positive:
            _sig_map.setdefault(r["feature"], []).append(r)
        for r in anti:
            _sig_map.setdefault(r["feature"], []).append(r)
        for r in absence:
            _sig_map.setdefault(r["feature"], []).append(r)

        or_pairs_found = []
        _seen_pairs: set = set()
        for feat_a in _sig_map:
            for pat, replace_part in _OR_PAIR_PATTERNS:
                m = pat.match(feat_a)
                if m:
                    prefix, suffix = m.groups()
                    feat_b = f"{prefix}{replace_part}{suffix}"
                    pair_key = tuple(sorted([feat_a, feat_b]))
                    if pair_key in _seen_pairs:
                        continue
                    if feat_b in _sig_map or feat_b in [
                        r["feature"]
                        for r in all_results
                        if abs(r.get("bad_rate_diff", 0)) > 0.02
                    ]:
                        _seen_pairs.add(pair_key)
                        or_pairs_found.append((feat_a, feat_b))

        if or_pairs_found and df is not None and n_dataset > 0:
            print(f"{'='*130}")
            print("🔀 方案 E: 语义 OR 组合 (自动检测的互补特征对)")
            print(f"{'='*130}")
            print(f"  ℹ️  OR 逻辑: 满足任一即通过 (不是 AND 全部满足)")
            print(
                f"  💡 适用场景: 互补语义 (如 longs+shorts 被套分数), 且 AND 样本量不足\n"
            )

            best_or_pair = None
            for feat_a, feat_b in or_pairs_found:
                if feat_a not in df.columns or feat_b not in df.columns:
                    continue
                # Collect "high" direction entries for each feature at various percentiles.
                # OR pairs (longs/shorts) = domain presence: HIGH value = concept present.
                # Use relaxed threshold for OR pairs — we care about domain presence, not signal strength.
                entries_a_high = [
                    r
                    for r in all_results
                    if r["feature"] == feat_a and r["direction"] == "high"
                ]
                entries_a_low = [
                    r
                    for r in all_results
                    if r["feature"] == feat_a
                    and r["direction"] == "low"
                    and abs(r.get("bad_rate_diff", 0)) > 0.02
                ]
                entries_b_high = [
                    r
                    for r in all_results
                    if r["feature"] == feat_b and r["direction"] == "high"
                ]
                entries_b_low = [
                    r
                    for r in all_results
                    if r["feature"] == feat_b
                    and r["direction"] == "low"
                    and abs(r.get("bad_rate_diff", 0)) > 0.02
                ]

                # Prefer high direction (domain presence); fallback to any
                pool_a = entries_a_high if entries_a_high else entries_a_low
                pool_b = entries_b_high if entries_b_high else entries_b_low
                if not pool_a:
                    continue

                # Determine operator and threshold
                def _or_rule(r):
                    if r["direction"] == "high":
                        return ">=", r["threshold"]
                    else:
                        return "<=", r["threshold"]

                # Try multiple percentile combinations and report each
                tested_combos = []
                for r_a in pool_a:
                    op_a, val_a = _or_rule(r_a)
                    op_func_a = _SIM_OPS.get(op_a)
                    mask_a = (
                        op_func_a(df[feat_a], val_a)
                        if op_func_a
                        else pd.Series(False, index=df.index)
                    )
                    if pool_b:
                        for r_b in pool_b:
                            op_b, val_b = _or_rule(r_b)
                            op_func_b = _SIM_OPS.get(op_b)
                            mask_b = (
                                op_func_b(df[feat_b], val_b)
                                if op_func_b
                                else pd.Series(False, index=df.index)
                            )
                            or_mask = mask_a | mask_b
                            tested_combos.append(
                                {
                                    "r_a": r_a,
                                    "r_b": r_b,
                                    "op_a": op_a,
                                    "val_a": val_a,
                                    "op_b": op_b,
                                    "val_b": val_b,
                                    "or_mask": or_mask,
                                    "desc": f"{feat_a} {op_a} {val_a:.4f} OR {feat_b} {op_b} {val_b:.4f}",
                                    "pct_info": f"{r_a['percentile']}+{r_b['percentile']}",
                                }
                            )
                    else:
                        or_mask = mask_a
                        tested_combos.append(
                            {
                                "r_a": r_a,
                                "r_b": None,
                                "op_a": op_a,
                                "val_a": val_a,
                                "op_b": None,
                                "val_b": None,
                                "or_mask": or_mask,
                                "desc": f"{feat_a} {op_a} {val_a:.4f} (single)",
                                "pct_info": r_a["percentile"],
                            }
                        )

                # Deduplicate by n_or (same threshold combo may repeat)
                seen_n = set()
                unique_combos = []
                for c in tested_combos:
                    n = int(c["or_mask"].sum())
                    if n not in seen_n:
                        seen_n.add(n)
                        unique_combos.append(c)
                # Sort by train_est descending (prefer more training samples)
                unique_combos.sort(key=lambda c: int(c["or_mask"].sum()), reverse=True)

                for combo in unique_combos[:5]:  # show top 5 variants
                    or_mask = combo["or_mask"]
                    n_or = int(or_mask.sum())
                    pct_or = n_or / n_dataset * 100
                    bad_or = (
                        float((df.loc[or_mask, label_col] == 0).mean())
                        if n_or > 0
                        else 0.0
                    )
                    train_est_or = int(n_or * _TRAIN_RATIO)

                    ratio_str = ""
                    ratio_ok = True
                    if n_gate_features and n_gate_features > 0:
                        ratio = train_est_or / n_gate_features
                        ratio_str = f", ratio={ratio:.1f}:1"
                        ratio_ok = ratio >= 20

                    if n_or >= MIN_TRAIN_SAMPLES and train_est_or >= MIN_TRAIN_SAMPLES:
                        status_or = "✅" if ratio_ok else "⚠️ ratio低"
                    else:
                        status_or = "⚠️ 不足"

                    print(f"  🔀 {combo['desc']} [{combo['pct_info']}]")
                    print(
                        f"     剩余: {n_or:,} ({pct_or:.1f}%), train_est: {train_est_or:,}{ratio_str}"
                    )
                    print(
                        f"     bad_rate: {bad_or:.1%} (vs 基线 {baseline_bad:.1%}, 差异 {bad_or - baseline_bad:+.1%})  {status_or}"
                    )

                    # Track best viable OR pair
                    if best_or_pair is None or train_est_or > best_or_pair["train_est"]:
                        or_sub_rules = [
                            {
                                "feature": feat_a,
                                "operator": combo["op_a"],
                                "value": round(combo["val_a"], 4),
                                "percentile": combo["r_a"]["percentile"],
                            }
                        ]
                        if combo["r_b"]:
                            or_sub_rules.append(
                                {
                                    "feature": feat_b,
                                    "operator": combo["op_b"],
                                    "value": round(combo["val_b"], 4),
                                    "percentile": combo["r_b"]["percentile"],
                                }
                            )
                        best_or_pair = {
                            "n_or": n_or,
                            "pct_or": pct_or,
                            "bad_or": bad_or,
                            "train_est": train_est_or,
                            "status": status_or,
                            "sub_rules": or_sub_rules,
                            "rationale": f"semantic OR: {feat_a} OR {feat_b}",
                        }

            # If best AND is insufficient but OR is sufficient, highlight
            if best_or_pair:
                _and_best_train = (
                    and_results[last_viable_idx]["train_est"]
                    if last_viable_idx >= 0 and and_results
                    else 0
                )
                _or_train = best_or_pair["train_est"]
                if _or_train > _and_best_train:
                    print(
                        f"\n  🎯 推荐: OR 方案优于 AND (训练样本 {_or_train:,} vs {_and_best_train:,})"
                    )
                    print(
                        f"     语义让计: prefilter 应定义 archetype 因果前提，不要贪滤; 细粒度留给 Gate"
                    )
            print()
        elif not or_pairs_found:
            # No pairs detected, just end
            print()
    if recommended:
        print(f"\n{'─'*130}")
        print(f"📝 推荐 rules: YAML (将以下内容复制到 archetypes/prefilter.yaml)")
        print(f"{'─'*130}")
        print(f"rules:")
        for s in recommended:
            rationale_parts = [f"{s['percentile']}阈值"]
            if s["cv"] is not None:
                rationale_parts.append(f"CV={s['cv']:.2f}")
            rationale_parts.append(f"bad_rate {s['bad_rate_diff']:+.1%}")
            rationale_parts.append(f"鲁棒性={s['robustness']:.2f}")
            rationale_parts.append(f"评分={s['composite']:.3f}")
            rationale_parts.append(f"覆盖率={s['coverage_pct']:.1f}%")
            rationale = ", ".join(rationale_parts)
            print(f'  - feature: {s["feature"]}')
            print(f'    operator: "{s["operator"]}"')
            print(f'    value: {s["value"]}')
            print(f'    rationale: "{rationale}"')
        print()
        print(
            f"  ℹ️  {len(recommended)} 条推荐 (composite >= 0.5), "
            f"human review 后复制到 config/strategies/{strategy}/archetypes/prefilter.yaml"
        )
        print(f"  ⚠️  切勿全部 AND! 请参考上方「实用方案推荐」选择合适的子集")

        # Also output any_of YAML for detected OR pairs
        _best_or = locals().get("best_or_pair")
        if _best_or and _best_or.get("sub_rules"):
            print(f"\n{'─'*130}")
            print(
                f"🔀 推荐 OR rules: YAML (any_of 语法, 直接可用于 archetypes/prefilter.yaml)"
            )
            print(f"{'─'*130}")
            print(f"rules:")
            print(f"  - any_of:")
            for sub in _best_or["sub_rules"]:
                print(f'      - feature: {sub["feature"]}')
                print(f'        operator: "{sub["operator"]}"')
                print(f'        value: {sub["value"]}')
            print(
                f'    rationale: "{_best_or["rationale"]}, '
                f'train_est={_best_or["train_est"]:,}, '
                f'bad_rate={_best_or["bad_or"]:.1%}"'
            )
            print()
    else:
        print(f"\n  ⚠️  无足够强的 prefilter 候选 (composite < 0.5)")

    # ── 阈值放宽工具: 特征有效但百分位过严时, 回退到更宽松的百分位 ──
    # 例: atr_percentile P95(通过率5%) 超过 min_pass_rate=10% → 回退到 P90(通过率10%)
    _RELAX_ORDER_HIGH = ["P90", "P80"]  # 高端特征: P95→P90→P80
    _RELAX_ORDER_LOW = ["P10", "P20"]  # 低端特征: P5→P10→P20

    def _find_relaxed_threshold(
        feat: str,
        direction: str,
        current_pct: str,
        all_results: List[Dict[str, Any]],
        sim_df: pd.DataFrame,
        n_dataset: int,
        min_pass_rate: float,
    ) -> Optional[Dict[str, Any]]:
        """Find a more relaxed percentile for the same feature that satisfies min_pass_rate."""
        relax_order = _RELAX_ORDER_HIGH if direction == "high" else _RELAX_ORDER_LOW
        # 只考虑比当前百分位更宽松的
        try:
            cur_num = int(current_pct.replace("P", ""))
        except (ValueError, AttributeError):
            return None
        for r in all_results:
            if r["feature"] != feat or r.get("direction") != direction:
                continue
            r_pct = r.get("percentile", "")
            if r_pct not in relax_order:
                continue
            try:
                r_num = int(r_pct.replace("P", ""))
            except (ValueError, AttributeError):
                continue
            # 对于 high 端: 更宽松 = 数值更小 (P95→P90→P80)
            # 对于 low 端: 更宽松 = 数值更大 (P5→P10→P20)
            if direction == "high" and r_num >= cur_num:
                continue
            if direction == "low" and r_num <= cur_num:
                continue
            # 检查通过率
            _op = ">=" if direction == "high" else "<="
            _op_fn = _SIM_OPS.get(_op)
            if _op_fn is None or feat not in sim_df.columns:
                continue
            _trial = sim_df[_op_fn(sim_df[feat], r["threshold"])]
            _cov = len(_trial) / n_dataset if n_dataset > 0 else 0
            if _cov >= min_pass_rate:
                return {
                    "threshold": r["threshold"],
                    "percentile": r_pct,
                    "direction": direction,
                }
        return None

    # Build top_rules for auto-generation (with rationale)
    # ── CRITICAL: simulate cumulative AND to avoid over-filtering ──
    # Guards:
    #   1. Absolute row minimum: 累积 AND 后至少保留 MIN_PREFILTER_ROWS 行
    #      (Gate/Evidence/Entry plateau 检测最低需 ~500 行即可)
    #   2. Actual train count (rows < split) >= MIN_TRAIN_SAMPLES
    #      (虽然模型用全量训练, 但 prefilter 后的子集需要足够 pre-split 数据做 plateau 验证)
    #
    # 为什么不用百分比?
    #   模型训练已用全量数据, prefilter 只影响 Gate Optimize 的 plateau 检测.
    #   Plateau 需要的是绝对样本量(~500), 不是比例.
    #   P5/P10 级特征虽然只选 5-10% 数据, 但信号更强, 应该允许选用.
    MIN_PREFILTER_ROWS = (
        min_prefilter_rows if min_prefilter_rows is not None else 500
    )  # plateau 检测的绝对最低行数
    top_rules = []
    _skip_log = []  # 记录跳过原因，最后汇总输出
    if df is not None and n_dataset > 0:
        _sim_df = df.copy()
        _train_ratio = locals().get("_TRAIN_RATIO", 0.44)
        # Resolve time column + split for actual train count
        _time_col_tr = _find_time_column(df)
        _split_ts = pd.Timestamp("2024-05-01")
        if _time_col_tr:
            _ts_all = _get_times(df, _time_col_tr)
            if hasattr(_ts_all, "tz") and _ts_all.dt.tz is not None:
                _split_ts = _split_ts.tz_localize(_ts_all.dt.tz)
        else:
            _ts_all = None
        _n_scanned = 0
        for s in recommended[:5]:  # scan up to 5 candidates
            if len(top_rules) >= 3:
                break
            _n_scanned += 1
            _feat = s["feature"]
            _op_str = s["operator"]
            _val = s["value"]
            _pct = s.get("percentile", "?")
            _op_func = _SIM_OPS.get(_op_str)
            if _op_func is None or _feat not in _sim_df.columns:
                continue
            _trial = _sim_df[_op_func(_sim_df[_feat], _val)]

            # Guard 1: absolute row minimum (plateau 检测需要足够样本)
            _coverage = len(_trial) / n_dataset if n_dataset > 0 else 0
            _is_cumulative = len(_sim_df) < n_dataset  # 前面已有规则通过

            # Guard 1a: min_pass_rate (通过率下限, 防止 prefilter 过严导致交易太少)
            if (
                min_prefilter_pass_rate is not None
                and _coverage < min_prefilter_pass_rate
            ):
                # ── 阈值放宽: 特征有效但阈值过严时, 自动回退到更宽松的百分位 ──
                _relaxed = _find_relaxed_threshold(
                    _feat,
                    s.get("direction", ""),
                    s.get("percentile", ""),
                    all_results,
                    _sim_df,
                    n_dataset,
                    min_prefilter_pass_rate,
                )
                if _relaxed is not None:
                    _old_pct = _pct
                    _val = _relaxed["threshold"]
                    _pct = _relaxed["percentile"]
                    # 重新计算 operator
                    if _relaxed["direction"] == "high":
                        _op_str = ">="
                    else:
                        _op_str = "<="
                    _op_func = _SIM_OPS.get(_op_str)
                    _trial = _sim_df[_op_func(_sim_df[_feat], _val)]
                    _coverage = len(_trial) / n_dataset if n_dataset > 0 else 0
                    # 更新 scored entry
                    s = {
                        **s,
                        "value": round(_val, 4),
                        "operator": _op_str,
                        "percentile": _pct,
                    }
                    print(
                        f"  🔄 top_rules 放宽 {_feat}: {_old_pct}→{_pct} "
                        f"(阈值={_val:.4f}, 通过率={_coverage:.1%})"
                    )
                else:
                    _reason = (
                        f"通过率不足: "
                        f"{'AND 后' if _is_cumulative else ''}通过率 {_coverage:.1%} < "
                        f"min_pass_rate {min_prefilter_pass_rate:.0%} (kpi_gates 约束), "
                        f"且无更宽松百分位可用"
                    )
                    _skip_log.append(
                        {
                            "rule": f"{_feat} {_op_str} {_val}",
                            "percentile": _pct,
                            "reason": _reason,
                            "guard": "pass_rate",
                        }
                    )
                    print(
                        f"  ⛔ top_rules 跳过 {_feat} {_op_str} {_val} ({_pct}): {_reason}"
                    )
                    continue

            # Guard 1b: absolute row minimum
            if len(_trial) < MIN_PREFILTER_ROWS:
                if _is_cumulative:
                    _reason = (
                        f"累积 AND 后行数不足: "
                        f"前序规则已筛至 {len(_sim_df):,} 行, "
                        f"再加此规则({_pct}阈值)仅剩 {len(_trial):,} 行 < {MIN_PREFILTER_ROWS} (plateau 最低需求)"
                    )
                else:
                    _reason = (
                        f"规则本身行数不足: "
                        f"{_pct}阈值仅选中 {len(_trial):,} 行 < {MIN_PREFILTER_ROWS} (plateau 最低需求)"
                    )
                _skip_log.append(
                    {
                        "rule": f"{_feat} {_op_str} {_val}",
                        "percentile": _pct,
                        "reason": _reason,
                        "guard": "rows",
                    }
                )
                print(
                    f"  ⛔ top_rules 跳过 {_feat} {_op_str} {_val} ({_pct}): {_reason}"
                )
                continue

            # Guard 2: actual train count (rows before split)
            if _ts_all is not None:
                _trial_ts = _ts_all.loc[_trial.index]
                _actual_train = int((_trial_ts < _split_ts).sum())
            else:
                _actual_train = int(len(_trial) * _train_ratio)
            if _actual_train < MIN_TRAIN_SAMPLES:
                _reason = (
                    f"split前样本不足: "
                    f"总行数={len(_trial):,}, 但 split 前仅 {_actual_train:,} 行 < {MIN_TRAIN_SAMPLES:,} (plateau时序验证要求)"
                )
                _skip_log.append(
                    {
                        "rule": f"{_feat} {_op_str} {_val}",
                        "percentile": _pct,
                        "reason": _reason,
                        "guard": "train_samples",
                    }
                )
                print(
                    f"  ⛔ top_rules 跳过 {_feat} {_op_str} {_val} ({_pct}): {_reason}"
                )
                continue
            # ✅ 通过所有护栏
            print(
                f"  ✅ top_rules 接受 {_feat} {_op_str} {_val} ({_pct}): "
                f"{len(_trial):,} 行 ({_coverage:.1%}), train={_actual_train if _ts_all is not None else '~' + str(_actual_train):,}"
            )
            _sim_df = _trial.copy()
            rationale_parts = [f"{s['percentile']}阈值"]
            if s["cv"] is not None:
                rationale_parts.append(f"CV={s['cv']:.2f}")
            rationale_parts.append(f"bad_rate {s['bad_rate_diff']:+.1%}")
            rationale_parts.append(f"鲁棒性={s['robustness']:.2f}")
            rationale_parts.append(f"评分={s['composite']:.3f}")
            top_rules.append(
                {
                    "feature": s["feature"],
                    "operator": s["operator"],
                    "value": s["value"],
                    "composite": s["composite"],
                    "rationale": ", ".join(rationale_parts),
                }
            )
    else:
        # Fallback: no df available, use top-3 without simulation
        for s in recommended[:3]:
            rationale_parts = [f"{s['percentile']}阈值"]
            if s["cv"] is not None:
                rationale_parts.append(f"CV={s['cv']:.2f}")
            rationale_parts.append(f"bad_rate {s['bad_rate_diff']:+.1%}")
            rationale_parts.append(f"鲁棒性={s['robustness']:.2f}")
            rationale_parts.append(f"评分={s['composite']:.3f}")
            top_rules.append(
                {
                    "feature": s["feature"],
                    "operator": s["operator"],
                    "value": s["value"],
                    "composite": s["composite"],
                    "rationale": ", ".join(rationale_parts),
                }
            )

    # ── top_rules 选择汇总 ──
    _n_total_scanned = locals().get("_n_scanned", len(recommended[:5]))
    if top_rules:
        print(
            f"\n  ✅ top_rules for --promote: {len(top_rules)} 条通过 "
            f"(扫描 {_n_total_scanned} 个候选, 跳过 {len(_skip_log)} 个)"
        )
        print(
            f"     护栏: 绝对行数>={MIN_PREFILTER_ROWS}, split前train>={MIN_TRAIN_SAMPLES:,}"
            + (
                f", 通过率>={min_prefilter_pass_rate:.0%}"
                if min_prefilter_pass_rate
                else ""
            )
        )
    else:
        print(
            f"\n  ⚠️  top_rules 为空: 扫描了 {_n_total_scanned} 个候选, 全部未通过安全护栏"
        )
        print(
            f"     护栏: 绝对行数>={MIN_PREFILTER_ROWS} (plateau最低需求), split前train>={MIN_TRAIN_SAMPLES:,}"
            + (
                f", 通过率>={min_prefilter_pass_rate:.0%}"
                if min_prefilter_pass_rate
                else ""
            )
        )
        if _skip_log:
            _rows_skip = sum(1 for x in _skip_log if x["guard"] == "rows")
            _train_skip = sum(1 for x in _skip_log if x["guard"] == "train_samples")
            _rate_skip = sum(1 for x in _skip_log if x["guard"] == "pass_rate")
            if _rate_skip:
                print(
                    f"     → {_rate_skip} 个因通过率<{min_prefilter_pass_rate:.0%} 跳过"
                )
            if _rows_skip:
                print(f"     → {_rows_skip} 个因行数<{MIN_PREFILTER_ROWS} 跳过")
            if _train_skip:
                print(f"     → {_train_skip} 个因训练样本<{MIN_TRAIN_SAMPLES:,} 跳过")
            print(
                f"     💡 提示: 可增加扫描范围(当前仅扫描 top-5), 或调整 kpi_gates.prefilter.min_pass_rate"
            )

    _best_or = locals().get("best_or_pair")

    return {
        "scored_candidates": scored,
        "recommended_rules": recommended,
        "or_pair": _best_or if _best_or and _best_or.get("sub_rules") else None,
        "top_rules": top_rules,
    }


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    """找到 DataFrame 中的时间列。"""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.name:
        return "__index__"
    for col in ["timestamp", "date", "datetime", "time", "ts"]:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # 尝试解析 index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            pd.to_datetime(df.index[:10])
            return "__index__"
        except (ValueError, TypeError):
            pass
    return None


def _get_times(df: pd.DataFrame, time_col: str) -> pd.Series:
    """获取时间序列。"""
    if time_col == "__index__":
        return pd.to_datetime(df.index)
    return pd.to_datetime(df[time_col])


def _temporal_stability_analysis(
    df: pd.DataFrame,
    significant_results: List[Dict[str, Any]],
    label_col: str,
    time_col: str,
) -> Dict[str, Any]:
    """
    对每个显著特征×阈值，计算不同窗口下的 rolling bad_rate_diff + CV。

    输出:
    - 每个窗口大小的平均 CV
    - 最优窗口选择
    - 每个特征的时间曲线 + 稳定性判定
    """
    times = _get_times(df, time_col)
    t_min, t_max = times.min(), times.max()
    total_months = (t_max.year - t_min.year) * 12 + (t_max.month - t_min.month)

    print(f"\n{'='*110}")
    print(f"🕰️  时间稳定性分析 (--temporal)")
    print(
        f"   时间范围: {t_min.strftime('%Y-%m')} → {t_max.strftime('%Y-%m')}, 共 {total_months} 个月"
    )

    # 去重 feature × percentile × direction
    combos = []
    seen = set()
    for r in significant_results:
        key = (r["feature"], r["percentile"], r["direction"])
        if key not in seen:
            seen.add(key)
            combos.append(r)

    if not combos:
        print("   ⚠️  无显著特征可分析")
        return {}

    print(f"   分析 {len(combos)} 个显著特征×阈值组合")
    print(f"   候选窗口: {', '.join(f'{w}m' for w in TEMPORAL_WINDOW_MONTHS)}")

    # 对每个窗口大小，对每个 combo 计算 rolling bad_rate_diff
    window_results: Dict[int, List[Dict]] = {}

    for wm in TEMPORAL_WINDOW_MONTHS:
        window_results[wm] = []

        # 生成窗口 (步长 1 个月)
        window_start = t_min
        windows = []
        while True:
            window_end = window_start + pd.DateOffset(months=wm)
            if window_end > t_max + pd.Timedelta(days=1):
                break
            windows.append((window_start, window_end))
            window_start = window_start + pd.DateOffset(months=1)

        if len(windows) < 3:
            continue

        for combo in combos:
            feat = combo["feature"]
            threshold = combo["threshold"]
            direction = combo["direction"]

            diffs = []
            window_details = []

            for w_start, w_end in windows:
                mask = (times >= w_start) & (times < w_end)
                w_df = df.loc[mask.values]

                if len(w_df) < TEMPORAL_MIN_SAMPLES_PER_WINDOW:
                    continue

                # 用全周期阈值（不是 per-window 阈值，保持可比性）
                if direction == "high":
                    signal_mask = w_df[feat] >= threshold
                else:
                    signal_mask = w_df[feat] <= threshold

                signal_df = w_df.loc[signal_mask]
                rest_df = w_df.loc[~signal_mask]

                if len(signal_df) < 30 or len(rest_df) < 30:
                    continue

                br_signal = (signal_df[label_col] == 0).mean()
                br_rest = (rest_df[label_col] == 0).mean()
                diff = br_signal - br_rest

                diffs.append(diff)
                window_details.append(
                    {
                        "period": f"{w_start.strftime('%Y-%m')}→{w_end.strftime('%Y-%m')}",
                        "n": len(w_df),
                        "bad_rate_diff": round(diff, 4),
                    }
                )

            if len(diffs) < 3:
                continue

            diffs_arr = np.array(diffs)
            mean_diff = float(np.mean(diffs_arr))
            std_diff = float(np.std(diffs_arr))
            cv = abs(std_diff / mean_diff) if abs(mean_diff) > 1e-6 else float("inf")

            window_results[wm].append(
                {
                    "feature": feat,
                    "percentile": combo["percentile"],
                    "direction": direction,
                    "threshold": threshold,
                    "full_period_diff": combo["bad_rate_diff"],
                    "mean_diff": round(mean_diff, 4),
                    "std_diff": round(std_diff, 4),
                    "cv": round(cv, 2),
                    "n_windows": len(diffs),
                    "latest_diff": round(diffs[-1], 4),
                    "windows": window_details,
                }
            )

    # 找最优窗口 (平均 CV 最小)
    best_window = None
    best_avg_cv = float("inf")
    window_summary: Dict[int, Dict] = {}

    for wm, results in window_results.items():
        if not results:
            continue
        cvs = [r["cv"] for r in results if r["cv"] < float("inf")]
        if not cvs:
            continue
        avg_cv = float(np.mean(cvs))
        window_summary[wm] = {"avg_cv": round(avg_cv, 2), "n_features": len(results)}
        if avg_cv < best_avg_cv:
            best_avg_cv = avg_cv
            best_window = wm

    # 输出窗口对比
    print(f"\n   窗口对比:")
    for wm in sorted(window_summary.keys()):
        ws = window_summary[wm]
        marker = " ← 最优" if wm == best_window else ""
        print(
            f"     {wm}m: avg CV={ws['avg_cv']:.2f}, {ws['n_features']} 个特征{marker}"
        )

    if best_window is None:
        print("   ❌ 无有效窗口 (每个窗口最少需 1080 样本)")
        return {}

    print(f"\n   ✅ 最优窗口: {best_window} 个月 (avg CV={best_avg_cv:.2f})")

    # 详细表格
    best_results = window_results[best_window]
    best_results.sort(key=lambda x: x["cv"])

    print(f"\n{'─'*110}")
    print(
        f"{'  特征 × 阈值':<40s} {'全周期':>8s} {'最近窗口':>8s} "
        f"{'CV':>8s} {'判定':>8s}"
    )
    print(f"{'─'*110}")

    for r in best_results:
        op_str = ">=" if r["direction"] == "high" else "<="
        feat_str = f"{r['feature']} {r['percentile']}({op_str}{r['threshold']:.3f})"
        full_str = f"{r['full_period_diff']:+.1%}"
        latest_str = f"{r['latest_diff']:+.1%}"
        cv_str = f"{r['cv']:.2f}"

        if r["cv"] < 0.3:
            verdict = "✅ 稳定"
        elif r["cv"] < 0.5:
            verdict = "⚠️  一般"
        else:
            verdict = "❌ 不稳"

        print(
            f"  {feat_str:<38s} {full_str:>8s} {latest_str:>8s} {cv_str:>8s} {verdict}"
        )

    # Rolling 曲线 (top 5)
    print(f"\n📈 Rolling bad_rate_diff 曲线 ({best_window}m 窗口, 前 5 个特征):")
    for r in best_results[:5]:
        op_str = ">=" if r["direction"] == "high" else "<="
        print(
            f"\n  {r['feature']} {r['percentile']}"
            f"({op_str}{r['threshold']:.3f}) [CV={r['cv']:.2f}]:"
        )
        for w in r["windows"]:
            diff = w["bad_rate_diff"]
            bar_len = int(abs(diff) * 200)  # 200x 放大
            bar = "█" * min(bar_len, 30)
            sign_indicator = "-" if diff < 0 else "+"
            print(f"    {w['period']}: {diff:+.1%} {sign_indicator}{bar}")

    return {
        "best_window_months": best_window,
        "best_avg_cv": round(best_avg_cv, 2),
        "window_summary": {str(k): v for k, v in window_summary.items()},
        "feature_stability": best_results,
    }


# ── 自动回写 last_evaluation ─────────────────────────────────────


def _write_prefilter_evaluation(
    config_path: str,
    positive: List[Dict],
    anti: List[Dict],
    absence: List[Dict],
    temporal_report: Optional[Dict] = None,
    data_source: str = "",
    n_rows: int = 0,
    baseline_bad_rate: float = 0.0,
    baseline_median_rr: float = 0.0,
) -> None:
    """回写分析结果到 prefilter.yaml 的 last_evaluation 段。

    保留 candidates 段和注释，仅替换 last_evaluation 段。
    """
    path = Path(config_path)
    if not path.exists():
        print(f"\n⚠️  {path} 不存在, 跳过回写")
        return

    # Build temporal CV map: (feature, percentile, direction) -> cv
    temporal_cv_map: Dict[tuple, float] = {}
    if temporal_report and "feature_stability" in temporal_report:
        for tr in temporal_report["feature_stability"]:
            key = (tr["feature"], tr.get("percentile", ""), tr.get("direction", ""))
            temporal_cv_map[key] = tr.get("cv", None)

    lines: list[str] = []
    lines.append("last_evaluation:")
    lines.append(f"  # ── 自动生成 ({date.today()}) ──")
    lines.append(f'  timestamp: "{date.today()}"')
    lines.append(f'  data_source: "{data_source}"')
    lines.append(f"  n_rows: {n_rows}")
    lines.append(f"  baseline_bad_rate: {baseline_bad_rate:.4f}")
    lines.append(f"  baseline_median_rr: {baseline_median_rr:+.4f}")
    lines.append("")

    def _fmt_signals(signals: List[Dict], max_items: int = 10) -> None:
        if not signals:
            lines.append("    []")
            return
        for s in signals[:max_items]:
            lines.append(f"    - feature: {s['feature']}")
            lines.append(f"      percentile: {s['percentile']}")
            lines.append(f"      direction: {s['direction']}")
            lines.append(f"      threshold: {s['threshold']}")
            lines.append(f"      bad_rate_signal: {s['bad_rate_signal']:.4f}")
            lines.append(f"      bad_rate_rest: {s['bad_rate_rest']:.4f}")
            lines.append(f"      bad_rate_diff: {s['bad_rate_diff']:+.4f}")
            key = (s["feature"], s.get("percentile", ""), s.get("direction", ""))
            cv = temporal_cv_map.get(key)
            if cv is not None:
                lines.append(f"      temporal_cv: {cv}")

    lines.append(f"  # ── 正信号 ({len(positive)} 个) ──")
    lines.append("  positive_signals:")
    _fmt_signals(positive)
    lines.append("")

    lines.append(f"  # ── 反信号 ({len(anti)} 个) ──")
    lines.append("  anti_signals:")
    _fmt_signals(anti)
    lines.append("")

    lines.append(f"  # ── 低端信号 ({len(absence)} 个) ──")
    lines.append("  absence_signals:")
    _fmt_signals(absence)
    lines.append("")

    eval_text = "\n".join(lines) + "\n"

    # Read & replace
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    marker = "\nlast_evaluation:"
    idx = content.find(marker)
    if idx >= 0:
        new_content = content[: idx + 1] + eval_text
    else:
        new_content = content.rstrip() + "\n\n" + eval_text

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    n_total = len(positive) + len(anti) + len(absence)
    print(f"\n💾 已回写 last_evaluation → {path}")
    print(f"   正信号: {len(positive)}, 反信号: {len(anti)}, 低端: {len(absence)}")


def _generate_promoted_prefilter(
    output_path: Path,
    recommendation_report: Optional[Dict],
    strategy: str,
    positive: List[Dict],
    anti: List[Dict],
    absence: List[Dict],
    temporal_report: Optional[Dict] = None,
    data_source: str = "",
    n_rows: int = 0,
    baseline_bad_rate: float = 0.0,
    baseline_median_rr: float = 0.0,
) -> None:
    """Generate archetypes/prefilter.yaml with rules: from recommendation results.

    Strategy:
      1. If OR pair was recommended -> use any_of rule
      2. Else use top-1 rule from recommendation
      3. Always append last_evaluation section
    """
    from datetime import date as _date

    lines = []
    lines.append(f"# {strategy.upper()} Archetype Prefilter (auto-generated)")
    lines.append(f"# 职责: archetype 成立的前置条件 — 不满足的样本不参与 Gate 训练")
    lines.append(f"# 自动生成: {_date.today()}")
    lines.append(f"# 数据源: {data_source}, {n_rows} 行")
    lines.append("")

    # Try to extract OR pair from recommendation
    or_pair = None
    if recommendation_report and "or_pair" in recommendation_report:
        or_pair = recommendation_report["or_pair"]

    # Also check locals from recommendation_report dict
    if or_pair and or_pair.get("sub_rules"):
        lines.append("rules:")
        lines.append("  - any_of:")
        for sub in or_pair["sub_rules"]:
            lines.append(f'      - feature: {sub["feature"]}')
            lines.append(f'        operator: "{sub["operator"]}"')
            lines.append(f'        value: {sub["value"]}')
        rationale = or_pair.get("rationale", "auto-generated OR rule")
        lines.append(f'    rationale: "{rationale}"')
    elif recommendation_report and recommendation_report.get("top_rules"):
        # Use top recommended rules (max 3 to avoid over-filtering)
        top = recommendation_report["top_rules"][:3]
        lines.append("rules:")
        for r in top:
            lines.append(f'  - feature: {r["feature"]}')
            lines.append(f'    operator: "{r["operator"]}"')
            lines.append(f'    value: {r["value"]}')
            rationale = r.get(
                "rationale", f"auto, composite={r.get('composite', 'N/A')}"
            )
            lines.append(f'    rationale: "{rationale}"')
    else:
        # Fallback: no rules could be generated
        lines.append("# ⚠️  无法自动生成 rules, 请手动配置")
        lines.append("# rules:")
        lines.append("#   - feature: xxx")
        lines.append('#     operator: ">="')
        lines.append("#     value: 0.0")
        print("  ⚠️  无推荐结果可用, 生成了模板文件 (无有效 rules)")

    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Append last_evaluation
    _write_prefilter_evaluation(
        config_path=str(output_path),
        positive=positive,
        anti=anti,
        absence=absence,
        temporal_report=temporal_report,
        data_source=data_source,
        n_rows=n_rows,
        baseline_bad_rate=baseline_bad_rate,
        baseline_median_rr=baseline_median_rr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="分位数分层分析：验证 archetype 语义特征的预测力"
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="predictions.parquet 路径 (训练管线输出)",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["bpc", "me", "fer", "lv"],
        help="策略名称",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="prefilter.yaml 路径 (默认: config/strategies/{strategy}/prefilter.yaml)",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="发现模式: 跳过 prefilter.yaml, 扫描 parquet 全部数值特征",
    )
    parser.add_argument(
        "--deps",
        default=DEFAULT_DEPS_PATH,
        help=f"feature_dependencies.yaml 路径 (默认: {DEFAULT_DEPS_PATH})",
    )
    parser.add_argument(
        "--percentiles",
        default=",".join(str(p) for p in DEFAULT_PERCENTILES),
        help=f"百分位列表 (逗号分隔, 默认: {DEFAULT_PERCENTILES})",
    )
    parser.add_argument(
        "--rr-col",
        default="forward_rr",
        help="forward_rr 列名 (默认: forward_rr)",
    )
    parser.add_argument(
        "--label-col",
        default="success_no_rr_extreme",
        help="标签列名 (默认: success_no_rr_extreme, 1=好 0=坏)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON 路径 (可选)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"每组最小样本量 (默认: {DEFAULT_MIN_SAMPLES})",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="启用时间稳定性分析: rolling bad_rate 曲线 + CV + 多窗口自动选择",
    )
    parser.add_argument(
        "--select-recent",
        type=int,
        default=None,
        metavar="MONTHS",
        help="Mode A: 用最近 N 个月做特征选择, 全量数据做 AND 覆盖率模拟 (可选加 --temporal 做 CV 分析)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="过滤数据开始日期 (如 2025-07-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="过滤数据结束日期 (如 2026-01-01)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="分析完成后自动复制 prefilter.yaml 到 archetypes/prefilter.yaml",
    )
    parser.add_argument(
        "--n-gate-features",
        type=int,
        default=None,
        metavar="N",
        help="Gate 训练特征数 (可选). 提供后会警告 sample:feature 比 < 20:1 的方案",
    )
    parser.add_argument(
        "--min-prefilter-pass-rate",
        type=float,
        default=None,
        metavar="RATE",
        help="Prefilter 最低通过率 (0~1). 如 0.15 表示至少 15%% 的 bars 必须通过 prefilter. "
        "用于防止 prefilter 过严导致交易太少. 由 research_pipeline.yaml kpi_gates 控制.",
    )
    parser.add_argument(
        "--min-prefilter-rows",
        type=int,
        default=None,
        metavar="N",
        help="Prefilter 后最低行数 (覆盖 MIN_PREFILTER_ROWS 默认值 500). "
        "由 research_pipeline.yaml kpi_gates 控制.",
    )
    args = parser.parse_args()

    min_samples = args.min_samples

    # ── 1. 加载数据 ──────────────────────────────────────────
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ 文件不存在: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"✅ 加载 {len(df)} 行, {len(df.columns)} 列 from {logs_path}")

    # --select-recent 不再强制 --temporal; 覆盖率/AND模拟不依赖 temporal
    # 用户需要 CV 分析时显式加 --temporal
    # if args.select_recent:
    #     args.temporal = True

    # ── 1b. 日期过滤 (可选) ──────────────────────────────
    if args.start_date or args.end_date:
        time_col = _find_time_column(df)
        if time_col is None:
            print("❌ 无法找到时间列, --start-date/--end-date 无法使用")
            return 1
        times = _get_times(df, time_col)
        n_before = len(df)
        if args.start_date:
            df = df.loc[times >= pd.Timestamp(args.start_date)]
            times = times.loc[df.index]
        if args.end_date:
            df = df.loc[times < pd.Timestamp(args.end_date)]
        print(f"📅 日期过滤: {n_before} → {len(df)} 行", end="")
        if args.start_date:
            print(f" (>= {args.start_date})", end="")
        if args.end_date:
            print(f" (< {args.end_date})", end="")
        print()
        if len(df) < TEMPORAL_MIN_SAMPLES_PER_WINDOW:
            print(
                f"❌ 过滤后样本量 {len(df)} < {TEMPORAL_MIN_SAMPLES_PER_WINDOW}, 统计不可信"
            )
            return 1

    # ── 1c. select-recent: 分离 df_full 和 df_recent ─────
    df_full = df  # 全周期数据 (temporal rolling 用)
    select_recent_months = args.select_recent
    if select_recent_months:
        time_col_sr = _find_time_column(df)
        if time_col_sr is None:
            print("❌ 无法找到时间列, --select-recent 无法使用")
            return 1
        times_sr = _get_times(df, time_col_sr)
        t_max = times_sr.max()
        cutoff = t_max - pd.DateOffset(months=select_recent_months)
        df = df.loc[times_sr.values >= cutoff].copy()
        print(
            f"🎯 Mode A: 特征选择窗口 = 最近 {select_recent_months} 个月 "
            f"({cutoff.strftime('%Y-%m-%d')} → {t_max.strftime('%Y-%m-%d')}), "
            f"{len(df)} 行"
        )
        print(f"   全周期数据保留 {len(df_full)} 行用于 temporal rolling")
        if len(df) < TEMPORAL_MIN_SAMPLES_PER_WINDOW:
            print(
                f"❌ 选择窗口样本量 {len(df)} < {TEMPORAL_MIN_SAMPLES_PER_WINDOW}, 统计不可信"
            )
            return 1

    # ── 2. 检查标签列 ────────────────────────────────────────
    rr_col = args.rr_col
    label_col = args.label_col

    # 自动生成标签 (与 optimize_gate_unified.py 一致)
    if label_col not in df.columns and rr_col in df.columns:
        df[label_col] = (df[rr_col] >= -0.8).astype(int)
        print(f"ℹ️  自动生成 '{label_col}' from '{rr_col}' (threshold: -0.8R)")
    # Mode A: df_full 也需要标签列 (用于 AND 模拟)
    if (
        select_recent_months
        and label_col not in df_full.columns
        and rr_col in df_full.columns
    ):
        df_full[label_col] = (df_full[rr_col] >= -0.8).astype(int)

    if label_col not in df.columns:
        print(f"❌ 标签列 '{label_col}' 不存在")
        print(
            f"   可用列: {[c for c in df.columns if 'success' in c or 'failure' in c or 'rr' in c][:20]}"
        )
        return 1

    if rr_col not in df.columns:
        # 尝试常见 fallback
        for candidate in ["bpc_impulse_return_atr", "rr", "return_atr"]:
            if candidate in df.columns:
                rr_col = candidate
                break
        if rr_col not in df.columns:
            print(f"⚠️  RR 列 '{args.rr_col}' 不存在, median RR 将显示 NaN")
            df[rr_col] = np.nan

    # 统计概览
    n_total = len(df)
    n_valid = df[label_col].notna().sum()
    bad_rate = (df[label_col] == 0).mean()
    med_rr = df[rr_col].median() if rr_col in df.columns else float("nan")

    symbols = df["symbol"].nunique() if "symbol" in df.columns else "?"
    print(f"   {n_valid} valid rows, {symbols} symbols")
    print(f"   全局 bad rate: {bad_rate:.1%}, median RR: {med_rr:+.2f}")

    # ── 3. 找候选特征 ────────────────────────────────────────
    config_path = (
        Path(args.config)
        if args.config
        else Path(f"config/strategies/{args.strategy}/prefilter.yaml")
    )
    deps_path = Path(args.deps)

    if getattr(args, "all_features", False):
        # ── 发现模式: 扫描 parquet 全部数值特征 ──
        _exclude = {
            rr_col,
            "forward_rr",
            "forward_rr_long",
            label_col,
            "success_no_rr_extreme",
            "failure_rr_extreme",
            "target",
            "sample_weight",
            "pred",
            "pred_proba",
            "timestamp",
            "datetime",
            "date",
            "symbol",
            "_symbol",
            "direction",
            "signal_direction",
        }
        features = []
        for col in sorted(df.columns):
            if col in _exclude:
                continue
            if df[col].dtype not in ["float64", "float32", "int64", "int32"]:
                continue
            s = df[col].dropna()
            if len(s) < min_samples * 2:
                continue
            features.append(col)
        print(f"\n🔍 --all-features: 扫描到 {len(features)} 个数值特征")
        if not features:
            print("❌ parquet 中无有效数值特征")
            return 1
    else:
        # ── 配置模式: 从 prefilter.yaml 读取 ──
        if not config_path.exists():
            print(f"❌ prefilter.yaml 不存在: {config_path}")
            return 1
        if not deps_path.exists():
            print(f"❌ feature_dependencies.yaml 不存在: {deps_path}")
            return 1
        print(f"\n📖 读取 {config_path}")
        features = _resolve_features_from_config(
            str(config_path), str(deps_path), list(df.columns)
        )
        if not features:
            print(f"❌ 从 prefilter.yaml 解析后无匹配列")
            print(f"   提示: 检查 prefilter.yaml 的 candidates 是否与 parquet 列名对应")
            return 1

        print(f"\n📊 找到 {len(features)} 个候选特征")
        for f in features:
            n_valid_f = df[f].notna().sum()
            n_nonzero = (df[f] != 0).sum() if df[f].notna().any() else 0
            print(f"   {f}: {n_valid_f} valid, {n_nonzero} nonzero")

    # ── 4. 逐特征分层分析 ────────────────────────────────────
    percentiles = [int(p) for p in args.percentiles.split(",")]
    all_results = []

    for feat in features:
        feat_results = analyze_feature(
            df, feat, percentiles, rr_col, label_col, min_samples
        )
        all_results.extend(feat_results)

    if not all_results:
        print("\n⚠️  所有特征均无有效分层结果 (样本不足)")
        return 0

    # ── 5. 分类和输出 ────────────────────────────────────────
    positive, anti, absence = _classify_results(all_results)

    print(f"\n{'='*110}")
    print(f"📊 {args.strategy.upper()} 语义特征分位数分层分析")
    print(
        f"   数据: {n_valid} rows, {symbols} symbols, bad rate={bad_rate:.1%}, medRR={med_rr:+.2f}"
    )
    print(f"{'='*110}")

    print(_format_table(positive, "✅ 正信号 (高端有信号 → bad rate 降低)"))
    print(_format_table(anti, "⚠️  反信号 (高端有信号 → bad rate 升高)"))
    print(_format_table(absence, "🔻 低端信号 (缺失信号 → bad rate 升高)"))

    # ── 6. 总结 ──────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("📋 总结")
    print(f"   正信号: {len(positive)} 条 (可做 guardrail / pre_filter 的正向条件)")
    print(f"   反信号: {len(anti)} 条 (高值=坏, 可做反向 hard_gate)")
    print(f"   低端信号: {len(absence)} 条 (缺失=坏, 可做最低门槛)")

    if positive:
        best = positive[0]
        print(
            f"\n   最强正信号: {best['feature']} {best['percentile']}, "
            f"bad_rate: {best['bad_rate_signal']:.1%} vs {best['bad_rate_rest']:.1%} "
            f"(差异 {best['bad_rate_diff']:+.1%})"
        )
    if anti:
        best = anti[0]
        print(
            f"   最强反信号: {best['feature']} {best['percentile']}, "
            f"bad_rate: {best['bad_rate_signal']:.1%} vs {best['bad_rate_rest']:.1%} "
            f"(差异 {best['bad_rate_diff']:+.1%})"
        )
    if absence:
        best = absence[0]
        print(
            f"   最强低端信号: {best['feature']} {best['percentile']}, "
            f"bad_rate: {best['bad_rate_signal']:.1%} vs {best['bad_rate_rest']:.1%} "
            f"(差异 {best['bad_rate_diff']:+.1%})"
        )

    # ── 6a. Prefilter AND 组合样本量测算 + Jaccard 冗余矩阵 ──
    combo_report = _prefilter_combination_analysis(
        df,
        positive,
        anti,
        absence,
        label_col,
        rr_col,
        n_valid,
    )

    # ── 6b. 时间稳定性分析 (可选) ─────────────────────
    temporal_report = None
    if args.temporal:
        # Mode A: rolling 在 df_full 上做; Mode B: rolling 在 df 上做
        df_for_rolling = df_full if select_recent_months else df
        time_col = _find_time_column(df_for_rolling)
        if time_col is None:
            print("\n⚠️  无法找到时间列, 跳过 --temporal 分析")
        else:
            # 只分析显著特征 (bad_rate_diff > 2%)
            significant = positive + anti + absence
            if not significant:
                print("\n⚠️  无显著特征, 跳过 --temporal 分析")
            else:
                if select_recent_months:
                    print(
                        f"\n🔄 Mode A: 用近 {select_recent_months}m 选出的 {len(significant)} 个特征, 在全周期 {len(df_for_rolling)} 行上做 rolling 验证"
                    )
                temporal_report = _temporal_stability_analysis(
                    df_for_rolling, significant, label_col, time_col
                )

    # ── 6c. 综合评分精选推荐 + rules: YAML ──────────────────
    # Mode A: AND 模拟用全量数据 (df_full), 因为训练管线用全量数据
    # Mode B: AND 模拟用 df (就是全量)
    df_for_sim = df_full if select_recent_months else df
    recommendation_report = _prefilter_recommendation(
        positive,
        anti,
        absence,
        all_results,
        temporal_report=temporal_report,
        strategy=args.strategy,
        df=df_for_sim,
        label_col=label_col,
        n_gate_features=args.n_gate_features,
        min_prefilter_pass_rate=args.min_prefilter_pass_rate,
        min_prefilter_rows=args.min_prefilter_rows,
    )

    # ── 6d. 回写 last_evaluation ──────────────────────────
    _write_prefilter_evaluation(
        config_path=str(config_path),
        positive=positive,
        anti=anti,
        absence=absence,
        temporal_report=temporal_report,
        data_source=str(logs_path.name),
        n_rows=n_total,
        baseline_bad_rate=float(bad_rate),
        baseline_median_rr=float(med_rr),
    )

    # ── 6e. --promote: 更新 archetypes/prefilter.yaml ──────
    #  关键逻辑:
    #    - 始终从本次推荐结果生成最新 rules (每次 promote 都是最新优化)
    #    - 绝不把候选声明文件 (candidates:) 覆盖到 archetypes/
    if args.promote:
        arch_prefilter = Path(config_path).parent / "archetypes" / "prefilter.yaml"
        arch_prefilter.parent.mkdir(parents=True, exist_ok=True)

        _generate_promoted_prefilter(
            arch_prefilter,
            recommendation_report=recommendation_report,
            strategy=args.strategy,
            positive=positive,
            anti=anti,
            absence=absence,
            temporal_report=temporal_report,
            data_source=str(logs_path.name),
            n_rows=n_total,
            baseline_bad_rate=float(bad_rate),
            baseline_median_rr=float(med_rr),
        )
        print(f"\n📦 Promoted prefilter.yaml → {arch_prefilter}")
        print(f"   (从本次推荐结果生成最新 rules)")

    # ── 7. 输出 JSON ─────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "strategy": args.strategy,
            "config_path": str(config_path),
            "data": {
                "n_rows": n_total,
                "n_valid": int(n_valid),
                "n_symbols": (
                    int(symbols) if isinstance(symbols, (int, np.integer)) else symbols
                ),
                "global_bad_rate": round(bad_rate, 4),
                "global_median_rr": round(med_rr, 2),
            },
            "features_analyzed": len(features),
            "features_list": features,
            "percentiles_used": percentiles,
            "summary": {
                "n_positive": len(positive),
                "n_anti": len(anti),
                "n_absence": len(absence),
            },
            "positive_signals": positive,
            "anti_signals": anti,
            "absence_signals": absence,
            "all_results": all_results,
        }

        if temporal_report:
            report["temporal_stability"] = temporal_report

        if combo_report:
            report["prefilter_combination"] = combo_report

        if recommendation_report:
            report["prefilter_recommendation"] = recommendation_report

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 报告已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
