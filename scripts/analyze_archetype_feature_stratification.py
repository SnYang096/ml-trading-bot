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
        help="Mode A: 用最近 N 个月做特征选择, 全周期做 temporal rolling 验证 (隐含 --temporal)",
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
    args = parser.parse_args()

    min_samples = args.min_samples

    # ── 1. 加载数据 ──────────────────────────────────────────
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ 文件不存在: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"✅ 加载 {len(df)} 行, {len(df.columns)} 列 from {logs_path}")

    # --select-recent 隐含 --temporal
    if args.select_recent:
        args.temporal = True

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

    # ── 6c. 回写 last_evaluation ──────────────────────────
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

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 报告已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
