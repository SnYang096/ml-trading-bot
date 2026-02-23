#!/usr/bin/env python3
"""
特征漂移检测 — PSI / KS test / NaN 率变化

对比训练基线 (training_baseline.json) 的特征分布与新数据的特征分布，
检测统计显著的漂移，输出漂移报告。

用法:
    # 对比训练基线 vs 新数据
    python scripts/local_monitor_feature_drift.py \
        --baseline results/train_final_xxx/me/training_baseline.json \
        --new-data data/live_latest.parquet \
        --output reports/drift_report.json

    # 对比两份 parquet (无基线 JSON)
    python scripts/local_monitor_feature_drift.py \
        --old-data results/train_final_xxx/me/predictions.parquet \
        --new-data data/live_latest.parquet \
        --output reports/drift_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ====================================================================
# PSI (Population Stability Index)
# ====================================================================


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    计算 PSI (Population Stability Index).

    PSI < 0.1  → 稳定 (无显著漂移)
    PSI 0.1~0.25 → 轻度漂移
    PSI > 0.25 → 严重漂移

    Args:
        expected: 训练期分布 (参考)
        actual:   新数据分布
        n_bins:   分箱数
        eps:      避免除零的极小值
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return float("nan")

    # 用 expected 的分位数作为分箱边界
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # 去重
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return float("nan")

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / expected_counts.sum() + eps
    actual_pct = actual_counts / actual_counts.sum() + eps

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


# ====================================================================
# KS Test
# ====================================================================


def compute_ks(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """
    计算 KS 统计量和 p-value.

    Returns:
        (ks_statistic, p_value)
    """
    from scipy import stats

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return float("nan"), float("nan")

    result = stats.ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


# ====================================================================
# NaN 率变化
# ====================================================================


def compute_nan_drift(
    old_nan_rate: float,
    new_values: np.ndarray,
) -> Dict[str, float]:
    """计算 NaN 率变化."""
    total = len(new_values)
    if total == 0:
        return {"old_nan_rate": old_nan_rate, "new_nan_rate": 0.0, "nan_drift": 0.0}

    new_nan_rate = float(np.isnan(new_values).sum() / total)
    return {
        "old_nan_rate": round(old_nan_rate, 4),
        "new_nan_rate": round(new_nan_rate, 4),
        "nan_drift": round(new_nan_rate - old_nan_rate, 4),
    }


# ====================================================================
# 漂移等级判定
# ====================================================================


def classify_drift(psi: float, ks_p: float, nan_drift: float) -> str:
    """
    综合判定漂移等级.

    Returns:
        "🟢 STABLE" / "🟡 DRIFT" / "🔴 SEVERE"
    """
    if np.isnan(psi):
        return "⚪ SKIP"

    severe = False
    drift = False

    if psi > 0.25:
        severe = True
    elif psi > 0.10:
        drift = True

    if not np.isnan(ks_p) and ks_p < 0.001:
        severe = True
    elif not np.isnan(ks_p) and ks_p < 0.01:
        drift = True

    if abs(nan_drift) > 0.10:
        severe = True
    elif abs(nan_drift) > 0.05:
        drift = True

    if severe:
        return "🔴 SEVERE"
    elif drift:
        return "🟡 DRIFT"
    return "🟢 STABLE"


# ====================================================================
# Main analysis
# ====================================================================


def analyze_drift(
    baseline_distributions: Dict[str, Dict[str, float]],
    new_data_path: Path,
    *,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    对比训练基线与新数据的特征分布.

    Args:
        baseline_distributions: training_baseline.json 的 feature_distributions
        new_data_path: 新数据 parquet
        top_n: 输出 top-N 最严重漂移特征

    Returns:
        完整漂移报告 dict
    """
    import pandas as pd

    df_new = pd.read_parquet(new_data_path)

    results: List[Dict[str, Any]] = []
    features_checked = 0
    features_drifted = 0
    features_severe = 0

    for feat, baseline_stats in baseline_distributions.items():
        if feat not in df_new.columns:
            results.append(
                {
                    "feature": feat,
                    "status": "⚪ MISSING",
                    "reason": "feature not in new data",
                }
            )
            continue

        new_values = df_new[feat].values.astype(float)
        features_checked += 1

        # 从 baseline 统计重建近似分布 (用 mean/std/p5/p95 生成正态样本)
        # 但更准确的方式是直接用 PSI 的分箱法
        # 我们直接用新数据 vs baseline 统计的偏差
        old_mean = baseline_stats.get("mean", 0)
        old_std = baseline_stats.get("std", 1)
        old_nan_rate = baseline_stats.get("nan_rate", 0)

        valid_new = new_values[~np.isnan(new_values)]
        if len(valid_new) < 10:
            results.append(
                {
                    "feature": feat,
                    "status": "⚪ SKIP",
                    "reason": f"too few valid samples ({len(valid_new)})",
                }
            )
            continue

        new_mean = float(np.mean(valid_new))
        new_std = float(np.std(valid_new))

        # 均值偏移 (标准化)
        mean_shift = (
            abs(new_mean - old_mean) / max(old_std, 1e-8) if old_std > 1e-8 else 0
        )
        # 标准差变化比
        std_ratio = new_std / max(old_std, 1e-8) if old_std > 1e-8 else 1.0

        # NaN 漂移
        nan_info = compute_nan_drift(old_nan_rate, new_values)

        # PSI (用简化方式: 基于 z-score 分箱)
        # 将新数据标准化到 baseline 的尺度
        z_old_p5 = (
            baseline_stats.get("p5", old_mean - 1.65 * old_std) - old_mean
        ) / max(old_std, 1e-8)
        z_old_p95 = (
            baseline_stats.get("p95", old_mean + 1.65 * old_std) - old_mean
        ) / max(old_std, 1e-8)

        # 近似 PSI: 基于均值偏移和标准差变化
        approx_psi = mean_shift**2 * 0.1 + abs(np.log(max(std_ratio, 0.1))) * 0.15
        if abs(nan_info["nan_drift"]) > 0.05:
            approx_psi += abs(nan_info["nan_drift"]) * 0.5

        # KS test (如果 scipy 可用)
        ks_stat, ks_p = float("nan"), float("nan")
        try:
            # 从 baseline 生成合成样本 (正态近似)
            np.random.seed(42)
            synthetic = np.random.normal(
                old_mean, max(old_std, 1e-8), size=min(len(valid_new), 10000)
            )
            ks_stat, ks_p = compute_ks(synthetic, valid_new)
        except ImportError:
            pass

        status = classify_drift(approx_psi, ks_p, nan_info["nan_drift"])

        if "DRIFT" in status:
            features_drifted += 1
        if "SEVERE" in status:
            features_severe += 1

        results.append(
            {
                "feature": feat,
                "status": status,
                "mean_shift_std": round(mean_shift, 3),
                "std_ratio": round(std_ratio, 3),
                "approx_psi": round(approx_psi, 4),
                "ks_statistic": round(ks_stat, 4) if not np.isnan(ks_stat) else None,
                "ks_p_value": round(ks_p, 6) if not np.isnan(ks_p) else None,
                "nan_drift": nan_info["nan_drift"],
                "old_mean": round(old_mean, 6),
                "new_mean": round(new_mean, 6),
                "old_std": round(old_std, 6),
                "new_std": round(new_std, 6),
            }
        )

    # 按 PSI 排序
    results.sort(key=lambda x: x.get("approx_psi", 0), reverse=True)

    # 汇总
    summary = {
        "features_checked": features_checked,
        "features_drifted": features_drifted,
        "features_severe": features_severe,
        "drift_rate": round(features_drifted / max(features_checked, 1), 3),
        "overall_status": (
            "🔴 SEVERE"
            if features_severe > 3
            else (
                "🟡 DRIFT" if features_drifted > features_checked * 0.2 else "🟢 STABLE"
            )
        ),
    }

    return {
        "summary": summary,
        "top_drifted": results[:top_n],
        "all_features": results,
    }


# ====================================================================
# Pretty print
# ====================================================================


def print_drift_report(report: Dict[str, Any]) -> None:
    """打印漂移报告到 stdout."""
    s = report["summary"]
    print(f"\n{'='*70}")
    print(f"📊 特征漂移检测报告")
    print(f"{'='*70}")
    print(f"   检查特征数: {s['features_checked']}")
    print(f"   漂移特征数: {s['features_drifted']} ({s['drift_rate']:.1%})")
    print(f"   严重漂移:   {s['features_severe']}")
    print(f"   整体状态:   {s['overall_status']}")

    top = report.get("top_drifted", [])
    if top:
        print(
            f"\n   {'Feature':<35} {'Status':<14} {'MeanShift':>10} {'PSI':>8} {'NaN±':>8}"
        )
        print(f"   {'-'*75}")
        for r in top:
            if r.get("status", "").startswith("⚪"):
                continue
            print(
                f"   {r['feature']:<35} {r['status']:<14} "
                f"{r.get('mean_shift_std', 0):>10.3f} "
                f"{r.get('approx_psi', 0):>8.4f} "
                f"{r.get('nan_drift', 0):>+8.4f}"
            )


# ====================================================================
# CLI
# ====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="特征漂移检测 (PSI/KS/NaN)")
    parser.add_argument(
        "--baseline", default=None, help="训练基线 JSON (training_baseline.json)"
    )
    parser.add_argument(
        "--old-data", default=None, help="旧数据 parquet (替代 baseline JSON)"
    )
    parser.add_argument("--new-data", required=True, help="新数据 parquet")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--top-n", type=int, default=20, help="输出 top-N 漂移特征")
    args = parser.parse_args()

    # 加载 baseline 分布
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(f"❌ Baseline 不存在: {baseline_path}")
            return 1
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        distributions = baseline.get("feature_distributions", {})
    elif args.old_data:
        # 从旧 parquet 动态计算分布
        from scripts.export_training_baseline import compute_feature_distributions

        distributions = compute_feature_distributions(Path(args.old_data))
    else:
        print("❌ 必须指定 --baseline 或 --old-data")
        return 1

    if not distributions:
        print("❌ Baseline 分布为空")
        return 1

    new_data_path = Path(args.new_data)
    if not new_data_path.exists():
        print(f"❌ 新数据不存在: {new_data_path}")
        return 1

    print(f"📊 对比: baseline ({len(distributions)} features) vs {new_data_path.name}")

    report = analyze_drift(distributions, new_data_path, top_n=args.top_n)
    print_drift_report(report)

    # 保存 JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\n   ✅ Report saved: {out_path}")

    # 返回 exit code: 0=stable, 1=drift, 2=severe
    status = report["summary"]["overall_status"]
    if "SEVERE" in status:
        return 2
    elif "DRIFT" in status:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
