#!/usr/bin/env python3
"""
Evidence Score Plateau Optimization

使用 good 样本的 sharpness 作为优化目标，找到 evidence 特征的最佳分位数划分。

Sharpness 定义:
    sharpness = mean(score | good) - mean(score | bad)
    或者: sharpness = P(score > 0.7 | good) - P(score > 0.7 | bad)

目标: 找到分位数划分，使得 good 样本的分数分布显著高于 bad 样本。

使用方法:
    python scripts/optimize_evidence_plateau.py \
        --strategy bpc \
        --logs results/trade_logs.parquet \
        --output results/evidence_optimization.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.archetype import (
    EvidenceFeature,
    StrategyArchetype,
    load_strategy_archetype,
)


@dataclass
class EvidenceOptimizationConfig:
    """Evidence optimization configuration"""

    min_sharpness: float = 0.05  # 最小 sharpness 要求
    min_samples_good: int = 50  # good 样本最少数量
    min_samples_bad: int = 50  # bad 样本最少数量
    plateau_width_bins: int = 2  # 平坦高原宽度（分位数 bin 数量变化）


# 语义标签到分数的映射
SEMANTIC_SCORE_MAP = {
    "suppress": 0.0,
    "downweight": 0.25,
    "neutral": 0.5,
    "favor": 0.75,
    "amplify": 1.0,
}


def compute_evidence_score(
    values: pd.Series,
    quantile_bins: List[float],
    quantile_labels: List[str],
    quantiles: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    计算 evidence 分数

    Args:
        values: 特征值序列
        quantile_bins: 分位数边界 [q1, q2, q3, q4]
        quantile_labels: 语义标签 [label1, label2, label3, label4, label5]
        quantiles: 分位数值字典 {"p25": 0.25, "p50": 0.5, ...}

    Returns:
        分数序列 (0-1 范围)
    """
    scores = []

    for val in values:
        if pd.isna(val):
            scores.append(0.5)  # 默认中性
            continue

        # 计算分位数位置
        if quantiles:
            # 使用提供的分位数值
            percentile = 0.5
            for q_name, q_val in sorted(quantiles.items(), key=lambda x: x[1]):
                if val <= q_val:
                    # 解析分位数名称 (e.g., "p25" -> 0.25)
                    try:
                        percentile = float(q_name.replace("p", "")) / 100
                    except ValueError:
                        percentile = 0.5
                    break
        else:
            # 假设值已经是百分位
            percentile = val

        # 根据分位数边界确定标签
        label = quantile_labels[-1] if quantile_labels else "neutral"
        for i, bin_val in enumerate(quantile_bins):
            if percentile <= bin_val:
                label = quantile_labels[i] if i < len(quantile_labels) else "neutral"
                break

        # 映射到分数
        score = SEMANTIC_SCORE_MAP.get(label, 0.5)
        scores.append(score)

    return pd.Series(scores, index=values.index)


def compute_sharpness(
    df: pd.DataFrame,
    feature_col: str,
    quantile_bins: List[float],
    quantile_labels: List[str],
    label_col: str = "is_good",
) -> Dict[str, float]:
    """
    计算给定分位数划分的 sharpness

    Args:
        df: DataFrame
        feature_col: 特征列名
        quantile_bins: 分位数边界
        quantile_labels: 语义标签
        label_col: 标签列

    Returns:
        包含 sharpness 和其他指标的字典
    """
    if feature_col not in df.columns:
        return {"sharpness": 0.0, "mean_good": 0.5, "mean_bad": 0.5}

    # 计算分数
    scores = compute_evidence_score(
        df[feature_col],
        quantile_bins,
        quantile_labels,
    )

    # 分组统计
    is_good = df[label_col] == 1
    is_bad = df[label_col] == 0

    mean_good = scores[is_good].mean() if is_good.any() else 0.5
    mean_bad = scores[is_bad].mean() if is_bad.any() else 0.5

    sharpness = mean_good - mean_bad

    # 计算高分比例差异
    high_score_threshold = 0.7
    p_high_good = (
        (scores[is_good] > high_score_threshold).mean() if is_good.any() else 0.0
    )
    p_high_bad = (scores[is_bad] > high_score_threshold).mean() if is_bad.any() else 0.0

    return {
        "sharpness": sharpness,
        "mean_good": mean_good,
        "mean_bad": mean_bad,
        "p_high_good": p_high_good,
        "p_high_bad": p_high_bad,
        "high_score_lift": p_high_good - p_high_bad,
    }


def generate_quantile_bin_candidates(
    base_bins: List[float] = [0.2, 0.4, 0.6, 0.8],
) -> List[List[float]]:
    """
    生成分位数边界候选集

    基于 base_bins 生成变体：
    - 移动边界 ±0.05, ±0.10
    - 调整对称性
    """
    candidates = [base_bins]

    # 生成移动变体
    shifts = [-0.10, -0.05, 0.05, 0.10]
    for shift in shifts:
        shifted = [min(1.0, max(0.0, b + shift)) for b in base_bins]
        if shifted not in candidates:
            candidates.append(shifted)

    # 生成非对称变体
    asymmetric_variants = [
        [0.15, 0.35, 0.65, 0.85],  # 更紧的中性区
        [0.25, 0.45, 0.55, 0.75],  # 更宽的中性区
        [0.10, 0.30, 0.70, 0.90],  # 极端分布
        [0.30, 0.45, 0.55, 0.70],  # 压缩分布
    ]
    for variant in asymmetric_variants:
        if variant not in candidates:
            candidates.append(variant)

    return candidates


def optimize_evidence_feature(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str = "is_good",
    config: Optional[EvidenceOptimizationConfig] = None,
) -> Dict[str, Any]:
    """
    优化单个 evidence 特征的分位数划分

    Args:
        df: DataFrame
        feature_col: 特征列名
        label_col: 标签列
        config: 优化配置

    Returns:
        优化结果
    """
    if config is None:
        config = EvidenceOptimizationConfig()

    if feature_col not in df.columns:
        return {
            "feature": feature_col,
            "status": "skip",
            "reason": f"Feature not found in DataFrame",
        }

    # 检查样本数
    n_good = (df[label_col] == 1).sum()
    n_bad = (df[label_col] == 0).sum()

    if n_good < config.min_samples_good or n_bad < config.min_samples_bad:
        return {
            "feature": feature_col,
            "status": "skip",
            "reason": f"Insufficient samples: good={n_good}, bad={n_bad}",
        }

    # 5档语义标签（固定）
    quantile_labels = ["suppress", "downweight", "neutral", "favor", "amplify"]

    # 生成候选分位数边界
    candidates = generate_quantile_bin_candidates()

    # 评估每个候选
    results = []
    for bins in candidates:
        metrics = compute_sharpness(df, feature_col, bins, quantile_labels, label_col)
        metrics["quantile_bins"] = bins
        results.append(metrics)

    # 找最佳候选（最大 sharpness）
    best = max(results, key=lambda x: x["sharpness"])

    # 找满足条件的候选（平坦高原）
    valid_results = [r for r in results if r["sharpness"] >= config.min_sharpness]

    if not valid_results:
        return {
            "feature": feature_col,
            "status": "no_valid_bins",
            "best_sharpness": best["sharpness"],
            "best_bins": best["quantile_bins"],
            "scan_results": results,
        }

    # 在有效结果中找稳定性最好的（sharpness 方差最小的邻域）
    # 简化：直接使用 sharpness 最大的
    recommended = max(valid_results, key=lambda x: x["sharpness"])

    return {
        "feature": feature_col,
        "status": "optimized",
        "recommended_bins": recommended["quantile_bins"],
        "sharpness": recommended["sharpness"],
        "mean_good": recommended["mean_good"],
        "mean_bad": recommended["mean_bad"],
        "high_score_lift": recommended["high_score_lift"],
        "num_valid_candidates": len(valid_results),
        "all_candidates_count": len(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence Score Plateau Optimization")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name (e.g., bpc)",
    )
    parser.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="Root directory for strategy configs",
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Trade logs parquet with features and labels",
    )
    parser.add_argument(
        "--label-col",
        default="is_good",
        help="Label column name (1=good, 0=bad)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.05,
        help="Minimum sharpness requirement",
    )
    args = parser.parse_args()

    # Load strategy archetype
    try:
        arch = load_strategy_archetype(args.strategy, args.strategies_root)
    except Exception as e:
        print(f"❌ Failed to load strategy '{args.strategy}': {e}")
        return 1

    print(f"✅ Loaded strategy: {arch.name}")
    print(f"   Evidence features: {len(arch.evidence.features)}")

    # Load logs
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"✅ Loaded {len(df)} rows from {logs_path}")

    # Check label column
    if args.label_col not in df.columns:
        print(f"❌ Label column '{args.label_col}' not found in DataFrame")
        print(f"   Available columns: {list(df.columns)[:20]}...")
        return 1

    n_good = (df[args.label_col] == 1).sum()
    n_bad = (df[args.label_col] == 0).sum()
    print(f"   Good samples: {n_good}, Bad samples: {n_bad}")

    # Create config
    config = EvidenceOptimizationConfig(
        min_sharpness=args.min_sharpness,
    )

    # Optimize each evidence feature
    results = {}

    print("\n📋 Optimizing Evidence Features:")
    for ef in arch.evidence.features:
        feature_col = ef.feature
        print(f"  Processing: {feature_col}")

        result = optimize_evidence_feature(df, feature_col, args.label_col, config)
        results[ef.id] = result

        if result.get("status") == "optimized":
            print(f"    ✅ Optimized:")
            print(f"       Bins: {result['recommended_bins']}")
            print(f"       Sharpness: {result['sharpness']:.3f}")
            print(
                f"       Mean (good/bad): {result['mean_good']:.3f} / {result['mean_bad']:.3f}"
            )
        else:
            print(f"    ⚠️  {result.get('status')}: {result.get('reason', 'N/A')}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_path}")

    # Summary
    n_optimized = sum(1 for r in results.values() if r.get("status") == "optimized")
    n_skip = sum(1 for r in results.values() if r.get("status") == "skip")
    n_no_valid = sum(1 for r in results.values() if r.get("status") == "no_valid_bins")

    print(f"\n📊 Summary:")
    print(f"   Optimized: {n_optimized}")
    print(f"   No valid bins: {n_no_valid}")
    print(f"   Skipped: {n_skip}")

    # Generate updated YAML config
    if n_optimized > 0:
        print("\n📝 Suggested evidence.yaml updates:")
        for ef_id, result in results.items():
            if result.get("status") == "optimized":
                print(f"  {ef_id}:")
                print(f"    quantile_bins: {result['recommended_bins']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
