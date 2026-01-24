#!/usr/bin/env python3
"""
实验2：平坦高原优化gate参数

使用平坦高原方法优化gate规则参数，分析分桶和分布，找到robust参数区间。

使用方法:
    python scripts/experiment_gate_plateau_optimization.py \
        --exec-log results/pipeline_<run_id>/execution_log.jsonl \
        --gated-logs results/pipeline_<run_id>/logs_execution_gated.parquet \
        --raw-logs results/pipeline_<run_id>/logs_execution.parquet \
        --out-dir results/experiments/gate_plateau
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_gate_plateau import (
    BucketConfig,
    OptimizationConfig,
    _scan_threshold,
    _find_plateau,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


def analyze_bucket_distribution(
    gated_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    """分析分桶分布"""
    bucket_config = BucketConfig()

    # 按archetype和volatility分桶
    buckets = {}
    for arch_name in ["TC", "TE", "FR", "ET"]:
        arch_mask = gated_df["gate_archetype"].str.contains(
            arch_name, case=False, na=False
        )
        arch_df = gated_df[arch_mask]

        if len(arch_df) == 0:
            continue

        # 计算volatility分桶（使用ATR或volatility特征）
        if "atr" in arch_df.columns:
            vol_col = "atr"
        elif "volatility" in arch_df.columns:
            vol_col = "volatility"
        else:
            vol_col = None

        if vol_col:
            vol_quantiles = arch_df[vol_col].quantile([0.33, 0.67])
            # 处理重复的quantile值
            unique_quantiles = vol_quantiles.unique()
            if len(unique_quantiles) < 2:
                # 如果quantile值相同，使用固定分桶
                arch_df = arch_df.copy()
                arch_df["vol_bucket"] = "unknown"
            else:
                arch_df = arch_df.copy()
                arch_df["vol_bucket"] = pd.cut(
                    arch_df[vol_col],
                    bins=[-np.inf, unique_quantiles[0], unique_quantiles[-1], np.inf],
                    labels=["low", "mid", "high"],
                    duplicates="drop",
                )
        else:
            arch_df["vol_bucket"] = "unknown"

        buckets[arch_name] = {
            "total": len(arch_df),
            "vol_distribution": (
                arch_df["vol_bucket"].value_counts().to_dict() if vol_col else {}
            ),
        }

    return {
        "buckets": buckets,
        "total_trades": len(gated_df[gated_df["gate_ok"] == True]),
    }


def optimize_all_rules(
    gated_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    execution_archetypes_path: str,
    out_dir: Path,
    min_trade_rate: float = 0.005,
    min_trades_per_bucket: int = 10,
    min_sharpe_threshold: float = 0.5,
    threshold_step: float = 0.05,
) -> Dict[str, Any]:
    """
    优化所有gate规则

    注意：使用gated_df中的特征（已经从FeatureStore加载），而不是raw_df
    """
    """优化所有gate规则"""
    arches = load_execution_archetypes_registry(execution_archetypes_path)
    bucket_config = BucketConfig()
    opt_config = OptimizationConfig(
        min_trade_rate=min_trade_rate,
        min_trades_per_bucket=min_trades_per_bucket,
        min_sharpe_threshold=min_sharpe_threshold,
    )

    all_results = {}

    for arch_name, arch in arches.items():
        if not arch.gate_rules:
            continue

        arch_results = []
        rules = (arch.gate_rules or {}).get("rules") or []

        # 优化quantile和value类型的规则
        optimizable_rules = [
            r
            for r in rules
            if r.get("kind", "").startswith("quantile_")
            or r.get("kind", "").startswith("value_")
        ]

        if not optimizable_rules:
            continue

        print(f"\n优化Archetype: {arch_name}")

        for rule in optimizable_rules:
            rule_name = rule.get("name", "unknown")
            feature_key = rule.get("key")
            rule_kind = rule.get("kind")
            current_threshold = rule.get("quantile") or rule.get("value", 0.5)

            if not feature_key or not rule_kind:
                continue

            print(f"  规则: {rule_name} ({feature_key}, {rule_kind})")

            # 确定阈值范围
            if rule_kind.startswith("quantile_"):
                threshold_range = (0.0, 1.0)
            else:
                # value_* 类型：根据特征分布确定范围
                # 优先从gated_df获取特征（已从FeatureStore加载）
                feature_df = gated_df if feature_key in gated_df.columns else raw_df
                if feature_key in feature_df.columns:
                    feature_vals = feature_df[feature_key].dropna()
                    if len(feature_vals) > 0:
                        p5 = float(feature_vals.quantile(0.05))
                        p95 = float(feature_vals.quantile(0.95))
                        # 扩展范围以确保覆盖当前阈值
                        current_val = current_threshold
                        if current_val < p5:
                            p5 = current_val * 0.8
                        if current_val > p95:
                            p95 = current_val * 1.2
                        threshold_range = (p5, p95)
                    else:
                        print(f"    特征 {feature_key} 没有有效值，跳过")
                        continue
                else:
                    print(f"    特征 {feature_key} 不存在于gated或raw logs，跳过")
                    continue

            # 扫描阈值
            # 使用gated_df作为数据源（包含所有特征），但需要移除gate_ok等列以避免干扰
            scan_df = gated_df.copy()
            # 移除gate相关列，只保留特征和基础列
            gate_cols = ["gate_ok", "gate_decision", "gate_reasons", "gate_archetype"]
            for col in gate_cols:
                if col in scan_df.columns:
                    scan_df = scan_df.drop(columns=[col])

            try:
                scan_results = _scan_threshold(
                    scan_df,  # 使用包含特征的gated_df
                    rule_name,
                    feature_key,
                    rule_kind,
                    threshold_range,
                    threshold_step,
                    bucket_config,
                    opt_config,
                    execution_archetypes_path,
                    base_gated_df=None,  # 不提供base，因为我们已经在scan_df中
                )

                if len(scan_results) == 0:
                    print(f"    未找到有效阈值")
                    continue

                # 找到平坦高原
                plateau = _find_plateau(scan_results, min_sharpe_threshold)

                if plateau:
                    plateau_start, plateau_end, plateau_median = plateau
                    print(
                        f"    平坦高原: [{plateau_start:.4f}, {plateau_end:.4f}], 推荐: {plateau_median:.4f}"
                    )
                else:
                    # 使用最佳阈值
                    best_idx = scan_results["robustness_score"].idxmax()
                    plateau_median = scan_results.loc[best_idx, "threshold"]
                    plateau_start = plateau_median
                    plateau_end = plateau_median
                    print(f"    未找到平坦高原，使用最佳阈值: {plateau_median:.4f}")

                # 获取最佳结果
                best_idx = scan_results["robustness_score"].idxmax()
                best_result = scan_results.loc[best_idx].to_dict()

                arch_results.append(
                    {
                        "rule_name": rule_name,
                        "feature_key": feature_key,
                        "rule_kind": rule_kind,
                        "current_threshold": current_threshold,
                        "plateau_start": float(plateau_start),
                        "plateau_end": float(plateau_end),
                        "recommended_threshold": float(plateau_median),
                        "robustness_score": float(
                            best_result.get("robustness_score", 0)
                        ),
                        "trade_rate": float(best_result.get("trade_rate", 0)),
                        "min_coverage": int(best_result.get("min_coverage", 0)),
                        "bucket_sharpes": best_result.get("bucket_sharpes", {}),
                    }
                )
            except Exception as e:
                print(f"    优化失败: {e}")
                continue

        if arch_results:
            all_results[arch_name] = arch_results

    return all_results


def generate_plateau_report(
    optimization_results: Dict[str, Any],
    bucket_analysis: Dict[str, Any],
    out_dir: Path,
) -> None:
    """生成平坦高原分析报告"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存优化结果JSON
    optimal_params_path = out_dir / "optimal_params.json"
    with open(optimal_params_path, "w", encoding="utf-8") as f:
        json.dump(optimization_results, f, indent=2, default=str)

    # 生成Markdown报告
    report_path = out_dir / "plateau_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Gate参数平坦高原优化报告\n\n")

        f.write("## 1. 分桶分布分析\n\n")
        f.write(f"- 总交易数: {bucket_analysis['total_trades']}\n\n")
        f.write("### 按Archetype分布\n\n")
        for arch_name, arch_data in bucket_analysis.get("buckets", {}).items():
            f.write(f"#### {arch_name}\n")
            f.write(f"- 总交易数: {arch_data['total']}\n")
            if arch_data.get("vol_distribution"):
                f.write("- Volatility分布:\n")
                for vol_bucket, count in arch_data["vol_distribution"].items():
                    f.write(f"  - {vol_bucket}: {count}\n")
            f.write("\n")

        f.write("## 2. 平坦高原优化结果\n\n")
        for arch_name, arch_results in optimization_results.items():
            f.write(f"### {arch_name}\n\n")
            for result in arch_results:
                f.write(f"#### {result['rule_name']}\n")
                f.write(f"- 特征: {result['feature_key']}\n")
                f.write(f"- 规则类型: {result['rule_kind']}\n")
                f.write(f"- 当前阈值: {result['current_threshold']:.4f}\n")
                f.write(
                    f"- 平坦高原区间: [{result['plateau_start']:.4f}, {result['plateau_end']:.4f}]\n"
                )
                f.write(f"- 推荐阈值: {result['recommended_threshold']:.4f}\n")
                f.write(f"- Robustness Score: {result['robustness_score']:.2f}\n")
                f.write(f"- 交易率: {result['trade_rate']:.2%}\n")
                f.write(f"- 最小覆盖: {result['min_coverage']}\n")
                f.write("\n")

    print(f"✅ 报告已生成:")
    print(f"   - 优化参数: {optimal_params_path}")
    print(f"   - 分析报告: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="平坦高原优化gate参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exec-log",
        required=True,
        help="Execution log文件（jsonl）",
    )
    parser.add_argument(
        "--gated-logs",
        required=True,
        help="Gated logs文件（parquet）",
    )
    parser.add_argument(
        "--raw-logs",
        default=None,
        help="原始logs文件（parquet，可选）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--min-trade-rate",
        type=float,
        default=0.005,
        help="最小交易率（默认0.5%）",
    )
    parser.add_argument(
        "--min-trades-per-bucket",
        type=int,
        default=10,
        help="每桶最少交易数",
    )
    parser.add_argument(
        "--min-sharpe-threshold",
        type=float,
        default=0.5,
        help="平坦高原的最低Sharpe要求",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="阈值扫描步长",
    )

    args = parser.parse_args()

    # 读取数据
    gated_df = pd.read_parquet(args.gated_logs)
    print(f"✅ 读取gated数据: {len(gated_df)} 行")

    if args.raw_logs:
        raw_df = pd.read_parquet(args.raw_logs)
        print(f"✅ 读取原始数据: {len(raw_df)} 行")
    else:
        raw_df = gated_df.copy()
        print("⚠️  未提供原始数据，使用gated数据作为基础")

    # 分析分桶分布
    print("\n📊 分析分桶分布...")
    bucket_analysis = analyze_bucket_distribution(gated_df, raw_df)

    # 优化所有规则
    print("\n🔍 开始平坦高原优化...")
    optimization_results = optimize_all_rules(
        gated_df=gated_df,
        raw_df=raw_df,
        execution_archetypes_path=args.execution_archetypes,
        out_dir=Path(args.out_dir),
        min_trade_rate=args.min_trade_rate,
        min_trades_per_bucket=args.min_trades_per_bucket,
        min_sharpe_threshold=args.min_sharpe_threshold,
        threshold_step=args.threshold_step,
    )

    # 生成报告
    print("\n📝 生成报告...")
    generate_plateau_report(optimization_results, bucket_analysis, Path(args.out_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
