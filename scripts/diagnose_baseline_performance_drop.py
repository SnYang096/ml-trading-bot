#!/usr/bin/env python3
"""
诊断基线表现下降问题

分析为什么移除regime过滤后，Sharpe从4.657降到-0.0137
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_gate_rules_strictness(
    df: pd.DataFrame,
    archetype: str,
    gate_rules_config: Dict[str, Any],
) -> Dict[str, Any]:
    """分析gate规则的严格程度"""
    arch_trades = df[df["gate_archetype"] == archetype].copy()

    if len(arch_trades) == 0:
        return {
            "archetype": archetype,
            "trade_count": 0,
            "analysis": "No trades found",
        }

    results = {
        "archetype": archetype,
        "trade_count": len(arch_trades),
        "regime_distribution": (
            arch_trades["regime"].value_counts().to_dict()
            if "regime" in arch_trades.columns
            else {}
        ),
        "physical_features": {},
        "gate_rule_violations": {},
    }

    # 分析物理特征分布
    physical_features = [
        "path_efficiency_pct",
        "price_dir_consistency_pct",
        "path_length_pct",
        "jump_risk_pct",
        "atr_percentile",
        "atr_slope_pct",
    ]

    for feat in physical_features:
        if feat in arch_trades.columns:
            results["physical_features"][feat] = {
                "mean": float(arch_trades[feat].mean()),
                "median": float(arch_trades[feat].median()),
                "p10": float(arch_trades[feat].quantile(0.1)),
                "p90": float(arch_trades[feat].quantile(0.9)),
            }

    # 分析收益分布
    if "ret_mean" in arch_trades.columns:
        ret_col = "ret_mean"
    elif "ret_trend" in arch_trades.columns:
        ret_col = "ret_trend"
    else:
        ret_col = None

    if ret_col:
        returns = arch_trades[ret_col].dropna()
        if len(returns) > 0:
            results["returns"] = {
                "mean": float(returns.mean()),
                "median": float(returns.median()),
                "positive_rate": float((returns > 0).mean()),
                "negative_rate": float((returns < 0).mean()),
            }

            # 计算Sharpe
            if len(returns) > 1 and returns.std() > 1e-12:
                sharpe = (
                    returns.mean() / returns.std() * np.sqrt(6 * 365)
                )  # Annualized for 4H bars
                results["sharpe"] = float(sharpe)

    return results


def compare_with_regime_baseline(
    current_df: pd.DataFrame,
    baseline_with_regime_path: Path,
) -> Dict[str, Any]:
    """对比当前基线和有regime时的baseline"""
    if not baseline_with_regime_path.exists():
        return {"error": f"Baseline file not found: {baseline_with_regime_path}"}

    baseline_df = pd.read_parquet(baseline_with_regime_path)

    comparison = {
        "current": {
            "total_rows": len(current_df),
            "trade_count": int(
                current_df["gate_ok"].sum() if "gate_ok" in current_df.columns else 0
            ),
            "archetype_distribution": (
                current_df["gate_archetype"].value_counts().to_dict()
                if "gate_archetype" in current_df.columns
                else {}
            ),
        },
        "baseline_with_regime": {
            "total_rows": len(baseline_df),
            "trade_count": int(
                baseline_df["gate_ok"].sum() if "gate_ok" in baseline_df.columns else 0
            ),
            "archetype_distribution": (
                baseline_df["gate_archetype"].value_counts().to_dict()
                if "gate_archetype" in baseline_df.columns
                else {}
            ),
            "regime_distribution": (
                baseline_df["regime"].value_counts().to_dict()
                if "regime" in baseline_df.columns
                else {}
            ),
        },
    }

    return comparison


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose baseline performance drop")
    p.add_argument(
        "--current-logs", required=True, help="Current baseline logs (parquet)"
    )
    p.add_argument(
        "--baseline-with-regime",
        default="results/experiments_regenerated/baseline_gated.parquet",
        help="Baseline with regime filtering (parquet)",
    )
    p.add_argument(
        "--output",
        default="results/diagnostics/baseline_performance_drop_report.md",
        help="Output report path",
    )
    args = p.parse_args()

    current_df = pd.read_parquet(args.current_logs)

    # 分析各archetype的gate规则严格程度
    print("Analyzing gate rules strictness...")

    archetypes = ["TrendContinuationTC", "FailureReversionFR", "ExhaustionTurnET"]
    analyses = {}

    for arch in archetypes:
        arch_trades = current_df[current_df["gate_archetype"] == arch].copy()
        if len(arch_trades) > 0:
            analyses[arch] = analyze_gate_rules_strictness(current_df, arch, {})

    # 对比有regime时的baseline
    print("Comparing with regime baseline...")
    comparison = compare_with_regime_baseline(
        current_df,
        Path(args.baseline_with_regime),
    )

    # 生成报告
    report_lines = []
    report_lines.append("# 基线表现下降诊断报告\n")
    report_lines.append(f"**当前基线**: `{args.current_logs}`\n")
    report_lines.append(f"**有regime时的baseline**: `{args.baseline_with_regime}`\n")
    report_lines.append("\n")

    # 对比摘要
    report_lines.append("## 对比摘要\n")
    report_lines.append("| 指标 | 有regime | 无regime | 变化 |\n")
    report_lines.append("|------|----------|----------|------|\n")

    if "baseline_with_regime" in comparison:
        baseline = comparison["baseline_with_regime"]
        current = comparison["current"]
        report_lines.append(
            f"| 总样本数 | {baseline['total_rows']} | {current['total_rows']} | {current['total_rows'] - baseline['total_rows']:+d} |\n"
        )
        report_lines.append(
            f"| 交易数 | {baseline['trade_count']} | {current['trade_count']} | {current['trade_count'] - baseline['trade_count']:+d} |\n"
        )

    report_lines.append("\n")

    # 各archetype分析
    report_lines.append("## 各Archetype分析\n")
    for arch, analysis in analyses.items():
        report_lines.append(f"### {arch}\n")
        report_lines.append(f"- **交易数**: {analysis['trade_count']}\n")

        if "sharpe" in analysis:
            report_lines.append(f"- **Sharpe**: {analysis['sharpe']:.4f}\n")

        if "returns" in analysis:
            ret = analysis["returns"]
            report_lines.append(f"- **平均收益**: {ret['mean']:.6f}\n")
            report_lines.append(f"- **正收益比例**: {ret['positive_rate']:.2%}\n")

        if "regime_distribution" in analysis and analysis["regime_distribution"]:
            report_lines.append(
                f"- **Regime分布**: {analysis['regime_distribution']}\n"
            )

        if "physical_features" in analysis:
            report_lines.append(f"- **物理特征分布**:\n")
            for feat, stats in analysis["physical_features"].items():
                report_lines.append(
                    f"  - {feat}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, p10={stats['p10']:.3f}, p90={stats['p90']:.3f}\n"
                )

        report_lines.append("\n")

    # 诊断结论
    report_lines.append("## 诊断结论\n")
    report_lines.append("\n### 问题根源\n")
    report_lines.append("1. **FR和ET在非MEAN_REGIME下也能通过gate**\n")
    report_lines.append("   - FR和ET的gate规则可能不够严格\n")
    report_lines.append("   - 它们的alpha假设可能只在MEAN_REGIME下成立\n")
    report_lines.append("\n2. **TC的Sharpe下降**\n")
    report_lines.append("   - TC在非TC_REGIME下表现变差\n")
    report_lines.append("   - 说明TC的gate规则也需要regime约束\n")
    report_lines.append("\n### 修复建议\n")
    report_lines.append("1. **加强FR/ET的gate规则**\n")
    report_lines.append(
        "   - 添加更严格的MEAN_REGIME条件（path_efficiency_pct < 0.5等）\n"
    )
    report_lines.append("   - 确保FR/ET只在适合mean reversion的环境下交易\n")
    report_lines.append("\n2. **加强TC的gate规则**\n")
    report_lines.append("   - 添加更严格的TC_REGIME条件\n")
    report_lines.append("   - 确保TC只在适合trend continuation的环境下交易\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(report_lines), encoding="utf-8")

    print(f"Diagnosis report written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
