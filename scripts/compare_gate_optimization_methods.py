#!/usr/bin/env python3
"""
对比Hard-Gate System和渐进式优化的结果

使用方法:
    python scripts/compare_gate_optimization_methods.py \
        --gated-logs results/pipeline_with_reflexivity_2024_full/logs_execution_gated.parquet \
        --raw-logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
        --output-dir results/gate_optimization_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_optimization_results(result_path: Path) -> Dict[str, Any]:
    """加载优化结果"""
    if not result_path.exists():
        return {}

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results(
    hard_gate_results: Dict[str, Any],
    progressive_results: Dict[str, Any],
) -> Dict[str, Any]:
    """对比两种优化方法的结果"""
    comparison = {
        "hard_gate": {
            "total_rules": len(hard_gate_results),
            "rules_by_priority": {},
            "avg_robustness": 0.0,
            "avg_trade_rate": 0.0,
        },
        "progressive": {
            "total_rules": len(progressive_results),
            "rules_by_priority": {},
            "avg_robustness": 0.0,
            "avg_trade_rate": 0.0,
        },
        "differences": [],
    }

    # 统计Hard-Gate结果
    if hard_gate_results:
        robustness_scores = []
        trade_rates = []
        for key, result in hard_gate_results.items():
            if isinstance(result, dict):
                priority = result.get("priority", 999)
                robustness = result.get("robustness_score", 0.0)
                trade_rate = result.get("trade_rate", 0.0)

                if priority not in comparison["hard_gate"]["rules_by_priority"]:
                    comparison["hard_gate"]["rules_by_priority"][priority] = 0
                comparison["hard_gate"]["rules_by_priority"][priority] += 1

                if robustness > 0:
                    robustness_scores.append(robustness)
                if trade_rate > 0:
                    trade_rates.append(trade_rate)

        if robustness_scores:
            comparison["hard_gate"]["avg_robustness"] = sum(robustness_scores) / len(
                robustness_scores
            )
        if trade_rates:
            comparison["hard_gate"]["avg_trade_rate"] = sum(trade_rates) / len(
                trade_rates
            )

    # 统计渐进式优化结果
    if progressive_results:
        robustness_scores = []
        trade_rates = []
        for key, result in progressive_results.items():
            if isinstance(result, dict):
                robustness = result.get("robustness_score", 0.0)
                trade_rate = result.get("trade_rate", 0.0)

                if robustness > 0:
                    robustness_scores.append(robustness)
                if trade_rate > 0:
                    trade_rates.append(trade_rate)

        if robustness_scores:
            comparison["progressive"]["avg_robustness"] = sum(robustness_scores) / len(
                robustness_scores
            )
        if trade_rates:
            comparison["progressive"]["avg_trade_rate"] = sum(trade_rates) / len(
                trade_rates
            )

    # 对比相同规则的阈值差异
    common_rules = set(hard_gate_results.keys()) & set(progressive_results.keys())
    for rule_key in common_rules:
        hg_result = hard_gate_results[rule_key]
        prog_result = progressive_results[rule_key]

        if isinstance(hg_result, dict) and isinstance(prog_result, dict):
            hg_threshold = hg_result.get("recommended_threshold") or hg_result.get(
                "final_threshold"
            )
            prog_threshold = prog_result.get(
                "recommended_threshold"
            ) or prog_result.get("final_threshold")

            if hg_threshold is not None and prog_threshold is not None:
                diff = abs(hg_threshold - prog_threshold)
                comparison["differences"].append(
                    {
                        "rule": rule_key,
                        "hard_gate_threshold": hg_threshold,
                        "progressive_threshold": prog_threshold,
                        "difference": diff,
                        "hard_gate_robustness": hg_result.get("robustness_score", 0.0),
                        "progressive_robustness": prog_result.get(
                            "robustness_score", 0.0
                        ),
                    }
                )

    return comparison


def generate_report(
    comparison: Dict[str, Any],
    output_dir: Path,
) -> None:
    """生成对比报告"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON报告
    json_path = output_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Markdown报告
    md_path = output_dir / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gate优化方法对比报告\n\n")
        f.write("## 概述\n\n")
        f.write(
            f"- **Hard-Gate System**: {comparison['hard_gate']['total_rules']} 个规则\n"
        )
        f.write(
            f"- **渐进式优化**: {comparison['progressive']['total_rules']} 个规则\n\n"
        )

        f.write("## 性能对比\n\n")
        f.write("| 指标 | Hard-Gate System | 渐进式优化 | 差异 |\n")
        f.write("|------|------------------|------------|------|\n")
        f.write(
            f"| 平均Robustness Score | {comparison['hard_gate']['avg_robustness']:.4f} | {comparison['progressive']['avg_robustness']:.4f} | {comparison['hard_gate']['avg_robustness'] - comparison['progressive']['avg_robustness']:.4f} |\n"
        )
        f.write(
            f"| 平均Trade Rate | {comparison['hard_gate']['avg_trade_rate']:.4f} | {comparison['progressive']['avg_trade_rate']:.4f} | {comparison['hard_gate']['avg_trade_rate'] - comparison['progressive']['avg_trade_rate']:.4f} |\n\n"
        )

        f.write("## Hard-Gate System规则分布（按优先级）\n\n")
        for priority in sorted(comparison["hard_gate"]["rules_by_priority"].keys()):
            count = comparison["hard_gate"]["rules_by_priority"][priority]
            f.write(f"- **Priority {priority}**: {count} 个规则\n")
        f.write("\n")

        if comparison["differences"]:
            f.write("## 阈值差异分析\n\n")
            f.write(
                "| 规则 | Hard-Gate阈值 | 渐进式阈值 | 差异 | Hard-Gate Robustness | 渐进式 Robustness |\n"
            )
            f.write(
                "|------|---------------|------------|------|---------------------|------------------|\n"
            )

            # 按差异排序
            sorted_diffs = sorted(
                comparison["differences"], key=lambda x: x["difference"], reverse=True
            )
            for diff in sorted_diffs[:20]:  # 只显示前20个差异最大的
                f.write(
                    f"| {diff['rule']} | {diff['hard_gate_threshold']:.4f} | {diff['progressive_threshold']:.4f} | {diff['difference']:.4f} | {diff['hard_gate_robustness']:.4f} | {diff['progressive_robustness']:.4f} |\n"
                )
            f.write("\n")

        f.write("## 结论\n\n")
        if (
            comparison["hard_gate"]["avg_robustness"]
            > comparison["progressive"]["avg_robustness"]
        ):
            f.write("- **Hard-Gate System**在Robustness Score方面表现更好\n")
        else:
            f.write("- **渐进式优化**在Robustness Score方面表现更好\n")

        if (
            comparison["hard_gate"]["avg_trade_rate"]
            > comparison["progressive"]["avg_trade_rate"]
        ):
            f.write("- **Hard-Gate System**在Trade Rate方面表现更好\n")
        else:
            f.write("- **渐进式优化**在Trade Rate方面表现更好\n")

    print(f"✅ 对比报告已生成:")
    print(f"   - JSON: {json_path}")
    print(f"   - Markdown: {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对比Hard-Gate System和渐进式优化的结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gated-logs",
        required=True,
        help="已应用gate的logs文件（parquet）",
    )
    parser.add_argument(
        "--raw-logs",
        required=True,
        help="原始logs文件（parquet）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    parser.add_argument(
        "--output-dir",
        default="results/gate_optimization_comparison",
        help="输出目录",
    )
    parser.add_argument(
        "--hard-gate-result",
        default=None,
        help="Hard-Gate System优化结果文件（可选，如果未提供则运行优化）",
    )
    parser.add_argument(
        "--progressive-result",
        default=None,
        help="渐进式优化结果文件（可选，如果未提供则运行优化）",
    )
    parser.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore根目录",
    )
    parser.add_argument(
        "--feature-store-layer",
        default=None,
        help="FeatureStore layer名称",
    )
    parser.add_argument(
        "--timeframe",
        default="240T",
        help="时间框架（如 240T）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="开始日期（可选）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期（可选）",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载或运行Hard-Gate System优化
    if args.hard_gate_result:
        hard_gate_path = Path(args.hard_gate_result)
    else:
        hard_gate_path = output_dir / "hard_gate_optimization.json"
        print("📊 运行Hard-Gate System优化...")
        import subprocess

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "optimize_gate_plateau_hard_gate.py"),
            "--gated-logs",
            args.gated_logs,
            "--raw-logs",
            args.raw_logs,
            "--execution-archetypes",
            args.execution_archetypes,
            "--output",
            str(hard_gate_path),
        ]
        if args.feature_store_layer:
            cmd.extend(
                [
                    "--feature-store-root",
                    args.feature_store_root,
                    "--feature-store-layer",
                    args.feature_store_layer,
                    "--timeframe",
                    args.timeframe,
                ]
            )
            if args.start_date:
                cmd.extend(["--start-date", args.start_date])
            if args.end_date:
                cmd.extend(["--end-date", args.end_date])

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print("❌ Hard-Gate System优化失败")
            return 1

    # 加载或运行渐进式优化
    if args.progressive_result:
        progressive_path = Path(args.progressive_result)
    else:
        progressive_path = output_dir / "progressive_optimization.json"
        print("📊 运行渐进式优化...")
        import subprocess

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "optimize_gate_plateau_progressive.py"),
            "--gated-logs",
            args.gated_logs,
            "--raw-logs",
            args.raw_logs,
            "--execution-archetypes",
            args.execution_archetypes,
            "--output",
            str(progressive_path),
            "--target-trades",
            "200",
        ]
        if args.feature_store_layer:
            cmd.extend(
                [
                    "--feature-store-root",
                    args.feature_store_root,
                    "--feature-store-layer",
                    args.feature_store_layer,
                    "--timeframe",
                    args.timeframe,
                ]
            )
            if args.start_date:
                cmd.extend(["--start-date", args.start_date])
            if args.end_date:
                cmd.extend(["--end-date", args.end_date])

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print("❌ 渐进式优化失败")
            return 1

    # 加载结果
    print("📊 加载优化结果...")
    hard_gate_results = load_optimization_results(hard_gate_path)
    progressive_results = load_optimization_results(progressive_path)

    if not hard_gate_results:
        print("⚠️  未找到Hard-Gate System优化结果")
    if not progressive_results:
        print("⚠️  未找到渐进式优化结果")

    if not hard_gate_results and not progressive_results:
        print("❌ 没有可对比的结果")
        return 1

    # 对比结果
    print("📊 对比优化结果...")
    comparison = compare_results(hard_gate_results, progressive_results)

    # 生成报告
    generate_report(comparison, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
