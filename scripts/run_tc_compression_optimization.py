#!/usr/bin/env python3
"""
运行TC archetype压缩优化实验

从全松阈值开始，逐步收紧，找到最优压缩点（压缩过度交易）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行TC archetype压缩优化实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        default="results/tc_compression_optimization",
        help="输出目录",
    )
    parser.add_argument(
        "--compression-target-trade-rate",
        type=float,
        default=0.02,
        help="压缩目标trade_rate（例如0.02表示压缩到2%）",
    )
    parser.add_argument(
        "--compression-min-robustness",
        type=float,
        default=0.5,
        help="压缩过程中最低robustness_score要求",
    )
    parser.add_argument(
        "--compression-step",
        type=float,
        default=0.01,
        help="压缩收紧步长",
    )
    parser.add_argument(
        "--global-trade-budget",
        type=float,
        default=0.02,
        help="全局trade_rate生存约束",
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
        help="时间框架",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="开始日期",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="不使用Docker",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成gated-logs（使用全松阈值）
    print("📋 步骤1: 生成gated-logs（全松阈值）...")
    gated_logs = output_dir / "logs_execution_gated_baseline.parquet"

    # 运行gate应用（使用全松阈值）
    cmd_gate = [
        "mlbot",
        "rule",
        "apply-tree-gate",
        "--logs",
        args.raw_logs,
        "--out",
        str(gated_logs),
        "--execution-archetypes",
        args.execution_archetypes,
    ]

    if args.feature_store_layer:
        cmd_gate.extend(
            [
                "--features-store-layer",
                args.feature_store_layer,
                "--features-store-root",
                args.feature_store_root,
            ]
        )

    if args.no_docker:
        cmd_gate.append("--no-docker")

    result = subprocess.run(cmd_gate, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Gate应用失败: {result.stderr}")
        return 1

    print(f"✅ Gated logs生成: {gated_logs}")

    # 运行TC优化
    print("\n📋 步骤2: 运行TC压缩优化...")
    output_json = output_dir / "tc_optimization_compression.json"

    cmd_optimize = [
        "python",
        "scripts/optimize_gate_plateau_hard_gate.py",
        "--gated-logs",
        str(gated_logs),
        "--raw-logs",
        args.raw_logs,
        "--execution-archetypes",
        args.execution_archetypes,
        "--output",
        str(output_json),
        "--compression-mode",
        "--compression-target-trade-rate",
        str(args.compression_target_trade_rate),
        "--compression-min-robustness",
        str(args.compression_min_robustness),
        "--compression-step",
        str(args.compression_step),
        "--global-trade-budget",
        str(args.global_trade_budget),
        "--archetype-filter",
        "TC",
        "--multi-objective-strategy",
        "max_compression_efficiency",
    ]

    if args.feature_store_layer:
        cmd_optimize.extend(
            [
                "--feature-store-layer",
                args.feature_store_layer,
                "--feature-store-root",
                args.feature_store_root,
                "--timeframe",
                args.timeframe,
            ]
        )
        if args.start_date:
            cmd_optimize.extend(["--start-date", args.start_date])
        if args.end_date:
            cmd_optimize.extend(["--end-date", args.end_date])

    result = subprocess.run(cmd_optimize, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ TC优化失败: {result.stderr}")
        return 1

    print(f"✅ TC优化完成: {output_json}")
    print(f"\n📊 优化结果:")
    print(result.stdout)

    # 验证优化结果（可选）
    print("\n📋 步骤3: 验证优化结果...")
    print("   提示: 可以运行以下命令验证优化结果:")
    print(f"   python scripts/run_gate_optimization_experiments.py \\")
    print(f"       --raw-logs {args.raw_logs} \\")
    print(f"       --optimization-results {output_json} \\")
    print(f"       --output-dir {output_dir / 'validation'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
