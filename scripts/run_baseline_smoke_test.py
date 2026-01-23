#!/usr/bin/env python3
"""
基线测试和冒烟测试脚本

运行完整的gate检查流程，生成各archetype的KPI报告，输出基线性能指标。

输出:
  - results/baseline_smoke_test/baseline_kpi.md - 各archetype的KPI报告
  - results/baseline_smoke_test/baseline_kpi.json - JSON格式数据
  - results/baseline_smoke_test/logs_baseline.parquet - 基线logs文件

指标:
  - 每个archetype的Sharpe、交易数、胜率、平均收益
  - 多archetype同时触发的统计
  - CVD判断的效果统计
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description="Baseline smoke test for archetype performance"
    )
    p.add_argument(
        "--logs",
        required=True,
        help="Input logs file (parquet) - e.g., results/e2e_kpi/logs_3action_2024_2025.parquet",
    )
    p.add_argument(
        "--output-dir",
        default="results/baseline_smoke_test",
        help="Output directory for baseline reports",
    )
    p.add_argument(
        "--features-store-layer",
        required=True,
        help="FeatureStore layer name (e.g., nnmh_highcap6_240T_2024_202510_v2)",
    )
    p.add_argument(
        "--features-store-root",
        default="feature_store",
        help="FeatureStore root directory",
    )
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Path to execution_archetypes.yaml",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"Error: Logs file not found: {logs_path}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("Baseline Smoke Test")
    print("=" * 80)
    print(f"Input logs: {logs_path}")
    print(f"Output directory: {output_dir}")
    print(f"FeatureStore layer: {args.features_store_layer}")
    print()

    # Step 1: Run gate check
    gated_logs_path = output_dir / "logs_baseline.parquet"
    cmd_gate = [
        sys.executable,
        "scripts/apply_archetype_gate.py",
        "--logs",
        str(logs_path),
        "--out",
        str(gated_logs_path),
        "--features-store-layer",
        args.features_store_layer,
        "--features-store-root",
        args.features_store_root,
        "--execution-archetypes",
        args.execution_archetypes,
    ]
    exit_code = run_command(cmd_gate, "Step 1: Apply archetype gate rules")
    if exit_code != 0:
        print(f"Error: Gate check failed with exit code {exit_code}", file=sys.stderr)
        return exit_code

    # Step 2: Generate KPI report
    kpi_md_path = output_dir / "baseline_kpi.md"
    kpi_json_path = output_dir / "baseline_kpi.json"
    cmd_kpi = [
        sys.executable,
        "scripts/diagnose_e2e_kpi.py",
        "--logs",
        str(gated_logs_path),
        "--output-md",
        str(kpi_md_path),
        "--output-json",
        str(kpi_json_path),
    ]
    exit_code = run_command(cmd_kpi, "Step 2: Generate KPI report")
    if exit_code != 0:
        print(
            f"Error: KPI report generation failed with exit code {exit_code}",
            file=sys.stderr,
        )
        return exit_code

    # Step 3: Generate archetype performance report
    arch_perf_path = output_dir / "archetype_performance.md"
    cmd_arch = [
        sys.executable,
        "scripts/analyze_archetype_performance.py",
        "--logs",
        str(gated_logs_path),
        "--output",
        str(arch_perf_path),
    ]
    exit_code = run_command(cmd_arch, "Step 3: Generate archetype performance report")
    if exit_code != 0:
        print(
            f"Warning: Archetype performance report generation failed with exit code {exit_code}",
            file=sys.stderr,
        )
        # Don't fail if this step fails, as it's optional

    print("\n" + "=" * 80)
    print("Baseline Smoke Test Complete")
    print("=" * 80)
    print(f"Output files:")
    print(f"  - Gated logs: {gated_logs_path}")
    print(f"  - KPI report (MD): {kpi_md_path}")
    print(f"  - KPI report (JSON): {kpi_json_path}")
    if arch_perf_path.exists():
        print(f"  - Archetype performance: {arch_perf_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
