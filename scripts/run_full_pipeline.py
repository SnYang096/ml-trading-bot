#!/usr/bin/env python3
"""
一键执行完整Pipeline工作流

自动执行所有步骤，每一步检查输出文件是否存在，记录日志，支持断点续传。

使用方法:
    python scripts/run_full_pipeline.py \
        --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
        --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
        --timeframe 240T \
        --start-date 2024-01-01 \
        --end-date 2024-12-31 \
        --model results/nnmultihead/.../model.pt \
        --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
        --run-id pipeline_2024_reflexivity_validation
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(
    cmd: List[str],
    log_file: Path,
    check_output: Optional[Path] = None,
    description: str = "",
) -> bool:
    """运行命令并记录日志"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志: {log_file}")
    if check_output:
        if check_output.exists():
            print(f"✅ 输出文件已存在: {check_output}，跳过此步骤")
            return True
        print(f"📝 将生成输出文件: {check_output}")
    print(f"{'='*60}\n")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        print(f"❌ 命令执行失败，退出码: {result.returncode}")
        print(f"查看日志: {log_file}")
        return False

    if check_output and not check_output.exists():
        print(f"⚠️ 警告: 预期的输出文件不存在: {check_output}")
        return False

    print(f"✅ 步骤完成")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="一键执行完整Pipeline工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 必需参数
    parser.add_argument("--task-spec", required=True, help="TaskSpec YAML文件路径")
    parser.add_argument("--symbols", required=True, help="交易对列表，逗号分隔")
    parser.add_argument("--timeframe", required=True, help="时间周期，如 240T")
    parser.add_argument("--start-date", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--model", required=True, help="模型文件路径 .pt")

    # 可选参数
    parser.add_argument(
        "--run-id",
        default=None,
        help="运行ID（默认使用时间戳）",
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
        "--data-path",
        default="data/parquet_data",
        help="原始数据目录",
    )
    parser.add_argument(
        "--returns-source",
        default="rr_execution",
        help="Returns计算方式",
    )
    parser.add_argument(
        "--warmup-months",
        type=int,
        default=1,
        help="FeatureStore构建时的warmup月数",
    )
    parser.add_argument(
        "--skip-featurestore",
        action="store_true",
        help="跳过FeatureStore构建步骤",
    )
    parser.add_argument(
        "--skip-reflexivity",
        action="store_true",
        help="跳过反身性特征添加步骤",
    )
    parser.add_argument(
        "--strategy",
        default="bpc",
        help="策略名称，如 bpc, htf",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="不使用Docker",
    )

    args = parser.parse_args()

    # 生成run_id
    if args.run_id is None:
        args.run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = Path(f"results/{args.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Pipeline工作流开始")
    print(f"运行ID: {args.run_id}")
    print(f"输出目录: {run_dir}")
    print(f"{'='*60}\n")

    docker_flag = [] if args.no_docker else []

    # 步骤0: FeatureStore构建
    if not args.skip_featurestore and args.feature_store_layer:
        fs_output = Path(args.feature_store_root) / args.feature_store_layer
        if not any(fs_output.glob("*/*/240T/*.parquet")):
            success = run_command(
                cmd=[
                    "python3",
                    "src/cli/main.py",
                    "nnmultihead",
                    "build-feature-store",
                    "--task-spec",
                    args.task_spec,
                    "--symbols",
                    args.symbols,
                    "--timeframe",
                    args.timeframe,
                    "--start-date",
                    args.start_date,
                    "--end-date",
                    args.end_date,
                    "--feature-store-root",
                    args.feature_store_root,
                    "--layer",
                    args.feature_store_layer,
                    "--warmup-months",
                    str(args.warmup_months),
                ]
                + docker_flag,
                log_file=run_dir / "featurestore_build.log",
                description="FeatureStore构建",
            )
            if not success:
                return 1
        else:
            print(f"✅ FeatureStore已存在，跳过构建步骤")

    # 步骤1: 模型预测
    preds_dir = run_dir / "preds"
    success = run_command(
        cmd=[
            "python3",
            "src/cli/main.py",
            "nnmultihead",
            "predict",
            "--task-spec",
            args.task_spec,
            "--symbols",
            args.symbols,
            "--timeframe",
            args.timeframe,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--model",
            args.model,
            "--feature-store-layer",
            args.feature_store_layer or "",
            "--feature-store-root",
            args.feature_store_root,
            "--output",
            str(preds_dir),
        ]
        + docker_flag,
        log_file=run_dir / "predict.log",
        check_output=preds_dir / f"preds_{args.symbols.split(',')[0]}.parquet",
        description="模型预测",
    )
    if not success:
        return 1

    # 步骤2: 构建Execution日志
    logs_file = run_dir / "logs_execution.parquet"
    success = run_command(
        cmd=[
            "python3",
            "src/cli/main.py",
            "nnmultihead",
            "build-execution-logs",
            "--preds",
            str(preds_dir),
            "--model",
            args.model,
            "--symbols",
            args.symbols,
            "--timeframe",
            args.timeframe,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--data-path",
            args.data_path,
            "--returns-source",
            args.returns_source,
            "--output",
            str(logs_file),
        ]
        + docker_flag,
        log_file=run_dir / "build_logs.log",
        check_output=logs_file,
        description="构建Execution日志",
    )
    if not success:
        return 1

    # 步骤3: 应用Gate过滤（regime 已内嵌在 gate 规则中）
    gated_file = run_dir / "logs_execution_gated.parquet"
    gate_cmd = [
        "python3",
        "src/cli/main.py",
        "gate",
        "apply-archetype",
        "--logs",
        str(logs_file),
        "--out",
        str(gated_file),
        "--features-store-layer",
        args.feature_store_layer or "",
        "--features-store-root",
        args.feature_store_root,
        "--strategies-root",
        "config/strategies",
        "--strategy",
        args.strategy,
    ] + docker_flag

    success = run_command(
        cmd=gate_cmd,
        log_file=run_dir / "gate.log",
        check_output=gated_file,
        description="应用Gate过滤",
    )
    if not success:
        return 1

    # 步骤4: 添加反身性特征（可选）
    exec_logs_dir = run_dir / "exec_logs"
    if not args.skip_reflexivity and args.feature_store_layer:
        success = run_command(
            cmd=[
                "python3",
                "scripts/add_reflexivity_features_to_logs.py",
                "--preds",
                str(preds_dir),
                "--logs",
                str(logs_file),
                "--out-dir",
                str(exec_logs_dir),
                "--feature-store-dir",
                args.feature_store_root,
                "--feature-store-layer",
                args.feature_store_layer,
                "--data-path",
                args.data_path,
                "--timeframe",
                args.timeframe,
                "--run-id",
                args.run_id,
                "--strategy-name",
                "pipeline-3action-e2e",
            ],
            log_file=run_dir / "add_reflexivity.log",
            check_output=exec_logs_dir / "features" / "2024-01.jsonl",
            description="添加反身性特征",
        )
        if not success:
            print("⚠️ 反身性特征添加失败，继续执行后续步骤")

    # 步骤5: 构建Stage Logs
    success = run_command(
        cmd=[
            "python3",
            "scripts/build_execution_log_stages.py",
            "--preds",
            str(preds_dir),
            "--logs",
            str(logs_file),
            "--gated-logs",
            str(gated_file),
            "--out-dir",
            str(exec_logs_dir),
            "--run-id",
            args.run_id,
            "--timeframe",
            args.timeframe,
            "--strategy-name",
            "pipeline-3action-e2e",
        ],
        log_file=run_dir / "build_stages.log",
        check_output=exec_logs_dir / "gate" / "2024-01.jsonl",
        description="构建Stage Logs",
    )
    if not success:
        return 1

    # 步骤6: 聚合Canonical Log
    canonical_file = run_dir / "execution_log.jsonl"
    success = run_command(
        cmd=[
            "python3",
            "scripts/aggregate_execution_log_stages.py",
            "--stage-dir",
            str(exec_logs_dir),
            "--out",
            str(canonical_file),
        ],
        log_file=run_dir / "aggregate.log",
        check_output=canonical_file,
        description="聚合Canonical Log",
    )
    if not success:
        return 1

    # 步骤7: 生成E2E KPI报告
    success = run_command(
        cmd=[
            "python3",
            "src/cli/main.py",
            "rule",
            "diagnose-e2e-kpi",
            "--logs",
            str(gated_file),
            "--gate",
            str(gated_file),
            "--output-md",
            str(run_dir / "e2e_kpi_report.md"),
            "--output-json",
            str(run_dir / "e2e_kpi_report.json"),
            "--no-regime-filter",
        ]
        + docker_flag,
        log_file=run_dir / "e2e_kpi.log",
        check_output=run_dir / "e2e_kpi_report.md",
        description="生成E2E KPI报告",
    )
    if not success:
        return 1

    print(f"\n{'='*60}")
    print(f"✅ Pipeline工作流完成")
    print(f"运行ID: {args.run_id}")
    print(f"输出目录: {run_dir}")
    print(f"Canonical Log: {canonical_file}")
    print(f"E2E报告: {run_dir / 'e2e_kpi_report.md'}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
