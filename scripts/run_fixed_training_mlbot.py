#!/usr/bin/env python3
"""
通过 mlbot 命令依次执行：长时间段固定训练 → 短时间段固定训练。

每个周期内对每个策略执行：
  1. mlbot train final（固定训练 + OOS 回测）
  2. 生成 monthly_results.json（build_monthly_results_from_backtest）
  3. mlbot train export-rules-to-readme --generate-rules（imodels 规则导出到 README）
  4. mlbot train export-monthly（若脚本存在则导出月度结果到 README）

用法（在项目根目录）：
  python scripts/run_fixed_training_mlbot.py
  python scripts/run_fixed_training_mlbot.py --long-only   # 只跑长时间段
  python scripts/run_fixed_training_mlbot.py --short-only  # 只跑短时间段
  python scripts/run_fixed_training_mlbot.py --no-docker   # 不使用 Docker（默认 --no-docker）
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "compression_breakout",
    "sr_breakout",
    "trend_following",
]

SYMBOL = "BTCUSDT"
TIMEFRAME = "240T"
DATA_PATH = "data/parquet_data"
SEED = "42"

# 长时间段：训练 2023-01-01 → 2024-12-31，OOS 2025-01-01 → 2025-05-31
LONG_TRAIN_START = "2023-01-01"
LONG_TRAIN_END = "2024-12-31"
LONG_OOS_START = "2025-01-01"
LONG_OOS_END = "2025-05-31"
LONG_OUTPUT_BASE = "results/fixed_long"

# 短时间段：训练 2024-07-01 → 2024-12-31，OOS 2025-01-01 → 2025-05-31
SHORT_TRAIN_START = "2024-07-01"
SHORT_TRAIN_END = "2024-12-31"
SHORT_OOS_START = "2025-01-01"
SHORT_OOS_END = "2025-05-31"
SHORT_OUTPUT_BASE = "results/fixed_short"


def _mlbot(args: list[str], docker: bool = False) -> int:
    """调用 mlbot CLI。若未安装则用 python -m cli.main + PYTHONPATH=src。"""
    cmd = ["mlbot"] + args
    env = os.environ.copy()
    if docker:
        return subprocess.run(cmd, cwd=ROOT, env=env).returncode
    # 开发环境：未安装时从 src 运行
    try:
        return subprocess.run(cmd, cwd=ROOT, env=env).returncode
    except FileNotFoundError:
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "cli.main"] + args,
            cwd=ROOT,
            env=env,
        ).returncode


def run_train_final(
    strategy: str,
    start: str,
    end: str,
    holdout_start: str,
    holdout_end: str,
    output_base: str,
    docker: bool,
) -> int:
    """对单个策略执行 mlbot train final。"""
    config = f"config/strategies/{strategy}"
    output_root = f"{output_base}/{strategy}"
    Path(output_root).mkdir(parents=True, exist_ok=True)
    args = [
        "train",
        "final",
        "--config",
        config,
        "--symbol",
        SYMBOL,
        "--timeframe",
        TIMEFRAME,
        "--data-path",
        DATA_PATH,
        "--start-date",
        start,
        "--end-date",
        end,
        "--holdout-start-date",
        holdout_start,
        "--holdout-end-date",
        holdout_end,
        "--output-root",
        output_root,
        "--seed",
        SEED,
    ]
    if not docker:
        args.append("--no-docker")
    return _mlbot(args, docker=docker)


def run_build_monthly() -> int:
    """生成 fixed_long / fixed_short 下各策略的 monthly_results.json。"""
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_monthly_results_from_backtest.py"),
        ],
        cwd=ROOT,
    ).returncode


def run_export_rules_to_readme(
    strategy: str, start_date: str, end_date: str, docker: bool
) -> int:
    """复用 mlbot train export-rules-to-readme --generate-rules（imodels 规则 → README）。"""
    args = [
        "train",
        "export-rules-to-readme",
        "--strategy-config",
        f"config/strategies/{strategy}",
        "--features-yaml",
        f"config/strategies/{strategy}/features.yaml",
        "--generate-rules",
        "--symbol",
        SYMBOL,
        "--timeframe",
        TIMEFRAME,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
    ]
    if not docker:
        args.append("--no-docker")
    return _mlbot(args, docker=docker)


def run_export_monthly(results_dir: str, strategy: str, docker: bool) -> int:
    """mlbot train export-monthly（若脚本存在）。"""
    script = ROOT / "scripts" / "export_monthly_results_to_readme.py"
    if not script.exists():
        return 0  # 脚本不存在则跳过
    args = [
        "train",
        "export-monthly",
        "--results-dir",
        results_dir,
        "--strategy",
        strategy,
    ]
    if not docker:
        args.append("--no-docker")
    return _mlbot(args, docker=docker)


def run_cycle(
    name: str,
    start: str,
    end: str,
    oos_start: str,
    oos_end: str,
    output_base: str,
    docker: bool,
) -> dict:
    """执行一个完整周期：train final → build monthly → export-rules → export-monthly。"""
    summary = {}
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"  训练: {start} → {end} (OOS 前); OOS: {oos_start} → {oos_end}")
    print(f"  输出: {output_base}")
    print(f"{'='*80}\n")

    for strategy in STRATEGIES:
        print(f"  [train final] {strategy} ...")
        rc = run_train_final(
            strategy, start, end, oos_start, oos_end, output_base, docker
        )
        summary[strategy] = {"train_final": "ok" if rc == 0 else "fail"}
        if rc != 0:
            print(f"    ❌ {strategy} train final 失败")
            continue
        print(f"    ✅ {strategy} train final 完成")

    print("\n  [build monthly] ...")
    rc = run_build_monthly()
    if rc == 0:
        print("    ✅ monthly_results.json 已生成")
    else:
        print("    ⚠️ build monthly 失败或跳过")

    for strategy in STRATEGIES:
        if summary.get(strategy, {}).get("train_final") != "ok":
            continue
        print(f"  [export-rules-to-readme] {strategy} ...")
        rc = run_export_rules_to_readme(strategy, start, end, docker)
        summary[strategy]["export_rules"] = "ok" if rc == 0 else "fail"
        if rc == 0:
            print(f"    ✅ {strategy} export-rules-to-readme 完成")
        else:
            print(f"    ⚠️ {strategy} export-rules-to-readme 失败")

    for strategy in STRATEGIES:
        if summary.get(strategy, {}).get("train_final") != "ok":
            continue
        print(f"  [export-monthly] {strategy} ...")
        rc = run_export_monthly(output_base, strategy, docker)
        summary[strategy]["export_monthly"] = "ok" if rc == 0 else "skip/fail"
        if rc == 0:
            print(f"    ✅ {strategy} export-monthly 完成")
        else:
            print(f"    ⚠️ {strategy} export-monthly 跳过或失败")

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Run fixed training (long + short) via mlbot"
    )
    ap.add_argument(
        "--long-only", action="store_true", help="Only run long-window cycle"
    )
    ap.add_argument(
        "--short-only", action="store_true", help="Only run short-window cycle"
    )
    ap.add_argument(
        "--docker", action="store_true", help="Use Docker for train final (default: no)"
    )
    args = ap.parse_args()

    docker = args.docker
    do_long = args.long_only or (not args.long_only and not args.short_only)
    do_short = args.short_only or (not args.long_only and not args.short_only)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": timestamp, "long": {}, "short": {}}

    if do_long:
        report["long"] = run_cycle(
            "长时间段固定训练",
            LONG_TRAIN_START,
            LONG_TRAIN_END,
            LONG_OOS_START,
            LONG_OOS_END,
            LONG_OUTPUT_BASE,
            docker,
        )
    if do_short:
        report["short"] = run_cycle(
            "短时间段固定训练",
            SHORT_TRAIN_START,
            SHORT_TRAIN_END,
            SHORT_OOS_START,
            SHORT_OOS_END,
            SHORT_OUTPUT_BASE,
            docker,
        )

    out_file = ROOT / f"results/fixed_mlbot_summary_{timestamp}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n摘要已保存: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
