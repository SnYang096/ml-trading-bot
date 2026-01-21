#!/usr/bin/env python3
"""
测试 FR/ET 交易，使用放松的阈值。

这个脚本会：
1. 使用放松的 Regime 分类条件（增加 MEAN_REGIME 数量）
2. 使用放松的 Router 阈值（增加 MEAN mode 输出）
3. 生成带语义分桶的 E2E 报告
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Test FR/ET with relaxed thresholds.")
    p.add_argument("--preds", required=True, help="Predictions parquet directory")
    p.add_argument("--output-dir", default="/tmp", help="Output directory")
    p.add_argument(
        "--test-semantic-buckets",
        action="store_true",
        help="Test semantic score buckets",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("测试 FR/ET 交易（使用放松的阈值）")
    print("=" * 80)

    # Step 1: 诊断当前状态
    print("\n[1/4] 诊断当前过滤情况...")
    cmd = [
        sys.executable,
        "scripts/diagnose_fr_et_filtering.py",
        "--preds",
        str(args.preds),
        "--output-md",
        str(output_dir / "fr_et_diagnosis_before.md"),
    ]
    subprocess.run(cmd, check=False)

    # Step 2: 使用放松的阈值重新分类 Regime
    print("\n[2/4] 使用放松的 Regime 条件重新分类...")
    cmd = [
        sys.executable,
        "scripts/diagnose_fr_et_filtering.py",
        "--preds",
        str(args.preds),
        "--relax-regime",
        "--output-md",
        str(output_dir / "fr_et_diagnosis_relaxed_regime.md"),
    ]
    subprocess.run(cmd, check=False)

    # Step 3: 使用放松的 Router 阈值
    print("\n[3/4] 使用放松的 Router 阈值...")
    cmd = [
        sys.executable,
        "scripts/diagnose_fr_et_filtering.py",
        "--preds",
        str(args.preds),
        "--relax-regime",
        "--relax-router",
        "--output-md",
        str(output_dir / "fr_et_diagnosis_relaxed_all.md"),
    ]
    subprocess.run(cmd, check=False)

    # Step 4: 如果启用，测试语义分桶
    if args.test_semantic_buckets:
        print("\n[4/4] 测试语义分桶...")
        print("⚠️  需要先运行完整的 pipeline 生成带语义分数的 regime 文件")
        print(
            "   然后运行: mlbot rule diagnose-e2e-kpi --logs <logs> --regime <regime> --gate <gate>"
        )

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
    print(f"\n报告保存在: {output_dir}")
    print("  - fr_et_diagnosis_before.md: 原始状态")
    print("  - fr_et_diagnosis_relaxed_regime.md: 放松 Regime 条件")
    print("  - fr_et_diagnosis_relaxed_all.md: 放松所有条件")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
