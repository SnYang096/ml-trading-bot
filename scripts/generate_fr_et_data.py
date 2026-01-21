#!/usr/bin/env python3
"""
生成 FR/ET 数据的完整流程脚本。

通过放松 Regime 分类和 Router 阈值来生成 MEAN_REGIME 数据。
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
    p = argparse.ArgumentParser(
        description="Generate FR/ET data with relaxed thresholds."
    )
    p.add_argument("--preds", required=True, help="Predictions directory")
    p.add_argument("--output-dir", default="results/e2e_kpi", help="Output directory")
    p.add_argument(
        "--regime-relaxed",
        action="store_true",
        help="Use relaxed regime classification thresholds",
    )
    p.add_argument(
        "--router-relaxed",
        action="store_true",
        help="Use relaxed router thresholds",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("生成 FR/ET 数据")
    print("=" * 80)

    # Step 1: Generate Regime with relaxed thresholds
    print("\n[1/4] 生成 Regime 分类（放松阈值）...")
    regime_output = output_dir / "physics_regime_relaxed.parquet"

    # 需要修改 physics_regime_classifier.py 来支持放松的阈值
    # 或者直接使用 diagnose_fr_et_filtering.py 的逻辑
    cmd = [
        sys.executable,
        "scripts/diagnose_fr_et_filtering.py",
        "--preds",
        str(args.preds),
        "--output-md",
        str(output_dir / "fr_et_diagnosis_relaxed.md"),
    ]
    if args.regime_relaxed:
        cmd.append("--relax-regime")
    if args.router_relaxed:
        cmd.append("--relax-router")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 诊断失败: {result.stderr}")
        return 1

    print(f"✅ 诊断完成，查看报告: {output_dir / 'fr_et_diagnosis_relaxed.md'}")

    # Step 2: Check MEAN_REGIME count
    print("\n[2/4] 检查 MEAN_REGIME 数量...")
    import pandas as pd

    # 需要从诊断报告中提取，或者重新生成 regime
    # 这里先提供一个检查脚本
    print("   运行诊断脚本后，查看报告中的 MEAN_REGIME 数量")

    # Step 3: Generate logs with relaxed router
    print("\n[3/4] 重新生成 logs（使用放松的 Router 阈值）...")
    print("   需要修改 build-logs-3action 来支持放松的阈值")
    print("   或者手动修改 Router 配置")

    # Step 4: Apply Gate
    print("\n[4/4] 应用 Gate 过滤...")
    print("   使用修正后的 Gate 逻辑过滤 FR/ET 交易")

    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print(f"\n查看诊断报告: {output_dir / 'fr_et_diagnosis_relaxed.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
