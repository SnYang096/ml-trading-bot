#!/usr/bin/env python3
"""
分析optional_blocks_enabled的去留

检查：
1. 当前TaskSpec中optional_blocks_enabled的使用情况
2. 自动推导是否覆盖所有需求
3. 评估去掉optional_blocks_enabled的影响
4. 提出简化方案
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Set, Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.auto_detect_compute_requirements import auto_detect_compute_requirements


def analyze_task_spec_usage() -> Dict[str, Any]:
    """分析所有TaskSpec中optional_blocks_enabled的使用情况"""
    tasks_dir = PROJECT_ROOT / "config/tasks"
    results = {}

    for task_file in tasks_dir.glob("*.yaml"):
        with open(task_file, "r", encoding="utf-8") as f:
            task_obj = yaml.safe_load(f) or {}

        fp_over = task_obj.get("feature_plan_overrides", {}) or {}
        ob_enabled = fp_over.get("optional_blocks_enabled", [])

        results[str(task_file.name)] = {
            "optional_blocks_enabled": (
                ob_enabled if isinstance(ob_enabled, list) else []
            ),
            "has_override": bool(fp_over),
        }

    return results


def check_auto_detection_coverage(task_spec_path: str) -> Dict[str, Any]:
    """检查自动推导是否覆盖所有需求"""
    try:
        auto_detected = auto_detect_compute_requirements(task_spec_path)

        # 读取TaskSpec获取用户显式指定的
        ts_path = Path(task_spec_path)
        if not ts_path.is_absolute():
            ts_path = PROJECT_ROOT / ts_path

        with open(ts_path, "r", encoding="utf-8") as f:
            ts_obj = yaml.safe_load(f) or {}

        fp_over = ts_obj.get("feature_plan_overrides", {}) or {}
        user_specified = set(fp_over.get("optional_blocks_enabled", []) or [])

        return {
            "auto_detected": sorted(auto_detected),
            "user_specified": sorted(user_specified),
            "coverage": sorted(auto_detected | user_specified),
            "missing_in_auto": sorted(user_specified - auto_detected),
            "extra_in_auto": sorted(auto_detected - user_specified),
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze optional_blocks_enabled usage")
    p.add_argument(
        "--task-spec",
        default="config/tasks/task_spec_highcap6_2024_202510.yaml",
        help="TaskSpec file to analyze",
    )
    p.add_argument(
        "--output",
        default="results/optional_blocks_enabled_analysis.json",
        help="Output JSON file",
    )
    args = p.parse_args()

    print("=" * 80)
    print("分析optional_blocks_enabled的去留")
    print("=" * 80)

    # 1. 分析所有TaskSpec的使用情况
    print("\n1. 所有TaskSpec中optional_blocks_enabled的使用情况:")
    print("-" * 80)
    task_usage = analyze_task_spec_usage()
    for task_name, info in task_usage.items():
        ob_enabled = info["optional_blocks_enabled"]
        print(f"  {task_name}:")
        if ob_enabled:
            print(f"    - optional_blocks_enabled: {ob_enabled}")
        else:
            print(f"    - optional_blocks_enabled: [] (空或未设置)")
        print(f"    - has_override: {info['has_override']}")

    # 2. 检查自动推导覆盖
    print("\n2. 自动推导覆盖情况:")
    print("-" * 80)
    coverage = check_auto_detection_coverage(args.task_spec)
    if "error" in coverage:
        print(f"  ❌ 错误: {coverage['error']}")
    else:
        print(f"  自动推导的blocks: {coverage['auto_detected']}")
        print(f"  用户显式指定的blocks: {coverage['user_specified']}")
        print(f"  最终启用的blocks: {coverage['coverage']}")
        if coverage["missing_in_auto"]:
            print(f"  ⚠️  用户指定但自动推导未覆盖: {coverage['missing_in_auto']}")
        if coverage["extra_in_auto"]:
            print(f"  ✅ 自动推导额外检测到: {coverage['extra_in_auto']}")

    # 3. 评估和建议
    print("\n3. 评估和建议:")
    print("-" * 80)

    if "error" not in coverage:
        if not coverage["missing_in_auto"]:
            print("  ✅ 自动推导完全覆盖用户需求")
            print(
                "  💡 建议: 可以移除用户显式指定的optional_blocks_enabled，完全依赖自动推导"
            )
        else:
            print(f"  ⚠️  自动推导未覆盖: {coverage['missing_in_auto']}")
            print(
                "  💡 建议: 保留optional_blocks_enabled用于模型训练需求（非gate/regime需求）"
            )

    print("\n  📝 总结:")
    print("    - optional_blocks_enabled有两个作用:")
    print("      1. 控制特征计算（gate/regime需求）- 现在由自动推导覆盖")
    print("      2. 定义额外blocks用于模型训练（非gate/regime需求）- 需要用户显式指定")
    print("    - 建议: 保留optional_blocks_enabled，但明确语义:")
    print("      - 自动推导: gate/regime需要的blocks（必须计算）")
    print("      - 用户显式指定: 模型训练需要的额外blocks（可选）")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "task_usage": task_usage,
        "auto_detection_coverage": coverage,
        "recommendation": {
            "keep_optional_blocks_enabled": True,
            "reason": "用于模型训练需求（非gate/regime需求）",
            "auto_detect_for": "gate/regime需求（必须计算）",
            "user_specify_for": "模型训练需求（可选）",
        },
    }

    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 分析结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
