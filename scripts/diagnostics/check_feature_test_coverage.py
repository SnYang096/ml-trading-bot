#!/usr/bin/env python3
"""
检查所有特征相关测试文件的四种关键测试覆盖情况

使用方法:
    python tests/check_feature_test_coverage.py
    python tests/check_feature_test_coverage.py --detailed
    python tests/check_feature_test_coverage.py --missing-only
"""

import re
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def get_feature_test_files() -> list[str]:
    """
    Discover feature-related test files.

    Repo has migrated most feature tests under:
      - tests/features/test_*.py
    """
    root = Path(__file__).resolve().parents[1]
    patterns = [
        root / "tests" / "features" / "test_*.py",
        root / "tests" / "test_*.py",
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend([str(p) for p in sorted(pat.parent.glob(pat.name))])
    # de-dup preserve order
    out: list[str] = []
    seen = set()
    for f in files:
        name = str(f)
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


# 四种关键测试的关键词模式
TEST_PATTERNS = {
    "1. 无未来函数测试": {
        "keywords": [
            r"def\s+test.*no.*future.*leak",
            r"def\s+test.*future.*leak",
            r"def\s+test.*lookahead",
            r"def\s+test.*causality.*no.*future",
            r"def\s+test.*rolling.*window.*no.*lookahead",
        ],
        "priority": "⭐⭐⭐⭐⭐",
        "required": True,
        "icon": "🔴",
    },
    "2. 多资产归一化测试": {
        "keywords": [
            r"def\s+test.*multi.*asset.*normalization",
            r"def\s+test.*normalization.*multi.*asset",
            r"def\s+test.*multi.*asset.*comparability",
            r"def\s+test.*cross.*asset",
        ],
        "priority": "⭐⭐⭐⭐",
        "required": True,
        "icon": "🟠",
    },
    "3. 流式vs批量一致性": {
        "keywords": [
            r"def\s+test.*streaming.*batch",
            r"def\s+test.*batch.*streaming",
            r"def\s+test.*streaming.*consistency",
            r"def\s+test.*streaming.*correctness",
        ],
        "priority": "⭐⭐⭐⭐",
        "required": True,
        "icon": "🟡",
    },
    "4. lag衰减平滑测试": {
        "keywords": [
            r"def\s+test.*lag.*decay",
            r"def\s+test.*lag.*correlation",
            r"def\s+test.*autocorrelation.*decay",
            r"def\s+test.*correlation.*decay",
        ],
        "priority": "⭐⭐⭐",
        "required": False,
        "icon": "🟢",
    },
}


def check_file(test_file):
    """检查单个文件的测试覆盖情况"""
    if not os.path.exists(test_file):
        return None

    file_name = os.path.basename(test_file)

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return {"error": str(e)}

    results = {}
    for test_type, config in TEST_PATTERNS.items():
        found_tests = []
        for pattern in config["keywords"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_tests.extend(matches)

        found_tests = list(set([t.strip() for t in found_tests if t]))
        results[test_type] = {
            "found": len(found_tests) > 0,
            "count": len(found_tests),
            "tests": found_tests,
        }

    return results


def print_summary(
    results,
    *,
    detailed: bool = False,
    missing_only: bool = False,
    files: list[str] | None = None,
):
    """打印汇总报告"""
    print("=" * 100)
    print("所有特征相关测试文件的四种关键测试覆盖情况")
    print("=" * 100)
    print()

    # 统计每个测试类型的覆盖情况
    test_type_stats = defaultdict(lambda: {"covered": 0, "missing": 0, "files": []})

    for test_file, file_results in results.items():
        if file_results is None or "error" in file_results:
            continue

        for test_type, test_result in file_results.items():
            if test_result["found"]:
                test_type_stats[test_type]["covered"] += 1
                test_type_stats[test_type]["files"].append(test_file)
            else:
                test_type_stats[test_type]["missing"] += 1

    # 打印每个文件的详细情况
    if detailed:
        print("📄 各文件详细情况")
        print("-" * 100)
        print()

        for test_file in files or []:
            file_name = os.path.basename(test_file)
            file_results = results.get(file_name)

            if file_results is None:
                print(f"❌ {file_name}: 文件不存在")
                continue

            if "error" in file_results:
                print(f"❌ {file_name}: {file_results['error']}")
                continue

            # 统计覆盖情况
            covered_count = sum(1 for r in file_results.values() if r["found"])
            total_count = len(file_results)

            status_icon = (
                "✅"
                if covered_count == total_count
                else "⚠️" if covered_count > 0 else "❌"
            )
            print(f"{status_icon} {file_name} ({covered_count}/{total_count})")

            for test_type, test_result in file_results.items():
                config = TEST_PATTERNS[test_type]
                icon = config["icon"]
                status = "✅" if test_result["found"] else "❌"
                print(f"   {icon} {status} {test_type}: {test_result['count']} 个测试")
                if detailed and test_result["tests"]:
                    for test in test_result["tests"][:3]:
                        print(f"      - {test}")
                    if len(test_result["tests"]) > 3:
                        print(f"      ... 还有 {len(test_result['tests']) - 3} 个")
            print()

    # 打印汇总统计
    print("=" * 100)
    print("📊 汇总统计")
    print("=" * 100)
    print()

    for test_type, config in TEST_PATTERNS.items():
        stats = test_type_stats[test_type]
        total = stats["covered"] + stats["missing"]
        coverage = (stats["covered"] / total * 100) if total > 0 else 0

        icon = config["icon"]
        priority = config["priority"]
        required = "必须" if config["required"] else "可选"

        print(f"{icon} {test_type} ({priority}, {required})")
        print(f"   覆盖率: {stats['covered']}/{total} ({coverage:.1f}%)")

        if missing_only and stats["missing"] > 0:
            print(f"   ❌ 缺失文件 ({stats['missing']} 个):")
            missing_files = []
            for test_file in files or []:
                file_name = os.path.basename(test_file)
                file_results = results.get(file_name)
                if file_results and test_type in file_results:
                    if not file_results[test_type]["found"]:
                        missing_files.append(file_name)

            for f in missing_files[:10]:
                print(f"      - {f}")
            if len(missing_files) > 10:
                print(f"      ... 还有 {len(missing_files) - 10} 个文件")

        print()

    # 打印最需要补充的文件
    if not missing_only:
        print("=" * 100)
        print("🎯 最需要补充的文件（按缺失测试数量排序）")
        print("=" * 100)
        print()

        file_missing_count = defaultdict(int)
        for test_file in files or []:
            file_name = os.path.basename(test_file)
            file_results = results.get(file_name)
            if file_results and "error" not in file_results:
                for test_type, config in TEST_PATTERNS.items():
                    if config["required"] and not file_results[test_type]["found"]:
                        file_missing_count[file_name] += 1

        sorted_files = sorted(
            file_missing_count.items(), key=lambda x: x[1], reverse=True
        )

        for file_name, missing_count in sorted_files[:10]:
            print(f"🔴 {file_name}: 缺失 {missing_count} 个必须测试")
            file_results = results[file_name]
            for test_type, config in TEST_PATTERNS.items():
                if config["required"] and not file_results[test_type]["found"]:
                    print(f"   - {config['icon']} {test_type}")


def main():
    parser = argparse.ArgumentParser(description="检查特征测试覆盖情况")
    parser.add_argument("--detailed", action="store_true", help="显示详细信息")
    parser.add_argument("--missing-only", action="store_true", help="只显示缺失的测试")
    args = parser.parse_args()

    files = get_feature_test_files()
    # 检查所有文件
    results = {}
    for test_file in files:
        file_name = os.path.basename(test_file)
        results[file_name] = check_file(test_file)

    # 打印报告
    print_summary(
        results, detailed=args.detailed, missing_only=args.missing_only, files=files
    )


if __name__ == "__main__":
    main()
