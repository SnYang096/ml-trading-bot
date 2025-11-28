#!/usr/bin/env python3
"""
检查代码库中是否有使用全局归一化（window=None）的情况

全局归一化在时序数据中会导致未来信息泄露，应该避免使用。
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple


def find_python_files(root_dir: Path) -> List[Path]:
    """查找所有Python文件"""
    python_files = []
    for path in root_dir.rglob("*.py"):
        # 排除测试文件和脚本文件（测试文件中的全局归一化是用于测试的）
        if "test" in str(path) or "scripts" in str(path):
            continue
        # 排除__pycache__和虚拟环境
        if "__pycache__" in str(path) or ".venv" in str(path) or "venv" in str(path):
            continue
        python_files.append(path)
    return python_files


def check_global_normalization(file_path: Path) -> List[Tuple[int, str]]:
    """检查文件中是否有使用全局归一化的情况"""
    issues = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # 检查是否有调用 normalize_by_group 或 normalize_dataframe
        for i, line in enumerate(lines, 1):
            # 检查是否有调用这些函数
            if "normalize_by_group" in line or "normalize_dataframe" in line:
                # 检查是否明确指定了 window=None 或没有指定 window 参数
                # 如果指定了 window=数字，则没问题

                # 情况1: window=None (明确指定)
                if "window=None" in line:
                    issues.append((i, f"明确使用全局归一化: {line.strip()}"))

                # 情况2: 调用函数但没有指定 window 参数（默认就是 None）
                # 排除函数定义（def 开头的行）
                if line.strip().startswith("def "):
                    continue

                # 检查是否是函数调用
                if "normalize_by_group(" in line or "normalize_dataframe(" in line:
                    # 检查这一行和后续几行是否有 window= 参数
                    window_found = False
                    check_lines = lines[
                        i - 1 : min(i + 5, len(lines))
                    ]  # 检查当前行和后续5行
                    for check_line in check_lines:
                        if "window=" in check_line:
                            window_found = True
                            # 如果 window 是数字，则没问题
                            if "window=" in check_line and any(
                                c.isdigit()
                                for c in check_line.split("window=")[1]
                                .split(",")[0]
                                .split(")")[0]
                            ):
                                break
                            # 如果 window=None，已经在上面检查了
                            break

                    # 如果没有找到 window 参数，可能是使用默认值（None）
                    if not window_found:
                        # 检查函数调用是否完整（在同一行）
                        if ")" in line:
                            issues.append(
                                (
                                    i,
                                    f"可能使用默认全局归一化（未指定window参数）: {line.strip()}",
                                )
                            )
                        else:
                            # 多行调用，标记为需要检查
                            issues.append(
                                (
                                    i,
                                    f"多行调用，需要检查是否指定window参数: {line.strip()}",
                                )
                            )

    except Exception as e:
        issues.append((0, f"解析文件时出错: {e}"))

    return issues


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    print("=" * 80)
    print("检查代码库中是否有使用全局归一化（window=None）的情况")
    print("=" * 80)
    print()

    python_files = find_python_files(src_dir)
    print(f"找到 {len(python_files)} 个Python文件需要检查")
    print()

    total_issues = 0
    files_with_issues = []

    for file_path in python_files:
        issues = check_global_normalization(file_path)
        if issues:
            files_with_issues.append((file_path, issues))
            total_issues += len(issues)

    if files_with_issues:
        print("⚠️  发现以下文件可能使用了全局归一化：")
        print()
        for file_path, issues in files_with_issues:
            rel_path = file_path.relative_to(project_root)
            print(f"📄 {rel_path}")
            for line_num, issue in issues:
                print(f"   行 {line_num}: {issue}")
            print()

        print("=" * 80)
        print(f"总计: {len(files_with_issues)} 个文件，{total_issues} 个潜在问题")
        print("=" * 80)
        print()
        print("建议：")
        print("1. 检查这些调用是否确实需要全局归一化（如EDA场景）")
        print("2. 如果是时序数据，应该使用滚动归一化（window=252等）")
        print("3. 如果确实需要全局归一化，应该设置 warn_global=False 以明确意图")
    else:
        print("✅ 未发现使用全局归一化的情况！")
        print("所有调用都明确指定了 window 参数，或者使用了滚动归一化。")

    print()


if __name__ == "__main__":
    main()
