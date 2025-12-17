#!/usr/bin/env python3
"""
迁移 BaselineFeatureEngineer 的静态方法到模块级函数。

这个脚本会：
1. 读取 baseline_features.py
2. 把所有 @staticmethod 方法移到类外部
3. 给公开方法（不以 _ 开头）添加 @register_feature 装饰器
4. 删除类定义（类已无内容）
5. 写回文件
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_FILE = PROJECT_ROOT / "src/features/time_series/baseline_features.py"


def migrate():
    """执行迁移"""
    with open(BASELINE_FILE, "r") as f:
        content = f.read()
        lines = content.split("\n")

    # 找到类定义的起始位置
    class_start = None
    for i, line in enumerate(lines):
        if line.startswith("class BaselineFeatureEngineer"):
            class_start = i
            break

    if class_start is None:
        print("❌ 未找到 BaselineFeatureEngineer 类")
        return False

    # 找到类结束位置（下一个顶级定义或文件末尾）
    class_end = len(lines)
    for i in range(class_start + 1, len(lines)):
        # 检查是否是类外的顶级代码（不以空格开头，且不是空行或注释）
        if lines[i] and not lines[i].startswith(" ") and not lines[i].startswith("#"):
            # 但要排除类内部的文档字符串结束
            if (
                lines[i].startswith("class ")
                or lines[i].startswith("def ")
                or lines[i].startswith("__all__")
            ):
                class_end = i
                break

    print(f"类定义范围: 行 {class_start+1} - {class_end}")

    # 提取类之前的内容
    before_class = lines[:class_start]

    # 提取类之后的内容
    after_class = lines[class_end:]

    # 提取类内部的内容（包括文档字符串和方法）
    class_body = lines[class_start:class_end]

    # 解析类体，提取所有静态方法
    methods = []
    i = 1  # 跳过 class 定义行

    # 跳过类文档字符串
    while i < len(class_body):
        line = class_body[i]
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            # 找到文档字符串结束
            if line.strip().count('"""') >= 2 or line.strip().count("'''") >= 2:
                i += 1
            else:
                quote = '"""' if '"""' in line else "'''"
                i += 1
                while i < len(class_body) and quote not in class_body[i]:
                    i += 1
                i += 1
            break
        elif line.strip() and not line.strip().startswith("#"):
            break
        i += 1

    # 现在解析方法
    while i < len(class_body):
        line = class_body[i]

        if line.strip() == "@staticmethod":
            method_start = i
            # 找到 def 行
            j = i + 1
            while j < len(class_body) and not class_body[j].strip().startswith("def "):
                j += 1

            if j >= len(class_body):
                break

            # 提取方法名
            def_line = class_body[j]
            match = re.match(r"\s+def (\w+)\(", def_line)
            if not match:
                i += 1
                continue

            method_name = match.group(1)

            # 找到方法结束位置
            k = j + 1
            while k < len(class_body):
                next_line = class_body[k]
                # 方法结束条件：遇到另一个装饰器或 def（同级缩进）
                if (
                    next_line.strip().startswith("@")
                    and len(next_line) - len(next_line.lstrip()) == 4
                ):
                    break
                if (
                    next_line.strip().startswith("def ")
                    and len(next_line) - len(next_line.lstrip()) == 4
                ):
                    break
                k += 1

            # 提取方法体（去掉 @staticmethod 装饰器，减少缩进）
            method_lines = class_body[j:k]
            # 去掉4个空格的缩进
            method_code = []
            for ml in method_lines:
                if ml.startswith("    "):
                    method_code.append(ml[4:])
                else:
                    method_code.append(ml)

            methods.append(
                {
                    "name": method_name,
                    "code": "\n".join(method_code),
                    "is_private": method_name.startswith("_"),
                }
            )

            i = k
        else:
            i += 1

    print(f"提取了 {len(methods)} 个静态方法")

    # 统计公开和私有方法
    public_methods = [m for m in methods if not m["is_private"]]
    private_methods = [m for m in methods if m["is_private"]]
    print(f"  - 公开方法: {len(public_methods)}")
    print(f"  - 私有方法: {len(private_methods)}")

    # 构建新文件内容
    new_content_parts = []

    # 添加 import register_feature（如果不存在）
    import_line = "from src.features.registry import register_feature"
    header_content = "\n".join(before_class)
    if import_line not in header_content:
        # 在其他 import 后添加
        import_added = False
        new_before_class = []
        for line in before_class:
            new_before_class.append(line)
            if line.startswith("from src.features") and not import_added:
                # 检查下一行是否已经有这个 import
                if import_line not in "\n".join(before_class):
                    new_before_class.append(import_line)
                    import_added = True
        before_class = new_before_class

    new_content_parts.append("\n".join(before_class))

    # 添加分隔注释
    new_content_parts.append("\n\n# " + "=" * 77)
    new_content_parts.append("# Baseline Feature Functions")
    new_content_parts.append("# (migrated from BaselineFeatureEngineer @staticmethod)")
    new_content_parts.append("# " + "=" * 77 + "\n")

    # 添加所有方法（公开方法添加装饰器）
    for method in methods:
        if not method["is_private"]:
            # 添加 @register_feature 装饰器
            new_content_parts.append(
                f'\n@register_feature("{method["name"]}", category="baseline")'
            )
        else:
            new_content_parts.append("")
        new_content_parts.append(method["code"])

    # 添加类之后的内容
    if after_class:
        new_content_parts.append("\n" + "\n".join(after_class))

    new_content = "\n".join(new_content_parts)

    # 写回文件
    with open(BASELINE_FILE, "w") as f:
        f.write(new_content)

    print(f"✅ 迁移完成！写入 {len(new_content)} 字符到 {BASELINE_FILE}")
    return True


if __name__ == "__main__":
    migrate()
