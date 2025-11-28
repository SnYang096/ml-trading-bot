#!/usr/bin/env python3
"""
验证测试文件结构的脚本
检查测试文件是否有明显的逻辑错误
"""

import ast
import sys
from pathlib import Path


def check_test_file(file_path):
    """检查测试文件的结构"""
    print(f"\n检查文件: {file_path}")
    print("=" * 70)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 解析 AST
        tree = ast.parse(content, filename=str(file_path))

        # 检查导入
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        print(f"✅ 语法检查通过")
        print(f"📦 导入的模块: {', '.join(imports[:5])}...")

        # 检查测试类和方法
        test_classes = []
        test_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    test_classes.append(node.name)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith(
                            "test_"
                        ):
                            test_methods.append(f"{node.name}.{item.name}")

        print(f"📋 测试类: {len(test_classes)} 个")
        for cls in test_classes:
            print(f"   - {cls}")

        print(f"🧪 测试方法: {len(test_methods)} 个")
        for method in test_methods[:10]:
            print(f"   - {method}")
        if len(test_methods) > 10:
            print(f"   ... 还有 {len(test_methods) - 10} 个")

        return True

    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    test_files = [
        project_root / "tests" / "test_hurst_features_improved.py",
        project_root / "tests" / "test_garch_evt_features.py",
    ]

    all_passed = True
    for test_file in test_files:
        if test_file.exists():
            if not check_test_file(test_file):
                all_passed = False
        else:
            print(f"⚠️  文件不存在: {test_file}")

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有测试文件结构检查通过")
        print("\n注意: 这只是结构检查，实际运行需要安装依赖包:")
        print("  - numpy")
        print("  - pandas")
        print("  - arch (可选，用于 GARCH 特征)")
        print("  - scipy (可选，用于 EVT 特征)")
    else:
        print("❌ 部分测试文件有问题")
        sys.exit(1)


if __name__ == "__main__":
    main()
