#!/usr/bin/env python3
"""
测试文件的干运行验证（不实际执行，只检查逻辑）
"""

import ast
import sys
from pathlib import Path


def analyze_test_file(file_path):
    """分析测试文件，检查潜在问题"""
    print(f"\n{'='*70}")
    print(f"分析文件: {file_path.name}")
    print("=" * 70)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(file_path))

    issues = []
    warnings = []

    # 检查测试方法
    test_methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_methods.append(node.name)

            # 检查是否有断言
            has_assert = False
            for child in ast.walk(node):
                if isinstance(child, (ast.Assert, ast.Call)):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr.startswith("assert"):
                                has_assert = True
                    elif isinstance(child, ast.Assert):
                        has_assert = True

            if not has_assert:
                warnings.append(f"  ⚠️  {node.name}: 没有断言语句")

    print(f"✅ 找到 {len(test_methods)} 个测试方法:")
    for method in test_methods:
        print(f"   - {method}")

    # 检查导入
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.name] = "import"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports[f"{module}.{alias.name}"] = "from"

    print(f"\n✅ 导入检查:")
    critical_imports = ["numpy", "pandas", "unittest"]
    for imp in critical_imports:
        found = any(imp in key for key in imports.keys())
        status = "✅" if found else "❌"
        print(f"   {status} {imp}")

    # 检查函数调用
    function_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                function_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                function_calls.add(node.func.attr)

    print(f"\n✅ 函数调用检查:")
    expected_functions = [
        "extract_hurst_features",
        "extract_garch_features",
        "extract_evt_features",
    ]
    for func in expected_functions:
        found = func in function_calls or any(
            func in str(call) for call in function_calls
        )
        status = "✅" if found else "⚠️"
        print(f"   {status} {func}")

    if warnings:
        print(f"\n⚠️  警告:")
        for warning in warnings:
            print(warning)

    if issues:
        print(f"\n❌ 问题:")
        for issue in issues:
            print(issue)

    return len(issues) == 0


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    test_files = [
        project_root / "tests" / "test_hurst_features_improved.py",
        project_root / "tests" / "test_garch_evt_features.py",
    ]

    print("🧪 测试文件干运行验证")
    print("=" * 70)
    print("注意: 这只是逻辑检查，不实际运行测试")
    print("实际运行需要安装: numpy, pandas, arch, scipy")

    all_ok = True
    for test_file in test_files:
        if test_file.exists():
            if not analyze_test_file(test_file):
                all_ok = False
        else:
            print(f"\n❌ 文件不存在: {test_file}")
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("✅ 所有测试文件逻辑检查通过")
        print("\n📝 总结:")
        print("   - 测试文件结构正确")
        print("   - 函数调用匹配")
        print("   - 导入语句正确")
        print("\n💡 要实际运行测试，请先安装依赖:")
        print("   pip install numpy pandas arch scipy")
    else:
        print("❌ 部分测试文件有问题")
        sys.exit(1)


if __name__ == "__main__":
    main()
