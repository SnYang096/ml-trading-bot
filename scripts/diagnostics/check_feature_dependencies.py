#!/usr/bin/env python3
"""
检查所有特征的依赖关系是否正确

检查：
1. 特征函数内部需要的列是否在 dependencies 或 required_columns 中声明
2. 是否有特征函数需要某些列但没有自动计算机制
3. 是否有类似 sr_strength_max 的依赖问题
"""

import sys
from pathlib import Path
import yaml
import ast
import re

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def extract_required_columns_from_code(func_code: str) -> set:
    """从代码中提取需要的列"""
    required = set()

    # 查找常见的列检查模式
    patterns = [
        r'if\s+["\'](\w+)["\']\s+not\s+in\s+.*\.columns',
        r'["\'](\w+)["\']\s+not\s+in\s+.*\.columns',
        r'\.get\(["\'](\w+)["\']',
        r'\[["\'](\w+)["\']\]',
        r"required_cols\s*=\s*\[(.*?)\]",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, func_code)
        for match in matches:
            if isinstance(match, str):
                # 处理列表
                if "[" in match:
                    cols = re.findall(r'["\'](\w+)["\']', match)
                    required.update(cols)
                else:
                    required.add(match)

    return required


def check_feature_dependencies():
    """检查所有特征的依赖关系"""
    print("=" * 80)
    print("特征依赖关系检查")
    print("=" * 80)

    # 加载特征配置
    feature_deps_path = project_root / "config" / "feature_dependencies.yaml"
    with open(feature_deps_path, "r", encoding="utf-8") as f:
        feature_deps = yaml.safe_load(f)

    features = feature_deps.get("features", {})

    print(f"\n📊 总特征数: {len(features)}")

    # 检查关键特征
    critical_features = [
        "sqs_hal_high",
        "sqs_hal_low",
        "sr_strength_max",
    ]

    issues = []

    for feature_name in critical_features:
        if feature_name not in features:
            continue

        feature_info = features[feature_name]
        print(f"\n{'='*80}")
        print(f"检查特征: {feature_name}")
        print(f"{'='*80}")

        dependencies = feature_info.get("dependencies", [])
        required_columns = feature_info.get("required_columns", [])
        compute_func = feature_info.get("compute_func", "")

        print(f"   依赖特征: {dependencies}")
        print(f"   必需列: {required_columns}")
        print(f"   计算函数: {compute_func}")

        # 检查函数实现
        try:
            # 尝试导入函数
            if "." in compute_func:
                module_path, func_name = compute_func.rsplit(".", 1)
                module = __import__(module_path, fromlist=[func_name])
                func = getattr(module, func_name)

                # 获取函数源码
                import inspect

                func_code = inspect.getsource(func)

                # 检查代码中的列检查
                if "atr" in func_code.lower() and "atr" not in required_columns:
                    # 检查是否有自动计算机制
                    if 'if "atr" not in' in func_code or "compute_atr" in func_code:
                        print(f"   ✅ 有 ATR 自动计算机制")
                    else:
                        print(f"   ⚠️  代码中使用 ATR 但可能没有自动计算")
                        issues.append(
                            {
                                "feature": feature_name,
                                "issue": "使用 ATR 但可能没有自动计算机制",
                                "code_snippet": func_code[:200],
                            }
                        )

                # 检查边界列
                boundary_cols = ["hal_high", "hal_low", "poc"]
                for col in boundary_cols:
                    if col in func_code.lower() and col not in required_columns:
                        if (
                            f'if "{col}" not in' in func_code
                            or f"add_poc_hal" in func_code
                        ):
                            print(f"   ✅ 有 {col} 自动计算机制")
                        else:
                            print(f"   ⚠️  代码中使用 {col} 但可能没有自动计算")
                            issues.append(
                                {
                                    "feature": feature_name,
                                    "issue": f"使用 {col} 但可能没有自动计算机制",
                                }
                            )

        except Exception as e:
            print(f"   ⚠️  无法检查函数实现: {e}")

    # 检查 sqs_hal_high 和 sqs_hal_low 的 ATR 依赖
    print(f"\n{'='*80}")
    print("检查 sqs_hal_high 和 sqs_hal_low 的 ATR 依赖")
    print(f"{'='*80}")

    for feature_name in ["sqs_hal_high", "sqs_hal_low"]:
        if feature_name not in features:
            continue

        feature_info = features[feature_name]
        compute_func = feature_info.get("compute_func", "")

        try:
            if "." in compute_func:
                module_path, func_name = compute_func.rsplit(".", 1)
                module = __import__(module_path, fromlist=[func_name])
                func = getattr(module, func_name)

                import inspect

                func_code = inspect.getsource(func)

                # 检查是否有 ATR 检查
                if (
                    'required_cols = ["high", "low", "close", "atr", "volume"]'
                    in func_code
                ):
                    print(f"   {feature_name}: 代码中检查 ATR 列")
                    if 'if "atr" not in' in func_code or "compute_atr" in func_code:
                        print(f"      ✅ 有 ATR 自动计算机制")
                    else:
                        print(
                            f"      ⚠️  没有 ATR 自动计算机制，如果 ATR 不存在会返回 0.0"
                        )
                        issues.append(
                            {
                                "feature": feature_name,
                                "issue": "需要 ATR 但没有自动计算机制",
                                "suggestion": "添加 ATR 自动计算，类似 sr_strength_max",
                            }
                        )
        except Exception as e:
            print(f"   ⚠️  无法检查 {feature_name}: {e}")

    # 总结
    print(f"\n{'='*80}")
    print("检查总结")
    print(f"{'='*80}")

    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"\n   {i}. {issue['feature']}:")
            print(f"      问题: {issue['issue']}")
            if "suggestion" in issue:
                print(f"      建议: {issue['suggestion']}")
    else:
        print(f"\n✅ 未发现明显的依赖问题")

    return issues


if __name__ == "__main__":
    check_feature_dependencies()
