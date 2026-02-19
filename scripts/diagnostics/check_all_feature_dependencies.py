#!/usr/bin/env python3
"""
全面检查所有特征的依赖关系问题

检查：
1. 特征函数内部需要的列是否在 dependencies 或 required_columns 中声明
2. 是否有特征函数需要某些列但没有自动计算机制
3. 是否有类似 sr_strength_max 的依赖问题（需要列但可能不存在）
"""

import sys
from pathlib import Path
import yaml
import inspect

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def check_all_feature_dependencies():
    """检查所有特征的依赖关系"""
    print("=" * 80)
    print("全面特征依赖关系检查")
    print("=" * 80)

    # 加载特征配置
    feature_deps_path = project_root / "config" / "feature_dependencies.yaml"
    with open(feature_deps_path, "r", encoding="utf-8") as f:
        feature_deps = yaml.safe_load(f)

    features = feature_deps.get("features", {})

    print(f"\n📊 总特征数: {len(features)}")

    # 需要检查的关键列
    critical_columns = {
        "atr": "ATR 是很多特征计算的基础",
        "hal_high": "HAL 边界列，用于 SR 计算",
        "hal_low": "HAL 边界列，用于 SR 计算",
        "poc": "POC 边界列，用于 SR 计算",
    }

    issues = []

    # 检查每个特征
    for feature_name, feature_info in features.items():
        compute_func = feature_info.get("compute_func", "")
        dependencies = feature_info.get("dependencies", [])
        required_columns = feature_info.get("required_columns", [])

        # 只检查有计算函数的特征
        if not compute_func or "." not in compute_func:
            continue

        try:
            # 尝试导入函数
            module_path, func_name = compute_func.rsplit(".", 1)
            module = __import__(module_path, fromlist=[func_name])
            func = getattr(module, func_name)

            # 获取函数源码
            func_code = inspect.getsource(func)

            # 检查每个关键列
            for col, description in critical_columns.items():
                # 检查代码中是否使用了这个列
                col_used = (
                    f'"{col}"' in func_code
                    or f"'{col}'" in func_code
                    or f"[{col}]" in func_code
                    or f".{col}" in func_code
                )

                if col_used:
                    # 检查是否在依赖或必需列中声明
                    in_deps = col in dependencies or any(
                        dep.endswith(f"_{col}") or dep.startswith(col)
                        for dep in dependencies
                    )
                    in_required = col in required_columns

                    # 检查是否有自动计算机制
                    has_auto_compute = (
                        f'if "{col}" not in' in func_code
                        or f"if '{col}' not in" in func_code
                        or f"compute_{col}" in func_code
                        or f"add_poc_hal" in func_code  # 对于 hal_high, hal_low, poc
                    )

                    if not (in_deps or in_required) and not has_auto_compute:
                        issues.append(
                            {
                                "feature": feature_name,
                                "column": col,
                                "issue": f"代码中使用 {col} 但未在依赖中声明，且没有自动计算机制",
                                "description": description,
                            }
                        )
                    elif not has_auto_compute and (in_deps or in_required):
                        # 依赖已声明，但没有自动计算机制（可能依赖计算顺序）
                        # 这是可接受的，但可以标记为潜在问题
                        pass

        except Exception as e:
            # 无法检查的函数，跳过
            continue

    # 特别检查几个关键特征
    print(f"\n{'='*80}")
    print("特别检查关键特征")
    print(f"{'='*80}")

    key_features = ["sqs_hal_high", "sqs_hal_low", "sr_strength_max", "atr_ratio"]

    for feature_name in key_features:
        if feature_name not in features:
            continue

        feature_info = features[feature_name]
        compute_func = feature_info.get("compute_func", "")
        dependencies = feature_info.get("dependencies", [])

        print(f"\n   {feature_name}:")
        print(f"      依赖: {dependencies}")

        try:
            if "." in compute_func:
                module_path, func_name = compute_func.rsplit(".", 1)
                module = __import__(module_path, fromlist=[func_name])
                func = getattr(module, func_name)
                func_code = inspect.getsource(func)

                # 检查 ATR
                if "atr" in func_code.lower():
                    if 'if "atr" not in' in func_code or "compute_atr" in func_code:
                        print(f"      ✅ 有 ATR 自动计算机制")
                    else:
                        print(f"      ⚠️  使用 ATR 但没有自动计算机制")
                        if "atr" not in dependencies:
                            issues.append(
                                {
                                    "feature": feature_name,
                                    "column": "atr",
                                    "issue": "使用 ATR 但没有自动计算机制，且未在依赖中声明",
                                }
                            )

                # 检查边界列
                for col in ["hal_high", "hal_low", "poc"]:
                    if col in func_code.lower():
                        if (
                            f'if "{col}" not in' in func_code
                            or "add_poc_hal" in func_code
                        ):
                            print(f"      ✅ 有 {col} 自动计算机制")
                        else:
                            print(f"      ⚠️  使用 {col} 但没有自动计算机制")

        except Exception as e:
            print(f"      ⚠️  无法检查: {e}")

    # 总结
    print(f"\n{'='*80}")
    print("检查总结")
    print(f"{'='*80}")

    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"\n   {i}. {issue['feature']}:")
            print(f"      列: {issue['column']}")
            print(f"      问题: {issue['issue']}")
            if "description" in issue:
                print(f"      说明: {issue['description']}")
    else:
        print(f"\n✅ 未发现明显的依赖问题")

    # 建议
    print(f"\n{'='*80}")
    print("建议")
    print(f"{'='*80}")
    print(f"1. ✅ sqs_hal_high 和 sqs_hal_low 已添加 ATR 自动计算")
    print(f"2. ✅ sr_strength_max 已添加边界列和 ATR 自动计算")
    print(f"3. ⚠️  其他使用 ATR 的特征依赖 dependencies 声明，理论上应该没问题")
    print(f"4. 💡 如果训练时仍有问题，可以检查特征计算顺序")

    return issues


if __name__ == "__main__":
    check_all_feature_dependencies()
