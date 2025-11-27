#!/usr/bin/env python3
"""
验证新的特征配置是否正确

检查：
1. 配置文件语法
2. 函数映射是否存在
3. 依赖关系是否完整
"""

import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.loader.feature_function_mapping import (
    FEATURE_FUNCTION_MAP,
    get_compute_func,
)


def validate_config():
    """验证配置文件"""
    print("=" * 80)
    print("验证新的特征配置")
    print("=" * 80)
    print()

    # 1. 加载配置文件
    print("📋 加载配置文件...")
    deps_path = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    with open(deps_path, "r", encoding="utf-8") as f:
        deps_config = yaml.safe_load(f)

    features = deps_config.get("features", {})
    print(f"   特征总数: {len(features)}")
    print()

    # 2. 检查新添加的衍生特征
    print("🔍 检查衍生特征配置...")
    derived_features = [
        "sr_strength_combined",
        "sr_distance_normalized",
        "dist_to_zz_high",
        "dist_to_zz_low",
        "dist_to_zz_high_atr",
        "dist_to_zz_low_atr",
        "cvd_slope_5",
        "atr_ratio",
        "bb_width_ratio",
        "compression_score",
        "tbr_ma_5",
        "tbr_spike",
    ]

    errors = []
    warnings = []

    for feat_name in derived_features:
        if feat_name not in features:
            errors.append(f"❌ {feat_name}: 配置文件中不存在")
            continue

        feat_config = features[feat_name]
        compute_func_name = feat_config.get("compute_func")

        if not compute_func_name:
            errors.append(f"❌ {feat_name}: 缺少 compute_func")
            continue

        # 检查函数映射
        try:
            func = get_compute_func(compute_func_name)
            print(f"   ✅ {feat_name}: {compute_func_name} -> {func.__name__}")
        except ValueError as e:
            errors.append(f"❌ {feat_name}: {compute_func_name} 函数映射不存在")
            print(f"   ❌ {feat_name}: {e}")

        # 检查依赖关系
        deps = feat_config.get("dependencies", [])
        required_cols = feat_config.get("required_columns", [])
        output_cols = feat_config.get("output_columns", [])

        if not output_cols:
            warnings.append(f"⚠️  {feat_name}: 缺少 output_columns")

    print()

    # 3. 检查交互特征
    print("🔍 检查交互特征配置...")
    interaction_features = [
        "vpin_x_wick_upper",
        "vpin_x_wick_lower",
        "vpin_x_wick_upper_rank",
        "vpin_x_wick_lower_rank",
    ]

    for feat_name in interaction_features:
        if feat_name not in features:
            errors.append(f"❌ {feat_name}: 配置文件中不存在")
            continue

        feat_config = features[feat_name]
        compute_func_name = feat_config.get("compute_func")

        if not compute_func_name:
            errors.append(f"❌ {feat_name}: 缺少 compute_func")
            continue

        try:
            func = get_compute_func(compute_func_name)
            print(f"   ✅ {feat_name}: {compute_func_name} -> {func.__name__}")
        except ValueError as e:
            errors.append(f"❌ {feat_name}: {compute_func_name} 函数映射不存在")
            print(f"   ❌ {feat_name}: {e}")

    print()

    # 4. 检查策略配置
    print("📋 检查策略配置...")
    strategy_path = (
        PROJECT_ROOT / "config" / "strategies" / "sr_reversal" / "features.yaml"
    )
    with open(strategy_path, "r", encoding="utf-8") as f:
        strategy_config = yaml.safe_load(f)

    requested = strategy_config.get("feature_pipeline", {}).get(
        "requested_features", []
    )
    print(f"   请求的特征数量: {len(requested)}")

    # 检查请求的特征是否都在配置文件中定义
    missing_in_deps = [f for f in requested if f not in features]
    if missing_in_deps:
        warnings.append(f"⚠️  策略请求的特征在依赖配置中不存在: {missing_in_deps[:5]}")
        print(f"   ⚠️  缺失的特征: {missing_in_deps[:5]}")
    else:
        print(f"   ✅ 所有请求的特征都在依赖配置中定义")

    print()

    # 5. 总结
    print("=" * 80)
    print("验证总结")
    print("=" * 80)

    if errors:
        print(f"❌ 发现 {len(errors)} 个错误:")
        for err in errors[:10]:
            print(f"   {err}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
        print()
        return False

    if warnings:
        print(f"⚠️  发现 {len(warnings)} 个警告:")
        for warn in warnings[:5]:
            print(f"   {warn}")
        print()

    print("✅ 配置验证通过！")
    print()
    return True


if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
