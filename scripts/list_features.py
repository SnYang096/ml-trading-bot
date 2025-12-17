#!/usr/bin/env python3
"""
列出所有已注册的特征函数

用法:
    python scripts/list_features.py              # 显示摘要统计
    python scripts/list_features.py --category   # 按分类显示
    python scripts/list_features.py --all        # 显示所有特征详情
    python scripts/list_features.py --search <name>  # 搜索特征
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保能导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import inspect
from typing import Dict, List


def import_all_feature_modules():
    """导入所有特征模块以触发装饰器注册"""
    # 小模块
    import src.features.loader.selector_utils
    import src.features.time_series.utils_wpt_features
    import src.features.time_series.utils_hilbert_features
    import src.features.time_series.utils_hurst_features
    import src.features.time_series.utils_spectrum_features
    import src.features.time_series.utils_garch_features
    import src.features.time_series.utils_evt_features
    import src.features.time_series.utils_dtw_features
    import src.features.time_series.utils_volume_profile
    import src.features.time_series.utils_liquidity_features
    import src.features.loader.feature_wrappers
    import src.features.time_series.dl_sequence_features

    # 中等模块
    import src.features.loader.talib_feature_wrappers
    import src.features.time_series.utils_volatility_features

    # 大模块
    import src.features.time_series.utils_interaction_features
    import src.features.time_series.baseline_features
    import src.features.time_series.utils_order_flow_features


def get_function_docstring(func) -> str:
    """获取函数的文档字符串（首行）"""
    doc = inspect.getdoc(func)
    if doc:
        # 只取第一行
        first_line = doc.split("\n")[0].strip()
        # 限制长度
        if len(first_line) > 60:
            return first_line[:57] + "..."
        return first_line
    return "(无描述)"


def get_function_module(func) -> str:
    """获取函数所在的模块名（简化）"""
    module = getattr(func, "__module__", "unknown")
    # 简化模块路径
    if module.startswith("src.features."):
        module = module.replace("src.features.", "")
    return module


def main():
    parser = argparse.ArgumentParser(description="列出已注册的特征函数")
    parser.add_argument("--all", "-a", action="store_true", help="显示所有特征详情")
    parser.add_argument("--category", "-c", action="store_true", help="按分类显示")
    parser.add_argument("--search", "-s", type=str, help="搜索特征名")
    parser.add_argument("--module", "-m", type=str, help="按模块过滤")
    args = parser.parse_args()

    # 导入模块触发注册
    import_all_feature_modules()

    from src.features.registry import get_registry

    registry = get_registry()

    # 收集信息
    features_by_category: Dict[str, List[tuple]] = {}
    features_by_module: Dict[str, List[str]] = {}

    for name in registry.list_features():
        meta = registry.get_metadata(name)
        func = registry.get(name)
        category = meta.get("category", "unknown")
        module = get_function_module(func)
        docstring = get_function_docstring(func)

        if category not in features_by_category:
            features_by_category[category] = []
        features_by_category[category].append((name, module, docstring))

        if module not in features_by_module:
            features_by_module[module] = []
        features_by_module[module].append(name)

    # 搜索模式
    if args.search:
        print(f"\n🔍 搜索: '{args.search}'")
        print("=" * 80)
        found = 0
        for name in registry.list_features():
            if args.search.lower() in name.lower():
                meta = registry.get_metadata(name)
                func = registry.get(name)
                category = meta.get("category", "unknown")
                module = get_function_module(func)
                docstring = get_function_docstring(func)
                print(f"\n  📦 {name}")
                print(f"     分类: {category}")
                print(f"     模块: {module}")
                print(f"     说明: {docstring}")
                found += 1
        print(f"\n找到 {found} 个匹配的特征\n")
        return

    # 模块过滤模式
    if args.module:
        print(f"\n📁 模块: '{args.module}'")
        print("=" * 80)
        for module, funcs in sorted(features_by_module.items()):
            if args.module.lower() in module.lower():
                print(f"\n  {module} ({len(funcs)} 个):")
                for name in sorted(funcs):
                    func = registry.get(name)
                    docstring = get_function_docstring(func)
                    print(f"    • {name}")
                    if args.all:
                        print(f"      {docstring}")
        return

    # 按分类显示
    if args.category or args.all:
        print("\n📊 特征分类详情")
        print("=" * 80)
        for category in sorted(features_by_category.keys()):
            features = features_by_category[category]
            print(f"\n🏷️  {category} ({len(features)} 个)")
            print("-" * 60)
            for name, module, docstring in sorted(features):
                if args.all:
                    print(f"  • {name}")
                    print(f"    模块: {module}")
                    print(f"    说明: {docstring}")
                else:
                    print(f"  • {name}")
        print()
        return

    # 默认：摘要统计
    print("\n" + "=" * 60)
    print("📈 特征注册表统计")
    print("=" * 60)

    print(f"\n✅ 总特征数: {registry.count}")

    print(f"\n📂 分类数: {len(features_by_category)}")
    print("-" * 40)
    for category in sorted(features_by_category.keys()):
        count = len(features_by_category[category])
        bar = "█" * (count // 5) + "░" * max(0, 14 - count // 5)
        print(f"  {category:20s} {bar} {count:3d}")

    print(f"\n📁 模块数: {len(features_by_module)}")
    print("-" * 40)
    for module in sorted(
        features_by_module.keys(), key=lambda m: -len(features_by_module[m])
    ):
        count = len(features_by_module[module])
        short_module = module.split(".")[-1] if "." in module else module
        print(f"  {short_module:35s} {count:3d}")

    print("\n" + "=" * 60)
    print("💡 使用提示:")
    print("  --all       显示所有特征详情")
    print("  --category  按分类显示")
    print("  --search X  搜索特征名")
    print("  --module X  按模块过滤")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
