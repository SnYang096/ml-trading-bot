#!/usr/bin/env python3
"""
清理 sr_strength_max 的缓存文件

由于修复了 compute_sr_strength_max 函数的代码结构错误，
旧的缓存可能包含错误的结果，需要删除并重新计算。
"""

import sys
from pathlib import Path
import hashlib
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.features.loader.feature_computer import FeatureComputer


def clear_sr_strength_max_cache():
    """清理 sr_strength_max 的缓存"""
    print("=" * 80)
    print("清理 sr_strength_max 缓存")
    print("=" * 80)

    cache_dir = project_root / "cache" / "features" / "monthly"

    if not cache_dir.exists():
        print(f"   ⚠️  缓存目录不存在: {cache_dir}")
        return

    print(f"\n📂 缓存目录: {cache_dir}")

    # 方法1: 通过特征名和参数匹配（更精确）
    print(f"\n🔍 方法1: 通过特征名和参数匹配缓存文件...")

    # 加载特征配置以获取参数
    try:
        import yaml

        feature_deps_path = project_root / "config" / "feature_dependencies.yaml"
        with open(feature_deps_path, "r", encoding="utf-8") as f:
            feature_deps = yaml.safe_load(f)

        feature_info = feature_deps.get("features", {}).get("sr_strength_max", {})
        compute_params = feature_info.get("compute_params", {})
        output_columns = feature_info.get("output_columns", ["sr_strength_max"])

        print(f"   特征配置:")
        print(f"      compute_params: {compute_params}")
        print(f"      output_columns: {output_columns}")

        # 生成可能的缓存键（需要检查所有月份）
        months = [
            "2025-01",
            "2025-02",
            "2025-03",
            "2025-04",
            "2025-05",
            "2025-06",
            "2025-07",
        ]
        possible_keys = []

        for month_key in months:
            params_str = str(sorted(compute_params.items()))
            output_cols_str = str(sorted(output_columns))
            code_version = "v3"  # 当前版本
            key_str = f"sr_strength_max_monthly_{month_key}_{params_str}_{output_cols_str}_{code_version}"
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            possible_keys.append(cache_key)

        print(f"\n   可能的缓存键数量: {len(possible_keys)}")

        # 查找并删除匹配的缓存文件
        deleted_count = 0
        for cache_key in possible_keys:
            cache_file = cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                print(f"   🗑️  删除: {cache_file.name}")
                cache_file.unlink()
                deleted_count += 1

        print(f"\n   ✅ 删除了 {deleted_count} 个匹配的缓存文件")

    except Exception as e:
        print(f"   ⚠️  方法1失败: {e}")
        import traceback

        traceback.print_exc()

    # 方法2: 通过文件名模式匹配（备用方法）
    print(f"\n🔍 方法2: 通过文件名模式匹配...")

    # 由于缓存键是 MD5，我们无法直接通过文件名匹配
    # 但可以尝试加载每个缓存文件，检查内容
    # 这是一个更慢但更可靠的方法

    try:
        cache_files = list(cache_dir.glob("*.pkl"))
        print(f"   总缓存文件数: {len(cache_files)}")

        # 尝试加载并检查（只检查前几个，避免太慢）
        checked_count = 0
        matched_count = 0

        for cache_file in cache_files[:100]:  # 只检查前100个
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    # 检查是否是 DataFrame 且包含 sr_strength_max 列
                    if isinstance(data, (pd.DataFrame, pd.Series)):
                        if (
                            isinstance(data, pd.DataFrame)
                            and "sr_strength_max" in data.columns
                        ):
                            print(f"   🗑️  删除（通过内容匹配）: {cache_file.name}")
                            cache_file.unlink()
                            matched_count += 1
                        elif (
                            isinstance(data, pd.Series)
                            and data.name == "sr_strength_max"
                        ):
                            print(f"   🗑️  删除（通过内容匹配）: {cache_file.name}")
                            cache_file.unlink()
                            matched_count += 1
                checked_count += 1
            except Exception:
                # 无法加载的文件，跳过
                continue

        print(f"   检查了 {checked_count} 个文件，匹配 {matched_count} 个")

    except Exception as e:
        print(f"   ⚠️  方法2失败: {e}")
        import traceback

        traceback.print_exc()

    # 方法3: 简单粗暴 - 删除所有可能的缓存（如果文件不多）
    print(f"\n🔍 方法3: 列出所有缓存文件，手动确认...")

    try:
        import pandas as pd

        cache_files = list(cache_dir.glob("*.pkl"))
        print(f"   总缓存文件数: {len(cache_files)}")

        if len(cache_files) < 1000:  # 如果文件不多，可以全部检查
            print(f"   ⚠️  文件数量较多，建议使用方法1或直接删除整个 monthly 目录")
            print(f"   或者更新 code_version 让旧缓存自动失效")
        else:
            print(f"   ℹ️  文件数量: {len(cache_files)}，建议使用方法1")

    except Exception as e:
        print(f"   ⚠️  方法3失败: {e}")

    print(f"\n✅ 缓存清理完成")
    print(f"\n💡 建议:")
    print(f"   1. 重新运行训练，系统会使用修复后的代码重新计算")
    print(f"   2. 或者更新 code_version（在 parallel_computer.py 中）让所有旧缓存失效")


if __name__ == "__main__":
    clear_sr_strength_max_cache()
