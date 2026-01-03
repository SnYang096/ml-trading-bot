#!/usr/bin/env python3
"""
迁移脚本：将特征计算函数名添加 _f 后缀

将 feature_dependencies.yaml 中所有特征计算函数名从：
  bb_width → bb_width_f
  roc_5 → roc_5_f

同时更新所有依赖关系和策略配置文件。
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def load_yaml(path: Path) -> Dict:
    """加载 YAML 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: Dict):
    """保存 YAML 文件"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )


def migrate_feature_dependencies(
    feature_deps_path: Path, backup: bool = True
) -> Dict[str, str]:
    """
    迁移 feature_dependencies.yaml

    Returns:
        mapping: 旧名称 -> 新名称的映射
    """
    print(f"📝 迁移 {feature_deps_path}...")

    # 备份
    if backup:
        backup_path = feature_deps_path.with_suffix(".yaml.backup")
        print(f"   💾 创建备份: {backup_path}")
        with open(feature_deps_path, "r", encoding="utf-8") as f:
            backup_path.write_text(f.read(), encoding="utf-8")

    # 加载
    deps = load_yaml(feature_deps_path)
    features = deps.get("features", {})

    # 创建映射：旧名称 -> 新名称
    name_mapping = {}
    for feat_name in list(features.keys()):
        if not feat_name.endswith("_f"):
            new_name = f"{feat_name}_f"
            name_mapping[feat_name] = new_name

    print(f"   📊 需要迁移 {len(name_mapping)} 个特征计算函数")

    # 1. 重命名特征计算函数（key）
    new_features = {}
    for old_name, feat_info in features.items():
        if old_name in name_mapping:
            new_name = name_mapping[old_name]
            new_features[new_name] = feat_info
        else:
            new_features[old_name] = feat_info

    # 2. 更新 dependencies
    for feat_name, feat_info in new_features.items():
        deps_list = feat_info.get("dependencies", [])
        updated_deps = []
        for dep in deps_list:
            if dep in name_mapping:
                updated_deps.append(name_mapping[dep])
            else:
                updated_deps.append(dep)
        feat_info["dependencies"] = updated_deps

    # 更新 deps
    deps["features"] = new_features

    # 保存
    save_yaml(feature_deps_path, deps)
    print(f"   ✅ 迁移完成")

    return name_mapping


def migrate_strategy_configs(strategy_configs_dir: Path, name_mapping: Dict[str, str]):
    """迁移策略配置文件"""
    print(f"\n📝 迁移策略配置文件...")

    # 查找所有 features*.yaml 文件
    config_files = list(strategy_configs_dir.rglob("features*.yaml"))
    print(f"   📊 找到 {len(config_files)} 个配置文件")

    migrated_count = 0
    for config_file in config_files:
        try:
            config = load_yaml(config_file)

            # 检查是否有 requested_features
            feature_pipeline = config.get("feature_pipeline", {})
            requested_features = feature_pipeline.get("requested_features", [])

            if not requested_features:
                continue

            # 更新 requested_features
            updated = False
            new_requested = []
            for item in requested_features:
                if item in name_mapping:
                    new_requested.append(name_mapping[item])
                    updated = True
                else:
                    new_requested.append(item)

            if updated:
                feature_pipeline["requested_features"] = new_requested
                config["feature_pipeline"] = feature_pipeline

                # 备份
                backup_path = config_file.with_suffix(".yaml.backup")
                with open(config_file, "r", encoding="utf-8") as f:
                    backup_path.write_text(f.read(), encoding="utf-8")

                # 保存
                save_yaml(config_file, config)
                migrated_count += 1
                print(f"   ✅ 迁移: {config_file}")

        except Exception as e:
            print(f"   ⚠️  跳过 {config_file}: {e}")

    print(f"   ✅ 迁移了 {migrated_count} 个配置文件")


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 开始迁移：特征计算函数名添加 _f 后缀")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    feature_deps_path = project_root / "config" / "feature_dependencies.yaml"
    strategy_configs_dir = project_root / "config" / "strategies"

    # 检查文件是否存在
    if not feature_deps_path.exists():
        print(f"❌ 文件不存在: {feature_deps_path}")
        return

    # 1. 迁移 feature_dependencies.yaml
    name_mapping = migrate_feature_dependencies(feature_deps_path, backup=True)

    if not name_mapping:
        print("\n✅ 没有需要迁移的特征，退出")
        return

    # 2. 迁移策略配置文件
    if strategy_configs_dir.exists():
        migrate_strategy_configs(strategy_configs_dir, name_mapping)

    # 3. 输出映射表（用于迁移记录/回溯）
    # NOTE: We no longer write this into config/ because runtime code does not consume it.
    # Keep it under results/ for audit/debug only.
    mapping_dir = project_root / "results" / "migrations"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = mapping_dir / "feature_name_mapping.yaml"
    mapping_data = {
        "old_to_new": name_mapping,
        "new_to_old": {v: k for k, v in name_mapping.items()},
        "note": "This mapping is for backward compatibility. Old feature names will be automatically mapped to new names with _f suffix.",
    }
    save_yaml(mapping_file, mapping_data)
    print(f"\n📝 保存名称映射: {mapping_file}")

    print("\n" + "=" * 70)
    print("✅ 迁移完成！")
    print("=" * 70)
    print(f"\n迁移统计：")
    print(f"  - 特征计算函数迁移数: {len(name_mapping)}")
    print(f"  - 映射文件: {mapping_file}")
    print(f"\n⚠️  注意：")
    print(f"  - 所有文件已备份（.yaml.backup）")
    print(f"  - 请检查迁移结果，确保无误")
    print(f"  - 代码中的兼容性处理需要手动添加")


if __name__ == "__main__":
    main()
