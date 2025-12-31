#!/usr/bin/env python3
"""
将语义节点添加到 features_all.yaml 中。

用法:
    python scripts/add_semantic_to_features_all.py \
        --strategy sr_reversal_rr_reg_long \
        --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
        --features-all config/strategies/sr_reversal_rr_reg_long/features_all.yaml
"""

import argparse
import yaml
from pathlib import Path
from typing import List


def load_yaml(path: Path) -> dict:
    """加载 YAML 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict):
    """保存 YAML 文件"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )


def extract_semantic_nodes(semantic_groups: dict) -> List[str]:
    """从语义 groups 中提取所有节点"""
    nodes = []
    for group in semantic_groups.get("groups", {}).values():
        nodes.extend(group)
    return nodes


def add_semantic_to_features_all(
    features_all_path: Path, semantic_nodes: List[str], dry_run: bool = False
) -> dict:
    """将语义节点添加到 features_all.yaml"""
    # 加载 features_all.yaml
    data = load_yaml(features_all_path)
    all_features = data.get("feature_pipeline", {}).get("requested_features", [])

    # 检查哪些语义节点已经存在
    existing_set = set(all_features)
    existing_semantic = [n for n in semantic_nodes if n in existing_set]
    missing_semantic = [n for n in semantic_nodes if n not in existing_set]

    # 添加缺失的语义节点
    if missing_semantic:
        all_features.extend(missing_semantic)
        data["feature_pipeline"]["requested_features"] = all_features

        if not dry_run:
            save_yaml(features_all_path, data)

    return {
        "total_features_before": len(all_features) - len(missing_semantic),
        "total_features_after": len(all_features),
        "semantic_nodes_count": len(semantic_nodes),
        "existing_semantic": existing_semantic,
        "added_semantic": missing_semantic,
        "added_count": len(missing_semantic),
    }


def main():
    parser = argparse.ArgumentParser(description="将语义节点添加到 features_all.yaml")
    parser.add_argument("--strategy", required=True, help="策略名称")
    parser.add_argument(
        "--semantic-groups", required=True, type=Path, help="语义 groups YAML 路径"
    )
    parser.add_argument(
        "--features-all", required=True, type=Path, help="features_all.yaml 路径"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只显示将要添加的节点，不实际修改文件"
    )
    args = parser.parse_args()

    # 加载语义 groups
    if not args.semantic_groups.exists():
        print(f"❌ 语义 groups 文件不存在: {args.semantic_groups}")
        return

    semantic_data = load_yaml(args.semantic_groups)
    semantic_nodes = extract_semantic_nodes(semantic_data)

    print("=" * 80)
    print("📊 将语义节点添加到 features_all.yaml")
    print("=" * 80)
    print(f"\n策略: {args.strategy}")
    print(f"语义 groups: {args.semantic_groups}")
    print(f"features_all.yaml: {args.features_all}")
    print(f"语义节点数: {len(semantic_nodes)}")
    print(f"语义节点列表: {semantic_nodes}")

    # 检查 features_all.yaml
    if not args.features_all.exists():
        print(f"\n❌ features_all.yaml 不存在: {args.features_all}")
        return

    # 添加语义节点
    result = add_semantic_to_features_all(
        args.features_all, semantic_nodes, dry_run=args.dry_run
    )

    print(f"\n📈 结果:")
    print(f"  添加前特征数: {result['total_features_before']}")
    print(f"  添加后特征数: {result['total_features_after']}")
    print(f"  语义节点数: {result['semantic_nodes_count']}")
    print(f"  已存在的语义节点: {len(result['existing_semantic'])}")
    if result["existing_semantic"]:
        print(f"    {result['existing_semantic']}")
    print(f"  新增的语义节点: {result['added_count']}")
    if result["added_semantic"]:
        print(f"    {result['added_semantic']}")

    if args.dry_run:
        print(f"\n⚠️  这是 dry-run，未实际修改文件")
    else:
        print(f"\n✅ 已更新 {args.features_all}")
        print(f"\n💡 下一步:")
        print(f"   重新运行 factor-eval 生成包含语义特征的 Pool B")


if __name__ == "__main__":
    main()
