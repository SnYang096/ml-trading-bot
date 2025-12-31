#!/usr/bin/env python3
"""
创建 features_semantic.yaml（只包含语义特征）。

这是一个可选工具，用于验证语义特征的 IC/IR。

用法:
    python scripts/create_semantic_features_yaml.py \
        --strategy sr_reversal_rr_reg_long \
        --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
        --output config/strategies/sr_reversal_rr_reg_long/features_semantic.yaml
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def create_semantic_features_yaml(
    strategy: str, semantic_groups_path: Path, output_path: Path
) -> dict:
    """创建 features_semantic.yaml"""
    # 加载语义 groups
    semantic_data = load_yaml(semantic_groups_path)
    semantic_nodes = extract_semantic_nodes(semantic_data)

    # 创建 features_semantic.yaml 结构
    features_semantic = {
        "name": f"{strategy}_semantic",
        "description": f"Semantic features for {strategy} strategy. Contains only semantic scene features from feature_groups_<strategy>_semantic.yaml.",
        "feature_pipeline": {
            "requested_features": sorted(semantic_nodes),
            "ensure_signal_column": {"name": "signal", "default_value": 0},
        },
        "notes": f"""
This file contains only semantic features extracted from {semantic_groups_path.name}.

Usage:
  1. Use this file to run factor-eval and verify semantic features' IC/IR
  2. This is optional - the main workflow uses semantic groups directly in feature-group-search

Total semantic features: {len(semantic_nodes)}
""",
    }

    # 保存
    save_yaml(output_path, features_semantic)

    return {
        "strategy": strategy,
        "semantic_nodes_count": len(semantic_nodes),
        "semantic_nodes": semantic_nodes,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="创建 features_semantic.yaml")
    parser.add_argument("--strategy", required=True, help="策略名称")
    parser.add_argument(
        "--semantic-groups", required=True, type=Path, help="语义 groups YAML 路径"
    )
    parser.add_argument("--output", required=True, type=Path, help="输出文件路径")
    args = parser.parse_args()

    # 检查语义 groups 文件
    if not args.semantic_groups.exists():
        print(f"❌ 语义 groups 文件不存在: {args.semantic_groups}")
        return

    # 创建 features_semantic.yaml
    result = create_semantic_features_yaml(
        args.strategy, args.semantic_groups, args.output
    )

    print("=" * 80)
    print("✅ 已创建 features_semantic.yaml")
    print("=" * 80)
    print(f"\n策略: {result['strategy']}")
    print(f"语义节点数: {result['semantic_nodes_count']}")
    print(f"输出路径: {result['output_path']}")
    print(f"\n语义节点列表:")
    for i, node in enumerate(result["semantic_nodes"], 1):
        print(f"  {i:2d}. {node}")

    print(f"\n💡 下一步:")
    print(f"   可选：运行 factor-eval 验证语义特征的 IC/IR")
    print(f"   mlbot analyze factor-eval \\")
    print(f"     -c {result['output_path']} \\")
    print(f"     -s BTCUSDT -t 240T \\")
    print(f"     --start-date 2023-01-01 --end-date 2025-10-31 \\")
    print(f"     --output-dir results/pools/{result['strategy']}/pool_b_semantic \\")
    print(
        f"     --export-yaml results/pools/{result['strategy']}/pool_b_semantic/features_pool_b_semantic.yaml \\"
    )
    print(f"     --no-docker")
    print(f"\n   注意：这是可选的验证步骤，主要工作流仍然使用语义 groups")


if __name__ == "__main__":
    main()
