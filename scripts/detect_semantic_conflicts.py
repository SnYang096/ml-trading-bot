#!/usr/bin/env python3
"""
检测语义特征内部可能的冲突。

通过以下方式检测冲突：
1. 检查是否有重复特征
2. 检查同源但不同语义的组（如 liquidity_void vs liquidity_void_scene）
3. 运行 feature-group-search 测试组合，看是否有性能下降

用法:
    python scripts/detect_semantic_conflicts.py \
        --strategy sr_reversal_rr_reg_long \
        --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
        --test-combinations
"""

import argparse
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_yaml(path: Path) -> dict:
    """加载 YAML 文件"""
    with open(path) as f:
        return yaml.safe_load(f)


def detect_duplicate_features(
    semantic_groups: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """检测重复特征"""
    feature_to_groups = defaultdict(list)
    for group_name, features in semantic_groups.items():
        for feat in features:
            feature_to_groups[feat].append(group_name)

    duplicates = {
        f: groups for f, groups in feature_to_groups.items() if len(groups) > 1
    }
    return duplicates


def detect_semantic_conflicts(semantic_groups: Dict[str, List[str]]) -> List[Dict]:
    """检测语义上可能冲突的组"""
    conflict_pairs = []
    group_names = list(semantic_groups.keys())

    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1 :]:
            # 检查是否有同源但不同语义的组
            base1 = g1.replace("_scene", "").replace("_semantic", "")
            base2 = g2.replace("_scene", "").replace("_semantic", "")

            # 同源但不同语义
            if base1 == base2 and g1 != g2:
                conflict_pairs.append(
                    {
                        "group1": g1,
                        "group2": g2,
                        "base": base1,
                        "reason": f"同源 {base1}，但一个是原始特征，一个是语义化特征",
                        "features1": semantic_groups[g1],
                        "features2": semantic_groups[g2],
                        "suggestion": "建议测试同时加入是否会冲突，通常语义化版本应该替代原始版本",
                    }
                )

            # 检查是否有语义上可能冲突的组（如 vpin_scene 和 trade_cluster_scene 在反转策略中）
            if "scene" in g1 and "scene" in g2:
                # 这些通常不会冲突，因为它们已经是语义化的
                pass

    return conflict_pairs


def generate_test_combinations(conflict_pairs: List[Dict]) -> List[Tuple[str, str]]:
    """生成需要测试的组合"""
    combinations = []
    for pair in conflict_pairs:
        combinations.append((pair["group1"], pair["group2"]))
    return combinations


def main():
    parser = argparse.ArgumentParser(description="检测语义特征内部冲突")
    parser.add_argument("--strategy", required=True, help="策略名称")
    parser.add_argument(
        "--semantic-groups", required=True, type=Path, help="语义 groups YAML 路径"
    )
    parser.add_argument(
        "--test-combinations", action="store_true", help="生成测试组合命令"
    )
    args = parser.parse_args()

    # 加载数据
    semantic_data = load_yaml(args.semantic_groups)
    semantic_groups = semantic_data.get("groups", {})

    # 检测冲突
    duplicates = detect_duplicate_features(semantic_groups)
    conflicts = detect_semantic_conflicts(semantic_groups)

    # 输出结果
    print("=" * 80)
    print("🔍 语义特征内部冲突检测")
    print("=" * 80)

    print(f"\n1. 重复特征检测:")
    if duplicates:
        print(f"   ⚠️  发现 {len(duplicates)} 个重复特征:")
        for feat, groups in duplicates.items():
            print(f"      {feat}: 出现在 {groups}")
    else:
        print(f"   ✅ 未发现重复特征")

    print(f"\n2. 语义冲突检测:")
    if conflicts:
        print(f"   ⚠️  发现 {len(conflicts)} 对可能的语义冲突:")
        for conflict in conflicts:
            print(f"\n      {conflict['group1']} vs {conflict['group2']}")
            print(f"      原因: {conflict['reason']}")
            print(f"      特征1: {conflict['features1']}")
            print(f"      特征2: {conflict['features2']}")
            print(f"      建议: {conflict['suggestion']}")
    else:
        print(f"   ✅ 未发现明显的语义冲突")

    # 生成测试命令
    if args.test_combinations and conflicts:
        print(f"\n3. 测试组合命令:")
        print(f"   建议运行以下命令测试冲突组合:")
        for conflict in conflicts:
            g1, g2 = conflict["group1"], conflict["group2"]
            print(f"\n   # 测试 {g1} + {g2} 是否冲突")
            print(f"   mlbot diagnose feature-group-search \\")
            print(f"     -c config/strategies/{args.strategy} \\")
            print(
                f"     --groups-yaml <(echo 'groups: {{ {g1}: {semantic_groups[g1]}, {g2}: {semantic_groups[g2]} }}') \\"
            )
            print(f"     --max-steps 2 \\")
            print(f"     --seeds 1,2,3 \\")
            print(f"     --output-dir results/conflict_test/{g1}_{g2}")


if __name__ == "__main__":
    main()
