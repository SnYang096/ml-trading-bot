#!/usr/bin/env python3
"""
分析语义特征 vs 全量特征的覆盖情况，并检测可能的冲突。

用法:
    python scripts/analyze_semantic_vs_all_features.py \
        --strategy sr_reversal_rr_reg_long \
        --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
        --all-features config/strategies/sr_reversal_rr_reg_long/features_all.yaml \
        --output-dir results/feature_analysis
"""

import argparse
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


def load_yaml(path: Path) -> dict:
    """加载 YAML 文件"""
    with open(path) as f:
        return yaml.safe_load(f)


def classify_features(features: List[str]) -> Dict[str, List[str]]:
    """按特征类型分类特征"""
    classified = defaultdict(list)
    for feat in features:
        if "scene" in feat.lower() or "semantic" in feat.lower():
            classified["语义化特征"].append(feat)
        elif "dtw" in feat.lower():
            classified["DTW 模式匹配"].append(feat)
        elif "vpin" in feat.lower():
            classified["VPIN 相关"].append(feat)
        elif "trade_cluster" in feat.lower():
            classified["Trade Cluster"].append(feat)
        elif "wpt" in feat.lower():
            classified["WPT 小波"].append(feat)
        elif "footprint" in feat.lower() or "fp_" in feat.lower():
            classified["Footprint"].append(feat)
        elif "hilbert" in feat.lower():
            classified["Hilbert"].append(feat)
        elif "spectrum" in feat.lower():
            classified["Spectrum"].append(feat)
        elif "liquidity_void" in feat.lower():
            classified["Liquidity Void"].append(feat)
        elif "volume_profile" in feat.lower() or "vp_" in feat.lower():
            classified["Volume Profile"].append(feat)
        elif "compression" in feat.lower():
            classified["Compression"].append(feat)
        elif "wick" in feat.lower():
            classified["Wick"].append(feat)
        elif "funding" in feat.lower():
            classified["Funding"].append(feat)
        elif "market_cap" in feat.lower():
            classified["Market Cap"].append(feat)
        elif any(
            x in feat.lower() for x in ["macd", "rsi", "sma", "atr", "bb_", "trend_r2"]
        ):
            classified["K线技术指标"].append(feat)
        elif any(
            x in feat.lower()
            for x in ["poc", "hal", "sqs", "sr_strength", "sr_distance"]
        ):
            classified["SR 结构"].append(feat)
        else:
            classified["其他"].append(feat)
    return dict(classified)


def analyze_coverage(
    semantic_groups: Dict[str, List[str]], all_features: List[str]
) -> Dict:
    """分析语义特征覆盖情况"""
    # 收集语义特征
    semantic_features = set()
    feature_to_group = {}
    for group_name, features in semantic_groups.items():
        for feat in features:
            semantic_features.add(feat)
            feature_to_group[feat] = group_name

    # 分类全量特征
    all_features_by_type = classify_features(all_features)

    # 统计覆盖情况
    coverage_stats = {}
    for feat_type, feats in all_features_by_type.items():
        covered = [f for f in feats if f in semantic_features]
        not_covered = [f for f in feats if f not in semantic_features]
        coverage_stats[feat_type] = {
            "total": len(feats),
            "covered": len(covered),
            "not_covered": len(not_covered),
            "coverage_pct": len(covered) / len(feats) * 100 if feats else 0,
            "covered_features": covered,
            "not_covered_features": not_covered[:20],  # 只保留前20个
        }

    return {
        "total_all_features": len(all_features),
        "total_semantic_features": len(semantic_features),
        "overall_coverage_pct": (
            len(semantic_features) / len(all_features) * 100 if all_features else 0
        ),
        "not_covered_count": len(set(all_features) - semantic_features),
        "coverage_by_type": coverage_stats,
        "semantic_groups": semantic_groups,
        "feature_to_group": feature_to_group,
    }


def detect_conflicts(semantic_groups: Dict[str, List[str]]) -> Dict:
    """检测语义特征内部可能的冲突"""
    # 检查重复特征
    feature_to_groups = defaultdict(list)
    for group_name, features in semantic_groups.items():
        for feat in features:
            feature_to_groups[feat].append(group_name)

    duplicates = {
        f: groups for f, groups in feature_to_groups.items() if len(groups) > 1
    }

    # 检查语义上可能冲突的组
    conflict_pairs = []
    group_names = list(semantic_groups.keys())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1 :]:
            # 检查是否有同源但不同语义的组
            if "vpin" in g1 and "vpin" in g2 and g1 != g2:
                conflict_pairs.append(
                    {
                        "group1": g1,
                        "group2": g2,
                        "reason": "同源 VPIN，可能冲突",
                        "features1": semantic_groups[g1],
                        "features2": semantic_groups[g2],
                    }
                )
            elif "trade_cluster" in g1 and "trade_cluster" in g2 and g1 != g2:
                conflict_pairs.append(
                    {
                        "group1": g1,
                        "group2": g2,
                        "reason": "同源 Trade Cluster，可能冲突",
                        "features1": semantic_groups[g1],
                        "features2": semantic_groups[g2],
                    }
                )
            elif "liquidity_void" in g1 and "liquidity_void" in g2 and g1 != g2:
                conflict_pairs.append(
                    {
                        "group1": g1,
                        "group2": g2,
                        "reason": "同源 Liquidity Void，可能冲突",
                        "features1": semantic_groups[g1],
                        "features2": semantic_groups[g2],
                    }
                )

    return {
        "duplicates": dict(duplicates),
        "conflict_pairs": conflict_pairs,
        "has_conflicts": len(duplicates) > 0 or len(conflict_pairs) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="分析语义特征 vs 全量特征")
    parser.add_argument("--strategy", required=True, help="策略名称")
    parser.add_argument(
        "--semantic-groups", required=True, type=Path, help="语义 groups YAML 路径"
    )
    parser.add_argument(
        "--all-features", required=True, type=Path, help="全量特征 YAML 路径"
    )
    parser.add_argument("--output-dir", type=Path, help="输出目录")
    args = parser.parse_args()

    # 加载数据
    semantic_data = load_yaml(args.semantic_groups)
    semantic_groups = semantic_data.get("groups", {})

    all_features_data = load_yaml(args.all_features)
    all_features = all_features_data.get("feature_pipeline", {}).get(
        "requested_features", []
    )

    # 分析
    coverage = analyze_coverage(semantic_groups, all_features)
    conflicts = detect_conflicts(semantic_groups)

    # 输出结果
    result = {
        "strategy": args.strategy,
        "coverage_analysis": coverage,
        "conflict_analysis": conflicts,
    }

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = args.output_dir / f"{args.strategy}_feature_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存到: {output_file}")

    # 打印摘要
    print("=" * 80)
    print("📊 语义特征 vs 全量特征分析结果")
    print("=" * 80)
    # 计算实际输出列数
    semantic_output_cols = {
        "liquidity_void_f": 6,
        "compression_score_f": 1,
        "compression_energy_f": 1,
        "liquidity_void_scene_semantic_scores_f": 4,
        "vpin_scene_semantic_scores_f": 4,
        "trade_cluster_scene_semantic_scores_f": 4,
        "wpt_scene_semantic_scores_f": 4,
        "volume_profile_scene_semantic_scores_f": 4,
        "wick_scene_semantic_scores_f": 4,
        "fp_imbalance_scene_semantic_scores_f": 4,
        "market_cap_normalized_orderflow_f": 5,
        "funding_scene_semantic_scores_f": 4,
    }

    semantic_nodes = [f for group in semantic_groups.values() for f in group]
    total_output_cols = sum(semantic_output_cols.get(f, 0) for f in semantic_nodes)

    print(f"\n总体覆盖:")
    print(f"  全量特征数: {coverage['total_all_features']}")
    print(f"  语义特征节点数: {coverage['total_semantic_features']}")
    print(f"  语义特征实际输出列数: {total_output_cols}")
    print(f"  覆盖率（按节点）: {coverage['overall_coverage_pct']:.1f}%")
    print(f"  未覆盖数: {coverage['not_covered_count']}")

    print(f"\n按类型覆盖情况:")
    for feat_type, stats in sorted(
        coverage["coverage_by_type"].items(), key=lambda x: x[1]["total"], reverse=True
    ):
        print(
            f"  {feat_type}: {stats['covered']}/{stats['total']} ({stats['coverage_pct']:.1f}%)"
        )

    print(f"\n冲突检测:")
    if conflicts["has_conflicts"]:
        if conflicts["duplicates"]:
            print(f"  ⚠️  发现重复特征: {len(conflicts['duplicates'])} 个")
        if conflicts["conflict_pairs"]:
            print(f"  ⚠️  发现可能的语义冲突: {len(conflicts['conflict_pairs'])} 对")
            for pair in conflicts["conflict_pairs"]:
                print(f"    {pair['group1']} vs {pair['group2']}: {pair['reason']}")
    else:
        print(f"  ✅ 未发现明显冲突")


if __name__ == "__main__":
    main()
