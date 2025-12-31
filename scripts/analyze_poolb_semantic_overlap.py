#!/usr/bin/env python3
"""
分析四个策略的 Pool B 与语义特征的交叉情况。

用法:
    python scripts/analyze_poolb_semantic_overlap.py
"""

from pathlib import Path
import yaml
from collections import defaultdict

# 语义特征的实际输出列（从 feature_dependencies.yaml 中提取）
SEMANTIC_OUTPUT_COLS = {
    "liquidity_void_f": [
        "liquidity_void_detected",
        "liquidity_void_speed",
        "liquidity_void_volume_ratio",
        "liquidity_void_price_impact",
        "liquidity_void_retracement",
        "liquidity_void_false_breakout_risk",
    ],
    "compression_score_f": ["compression_score"],
    "compression_energy_f": ["compression_energy"],
    "liquidity_void_scene_semantic_scores_f": [
        "liquidity_void_compression_score",
        "liquidity_void_ignition_score",
        "liquidity_void_absorption_score",
        "liquidity_void_exhaustion_score",
    ],
    "vpin_scene_semantic_scores_f": [
        "vpin_compression_score",
        "vpin_ignition_score",
        "vpin_absorption_score",
        "vpin_exhaustion_scene_score",
    ],
    "trade_cluster_scene_semantic_scores_f": [
        "trade_cluster_compression_score",
        "trade_cluster_ignition_score",
        "trade_cluster_absorption_scene_score",
        "trade_cluster_exhaustion_scene_score",
    ],
    "wpt_scene_semantic_scores_f": [
        "wpt_compression_score",
        "wpt_ignition_score",
        "wpt_absorption_score",
        "wpt_exhaustion_score",
    ],
    "volume_profile_scene_semantic_scores_f": [
        "vp_compression_score",
        "vp_ignition_score",
        "vp_absorption_score",
        "vp_exhaustion_score",
    ],
    "wick_scene_semantic_scores_f": [
        "wick_compression_score",
        "wick_ignition_score",
        "wick_absorption_score",
        "wick_exhaustion_score",
    ],
    "fp_imbalance_scene_semantic_scores_f": [
        "fp_imbalance_compression_score",
        "fp_imbalance_ignition_score",
        "fp_imbalance_absorption_score",
        "fp_imbalance_exhaustion_scene_score",
    ],
    "market_cap_normalized_orderflow_f": [
        "market_cap_usd",
        "dollar_volume_over_mcap",
        "turnover_over_mcap",
        "net_buy_usd_over_mcap",
        "abs_net_buy_usd_over_mcap",
    ],
    "funding_scene_semantic_scores_f": [
        "funding_compression_score",
        "funding_ignition_score",
        "funding_absorption_score",
        "funding_exhaustion_scene_score",
    ],
}

STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "sr_breakout",
    "compression_breakout",
    "trend_following",
]

SEMANTIC_GROUPS_FILES = {
    "sr_reversal_rr_reg_long": Path("config/feature_groups_sr_reversal_semantic.yaml"),
    "sr_breakout": Path("config/feature_groups_sr_breakout_semantic.yaml"),
    "compression_breakout": Path(
        "config/feature_groups_compression_breakout_semantic.yaml"
    ),
    "trend_following": Path("config/feature_groups_trend_following_semantic.yaml"),
}


def load_semantic_groups(strategy: str) -> dict:
    """加载语义 groups"""
    file_path = SEMANTIC_GROUPS_FILES.get(strategy)
    if not file_path or not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("groups", {})


def get_semantic_output_cols(strategy: str, semantic_groups: dict) -> list:
    """获取语义特征的实际输出列"""
    semantic_nodes = [f for group in semantic_groups.values() for f in group]
    output_cols = []
    for node in semantic_nodes:
        if node in SEMANTIC_OUTPUT_COLS:
            output_cols.extend(SEMANTIC_OUTPUT_COLS[node])
    return output_cols


def load_pool_b(strategy: str) -> list:
    """加载 Pool B 特征"""
    pool_b_yaml = Path(f"results/pools/{strategy}/pool_b/features_pool_b.yaml")
    if not pool_b_yaml.exists():
        return []

    with open(pool_b_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("feature_pipeline", {}).get("requested_features", [])


def analyze_overlap(strategy: str):
    """分析单个策略的交叉情况"""
    print(f"\n{'='*80}")
    print(f"策略: {strategy}")
    print(f"{'='*80}")

    # 加载数据
    semantic_groups = load_semantic_groups(strategy)
    pool_b_features = load_pool_b(strategy)

    semantic_nodes = [f for group in semantic_groups.values() for f in group]
    semantic_output_cols = get_semantic_output_cols(strategy, semantic_groups)

    # 统计
    pool_b_exists = len(pool_b_features) > 0
    semantic_exists = len(semantic_nodes) > 0

    print(f"\n📊 基本信息:")
    print(f"  Pool B 是否存在: {'✅' if pool_b_exists else '❌'}")
    if pool_b_exists:
        print(f"  Pool B 特征数: {len(pool_b_features)}")
    print(f"  语义 groups 是否存在: {'✅' if semantic_exists else '❌'}")
    if semantic_exists:
        print(f"  语义 groups 节点数: {len(semantic_nodes)}")
        print(f"  语义特征输出列数: {len(semantic_output_cols)}")

    if not pool_b_exists or not semantic_exists:
        return

    # 节点级别交叉
    pool_b_set = set(pool_b_features)
    semantic_set = set(semantic_nodes)

    node_overlap = pool_b_set & semantic_set
    pool_b_only_nodes = pool_b_set - semantic_set
    semantic_only_nodes = semantic_set - pool_b_set

    print(f"\n📋 节点级别交叉:")
    print(f"  交叉特征数（节点）: {len(node_overlap)}")
    if node_overlap:
        print(f"  交叉特征: {list(node_overlap)}")
    print(f"  Pool B 独有特征数: {len(pool_b_only_nodes)}")
    if pool_b_only_nodes and len(pool_b_only_nodes) <= 10:
        print(f"  Pool B 独有特征: {list(pool_b_only_nodes)}")
    print(f"  语义 groups 独有特征数: {len(semantic_only_nodes)}")
    if semantic_only_nodes and len(semantic_only_nodes) <= 10:
        print(f"  语义 groups 独有特征: {list(semantic_only_nodes)}")

    # 输出列级别交叉
    pool_b_cols_set = pool_b_set  # Pool B 中的特征可能是节点名或列名
    semantic_cols_set = set(semantic_output_cols)

    col_overlap = pool_b_cols_set & semantic_cols_set
    pool_b_only_cols = pool_b_cols_set - semantic_cols_set

    print(f"\n📋 输出列级别交叉:")
    print(f"  Pool B 中包含语义特征输出列数: {len(col_overlap)}")
    if col_overlap:
        print(f"  包含的列（前10）: {list(col_overlap)[:10]}")
    print(f"  Pool B 中非语义特征列数: {len(pool_b_only_cols)}")

    # 覆盖率
    if semantic_output_cols:
        coverage = len(col_overlap) / len(semantic_output_cols) * 100
        print(f"\n📈 覆盖率:")
        print(
            f"  Pool B 覆盖语义特征输出列: {coverage:.1f}% ({len(col_overlap)}/{len(semantic_output_cols)})"
        )


def main():
    print("=" * 80)
    print("📊 四个策略的 Pool B 与语义特征交叉情况分析")
    print("=" * 80)

    # 分析每个策略
    for strategy in STRATEGIES:
        analyze_overlap(strategy)

    # 汇总
    print(f"\n{'='*80}")
    print("📈 汇总统计")
    print(f"{'='*80}")

    pool_b_count = sum(1 for s in STRATEGIES if load_pool_b(s))
    semantic_count = sum(1 for s in STRATEGIES if load_semantic_groups(s))

    print(f"\n已生成 Pool B 的策略: {pool_b_count}/{len(STRATEGIES)}")
    print(f"有语义 groups 的策略: {semantic_count}/{len(STRATEGIES)}")

    print(f"\n策略列表:")
    for strategy in STRATEGIES:
        pool_b_exists = len(load_pool_b(strategy)) > 0
        semantic_exists = len(load_semantic_groups(strategy)) > 0
        status = []
        if pool_b_exists:
            status.append("Pool B ✅")
        if semantic_exists:
            status.append("语义 groups ✅")
        print(f"  {strategy}: {', '.join(status) if status else '❌ 无'}")


if __name__ == "__main__":
    main()
