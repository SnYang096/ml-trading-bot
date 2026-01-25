#!/usr/bin/env python3
"""
分析MEAN_REGIME分类条件的放宽策略

测试多种参数组合，找出最优平衡点（样本数 vs 质量）
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product


def calculate_sharpe(returns: pd.Series) -> float:
    """计算Sharpe比率（简化版，假设无风险利率为0）"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0


def test_relaxation_strategy(
    df: pd.DataFrame,
    conditions: Dict[str, np.ndarray],
    strategy_name: str,
) -> Dict:
    """测试一个放宽策略"""
    # 所有条件必须满足
    all_met = np.ones(len(df), dtype=bool)
    for cond_mask in conditions.values():
        all_met = all_met & cond_mask

    count = all_met.sum()
    if count == 0:
        return {
            "strategy": strategy_name,
            "count": 0,
            "mean_ret": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "median_ret": 0.0,
        }

    subset = df[all_met]
    if "ret_mean" not in subset.columns:
        return {
            "strategy": strategy_name,
            "count": count,
            "mean_ret": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "median_ret": 0.0,
        }

    ret_mean = subset["ret_mean"]
    return {
        "strategy": strategy_name,
        "count": int(count),
        "mean_ret": float(ret_mean.mean()),
        "win_rate": float((ret_mean > 0).sum() / len(ret_mean)),
        "sharpe": float(calculate_sharpe(ret_mean)),
        "median_ret": float(ret_mean.median()),
        "std_ret": float(ret_mean.std()),
    }


def analyze_mean_regime_relaxation():
    """分析MEAN_REGIME条件放宽策略"""

    print("=" * 80)
    print("MEAN_REGIME条件放宽优化分析")
    print("=" * 80)

    # 读取数据
    regime_file = Path("results/e2e_kpi/logs_3action_regime_optimized.parquet")
    if not regime_file.exists():
        print(f"❌ 文件不存在: {regime_file}")
        return

    df = pd.read_parquet(regime_file)
    print(f"\n总样本数: {len(df)}")

    # 当前MEAN_REGIME样本
    current_mean = df[df["regime"] == "MEAN_REGIME"]
    print(f"当前MEAN_REGIME样本数: {len(current_mean)}")

    if len(current_mean) > 0 and "ret_mean" in current_mean.columns:
        current_ret = current_mean["ret_mean"]
        print(f"当前MEAN_REGIME平均ret_mean: {current_ret.mean():.6f}")
        print(f"当前MEAN_REGIME胜率: {(current_ret > 0).sum() / len(current_ret):.1%}")

    # 定义基础条件（当前条件）
    base_conditions = {
        "deviation_z_abs_pct >= 0.6": (df["deviation_z_abs_pct"] >= 0.6)
        & df["deviation_z_abs_pct"].notna(),
        "path_length_pct >= 0.5": (df["path_length_pct"] >= 0.5)
        & df["path_length_pct"].notna(),
        "price_dir_consistency_pct <= 0.5": (df["price_dir_consistency_pct"] <= 0.5)
        & df["price_dir_consistency_pct"].notna(),
        "atr_percentile >= 0.5": (df["atr_percentile"] >= 0.5)
        & df["atr_percentile"].notna(),
        "path_efficiency_pct <= 0.4": (df["path_efficiency_pct"] <= 0.4)
        & df["path_efficiency_pct"].notna(),
        "jump_risk_pct <= 0.3": (df["jump_risk_pct"] <= 0.3)
        & df["jump_risk_pct"].notna(),
    }

    # 测试策略A：保守策略（只放宽最严格的1-2个条件）
    print("\n" + "=" * 80)
    print("策略A：保守策略（只放宽最严格的条件）")
    print("=" * 80)

    strategy_a_results = []

    # A1: 只放宽deviation_z_abs
    conditions_a1 = base_conditions.copy()
    conditions_a1["deviation_z_abs_pct >= 0.5"] = (
        df["deviation_z_abs_pct"] >= 0.5
    ) & df["deviation_z_abs_pct"].notna()
    del conditions_a1["deviation_z_abs_pct >= 0.6"]
    result_a1 = test_relaxation_strategy(
        df, conditions_a1, "A1: 放宽deviation_z_abs到0.5"
    )
    strategy_a_results.append(result_a1)

    # A2: 只放宽jump_risk
    conditions_a2 = base_conditions.copy()
    conditions_a2["jump_risk_pct <= 0.4"] = (df["jump_risk_pct"] <= 0.4) & df[
        "jump_risk_pct"
    ].notna()
    del conditions_a2["jump_risk_pct <= 0.3"]
    result_a2 = test_relaxation_strategy(df, conditions_a2, "A2: 放宽jump_risk到0.4")
    strategy_a_results.append(result_a2)

    # A3: 只放宽path_efficiency
    conditions_a3 = base_conditions.copy()
    conditions_a3["path_efficiency_pct <= 0.5"] = (
        df["path_efficiency_pct"] <= 0.5
    ) & df["path_efficiency_pct"].notna()
    del conditions_a3["path_efficiency_pct <= 0.4"]
    result_a3 = test_relaxation_strategy(
        df, conditions_a3, "A3: 放宽path_efficiency到0.5"
    )
    strategy_a_results.append(result_a3)

    # A4: 放宽deviation_z_abs + jump_risk
    conditions_a4 = base_conditions.copy()
    conditions_a4["deviation_z_abs_pct >= 0.5"] = (
        df["deviation_z_abs_pct"] >= 0.5
    ) & df["deviation_z_abs_pct"].notna()
    conditions_a4["jump_risk_pct <= 0.4"] = (df["jump_risk_pct"] <= 0.4) & df[
        "jump_risk_pct"
    ].notna()
    del conditions_a4["deviation_z_abs_pct >= 0.6"]
    del conditions_a4["jump_risk_pct <= 0.3"]
    result_a4 = test_relaxation_strategy(
        df, conditions_a4, "A4: 放宽deviation_z_abs到0.5 + jump_risk到0.4"
    )
    strategy_a_results.append(result_a4)

    for result in strategy_a_results:
        print(f"\n{result['strategy']}:")
        print(f"  样本数: {result['count']}")
        print(f"  平均ret_mean: {result['mean_ret']:.6f}")
        print(f"  胜率: {result['win_rate']:.1%}")
        print(f"  Sharpe: {result['sharpe']:.3f}")

    # 测试策略B：适度策略（放宽多个条件但保持核心约束）
    print("\n" + "=" * 80)
    print("策略B：适度策略（放宽多个条件）")
    print("=" * 80)

    strategy_b_results = []

    # B1: 适度放宽所有条件
    conditions_b1 = {
        "deviation_z_abs_pct >= 0.5": (df["deviation_z_abs_pct"] >= 0.5)
        & df["deviation_z_abs_pct"].notna(),
        "path_length_pct >= 0.4": (df["path_length_pct"] >= 0.4)
        & df["path_length_pct"].notna(),
        "price_dir_consistency_pct <= 0.6": (df["price_dir_consistency_pct"] <= 0.6)
        & df["price_dir_consistency_pct"].notna(),
        "atr_percentile >= 0.4": (df["atr_percentile"] >= 0.4)
        & df["atr_percentile"].notna(),
        "path_efficiency_pct <= 0.5": (df["path_efficiency_pct"] <= 0.5)
        & df["path_efficiency_pct"].notna(),
        "jump_risk_pct <= 0.4": (df["jump_risk_pct"] <= 0.4)
        & df["jump_risk_pct"].notna(),
    }
    result_b1 = test_relaxation_strategy(df, conditions_b1, "B1: 适度放宽所有条件")
    strategy_b_results.append(result_b1)

    # B2: 放宽核心条件
    conditions_b2 = {
        "deviation_z_abs_pct >= 0.5": (df["deviation_z_abs_pct"] >= 0.5)
        & df["deviation_z_abs_pct"].notna(),
        "path_length_pct >= 0.5": (df["path_length_pct"] >= 0.5)
        & df["path_length_pct"].notna(),
        "price_dir_consistency_pct <= 0.6": (df["price_dir_consistency_pct"] <= 0.6)
        & df["price_dir_consistency_pct"].notna(),
        "atr_percentile >= 0.5": (df["atr_percentile"] >= 0.5)
        & df["atr_percentile"].notna(),
        "path_efficiency_pct <= 0.5": (df["path_efficiency_pct"] <= 0.5)
        & df["path_efficiency_pct"].notna(),
        "jump_risk_pct <= 0.4": (df["jump_risk_pct"] <= 0.4)
        & df["jump_risk_pct"].notna(),
    }
    result_b2 = test_relaxation_strategy(
        df, conditions_b2, "B2: 放宽核心条件（保持path_length和atr）"
    )
    strategy_b_results.append(result_b2)

    for result in strategy_b_results:
        print(f"\n{result['strategy']}:")
        print(f"  样本数: {result['count']}")
        print(f"  平均ret_mean: {result['mean_ret']:.6f}")
        print(f"  胜率: {result['win_rate']:.1%}")
        print(f"  Sharpe: {result['sharpe']:.3f}")

    # 测试策略C：激进策略（大幅放宽以最大化样本数）
    print("\n" + "=" * 80)
    print("策略C：激进策略（大幅放宽）")
    print("=" * 80)

    strategy_c_results = []

    # C1: 大幅放宽所有条件
    conditions_c1 = {
        "deviation_z_abs_pct >= 0.4": (df["deviation_z_abs_pct"] >= 0.4)
        & df["deviation_z_abs_pct"].notna(),
        "path_length_pct >= 0.3": (df["path_length_pct"] >= 0.3)
        & df["path_length_pct"].notna(),
        "price_dir_consistency_pct <= 0.7": (df["price_dir_consistency_pct"] <= 0.7)
        & df["price_dir_consistency_pct"].notna(),
        "atr_percentile >= 0.3": (df["atr_percentile"] >= 0.3)
        & df["atr_percentile"].notna(),
        "path_efficiency_pct <= 0.6": (df["path_efficiency_pct"] <= 0.6)
        & df["path_efficiency_pct"].notna(),
        "jump_risk_pct <= 0.5": (df["jump_risk_pct"] <= 0.5)
        & df["jump_risk_pct"].notna(),
    }
    result_c1 = test_relaxation_strategy(df, conditions_c1, "C1: 大幅放宽所有条件")
    strategy_c_results.append(result_c1)

    for result in strategy_c_results:
        print(f"\n{result['strategy']}:")
        print(f"  样本数: {result['count']}")
        print(f"  平均ret_mean: {result['mean_ret']:.6f}")
        print(f"  胜率: {result['win_rate']:.1%}")
        print(f"  Sharpe: {result['sharpe']:.3f}")

    # 测试策略D：网格搜索（测试关键参数的组合）
    print("\n" + "=" * 80)
    print("策略D：关键参数网格搜索")
    print("=" * 80)

    # 测试关键参数的组合
    deviation_values = [0.5, 0.55, 0.6]
    jump_risk_values = [0.3, 0.35, 0.4]
    path_efficiency_values = [0.4, 0.45, 0.5]

    strategy_d_results = []

    for dev, jr, pe in product(
        deviation_values, jump_risk_values, path_efficiency_values
    ):
        conditions_d = {
            f"deviation_z_abs_pct >= {dev}": (df["deviation_z_abs_pct"] >= dev)
            & df["deviation_z_abs_pct"].notna(),
            "path_length_pct >= 0.5": (df["path_length_pct"] >= 0.5)
            & df["path_length_pct"].notna(),
            "price_dir_consistency_pct <= 0.5": (df["price_dir_consistency_pct"] <= 0.5)
            & df["price_dir_consistency_pct"].notna(),
            "atr_percentile >= 0.5": (df["atr_percentile"] >= 0.5)
            & df["atr_percentile"].notna(),
            f"path_efficiency_pct <= {pe}": (df["path_efficiency_pct"] <= pe)
            & df["path_efficiency_pct"].notna(),
            f"jump_risk_pct <= {jr}": (df["jump_risk_pct"] <= jr)
            & df["jump_risk_pct"].notna(),
        }
        strategy_name = f"D: dev={dev}, jr={jr}, pe={pe}"
        result_d = test_relaxation_strategy(df, conditions_d, strategy_name)
        if result_d["count"] > 0:  # 只保存有样本的策略
            strategy_d_results.append(result_d)

    # 按样本数排序
    strategy_d_results.sort(key=lambda x: x["count"], reverse=True)

    print(f"\n找到 {len(strategy_d_results)} 个有效策略组合")
    print("\n前10个策略（按样本数排序）:")
    for i, result in enumerate(strategy_d_results[:10], 1):
        print(f"\n{i}. {result['strategy']}:")
        print(
            f"   样本数: {result['count']}, 平均ret_mean: {result['mean_ret']:.6f}, 胜率: {result['win_rate']:.1%}, Sharpe: {result['sharpe']:.3f}"
        )

    # 合并所有结果
    all_results = {
        "current": {
            "strategy": "当前条件",
            "count": len(current_mean),
            "mean_ret": (
                float(current_mean["ret_mean"].mean())
                if len(current_mean) > 0 and "ret_mean" in current_mean.columns
                else 0.0
            ),
            "win_rate": (
                float((current_mean["ret_mean"] > 0).sum() / len(current_mean))
                if len(current_mean) > 0 and "ret_mean" in current_mean.columns
                else 0.0
            ),
            "sharpe": (
                float(calculate_sharpe(current_mean["ret_mean"]))
                if len(current_mean) > 0 and "ret_mean" in current_mean.columns
                else 0.0
            ),
        },
        "strategy_a": strategy_a_results,
        "strategy_b": strategy_b_results,
        "strategy_c": strategy_c_results,
        "strategy_d": strategy_d_results,
    }

    # 保存结果
    output_file = Path("results/mean_regime_relaxation_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ 分析结果已保存到: {output_file}")

    return all_results


if __name__ == "__main__":
    analyze_mean_regime_relaxation()
