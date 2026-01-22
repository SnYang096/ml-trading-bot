#!/usr/bin/env python3
"""
深入分析FR/ET的数据分布，找出适合mean reversion的特征模式
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def analyze_fr_et_distribution():
    """分析FR/ET的数据分布"""

    # 加载数据
    all_off = pd.read_parquet(
        "results/experiments_regenerated/all_veto_off_gated.parquet"
    )
    fr_et = all_off[
        all_off["gate_archetype"].str.contains("FR|ET", case=False, na=False)
    ].copy()

    print("=" * 80)
    print("FR/ET深度数据分布分析")
    print("=" * 80)
    print(f"\n总FR/ET交易数: {len(fr_et)}")
    print(f"总交易数: {len(all_off[all_off['gate_ok'] == True])}")

    if len(fr_et) == 0:
        print("没有FR/ET交易数据")
        return

    results = {}

    # 1. 分析收益好的vs收益差的FR/ET特征差异
    fr_et_sorted = fr_et.sort_values("ret_mean", ascending=False)
    top_10pct = fr_et_sorted.head(int(len(fr_et) * 0.1))
    bottom_10pct = fr_et_sorted.tail(int(len(fr_et) * 0.1))

    print("\n" + "=" * 80)
    print("1. 收益最好的10% vs 收益最差的10% FR/ET特征对比")
    print("=" * 80)

    features_to_compare = [
        "path_efficiency_pct",
        "price_dir_consistency_pct",
        "deviation_z_abs_pct",
        "atr_percentile",
        "jump_risk_pct",
        "path_length_pct",
        "atr_slope_pct",
        "fr_semantic_score",
        "et_semantic_score",
        "cvd_change_5",
        "vpin",
        "volume_ratio",
        "bb_width_normalized",
    ]

    top_bottom_comparison = {}
    for feat in features_to_compare:
        if feat in fr_et.columns:
            top_vals = top_10pct[feat].dropna()
            bottom_vals = bottom_10pct[feat].dropna()
            if len(top_vals) > 0 and len(bottom_vals) > 0:
                top_bottom_comparison[feat] = {
                    "top_mean": float(top_vals.mean()),
                    "top_median": float(top_vals.median()),
                    "top_std": float(top_vals.std()),
                    "bottom_mean": float(bottom_vals.mean()),
                    "bottom_median": float(bottom_vals.median()),
                    "bottom_std": float(bottom_vals.std()),
                }
                print(f"\n{feat}:")
                print(
                    f"  最好10%: mean={top_vals.mean():.3f}, median={top_vals.median():.3f}, std={top_vals.std():.3f}"
                )
                print(
                    f"  最差10%: mean={bottom_vals.mean():.3f}, median={bottom_vals.median():.3f}, std={bottom_vals.std():.3f}"
                )
                if top_vals.mean() != bottom_vals.mean():
                    diff_pct = (
                        (
                            (top_vals.mean() - bottom_vals.mean())
                            / abs(bottom_vals.mean())
                            * 100
                        )
                        if bottom_vals.mean() != 0
                        else 0
                    )
                    print(f"  差异: {diff_pct:+.1f}%")

    results["top_bottom_comparison"] = top_bottom_comparison

    # 2. 找出"黄金特征区间"（收益>0的样本）
    profitable = fr_et[fr_et["ret_mean"] > 0]
    unprofitable = fr_et[fr_et["ret_mean"] <= 0]

    print("\n" + "=" * 80)
    print("2. 盈利vs亏损FR/ET特征对比")
    print("=" * 80)
    print(f"盈利样本数: {len(profitable)} ({len(profitable)/len(fr_et)*100:.1f}%)")
    print(f"亏损样本数: {len(unprofitable)} ({len(unprofitable)/len(fr_et)*100:.1f}%)")

    profitable_vs_unprofitable = {}
    for feat in features_to_compare:
        if feat in fr_et.columns:
            prof_vals = profitable[feat].dropna()
            unprof_vals = unprofitable[feat].dropna()
            if len(prof_vals) > 0 and len(unprof_vals) > 0:
                profitable_vs_unprofitable[feat] = {
                    "profitable_mean": float(prof_vals.mean()),
                    "profitable_median": float(prof_vals.median()),
                    "unprofitable_mean": float(unprof_vals.mean()),
                    "unprofitable_median": float(unprof_vals.median()),
                }
                print(f"\n{feat}:")
                print(
                    f"  盈利: mean={prof_vals.mean():.3f}, median={prof_vals.median():.3f}"
                )
                print(
                    f"  亏损: mean={unprof_vals.mean():.3f}, median={unprof_vals.median():.3f}"
                )

    results["profitable_vs_unprofitable"] = profitable_vs_unprofitable

    # 3. 分析不同regime下的FR/ET表现
    print("\n" + "=" * 80)
    print("3. FR/ET在不同Regime下的详细表现")
    print("=" * 80)

    regime_performance = {}
    if "regime" in fr_et.columns:
        regime_groups = fr_et.groupby("regime")
        for regime, group in regime_groups:
            regime_perf = {
                "count": int(len(group)),
                "mean_return": float(group["ret_mean"].mean()),
                "std_return": float(group["ret_mean"].std()),
                "win_rate": float((group["ret_mean"] > 0).sum() / len(group)),
            }

            if len(group) > 10:  # 只对样本数足够的regime显示特征
                for feat in [
                    "path_efficiency_pct",
                    "price_dir_consistency_pct",
                    "deviation_z_abs_pct",
                ]:
                    if feat in group.columns:
                        vals = group[feat].dropna()
                        if len(vals) > 0:
                            regime_perf[feat] = float(vals.mean())

            regime_performance[regime] = regime_perf

            print(f"\n{regime} (n={len(group)}):")
            print(f"  平均收益: {group['ret_mean'].mean():.6f}")
            print(f"  收益std: {group['ret_mean'].std():.6f}")
            print(f"  胜率: {(group['ret_mean'] > 0).sum() / len(group):.1%}")
            if len(group) > 10:
                for feat in [
                    "path_efficiency_pct",
                    "price_dir_consistency_pct",
                    "deviation_z_abs_pct",
                ]:
                    if feat in group.columns:
                        vals = group[feat].dropna()
                        if len(vals) > 0:
                            print(f"  {feat}: {vals.mean():.3f}")

    results["regime_performance"] = regime_performance

    # 4. 分析gate_rules的过滤效果
    print("\n" + "=" * 80)
    print("4. FR/ET Gate Rules过滤效果分析")
    print("=" * 80)

    passed = fr_et[fr_et["gate_ok"] == True]
    failed = fr_et[fr_et["gate_ok"] == False]

    print(f"\n总FR/ET候选数: {len(fr_et)}")
    print(f"通过gate: {len(passed)} ({len(passed)/len(fr_et)*100:.1f}%)")
    print(f"被gate拒绝: {len(failed)} ({len(failed)/len(fr_et)*100:.1f}%)")

    gate_filter_analysis = {
        "total_candidates": int(len(fr_et)),
        "passed_count": int(len(passed)),
        "failed_count": int(len(failed)),
        "pass_rate": float(len(passed) / len(fr_et)) if len(fr_et) > 0 else 0.0,
    }

    if len(passed) > 0:
        gate_filter_analysis["passed_mean_return"] = float(passed["ret_mean"].mean())
        gate_filter_analysis["passed_win_rate"] = float(
            (passed["ret_mean"] > 0).sum() / len(passed)
        )
        gate_filter_analysis["passed_std_return"] = float(passed["ret_mean"].std())

    if len(passed) > 0 and len(failed) > 0:
        print("\n通过vs被拒绝的特征对比:")
        gate_feature_comparison = {}
        for feat in features_to_compare:
            if feat in fr_et.columns:
                passed_vals = passed[feat].dropna()
                failed_vals = failed[feat].dropna()
                if len(passed_vals) > 0 and len(failed_vals) > 0:
                    gate_feature_comparison[feat] = {
                        "passed_mean": float(passed_vals.mean()),
                        "passed_median": float(passed_vals.median()),
                        "failed_mean": float(failed_vals.mean()),
                        "failed_median": float(failed_vals.median()),
                    }
                    print(f"\n{feat}:")
                    print(
                        f"  通过: mean={passed_vals.mean():.3f}, median={passed_vals.median():.3f}"
                    )
                    print(
                        f"  拒绝: mean={failed_vals.mean():.3f}, median={failed_vals.median():.3f}"
                    )

        gate_filter_analysis["feature_comparison"] = gate_feature_comparison

        print("\n通过gate的FR/ET收益:")
        print(f"  平均收益: {passed['ret_mean'].mean():.6f}")
        print(f"  胜率: {(passed['ret_mean'] > 0).sum() / len(passed):.1%}")
        print(f"  收益std: {passed['ret_mean'].std():.6f}")

    results["gate_filter_analysis"] = gate_filter_analysis

    # 5. 找出"黄金特征区间"
    print("\n" + "=" * 80)
    print("5. FR/ET黄金特征区间（基于盈利样本）")
    print("=" * 80)

    golden_ranges = {}
    for feat in [
        "path_efficiency_pct",
        "price_dir_consistency_pct",
        "deviation_z_abs_pct",
    ]:
        if feat in profitable.columns and len(profitable) > 0:
            vals = profitable[feat].dropna()
            if len(vals) > 0:
                golden_ranges[feat] = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "p25": float(vals.quantile(0.25)),
                    "p75": float(vals.quantile(0.75)),
                }
                print(f"\n{feat} (盈利样本):")
                print(f"  范围: [{vals.min():.3f}, {vals.max():.3f}]")
                print(f"  均值: {vals.mean():.3f}, 中位数: {vals.median():.3f}")
                print(
                    f"  25%-75%分位: [{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]"
                )

    results["golden_ranges"] = golden_ranges

    # 保存结果
    output_file = Path("results/fr_et_distribution_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n分析结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    analyze_fr_et_distribution()
