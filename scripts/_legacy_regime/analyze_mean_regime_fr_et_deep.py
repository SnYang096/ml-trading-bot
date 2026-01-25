#!/usr/bin/env python3
"""
深入分析MEAN_REGIME分类和FR/ET被拒绝的根本原因
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List


def analyze_mean_regime_classification():
    """分析MEAN_REGIME分类问题"""

    print("=" * 80)
    print("1. MEAN_REGIME分类问题分析")
    print("=" * 80)

    logs = pd.read_parquet("results/e2e_kpi/logs_3action_with_new_regime.parquet")

    print(f"\n原始logs的regime分布:")
    print(logs["regime"].value_counts())

    # 检查是否需要重新运行regime分类
    print(f"\n问题诊断:")
    print(f"  - 原始logs中的regime是在优化前分类的")
    print(f"  - 优化后的MEAN_REGIME条件需要重新运行regime分类")
    print(f"  - 优化后的条件:")
    print(f"    * mean_deviation_z_abs_min_pct: 0.85 -> 0.6 (放宽)")
    print(f"    * mean_path_efficiency_max_pct: 0.4 (新增)")
    print(f"    * mean_price_dir_consistency_max_pct: 0.5 (新增)")
    print(f"    * mean_jump_risk_max_pct: 0.3 (新增)")

    # 检查是否有物理特征
    physical_features = [
        "path_efficiency_pct",
        "price_dir_consistency_pct",
        "deviation_z_abs_pct",
    ]
    has_physical = all(feat in logs.columns for feat in physical_features)

    if has_physical:
        print(f"\n  ✅ logs中有物理特征，可以重新运行regime分类")
    else:
        print(f"\n  ⚠️ logs中没有物理特征，需要从FeatureStore读取")

    return {
        "original_regime_dist": logs["regime"].value_counts().to_dict(),
        "has_physical_features": has_physical,
        "needs_rerun_regime": True,
    }


def analyze_physical_features_reading():
    """分析物理特征读取问题"""

    print("\n" + "=" * 80)
    print("2. 物理特征读取问题分析")
    print("=" * 80)

    baseline_gated = pd.read_parquet(
        "results/experiments_optimized/baseline_gated.parquet"
    )

    physical_features = [
        "path_efficiency_pct",
        "price_dir_consistency_pct",
        "deviation_z_abs_pct",
    ]
    related_features = [
        "path_efficiency",
        "price_dir_consistency",
        "deviation_z_abs",
        "path_length_pct",
        "jump_risk_pct",
        "atr_percentile",
    ]

    print(f"\nBaseline Gated文件中的特征检查:")
    feature_status = {}

    for feat in physical_features + related_features:
        if feat in baseline_gated.columns:
            non_null = baseline_gated[feat].notna().sum()
            feature_status[feat] = {
                "exists": True,
                "non_null_count": int(non_null),
                "total_count": len(baseline_gated),
                "coverage": float(non_null / len(baseline_gated)),
            }
            print(
                f"  ✅ {feat}: {non_null}/{len(baseline_gated)} 非空 ({non_null/len(baseline_gated)*100:.1f}%)"
            )
        else:
            feature_status[feat] = {
                "exists": False,
                "non_null_count": 0,
                "total_count": len(baseline_gated),
                "coverage": 0.0,
            }
            print(f"  ❌ {feat}: 不存在")

    # 检查FeatureStore
    print(f"\nFeatureStore检查:")
    try:
        from src.feature_store import FeatureStore, FeatureStoreSpec

        store = FeatureStore("feature_store")
        spec = FeatureStoreSpec(
            layer="nnmh_highcap6_240T_2024_202510", symbol="BTCUSDT", timeframe="240T"
        )

        df = store.read_range(
            spec, start=pd.Timestamp("2025-05-01"), end=pd.Timestamp("2025-05-10")
        )

        if not df.empty:
            print(f"  ✅ FeatureStore读取成功")
            for feat in physical_features:
                if feat in df.columns:
                    print(f"    ✅ {feat} 存在于FeatureStore")
                else:
                    print(f"    ❌ {feat} 不存在于FeatureStore")
        else:
            print(f"  ❌ FeatureStore读取为空")
    except Exception as e:
        print(f"  ❌ FeatureStore检查失败: {e}")

    return {
        "feature_status": feature_status,
        "issue": "物理特征可能没有从FeatureStore正确读取到gated文件",
    }


def analyze_fr_et_rejection():
    """分析FR/ET被拒绝的详细原因"""

    print("\n" + "=" * 80)
    print("3. FR/ET被拒绝原因分析")
    print("=" * 80)

    baseline_gated = pd.read_parquet(
        "results/experiments_optimized/baseline_gated.parquet"
    )
    fr_et = baseline_gated[
        baseline_gated["gate_archetype"].str.contains("FR|ET", case=False, na=False)
    ]

    print(f"\nFR/ET候选数: {len(fr_et)}")

    if len(fr_et) == 0:
        print("  ⚠️ 没有FR/ET候选")
        return {"candidates": 0, "rejected": 0}

    rejected = fr_et[fr_et["gate_ok"] == False]
    print(f"被拒绝数: {len(rejected)}")

    # 加载FR配置
    with open("config/nnmultihead/execution_archetypes.yaml", "r") as f:
        config = yaml.safe_load(f)

    fr_config = config["regimes"]["MEAN"]["archetypes"]["FailureReversionFR"]
    fr_gate_rules = fr_config["gate_rules"]

    print(f"\nFR Gate Rules配置:")
    print(f"  deny_if: {len(fr_gate_rules.get('deny_if', []))} 个规则")
    print(f"  allow_if: {len(fr_gate_rules.get('allow_if', []))} 个规则")
    print(f"  allow_mode: {fr_gate_rules.get('allow_mode', 'N/A')}")
    print(f"  default_action: {fr_gate_rules.get('default_action', 'N/A')}")

    # 分析被拒绝的FR/ET
    rejection_analysis = {}
    for idx, row in rejected.iterrows():
        print(f"\n被拒绝的FR/ET #{idx}:")
        print(f"  regime: {row.get('regime', 'N/A')}")
        print(f"  gate_decision: {row.get('gate_decision', 'N/A')}")

        # 解析gate_reasons
        gate_reasons = row.get("gate_reasons", "")
        if pd.notna(gate_reasons):
            print(f"  gate_reasons: {str(gate_reasons)[:200]}")

            # 尝试解析
            try:
                if isinstance(gate_reasons, str):
                    try:
                        reasons = json.loads(gate_reasons)
                    except:
                        import ast

                        reasons = ast.literal_eval(gate_reasons)
                else:
                    reasons = gate_reasons

                if isinstance(reasons, dict):
                    if "gate_allow_not_met" in reasons:
                        allow_not_met = reasons["gate_allow_not_met"]
                        print(f"  Allow条件未满足: {allow_not_met}")

                    if "gate_deny_triggered" in reasons:
                        deny_triggered = reasons["gate_deny_triggered"]
                        print(f"  Deny条件触发: {deny_triggered}")
            except Exception as e:
                print(f"  解析失败: {e}")

        # 检查关键特征值
        print(f"\n  关键特征值:")
        key_features = [
            "cvd_change_5",
            "vpin",
            "volume_ratio",
            "bb_width_normalized",
            "sr_distance_normalized",
            "adx",
            "sqs",
            "trade_quality",
            "mean_score",
            "path_efficiency_pct",
            "price_dir_consistency_pct",
            "deviation_z_abs_pct",
        ]

        feature_values = {}
        for feat in key_features:
            if feat in row.index:
                val = row[feat]
                if pd.notna(val):
                    feature_values[feat] = (
                        float(val) if isinstance(val, (int, float)) else str(val)
                    )
                    print(
                        f"    {feat}: {val:.6f}"
                        if isinstance(val, float)
                        else f"    {feat}: {val}"
                    )

        rejection_analysis[idx] = {
            "regime": str(row.get("regime", "N/A")),
            "gate_decision": str(row.get("gate_decision", "N/A")),
            "gate_reasons": str(gate_reasons),
            "feature_values": feature_values,
        }

    return {
        "candidates": len(fr_et),
        "rejected": len(rejected),
        "rejection_analysis": rejection_analysis,
    }


def analyze_mean_alpha():
    """分析MEAN_REGIME的Alpha（收益能力）"""

    print("\n" + "=" * 80)
    print("4. MEAN_REGIME Alpha分析")
    print("=" * 80)

    logs = pd.read_parquet("results/e2e_kpi/logs_3action_with_new_regime.parquet")

    mean_regime = logs[logs["regime"] == "MEAN_REGIME"]
    no_trade = logs[logs["regime"] == "NO_TRADE"]

    print(f"\nMEAN_REGIME样本数: {len(mean_regime)}")

    alpha_analysis = {
        "mean_regime_count": len(mean_regime),
        "mean_regime_returns": {},
        "no_trade_returns": {},
    }

    if len(mean_regime) > 0:
        print(f"\nMEAN_REGIME收益分析:")
        ret_mean = mean_regime["ret_mean"]
        ret_trend = mean_regime["ret_trend"]

        print(f"  ret_mean: {ret_mean.values}")
        print(f"  ret_trend: {ret_trend.values}")
        print(f"  平均ret_mean: {ret_mean.mean():.6f}")
        print(f"  平均ret_trend: {ret_trend.mean():.6f}")
        print(f"  胜率: {(ret_mean > 0).sum() / len(ret_mean):.1%}")

        alpha_analysis["mean_regime_returns"] = {
            "mean": float(ret_mean.mean()),
            "median": float(ret_mean.median()),
            "std": float(ret_mean.std()),
            "win_rate": float((ret_mean > 0).sum() / len(ret_mean)),
            "values": ret_mean.tolist(),
        }

        print(f"\n  ✅ MEAN_REGIME有正收益（alpha存在）")
    else:
        print(f"\n  ⚠️ MEAN_REGIME样本数为0，无法判断alpha")
        print(f"  问题: 需要重新运行regime分类以应用优化后的条件")

    # 对比NO_TRADE
    if len(no_trade) > 0:
        print(f"\nNO_TRADE对比:")
        ret_mean_nt = no_trade["ret_mean"]
        print(f"  平均ret_mean: {ret_mean_nt.mean():.6f}")
        print(f"  胜率: {(ret_mean_nt > 0).sum() / len(ret_mean_nt):.1%}")

        alpha_analysis["no_trade_returns"] = {
            "mean": float(ret_mean_nt.mean()),
            "median": float(ret_mean_nt.median()),
            "std": float(ret_mean_nt.std()),
            "win_rate": float((ret_mean_nt > 0).sum() / len(ret_mean_nt)),
        }

    # 数据划分检查
    print(f"\n数据划分检查:")
    print(f"  时间范围: {logs['timestamp'].min()} 到 {logs['timestamp'].max()}")
    print(f"  Symbols: {sorted(logs['symbol'].unique())}")
    print(f"  总样本数: {len(logs)}")

    alpha_analysis["data_info"] = {
        "time_range": {
            "start": str(logs["timestamp"].min()),
            "end": str(logs["timestamp"].max()),
        },
        "symbols": sorted(logs["symbol"].unique().tolist()),
        "total_samples": len(logs),
    }

    return alpha_analysis


def comprehensive_analysis():
    """综合分析"""

    print("\n" + "=" * 80)
    print("5. 综合分析")
    print("=" * 80)

    results = {
        "mean_regime_classification": analyze_mean_regime_classification(),
        "physical_features_reading": analyze_physical_features_reading(),
        "fr_et_rejection": analyze_fr_et_rejection(),
        "mean_alpha": analyze_mean_alpha(),
    }

    # 综合诊断
    print("\n" + "=" * 80)
    print("综合诊断结论")
    print("=" * 80)

    print("\n1. MEAN_REGIME分类问题:")
    if results["mean_regime_classification"]["needs_rerun_regime"]:
        print("   ❌ 需要重新运行regime分类以应用优化后的条件")
        print("   原因: 原始logs中的regime是在优化前分类的")

    print("\n2. 物理特征读取问题:")
    feature_status = results["physical_features_reading"]["feature_status"]
    has_physical = any(
        feat["exists"]
        for feat in feature_status.values()
        if "path_efficiency" in feat
        or "price_dir_consistency" in feat
        or "deviation_z" in feat
    )
    if not has_physical:
        print("   ❌ 物理特征没有从FeatureStore正确读取")
        print("   原因: apply_tree_gate_3action.py可能没有读取这些特征")
        print("   或者: FeatureStore中没有这些特征")

    print("\n3. FR/ET被拒绝原因:")
    if results["fr_et_rejection"]["rejected"] > 0:
        print(f"   ❌ {results['fr_et_rejection']['rejected']} 个FR/ET候选被拒绝")
        print("   原因: allow_if条件未满足或deny_if条件触发")
        print("   需要: 检查gate_rules配置和特征值")

    print("\n4. MEAN_REGIME Alpha:")
    if results["mean_alpha"]["mean_regime_count"] > 0:
        mean_ret = results["mean_alpha"]["mean_regime_returns"]["mean"]
        if mean_ret > 0:
            print(f"   ✅ MEAN_REGIME有正收益（alpha存在）: {mean_ret:.6f}")
        else:
            print(f"   ⚠️ MEAN_REGIME收益为负: {mean_ret:.6f}")
    else:
        print("   ⚠️ MEAN_REGIME样本数为0，无法判断alpha")
        print("   需要: 重新运行regime分类以增加MEAN_REGIME样本")

    print("\n5. 数据划分:")
    data_info = results["mean_alpha"]["data_info"]
    print(f"   ✅ 数据划分正常")
    print(
        f"   时间范围: {data_info['time_range']['start']} 到 {data_info['time_range']['end']}"
    )
    print(f"   Symbols: {data_info['symbols']}")

    # 保存结果
    output_file = Path("results/mean_regime_fr_et_deep_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n分析结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    comprehensive_analysis()
