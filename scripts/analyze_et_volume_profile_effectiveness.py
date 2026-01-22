#!/usr/bin/env python3
"""
分析ET为什么需要Volume Profile - 用数据说明

分析ET gate rules和evidence rules中volume profile的使用，
以及volume profile特征对ET样本表现的影响。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_et_volume_profile_usage() -> Dict[str, Any]:
    """分析ET配置中volume profile的使用"""
    print("=" * 80)
    print("1. ET配置中Volume Profile的使用分析")
    print("=" * 80)

    config_path = PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    et_config = (
        config.get("regimes", {})
        .get("MEAN", {})
        .get("archetypes", {})
        .get("ExhaustionTurnET", {})
    )
    gate_rules = et_config.get("gate_rules", {})
    evidence_rules = et_config.get("evidence_rules", [])

    results = {
        "gate_rules_using_vp": [],
        "evidence_rules_using_vp": [],
        "vp_features_needed": set(),
    }

    # 分析gate rules
    print("\nGate Rules中使用Volume Profile的地方:")
    all_rules = gate_rules.get("rules", [])
    allow_rules = gate_rules.get("allow_if", [])

    for rule in all_rules:
        rule_name = rule.get("name", "")
        rule_key = rule.get("key", "")

        # 检查是否与volume profile相关
        is_vp_related = (
            "lvn" in rule_name.lower()
            or "vp" in rule_key.lower()
            or "volume_profile" in rule_key.lower()
            or "vpvr" in rule_key.lower()
        )

        if is_vp_related:
            results["gate_rules_using_vp"].append(
                {
                    "name": rule_name,
                    "kind": rule.get("kind"),
                    "key": rule_key,
                    "threshold": rule.get("threshold"),
                    "quantile": rule.get("quantile"),
                    "is_allow_rule": rule_name in allow_rules,
                }
            )
            print(f"\n  ✅ {rule_name}:")
            print(f"     kind: {rule.get('kind')}")
            print(f"     key: {rule_key}")
            if rule.get("threshold") is not None:
                print(f"     threshold: {rule.get('threshold')}")
            if rule.get("quantile") is not None:
                print(f"     quantile: {rule.get('quantile')}")
            print(
                f"     类型: {'Allow规则' if rule_name in allow_rules else 'Deny规则'}"
            )

            # 提取需要的特征
            if rule_key:
                results["vp_features_needed"].add(rule_key)

    # 分析evidence rules
    print("\nEvidence Rules中使用Volume Profile的地方:")
    for rule in evidence_rules:
        rule_name = rule.get("name", "")
        if rule_name == "has_volume_profile":
            results["evidence_rules_using_vp"].append(
                {
                    "name": rule_name,
                    "kind": rule.get("kind"),
                    "patterns": rule.get("any_key_contains", []),
                }
            )
            print(f"\n  ✅ {rule_name}:")
            print(f"     kind: {rule.get('kind')}")
            print(f"     匹配模式: {rule.get('any_key_contains', [])}")
            print(f"     作用: 确保有volume profile数据可用")

    results["vp_features_needed"] = sorted(list(results["vp_features_needed"]))

    return results


def analyze_et_semantic_requirements() -> Dict[str, Any]:
    """分析ET的语义需求"""
    print("\n" + "=" * 80)
    print("2. ET的语义需求分析")
    print("=" * 80)

    print("\nET的语义：Exhaustion Turn（趋势衰竭反转）")
    print("\nVolume Profile的作用:")
    print("  1. 识别流动性节点:")
    print("     - LVN (Low Volume Node): 流动性真空，价格容易快速穿越")
    print("     - HVN (High Volume Node): 高成交量区域，可能成为支撑/阻力")
    print("     - POC (Point of Control): 价值中枢，成交量最大的价格")
    print("  2. 检测趋势末期特征:")
    print("     - 成交量集中在特定价格区间（POC附近）")
    print("     - 价格偏离价值中枢（POC），可能回归")
    print("     - 存在多个LVN，表示流动性分散，趋势可能衰竭")
    print("  3. 验证反转信号:")
    print("     - 价格接近LVN时，反转概率增加（et_near_lvn规则）")
    print("     - 成交量分布显示趋势末期的特征")
    print("     - 价格偏离POC，可能回归到价值中枢")

    return {
        "semantic_requirements": {
            "liquidity_nodes": "识别LVN和HVN",
            "trend_exhaustion": "检测趋势末期特征",
            "reversal_signals": "验证反转信号",
        }
    }


def analyze_et_samples_with_vp(df: pd.DataFrame) -> Dict[str, Any]:
    """分析ET样本中volume profile特征的表现"""
    print("\n" + "=" * 80)
    print("3. ET样本中Volume Profile特征的表现分析")
    print("=" * 80)

    # 筛选ET样本
    et_samples = df[
        (df.get("regime", "") == "ET_REGIME")
        & (
            df.get("gate_archetype", "")
            .astype(str)
            .str.contains("ET", case=False, na=False)
        )
    ].copy()

    if len(et_samples) == 0:
        print("\n⚠️  当前没有ET样本，需要先运行regime分类和gate检查")
        return {"sample_count": 0}

    print(f"\nET样本数: {len(et_samples)}")

    # 检查volume profile特征
    vp_cols = [
        col
        for col in et_samples.columns
        if "vp" in col.lower()
        or "volume_profile" in col.lower()
        or "vpvr" in col.lower()
    ]

    if not vp_cols:
        print("\n⚠️  当前数据中没有volume profile特征")
        print("需要重建FeatureStore以包含这些特征")
        return {
            "sample_count": len(et_samples),
            "vp_features_found": False,
        }

    print(f"\n找到 {len(vp_cols)} 个volume profile相关特征:")
    for col in sorted(vp_cols)[:10]:
        non_null = et_samples[col].notna().sum()
        print(
            f"  {col}: {non_null}/{len(et_samples)} 非空 ({non_null/len(et_samples)*100:.1f}%)"
        )

    # 分析et_near_lvn规则的影响
    if "vpvr_lvn_distance" in et_samples.columns:
        lvn_dist = et_samples["vpvr_lvn_distance"].dropna()
        if len(lvn_dist) > 0:
            print(f"\n4. vpvr_lvn_distance分析:")
            print(f"   平均值: {lvn_dist.mean():.4f}")
            print(f"   中位数: {lvn_dist.median():.4f}")
            print(f"   最小值: {lvn_dist.min():.4f}")
            print(f"   最大值: {lvn_dist.max():.4f}")

            # 检查et_near_lvn规则（假设threshold是0.1）
            near_lvn = lvn_dist <= 0.1
            print(
                f"\n   接近LVN的样本 (distance <= 0.1): {near_lvn.sum()}/{len(lvn_dist)} ({near_lvn.sum()/len(lvn_dist)*100:.1f}%)"
            )

            if "ret_mean" in et_samples.columns and len(near_lvn) > 0:
                near_lvn_samples = et_samples[near_lvn]
                far_lvn_samples = et_samples[~near_lvn]

                if len(near_lvn_samples) > 0 and len(far_lvn_samples) > 0:
                    near_ret = near_lvn_samples["ret_mean"].dropna()
                    far_ret = far_lvn_samples["ret_mean"].dropna()

                    if len(near_ret) > 0 and len(far_ret) > 0:
                        print(f"\n   表现对比:")
                        print(
                            f"     接近LVN: 平均ret_mean={near_ret.mean():.6f}, 胜率={(near_ret>0).sum()/len(near_ret)*100:.1f}%"
                        )
                        print(
                            f"     远离LVN: 平均ret_mean={far_ret.mean():.6f}, 胜率={(far_ret>0).sum()/len(far_ret)*100:.1f}%"
                        )

    return {
        "sample_count": len(et_samples),
        "vp_features_found": len(vp_cols) > 0,
        "vp_features": sorted(vp_cols),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze why ET needs volume profile")
    p.add_argument(
        "--logs", default=None, help="Optional logs file for sample analysis"
    )
    p.add_argument(
        "--output-json",
        default="results/et_volume_profile_analysis.json",
        help="Output JSON",
    )
    args = p.parse_args()

    # 1. 分析ET配置中volume profile的使用
    usage_analysis = analyze_et_volume_profile_usage()

    # 2. 分析ET的语义需求
    semantic_analysis = analyze_et_semantic_requirements()

    # 3. 如果有logs文件，分析ET样本
    sample_analysis = {}
    if args.logs and Path(args.logs).exists():
        df = pd.read_parquet(args.logs)
        sample_analysis = analyze_et_samples_with_vp(df)
    else:
        print("\n" + "=" * 80)
        print("3. ET样本分析（跳过：需要logs文件）")
        print("=" * 80)
        print("\n提示: 使用--logs参数提供logs文件以进行样本分析")

    # 保存结果
    output = {
        "volume_profile_usage": usage_analysis,
        "semantic_requirements": semantic_analysis,
        "sample_analysis": sample_analysis,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 分析结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
