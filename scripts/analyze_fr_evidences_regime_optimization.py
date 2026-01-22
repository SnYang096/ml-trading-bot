#!/usr/bin/env python3
"""
FR Evidences单独深度分析

分析目标:
1. 分析不同regime下FR evidences的表现
2. 找出适合FR的regime特征范围
3. 优化evidence参数（quantile阈值等）
4. 扩大数据范围寻找更多样本
"""

import pandas as pd
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.feature_store import FeatureStore, FeatureStoreSpec


def calculate_sharpe(returns: pd.Series) -> float:
    """计算Sharpe比率"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0


def compute_quantiles_from_data(
    df: pd.DataFrame, symbol_col: str = "symbol"
) -> Dict[str, Any]:
    """从数据中计算quantiles"""
    quantiles = {}
    symbols = df[symbol_col].unique() if symbol_col in df.columns else ["ALL"]

    for symbol in symbols:
        symbol_df = df[df[symbol_col] == symbol] if symbol_col in df.columns else df
        quantiles[symbol] = {}

        numeric_cols = symbol_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ["ret_mean", "timestamp"]:
                continue
            values = symbol_df[col].dropna()
            if len(values) > 0:
                quantiles[symbol][col] = {}
                for q in [
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.8,
                    0.9,
                    0.95,
                ]:
                    try:
                        quantiles[symbol][col][f"{q:.2f}"] = float(values.quantile(q))
                    except Exception:
                        pass

    return quantiles


def check_required_evidence(
    evidence_flags: Dict[str, bool],
    required_evidence: List[str],
) -> bool:
    """检查是否满足required_evidence"""
    if not required_evidence:
        return True
    return all(evidence_flags.get(ev, False) for ev in required_evidence)


def apply_evidence_filter(
    df: pd.DataFrame,
    evidence_rules: List[Dict[str, Any]],
    required_evidence: List[str],
    quantiles: Dict[str, Any] | None = None,
    symbol_col: str = "symbol",
) -> pd.Series:
    """应用evidence_rules过滤样本"""
    mask = pd.Series(False, index=df.index)

    for idx, row in df.iterrows():
        features = row.to_dict()
        symbol = features.get(symbol_col, "ALL")
        symbol_quantiles = quantiles.get(symbol, {}) if quantiles else None

        try:
            evidence_flags = compute_execution_evidence(
                features=features,
                rules=evidence_rules,
                quantiles=symbol_quantiles,
            )

            if check_required_evidence(evidence_flags, required_evidence):
                mask.loc[idx] = True
        except Exception:
            continue

    return mask


def analyze_fr_by_regime(
    df: pd.DataFrame,
    fr_evidence_rules: List[Dict[str, Any]],
    fr_required_evidence: List[str],
    quantiles: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """分析不同regime下FR的表现"""
    results = {}

    regimes = df["regime"].unique() if "regime" in df.columns else ["ALL"]

    for regime in regimes:
        regime_df = df[df["regime"] == regime] if "regime" in df.columns else df

        if len(regime_df) == 0:
            continue

        # 应用FR evidences
        evidence_mask = apply_evidence_filter(
            regime_df, fr_evidence_rules, fr_required_evidence, quantiles
        )
        evidence_passed = regime_df[evidence_mask]

        if len(evidence_passed) == 0:
            results[regime] = {
                "total_samples": len(regime_df),
                "evidence_passed": 0,
                "mean_ret": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
            }
            continue

        if "ret_mean" not in evidence_passed.columns:
            results[regime] = {
                "total_samples": len(regime_df),
                "evidence_passed": len(evidence_passed),
                "mean_ret": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
            }
            continue

        ret_mean = evidence_passed["ret_mean"]
        results[regime] = {
            "total_samples": len(regime_df),
            "evidence_passed": len(evidence_passed),
            "mean_ret": float(ret_mean.mean()),
            "win_rate": float((ret_mean > 0).sum() / len(ret_mean)),
            "sharpe": float(calculate_sharpe(ret_mean)),
            "median_ret": float(ret_mean.median()),
            "std_ret": float(ret_mean.std()),
        }

    return results


def analyze_evidence_parameters(
    df: pd.DataFrame,
    base_evidence_rules: List[Dict[str, Any]],
    fr_required_evidence: List[str],
    quantiles: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """分析不同evidence参数组合的表现"""
    results = []

    # 找到has_orderflow规则
    has_orderflow_rule = None
    for rule in base_evidence_rules:
        if rule.get("name") == "has_orderflow":
            has_orderflow_rule = rule.copy()
            break

    if not has_orderflow_rule:
        return results

    # 测试不同的quantile阈值
    quantile_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    for q in quantile_values:
        # 创建修改后的evidence rules
        modified_rules = []
        for rule in base_evidence_rules:
            rule_copy = rule.copy()
            if rule_copy.get("name") == "has_orderflow":
                rule_copy["quantile"] = q
            modified_rules.append(rule_copy)

        # 应用evidences
        evidence_mask = apply_evidence_filter(
            df, modified_rules, fr_required_evidence, quantiles
        )
        evidence_passed = df[evidence_mask]

        if len(evidence_passed) == 0:
            continue

        if "ret_mean" not in evidence_passed.columns:
            continue

        ret_mean = evidence_passed["ret_mean"]
        results.append(
            {
                "has_orderflow_quantile": q,
                "evidence_passed": len(evidence_passed),
                "mean_ret": float(ret_mean.mean()),
                "win_rate": float((ret_mean > 0).sum() / len(ret_mean)),
                "sharpe": float(calculate_sharpe(ret_mean)),
            }
        )

    return results


def find_optimal_regime_features(
    df: pd.DataFrame,
    fr_evidence_passed: pd.DataFrame,
) -> Dict[str, Any]:
    """找出适合FR的regime特征范围"""
    if len(fr_evidence_passed) == 0:
        return {}

    # 分析通过FR evidences的样本的物理特征分布
    physical_features = [
        "path_efficiency_pct",
        "price_dir_consistency_pct",
        "deviation_z_abs_pct",
        "path_length_pct",
        "jump_risk_pct",
        "atr_percentile",
    ]

    results = {}

    for feat in physical_features:
        if feat not in fr_evidence_passed.columns:
            continue

        values = fr_evidence_passed[feat].dropna()
        if len(values) == 0:
            continue

        # 分析正收益和负收益样本的特征分布
        positive_ret = fr_evidence_passed[fr_evidence_passed["ret_mean"] > 0]
        negative_ret = fr_evidence_passed[fr_evidence_passed["ret_mean"] <= 0]

        if len(positive_ret) > 0 and feat in positive_ret.columns:
            pos_values = positive_ret[feat].dropna()
            if len(pos_values) > 0:
                results[f"{feat}_positive"] = {
                    "mean": float(pos_values.mean()),
                    "median": float(pos_values.median()),
                    "min": float(pos_values.min()),
                    "max": float(pos_values.max()),
                    "p25": float(pos_values.quantile(0.25)),
                    "p75": float(pos_values.quantile(0.75)),
                }

        if len(negative_ret) > 0 and feat in negative_ret.columns:
            neg_values = negative_ret[feat].dropna()
            if len(neg_values) > 0:
                results[f"{feat}_negative"] = {
                    "mean": float(neg_values.mean()),
                    "median": float(neg_values.median()),
                    "min": float(neg_values.min()),
                    "max": float(neg_values.max()),
                    "p25": float(neg_values.quantile(0.25)),
                    "p75": float(neg_values.quantile(0.75)),
                }

        # 计算所有样本的分布
        results[f"{feat}_all"] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="FR Evidences深度分析")
    parser.add_argument("--logs", required=True, help="Logs文件路径")
    parser.add_argument("--feature-store-root", default="feature_store")
    parser.add_argument(
        "--feature-store-layer",
        default="nnmh_highcap6_240T_2024_202510_v2",
        help="FeatureStore layer name (default: nnmh_highcap6_240T_2024_202510_v2)",
    )
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--execution-archetypes", default="config/nnmultihead/execution_archetypes.yaml"
    )
    parser.add_argument(
        "--output", default="results/fr_evidences_regime_optimization.json"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FR Evidences深度分析 - Regime和Evidence参数优化")
    print("=" * 80)

    # 读取数据
    df = pd.read_parquet(args.logs)
    print(f"\n总样本数: {len(df)}")

    # 检查必需的订单流特征
    required_features = ["vpin", "cvd_change_5", "cvd_change_5_normalized"]
    missing_features = [f for f in required_features if f not in df.columns]

    # 从FeatureStore读取缺失的特征（如果需要）
    if missing_features:
        print(f"\n尝试从FeatureStore读取特征...")
        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
        if symbols:
            from src.feature_store import FeatureStore, FeatureStoreSpec

            store = FeatureStore(args.feature_store_root)
            parts = []
            for sym in symbols:
                spec = FeatureStoreSpec(
                    layer=args.feature_store_layer,
                    symbol=str(sym),
                    timeframe=args.timeframe,
                )
                start_ts = (
                    pd.Timestamp(args.start_date)
                    if args.start_date
                    else pd.Timestamp("1970-01-01")
                )
                end_ts = (
                    pd.Timestamp(args.end_date)
                    if args.end_date
                    else pd.Timestamp("2100-01-01")
                )
                feat_df = store.read_range(spec, start=start_ts, end=end_ts)
                if not feat_df.empty:
                    if "symbol" not in feat_df.columns:
                        feat_df["symbol"] = sym
                    parts.append(feat_df)

            if parts:
                feats_df = pd.concat(parts, axis=0, ignore_index=False)
                if (
                    "timestamp" not in feats_df.columns
                    and getattr(feats_df.index, "name", None) == "timestamp"
                ):
                    feats_df = feats_df.reset_index()

                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                if "timestamp" in feats_df.columns:
                    feats_df["timestamp"] = pd.to_datetime(
                        feats_df["timestamp"], errors="coerce"
                    )

                df = df.merge(
                    feats_df,
                    on=["symbol", "timestamp"],
                    how="left",
                    suffixes=("", "_feat"),
                )
                print(
                    f"✅ 从FeatureStore读取了 {len([f for f in missing_features if f in feats_df.columns])} 个缺失特征"
                )
            else:
                print(f"⚠️  FeatureStore中没有找到数据")

    # 再次检查必需特征（严格要求，一个都不能少）
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"\n❌ 错误: 仍然缺少必需的订单流特征: {missing_features}")
        print(f"订单流特征一个都不能少！缺少这些特征无法进行有效的evidences分析。")
        print(f"\n解决方案:")
        print(f"1. 重新生成FeatureStore，确保包含所有订单流特征")
        print(f"2. 检查FeatureStore配置，确认包含vpin等特征的计算")
        print(f"3. 确保tick数据可用（vpin计算需要tick数据）")
        return 1

    print(f"\n✅ 所有必需的订单流特征都已就绪")

    # 读取FR配置
    with open(args.execution_archetypes, "r") as f:
        config = yaml.safe_load(f)

    fr_config = None
    for regime_name, regime_data in config.get("regimes", {}).items():
        for arch_name, arch_data in regime_data.get("archetypes", {}).items():
            if arch_name == "FailureReversionFR":
                fr_config = arch_data
                break
        if fr_config:
            break

    if not fr_config:
        print("❌ 未找到FR配置")
        return 1

    fr_evidence_rules = fr_config.get("evidence_rules", [])
    fr_required_evidence = fr_config.get("required_evidence", [])

    print(f"\nFR required_evidence: {fr_required_evidence}")

    # 计算quantiles
    print("\n计算quantiles...")
    quantiles = compute_quantiles_from_data(df)

    # 分析1: 不同regime下FR的表现
    print("\n" + "=" * 80)
    print("分析1: 不同Regime下FR Evidences的表现")
    print("=" * 80)

    regime_results = analyze_fr_by_regime(
        df, fr_evidence_rules, fr_required_evidence, quantiles
    )

    print("\n| Regime | 总样本数 | 通过evidences | 平均ret_mean | 胜率 | Sharpe |")
    print("|--------|----------|---------------|--------------|------|--------|")
    for regime, result in sorted(regime_results.items()):
        print(
            f"| {regime} | {result['total_samples']} | {result['evidence_passed']} | {result['mean_ret']:.6f} | {result['win_rate']:.1%} | {result['sharpe']:.3f} |"
        )

    # 分析2: Evidence参数优化
    print("\n" + "=" * 80)
    print("分析2: Evidence参数优化（has_orderflow quantile阈值）")
    print("=" * 80)

    param_results = analyze_evidence_parameters(
        df, fr_evidence_rules, fr_required_evidence, quantiles
    )

    if param_results:
        print(
            "\n| has_orderflow quantile | 通过evidences | 平均ret_mean | 胜率 | Sharpe |"
        )
        print("|----------------------|---------------|--------------|------|--------|")
        for result in sorted(param_results, key=lambda x: x["sharpe"], reverse=True):
            print(
                f"| {result['has_orderflow_quantile']:.2f} | {result['evidence_passed']} | {result['mean_ret']:.6f} | {result['win_rate']:.1%} | {result['sharpe']:.3f} |"
            )

    # 分析3: 找出适合FR的regime特征范围
    print("\n" + "=" * 80)
    print("分析3: 适合FR的Regime特征范围")
    print("=" * 80)

    # 找出所有通过FR evidences的样本
    all_evidence_mask = apply_evidence_filter(
        df, fr_evidence_rules, fr_required_evidence, quantiles
    )
    fr_evidence_passed = df[all_evidence_mask]

    if len(fr_evidence_passed) > 0:
        feature_ranges = find_optimal_regime_features(df, fr_evidence_passed)

        print("\n通过FR evidences的样本的物理特征分布:")
        for feat_key, stats in feature_ranges.items():
            if feat_key.endswith("_all"):
                feat_name = feat_key.replace("_all", "")
                print(f"\n{feat_name} (所有样本):")
                print(f"  均值: {stats['mean']:.3f}, 中位数: {stats['median']:.3f}")
                print(f"  范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  25%-75%分位: [{stats['p25']:.3f}, {stats['p75']:.3f}]")

    # 分析4: 扩大数据范围（如果可能）
    print("\n" + "=" * 80)
    print("分析4: 数据范围扩展分析")
    print("=" * 80)

    if args.start_date and args.end_date:
        print(f"当前数据范围: {args.start_date} 到 {args.end_date}")
        print(f"当前样本数: {len(df)}")
        print(f"通过FR evidences的样本数: {len(fr_evidence_passed)}")
        print(f"\n建议: 可以尝试扩大数据范围以寻找更多适合FR的样本")

    # 保存结果
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "regime_results": regime_results,
        "parameter_results": param_results,
        "feature_ranges": feature_ranges if len(fr_evidence_passed) > 0 else {},
        "total_samples": len(df),
        "fr_evidence_passed": len(fr_evidence_passed),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ 分析结果已保存到: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
