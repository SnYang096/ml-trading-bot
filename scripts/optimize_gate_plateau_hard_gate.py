#!/usr/bin/env python3
"""
Gate平坦高原优化 - Hard-Gate System版本

实现Hard-Gate System规则调优协议：
1. 规则按照语义优先级排序（安全性 -> 市场状态 -> 执行策略）
2. 规则按顺序逐一优化，不允许联合优化
3. 每个规则一旦优化完成，其参数就被冻结
4. 后续规则的调优基于前序规则生成的过滤数据集进行
5. Plateau评估必须考虑所有上游固定的规则条件

使用方法:
    python scripts/optimize_gate_plateau_hard_gate.py \
        --gated-logs results/pipeline_with_reflexivity_2024_full/logs_execution_gated.parquet \
        --raw-logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
        --output results/gate_optimization_hard_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_gate_plateau import (
    _scan_threshold,
    _find_plateau,
    _apply_single_rule_veto,
    _compute_robustness_score,
    BucketConfig,
    OptimizationConfig,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from scripts.apply_archetype_gate import _read_feature_store_range


def extract_required_features(
    execution_archetypes_path: str,
) -> List[str]:
    """
    从execution_archetypes.yaml提取所有gate规则使用的特征

    Returns:
        特征列表（去重并排序）
    """
    arches = load_execution_archetypes_registry(execution_archetypes_path)
    features = set()

    for arch in arches.values():
        if not arch.gate_rules:
            continue
        rules = arch.gate_rules.get("rules", [])
        for rule in rules:
            feature_key = rule.get("key")
            if feature_key:
                features.add(feature_key)

    return sorted(list(features))


def load_features_from_featurestore(
    logs_df: pd.DataFrame,
    feature_store_root: str,
    feature_store_layer: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    从FeatureStore加载特征并merge到logs DataFrame

    复用apply_archetype_gate.py的逻辑，保持一致性
    """
    symbols = sorted(logs_df["symbol"].astype(str).unique().tolist())

    # 从FeatureStore读取特征
    feats = _read_feature_store_range(
        features_store_root=feature_store_root,
        layer=feature_store_layer,
        symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
    )

    if feats.empty:
        raise ValueError(
            f"FeatureStore读取失败: layer={feature_store_layer}, "
            f"symbols={symbols}, timeframe={timeframe}"
        )

    # 处理timestamp列（复用apply_archetype_gate.py的逻辑）
    feats = feats.copy()
    if getattr(feats.index, "name", None) == "timestamp":
        if "timestamp" in feats.columns:
            feats = feats.reset_index(drop=True)
        else:
            feats = feats.reset_index()

    if "timestamp" not in feats.columns:
        if getattr(feats.index, "name", None) == "timestamp":
            feats = feats.reset_index()

    feats["symbol"] = feats["symbol"].astype(str)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], errors="coerce")
    logs_df = logs_df.copy()
    logs_df["symbol"] = logs_df["symbol"].astype(str)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")

    # Merge特征
    merged = logs_df.merge(
        feats, on=["symbol", "timestamp"], how="left", suffixes=("", "_feat")
    )

    # 处理重复列（优先使用logs_df中的列）
    feat_suffix_cols = [c for c in merged.columns if c.endswith("_feat")]
    cols_to_drop = []
    cols_to_rename = {}
    for feat_col in feat_suffix_cols:
        original_col = feat_col[:-5]  # Remove "_feat" suffix
        if original_col in merged.columns:
            cols_to_drop.append(feat_col)
        else:
            cols_to_rename[feat_col] = original_col

    if cols_to_drop:
        merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])
    if cols_to_rename:
        merged = merged.rename(
            columns={k: v for k, v in cols_to_rename.items() if k in merged.columns}
        )

    return merged


def load_rule_priorities(
    execution_archetypes_path: str,
) -> Dict[str, Dict[str, int]]:
    """
    从execution_archetypes.yaml加载规则优先级

    Returns:
        {archetype_name: {rule_name: priority}}
    """
    with open(execution_archetypes_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    priorities = {}
    regimes = config.get("regimes", {})

    for regime_name, regime_config in regimes.items():
        archetypes = regime_config.get("archetypes", {})
        for arch_name, arch_config in archetypes.items():
            gate_rules = arch_config.get("gate_rules", {})
            rules = gate_rules.get("rules", [])

            arch_priorities = {}
            for rule in rules:
                rule_name = rule.get("name", "")
                priority = rule.get("priority", 999)  # 默认优先级最低
                if rule_name:
                    arch_priorities[rule_name] = priority

            if arch_priorities:
                priorities[arch_name] = arch_priorities

    # 处理overlays
    overlays = config.get("overlays", {})
    for arch_name, arch_config in overlays.items():
        gate_rules = arch_config.get("gate_rules", {})
        rules = gate_rules.get("rules", [])

        arch_priorities = {}
        for rule in rules:
            rule_name = rule.get("name", "")
            priority = rule.get("priority", 999)
            if rule_name:
                arch_priorities[rule_name] = priority

        if arch_priorities:
            priorities[arch_name] = arch_priorities

    return priorities


def apply_frozen_rules(
    df: pd.DataFrame,
    frozen_rules: List[Dict[str, Any]],
    archetype: str,
) -> pd.DataFrame:
    """
    应用已冻结的规则，返回过滤后的DataFrame

    Args:
        df: 原始DataFrame
        frozen_rules: 已冻结的规则列表，每个规则包含 {name, feature_key, rule_kind, threshold}
        archetype: archetype名称（用于过滤）

    Returns:
        过滤后的DataFrame（只包含通过所有冻结规则的样本）
    """
    if not frozen_rules:
        return df.copy()

    # 应用所有冻结规则
    mask = pd.Series(True, index=df.index)

    for rule in frozen_rules:
        feature_key = rule.get("feature_key")
        rule_kind = rule.get("rule_kind")
        threshold = rule.get("threshold") or rule.get("quantile")

        if not feature_key or not rule_kind or threshold is None:
            continue

        # 应用单个规则
        rule_veto = _apply_single_rule_veto(
            df,
            feature_key,
            rule_kind,
            threshold,
        )

        # 合并到总mask（必须通过所有规则）
        mask = mask & rule_veto

    return df[mask].copy()


def optimize_rule_hard_gate(
    df: pd.DataFrame,
    rule_info: Dict[str, Any],
    frozen_rules: List[Dict[str, Any]],
    bucket_config: BucketConfig,
    opt_config: OptimizationConfig,
    execution_archetypes_path: str,
) -> Optional[Dict[str, Any]]:
    """
    在Hard-Gate System下优化单个规则

    Args:
        df: 原始DataFrame
        rule_info: 规则信息 {archetype, rule_name, feature_key, rule_kind, current_threshold, priority}
        frozen_rules: 已冻结的规则列表
        bucket_config: 分桶配置
        opt_config: 优化配置
        execution_archetypes_path: archetypes配置文件路径

    Returns:
        优化结果字典或None
    """
    arch_name = rule_info["archetype"]
    rule_name = rule_info["rule_name"]
    feature_key = rule_info["feature_key"]
    rule_kind = rule_info["rule_kind"]
    current_threshold = rule_info.get("current_threshold")

    print(f"\n  🔍 优化规则: {rule_name} (优先级: {rule_info.get('priority', 'N/A')})")

    # 步骤1: 应用所有已冻结的规则，得到过滤后的数据集
    df_filtered = apply_frozen_rules(df, frozen_rules, arch_name)

    if len(df_filtered) == 0:
        print(f"    ⚠️  应用冻结规则后无数据，跳过")
        return None

    print(f"    应用冻结规则后剩余样本: {len(df_filtered)} / {len(df)}")

    # 步骤2: 确定阈值扫描范围
    if rule_kind.startswith("quantile_"):
        threshold_range = (0.0, 1.0)
    else:
        if feature_key in df_filtered.columns:
            feature_vals = df_filtered[feature_key].dropna()
            if len(feature_vals) > 0:
                p5 = float(feature_vals.quantile(0.05))
                p95 = float(feature_vals.quantile(0.95))
                threshold_range = (p5, p95)
            else:
                print(f"    ⚠️  特征 {feature_key} 无有效值，跳过")
                return None
        else:
            print(f"    ⚠️  特征 {feature_key} 不存在，跳过")
            return None

    # 步骤3: 构建base_gated_df（包含冻结规则的结果）
    base_gated_df = None
    if frozen_rules:
        base_mask = pd.Series(True, index=df.index)
        for rule in frozen_rules:
            feature_key_frozen = rule.get("feature_key")
            rule_kind_frozen = rule.get("rule_kind")
            threshold_frozen = rule.get("threshold") or rule.get("quantile")

            if feature_key_frozen and rule_kind_frozen and threshold_frozen is not None:
                rule_veto = _apply_single_rule_veto(
                    df,
                    feature_key_frozen,
                    rule_kind_frozen,
                    threshold_frozen,
                )
                base_mask = base_mask & rule_veto

        base_gated_df = df.copy()
        base_gated_df["gate_ok"] = base_mask

    # 步骤4: 扫描阈值（在过滤后的数据基础上）
    try:
        results = _scan_threshold(
            df_filtered,  # 使用过滤后的数据
            rule_name,
            feature_key,
            rule_kind,
            threshold_range,
            opt_config.threshold_step,
            bucket_config,
            opt_config,
            execution_archetypes_path,
            base_gated_df=base_gated_df,  # 传递冻结规则的结果
        )

        if len(results) == 0:
            print(f"    ⚠️  未找到满足约束的阈值")
            return None

        # 步骤5: 找到平台高原
        plateau = _find_plateau(
            results,
            opt_config.min_sharpe_threshold,
            use_trade_rate_fallback=True,
        )

        if plateau:
            plateau_start, plateau_end, plateau_median = plateau
            recommended_threshold = plateau_median
            print(
                f"    ✅ 平台高原: [{plateau_start:.4f}, {plateau_end:.4f}], 推荐阈值: {plateau_median:.4f}"
            )
        else:
            best_idx = results["robustness_score"].idxmax()
            recommended_threshold = results.loc[best_idx, "threshold"]
            print(f"    ⚠️  未找到平台高原，使用最佳阈值: {recommended_threshold:.4f}")

        # 获取最佳结果
        best_result = results.loc[results["robustness_score"].idxmax()]

        return {
            "archetype": arch_name,
            "rule_name": rule_name,
            "feature_key": feature_key,
            "rule_kind": rule_kind,
            "current_threshold": current_threshold,
            "recommended_threshold": recommended_threshold,
            "robustness_score": float(best_result["robustness_score"]),
            "trade_rate": float(best_result["trade_rate"]),
            "priority": rule_info.get("priority", 999),
        }

    except Exception as e:
        print(f"    ❌ 优化失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate平坦高原优化 - Hard-Gate System版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gated-logs",
        required=True,
        help="已应用gate的logs文件（parquet）",
    )
    parser.add_argument(
        "--raw-logs",
        required=True,
        help="原始logs文件（parquet）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    parser.add_argument(
        "--output",
        default="results/gate_optimization_hard_gate.json",
        help="输出JSON文件",
    )
    parser.add_argument(
        "--min-trade-rate",
        type=float,
        default=0.001,
        help="最小交易率",
    )
    parser.add_argument(
        "--min-trades-per-bucket",
        type=int,
        default=5,
        help="每桶最少交易数",
    )
    parser.add_argument(
        "--min-sharpe-threshold",
        type=float,
        default=0.1,
        help="平台高原的最低Sharpe要求",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="阈值扫描步长",
    )
    parser.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore根目录",
    )
    parser.add_argument(
        "--feature-store-layer",
        default=None,
        help="FeatureStore layer名称（如果提供，将从FeatureStore加载特征）",
    )
    parser.add_argument(
        "--timeframe",
        default="240T",
        help="时间框架（如 240T）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="开始日期（可选，用于FeatureStore读取）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期（可选，用于FeatureStore读取）",
    )

    args = parser.parse_args()

    # 读取数据
    df_raw = pd.read_parquet(args.raw_logs)
    print(f"✅ 读取原始数据: {len(df_raw)} 行")

    # 提取gate规则所需的所有特征
    required_features = extract_required_features(args.execution_archetypes)
    print(f"📋 Gate规则需要 {len(required_features)} 个特征")

    # 检查哪些特征缺失
    available_features = set(df_raw.columns)
    missing_features = [f for f in required_features if f not in available_features]

    # 如果缺少特征，尝试从FeatureStore加载
    if missing_features and args.feature_store_layer:
        print(
            f"⚠️  logs文件缺少 {len(missing_features)} 个特征，尝试从FeatureStore加载..."
        )
        print(
            f"   缺失特征: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
        )

        try:
            # 确定时间范围
            if "timestamp" in df_raw.columns:
                timestamps = pd.to_datetime(df_raw["timestamp"], errors="coerce")
                if not timestamps.isna().all():
                    start_ts = timestamps.min()
                    end_ts = timestamps.max()
                    start_date = args.start_date or start_ts.strftime("%Y-%m-%d")
                    end_date = args.end_date or end_ts.strftime("%Y-%m-%d")
                else:
                    start_date = args.start_date
                    end_date = args.end_date
            else:
                start_date = args.start_date
                end_date = args.end_date

            # 从FeatureStore加载特征
            df_raw = load_features_from_featurestore(
                df_raw,
                feature_store_root=args.feature_store_root,
                feature_store_layer=args.feature_store_layer,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            print(
                f"✅ 从FeatureStore加载特征成功，DataFrame现在有 {len(df_raw.columns)} 列"
            )

            # 再次检查缺失的特征
            available_features = set(df_raw.columns)
            still_missing = [
                f for f in required_features if f not in available_features
            ]
            if still_missing:
                print(f"⚠️  仍有 {len(still_missing)} 个特征在FeatureStore中缺失:")
                print(
                    f"   {still_missing[:10]}{'...' if len(still_missing) > 10 else ''}"
                )
                print(f"   建议重新构建FeatureStore以包含这些特征")
        except Exception as e:
            print(f"❌ 从FeatureStore加载特征失败: {e}")
            print(f"   建议检查FeatureStore配置或重新构建FeatureStore")
            if missing_features:
                print(
                    f"   缺失的特征: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
                )
                return 1
    elif missing_features:
        print(f"⚠️  logs文件缺少 {len(missing_features)} 个特征:")
        print(
            f"   {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
        )
        print(f"   请提供 --feature-store-layer 参数以从FeatureStore加载特征")
        print(f"   或重新构建包含这些特征的logs文件")
        return 1

    # 加载archetypes
    arches = load_execution_archetypes_registry(args.execution_archetypes)
    arches = {
        k: v for k, v in arches.items() if k != "VolMeanCompressionExpansionReversion"
    }

    # 加载规则优先级
    rule_priorities = load_rule_priorities(args.execution_archetypes)
    print(f"✅ 加载规则优先级: {len(rule_priorities)} 个archetype")

    # 配置
    bucket_config = BucketConfig()
    opt_config = OptimizationConfig(
        min_trade_rate=args.min_trade_rate,
        min_trades_per_bucket=args.min_trades_per_bucket,
        min_sharpe_threshold=args.min_sharpe_threshold,
        threshold_step=args.threshold_step,
    )

    # 提取需要优化的规则，并按优先级排序
    all_rules_to_optimize = []
    available_features = set(df_raw.columns)

    print(f"📊 数据文件包含 {len(available_features)} 个特征列")

    for arch_name, arch in arches.items():
        if not arch.gate_rules:
            continue

        rules = (arch.gate_rules or {}).get("rules", [])
        arch_priorities = rule_priorities.get(arch_name, {})

        for rule in rules:
            rule_name = rule.get("name", "")
            feature_key = rule.get("key")
            rule_kind = rule.get("kind")

            # 只优化value_*和quantile_*类型的规则
            if not (
                rule_kind.startswith("value_") or rule_kind.startswith("quantile_")
            ):
                continue

            if not rule_name or not feature_key:
                continue

            # 检查特征是否存在于数据中
            if feature_key not in available_features:
                continue  # 静默跳过，不打印警告

            priority = arch_priorities.get(rule_name, 999)
            current_threshold = rule.get("threshold") or rule.get("quantile")

            all_rules_to_optimize.append(
                {
                    "archetype": arch_name,
                    "rule_name": rule_name,
                    "feature_key": feature_key,
                    "rule_kind": rule_kind,
                    "current_threshold": current_threshold,
                    "priority": priority,
                }
            )

    # 按优先级排序（优先级越小越先优化）
    all_rules_to_optimize.sort(
        key=lambda x: (x["priority"], x["archetype"], x["rule_name"])
    )

    print(f"\n📋 找到 {len(all_rules_to_optimize)} 个需要优化的规则")
    print(f"📊 优先级分布:")
    priority_counts = {}
    for rule in all_rules_to_optimize:
        p = rule["priority"]
        priority_counts[p] = priority_counts.get(p, 0) + 1
    for p in sorted(priority_counts.keys()):
        print(f"  优先级 {p}: {priority_counts[p]} 个规则")

    # Hard-Gate System: 按优先级顺序逐一优化
    optimization_results = {}
    frozen_rules_by_arch: Dict[str, List[Dict[str, Any]]] = {}

    for rule_info in all_rules_to_optimize:
        arch_name = rule_info["archetype"]
        frozen_rules = frozen_rules_by_arch.get(arch_name, [])

        # 优化当前规则
        result = optimize_rule_hard_gate(
            df_raw,
            rule_info,
            frozen_rules,
            bucket_config,
            opt_config,
            args.execution_archetypes,
        )

        if result:
            # 保存结果
            key = f"{arch_name}_{rule_info['rule_name']}"
            optimization_results[key] = result

            # 冻结当前规则（添加到frozen_rules）
            frozen_rule = {
                "name": rule_info["rule_name"],
                "feature_key": rule_info["feature_key"],
                "rule_kind": rule_info["rule_kind"],
                "threshold": result["recommended_threshold"],
                "quantile": (
                    result["recommended_threshold"]
                    if rule_info["rule_kind"].startswith("quantile_")
                    else None
                ),
            }
            frozen_rules.append(frozen_rule)
            frozen_rules_by_arch[arch_name] = frozen_rules

            print(
                f"    ✅ 规则已优化并冻结: {rule_info['rule_name']} = {result['recommended_threshold']:.4f}"
            )

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(optimization_results, f, indent=2, default=str)

    print(f"\n✅ Hard-Gate System优化结果已保存: {output_path}")
    print(f"📊 共优化 {len(optimization_results)} 个规则")

    return 0


if __name__ == "__main__":
    sys.exit(main())
