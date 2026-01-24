#!/usr/bin/env python3
"""
Gate平坦高原优化 - 渐进式优化版本

实现三步渐进式优化：
1. 第一步：大幅放宽规则，增加交易数到200+
2. 第二步：在足够数据基础上，使用平坦高原优化
3. 第三步：逐步收紧，找到最优阈值

使用方法:
    python scripts/optimize_gate_plateau_progressive.py \
        --gated-logs results/pipeline_with_reflexivity_2024_full/logs_execution_gated.parquet \
        --raw-logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
        --output results/gate_optimization_progressive.json \
        --target-trades 200 \
        --tighten-step 0.05
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
    _compute_robustness_score,
    compute_pareto_frontier,
    select_multi_objective_threshold,
    BucketConfig,
    OptimizationConfig,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from scripts.apply_archetype_gate import _read_feature_store_range


def _compute_min_trade_per_rule(
    *,
    total_rules: int,
    optimized_count: int,
    global_trade_budget: Optional[float],
) -> Optional[float]:
    """
    Compute per-rule minimum trade rate from a global trade budget.
    Uses multiplicative budget: min_trade_per_rule = budget ** (1 / remaining_rules).
    """
    if global_trade_budget is None or global_trade_budget <= 0:
        return None
    remaining_rules = max(1, total_rules - optimized_count)
    return float(global_trade_budget) ** (1.0 / float(remaining_rules))


def extract_required_features(
    execution_archetypes_path: str,
) -> List[str]:
    """从execution_archetypes.yaml提取所有gate规则使用的特征"""
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
    """从FeatureStore加载特征并merge到logs DataFrame"""
    symbols = sorted(logs_df["symbol"].astype(str).unique().tolist())

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

    # 处理timestamp列
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

    # 处理重复列
    feat_suffix_cols = [c for c in merged.columns if c.endswith("_feat")]
    cols_to_drop = []
    cols_to_rename = {}
    for feat_col in feat_suffix_cols:
        original_col = feat_col[:-5]
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


def relax_rule_threshold(
    rule: Dict[str, Any],
    relax_factor: float = 0.5,
) -> Dict[str, Any]:
    """
    放宽规则阈值

    Args:
        rule: 规则字典
        relax_factor: 放宽因子（0.5表示放宽50%，即threshold * 0.5）

    Returns:
        放宽后的规则
    """
    relaxed_rule = deepcopy(rule)
    rule_kind = rule.get("kind", "")

    # 获取当前阈值（优先使用threshold，其次quantile）
    current_threshold = None
    if "threshold" in rule and rule["threshold"] is not None:
        current_threshold = rule["threshold"]
        threshold_key = "threshold"
    elif "quantile" in rule and rule["quantile"] is not None:
        current_threshold = rule["quantile"]
        threshold_key = "quantile"
    else:
        # 如果没有阈值，返回原规则
        return relaxed_rule

    # 计算放宽后的阈值
    if rule_kind in ("value_lt", "quantile_lt", "value_lte", "quantile_lte"):
        # 对于<规则，降低阈值（更宽松）
        # 例如：threshold=0.6, relax_factor=0.5 -> 0.6 * (1-0.5) = 0.3
        relaxed_threshold = current_threshold * (1 - relax_factor)
    elif rule_kind in ("value_gt", "quantile_gt", "value_gte", "quantile_gte"):
        # 对于>规则，增加阈值（更宽松）
        # 例如：threshold=0.6, relax_factor=0.5 -> 0.6 * (1+0.5) = 0.9
        relaxed_threshold = current_threshold * (1 + relax_factor)
    else:
        relaxed_threshold = current_threshold

    # 更新阈值
    relaxed_rule[threshold_key] = relaxed_threshold

    return relaxed_rule


def apply_relaxed_rules(
    df: pd.DataFrame,
    relaxed_arches: Dict[str, Any],
    quantiles: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    应用放宽后的规则，返回gate_ok结果

    Args:
        df: 原始DataFrame
        relaxed_arches: 已放宽的archetypes配置（gate_rules中的规则已放宽）
        quantiles: 分位数字典（可选）

    Returns:
        DataFrame with gate_ok column
    """
    result_df = df.copy()
    result_df["gate_ok"] = False  # 默认不通过
    result_df["gate_archetype"] = None

    # 对每个样本，尝试所有archetype的gate规则
    for idx, row in result_df.iterrows():
        features = row.to_dict()

        # 尝试每个archetype
        for arch_name, arch in relaxed_arches.items():
            if not arch.gate_rules:
                # 如果没有gate规则，默认通过
                result_df.loc[idx, "gate_ok"] = True
                result_df.loc[idx, "gate_archetype"] = arch_name
                break

            # 获取该样本的分位数（如果有）
            sym_quantiles = None
            if quantiles:
                symbol = str(row.get("symbol", ""))
                if isinstance(quantiles, dict):
                    sym_quantiles = quantiles.get(symbol) or quantiles

            # 应用gate规则
            ok, reasons = apply_gate_rules(
                gate_rules=arch.gate_rules,
                features=features,
                quantiles=sym_quantiles,
            )

            if ok:
                # 如果通过，标记为通过并记录archetype
                result_df.loc[idx, "gate_ok"] = True
                result_df.loc[idx, "gate_archetype"] = arch_name
                break

    return result_df


def step1_relax_rules(
    df: pd.DataFrame,
    arches: Dict[str, Any],
    target_trades: int = 200,
    max_iterations: int = 10,
    quantiles: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    第一步：大幅放宽规则，增加交易数到target_trades

    Args:
        df: 原始DataFrame
        arches: archetypes配置
        target_trades: 目标交易数
        max_iterations: 最大迭代次数
        quantiles: 分位数字典（可选）

    Returns:
        (relaxed_arches, gated_df)
    """
    print(f"\n📊 第一步：大幅放宽规则，目标交易数: {target_trades}")

    relax_factor = 0.5
    relaxed_arches = deepcopy(arches)

    # 需要修改archetypes的gate_rules，但ExecutionArchetype是frozen dataclass
    # 所以我们需要创建一个新的字典结构来存储放宽后的规则
    relaxed_arches_dict = {}
    for arch_name, arch in relaxed_arches.items():
        relaxed_arches_dict[arch_name] = {
            "name": arch_name,
            "regime": arch.regime,
            "gate_rules": deepcopy(arch.gate_rules) if arch.gate_rules else {},
            "required_conditions": arch.required_conditions,
            "required_evidence": arch.required_evidence,
            "evidence_rules": arch.evidence_rules,
            "execution_constraints": arch.execution_constraints,
        }

    for iteration in range(max_iterations):
        # 放宽所有规则
        for arch_name, arch_dict in relaxed_arches_dict.items():
            gate_rules = arch_dict.get("gate_rules", {})
            if not gate_rules:
                continue

            rules = gate_rules.get("rules", [])
            if not rules:
                continue

            # 放宽所有规则
            relaxed_rules = [relax_rule_threshold(r, relax_factor) for r in rules]
            gate_rules["rules"] = relaxed_rules
            arch_dict["gate_rules"] = gate_rules

        # 创建临时的archetypes对象用于apply_gate_rules
        # 由于apply_gate_rules只需要gate_rules字典，我们可以直接传递
        temp_arches = {}
        for arch_name, arch_dict in relaxed_arches_dict.items():
            # 创建一个简单的对象来存储gate_rules
            class TempArch:
                def __init__(self, gate_rules):
                    self.gate_rules = gate_rules

            temp_arches[arch_name] = TempArch(arch_dict["gate_rules"])

        # 应用放宽后的规则
        gated_df = apply_relaxed_rules(df, temp_arches, quantiles=quantiles)
        allowed_count = int(gated_df["gate_ok"].sum())

        print(
            f"  迭代 {iteration + 1}: relax_factor={relax_factor:.2f}, 允许交易数={allowed_count}"
        )

        if allowed_count >= target_trades:
            print(f"  ✅ 达到目标交易数: {allowed_count} >= {target_trades}")
            break

        # 继续放宽
        relax_factor = min(relax_factor + 0.1, 0.9)  # 最多放宽90%

    # 返回放宽后的archetypes配置（以字典形式）和gated DataFrame
    return relaxed_arches_dict, gated_df


def step2_plateau_optimization(
    df: pd.DataFrame,
    relaxed_arches: Dict[str, Any],
    bucket_config: BucketConfig,
    opt_config: OptimizationConfig,
    execution_archetypes_path: str,
    global_trade_budget: Optional[float] = None,
) -> Dict[str, Any]:
    """
    第二步：在第一步的基础上，使用平坦高原优化

    Args:
        df: 第一步放宽后通过gate的DataFrame
        relaxed_arches: 放宽后的archetypes配置（字典格式）
        bucket_config: 分桶配置
        opt_config: 优化配置
        execution_archetypes_path: archetypes配置文件路径

    Returns:
        优化后的规则阈值字典
    """
    print(f"\n📊 第二步：平坦高原优化")

    optimization_results = {}

    total_rules = 0
    for arch_dict in relaxed_arches.values():
        gate_rules = arch_dict.get("gate_rules", {})
        rules = gate_rules.get("rules", [])
        total_rules += len(
            [r for r in rules if r.get("kind", "").startswith(("quantile_", "value_"))]
        )

    optimized_count = 0
    for arch_name, arch_dict in relaxed_arches.items():
        gate_rules = arch_dict.get("gate_rules", {})
        if not gate_rules:
            continue

        arch_results = []
        rules = gate_rules.get("rules", [])

        # 优化所有规则类型
        optimizable_rules = [
            r
            for r in rules
            if r.get("kind", "").startswith("quantile_")
            or r.get("kind", "").startswith("value_")
        ]

        if not optimizable_rules:
            continue

        print(f"  优化Archetype: {arch_name}")

        for rule in optimizable_rules:
            rule_name = rule.get("name", "unknown")
            feature_key = rule.get("key")
            rule_kind = rule.get("kind")
            current_threshold = rule.get("threshold") or rule.get("quantile", 0.5)

            if not feature_key or not rule_kind:
                continue

            print(f"    规则: {rule_name}")

            # 确定阈值范围
            if rule_kind.startswith("quantile_"):
                threshold_range = (0.0, 1.0)
            else:
                if feature_key in df.columns:
                    feature_vals = df[feature_key].dropna()
                    if len(feature_vals) > 0:
                        p5 = float(feature_vals.quantile(0.05))
                        p95 = float(feature_vals.quantile(0.95))
                        threshold_range = (p5, p95)
                    else:
                        continue
                else:
                    continue

            # 扫描阈值
            try:
                results = _scan_threshold(
                    df,
                    rule_name,
                    feature_key,
                    rule_kind,
                    threshold_range,
                    opt_config.threshold_step,
                    bucket_config,
                    opt_config,
                    execution_archetypes_path,
                    base_gated_df=None,
                )

                if len(results) == 0:
                    continue

                min_trade_per_rule = _compute_min_trade_per_rule(
                    total_rules=total_rules,
                    optimized_count=optimized_count,
                    global_trade_budget=global_trade_budget,
                )
                if min_trade_per_rule is not None:
                    constrained = results[
                        results["trade_rate"] >= min_trade_per_rule
                    ].copy()
                    if len(constrained) == 0:
                        print(
                            f"      ⚠️ guardrail未满足: min_trade_per_rule={min_trade_per_rule:.4f} "
                            f"(使用当前阈值 {current_threshold:.4f})"
                        )
                        arch_results.append(
                            {
                                "rule_name": rule_name,
                                "feature_key": feature_key,
                                "rule_kind": rule_kind,
                                "current_threshold": current_threshold,
                                "recommended_threshold": current_threshold,
                                "guardrail_min_trade_rate": min_trade_per_rule,
                                "guardrail_fallback": True,
                            }
                        )
                        optimized_count += 1
                        continue
                    results = constrained

                # 找到平台高原
                plateau = _find_plateau(
                    results,
                    opt_config.min_sharpe_threshold,
                    use_trade_rate_fallback=True,
                )

                if plateau:
                    plateau_start, plateau_end, plateau_median = plateau

                    # 多目标优化：使用max_robustness策略
                    pareto_front = compute_pareto_frontier(results)
                    if len(pareto_front) > 0:
                        # 使用max_robustness策略选择最优阈值
                        best_multi_threshold = select_multi_objective_threshold(
                            results,
                            pareto_front,
                            strategy="max_robustness",
                            min_robustness=opt_config.min_sharpe_threshold,
                            min_trade_rate=opt_config.min_trade_rate,
                            trade_rate_weight=0.5,
                            robustness_weight=0.5,
                        )

                        if best_multi_threshold is not None:
                            # 如果多目标优化的阈值在平台高原范围内，使用它
                            if plateau_start <= best_multi_threshold <= plateau_end:
                                recommended_threshold = best_multi_threshold
                                print(
                                    f"      平台高原: [{plateau_start:.4f}, {plateau_end:.4f}]"
                                )
                                print(
                                    f"      多目标优化(max_robustness): {best_multi_threshold:.4f}"
                                )
                            else:
                                recommended_threshold = plateau_median
                                print(
                                    f"      平台高原: [{plateau_start:.4f}, {plateau_end:.4f}], 推荐阈值: {plateau_median:.4f}"
                                )
                        else:
                            recommended_threshold = plateau_median
                            print(
                                f"      平台高原: [{plateau_start:.4f}, {plateau_end:.4f}], 推荐阈值: {plateau_median:.4f}"
                            )
                    else:
                        recommended_threshold = plateau_median
                        print(
                            f"      平台高原: [{plateau_start:.4f}, {plateau_end:.4f}], 推荐阈值: {plateau_median:.4f}"
                        )
                else:
                    # 未找到平台高原，使用多目标优化选择
                    pareto_front = compute_pareto_frontier(results)
                    if len(pareto_front) > 0:
                        best_multi_threshold = select_multi_objective_threshold(
                            results,
                            pareto_front,
                            strategy="max_robustness",
                            min_robustness=opt_config.min_sharpe_threshold,
                            min_trade_rate=opt_config.min_trade_rate,
                        )
                        if best_multi_threshold is not None:
                            recommended_threshold = best_multi_threshold
                            print(
                                f"      未找到平台高原，使用多目标优化(max_robustness): {best_multi_threshold:.4f}"
                            )
                        else:
                            best_idx = results["robustness_score"].idxmax()
                            recommended_threshold = results.loc[best_idx, "threshold"]
                            print(
                                f"      未找到平台高原，使用最佳robustness阈值: {recommended_threshold:.4f}"
                            )
                    else:
                        best_idx = results["robustness_score"].idxmax()
                        recommended_threshold = results.loc[best_idx, "threshold"]
                        print(
                            f"      未找到平台高原，使用最佳robustness阈值: {recommended_threshold:.4f}"
                        )

                arch_results.append(
                    {
                        "rule_name": rule_name,
                        "feature_key": feature_key,
                        "rule_kind": rule_kind,
                        "current_threshold": current_threshold,
                        "recommended_threshold": recommended_threshold,
                        "guardrail_min_trade_rate": min_trade_per_rule,
                        "guardrail_fallback": False,
                    }
                )
                optimized_count += 1

            except Exception as e:
                print(f"      优化失败: {e}")
                import traceback

                traceback.print_exc()
                continue

        if arch_results:
            optimization_results[arch_name] = arch_results

    return optimization_results


def step3_tighten_thresholds(
    df: pd.DataFrame,
    optimized_rules: Dict[str, Any],
    relaxed_arches: Dict[str, Any],
    bucket_config: BucketConfig,
    opt_config: OptimizationConfig,
    execution_archetypes_path: str,
    tighten_step: float = 0.05,
    quantiles: Optional[Dict[str, Any]] = None,
    global_trade_budget: Optional[float] = None,
) -> Dict[str, Any]:
    """
    第三步：逐步收紧，找到最优阈值

    Args:
        df: 原始DataFrame（用于计算robustness）
        optimized_rules: 第二步优化后的规则结果
        relaxed_arches: 放宽后的archetypes配置
        bucket_config: 分桶配置
        opt_config: 优化配置
        execution_archetypes_path: archetypes配置文件路径
        tighten_step: 收紧步长
        quantiles: 分位数字典（可选）

    Returns:
        收紧后的规则阈值字典
    """
    print(f"\n📊 第三步：逐步收紧阈值，步长: {tighten_step}")

    from scripts.optimize_gate_plateau import _apply_single_rule_veto

    final_results = {}

    total_rules = sum(len(v) for v in optimized_rules.values())
    optimized_count = 0
    for arch_name, arch_results in optimized_rules.items():
        if not arch_results:
            continue

        print(f"  收紧Archetype: {arch_name}")
        final_arch_results = []

        # 获取该archetype的gate_rules
        arch_dict = relaxed_arches.get(arch_name, {})
        gate_rules = arch_dict.get("gate_rules", {})
        all_rules = gate_rules.get("rules", [])

        for result in arch_results:
            rule_name = result["rule_name"]
            feature_key = result["feature_key"]
            rule_kind = result["rule_kind"]
            current_threshold = result["recommended_threshold"]

            print(f"    规则: {rule_name}, 当前阈值: {current_threshold:.4f}")

            # 计算当前阈值的robustness_score
            current_veto = _apply_single_rule_veto(
                df,
                feature_key,
                rule_kind,
                current_threshold,
            )
            df_current = df[current_veto].copy()
            if len(df_current) > 0:
                current_robustness, _, _ = _compute_robustness_score(
                    df_current,
                    bucket_config,
                    min_trades_per_bucket=opt_config.min_trades_per_bucket,
                )
            else:
                current_robustness = 0.0

            min_trade_per_rule = _compute_min_trade_per_rule(
                total_rules=total_rules,
                optimized_count=optimized_count,
                global_trade_budget=global_trade_budget,
            )
            current_trade_rate = float(len(df_current) / len(df)) if len(df) else 0.0
            if (
                min_trade_per_rule is not None
                and current_trade_rate < min_trade_per_rule
            ):
                print(
                    f"      ⚠️ guardrail未满足: min_trade_per_rule={min_trade_per_rule:.4f} "
                    f"(当前trade_rate={current_trade_rate:.4f})"
                )
                final_arch_results.append(
                    {
                        **result,
                        "final_threshold": current_threshold,
                        "final_robustness": 0.0,
                        "guardrail_min_trade_rate": min_trade_per_rule,
                        "guardrail_fallback": True,
                    }
                )
                optimized_count += 1
                continue

            best_threshold = current_threshold
            best_robustness = current_robustness

            # 确定收紧方向
            if rule_kind in ("value_lt", "quantile_lt", "value_lte", "quantile_lte"):
                # 对于<规则，收紧意味着增加阈值（更严格）
                tighten_direction = 1
            elif rule_kind in ("value_gt", "quantile_gt", "value_gte", "quantile_gte"):
                # 对于>规则，收紧意味着降低阈值（更严格）
                tighten_direction = -1
            else:
                tighten_direction = 0

            if tighten_direction != 0:
                # 逐步收紧，检查robustness_score
                for i in range(5):  # 最多收紧5次
                    test_threshold = (
                        current_threshold + tighten_direction * tighten_step * (i + 1)
                    )

                    # 确保阈值在合理范围内
                    if rule_kind.startswith("quantile_"):
                        test_threshold = max(0.0, min(1.0, test_threshold))
                    else:
                        # 对于value_*规则，使用特征的分位数范围
                        if feature_key in df.columns:
                            feature_vals = df[feature_key].dropna()
                            if len(feature_vals) > 0:
                                p1 = float(feature_vals.quantile(0.01))
                                p99 = float(feature_vals.quantile(0.99))
                                test_threshold = max(p1, min(p99, test_threshold))

                    # 应用测试阈值，计算robustness
                    test_veto = _apply_single_rule_veto(
                        df,
                        feature_key,
                        rule_kind,
                        test_threshold,
                    )
                    df_test = df[test_veto].copy()

                    if len(df_test) == 0:
                        # 如果收紧后没有交易，停止
                        break
                    test_trade_rate = float(len(df_test) / len(df)) if len(df) else 0.0
                    if (
                        min_trade_per_rule is not None
                        and test_trade_rate < min_trade_per_rule
                    ):
                        print(
                            f"      收紧到 {test_threshold:.4f} 触发guardrail "
                            f"(trade_rate={test_trade_rate:.4f} < {min_trade_per_rule:.4f})"
                        )
                        break

                    test_robustness, _, _ = _compute_robustness_score(
                        df_test,
                        bucket_config,
                        min_trades_per_bucket=opt_config.min_trades_per_bucket,
                    )

                    # 如果robustness下降超过5%，停止收紧
                    if test_robustness < best_robustness * 0.95:
                        print(
                            f"      收紧到 {test_threshold:.4f} 时robustness下降，停止收紧"
                        )
                        break

                    # 如果robustness更好或相等，更新最佳阈值
                    if test_robustness >= best_robustness:
                        best_threshold = test_threshold
                        best_robustness = test_robustness
                        print(
                            f"      收紧到 {test_threshold:.4f}, robustness: {test_robustness:.4f}"
                        )
                    else:
                        # robustness略有下降但不超过5%，继续尝试
                        pass

            print(
                f"      最终阈值: {best_threshold:.4f}, robustness: {best_robustness:.4f}"
            )

            final_arch_results.append(
                {
                    **result,
                    "final_threshold": best_threshold,
                    "final_robustness": best_robustness,
                    "guardrail_min_trade_rate": min_trade_per_rule,
                    "guardrail_fallback": False,
                }
            )
            optimized_count += 1

        if final_arch_results:
            final_results[arch_name] = final_arch_results

    return final_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate平坦高原优化 - 渐进式优化版本",
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
        default="results/gate_optimization_progressive.json",
        help="输出JSON文件",
    )
    parser.add_argument(
        "--target-trades",
        type=int,
        default=200,
        help="第一步目标交易数",
    )
    parser.add_argument(
        "--tighten-step",
        type=float,
        default=0.05,
        help="第三步收紧步长",
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
        "--global-trade-budget",
        type=float,
        default=None,
        help="全局trade_rate生存约束（如4H=0.12），用于受控Pareto",
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

    # 配置
    bucket_config = BucketConfig()
    opt_config = OptimizationConfig(
        min_trade_rate=args.min_trade_rate,
        min_trades_per_bucket=args.min_trades_per_bucket,
        min_sharpe_threshold=args.min_sharpe_threshold,
        threshold_step=args.threshold_step,
    )

    # 加载archetypes（如果还没有加载）
    if "arches" not in locals():
        arches = load_execution_archetypes_registry(args.execution_archetypes)
        arches = {
            k: v
            for k, v in arches.items()
            if k != "VolMeanCompressionExpansionReversion"
        }

    # 读取quantiles（如果有）
    quantiles = None
    # TODO: 如果需要quantiles，可以从quantiles文件加载

    # 第一步：大幅放宽规则
    relaxed_arches, gated_df = step1_relax_rules(
        df_raw,
        arches,
        target_trades=args.target_trades,
        quantiles=quantiles,
    )

    # 第二步：平坦高原优化（在放宽后通过gate的数据基础上）
    gated_allowed = gated_df[gated_df["gate_ok"] == True].copy()
    if len(gated_allowed) == 0:
        print("⚠️  警告：放宽后没有通过gate的交易，无法进行第二步优化")
        return 1

    optimized_rules = step2_plateau_optimization(
        gated_allowed,
        relaxed_arches,
        bucket_config,
        opt_config,
        args.execution_archetypes,
        global_trade_budget=args.global_trade_budget,
    )

    # 第三步：逐步收紧
    final_rules = step3_tighten_thresholds(
        df_raw,
        optimized_rules,
        relaxed_arches,
        bucket_config,
        opt_config,
        args.execution_archetypes,
        tighten_step=args.tighten_step,
        quantiles=quantiles,
        global_trade_budget=args.global_trade_budget,
    )

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_rules, f, indent=2, default=str)

    print(f"\n✅ 渐进式优化结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
