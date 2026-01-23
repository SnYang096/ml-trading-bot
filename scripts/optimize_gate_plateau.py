#!/usr/bin/env python3
"""
Gate 平台高原阈值搜索

使用 Robustness Score (min Sharpe) 作为优化目标，找到"平坦高原"区间。

优化方法：
1. 对每个 Gate rule 单独扫描阈值
2. 计算分桶后的 Conditional min Sharpe
3. 找到"Sharpe ≥ S_min 的最大阈值区间"（平台高原）
4. 选择最宽高原的中位数作为最终阈值

约束条件：
- trade_rate(θ) ≥ R_min
- coverage_per_bucket ≥ N_min
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


@dataclass
class BucketConfig:
    """分桶配置"""

    archetype_buckets: List[str] = None  # ["TC", "TE", "FR", "ET"]
    vol_buckets: List[Tuple[str, float, float]] = (
        None  # [("low", 0.0, 0.33), ("mid", 0.33, 0.67), ("high", 0.67, 1.0)]
    )

    def __post_init__(self):
        if self.archetype_buckets is None:
            self.archetype_buckets = ["TC", "TE", "FR", "ET"]
        if self.vol_buckets is None:
            self.vol_buckets = [
                ("low", 0.0, 0.33),
                ("mid", 0.33, 0.67),
                ("high", 0.67, 1.0),
            ]


@dataclass
class OptimizationConfig:
    """优化配置"""

    min_trade_rate: float = 0.005  # 0.5%
    min_trades_per_bucket: int = 10
    min_sharpe_threshold: float = 0.5  # 平台高原的最低Sharpe要求
    threshold_step: float = 0.05
    threshold_range: Tuple[float, float] = (0.2, 0.8)


def _compute_sharpe(returns: pd.Series, annualize: bool = True) -> float:
    """计算Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0.0
    sharpe = mean_return / std_return
    if annualize:
        # 假设240个bar/年（4小时K线）
        sharpe *= np.sqrt(240)
    return sharpe


def _assign_vol_bucket(
    atr_percentile: float, vol_buckets: List[Tuple[str, float, float]]
) -> str:
    """根据atr_percentile分配vol bucket"""
    if pd.isna(atr_percentile):
        return "unknown"
    for name, low, high in vol_buckets:
        if low <= atr_percentile < high:
            return name
    return "unknown"


def _normalize_archetype(archetype: str) -> str:
    """标准化archetype名称到简短形式"""
    arch_upper = str(archetype).upper()
    if "TRENDCONTINUATION" in arch_upper or arch_upper == "TC":
        return "TC"
    elif "TRENDEXPANSION" in arch_upper or arch_upper == "TE":
        return "TE"
    elif "FAILUREREVERSION" in arch_upper or arch_upper == "FR":
        return "FR"
    elif "EXHAUSTIONTURN" in arch_upper or arch_upper == "ET":
        return "ET"
    return "UNKNOWN"


def _compute_robustness_score(
    df: pd.DataFrame,
    bucket_config: BucketConfig,
    *,
    return_col: str = "ret_mean",
    archetype_col: str = "gate_archetype",
    atr_percentile_col: str = "atr_percentile",
    gate_ok_col: str = "gate_ok",
    min_trades_per_bucket: int = 10,
) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    计算Robustness Score (min Sharpe across buckets)

    Returns:
        (robustness_score, bucket_sharpes, bucket_counts)
    """
    bucket_sharpes: Dict[str, Dict[str, float]] = defaultdict(dict)
    bucket_counts: Dict[str, int] = {}

    # 过滤有效交易（如果有gate_ok列）
    if gate_ok_col in df.columns:
        valid = df[df[gate_ok_col].astype(bool)].copy()
    else:
        # 如果没有gate_ok列，假设所有行都是有效交易
        valid = df.copy()

    if len(valid) == 0:
        return 0.0, {}, {}

    # 分配buckets
    if archetype_col in valid.columns:
        valid["archetype_normalized"] = valid[archetype_col].apply(_normalize_archetype)
    else:
        valid["archetype_normalized"] = "UNKNOWN"

    if atr_percentile_col in valid.columns:
        valid["vol_bucket"] = valid[atr_percentile_col].apply(
            lambda x: _assign_vol_bucket(x, bucket_config.vol_buckets)
        )
    else:
        valid["vol_bucket"] = "unknown"

    min_sharpe = float("inf")
    worst_bucket = None

    # 计算每个bucket的Sharpe (Archetype × Vol)
    for arch in bucket_config.archetype_buckets:
        for vol_name, _, _ in bucket_config.vol_buckets:
            # 过滤archetype（使用标准化后的名称）
            if "archetype_normalized" in valid.columns:
                arch_mask = valid["archetype_normalized"] == arch
            elif archetype_col in valid.columns:
                arch_mask = valid[archetype_col].str.contains(
                    arch, case=False, na=False
                )
            else:
                arch_mask = pd.Series(False, index=valid.index)

            bucket_data = valid[arch_mask & (valid["vol_bucket"] == vol_name)]

            if len(bucket_data) == 0:
                continue

            if return_col not in bucket_data.columns:
                continue

            returns = pd.to_numeric(bucket_data[return_col], errors="coerce").dropna()
            if len(returns) < min_trades_per_bucket:
                continue

            sharpe = _compute_sharpe(returns)

            bucket_key = f"{arch}_{vol_name}"
            bucket_sharpes[arch][bucket_key] = sharpe
            bucket_counts[bucket_key] = len(returns)

            if sharpe < min_sharpe:
                min_sharpe = sharpe
                worst_bucket = bucket_key

    if min_sharpe == float("inf"):
        return 0.0, {}, {}

    return min_sharpe, dict(bucket_sharpes), bucket_counts


def _apply_single_rule_veto(
    df: pd.DataFrame,
    feature_key: str,
    rule_kind: str,
    threshold: float,
    archetype_col: str = "gate_archetype",
) -> pd.Series:
    """
    应用单个gate规则的veto逻辑

    Returns:
        gate_ok: Series of bool, True表示通过gate
    """
    if feature_key not in df.columns:
        # 如果特征缺失，默认全部通过（不veto）
        return pd.Series(True, index=df.index)

    feature_values = pd.to_numeric(df[feature_key], errors="coerce")

    # 计算quantile（基于整个数据集）
    feature_quantile = feature_values.rank(pct=True, na_option="keep")

    # 应用规则
    if rule_kind in ("quantile_lt", "quantile_lte"):
        # deny_if: 如果特征quantile < threshold，则拒绝
        # 所以gate_ok = feature_quantile >= threshold
        gate_ok = feature_quantile >= threshold
    elif rule_kind in ("quantile_gt", "quantile_gte"):
        # deny_if: 如果特征quantile > threshold，则拒绝
        # 所以gate_ok = feature_quantile <= threshold
        gate_ok = feature_quantile <= threshold
    else:
        # 未知规则类型，默认通过
        gate_ok = pd.Series(True, index=df.index)

    # 处理NaN（缺失值）
    gate_ok = gate_ok.fillna(True)  # 缺失值默认通过

    return gate_ok


def _scan_threshold(
    df: pd.DataFrame,
    rule_name: str,
    feature_key: str,
    rule_kind: str,  # "quantile_lt", "quantile_gt", "quantile_gte", etc.
    threshold_range: Tuple[float, float],
    threshold_step: float,
    bucket_config: BucketConfig,
    opt_config: OptimizationConfig,
    execution_archetypes_path: str,
    base_gated_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    扫描单个规则的阈值，返回结果DataFrame

    Args:
        base_gated_df: 已经应用了其他gate规则的DataFrame（可选）
    """
    results = []

    # 确定扫描方向
    thresholds = np.arange(
        threshold_range[0], threshold_range[1] + threshold_step, threshold_step
    )

    for threshold in thresholds:
        # 应用当前规则的veto
        rule_veto_ok = _apply_single_rule_veto(
            df,
            feature_key,
            rule_kind,
            threshold,
        )

        # 如果提供了base_gated_df，需要结合已有的gate结果
        if base_gated_df is not None:
            # 只考虑已经通过其他gate的样本
            base_gate_ok = base_gated_df["gate_ok"].astype(bool)
            gate_ok = base_gate_ok & rule_veto_ok
        else:
            gate_ok = rule_veto_ok

        # 计算trade rate
        trade_rate = gate_ok.sum() / len(df) if len(df) > 0 else 0.0

        if trade_rate < opt_config.min_trade_rate:
            continue

        # 应用gate并计算robustness
        df_gated = df[gate_ok].copy()
        if len(df_gated) == 0:
            continue

        # 需要确保有gate_archetype列
        if "gate_archetype" not in df_gated.columns:
            # 从base_gated_df获取
            if base_gated_df is not None and "gate_archetype" in base_gated_df.columns:
                df_gated = df_gated.merge(
                    base_gated_df[["gate_archetype"]],
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            else:
                # 无法计算，跳过
                continue

        # 添加gate_ok列（用于robustness计算）
        df_gated["gate_ok"] = True

        robustness, bucket_sharpes, bucket_counts = _compute_robustness_score(
            df_gated,
            bucket_config,
            min_trades_per_bucket=opt_config.min_trades_per_bucket,
        )

        # 检查coverage
        all_bucket_counts = []
        for arch_sharpes in bucket_sharpes.values():
            for bucket_key, sharpe in arch_sharpes.items():
                # 从bucket_counts获取count
                if bucket_key in bucket_counts:
                    all_bucket_counts.append(bucket_counts[bucket_key])

        # 简化coverage检查：使用总交易数
        min_coverage = len(df_gated)

        if min_coverage < opt_config.min_trades_per_bucket:
            continue

        # 找到worst bucket
        worst_bucket = None
        worst_sharpe = float("inf")
        for arch_sharpes in bucket_sharpes.values():
            for bucket_key, sharpe in arch_sharpes.items():
                if sharpe < worst_sharpe:
                    worst_sharpe = sharpe
                    worst_bucket = bucket_key

        results.append(
            {
                "rule_name": rule_name,
                "feature_key": feature_key,
                "rule_kind": rule_kind,
                "threshold": threshold,
                "robustness_score": robustness,
                "trade_rate": trade_rate,
                "min_coverage": min_coverage,
                "worst_bucket": worst_bucket,
            }
        )

    return pd.DataFrame(results)


def _find_plateau(
    results: pd.DataFrame,
    min_sharpe_threshold: float,
) -> Optional[Tuple[float, float, float]]:
    """
    找到平台高原区间

    Returns:
        (plateau_start, plateau_end, plateau_median) or None
    """
    if len(results) == 0:
        return None

    # 过滤满足最低Sharpe要求的阈值
    valid = results[results["robustness_score"] >= min_sharpe_threshold].copy()
    if len(valid) == 0:
        return None

    # 按threshold排序
    valid = valid.sort_values("threshold")

    # 找到连续的最大区间
    best_start = None
    best_end = None
    best_width = 0

    i = 0
    while i < len(valid):
        start = valid.iloc[i]["threshold"]
        j = i
        while (
            j < len(valid) and valid.iloc[j]["robustness_score"] >= min_sharpe_threshold
        ):
            j += 1
        end = valid.iloc[j - 1]["threshold"] if j > i else start
        width = end - start

        if width > best_width:
            best_width = width
            best_start = start
            best_end = end

        i = j

    if best_start is None:
        return None

    plateau_median = (best_start + best_end) / 2.0
    return (best_start, best_end, plateau_median)


def main() -> int:
    p = argparse.ArgumentParser(description="Gate 平台高原阈值搜索")
    p.add_argument(
        "--gated-logs",
        required=True,
        help="已应用gate的logs文件（parquet），用于分析当前gate效果",
    )
    p.add_argument(
        "--raw-logs",
        default=None,
        help="原始logs文件（parquet），用于重新应用gate规则（可选）",
    )
    p.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    p.add_argument(
        "--output",
        default="results/gate_optimization.json",
        help="输出JSON文件",
    )
    p.add_argument(
        "--min-trade-rate",
        type=float,
        default=0.005,
        help="最小交易率（默认0.5%%）",
    )
    p.add_argument(
        "--min-trades-per-bucket",
        type=int,
        default=10,
        help="每桶最少交易数",
    )
    p.add_argument(
        "--min-sharpe-threshold",
        type=float,
        default=0.5,
        help="平台高原的最低Sharpe要求",
    )
    p.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="阈值扫描步长",
    )
    args = p.parse_args()

    # 读取数据
    df_gated = pd.read_parquet(args.gated_logs)
    print(f"✅ 读取gated数据: {len(df_gated)} 行")

    # 读取原始数据（如果提供）
    df_raw = None
    if args.raw_logs:
        df_raw = pd.read_parquet(args.raw_logs)
        print(f"✅ 读取原始数据: {len(df_raw)} 行")
    else:
        # 使用gated数据作为基础（简化分析）
        df_raw = df_gated.copy()
        print("⚠️  未提供原始数据，使用gated数据作为基础（分析可能不准确）")

    # 配置
    bucket_config = BucketConfig()
    opt_config = OptimizationConfig(
        min_trade_rate=args.min_trade_rate,
        min_trades_per_bucket=args.min_trades_per_bucket,
        min_sharpe_threshold=args.min_sharpe_threshold,
        threshold_step=args.threshold_step,
    )

    # 加载archetypes配置，提取需要优化的规则
    arches = load_execution_archetypes_registry(args.execution_archetypes)
    arches = {
        k: v for k, v in arches.items() if k != "VolMeanCompressionExpansionReversion"
    }

    # 提取需要优化的规则（价格轨迹特征相关）
    rules_to_optimize = []
    for arch_name, arch in arches.items():
        gate_rules = arch.gate_rules.get("rules", [])
        for rule in gate_rules:
            rule_name = rule.get("name", "")
            feature_key = rule.get("key", "")
            rule_kind = rule.get("kind", "")

            # 只优化价格轨迹特征相关的规则
            price_trajectory_features = [
                "path_efficiency_pct",
                "jump_risk_pct",
                "deviation_z_abs_pct",
                "atr_slope_pct",
                "price_dir_consistency_pct",
                "dir_sign_consistency_pct",
            ]

            if feature_key in price_trajectory_features and rule_kind.startswith(
                "quantile_"
            ):
                rules_to_optimize.append(
                    {
                        "archetype": arch_name,
                        "rule_name": rule_name,
                        "feature_key": feature_key,
                        "rule_kind": rule_kind,
                        "current_threshold": rule.get("quantile")
                        or rule.get("threshold"),
                    }
                )

    print(f"\n📋 找到 {len(rules_to_optimize)} 个需要优化的规则")

    # 优化每个规则
    optimization_results = {}

    for rule_info in rules_to_optimize:
        arch_name = rule_info["archetype"]
        rule_name = rule_info["rule_name"]
        feature_key = rule_info["feature_key"]
        rule_kind = rule_info["rule_kind"]

        print(f"\n🔍 优化: {arch_name} / {rule_name} ({feature_key}, {rule_kind})")

        # 扫描阈值
        results = _scan_threshold(
            df_raw,
            rule_name,
            feature_key,
            rule_kind,
            opt_config.threshold_range,
            opt_config.threshold_step,
            bucket_config,
            opt_config,
            args.execution_archetypes,
            base_gated_df=df_gated if args.raw_logs else None,
        )

        if len(results) == 0:
            print(f"  ⚠️  没有找到满足约束的阈值")
            continue

        # 找到平台高原
        plateau = _find_plateau(results, opt_config.min_sharpe_threshold)

        if plateau:
            plateau_start, plateau_end, plateau_median = plateau
            print(
                f"  ✅ 平台高原: [{plateau_start:.3f}, {plateau_end:.3f}], 中位数: {plateau_median:.3f}"
            )

            # 找到高原区间的结果
            plateau_results = results[
                (results["threshold"] >= plateau_start)
                & (results["threshold"] <= plateau_end)
            ]
            best_result = plateau_results.loc[
                plateau_results["robustness_score"].idxmax()
            ]

            optimization_results[f"{arch_name}_{rule_name}"] = {
                "archetype": arch_name,
                "rule_name": rule_name,
                "feature_key": feature_key,
                "rule_kind": rule_kind,
                "current_threshold": rule_info["current_threshold"],
                "plateau_start": float(plateau_start),
                "plateau_end": float(plateau_end),
                "recommended_threshold": float(plateau_median),
                "robustness_score": float(best_result["robustness_score"]),
                "trade_rate": float(best_result["trade_rate"]),
                "min_coverage": int(best_result["min_coverage"]),
            }
        else:
            print(
                f"  ⚠️  没有找到平台高原（Sharpe < {opt_config.min_sharpe_threshold}）"
            )

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(optimization_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 优化结果已保存: {output_path}")
    print(f"   共优化 {len(optimization_results)} 个规则")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
