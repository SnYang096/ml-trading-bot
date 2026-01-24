#!/usr/bin/env python3
"""
Gate平坦高原优化 - 自适应版本

根据数据量自动选择优化策略：
- 数据少时：使用特征分布优化
- 数据多时：使用平坦高原优化
- 支持多种KPI（Sharpe, Win Rate, Trade Rate组合）

使用方法:
    python scripts/optimize_gate_plateau_adaptive.py \
        --gated-logs results/pipeline_with_reflexivity_2024_full/logs_execution_gated.parquet \
        --raw-logs results/pipeline_with_reflexivity_2024_full/logs_execution.parquet \
        --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
        --output results/gate_optimization_adaptive.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_gate_plateau import (
    _compute_robustness_score,
    _apply_single_rule_veto,
    _find_plateau,
    BucketConfig,
    OptimizationConfig,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


def compute_feature_distribution_score(
    allowed_df: pd.DataFrame,
    vetoed_df: pd.DataFrame,
    feature_key: str,
    archetype: str,
) -> float:
    """
    计算特征分布优化分数

    比较允许和阻止交易的特征分布，分数越高表示阈值设置越合理

    Returns:
        分数 (0-1之间，越高越好)
    """
    if feature_key not in allowed_df.columns or feature_key not in vetoed_df.columns:
        return 0.0

    allowed_vals = allowed_df[feature_key].dropna()
    vetoed_vals = vetoed_df[feature_key].dropna()

    if len(allowed_vals) == 0 or len(vetoed_vals) == 0:
        return 0.0

    # 计算允许和阻止交易的特征均值差异
    allowed_mean = allowed_vals.mean()
    vetoed_mean = vetoed_vals.mean()

    # 如果特征值越大越好（如path_efficiency），允许的应该 > 阻止的
    # 如果特征值越小越好（如jump_risk），允许的应该 < 阻止的
    # 这里使用绝对差异作为分数
    diff = abs(allowed_mean - vetoed_mean)

    # 归一化到0-1
    feature_range = max(allowed_vals.max(), vetoed_vals.max()) - min(
        allowed_vals.min(), vetoed_vals.min()
    )
    if feature_range > 0:
        score = min(diff / feature_range, 1.0)
    else:
        score = 0.0

    return score


def compute_trade_rate_win_rate_score(
    df: pd.DataFrame,
    gate_ok: pd.Series,
    return_col: str = "ret_mean",
) -> float:
    """
    计算Trade Rate + Win Rate组合分数

    Returns:
        组合分数
    """
    if len(df) == 0:
        return 0.0

    trade_rate = gate_ok.sum() / len(df)

    if return_col not in df.columns:
        return trade_rate

    traded_df = df[gate_ok]
    if len(traded_df) == 0:
        return 0.0

    returns = pd.to_numeric(traded_df[return_col], errors="coerce").dropna()
    if len(returns) == 0:
        return trade_rate

    win_rate = (returns > 0).sum() / len(returns)

    # 组合分数：trade_rate * win_rate
    score = trade_rate * win_rate

    return score


def optimize_rule_with_feature_distribution(
    df: pd.DataFrame,
    rule_name: str,
    feature_key: str,
    rule_kind: str,
    threshold_range: Tuple[float, float],
    threshold_step: float,
    archetype: str,
) -> pd.DataFrame:
    """
    使用特征分布优化规则阈值

    不依赖Sharpe，只依赖特征统计
    """
    results = []

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

        allowed_df = df[rule_veto_ok]
        vetoed_df = df[~rule_veto_ok]

        if len(allowed_df) == 0:
            continue

        # 计算trade_rate
        trade_rate = rule_veto_ok.sum() / len(df)

        # 计算特征分布分数
        feature_score = compute_feature_distribution_score(
            allowed_df, vetoed_df, feature_key, archetype
        )

        # 计算trade_rate + win_rate分数
        tr_wr_score = compute_trade_rate_win_rate_score(df, rule_veto_ok)

        # 组合分数：特征分布 + trade_rate + win_rate
        combined_score = 0.4 * feature_score + 0.3 * trade_rate + 0.3 * tr_wr_score

        results.append(
            {
                "rule_name": rule_name,
                "feature_key": feature_key,
                "rule_kind": rule_kind,
                "threshold": threshold,
                "robustness_score": combined_score,  # 使用combined_score作为robustness
                "trade_rate": trade_rate,
                "feature_score": feature_score,
                "tr_wr_score": tr_wr_score,
            }
        )

    return pd.DataFrame(results)


def optimize_rule_adaptive(
    df: pd.DataFrame,
    rule_name: str,
    feature_key: str,
    rule_kind: str,
    threshold_range: Tuple[float, float],
    threshold_step: float,
    bucket_config: BucketConfig,
    opt_config: OptimizationConfig,
    execution_archetypes_path: str,
    base_gated_df: Optional[pd.DataFrame] = None,
    archetype: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    自适应优化规则阈值

    根据数据量选择优化策略
    """
    # 检测数据量
    total_trades = len(df)

    # 如果数据量少，使用特征分布优化
    if total_trades < 100:
        return optimize_rule_with_feature_distribution(
            df,
            rule_name,
            feature_key,
            rule_kind,
            threshold_range,
            threshold_step,
            archetype,
        )

    # 数据量多时，使用平坦高原优化（从optimize_gate_plateau导入）
    # 这里简化处理，直接调用_scan_threshold的等价逻辑
    from scripts.optimize_gate_plateau import _scan_threshold

    return _scan_threshold(
        df,
        rule_name,
        feature_key,
        rule_kind,
        threshold_range,
        threshold_step,
        bucket_config,
        opt_config,
        execution_archetypes_path,
        base_gated_df=base_gated_df,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate平坦高原优化 - 自适应版本",
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
        default=None,
        help="原始logs文件（parquet），用于重新应用gate规则（可选）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )
    parser.add_argument(
        "--output",
        default="results/gate_optimization_adaptive.json",
        help="输出JSON文件",
    )
    parser.add_argument(
        "--min-trade-rate",
        type=float,
        default=0.005,
        help="最小交易率（默认0.5%%）",
    )
    parser.add_argument(
        "--min-trades-per-bucket",
        type=int,
        default=10,
        help="每桶最少交易数",
    )
    parser.add_argument(
        "--min-sharpe-threshold",
        type=float,
        default=0.3,
        help="平台高原的最低Sharpe要求（数据少时会自动降低）",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="阈值扫描步长",
    )

    args = parser.parse_args()

    # 读取数据
    gated_df = pd.read_parquet(args.gated_logs)
    print(f"✅ 读取gated数据: {len(gated_df)} 行")

    raw_df = None
    if args.raw_logs:
        raw_df = pd.read_parquet(args.raw_logs)
        print(f"✅ 读取原始数据: {len(raw_df)} 行")
    else:
        raw_df = gated_df.copy()
        print("⚠️  未提供原始数据，使用gated数据作为基础")

    # 检测数据量
    total_trades = len(raw_df)
    allowed_trades = gated_df[
        gated_df.get("gate_ok", pd.Series([False] * len(gated_df))) == True
    ]
    print(f"\n📊 数据量分析:")
    print(f"  总决策数: {total_trades}")
    print(f"  当前允许交易数: {len(allowed_trades)}")

    if total_trades < 100:
        print(f"  ⚠️  数据量较少，将使用特征分布优化策略")
    else:
        print(f"  ✅ 数据量充足，将使用平坦高原优化策略")

    # 加载archetypes
    arches = load_execution_archetypes_registry(args.execution_archetypes)
    bucket_config = BucketConfig()
    opt_config = OptimizationConfig(
        min_trade_rate=args.min_trade_rate,
        min_trades_per_bucket=args.min_trades_per_bucket,
        min_sharpe_threshold=args.min_sharpe_threshold,
    )

    # 优化每个规则
    optimization_results = {}

    for arch_name, arch in arches.items():
        if not arch.gate_rules:
            continue

        arch_results = []
        rules = (arch.gate_rules or {}).get("rules") or []

        # 优化所有规则类型
        optimizable_rules = [
            r
            for r in rules
            if r.get("kind", "").startswith("quantile_")
            or r.get("kind", "").startswith("value_")
        ]

        if not optimizable_rules:
            continue

        print(f"\n🔍 优化Archetype: {arch_name}")

        for rule in optimizable_rules:
            rule_name = rule.get("name", "unknown")
            feature_key = rule.get("key")
            rule_kind = rule.get("kind")
            current_threshold = rule.get("quantile") or rule.get("value", 0.5)

            if not feature_key or not rule_kind:
                continue

            print(f"  规则: {rule_name} ({feature_key}, {rule_kind})")

            # 确定阈值范围
            if rule_kind.startswith("quantile_"):
                threshold_range = (0.0, 1.0)
            else:
                # value_* 类型：根据特征分布确定范围
                feature_df = raw_df if feature_key in raw_df.columns else gated_df
                if feature_key in feature_df.columns:
                    feature_vals = feature_df[feature_key].dropna()
                    if len(feature_vals) > 0:
                        p5 = float(feature_vals.quantile(0.05))
                        p95 = float(feature_vals.quantile(0.95))
                        current_val = current_threshold
                        if current_val < p5:
                            p5 = current_val * 0.8
                        if current_val > p95:
                            p95 = current_val * 1.2
                        threshold_range = (p5, p95)
                    else:
                        print(f"    特征 {feature_key} 没有有效值，跳过")
                        continue
                else:
                    print(f"    特征 {feature_key} 不存在，跳过")
                    continue

            # 自适应优化
            try:
                results = optimize_rule_adaptive(
                    raw_df,
                    rule_name,
                    feature_key,
                    rule_kind,
                    threshold_range,
                    args.threshold_step,
                    bucket_config,
                    opt_config,
                    args.execution_archetypes,
                    base_gated_df=gated_df if args.raw_logs else None,
                    archetype=arch_name,
                )

                if len(results) == 0:
                    print(f"    未找到有效阈值")
                    continue

                # 找到平台高原（或最佳阈值）
                # 对于特征分布优化，使用combined_score作为robustness
                min_threshold = args.min_sharpe_threshold
                if total_trades < 100:
                    # 数据少时，降低阈值要求
                    min_threshold = min_threshold * 0.3

                plateau = _find_plateau(
                    results, min_threshold, use_trade_rate_fallback=True
                )

                if plateau:
                    plateau_start, plateau_end, plateau_median = plateau
                    print(
                        f"    平坦高原: [{plateau_start:.4f}, {plateau_end:.4f}], 推荐: {plateau_median:.4f}"
                    )
                else:
                    # 使用最佳阈值
                    best_idx = results["robustness_score"].idxmax()
                    plateau_median = results.loc[best_idx, "threshold"]
                    plateau_start = plateau_median
                    plateau_end = plateau_median
                    print(f"    未找到平坦高原，使用最佳阈值: {plateau_median:.4f}")

                # 保存结果
                best_idx = results["robustness_score"].idxmax()
                best_result = results.loc[best_idx].to_dict()

                arch_results.append(
                    {
                        "rule_name": rule_name,
                        "feature_key": feature_key,
                        "rule_kind": rule_kind,
                        "current_threshold": current_threshold,
                        "recommended_threshold": plateau_median,
                        "plateau_start": plateau_start,
                        "plateau_end": plateau_end,
                        "robustness_score": float(
                            best_result.get("robustness_score", 0.0)
                        ),
                        "trade_rate": float(best_result.get("trade_rate", 0.0)),
                    }
                )

            except Exception as e:
                print(f"    优化失败: {e}")
                continue

        if arch_results:
            optimization_results[arch_name] = arch_results

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(optimization_results, f, indent=2, default=str)

    print(f"\n✅ 优化结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
