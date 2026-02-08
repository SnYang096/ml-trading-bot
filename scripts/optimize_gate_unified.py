#!/usr/bin/env python3
"""
Unified Gate Optimization Script - Production Grade Gate Parameter Optimizer
┌───────────────────────────────────────────────────────────────┐
│                    整体流程                                      │
├───────────────────────────────────────────────────────────────┤
│  1. Threshold Scan    →   扫描所有阈值，计算 Lift               │
│  2. Plateau Detection →   找到「稳定平台」区间                   │
│  3. Robustness Score  →   在平台内选择「最不容易炸」的点         │
└───────────────────────────────────────────────────────────────┘
整合三种优化方法：
1. Lift-based optimization (基于条件选择性)
2. Robustness-based optimization (基于稳定性)
3. Hard-Gate System (按优先级顺序优化)

核心改进：
- plateau 定义从「连续」升级为「稳定」
- fallback 从「最强」升级为「最稳」
- 引入 Robustness Score 作为最终决策指标
- 支持区间门控而非单点门控
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.archetype import (
    GateRule,
    StrategyArchetype,
    load_strategy_archetype,
)


@dataclass
class RobustnessScore:
    """Robustness Score dataclass - 用于评估门控参数的稳定性"""

    param_stability: float  # 参数扰动稳定性 (0-1)
    temporal_stability: float  # 时间稳定性 (0-1)
    sample_efficiency: float  # 样本效率 (0-1)
    overall_score: float  # 综合评分

    def to_dict(self) -> Dict[str, float]:
        return {
            "param_stability": self.param_stability,
            "temporal_stability": self.temporal_stability,
            "sample_efficiency": self.sample_efficiency,
            "overall_score": self.overall_score,
        }


@dataclass
class UnifiedOptimizationConfig:
    """统一优化配置"""

    # Lift相关参数
    min_lift: float = 0.10  # 最低lift要求
    min_pass_rate: float = 0.20  # 最低通过率
    max_pass_rate: float = 0.80  # 最高通过率

    # 稳定性相关参数
    min_plateau_width: float = 0.05  # 平台最小宽度
    max_lift_std_ratio: float = 0.3  # lift_std / lift_mean 的最大比率
    min_samples_good: int = 50  # good样本最小数量
    min_samples_bad: int = 50  # bad样本最小数量

    # Robustness相关参数
    param_sensitivity_epsilon: float = 0.02  # 参数敏感性阈值
    temporal_cv_folds: int = 5  # 时间交叉验证折数

    # 优化相关参数
    threshold_step: float = 0.05  # 阈值扫描步长
    threshold_range: Tuple[float, float] = (0.05, 0.95)  # 阈值范围


def compute_lift_for_threshold(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,  # 'lt', 'gt', 'le', 'ge'
    threshold: float,
    label_col: str = "is_good",
) -> Dict[str, float]:
    """
    计算给定阈值下的 lift 指标

    Args:
        df: 包含特征和标签的 DataFrame
        feature_col: 特征列名
        operator: 比较运算符
        threshold: 阈值
        label_col: 标签列 (1=good, 0=bad)

    Returns:
        包含 lift, pass_rate_good, pass_rate_bad, pass_rate_all 的字典
    """
    if feature_col not in df.columns:
        return {
            "lift": 0.0,
            "pass_rate_good": 0.0,
            "pass_rate_bad": 0.0,
            "pass_rate_all": 0.0,
        }

    # 计算 pass 条件
    feat_values = df[feature_col]
    if operator == "lt":
        passed = feat_values < threshold
    elif operator == "le":
        passed = feat_values <= threshold
    elif operator == "gt":
        passed = feat_values > threshold
    elif operator == "ge":
        passed = feat_values >= threshold
    else:
        passed = pd.Series([True] * len(df))

    # 分组
    is_good = df[label_col] == 1
    is_bad = df[label_col] == 0

    n_good = is_good.sum()
    n_bad = is_bad.sum()
    n_all = len(df)

    if n_good == 0 or n_bad == 0 or n_all == 0:
        return {
            "lift": 0.0,
            "pass_rate_good": 0.0,
            "pass_rate_bad": 0.0,
            "pass_rate_all": 0.0,
        }

    # 计算各组通过率
    pass_rate_good = (passed & is_good).sum() / n_good if n_good > 0 else 0.0
    pass_rate_bad = (passed & is_bad).sum() / n_bad if n_bad > 0 else 0.0
    pass_rate_all = passed.sum() / n_all if n_all > 0 else 0.0

    # 计算 lift
    # lift = pass_rate_good / pass_rate_bad - 1 (如果 pass_rate_bad > 0.01)
    if pass_rate_bad > 0.01:  # 避免除零
        lift = pass_rate_good / pass_rate_bad - 1.0
    else:
        # 使用替代公式: lift = (pass_rate_good - pass_rate_all) / pass_rate_all
        if pass_rate_all > 0:
            lift = (pass_rate_good - pass_rate_all) / pass_rate_all
        else:
            lift = 0.0

    return {
        "lift": lift,
        "pass_rate_good": pass_rate_good,
        "pass_rate_bad": pass_rate_bad,
        "pass_rate_all": pass_rate_all,
        "n_good": int(n_good),
        "n_bad": int(n_bad),
        "n_passed": int(passed.sum()),
    }


def compute_robustness_score(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,
    threshold: float,
    label_col: str = "is_good",
    config: UnifiedOptimizationConfig = None,
) -> RobustnessScore:
    """
    计算 Execution-Robust Gate Score v2

    核心理念：
    - Robustness 不是 "lift 的稳定性"，而是 "decision boundary 的平缓性"
    - lift 只做 gate-level feasibility，不进入 robustness 计算
    - robustness 要回答的是："这个 gate 会不会炸"

    Args:
        df: DataFrame
        feature_col: 特征列
        operator: 操作符
        threshold: 阈值
        label_col: 标签列
        config: 配置对象

    Returns:
        RobustnessScore 对象
    """
    if config is None:
        config = UnifiedOptimizationConfig()

    # 基础指标
    base_metrics = compute_lift_for_threshold(
        df, feature_col, operator, threshold, label_col
    )
    base_pass_rate_all = base_metrics["pass_rate_all"]
    base_pass_rate_good = base_metrics["pass_rate_good"]
    base_pass_rate_bad = base_metrics["pass_rate_bad"]

    # =========================================================================
    # 1. Decision Boundary 平缓性 (Param Stability)
    # 核心问题：阈值附近 pass/fail 的判定结构稳不稳？
    # 衡量：pass_rate 随阈值扰动的变化程度（越小越平缓）
    # =========================================================================
    perturbations = [
        config.param_sensitivity_epsilon,
        -config.param_sensitivity_epsilon,
    ]
    pass_rate_changes = []

    for eps in perturbations:
        new_threshold = threshold + eps
        new_threshold = max(
            config.threshold_range[0], min(config.threshold_range[1], new_threshold)
        )
        perturbed_metrics = compute_lift_for_threshold(
            df, feature_col, operator, new_threshold, label_col
        )

        # 计算 pass_rate 的变化（三个维度）
        delta_all = abs(perturbed_metrics["pass_rate_all"] - base_pass_rate_all)
        delta_good = abs(perturbed_metrics["pass_rate_good"] - base_pass_rate_good)
        delta_bad = abs(perturbed_metrics["pass_rate_bad"] - base_pass_rate_bad)

        # 综合变化：重点关注 good/bad 的分离度是否稳定
        combined_change = delta_all + 0.5 * (delta_good + delta_bad)
        pass_rate_changes.append(combined_change)

    if pass_rate_changes:
        avg_pass_rate_change = np.mean(pass_rate_changes)
        # decision boundary 越平缓，变化越小，稳定性越高
        # 使用更敏感的衡量：变化超过 5% 就开始惩罚
        param_stability = 1.0 / (1.0 + 10 * avg_pass_rate_change)
    else:
        param_stability = 1.0

    # =========================================================================
    # 2. 时间稳定性 (Temporal Stability)
    # 核心问题：前后半段数据的 pass_rate 是否一致？
    # 衡量：样本结构的时间稳定性
    # =========================================================================
    if len(df) >= 100:
        mid_point = len(df) // 2
        df_first_half = df.iloc[:mid_point].copy()
        df_second_half = df.iloc[mid_point:].copy()

        metrics_first = compute_lift_for_threshold(
            df_first_half, feature_col, operator, threshold, label_col
        )
        metrics_second = compute_lift_for_threshold(
            df_second_half, feature_col, operator, threshold, label_col
        )

        # 比较 pass_rate 的时间差异
        temporal_diff_all = abs(
            metrics_first["pass_rate_all"] - metrics_second["pass_rate_all"]
        )
        temporal_diff_good = abs(
            metrics_first["pass_rate_good"] - metrics_second["pass_rate_good"]
        )
        temporal_diff_bad = abs(
            metrics_first["pass_rate_bad"] - metrics_second["pass_rate_bad"]
        )

        # 综合时间差异
        combined_temporal_diff = temporal_diff_all + 0.5 * (
            temporal_diff_good + temporal_diff_bad
        )
        temporal_stability = 1.0 / (1.0 + 5 * combined_temporal_diff)
    else:
        temporal_stability = 1.0

    # =========================================================================
    # 3. 样本效率 (Sample Efficiency)
    # 核心问题：有足够的 good/bad 样本支撑这个判定吗？
    # =========================================================================
    n_good = base_metrics["n_good"]
    n_bad = base_metrics["n_bad"]
    n_passed = base_metrics["n_passed"]

    min_samples = min(config.min_samples_good, config.min_samples_bad)
    eff_good = (
        min(np.log(max(n_good, 1)) / np.log(max(min_samples, 1)), 1.0)
        if min_samples > 0
        else 1.0
    )
    eff_bad = (
        min(np.log(max(n_bad, 1)) / np.log(max(min_samples, 1)), 1.0)
        if min_samples > 0
        else 1.0
    )

    # 额外惩罚：如果 passed 样本太少，结果不可靠
    passed_ratio = n_passed / len(df) if len(df) > 0 else 0
    passed_penalty = 1.0 if passed_ratio > 0.1 else passed_ratio / 0.1

    sample_efficiency = min(eff_good, eff_bad) * passed_penalty

    # =========================================================================
    # 综合评分
    # 注意：不使用 lift，只使用 decision boundary 的稳定性指标
    # =========================================================================
    overall_score = (
        0.45 * param_stability  # decision boundary 平缓性
        + 0.35 * temporal_stability  # 时间稳定性
        + 0.20 * sample_efficiency  # 样本充足度
    )

    return RobustnessScore(
        param_stability=param_stability,
        temporal_stability=temporal_stability,
        sample_efficiency=sample_efficiency,
        overall_score=overall_score,
    )


def find_stable_lift_plateau(
    results: List[Dict[str, Any]],
    config: UnifiedOptimizationConfig,
) -> Optional[Dict[str, Any]]:
    """
    找到稳定的lift平台区间 - 改进版，基于稳定性而非连续性

    Args:
        results: 扫描结果列表
        config: 配置对象

    Returns:
        稳定平台信息，包含 start, end, mid, metrics 等
    """
    if not results:
        return None

    # 按阈值排序
    results_sorted = sorted(results, key=lambda x: x["threshold"])

    # 筛选满足基础条件的阈值
    valid_thresholds = []
    for r in results_sorted:
        if (
            r["lift"] >= config.min_lift
            and config.min_pass_rate <= r["pass_rate_all"] <= config.max_pass_rate
            and r.get("n_good", 0) >= config.min_samples_good
            and r.get("n_bad", 0) >= config.min_samples_bad
        ):
            valid_thresholds.append(r)

    if len(valid_thresholds) < 2:
        return None

    # 寻找稳定的连续区间（基于lift变化的稳定性）
    stable_intervals = []

    i = 0
    while i < len(valid_thresholds):
        start_idx = i
        current_lift = valid_thresholds[i]["lift"]
        current_threshold = valid_thresholds[i]["threshold"]

        # 寻找稳定区间
        j = i + 1
        while j < len(valid_thresholds):
            next_threshold = valid_thresholds[j]["threshold"]
            next_lift = valid_thresholds[j]["lift"]

            # 检查阈值连续性（相邻阈值间隔不能过大）
            if next_threshold - current_threshold > config.threshold_step * 2:
                break

            # 检查lift稳定性（lift变化不能过大）
            lift_change = abs(next_lift - current_lift)
            if lift_change > config.max_lift_std_ratio * abs(current_lift):
                break

            current_threshold = next_threshold
            current_lift = next_lift
            j += 1

        # 如果找到了长度>=2的稳定区间
        if j - start_idx >= 2:
            interval = valid_thresholds[start_idx:j]
            # 计算此区间的稳定性指标
            lifts = [r["lift"] for r in interval]
            lift_mean = np.mean(lifts)
            lift_std = np.std(lifts) if len(lifts) > 1 else 0.0
            lift_min = np.min(lifts)
            lift_max = np.max(lifts)

            # 检查区间宽度是否满足要求
            interval_width = interval[-1]["threshold"] - interval[0]["threshold"]
            if interval_width >= config.min_plateau_width:
                stable_intervals.append(
                    {
                        "interval": interval,
                        "start_idx": start_idx,
                        "end_idx": j - 1,
                        "start_threshold": interval[0]["threshold"],
                        "end_threshold": interval[-1]["threshold"],
                        "width": interval_width,
                        "lift_mean": lift_mean,
                        "lift_std": lift_std,
                        "lift_min": lift_min,
                        "lift_max": lift_max,
                        "lift_stability_ratio": (
                            lift_std / lift_mean if lift_mean != 0 else float("inf")
                        ),
                        "num_points": len(interval),
                    }
                )

        i = j if j > i else i + 1

    if not stable_intervals:
        return None

    # 选择最好的稳定区间
    # 优先级：1. lift_stability_ratio 越小越稳定 2. 宽度越大越好 3. lift均值越高越好
    # 注意：使用 min 配合 (ratio, -width, -lift_mean)
    best_interval = min(
        stable_intervals,
        key=lambda x: (x["lift_stability_ratio"], -x["width"], -x["lift_mean"]),
    )

    interval = best_interval["interval"]
    start_th = best_interval["start_threshold"]
    end_th = best_interval["end_threshold"]
    mid_th = (start_th + end_th) / 2

    # 找到mid附近最接近的阈值的metrics
    mid_metrics = min(interval, key=lambda x: abs(x["threshold"] - mid_th))

    return {
        "plateau_start": start_th,
        "plateau_end": end_th,
        "plateau_mid": mid_th,
        "recommended_threshold": mid_th,
        "lift_mean": best_interval["lift_mean"],
        "lift_std": best_interval["lift_std"],
        "lift_min": best_interval["lift_min"],
        "lift_max": best_interval["lift_max"],
        "lift_stability_ratio": best_interval["lift_stability_ratio"],
        "pass_rate_at_mid": mid_metrics["pass_rate_all"],
        "lift_at_mid": mid_metrics["lift"],
        "plateau_width": best_interval["width"],
        "num_valid_thresholds": best_interval["num_points"],
        "interval_details": [r for r in interval],  # 包含所有点的详细信息
    }


def scan_thresholds_for_lift(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,
    threshold_range: Tuple[float, float],
    step: float,
    label_col: str = "is_good",
) -> List[Dict[str, Any]]:
    """
    扫描阈值范围，计算每个阈值的 lift

    Returns:
        阈值 -> lift 指标的列表
    """
    results = []
    low, high = threshold_range
    thresholds = np.arange(low, high + step / 2, step)

    for th in thresholds:
        metrics = compute_lift_for_threshold(df, feature_col, operator, th, label_col)
        metrics["threshold"] = float(th)
        results.append(metrics)

    return results


def _parse_gate_when_condition(
    when: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    解析 Gate 规则的 when 条件，支持多种格式

    支持格式:
    1. 直接格式: {feature: {value_lt: 0.5}}
    2. any_of 格式: {any_of: [{feature: {value_lt: 0.5}}, ...]}
    3. all_of 格式: {all_of: [{feature: {value_lt: 0.5}}, ...]}
    4. quantile 格式: {feature: {quantile_gt: 0.85}}

    Returns:
        (feature_col, operator, threshold)
    """
    if not isinstance(when, dict):
        return None, None, None

    # 处理 any_of / all_of 嵌套结构 - 取第一个条件优化
    if "any_of" in when:
        conditions = when["any_of"]
        if conditions and isinstance(conditions, list):
            return _parse_gate_when_condition(conditions[0])

    if "all_of" in when:
        conditions = when["all_of"]
        if conditions and isinstance(conditions, list):
            return _parse_gate_when_condition(conditions[0])

    # 直接格式: {feature_name: {value_lt/value_gt/quantile_lt/quantile_gt: threshold}}
    for feature_col, value_dict in when.items():
        if feature_col in ("any_of", "all_of"):
            continue

        if not isinstance(value_dict, dict):
            continue

        # 解析操作符和阈值
        for op_key, threshold in value_dict.items():
            # value_lt, value_gt, value_lte, value_gte
            if op_key.startswith("value_"):
                op_suffix = op_key[6:]  # 去掉 "value_" 前缀
                operator = op_suffix.replace("lte", "le").replace("gte", "ge")
                return feature_col, operator, float(threshold)

            # quantile_lt, quantile_gt
            if op_key.startswith("quantile_"):
                op_suffix = op_key[9:]  # 去掉 "quantile_" 前缀
                operator = op_suffix.replace("lte", "le").replace("gte", "ge")
                return feature_col, operator, float(threshold)

    return None, None, None


def optimize_gate_rule_unified(
    df: pd.DataFrame,
    rule: GateRule,
    label_col: str = "is_good",
    config: Optional[UnifiedOptimizationConfig] = None,
    step: float = 0.05,
) -> Dict[str, Any]:
    """
    统一的门控规则优化函数

    Args:
        df: 包含特征和标签的 DataFrame
        rule: GateRule 对象
        label_col: 标签列名
        config: 优化配置
        step: 阈值扫描步长

    Returns:
        优化结果
    """
    if config is None:
        config = UnifiedOptimizationConfig()

    # 从规则的 when 条件中提取特征和运算符
    when = rule.when
    feature_col, operator, current_threshold = _parse_gate_when_condition(when)

    if feature_col is None or operator is None:
        return {
            "rule_id": rule.id,
            "status": "skip",
            "reason": f"Could not parse rule when condition: {when}",
        }

    # 检查特征是否存在
    if feature_col not in df.columns:
        return {
            "rule_id": rule.id,
            "feature": feature_col,
            "status": "skip",
            "reason": f"Feature {feature_col} not found in DataFrame",
        }

    # 确定阈值范围
    # 对于 quantile 特征，范围是 [0, 1]
    if "_pct" in feature_col or "quantile" in feature_col:
        threshold_range = (0.05, 0.95)  # 避免边界值
    else:
        # 使用数据分位数确定范围
        q_low = df[feature_col].quantile(0.05)
        q_high = df[feature_col].quantile(0.95)
        threshold_range = (q_low, q_high)
        step = max(step, (q_high - q_low) / 20)  # 自适应步长

    # 扫描阈值
    results = scan_thresholds_for_lift(
        df, feature_col, operator, threshold_range, step, label_col
    )

    # 先进行lift筛选（可行性检查）
    valid_results = [
        r
        for r in results
        if r["lift"] >= config.min_lift
        and config.min_pass_rate <= r["pass_rate_all"] <= config.max_pass_rate
        and r.get("n_good", 0) >= config.min_samples_good
        and r.get("n_bad", 0) >= config.min_samples_bad
    ]

    if not valid_results:
        return {
            "rule_id": rule.id,
            "feature": feature_col,
            "status": "no_valid_threshold",
            "reason": "No threshold meets basic lift/pass_rate requirements",
            "scan_results": results,
        }

    # 寻找稳定的平台区间
    stable_plateau = find_stable_lift_plateau(results, config)

    if stable_plateau is None:
        # 找不到稳定平台，使用robustness导向的选择策略
        # 在满足基础条件的阈值中选择robustness最高的
        best_result = None
        best_robustness_score = -1

        for r in valid_results:
            robustness = compute_robustness_score(
                df, feature_col, operator, r["threshold"], label_col, config
            )
            if robustness.overall_score > best_robustness_score:
                best_robustness_score = robustness.overall_score
                best_result = {
                    **r,
                    "robustness_score": robustness.to_dict(),
                    "recommended_threshold": r["threshold"],
                }

        if best_result:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "operator": operator,
                "current_threshold": current_threshold,
                "status": "no_stable_plateau",
                "robustness_selection": True,
                **best_result,
                "scan_results": results,
            }
        else:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "status": "no_robust_threshold",
                "reason": "No robust threshold found even with basic requirements met",
                "scan_results": results,
            }

    # 在稳定平台内选择最稳健的阈值
    plateau_results = stable_plateau["interval_details"]
    best_result_in_plateau = None
    best_robustness_in_plateau = -1

    for r in plateau_results:
        robustness = compute_robustness_score(
            df, feature_col, operator, r["threshold"], label_col, config
        )
        if robustness.overall_score > best_robustness_in_plateau:
            best_robustness_in_plateau = robustness.overall_score
            best_result_in_plateau = {
                **r,
                "robustness_score": robustness.to_dict(),
                "recommended_threshold": r["threshold"],
            }

    # 如果平台内的最佳点与中位数不同，提供选择依据
    if best_result_in_plateau["threshold"] != stable_plateau["plateau_mid"]:
        # 比较两者，选择robustness更好的
        # 找到mid附近的结果（使用更宽松的匹配）
        mid_candidates = [
            r
            for r in plateau_results
            if abs(r["threshold"] - stable_plateau["plateau_mid"]) < step * 1.5
        ]
        if mid_candidates:
            mid_result = min(
                mid_candidates,
                key=lambda r: abs(r["threshold"] - stable_plateau["plateau_mid"]),
            )
            mid_robustness = compute_robustness_score(
                df, feature_col, operator, mid_result["threshold"], label_col, config
            )

            if best_robustness_in_plateau > mid_robustness.overall_score:
                recommended_threshold = best_result_in_plateau["threshold"]
            else:
                recommended_threshold = stable_plateau["plateau_mid"]
                best_result_in_plateau = {
                    **mid_result,
                    "robustness_score": mid_robustness.to_dict(),
                    "recommended_threshold": stable_plateau["plateau_mid"],
                }
        else:
            # 如果找不到mid附近的结果，使用best_result_in_plateau
            recommended_threshold = best_result_in_plateau["threshold"]
    else:
        recommended_threshold = best_result_in_plateau["threshold"]

    return {
        "rule_id": rule.id,
        "feature": feature_col,
        "operator": operator,
        "current_threshold": current_threshold,
        "status": "stable_plateau_found",
        "robustness_selection": True,
        **stable_plateau,
        "recommended_threshold": recommended_threshold,
        "best_result_in_plateau": best_result_in_plateau,
        "scan_results": results,
    }


def _generate_html_report(
    df: pd.DataFrame,
    opt_results: Dict[str, Any],
    output_path: Path,
    label_col: str = "is_good",
) -> None:
    """生成美化的 HTML 报告"""
    from datetime import datetime

    # 计算汇总指标
    n_all = len(df)
    n_good_all = (df[label_col] == 1).sum()
    n_bad_all = n_all - n_good_all
    good_rate_all = n_good_all / n_all if n_all > 0 else 0

    # 检查是否有 gate_decision 列
    if "gate_decision" in df.columns:
        allowed = df[df["gate_decision"] == "allow"]
        n_allowed = len(allowed)
        good_rate_allowed = allowed[label_col].mean() if len(allowed) > 0 else 0
        lift = (good_rate_allowed / good_rate_all - 1) if good_rate_all > 0 else 0
        pass_rate = n_allowed / n_all if n_all > 0 else 0
        veto_df = df[df["gate_decision"] == "veto"]
        bad_rejection_rate = (
            ((veto_df[label_col] == 0).sum() / n_bad_all) if n_bad_all > 0 else 0
        )
        good_retention_rate = (
            (allowed[label_col].sum() / n_good_all) if n_good_all > 0 else 0
        )
    else:
        n_allowed = n_all
        good_rate_allowed = good_rate_all
        lift = 0
        pass_rate = 1.0
        bad_rejection_rate = 0
        good_retention_rate = 1.0
        allowed = df

    # Sharpe (额外参考)
    rr_cols = ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]
    rr_col = next((c for c in rr_cols if c in df.columns), None)
    if rr_col:
        sharpe_all = df[rr_col].mean() / df[rr_col].std() if df[rr_col].std() > 0 else 0
        sharpe_allowed = (
            allowed[rr_col].mean() / allowed[rr_col].std()
            if len(allowed) > 0 and allowed[rr_col].std() > 0
            else 0
        )
    else:
        sharpe_all = 0
        sharpe_allowed = 0

    # 生成规则表格 HTML
    rules_html = ""
    for rule_id, result in opt_results.items():
        status = result.get("status", "N/A")

        if status == "stable_plateau_found":
            status_html = '<span class="status-ok">✅ 稳定平台</span>'
            threshold = result.get("recommended_threshold")
            th_str = (
                f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "N/A"
            )
            lift_val = result.get("lift_at_mid", result.get("lift", 0))
            lift_html = (
                f"{lift_val*100:.2f}%" if isinstance(lift_val, (int, float)) else "N/A"
            )
            pr = result.get("pass_rate_at_mid", result.get("pass_rate_all", 0))
            pr_html = f"{pr*100:.1f}%" if isinstance(pr, (int, float)) else "N/A"
            rob = (
                result.get("best_result_in_plateau", {})
                .get("robustness_score", {})
                .get("overall_score")
            )
            if rob is None:
                rob = result.get("robustness_score", {}).get("overall_score")
            rob_html = f"{rob:.3f}" if isinstance(rob, (int, float)) else "N/A"
        elif status == "no_stable_plateau":
            status_html = '<span class="status-warn">⚠️ 无稳定平台</span>'
            threshold = result.get("recommended_threshold")
            th_str = (
                f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "N/A"
            )
            lift_val = result.get("lift", 0)
            lift_html = (
                f"{lift_val*100:.2f}%" if isinstance(lift_val, (int, float)) else "N/A"
            )
            pr = result.get("pass_rate_all", 0)
            pr_html = f"{pr*100:.1f}%" if isinstance(pr, (int, float)) else "N/A"
            rob = result.get("robustness_score", {}).get("overall_score")
            rob_html = f"{rob:.3f}" if isinstance(rob, (int, float)) else "N/A"
        else:
            status_html = '<span class="status-fail">❌ 无效</span>'
            th_str = "-"
            lift_html = "-"
            pr_html = "-"
            rob_html = "-"

        rules_html += f"""
            <tr>
                <td><code>{rule_id}</code></td>
                <td>{status_html}</td>
                <td><strong>{th_str}</strong></td>
                <td>{lift_html}</td>
                <td>{pr_html}</td>
                <td>{rob_html}</td>
            </tr>"""

    # 完整 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gate 优化报告 - Execution-Robust v2</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #1a73e8; margin-bottom: 30px; font-size: 28px; }}
        h2 {{ color: #34495e; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; margin: 30px 0 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .kpi-item {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .kpi-item.primary {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .kpi-item.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .kpi-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .kpi-label {{ font-size: 14px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; color: #2c3e50; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .status-ok {{ color: #27ae60; font-weight: bold; }}
        .status-warn {{ color: #f39c12; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .metric-bar {{ height: 8px; background: #ecf0f1; border-radius: 4px; overflow: hidden; margin-top: 5px; }}
        .metric-fill {{ height: 100%; background: linear-gradient(90deg, #11998e, #38ef7d); border-radius: 4px; }}
        .secondary {{ color: #7f8c8d; font-size: 14px; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .timestamp {{ text-align: center; color: #95a5a6; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🎯 Gate 层效果评估报告</h1>
    <p style="text-align:center;color:#7f8c8d;margin-bottom:30px;">Execution-Robust v2</p>
    
    <h2>📊 核心 KPI</h2>
    <div class="card">
        <div class="kpi-grid">
            <div class="kpi-item primary">
                <div class="kpi-label">LIFT (核心指标)</div>
                <div class="kpi-value">+{lift*100:.2f}%</div>
                <div class="kpi-label">Good Rate: {good_rate_all*100:.1f}% → {good_rate_allowed*100:.1f}%</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Pass Rate</div>
                <div class="kpi-value">{pass_rate*100:.1f}%</div>
                <div class="kpi-label">{n_allowed:,} / {n_all:,} trades</div>
            </div>
            <div class="kpi-item warning">
                <div class="kpi-label">Bad Rejection Rate</div>
                <div class="kpi-value">{bad_rejection_rate*100:.1f}%</div>
                <div class="kpi-label">拒绝坏样本比例</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Good Retention Rate</div>
                <div class="kpi-value">{good_retention_rate*100:.1f}%</div>
                <div class="kpi-label">保留好样本比例</div>
            </div>
        </div>
    </div>
    
    <h2>🔧 Gate 规则优化结果</h2>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>规则 ID</th>
                    <th>状态</th>
                    <th>推荐阈值</th>
                    <th>Lift</th>
                    <th>Pass Rate</th>
                    <th>Robustness</th>
                </tr>
            </thead>
            <tbody>{rules_html}
            </tbody>
        </table>
    </div>
    
    <h2>📈 样本分布</h2>
    <div class="card">
        <table>
            <tr><th>类别</th><th>数量</th><th>比例</th><th>分布</th></tr>
            <tr>
                <td>总样本</td><td>{n_all:,}</td><td>100%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:100%"></div></div></td>
            </tr>
            <tr>
                <td>├─ Good 样本</td><td>{n_good_all:,}</td><td>{good_rate_all*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{good_rate_all*100}%"></div></div></td>
            </tr>
            <tr>
                <td>├─ Bad 样本</td><td>{n_bad_all:,}</td><td>{(1-good_rate_all)*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{(1-good_rate_all)*100}%;background:linear-gradient(90deg,#e74c3c,#c0392b)"></div></div></td>
            </tr>
            <tr>
                <td>Gate Allow</td><td>{n_allowed:,}</td><td>{pass_rate*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{pass_rate*100}%"></div></div></td>
            </tr>
            <tr>
                <td>└─ Good in Allow</td><td>{int(allowed[label_col].sum()):,}</td><td>{good_rate_allowed*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{good_rate_allowed*100}%"></div></div></td>
            </tr>
        </table>
    </div>
    
    <div class="secondary">
        <h3 style="margin-bottom:10px;">📎 额外参考 (Sharpe Ratio)</h3>
        <p>基准 Sharpe (无 Gate): <strong>{sharpe_all:.4f}</strong></p>
        <p>Allow Sharpe: <strong>{sharpe_allowed:.4f}</strong></p>
        <p style="color:#95a5a6;font-size:12px;margin-top:10px;">注：Sharpe Ratio 仅作为参考指标，Gate 优化以 Lift 为核心目标</p>
    </div>
    
    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Gate Optimization - Production Grade Gate Parameter Optimizer"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name (e.g., bpc)",
    )
    parser.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="Root directory for strategy configs",
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Trade logs parquet with features and labels",
    )
    parser.add_argument(
        "--label-col",
        default="is_good",
        help="Label column name (1=good, 0=bad)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=0.10,
        help="Minimum lift requirement",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.20,
        help="Minimum pass rate",
    )
    parser.add_argument(
        "--max-pass-rate",
        type=float,
        default=0.80,
        help="Maximum pass rate",
    )
    parser.add_argument(
        "--min-plateau-width",
        type=float,
        default=0.05,
        help="Minimum plateau width",
    )
    parser.add_argument(
        "--max-lift-std-ratio",
        type=float,
        default=0.3,
        help="Maximum lift std / lift mean ratio for stability",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Threshold scan step size",
    )
    parser.add_argument(
        "--write-back-intervals",
        action="store_true",
        help="Write back interval thresholds (start, end) instead of single point",
    )
    parser.add_argument(
        "--interval-method",
        choices=["plateau", "robustness"],
        default="plateau",
        help="Method to determine intervals: plateau bounds or robustness-driven",
    )
    args = parser.parse_args()

    # Load logs
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"✅ Loaded {len(df)} rows from {logs_path}")

    # 自动生成 is_good 列 (如果不存在)
    rr_col = None
    for candidate in ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]:
        if candidate in df.columns:
            rr_col = candidate
            break

    if args.label_col not in df.columns:
        if rr_col is not None:
            # 基于 rr_extreme 标签定义: Good = RR >= -0.8, Bad = RR < -0.8
            df[args.label_col] = (df[rr_col] >= -0.8).astype(int)
            print(
                f"ℹ️ Auto-generated '{args.label_col}' column from '{rr_col}' (threshold: -0.8)"
            )
        else:
            print(f"❌ Cannot auto-generate '{args.label_col}': no RR column found")
            print(f"   Tried: bpc_impulse_return_atr, forward_rr, rr, return_atr")
            return 1

    # Check label column
    if args.label_col not in df.columns:
        print(f"❌ Label column '{args.label_col}' not found in DataFrame")
        print(f"   Available columns: {list(df.columns)[:20]}...")
        return 1

    n_good = (df[args.label_col] == 1).sum()
    n_bad = (df[args.label_col] == 0).sum()
    print(f"   Good samples: {n_good}, Bad samples: {n_bad}")
    print(f"   Good rate: {n_good/(n_good+n_bad):.3f}")

    # Create config
    config = UnifiedOptimizationConfig(
        min_lift=args.min_lift,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
        min_plateau_width=args.min_plateau_width,
        max_lift_std_ratio=args.max_lift_std_ratio,
        threshold_step=args.step,
    )

    # Run optimizations
    all_results = {}

    # Load strategy archetype
    try:
        arch = load_strategy_archetype(args.strategy, args.strategies_root)
        print(f"✅ Loaded strategy: {arch.name}")
        print(f"   Hard gates: {len(arch.gate.hard_gates)}")
        print(f"   Soft filters: {len(arch.gate.soft_filters)}")

        # Process hard gates
        print("\n📋 Optimizing Hard Gates:")
        for rule in arch.gate.hard_gates:
            print(f"  Processing: {rule.id}")
            result = optimize_gate_rule_unified(
                df, rule, args.label_col, config, args.step
            )
            all_results[rule.id] = result

            if result.get("status") in ["stable_plateau_found", "no_stable_plateau"]:
                rec_thresh = result.get(
                    "recommended_threshold", result.get("threshold")
                )
                lift_val = result.get("lift_at_mid", result.get("lift"))
                pass_rate = result.get("pass_rate_at_mid", result.get("pass_rate_all"))
                rob_score = result.get("robustness_score", {}).get("overall_score")

                status_msg = (
                    "✅ Stable plateau"
                    if result.get("status") == "stable_plateau_found"
                    else "⚠️ No stable plateau"
                )
                # 安全格式化，避免 None 或字符串导致错误
                th_str = (
                    f"{rec_thresh:.3f}"
                    if isinstance(rec_thresh, (int, float))
                    else str(rec_thresh)
                )
                lift_str = (
                    f"{lift_val:.3f}"
                    if isinstance(lift_val, (int, float))
                    else str(lift_val)
                )
                pr_str = (
                    f"{pass_rate:.3f}"
                    if isinstance(pass_rate, (int, float))
                    else str(pass_rate)
                )
                rob_str = (
                    f"{rob_score:.3f}"
                    if isinstance(rob_score, (int, float))
                    else str(rob_score)
                )
                print(
                    f"    {status_msg}: Threshold={th_str}, Lift={lift_str}, PassRate={pr_str}, Robustness={rob_str}"
                )
            else:
                print(f"    ⚠️  {result.get('status')}: {result.get('reason', 'N/A')}")

        # Process soft filters
        print("\n📋 Optimizing Soft Filters:")
        for rule in arch.gate.soft_filters:
            print(f"  Processing: {rule.id}")
            result = optimize_gate_rule_unified(
                df, rule, args.label_col, config, args.step
            )
            all_results[rule.id] = result

            if result.get("status") in ["stable_plateau_found", "no_stable_plateau"]:
                rec_thresh = result.get(
                    "recommended_threshold", result.get("threshold")
                )
                lift_val = result.get("lift_at_mid", result.get("lift"))
                pass_rate = result.get("pass_rate_at_mid", result.get("pass_rate_all"))
                rob_score = result.get("robustness_score", {}).get("overall_score")

                status_msg = (
                    "✅ Stable plateau"
                    if result.get("status") == "stable_plateau_found"
                    else "⚠️ No stable plateau"
                )
                # 安全格式化
                th_str = (
                    f"{rec_thresh:.3f}"
                    if isinstance(rec_thresh, (int, float))
                    else str(rec_thresh)
                )
                lift_str = (
                    f"{lift_val:.3f}"
                    if isinstance(lift_val, (int, float))
                    else str(lift_val)
                )
                pr_str = (
                    f"{pass_rate:.3f}"
                    if isinstance(pass_rate, (int, float))
                    else str(pass_rate)
                )
                rob_str = (
                    f"{rob_score:.3f}"
                    if isinstance(rob_score, (int, float))
                    else str(rob_score)
                )
                print(
                    f"    {status_msg}: Threshold={th_str}, Lift={lift_str}, PassRate={pr_str}, Robustness={rob_str}"
                )
            else:
                print(f"    ⚠️  {result.get('status')}: {result.get('reason', 'N/A')}")
    except Exception as e:
        print(f"❌ Failed to load strategy '{args.strategy}': {e}")
        return 1

    # Prepare results for output
    final_results = {}
    for k, v in all_results.items():
        # Create clean result without large data
        clean_v = {kk: vv for kk, vv in v.items() if kk != "scan_results"}
        # Also remove interval_details to reduce output size
        if "interval_details" in clean_v:
            del clean_v["interval_details"]

        # Add interval information if requested
        if args.write_back_intervals:
            if v.get("status") == "stable_plateau_found":
                if args.interval_method == "plateau":
                    # Use plateau bounds
                    clean_v["threshold_interval"] = {
                        "start": v["plateau_start"],
                        "end": v["plateau_end"],
                        "method": "plateau_bounds",
                    }
                else:  # robustness method
                    # Calculate interval based on robustness considerations
                    center = v["recommended_threshold"]
                    # Use plateau width divided by 2 as buffer
                    half_width = (
                        v["plateau_width"] / 2
                        if v.get("plateau_width", 0) > 0
                        else args.step
                    )
                    clean_v["threshold_interval"] = {
                        "start": max(v["plateau_start"], center - half_width),
                        "end": min(v["plateau_end"], center + half_width),
                        "method": "robustness_centered",
                    }
            elif v.get("status") == "no_stable_plateau":
                # For non-stable cases, create small interval around recommended threshold
                center = v["recommended_threshold"]
                buffer = args.step
                clean_v["threshold_interval"] = {
                    "start": center - buffer,
                    "end": center + buffer,
                    "method": "single_point_with_buffer",
                }

        final_results[k] = clean_v

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Results saved to: {output_path}")

    # Summary
    n_plateau = sum(
        1 for r in final_results.values() if r.get("status") == "stable_plateau_found"
    )
    n_no_plateau = sum(
        1 for r in final_results.values() if r.get("status") == "no_stable_plateau"
    )
    n_skip = sum(
        1
        for r in final_results.values()
        if r.get("status") in ("skip", "no_valid_threshold", "no_robust_threshold")
    )
    n_with_intervals = sum(
        1 for r in final_results.values() if "threshold_interval" in r
    )

    print(f"\n📊 Summary:")
    print(f"   Stable plateaus found: {n_plateau}")
    print(f"   No stable plateau (robustness selection used): {n_no_plateau}")
    print(f"   Skipped/Failed: {n_skip}")
    if args.write_back_intervals:
        print(f"   Rules with interval thresholds: {n_with_intervals}")

    # ==========================================================================
    # 生成美化的 HTML 报告
    # ==========================================================================
    html_path = output_path.with_suffix(".html")
    _generate_html_report(df, final_results, html_path, args.label_col)
    print(f"✅ HTML report saved to: {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
