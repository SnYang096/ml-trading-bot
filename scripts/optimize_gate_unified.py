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
    # lift = pass_rate_good / pass_rate_bad - 1
    # lift=1.0 表示 good 通过率是 bad 的 2 倍, 是有意义的最低标准
    # lift=0.1 只提升 10%, 几乎无用
    min_lift: float = 1.0  # 最低 lift 要求 (之前 0.10 太低, 会保留无效规则)
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

    # Hard gate 严格模式：没有 plateau 则不允许 fallback 到单点
    # 默认 False，允许 robustness fallback（过渡态）
    strict_hard: bool = False

    # =========================================================================
    # Hard Gate NaN Lift 例外通道配置 https://chatgpt.com/s/t_69886fc99e7481918ad7880c8a65f732
    # 语义：Hard Gate 的合法性来源是"结构稳定性 + 执行风险厌恶"，不是 lift
    # 只有同时满足以下所有条件，才允许 lift=NaN 的 Hard Gate 进入 robustness 仲裁
    # =========================================================================
    allow_hard_nan_lift: bool = True  # 是否允许 Hard Gate 的 NaN lift 例外
    nan_lift_max_pass_rate_bad: float = 0.01  # pass_rate_bad 必须极低
    nan_lift_min_pass_rate_good: float = 0.20  # pass_rate_good 必须足够高
    nan_lift_min_coverage: float = 0.15  # 特征覆盖率下限
    nan_lift_min_robustness: float = 0.60  # robustness 下限
    nan_lift_min_plateau_width: float = 0.03  # 阈值区间宽度下限（排除点估计）


def compute_lift_for_threshold(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,  # 'lt', 'gt', 'le', 'ge' - 这是 DENY 条件的 operator
    threshold: float,
    label_col: str = "is_good",
) -> Dict[str, float]:
    """
    计算给定阈值下的 lift 指标

    Args:
        df: 包含特征和标签的 DataFrame
        feature_col: 特征列名
        operator: DENY 条件的比较运算符 (来自 gate.yaml 的 value_lt/value_gt 等)
        threshold: 阈值
        label_col: 标签列 (1=good, 0=bad)

    Returns:
        包含 lift, pass_rate_good, pass_rate_bad, pass_rate_all 的字典

    Note:
        operator 是 DENY 条件，所以 PASS 条件是其反面:
        - deny when `value_lt X` → pass when `>= X`
        - deny when `value_gt X` → pass when `<= X`
        - deny when `value_le X` → pass when `> X`
        - deny when `value_ge X` → pass when `< X`
    """
    if feature_col not in df.columns:
        return {
            "lift": 0.0,
            "pass_rate_good": 0.0,
            "pass_rate_bad": 0.0,
            "pass_rate_all": 0.0,
        }

    # 计算 PASS 条件 (deny 条件的反面)
    feat_values = df[feature_col]

    # ❗ 问题 1 修复: NaN 特征的显式处理
    # Policy: NaN = 不参与 gate（既不是 pass 也不是 deny）
    # 这样避免数据缺失导致的假稳态
    valid_mask = feat_values.notna()
    n_valid = valid_mask.sum()

    if operator == "lt":  # deny when < threshold, so pass when >= threshold
        passed = valid_mask & (feat_values >= threshold)
    elif operator == "le":  # deny when <= threshold, so pass when > threshold
        passed = valid_mask & (feat_values > threshold)
    elif operator == "gt":  # deny when > threshold, so pass when <= threshold
        passed = valid_mask & (feat_values <= threshold)
    elif operator == "ge":  # deny when >= threshold, so pass when < threshold
        passed = valid_mask & (feat_values < threshold)
    else:
        # ❗ 无效 operator 不应静默通过，返回错误状态
        raise ValueError(
            f"Invalid operator: {operator}. Expected one of: lt, le, gt, ge"
        )

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

    # =========================================================================
    # ❗ Bug A 修复: pass_rate 分母用 valid (参与 gate 的) 样本数
    # NaN 样本不参与 gate，所以不应计入分母
    # 语义: "在参与 gate 的样本中，通过率是多少"
    # =========================================================================
    valid_good = (is_good & valid_mask).sum()
    valid_bad = (is_bad & valid_mask).sum()

    # 计算各组通过率（分母是 valid 样本数）
    pass_rate_good = (passed & is_good).sum() / valid_good if valid_good > 0 else 0.0
    pass_rate_bad = (passed & is_bad).sum() / valid_bad if valid_bad > 0 else 0.0
    pass_rate_all = passed.sum() / n_valid if n_valid > 0 else 0.0

    # 计算 lift
    # lift = pass_rate_good / pass_rate_bad - 1
    # ❗ 当 pass_rate_bad 太小时，直接返回 NaN，不使用任何 fallback 公式
    # 避免制造“假高原”和虚假高 lift
    if pass_rate_bad < 0.01:
        # bad 样本几乎全被拒绝，lift 无法可靠计算
        lift = float("nan")
    elif pass_rate_bad > 0:
        lift = pass_rate_good / pass_rate_bad - 1.0
    else:
        lift = float("nan")

    return {
        "lift": lift,
        "lift_valid": np.isfinite(lift),  # ❗ 问题 2 修复: 显式标记 lift 是否有效
        "pass_rate_good": pass_rate_good,
        "pass_rate_bad": pass_rate_bad,
        "pass_rate_all": pass_rate_all,
        "n_good": int(n_good),
        "n_bad": int(n_bad),
        "n_passed": int(passed.sum()),
        "n_valid": int(n_valid),  # 有效样本数（非 NaN）
        # ❗ Bug 1 修复: 输出 valid_good / valid_bad，供 robustness 使用
        "valid_good": int(valid_good),  # 参与 gate 的 good 样本数
        "valid_bad": int(valid_bad),  # 参与 gate 的 bad 样本数
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

    # =========================================================================
    # ❗ Bug 1 修复: robustness 与 lift 完全解耦
    # robustness = "决策边界稳不稳"  (decision boundary stability)
    # lift = "值不值得 gate"       (另一个阶段的决策)
    # 这里只检查样本约束，不检查 lift 是否有限
    # ❗ Bug 1 增强: 使用 valid_good / valid_bad（参与 gate 的样本数）
    # 而不是全样本数，避免低覆盖率 feature 被误判为"样本充分"
    # =========================================================================
    valid_good = base_metrics.get("valid_good", base_metrics["n_good"])
    valid_bad = base_metrics.get("valid_bad", base_metrics["n_bad"])

    if valid_bad < config.min_samples_bad or valid_good < config.min_samples_good:
        # 样本不足，统计不可信
        return RobustnessScore(
            param_stability=0.0,
            temporal_stability=0.0,
            sample_efficiency=0.0,
            overall_score=0.0,
        )

    base_pass_rate_all = base_metrics["pass_rate_all"]
    base_pass_rate_good = base_metrics["pass_rate_good"]
    base_pass_rate_bad = base_metrics["pass_rate_bad"]

    # =========================================================================
    # 1. Decision Boundary 平缓性 (Param Stability)
    # 核心问题：阈值附近 pass/fail 的判定结构稳不稳？
    # 衡量：pass_rate 随阈值扰动的变化程度（越小越平缓）
    # ❗ Bug fix #4: 对变化做相对 scale，避免对不同量纲 feature 不可比
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

    if pass_rate_changes and base_pass_rate_all > 0:
        avg_pass_rate_change = np.mean(pass_rate_changes)
        # ❗ 问题 4 修复: 加 floor 避免 tight gate 虚高
        # 当 base_pass_rate_all 很小时，小变化会被放大
        # 设置 floor=0.1，确保只惩罚 unstable，不惩罚 tight
        denom = max(base_pass_rate_all, 0.1)
        relative_change = avg_pass_rate_change / denom
        # decision boundary 越平缓，变化越小，稳定性越高
        param_stability = 1.0 / (1.0 + 10 * relative_change)

        # ❗ Bug B 修复: 结构性惩罚 - 极端不对称 gate 不应被当成稳定结构
        # 当 pass_rate_bad 极低时，这是 "全杀 bad" 型 knife-edge gate
        # 看起来稳定，其实是因为已经没有 bad 可以 pass 了
        if base_pass_rate_bad < 0.05:
            param_stability *= 0.7
    else:
        param_stability = 1.0

    # =========================================================================
    # 2. 时间稳定性 (Temporal Stability)
    # 核心问题：前后半段数据的 pass_rate 是否一致？
    # ❗ 问题 3 修复: multi-asset 场景用 per-symbol 计算，然后取中位数
    # =========================================================================
    temporal_stability = 1.0
    temporal_stability_valid = True  # 标记这个 score 是否可信

    if len(df) >= 100:
        # multi-asset 检测
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            # ❗ 问题 3 修复: per-symbol temporal stability
            # ❗ 问题 5 修复: 使用样本数加权平均，避免小币与 BTC 权重相同
            per_symbol_scores = []
            per_symbol_weights = []  # 样本数作为权重
            for sym, sym_df in df.groupby("symbol"):
                if len(sym_df) >= 50:  # 每个 symbol 至少 50 个样本
                    if "timestamp" in sym_df.columns:
                        sym_df = sym_df.sort_values("timestamp")
                    mid_point = len(sym_df) // 2
                    df_first = sym_df.iloc[:mid_point]
                    df_second = sym_df.iloc[mid_point:]

                    m1 = compute_lift_for_threshold(
                        df_first, feature_col, operator, threshold, label_col
                    )
                    m2 = compute_lift_for_threshold(
                        df_second, feature_col, operator, threshold, label_col
                    )

                    diff = abs(m1["pass_rate_all"] - m2["pass_rate_all"]) + 0.5 * (
                        abs(m1["pass_rate_good"] - m2["pass_rate_good"])
                        + abs(m1["pass_rate_bad"] - m2["pass_rate_bad"])
                    )

                    # ❗ Bug C 修复: temporal diff 加入 feature coverage drift
                    # 如果前后半段 NaN 比例不同，说明特征覆盖率不稳
                    valid_ratio_1 = (
                        m1["n_valid"] / len(df_first) if len(df_first) > 0 else 0
                    )
                    valid_ratio_2 = (
                        m2["n_valid"] / len(df_second) if len(df_second) > 0 else 0
                    )
                    valid_ratio_diff = abs(valid_ratio_1 - valid_ratio_2)
                    diff += 0.5 * valid_ratio_diff  # 加入 coverage drift 惩罚

                    score = 1.0 / (1.0 + 5 * diff)
                    per_symbol_scores.append(score)
                    per_symbol_weights.append(len(sym_df))

            if per_symbol_scores:
                # ❗ 问题 5 修复: 加权平均，大币种权重更高
                temporal_stability = float(
                    np.average(per_symbol_scores, weights=per_symbol_weights)
                )
            else:
                temporal_stability = 0.5  # 没有足够样本的 symbol
                temporal_stability_valid = False
        else:
            # 单资产: 确保按时间排序
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
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

            # ❗ Bug C 修复: temporal diff 加入 feature coverage drift
            # 如果前后半段 NaN 比例不同，说明特征覆盖率不稳
            valid_ratio_first = (
                metrics_first["n_valid"] / len(df_first_half)
                if len(df_first_half) > 0
                else 0
            )
            valid_ratio_second = (
                metrics_second["n_valid"] / len(df_second_half)
                if len(df_second_half) > 0
                else 0
            )
            valid_ratio_diff = abs(valid_ratio_first - valid_ratio_second)
            combined_temporal_diff += 0.5 * valid_ratio_diff  # coverage drift 惩罚

            temporal_stability = 1.0 / (1.0 + 5 * combined_temporal_diff)

    # =========================================================================
    # 3. 样本效率 (Sample Efficiency)
    # 核心问题：有足够的 good/bad 样本支撑这个判定吗？
    # ❗ Bug 2 修复: 使用 valid_good/valid_bad（参与 gate 的样本数）
    # 而不是全样本数，避免低覆盖率 feature 被误判为"样本充分"
    # =========================================================================
    n_good = base_metrics.get("valid_good", base_metrics["n_good"])
    n_bad = base_metrics.get("valid_bad", base_metrics["n_bad"])
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
    actual_step: float = None,  # ❗ 问题 4 修复: 传入实际扫描步长
) -> Optional[Dict[str, Any]]:
    """
    找到稳定的lift平台区间 - 改进版，基于稳定性而非连续性

    Args:
        results: 扫描结果列表（应传入已筛选的 valid_results，避免在不可执行域上建立 plateau）
        config: 配置对象
        actual_step: 实际扫描步长（用于 plateau 连续性判断）

    Returns:
        稳定平台信息，包含 start, end, mid, metrics 等
    """
    if not results:
        return None

    # ❗ Bug fix: 过滤 NaN lift，避免污染 plateau 搜索
    # NaN >= anything 是 False，但 np.mean([..., NaN]) = NaN
    results = [r for r in results if np.isfinite(r.get("lift", float("nan")))]
    if not results:
        return None

    # 按阈值排序
    results_sorted = sorted(results, key=lambda x: x["threshold"])

    # ❗ 问题 4 修复: 使用实际步长，避免与 config.threshold_step 不一致
    step_for_continuity = (
        actual_step if actual_step is not None else config.threshold_step
    )

    # 筛选满足基础条件的阈值
    # 注意：如果调用方已经传入 valid_results，这里会重复筛选（但不会出错）
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
        # ❗ Bug fix #2: 使用 anchor_lift 而不是 rolling current_lift
        # 避免 plateau 结果依赖扫描顺序
        anchor_lift = valid_thresholds[i]["lift"]
        current_threshold = valid_thresholds[i]["threshold"]

        # 寻找稳定区间
        j = i + 1
        while j < len(valid_thresholds):
            next_threshold = valid_thresholds[j]["threshold"]
            next_lift = valid_thresholds[j]["lift"]

            # 检查阈值连续性（相邻阈值间隔不能过大）
            # ❗ 问题 4 修复: 使用实际步长
            if next_threshold - current_threshold > step_for_continuity * 2:
                break

            # =========================================================================
            # ❗ 问题 2 修复: plateau 加入 pass_rate 稳定性条件
            # 实盘真正炸的不是 lift 波动，而是 pass/fail 边界抖动
            # =========================================================================
            anchor_pass_rate = valid_thresholds[start_idx]["pass_rate_all"]
            pass_rate_change = abs(
                valid_thresholds[j]["pass_rate_all"] - anchor_pass_rate
            )
            if pass_rate_change > 0.15:  # pass_rate 变化超过 15% 就认为不稳定
                break

            # ❗ Bug fix #2: lift稳定性基于 anchor，不是 rolling
            # 这样 plateau 是区间性质，不是路径性质
            lift_change = abs(next_lift - anchor_lift)
            if lift_change > config.max_lift_std_ratio * abs(anchor_lift):
                break

            # ❗ Bug 2 修复: plateau 要求 coverage 稳定，不仅仅是 decision 稳定
            # 如果 n_valid 在不同阈值之间变化过大，说明 coverage 不稳定
            anchor_n_valid = valid_thresholds[start_idx].get("n_valid", 0)
            current_n_valid = valid_thresholds[j].get("n_valid", 0)
            if anchor_n_valid > 0:
                coverage_change = abs(current_n_valid - anchor_n_valid) / anchor_n_valid
                if coverage_change > 0.1:  # coverage 变化超过 10% 就认为不稳定
                    break

            current_threshold = next_threshold
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

            # ❗ Bug fix #6: 使用相对宽度，避免不同 feature 的 plateau 宽度不可比
            # threshold_range 会在上层传入，这里使用 valid_thresholds 的范围作为近似
            if len(valid_thresholds) > 1:
                total_range = (
                    valid_thresholds[-1]["threshold"] - valid_thresholds[0]["threshold"]
                )
                relative_width = (
                    interval_width / total_range if total_range > 0 else 0.0
                )
            else:
                relative_width = 0.0

            # ❗ 问题 3 修复: 使用相对宽度判断，避免不同 feature 的 plateau 宽度不可比
            # 绝对宽度仍然保留作为下限（防止极端情况）
            if (
                relative_width >= config.min_plateau_width
                or interval_width >= config.min_plateau_width
            ):
                stable_intervals.append(
                    {
                        "interval": interval,
                        "start_idx": start_idx,
                        "end_idx": j - 1,
                        "start_threshold": interval[0]["threshold"],
                        "end_threshold": interval[-1]["threshold"],
                        "width": interval_width,
                        "relative_width": relative_width,  # ❗ Bug fix #6
                        "lift_mean": lift_mean,
                        "lift_std": lift_std,
                        "lift_min": lift_min,
                        "lift_max": lift_max,
                        "lift_stability_ratio": (
                            lift_std / max(lift_mean, 0.2)
                            if lift_mean != 0
                            else float("inf")
                        ),  # ❗ Bug 3 修复: 使用 floor 避免弱 lift 平台被过度惩罚
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
        "recommended_threshold_type": "plateau_mid",  # ❗ Bug fix #3: 显式语义
        "lift_mean": best_interval["lift_mean"],
        "lift_std": best_interval["lift_std"],
        "lift_min": best_interval["lift_min"],
        "lift_max": best_interval["lift_max"],
        "lift_stability_ratio": best_interval["lift_stability_ratio"],
        "pass_rate_at_mid": mid_metrics["pass_rate_all"],
        "lift_at_mid": mid_metrics["lift"],
        "plateau_width": best_interval["width"],
        "plateau_relative_width": best_interval.get(
            "relative_width", 0.0
        ),  # ❗ Bug fix #6
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
            # ❗ 问题 6 确认: quantile_* 语义假设
            # 当前实现假设 feature 已经是预计算的 quantile score (0-1)
            # 即 quantile_gt 0.8 意思是 "feature_value > 0.8"
            # 而不是 "feature_value > df[feature].quantile(0.8)"
            # 如果 feature 是原始值，需要在特征工程阶段先转为 quantile
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
        # =========================================================================
        # Hard Gate NaN Lift 例外通道
        # 语义：Hard Gate 的合法性来源是"结构稳定性 + 执行风险厌恶"，不是 lift
        # 只有同时满足所有严格条件，才允许 lift=NaN 的 Hard Gate 进入 robustness 仲裁
        # =========================================================================
        if config.allow_hard_nan_lift and rule.tag and "gate_" in rule.id:
            # 检查是否有符合 NaN lift 例外条件的阈值
            nan_lift_candidates = []
            for r in results:
                # 条件 1: lift 是 NaN（pass_rate_bad 极低）
                if not np.isfinite(r.get("lift", 0)):
                    # 条件 2: pass_rate_bad 必须极低
                    if r.get("pass_rate_bad", 1.0) > config.nan_lift_max_pass_rate_bad:
                        continue
                    # 条件 3: pass_rate_good 必须足够高
                    if r.get("pass_rate_good", 0) < config.nan_lift_min_pass_rate_good:
                        continue
                    # 条件 4: 覆盖率必须足够
                    n_valid = r.get("n_valid", 0)
                    n_all = len(df)
                    coverage = n_valid / n_all if n_all > 0 else 0
                    if coverage < config.nan_lift_min_coverage:
                        continue
                    nan_lift_candidates.append(r)

            if nan_lift_candidates:
                # 在候选中找 robustness 最高的
                best_nan_result = None
                best_nan_robustness = -1

                for r in nan_lift_candidates:
                    robustness = compute_robustness_score(
                        df, feature_col, operator, r["threshold"], label_col, config
                    )
                    # 条件 5: robustness 必须足够高
                    if robustness.overall_score >= config.nan_lift_min_robustness:
                        if robustness.overall_score > best_nan_robustness:
                            best_nan_robustness = robustness.overall_score
                            best_nan_result = {
                                **r,
                                "robustness_score": robustness.to_dict(),
                                "recommended_threshold": r["threshold"],
                                "recommended_threshold_type": "robust_but_unproven",
                            }

                if best_nan_result:
                    # 条件 6: 检查阈值区间宽度（排除点估计）
                    # 找到所有通过条件的阈值范围
                    valid_thresholds = [
                        r["threshold"]
                        for r in nan_lift_candidates
                        if compute_robustness_score(
                            df, feature_col, operator, r["threshold"], label_col, config
                        ).overall_score
                        >= config.nan_lift_min_robustness
                    ]
                    if len(valid_thresholds) >= 2:
                        interval_width = max(valid_thresholds) - min(valid_thresholds)
                        if interval_width >= config.nan_lift_min_plateau_width:
                            return {
                                "rule_id": rule.id,
                                "feature": feature_col,
                                "operator": operator,
                                "current_threshold": current_threshold,
                                "status": "robust_but_unproven",  # 明确标记语义
                                "eligibility": "deny_only",  # 只能作为 deny-only safety gate
                                "robustness_selection": True,
                                "nan_lift_exception": True,
                                "nan_lift_reason": "Hard Gate with pass_rate_bad < 1%, structural stability validated",
                                "interval_width": interval_width,
                                **best_nan_result,
                                "scan_results": results,
                            }

        return {
            "rule_id": rule.id,
            "feature": feature_col,
            "status": "no_valid_threshold",
            "reason": "No threshold meets basic lift/pass_rate requirements",
            "scan_results": results,
        }

    # 寻找稳定的平台区间
    # ❗ Bug fix: 必须传入 valid_results，不能在不可执行域上建立 plateau
    # ❗ 问题 4 修复: 传入实际步长，保证连续性判断一致
    stable_plateau = find_stable_lift_plateau(valid_results, config, actual_step=step)

    if stable_plateau is None:
        # ❗ 设计决策点：Hard gate 没有 plateau 时是否允许 fallback
        # strict_hard=True: 不允许 fallback，Hard gate 必须有结构支撑
        # strict_hard=False: 允许 fallback 到 robustness 单点（过渡态）
        if config.strict_hard:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "operator": operator,
                "current_threshold": current_threshold,
                "status": "no_stable_plateau_strict",
                "reason": "Hard gate requires stable plateau (strict mode enabled)",
                "scan_results": results,
            }

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
                    "recommended_threshold_type": "robust_fallback",  # ❗ Bug fix #3
                }

        if best_result:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "operator": operator,
                "current_threshold": current_threshold,
                "status": "no_stable_plateau",
                "robustness_selection": True,
                "fallback_warning": "Hard gate (weak mode): robustness fallback enabled",
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
                "recommended_threshold_type": "plateau_best",  # ❗ Bug fix #3
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
                recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3
            else:
                recommended_threshold = stable_plateau["plateau_mid"]
                recommended_threshold_type = "plateau_mid"  # ❗ Bug fix #3
                best_result_in_plateau = {
                    **mid_result,
                    "robustness_score": mid_robustness.to_dict(),
                    "recommended_threshold": stable_plateau["plateau_mid"],
                    "recommended_threshold_type": "plateau_mid",
                }
        else:
            # 如果找不到mid附近的结果，使用best_result_in_plateau
            recommended_threshold = best_result_in_plateau["threshold"]
            recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3
    else:
        recommended_threshold = best_result_in_plateau["threshold"]
        recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3

    return {
        "rule_id": rule.id,
        "feature": feature_col,
        "operator": operator,
        "current_threshold": current_threshold,
        "status": "stable_plateau_found",
        "robustness_selection": True,
        **stable_plateau,
        "recommended_threshold": recommended_threshold,
        "recommended_threshold_type": recommended_threshold_type,  # ❗ Bug fix #3
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


# Optimization statuses that are considered "validated" for promotion
_VALID_OPT_STATUSES = {
    "stable_plateau_found",
    "no_stable_plateau",  # robustness fallback is still validated
    "nan_lift_hard_gate",  # NaN-lift exception for hard gates
}

# Prefilter operator → gate deny operator (直接映射, operator 本身已是 deny 方向)
# 分析脚本中 operator="<" 表示 deny when col < threshold,
# 因此 gate deny 条件应保持相同方向: value_lt → deny when col < threshold
_PREFILTER_OP_MAP = {
    ">=": "value_ge",  # deny when col >= X
    ">": "value_gt",  # deny when col > X
    "<=": "value_le",  # deny when col <= X
    "<": "value_lt",  # deny when col < X
}


def _load_prefilter_as_frozen_gates(prefilter_path: Path) -> List[Dict]:
    """
    将 prefilter 规则转换为 frozen hard_gates (deny 格式)。

    确保 prefilter 条件在推理时作为 gate 的一部分执行一次，
    保障训练-推理数据分布一致性。
    frozen=true 的规则不可被优化器移除或修改阈值。
    """
    import yaml as _yaml

    if not prefilter_path.exists():
        return []

    raw = _yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}
    rules = raw.get("rules", [])
    if not rules:
        return []

    gate_rules: List[Dict] = []
    for rule in rules:
        if "any_of" in rule:
            # OR 规则: deny when ALL sub-conditions are NOT met
            # allow = (A OR B) → deny = (NOT A AND NOT B)
            all_of_items = []
            features_desc = []
            for sub in rule["any_of"]:
                feat = sub.get("feature", "")
                op = sub.get("operator", "")
                val = sub.get("value")
                gate_op = _PREFILTER_OP_MAP.get(op)
                if gate_op and feat and val is not None:
                    all_of_items.append({feat: {gate_op: val}})
                    features_desc.append(f"{feat}{op}{val}")

            if all_of_items:
                feats_short = "_".join(
                    "_".join(f.split("_")[:2]) for item in all_of_items for f in item
                )
                gate_rules.append(
                    {
                        "id": f"prefilter_{feats_short}",
                        "tag": f"PREFILTER_{feats_short.upper()}",
                        "phase": "hard_gate",
                        "priority": 1,
                        "reason": f"prefilter OR: {' OR '.join(features_desc)}",
                        "when": {"all_of": all_of_items},
                        "then": {"action": "deny"},
                        "frozen": True,
                        "comment": "prefilter条件 (训练-推理一致性, frozen=true)",
                    }
                )
        else:
            # 简单 AND 规则 → 每条转为单独的 frozen hard_gate
            feat = rule.get("feature", "")
            op = rule.get("operator", "")
            val = rule.get("value")
            gate_op = _PREFILTER_OP_MAP.get(op)

            if gate_op and feat and val is not None:
                gate_rules.append(
                    {
                        "id": f"prefilter_{feat}",
                        "tag": f"PREFILTER_{feat.upper()}",
                        "phase": "hard_gate",
                        "priority": 1,
                        "reason": f"prefilter: {feat} {op} {val}",
                        "when": {feat: {gate_op: val}},
                        "then": {"action": "deny"},
                        "frozen": True,
                        "comment": "prefilter条件 (训练-推理一致性, frozen=true)",
                    }
                )

    return gate_rules


def _promote_gate_to_archetypes(
    strategy: str,
    strategies_root: str,
    arch: "StrategyArchetype",
    optimization_results: Dict[str, Any],
    source_gate_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    min_combined_pass_rate: float = 0.05,
) -> None:
    """
    将优化后的 gate 规则写入 archetypes/gate.yaml。

    读取源 gate YAML (草稿或现有 gate.yaml)，用优化结果更新阈值，
    写入 archetypes/gate.yaml。

    关键行为:
      - 优化失败的规则 (no_valid_threshold/skip) 会被移除
      - 累积 AND pass rate 过低时自动裁剪最弱规则 (防止全部 veto)
    """
    import yaml

    root = Path(strategies_root)
    arch_dir = root / strategy / "archetypes"
    target_path = arch_dir / "gate.yaml"

    # ── 语义锁定通过 gate 规则的 frozen: true 字段实现 ──
    # prefilter.yaml 中 locked: true 的规则会被 _load_prefilter_as_frozen_gates()
    # 转换为 frozen: true 的 hard_gate, 优化器对 frozen 规则跳过阈值优化 (opt=None),
    # 从而在下方 "if not opt: kept_rules.append(rule)" 路径被自动保留, 无需 meta.yaml.

    # 读取源 YAML (草稿或现有 gate.yaml)
    if source_gate_path:
        source = Path(source_gate_path)
    else:
        source = target_path

    if not source.exists():
        print(f"\u26a0\ufe0f  Cannot promote: source gate not found: {source}")
        return

    raw_text = source.read_text(encoding="utf-8")
    config = yaml.safe_load(raw_text) or {}

    # ── Filter hard_gates: only keep rules that passed optimization ──
    hard_gates = config.get("hard_gates", [])
    kept_rules = []
    removed_rules = []
    updated_count = 0

    for rule in hard_gates:
        rule_id = rule.get("id", "")
        opt = optimization_results.get(rule_id)

        if not opt:
            # No optimization result for this rule → keep as-is (e.g. frozen)
            kept_rules.append(rule)
            continue

        status = opt.get("status", "")
        rec = opt.get("recommended_threshold")

        if status not in _VALID_OPT_STATUSES or rec is None:
            # Optimization failed → remove from production gate
            # (frozen 规则已在 "if not opt" 分支被保留, 不会到达这里)
            removed_rules.append(
                {
                    "id": rule_id,
                    "status": status,
                    "reason": opt.get("reason", "unknown"),
                }
            )
            continue

        # Update threshold from optimization result
        when = rule.get("when", {})
        for feature, conditions in when.items():
            if isinstance(conditions, dict):
                for cond_key in list(conditions.keys()):
                    if cond_key.startswith("value_"):
                        conditions[cond_key] = round(rec, 4)
                        updated_count += 1

        # Add optimization metadata to comment
        lift = opt.get("lift_at_mid", opt.get("lift"))
        rule["comment"] = (
            f"optimizer: {status}, " f"threshold={rec:.4f}, " f"lift={lift:.3f}"
            if isinstance(lift, (int, float))
            else f"optimizer: {status}"
        )
        kept_rules.append(rule)

    # ── 注入 prefilter 条件为 frozen hard_gates (训练-推理一致性) ──
    prefilter_path = arch_dir / "prefilter.yaml"
    prefilter_gates = _load_prefilter_as_frozen_gates(prefilter_path)
    if prefilter_gates:
        print(
            f"\n  \U0001f512 注入 {len(prefilter_gates)} 条 prefilter 为 frozen hard_gates"
        )
        for pg in prefilter_gates:
            print(f"     - {pg['id']}: {pg['reason']}")

    # 合并: prefilter (frozen) + optimized gates
    # 去重: 同 id 只保留第一条 (防止 gate_draft 中同一特征多次分裂导致重复)
    _seen_ids: set = set()
    deduped_rules: list = []
    for rule in kept_rules:
        rid = rule.get("id", "")
        if rid in _seen_ids:
            removed_rules.append(
                {"id": rid, "status": "duplicate", "reason": f"duplicate of {rid}"}
            )
            continue
        _seen_ids.add(rid)
        deduped_rules.append(rule)
    kept_rules = deduped_rules
    all_rules = prefilter_gates + kept_rules
    config["hard_gates"] = all_rules

    # ── 累积 AND pass rate 模拟: 防止多条规则组合后 pass rate 过低 ──
    if df is not None and all_rules and min_combined_pass_rate > 0:
        import operator as op_module

        _GATE_OPS = {
            "value_lt": op_module.lt,
            "value_le": op_module.le,
            "value_gt": op_module.gt,
            "value_ge": op_module.ge,
        }

        def _apply_when_to_mask(when, data, allow_mask, gate_ops):
            """Apply a when clause to allow_mask. Handles simple + all_of."""
            if "all_of" in when:
                # all_of: deny when ALL sub-conditions match (AND)
                compound_deny = pd.Series(True, index=data.index)
                for sub in when["all_of"]:
                    if isinstance(sub, dict):
                        for feat, conds in sub.items():
                            if isinstance(conds, dict):
                                for ck, th in conds.items():
                                    op_func = gate_ops.get(ck)
                                    if op_func and feat in data.columns:
                                        compound_deny &= op_func(data[feat], th)
                allow_mask &= ~compound_deny
            else:
                for feature, conditions in when.items():
                    if isinstance(conditions, dict):
                        for cond_key, threshold in conditions.items():
                            op_func = gate_ops.get(cond_key)
                            if op_func and feature in data.columns:
                                deny_mask = op_func(data[feature], threshold)
                                allow_mask &= ~deny_mask
            return allow_mask

        def _simulate_combined_pass_rate(rules, data):
            """Simulate cumulative AND pass rate for all rules (incl. prefilter)."""
            allow_mask = pd.Series(True, index=data.index)
            for rule in rules:
                when = rule.get("when", {})
                allow_mask = _apply_when_to_mask(when, data, allow_mask, _GATE_OPS)
            n_allow = allow_mask.sum()
            return n_allow / len(data) if len(data) > 0 else 0.0

        combined_rate = _simulate_combined_pass_rate(all_rules, df)
        n_allow = int(combined_rate * len(df))
        n_prefilter = len(prefilter_gates)
        n_opt = len(kept_rules)
        print(
            f"\n  📊 累积 AND pass rate: {combined_rate:.1%} "
            f"({n_allow}/{len(df)}, {n_prefilter} prefilter + {n_opt} optimized)"
        )

        # ── Interaction Screening + Bell Partition ──
        n_rules = len(all_rules)
        rr_col_gate = None
        for cand in ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]:
            if cand in df.columns:
                rr_col_gate = cand
                break

        def _get_rule_deny_mask(rule, data, gate_ops):
            """Get deny mask for a single gate rule."""
            allow = pd.Series(True, index=data.index)
            when = rule.get("when", {})
            allow = _apply_when_to_mask(when, data, allow, gate_ops)
            return ~allow

        interaction_map = {}
        bell_applied = False

        if n_rules >= 2 and rr_col_gate is not None:
            rr_arr = df[rr_col_gate].values.astype(float)

            # Compute individual deny masks
            deny_masks = []
            for rule in all_rules:
                dm = _get_rule_deny_mask(rule, df, _GATE_OPS)
                deny_masks.append(dm.values)

            # ── Interaction Screening (2×2 Uplift Interaction Test) ──
            print(f"\n  📊 Gate Interaction Screening ({n_rules} rules)")
            print(
                f"   {'Pair':<55s} {'r00':>7s} {'r10':>7s} "
                f"{'r01':>7s} {'r11':>7s} {'type':>14s}"
            )
            print(f"   {'-'*105}")

            for i in range(n_rules):
                for j in range(i + 1, n_rules):
                    dA = deny_masks[i]
                    dB = deny_masks[j]
                    g00 = ~dA & ~dB
                    g10 = dA & ~dB
                    g01 = ~dA & dB
                    g11 = dA & dB

                    r00 = float(np.nanmean(rr_arr[g00])) if g00.sum() > 10 else 0
                    r10 = float(np.nanmean(rr_arr[g10])) if g10.sum() > 10 else 0
                    r01 = float(np.nanmean(rr_arr[g01])) if g01.sum() > 10 else 0
                    r11 = float(np.nanmean(rr_arr[g11])) if g11.sum() > 10 else 0

                    additive = r10 + r01 - r00
                    delta = r11 - additive

                    fA = all_rules[i].get("id", f"rule_{i}")
                    fB = all_rules[j].get("id", f"rule_{j}")

                    if g11.sum() < 10:
                        itype = "insufficient"
                    elif abs(delta) < 0.05 * max(abs(r10 - r00), abs(r01 - r00), 0.01):
                        itype = "independent"
                    elif r11 > max(r10, r01) and delta > 0:
                        itype = "synergistic"
                    elif abs(r11 - max(r10, r01)) < 0.05 * max(
                        abs(r10), abs(r01), 0.01
                    ):
                        itype = "substitutive"
                    elif r11 < min(r10, r01):
                        itype = "antagonistic"
                    else:
                        itype = "substitutive"

                    interaction_map[(i, j)] = {
                        "type": itype,
                        "delta": delta,
                        "r00": r00,
                        "r10": r10,
                        "r01": r01,
                        "r11": r11,
                    }
                    pair_name = f"{fA} × {fB}"
                    print(
                        f"   {pair_name:<55s} {r00:>+6.3f} {r10:>+6.3f} "
                        f"{r01:>+6.3f} {r11:>+6.3f} {itype:>14s}"
                    )

            # ── Bell Partition: search optimal AND/OR structure ──
            def _bell_partitions_gate(items):
                """Generate all Bell partitions. N=2→2, N=3→5, N=4→15."""
                if len(items) <= 1:
                    yield [items]
                    return
                first = items[0]
                rest = items[1:]
                for partition in _bell_partitions_gate(rest):
                    yield [[first]] + partition
                    for bi in range(len(partition)):
                        new_part = [g[:] for g in partition]
                        new_part[bi] = [first] + new_part[bi]
                        yield new_part

            def _interaction_penalty_gate(partition, imap):
                """Penalty for partition violating interaction structure."""
                penalty = 0.0
                for group in partition:
                    for a in range(len(group)):
                        for b in range(a + 1, len(group)):
                            key = (
                                min(group[a], group[b]),
                                max(group[a], group[b]),
                            )
                            info = imap.get(key, {})
                            if info.get("type") == "synergistic":
                                penalty += 0.15
                            elif info.get("type") == "antagonistic":
                                penalty += 0.30
                all_groups = partition
                for gi in range(len(all_groups)):
                    for gj in range(gi + 1, len(all_groups)):
                        for a in all_groups[gi]:
                            for b in all_groups[gj]:
                                key = (min(a, b), max(a, b))
                                info = imap.get(key, {})
                                if info.get("type") == "substitutive":
                                    penalty += 0.10
                                elif info.get("type") == "independent":
                                    penalty += 0.05
                return penalty

            def _eval_gate_partition(partition, dmasks, rr, imap, min_pr):
                """Evaluate Bell Partition for gate rules.
                Within group: AND deny (both must deny) = OR pass.
                Between groups: OR deny (any group denies) = AND pass."""
                combined_deny = np.zeros(len(rr), dtype=bool)
                for group in partition:
                    group_deny = np.ones(len(rr), dtype=bool)
                    for idx in group:
                        group_deny = group_deny & dmasks[idx]
                    combined_deny = combined_deny | group_deny

                pass_mask = ~combined_deny
                n_p = int(pass_mask.sum())
                n_d = int(combined_deny.sum())
                if n_p < 30 or n_d < 10:
                    return None
                pr = n_p / len(rr)
                if pr < min_pr:
                    return None

                rr_p = rr[pass_mask]
                rr_d = rr[combined_deny]
                effect = float(np.nanmean(rr_p)) - float(np.nanmean(rr_d))
                baseline_rr = float(np.nanmean(rr))
                deny_rr = float(np.nanmean(rr_d))
                tail_cap = max(baseline_rr - deny_rr, 0)

                _log_pr = float(np.log(max(pr, 0.01)))
                i_pen = _interaction_penalty_gate(partition, imap)
                _base = max(abs(baseline_rr), 0.01)
                score = (
                    0.35 * min(tail_cap / _base, 1.0)
                    + 0.30 * max(effect, 0) / _base
                    + 0.20 * (_log_pr + 3) / 3
                    - i_pen
                )
                return {
                    "pass_rate": pr,
                    "effect": effect,
                    "tail_capture": tail_cap,
                    "score": score,
                    "deny_mask": combined_deny,
                    "partition": partition,
                    "penalty": i_pen,
                }

            indices = list(range(n_rules))
            all_partitions = list(_bell_partitions_gate(indices))

            # Gate 约束:
            # 1. partition 必须 ≥2 组 (单组 = 全 OR-pass = 无 gate)
            # 2. pass_rate ≤ 85% (gate 必须有实际筛选效果)
            GATE_MAX_PASS_RATE = 0.85
            part_results = []
            for part in all_partitions:
                if len(part) < 2:
                    continue  # 单组 = 取消 gate, 不允许
                res = _eval_gate_partition(
                    part,
                    deny_masks,
                    rr_arr,
                    interaction_map,
                    min_combined_pass_rate,
                )
                if res is not None and res["pass_rate"] <= GATE_MAX_PASS_RATE:
                    part_results.append(res)

            if part_results:
                part_results.sort(key=lambda x: -x["score"])
                best = part_results[0]

                def _fmt_gate_partition(part, rules):
                    groups = []
                    for g in part:
                        names = [rules[i].get("id", f"r{i}") for i in g]
                        if len(names) == 1:
                            groups.append(names[0])
                        else:
                            groups.append("(" + " ∨ ".join(names) + ")")
                    return " ∧ ".join(groups)

                print(
                    f"\n  📊 Gate Bell Partition "
                    f"({len(all_partitions)} structures, "
                    f"{len(part_results)} pass min_pr≥{min_combined_pass_rate:.0%})"
                )
                print(
                    f"   score = 0.35×tail_cap + 0.30×effect "
                    f"+ 0.20×log(pr) - penalty"
                )
                for pi, r in enumerate(part_results[:5]):
                    marker = " ← BEST" if pi == 0 else ""
                    pen_str = f" pen={r['penalty']:.2f}" if r["penalty"] > 0 else ""
                    print(
                        f"   {pi+1}. "
                        f"{_fmt_gate_partition(r['partition'], all_rules)}"
                        f"  effect={r['effect']:+.4f} "
                        f"tail={r['tail_capture']:.4f} "
                        f"pass={r['pass_rate']:.1%} "
                        f"score={r['score']:.4f}{pen_str}{marker}"
                    )

                best_part = best["partition"]
                is_pure_and = all(len(g) == 1 for g in best_part)

                if not is_pure_and and best["pass_rate"] >= min_combined_pass_rate:
                    # Restructure: merge OR-pass groups into compound gates
                    print(
                        f"\n  ✅ 最优结构非 pure-AND, "
                        f"重构 gate 规则 (OR-pass groups)..."
                    )
                    new_rules = []
                    for gid, group in enumerate(best_part):
                        if len(group) == 1:
                            new_rules.append(all_rules[group[0]])
                        else:
                            # Merge into single compound gate (AND-deny)
                            merged_when = {"all_of": []}
                            merged_ids = []
                            for idx in group:
                                r = all_rules[idx]
                                when = r.get("when", {})
                                if "all_of" in when:
                                    merged_when["all_of"].extend(when["all_of"])
                                else:
                                    for feat, conds in when.items():
                                        merged_when["all_of"].append({feat: conds})
                                merged_ids.append(r.get("id", f"r{idx}"))

                            grp_rules = [all_rules[i] for i in group]
                            merged_rule = {
                                "id": "bell_or_pass_" + "_".join(merged_ids),
                                "tag": f"BELL_OR_PASS_{gid}",
                                "phase": "hard_gate",
                                "priority": min(
                                    r.get("priority", 10) for r in grp_rules
                                ),
                                "reason": (
                                    "Bell Partition OR-pass: " + " ∨ ".join(merged_ids)
                                ),
                                "when": merged_when,
                                "then": {"action": "deny"},
                                "comment": (
                                    "OR-pass: deny only when ALL of "
                                    f"[{', '.join(merged_ids)}] "
                                    "deny simultaneously"
                                ),
                            }
                            if any(all_rules[i].get("frozen") for i in group):
                                merged_rule["frozen"] = True
                            new_rules.append(merged_rule)
                            print(
                                f"     🔗 合并 {merged_ids} "
                                f"→ OR-pass group (AND-deny)"
                            )

                    all_rules = new_rules
                    config["hard_gates"] = all_rules
                    combined_rate = best["pass_rate"]
                    bell_applied = True
                    print(
                        f"  ✅ Bell Partition 重构: "
                        f"{len(new_rules)} gates, "
                        f"pass rate={combined_rate:.1%}"
                    )
                elif is_pure_and:
                    print(f"\n  ✅ 最优结构 = pure-AND, 保持原样")

        # ── Fallback: interaction-aware 裁剪 ──
        if not bell_applied and combined_rate < min_combined_pass_rate:
            print(
                f"  ⚠️  累积 pass rate {combined_rate:.1%} < "
                f"下限 {min_combined_pass_rate:.0%}"
            )

            # Build interaction-aware priority for each rule
            # Substitutive rules are cheaper to remove (redundant)
            print(f"  🔧 Interaction-aware 裁剪...")
            prunable = []
            for rule in kept_rules:
                if rule.get("frozen"):
                    continue
                rule_id = rule.get("id", "")
                opt = optimization_results.get(rule_id, {})
                lift = opt.get("lift_at_mid", opt.get("lift", 0))
                lv = lift if isinstance(lift, (int, float)) else 0

                # Count substitutive relationships for this rule
                ri_all = len(prefilter_gates) + kept_rules.index(rule)
                sub_count = 0
                for (a, b), info in interaction_map.items():
                    if ri_all in (a, b):
                        if info.get("type") == "substitutive":
                            sub_count += 1
                # Lower = remove first (substitutive + low lift)
                prune_priority = abs(lv) - sub_count * 0.05
                prunable.append((rule, prune_priority))
            prunable.sort(key=lambda x: x[1])

            remaining_opt = [r for r, _ in prunable]
            pruned_ids = []

            while remaining_opt and combined_rate < min_combined_pass_rate:
                weakest = remaining_opt.pop(0)
                wid = weakest.get("id", "unknown")
                wpri = next((pv for r, pv in prunable if r is weakest), 0)
                pruned_ids.append(wid)
                test_rules = prefilter_gates + remaining_opt
                combined_rate = _simulate_combined_pass_rate(test_rules, df)
                print(
                    f"    ✂️  移除 {wid} (priority={wpri:.3f}) "
                    f"→ pass rate={combined_rate:.1%}"
                )

            kept_rules = remaining_opt

            # ── Phase 2: frozen prefilter gates ──
            if combined_rate < min_combined_pass_rate and len(prefilter_gates) > 1:
                print(
                    f"  🔧 Phase 2: optimized 裁完仍 "
                    f"{combined_rate:.1%}, "
                    f"裁剪 frozen prefilter gates "
                    f"(保留最强 1 条)..."
                )
                pf_prunable = list(reversed(prefilter_gates[1:]))
                for pf_rule in pf_prunable:
                    if combined_rate >= min_combined_pass_rate:
                        break
                    pfid = pf_rule.get("id", "unknown")
                    prefilter_gates = [g for g in prefilter_gates if g is not pf_rule]
                    pruned_ids.append(pfid)
                    test_rules = prefilter_gates + kept_rules
                    combined_rate = _simulate_combined_pass_rate(test_rules, df)
                    print(
                        f"    ✂️  移除 frozen {pfid} "
                        f"→ pass rate={combined_rate:.1%} "
                        f"(剩余 {len(prefilter_gates)} prefilter)"
                    )

            all_rules = prefilter_gates + kept_rules
            config["hard_gates"] = all_rules

            if pruned_ids:
                removed_rules.extend(
                    {
                        "id": rid,
                        "status": "cumulative_pruned",
                        "reason": (f"AND pass rate " f"< {min_combined_pass_rate:.0%}"),
                    }
                    for rid in pruned_ids
                )
                print(
                    f"  ✅ 裁剪后: {len(prefilter_gates)} prefilter"
                    f" + {len(kept_rules)} optimized, "
                    f"pass rate={combined_rate:.1%}"
                )
        elif not bell_applied:
            print(
                f"  ✅ 累积 pass rate {combined_rate:.1%} >= "
                f"下限 {min_combined_pass_rate:.0%}, 无需裁剪"
            )

    # Report removed rules
    if removed_rules:
        print(
            f"\n  \u26a0\ufe0f  {len(removed_rules)} 条规则优化失败, 已从 gate.yaml 移除:"
        )
        for rm in removed_rules:
            print(f"     - {rm['id']}: {rm['status']} ({rm['reason']})")

    if not kept_rules:
        print(
            f"\n  \u274c  所有规则优化失败, gate.yaml 将只含 guardrails (无 hard_gates)"
        )
        print(f"     建议: 检查训练数据量是否充足, 或放宽优化参数")

    # 写入 archetypes/gate.yaml
    n_total = len(hard_gates)
    header = (
        f"# {strategy.upper()} Gate (optimized, auto-promoted)\n"
        f"# 来源: {source}\n"
        f"# 优化规则: {updated_count}/{n_total} 条通过优化"
        f"{f', {len(removed_rules)} 条已移除' if removed_rules else ''}\n\n"
    )
    yaml_content = yaml.dump(
        config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    target_path.write_text(header + yaml_content, encoding="utf-8")
    print(
        f"\n\U0001f4e6 Promoted to {target_path} ({updated_count} thresholds updated, "
        f"{len(kept_rules)} rules kept, {len(removed_rules)} removed)"
    )


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
    parser.add_argument(
        "--gate-path",
        default=None,
        help="Custom gate YAML path (e.g., config/strategies/fer/gate_draft.yaml). "
        "Default: archetypes/gate.yaml",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After optimization, write updated gate.yaml with optimized thresholds "
        "to archetypes/gate.yaml (promote draft to production)",
    )
    parser.add_argument(
        "--prefilter",
        default=None,
        help="Prefilter YAML path. If provided, filter logs by prefilter rules "
        "before optimization (ensures plateau validation on production distribution)",
    )
    parser.add_argument(
        "--min-combined-pass-rate",
        type=float,
        default=0.05,
        metavar="RATE",
        help="累积 AND pass rate 下限 (0~1). 多条 gate 规则组合后至少要保留这个比例的 bars. "
        "如果低于这个阈值, 会按 lift 从弱到强自动裁剪规则. 默认 0.05 (5%%). "
        "由 research_pipeline.yaml kpi_gates.gate.min_combined_pass_rate 控制.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Only use data before this date for optimization (IS cutoff, avoid OOS lookahead)",
    )
    args = parser.parse_args()

    # Load logs
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"✅ Loaded {len(df)} rows from {logs_path}")

    # Apply cutoff date (IS only — avoid OOS lookahead)
    if args.cutoff_date:
        _ts = "timestamp" if "timestamp" in df.columns else None
        if _ts is None and df.index.name == "timestamp":
            df = df.reset_index()
            _ts = "timestamp"
        if _ts:
            df[_ts] = pd.to_datetime(df[_ts])
            _n0 = len(df)
            df = df[df[_ts] < args.cutoff_date]
            print(f"   IS cutoff {args.cutoff_date}: {_n0} → {len(df)} rows")

    # ── Prefilter: 在生产分布上验证 plateau ──
    if args.prefilter:
        _pf_path = Path(args.prefilter)
        if _pf_path.exists():
            import yaml
            import operator as _op

            _PF_OPS = {
                ">=": _op.ge,
                ">": _op.gt,
                "<=": _op.le,
                "<": _op.lt,
                "==": _op.eq,
                "!=": _op.ne,
            }
            with open(_pf_path, "r", encoding="utf-8") as _f:
                _pf_cfg = yaml.safe_load(_f)
            _pf_rules = _pf_cfg.get("rules", []) if _pf_cfg else []
            if _pf_rules:
                _n_before = len(df)
                for _rule in _pf_rules:
                    if "any_of" in _rule:
                        # OR 组: 满足任一条件即通过
                        _or_mask = pd.Series(False, index=df.index)
                        for _sub in _rule["any_of"]:
                            _sf = _sub["feature"]
                            _sop = _PF_OPS.get(_sub["operator"])
                            if _sop and _sf in df.columns:
                                _or_mask |= _sop(df[_sf], _sub["value"])
                        df = df[_or_mask].copy()
                    else:
                        _feat = _rule.get("feature", "")
                        _op_str = _rule.get("operator", "")
                        _val = _rule.get("value")
                        _op_func = _PF_OPS.get(_op_str)
                        if _op_func and _feat in df.columns:
                            df = df[_op_func(df[_feat], _val)].copy()
                print(
                    f"🛡️  Prefilter applied: {_n_before} → {len(df)} rows "
                    f"({len(df)/_n_before:.1%} retained)"
                )
            else:
                print(f"ℹ️  Prefilter {_pf_path}: rules 为空, 不过滤")
        else:
            print(f"⚠️  Prefilter file not found: {args.prefilter}, 跳过")

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
        arch = load_strategy_archetype(
            args.strategy,
            args.strategies_root,
            gate_path=args.gate_path,
        )
        print(f"✅ Loaded strategy: {arch.name}")
        print(f"   Hard gates: {len(arch.gate.hard_gates)}")

        # Process hard gates
        print("\n📋 Optimizing Hard Gates:")
        for rule in arch.gate.hard_gates:
            print(f"  Processing: {rule.id}")

            # 跳过 frozen 规则
            if getattr(rule, "frozen", False):
                print(f"    ⚠️  FROZEN: 禁止优化，保持当前阈值")
                all_results[rule.id] = {
                    "rule_id": rule.id,
                    "status": "frozen",
                    "reason": "Rule marked as frozen, threshold optimization disabled",
                }
                continue

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

    # ==========================================================================
    # --promote: 将优化后的规则写入 archetypes/gate.yaml
    # ==========================================================================
    if args.promote:
        _promote_gate_to_archetypes(
            args.strategy,
            args.strategies_root,
            arch,
            all_results,
            args.gate_path,
            df=df,
            min_combined_pass_rate=args.min_combined_pass_rate,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
