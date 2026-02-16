#!/usr/bin/env python3
"""
Evidence Score Plateau Optimization

用你现在的 KPI 来对齐（非常重要）

你已经把 KPI 改成了：

⭐ bad_suppression
P(score < 0.3 | bad) − P(score < 0.3 | good)

这意味着 Evidence 层优化在做什么？

它在优化的是：

“在不误伤 good 的情况下，尽可能压制 bad 的风险敞口。”

目标: 找到分位数划分，使得 good 样本的分数分布显著高于 bad 样本。

使用方法:
    python scripts/optimize_evidence_plateau.py \
        --strategy bpc \
        --logs results/trade_logs.parquet \
        --output results/evidence_optimization.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.archetype import (
    EvidenceFeature,
    StrategyArchetype,
    load_strategy_archetype,
)


# ================================================================
# FeatureStore 补全缺失特征 (复用 eval_soft_gates 的 merge-by-(symbol,close) 逻辑)
# ================================================================


def _load_missing_features_from_store(
    df: pd.DataFrame,
    missing_features: list,
    features_store_root: str,
    features_store_layer: str,
    timeframe: str = "240T",
) -> pd.DataFrame:
    """从 FeatureStore 加载缺失特征，通过 (symbol, close) 对齐 merge 到 df。"""
    from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec

    store = FeatureStore(features_store_root)
    sym_col = "symbol" if "symbol" in df.columns else "_symbol"
    if sym_col not in df.columns:
        print("   ⚠️  No symbol column, cannot merge from FeatureStore")
        return df
    symbols = df[sym_col].unique().tolist()

    fs_parts = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=features_store_layer, symbol=sym, timeframe=timeframe
        )
        try:
            df_sym = store.read_range(
                spec,
                start=pd.Timestamp("1970-01-01"),
                end=pd.Timestamp("2100-01-01"),
            )
            if df_sym.empty:
                continue
            df_sym = df_sym.copy()
            if df_sym.index.name == "timestamp":
                df_sym = df_sym.reset_index()
            keep_cols = ["close"] + [f for f in missing_features if f in df_sym.columns]
            if len(keep_cols) <= 1:
                continue
            df_sym = df_sym[keep_cols].copy()
            df_sym[sym_col] = sym
            fs_parts.append(df_sym)
        except Exception as e:
            print(f"   ⚠️  FeatureStore read failed for {sym}: {e}")

    if not fs_parts:
        return df

    fs_all = pd.concat(fs_parts, ignore_index=True)
    loaded_feats = [c for c in fs_all.columns if c in missing_features]
    if not loaded_feats:
        return df

    # 对齐: 用 (symbol, close_rounded) 做 merge key
    df["_merge_key"] = df["close"].round(8).astype(str) + "|" + df[sym_col].astype(str)
    fs_all["_merge_key"] = (
        fs_all["close"].round(8).astype(str) + "|" + fs_all[sym_col].astype(str)
    )

    fs_dedup = fs_all.drop_duplicates(subset=["_merge_key"], keep="last")
    merge_cols = ["_merge_key"] + loaded_feats
    merged = df.merge(fs_dedup[merge_cols], on="_merge_key", how="left")
    merged.drop(columns=["_merge_key"], inplace=True)

    matched = merged[loaded_feats[0]].notna().sum()
    print(
        f"   📦 FeatureStore merge: {matched}/{len(df)} rows matched, loaded {loaded_feats}"
    )

    return merged


@dataclass
class EvidenceOptimizationConfig:
    """Evidence optimization configuration"""

    # ❗ 问题 4 修复: 降低 min_sharpness，只防反向 evidence
    # 在 relative-rank + imbalance 下，sharpness 容易接近 0，但不代表 evidence 无用
    min_sharpness: float = (
        -0.02
    )  # 只防"反向 evidence"，不防"弱但稳定的 risk-control evidence"
    min_samples_good: int = 50  # good 样本最少数量
    min_samples_bad: int = 50  # bad 样本最少数量
    # 注: plateau 判断改用 top 20% + neighbor >= 3 + CV + semantic drift，不再用固定 bins 宽度

    # ❗ 隐患 1 修复: Evidence 必须同时满足双重约束
    # 否则会"奖励无用 evidence"（只压 bad，但不拉开 good 结构差异）
    min_bad_suppression: float = 0.05  # bad 被选择性压制的最低要求
    min_good_amplification: float = 0.05  # good 被选择性放大的最低要求

    # ❗ 隐患 2 修复: Plateau 需要空间连续性约束
    plateau_max_bins_distance: float = 0.15  # bins 空间最大距离

    # ❗ 问题 2 修复: plateau 邻域需要语义稳定性约束
    # 不只是 bins 近，而是"行为结果近"
    # 注: quantile-threshold 模式下 bins 移动引起的 drift 比 rank 模式大
    # 0.10 = 允许 p_favor_good/p_suppress_bad 在邻域内波动 ±10%
    plateau_max_semantic_drift: float = 0.10  # p_favor_good / p_suppress_bad 最大漂移

    # ❗ 隐患 3 修复: Evidence score 应该是相对排名，而不是绝对分数
    use_relative_rank: bool = True  # 是否使用相对排名模式
    relative_rank_bins: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8]
    )

    # ❗ 问题 1 修复: 优化期使用近似 gate-conditioned rank
    # 默认使用 good 样本作为 gate 近似，确保优化期和实盘期语义一致
    use_gate_proxy_in_optimization: bool = True


# 语义标签到分数的映射
SEMANTIC_SCORE_MAP = {
    "suppress": 0.0,
    "downweight": 0.25,
    "neutral": 0.5,
    "favor": 0.75,
    "amplify": 1.0,
}


def compute_evidence_score(
    values: pd.Series,
    quantile_bins: List[float],
    quantile_labels: List[str],
    quantiles: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    计算 evidence 分数

    Args:
        values: 特征值序列
        quantile_bins: 分位数边界 [q1, q2, q3, q4]
        quantile_labels: 语义标签 [label1, label2, label3, label4, label5]
        quantiles: 分位数值字典 {"p25": 0.25, "p50": 0.5, ...}

    Returns:
        分数序列 (0-1 范围)
    """
    scores = []

    for val in values:
        if pd.isna(val):
            scores.append(0.5)  # 默认中性
            continue

        # 计算分位数位置
        if quantiles:
            # 使用提供的分位数值
            percentile = 0.5
            for q_name, q_val in sorted(quantiles.items(), key=lambda x: x[1]):
                if val <= q_val:
                    # 解析分位数名称 (e.g., "p25" -> 0.25)
                    try:
                        percentile = float(q_name.replace("p", "")) / 100
                    except ValueError:
                        percentile = 0.5
                    break
        else:
            # 假设值已经是百分位
            percentile = val

        # 根据分位数边界确定标签
        label = quantile_labels[-1] if quantile_labels else "neutral"
        for i, bin_val in enumerate(quantile_bins):
            if percentile <= bin_val:
                label = quantile_labels[i] if i < len(quantile_labels) else "neutral"
                break

        # 映射到分数
        score = SEMANTIC_SCORE_MAP.get(label, 0.5)
        scores.append(score)

    return pd.Series(scores, index=values.index)


def _value_to_label(
    pct_rank: float,
    rank_bins: List[float],
    quantile_labels: List[str],
) -> str:
    """根据百分位排名确定语义标签"""
    if pd.isna(pct_rank):
        return "neutral"
    label = quantile_labels[-1]  # 默认是最后一个 (amplify)
    for i, bin_val in enumerate(rank_bins):
        if pct_rank <= bin_val:
            label = quantile_labels[i] if i < len(quantile_labels) else "neutral"
            break
    return label


def compute_evidence_score_relative(
    values: pd.Series,
    rank_bins: List[float] = None,
    quantile_labels: List[str] = None,
    direction: str = "higher_is_better",
    gate_mask: pd.Series = None,
) -> pd.Series:
    """
    使用 gate-conditioned 分位数边界计算 evidence 分数

    核心设计:
    - 用 gate 放行样本的分布计算分位数阈值（参考分布）
    - 用这些阈值对 **所有样本** 赋分（包括 bad 样本）
    - 这样 bad 样本如果特征值偏低，就会被映射到 suppress 区间

    在多 symbol 场景下，绝对分数会失效（BTC 上 amplify=1.0 很强，ETH 上可能只是 baseline）
    正确做法: 根据 gate-conditioned 分位数边界分配分数

    Args:
        values: 特征值序列
        rank_bins: 排名分界点，默认 [0.2, 0.4, 0.6, 0.8]
        quantile_labels: 语义标签，默认 ["suppress", "downweight", "neutral", "favor", "amplify"]
        direction: 特征方向
            - "higher_is_better": 值越大越好 (如 strength, momentum)
            - "lower_is_better": 值越小越好 (如 volatility, drawdown, risk)
        gate_mask: Gate 放行的样本掩码 (Boolean Series)
            - 如果提供，用 gate 放行样本的分布计算分位数边界
            - 然后用这些边界对所有样本赋分
            - 生产语义: "以 gate 放行样本的分布为基准，评估每个样本的相对位置"

    Returns:
        分数序列 (0-1 范围)
    """
    if rank_bins is None:
        rank_bins = [0.2, 0.4, 0.6, 0.8]
    if quantile_labels is None:
        quantile_labels = ["suppress", "downweight", "neutral", "favor", "amplify"]

    # 初始化分数 (默认中性)
    scores = pd.Series(0.5, index=values.index)

    # 确定参考分布的样本范围
    if gate_mask is not None:
        ref_values = values[gate_mask].dropna()
    else:
        ref_values = values.dropna()

    if len(ref_values) == 0:
        return scores

    # 用参考分布计算分位数阈值
    # rank_bins = [0.2, 0.4, 0.6, 0.8] → 计算 20th, 40th, 60th, 80th 百分位的实际特征值
    quantile_thresholds = [ref_values.quantile(q) for q in rank_bins]

    # 对于 "lower_is_better" 特征，反转方向
    # 即: 值越低应该得到越高的分数 (amplify)
    if direction in ("lower_is_better", "negative"):
        # 反转标签顺序，使得低值 → amplify，高值 → suppress
        quantile_labels_effective = list(reversed(quantile_labels))
    else:
        quantile_labels_effective = quantile_labels

    # 对 **所有样本** 赋分（包括 bad 样本）
    def assign_score(val):
        if pd.isna(val):
            return 0.5  # NaN 默认中性
        # 找到该值落在哪个分位数区间
        for i, threshold in enumerate(quantile_thresholds):
            if val <= threshold:
                label = (
                    quantile_labels_effective[i]
                    if i < len(quantile_labels_effective)
                    else "neutral"
                )
                return SEMANTIC_SCORE_MAP.get(label, 0.5)
        # 超过最后一个阈值
        label = (
            quantile_labels_effective[-1] if quantile_labels_effective else "neutral"
        )
        return SEMANTIC_SCORE_MAP.get(label, 0.5)

    scores = values.apply(assign_score)

    return scores


def compute_sharpness(
    df: pd.DataFrame,
    feature_col: str,
    quantile_bins: List[float],
    quantile_labels: List[str],
    label_col: str = "is_good",
    use_relative_rank: bool = True,
    direction: str = "higher_is_better",
    gate_mask: pd.Series = None,  # ❗ 问题 1 修复: gate-conditioned rank
) -> Dict[str, float]:
    """
    计算给定分位数划分的 sharpness

    Args:
        df: DataFrame
        feature_col: 特征列名
        quantile_bins: 分位数边界 (e.g., [0.2, 0.4, 0.6, 0.8])
        quantile_labels: 语义标签
        label_col: 标签列
        use_relative_rank: 是否使用相对排名模式 (多 symbol 场景必须 True)
        direction: 特征方向 ("higher_is_better" 或 "lower_is_better")
        gate_mask: Gate 放行的样本掩码，用于 gate-conditioned rank

    Returns:
        包含 sharpness 和其他指标的字典
    """
    if feature_col not in df.columns:
        return {"sharpness": 0.0, "mean_good": 0.5, "mean_bad": 0.5}

    feature_values = df[feature_col].dropna()
    if len(feature_values) == 0:
        return {"sharpness": 0.0, "mean_good": 0.5, "mean_bad": 0.5}

    # ❗ 隐患 3 修复: 根据模式选择计算方式
    if use_relative_rank:
        # 多 symbol 场景: 使用 within-gate relative rank
        scores = compute_evidence_score_relative(
            df[feature_col],
            rank_bins=quantile_bins,
            quantile_labels=quantile_labels,
            direction=direction,
            gate_mask=gate_mask,  # ❗ 问题 1 修复: 传入 gate_mask
        )
    else:
        # 单 symbol 场景: 使用原来的绝对分位数方式
        # ❗ Bug 1 修复: 计算实际的 empirical quantiles
        quantiles = {}
        for i, q in enumerate(quantile_bins):
            q_name = f"p{int(q*100)}"
            quantiles[q_name] = feature_values.quantile(q)

        scores = compute_evidence_score(
            df[feature_col],
            quantile_bins,
            quantile_labels,
            quantiles=quantiles,
        )

    # 分组统计
    is_good = df[label_col] == 1
    is_bad = df[label_col] == 0

    mean_good = scores[is_good].mean() if is_good.any() else 0.5
    mean_bad = scores[is_bad].mean() if is_bad.any() else 0.5

    sharpness = mean_good - mean_bad

    # ❗ Bug 3 修复: 主 KPI 改为 bad_suppression
    # 在极端 class imbalance (Good=92%) 下，mean difference 会失真
    # 正确的衡量是: "bad 样本被压制了多少"
    suppress_threshold = 0.3  # score < 0.3 算被压制
    favor_threshold = 0.7  # score > 0.7 算被放大

    p_suppress_bad = (
        (scores[is_bad] < suppress_threshold).mean() if is_bad.any() else 0.0
    )
    p_suppress_good = (
        (scores[is_good] < suppress_threshold).mean() if is_good.any() else 0.0
    )
    p_favor_good = (scores[is_good] > favor_threshold).mean() if is_good.any() else 0.0
    p_favor_bad = (scores[is_bad] > favor_threshold).mean() if is_bad.any() else 0.0

    # bad_suppression = bad 被压制的比例 - good 被压制的比例
    # 正值越大越好：表示 bad 被选择性压制
    bad_suppression = p_suppress_bad - p_suppress_good

    # good_amplification = good 被放大的比例 - bad 被放大的比例
    good_amplification = p_favor_good - p_favor_bad

    # 计算高分比例差异（保留原有逻辑）
    high_score_threshold = 0.7
    p_high_good = (
        (scores[is_good] > high_score_threshold).mean() if is_good.any() else 0.0
    )
    p_high_bad = (scores[is_bad] > high_score_threshold).mean() if is_bad.any() else 0.0

    return {
        "sharpness": sharpness,  # 保留作为参考
        "mean_good": mean_good,
        "mean_bad": mean_bad,
        "p_high_good": p_high_good,
        "p_high_bad": p_high_bad,
        "high_score_lift": p_high_good - p_high_bad,
        # ❗ Bug 3 修复: 新增主 KPI
        "bad_suppression": bad_suppression,  # ⭐ 主 KPI: bad 被选择性压制
        "good_amplification": good_amplification,  # 辅助 KPI
        "p_suppress_bad": p_suppress_bad,
        "p_suppress_good": p_suppress_good,
        "p_favor_good": p_favor_good,
        "p_favor_bad": p_favor_bad,
    }


def generate_quantile_bin_candidates(
    base_bins: List[float] = [0.2, 0.4, 0.6, 0.8],
) -> List[List[float]]:
    """
    生成分位数边界候选集

    基于 base_bins 生成变体：
    - 移动边界 ±0.05, ±0.10
    - 调整对称性
    """
    candidates = [base_bins]

    # 生成移动变体
    shifts = [-0.10, -0.05, 0.05, 0.10]
    for shift in shifts:
        shifted = [min(1.0, max(0.0, b + shift)) for b in base_bins]
        if shifted not in candidates:
            candidates.append(shifted)

    # 生成非对称变体
    asymmetric_variants = [
        [0.15, 0.35, 0.65, 0.85],  # 更紧的中性区
        [0.25, 0.45, 0.55, 0.75],  # 更宽的中性区
        [0.10, 0.30, 0.70, 0.90],  # 极端分布
        [0.30, 0.45, 0.55, 0.70],  # 压缩分布
    ]
    for variant in asymmetric_variants:
        if variant not in candidates:
            candidates.append(variant)

    return candidates


def optimize_evidence_feature(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str = "is_good",
    config: Optional[EvidenceOptimizationConfig] = None,
    direction: str = "higher_is_better",  # ❗ Bug 1 修复: 特征方向
) -> Dict[str, Any]:
    """
    优化单个 evidence 特征的分位数划分

    Args:
        df: DataFrame
        feature_col: 特征列名
        label_col: 标签列
        config: 优化配置
        direction: 特征方向
            - "higher_is_better": 值越大越好 (如 strength, momentum)
            - "lower_is_better": 值越小越好 (如 volatility, risk)

    Returns:
        优化结果
    """
    if config is None:
        config = EvidenceOptimizationConfig()

    if feature_col not in df.columns:
        return {
            "feature": feature_col,
            "status": "skip",
            "reason": f"Feature not found in DataFrame",
        }

    # 检查样本数
    n_good = (df[label_col] == 1).sum()
    n_bad = (df[label_col] == 0).sum()

    if n_good < config.min_samples_good or n_bad < config.min_samples_bad:
        return {
            "feature": feature_col,
            "status": "skip",
            "reason": f"Insufficient samples: good={n_good}, bad={n_bad}",
        }

    # 5档语义标签（固定）
    quantile_labels = ["suppress", "downweight", "neutral", "favor", "amplify"]

    # 生成候选分位数边界
    candidates = generate_quantile_bin_candidates()

    # ❗ 问题 1 修复: 优化期使用近似 gate-conditioned rank
    # 默认使用 good 样本作为 gate 近似，确保优化期和实盘期语义一致
    # 原因: 实盘时 Evidence 只在 Gate 放行的样本中生效，优化时也应该在类似空间中排名
    gate_proxy = df[label_col] == 1 if config.use_gate_proxy_in_optimization else None

    # 评估每个候选
    results = []
    for bins in candidates:
        # ❗ 隐患 3 + Bug 1 + 问题 1 修复
        metrics = compute_sharpness(
            df,
            feature_col,
            bins,
            quantile_labels,
            label_col,
            use_relative_rank=config.use_relative_rank,
            direction=direction,
            gate_mask=gate_proxy,  # ❗ 问题 1: 使用 gate proxy
        )
        metrics["quantile_bins"] = bins
        results.append(metrics)

    # ❗ Bug 2 & 3 修复: 使用 bad_suppression 作为主 KPI
    # 找最佳候选（最大 bad_suppression）
    best = max(results, key=lambda x: x.get("bad_suppression", 0))

    # ❗ 隐患 1 修复: Evidence 必须同时满足双重约束
    # 条件 1: bad_suppression > min_bad_suppression (压 bad)
    # 条件 2: good_amplification > min_good_amplification (拉开 good 结构差异)
    # 条件 3: sharpness >= min_sharpness (基本分离度)
    # 否则会"奖励无用 evidence" —— 只是"轻微不伤 good 的无效过滤器"
    valid_results = [
        r
        for r in results
        if r.get("bad_suppression", 0) > config.min_bad_suppression
        and r.get("good_amplification", 0) > config.min_good_amplification
        and r.get("sharpness", 0) >= config.min_sharpness
    ]

    if not valid_results:
        return {
            "feature": feature_col,
            "status": "no_valid_bins",
            "reason": "No bins satisfy dual constraint (bad_suppression + good_amplification)",
            "best_sharpness": best.get("sharpness", 0),
            "best_bad_suppression": best.get("bad_suppression", 0),
            "best_good_amplification": best.get("good_amplification", 0),
            "best_bins": best["quantile_bins"],
            "scan_results": results,
        }

    # ❗ Bug 2 修复: 真正的 plateau 判断
    # 排序并找到 bad_suppression 前 20% 的候选
    valid_sorted = sorted(
        valid_results, key=lambda x: x.get("bad_suppression", 0), reverse=True
    )
    top_count = max(3, len(valid_sorted) // 5)
    top_candidates = valid_sorted[:top_count]

    # 计算 top 候选的 bad_suppression 方差
    top_bad_supp = [r.get("bad_suppression", 0) for r in top_candidates]
    bad_supp_std = np.std(top_bad_supp) if len(top_bad_supp) > 1 else 0
    bad_supp_mean = np.mean(top_bad_supp) if top_bad_supp else 0

    # plateau 判断条件 1: CV < 50% (数值稳定)
    cv = bad_supp_std / bad_supp_mean if bad_supp_mean > 0.01 else float("inf")
    cv_stable = cv < 0.5

    # ❗ 隐患 2 修复: Plateau 需要 bins 空间连续性约束
    def bins_distance(b1: list, b2: list) -> float:
        """计算两个 bins 配置的空间距离"""
        return np.mean([abs(a - b) for a, b in zip(b1, b2)])

    max_dist = 0.0
    if len(top_candidates) > 1:
        for i, r1 in enumerate(top_candidates):
            for r2 in top_candidates[i + 1 :]:
                dist = bins_distance(r1["quantile_bins"], r2["quantile_bins"])
                max_dist = max(max_dist, dist)

    # plateau 判断条件 2: bins 空间连续 (max_dist < threshold)
    bins_continuous = max_dist < config.plateau_max_bins_distance

    # ❗ 问题 3 修复: 先选择 recommended，然后基于 recommended 做 plateau 判断
    # 原因: best 可能在 invalid，而 recommended 在 valid，两者 bins 可能相距很远
    recommended = max(valid_results, key=lambda x: x.get("bad_suppression", 0))
    ref_bins = recommended["quantile_bins"]  # ❗ 问题 3: 用 recommended 而不是 best

    def is_neighbor(b: list, ref: list, threshold: float = 0.1) -> bool:
        """判断 b 是否是 ref 的邻域"""
        return bins_distance(b, ref) < threshold

    # 找到推荐点的邻域候选
    neighbors = [r for r in valid_results if is_neighbor(r["quantile_bins"], ref_bins)]

    # ❗ 问题 2 修复: 邻域需要语义稳定性约束
    # 不只是 bins 近，而是"行为结果近"
    def semantic_drift(r1: dict, r2: dict) -> float:
        """计算两个结果的语义漂移（行为结果差异）"""
        drift_favor = abs(r1.get("p_favor_good", 0) - r2.get("p_favor_good", 0))
        drift_suppress = abs(r1.get("p_suppress_bad", 0) - r2.get("p_suppress_bad", 0))
        return max(drift_favor, drift_suppress)

    # 语义漂移检查：只对比 recommended vs 每个邻居
    # （不再做 O(n²) 全对比，因为远处邻居之间的 drift 不该影响 recommended 的质量）
    # 只保留与 recommended 语义距离在阈值内的邻居
    semantic_neighbors = [
        r
        for r in neighbors
        if semantic_drift(r, recommended) < config.plateau_max_semantic_drift
    ]
    max_semantic_drift = max(
        (semantic_drift(r, recommended) for r in semantic_neighbors), default=0.0
    )

    # plateau 判断条件 3a: 邻域内 bad_suppression 稳定
    neighbor_bad_supp = [r.get("bad_suppression", 0) for r in semantic_neighbors]
    neighbor_cv = (
        np.std(neighbor_bad_supp) / np.mean(neighbor_bad_supp)
        if len(semantic_neighbors) > 1 and np.mean(neighbor_bad_supp) > 0.01
        else float("inf")
    )

    # plateau 判断条件 3b: 语义邻居数量充足
    semantic_stable = len(semantic_neighbors) >= 3

    # plateau 判断条件 3: 至少 3 个语义邻居 + CV 稳定
    neighbor_plateau = semantic_stable and neighbor_cv < 0.5

    # 最终 plateau 判断: 数值稳定 AND 空间连续 AND 邻域稳定
    is_plateau = cv_stable and bins_continuous and neighbor_plateau

    # ❗ 简化设计: 没有 plateau 的 evidence 直接移除，不进入生产系统
    # 原因: knife-edge 增加不可解释的波动、破坏参数可迁移性、让系统复杂度虚增
    if not is_plateau:
        return {
            "feature": feature_col,
            "status": "rejected",  # 直接移除，不是 knife_edge
            "reason": "no_plateau",
            "direction": direction,
            "recommended_bins": recommended["quantile_bins"],
            "bad_suppression": recommended.get("bad_suppression", 0),
            "good_amplification": recommended.get("good_amplification", 0),
            "plateau_cv": cv,
            "cv_stable": cv_stable,
            "bins_continuous": bins_continuous,
            "neighbor_plateau": neighbor_plateau,
            "semantic_stable": semantic_stable,
        }

    return {
        "feature": feature_col,
        "status": "optimized",  # 只有 plateau 才能进入生产
        "direction": direction,
        "recommended_bins": recommended["quantile_bins"],
        "sharpness": recommended.get("sharpness", 0),
        "bad_suppression": recommended.get("bad_suppression", 0),
        "good_amplification": recommended.get("good_amplification", 0),
        "mean_good": recommended.get("mean_good", 0),
        "mean_bad": recommended.get("mean_bad", 0),
        "high_score_lift": recommended.get("high_score_lift", 0),
        "p_suppress_bad": recommended.get("p_suppress_bad", 0),
        "p_favor_good": recommended.get("p_favor_good", 0),
        "is_plateau": True,
        "plateau_cv": cv,
        "num_neighbors": len(semantic_neighbors),
        "max_semantic_drift": max_semantic_drift,
        "num_valid_candidates": len(valid_results),
    }


def _generate_html_report(
    df: pd.DataFrame,
    opt_results: Dict[str, Any],
    output_path: Path,
    label_col: str = "is_good",
) -> None:
    """生成美化的 Evidence 优化 HTML 报告"""
    from datetime import datetime

    # 计算汇总指标
    n_all = len(df)
    n_good = (df[label_col] == 1).sum()
    n_bad = n_all - n_good
    good_rate = n_good / n_all if n_all > 0 else 0

    # 统计优化结果
    n_optimized = sum(1 for r in opt_results.values() if r.get("status") == "optimized")
    n_knife_edge = sum(
        1 for r in opt_results.values() if r.get("status") == "knife_edge"
    )
    n_no_valid = sum(
        1 for r in opt_results.values() if r.get("status") == "no_valid_bins"
    )
    n_skip = sum(1 for r in opt_results.values() if r.get("status") == "skip")

    # 计算主 KPI: bad_suppression
    optimized_results = [
        r
        for r in opt_results.values()
        if r.get("status") in ["optimized", "knife_edge"]
    ]
    bad_supp_values = [r.get("bad_suppression", 0) for r in optimized_results]
    avg_bad_supp = np.mean(bad_supp_values) if bad_supp_values else 0
    max_bad_supp = max(bad_supp_values) if bad_supp_values else 0

    # 保留 sharpness 作为参考
    sharpness_values = [r.get("sharpness", 0) for r in optimized_results]
    avg_sharpness = np.mean(sharpness_values) if sharpness_values else 0

    # 生成特征表格 HTML
    features_html = ""
    for ef_id, result in opt_results.items():
        status = result.get("status", "N/A")
        feature = result.get("feature", ef_id)

        if status == "optimized":
            status_html = '<span class="status-ok">✅ 稳定平台</span>'
            bins = result.get("recommended_bins", [])
            bins_str = str(bins) if bins else "N/A"
            bad_supp = result.get("bad_suppression", 0)
            bad_supp_html = (
                f"<strong>{bad_supp:.3f}</strong>"
                if isinstance(bad_supp, (int, float))
                else "N/A"
            )
            good_amp = result.get("good_amplification", 0)
            good_amp_html = (
                f"<strong>{good_amp:.3f}</strong>"
                if isinstance(good_amp, (int, float))
                else "N/A"
            )
            p_suppress_bad = result.get("p_suppress_bad", 0)
            p_favor_good = result.get("p_favor_good", 0)
            kpi_detail_html = f"{p_suppress_bad:.1%} / {p_favor_good:.1%}"
            sharpness = result.get("sharpness", 0)
            sharpness_html = (
                f"{sharpness:.3f}" if isinstance(sharpness, (int, float)) else "N/A"
            )
            plateau_cv = result.get("plateau_cv", 0)
            cv_html = f"{plateau_cv:.2f}"
        elif status == "knife_edge":
            status_html = '<span class="status-warn">⚠️ Knife-edge</span>'
            bins = result.get("recommended_bins", [])
            bins_str = str(bins) if bins else "N/A"
            bad_supp = result.get("bad_suppression", 0)
            bad_supp_html = (
                f"{bad_supp:.3f}" if isinstance(bad_supp, (int, float)) else "N/A"
            )
            good_amp = result.get("good_amplification", 0)
            good_amp_html = (
                f"{good_amp:.3f}" if isinstance(good_amp, (int, float)) else "N/A"
            )
            p_suppress_bad = result.get("p_suppress_bad", 0)
            p_favor_good = result.get("p_favor_good", 0)
            kpi_detail_html = f"{p_suppress_bad:.1%} / {p_favor_good:.1%}"
            sharpness = result.get("sharpness", 0)
            sharpness_html = (
                f"{sharpness:.3f}" if isinstance(sharpness, (int, float)) else "N/A"
            )
            plateau_cv = result.get("plateau_cv", 0)
            cv_html = f"<span style='color:#e74c3c'>{plateau_cv:.2f}</span>"
        elif status == "no_valid_bins":
            status_html = '<span class="status-fail">❌ 无有效划分</span>'
            bins_str = "-"
            bad_supp_html = (
                f"{result.get('best_bad_suppression', 0):.3f}"
                if result.get("best_bad_suppression")
                else "-"
            )
            good_amp_html = (
                f"{result.get('best_good_amplification', 0):.3f}"
                if result.get("best_good_amplification")
                else "-"
            )
            kpi_detail_html = "-"
            sharpness_html = "-"
            cv_html = "-"
        else:
            status_html = '<span class="status-fail">⏭ 跳过</span>'
            bins_str = "-"
            bad_supp_html = "-"
            good_amp_html = "-"
            kpi_detail_html = "-"
            sharpness_html = "-"
            cv_html = "-"

        features_html += f"""
            <tr>
                <td><code>{ef_id}</code></td>
                <td>{feature}</td>
                <td>{status_html}</td>
                <td>{bad_supp_html}</td>
                <td>{good_amp_html}</td>
                <td>{kpi_detail_html}</td>
                <td>{sharpness_html}</td>
                <td>{cv_html}</td>
                <td style="font-size:11px;">{bins_str}</td>
            </tr>"""

    # 完整 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evidence 优化报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #8e44ad; margin-bottom: 30px; font-size: 28px; }}
        h2 {{ color: #34495e; border-bottom: 3px solid #8e44ad; padding-bottom: 10px; margin: 30px 0 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; }}
        .kpi-item {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .kpi-item.primary {{ background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%); }}
        .kpi-item.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .kpi-item.warning {{ background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%); }}
        .kpi-value {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
        .kpi-label {{ font-size: 13px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; color: #2c3e50; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .status-ok {{ color: #27ae60; font-weight: bold; }}
        .status-warn {{ color: #f39c12; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .secondary {{ color: #7f8c8d; font-size: 14px; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .timestamp {{ text-align: center; color: #95a5a6; font-size: 12px; margin-top: 30px; }}
        .hint {{ background: #fff3cd; border-left: 4px solid #f39c12; padding: 15px; margin-top: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Evidence 层优化报告</h1>
    <p style="text-align:center;color:#7f8c8d;margin-bottom:30px;">Relative Rank Based Evidence Score Optimization (多 Symbol 模式)</p>

    <h2>🎯 核心 KPI</h2>
    <div class="card">
        <div class="kpi-grid">
            <div class="kpi-item primary">
                <div class="kpi-label">平均 Bad Suppression</div>
                <div class="kpi-value">{avg_bad_supp:.3f}</div>
                <div class="kpi-label">⭐ 主 KPI: Bad 被选择性压制</div>
            </div>
            <div class="kpi-item success">
                <div class="kpi-label">最大 Bad Suppression</div>
                <div class="kpi-value">{max_bad_supp:.3f}</div>
                <div class="kpi-label">最佳特征</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">稳定平台</div>
                <div class="kpi-value">{n_optimized}</div>
                <div class="kpi-label">共 {len(opt_results)} 个特征</div>
            </div>
            <div class="kpi-item warning">
                <div class="kpi-label">Knife-edge</div>
                <div class="kpi-value">{n_knife_edge}</div>
                <div class="kpi-label">无稳定平台</div>
            </div>
        </div>
        <div class="secondary">
            <strong>参考指标:</strong> 平均 Sharpness = {avg_sharpness:.3f} | 无效: {n_no_valid} | 跳过: {n_skip}
        </div>
    </div>

    <h2>📋 样本分布</h2>
    <div class="card">
        <div class="kpi-grid">
            <div class="kpi-item">
                <div class="kpi-label">总样本</div>
                <div class="kpi-value">{n_all:,}</div>
            </div>
            <div class="kpi-item success">
                <div class="kpi-label">Good 样本</div>
                <div class="kpi-value">{n_good:,}</div>
                <div class="kpi-label">{good_rate*100:.1f}%</div>
            </div>
            <div class="kpi-item warning">
                <div class="kpi-label">Bad 样本</div>
                <div class="kpi-value">{n_bad:,}</div>
                <div class="kpi-label">{(1-good_rate)*100:.1f}%</div>
            </div>
        </div>
    </div>

    <h2>🔧 Evidence 特征优化结果</h2>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Evidence ID</th>
                    <th>Feature</th>
                    <th>状态</th>
                    <th>Bad Supp ⭐</th>
                    <th>Good Amp ⭐</th>
                    <th>P(supp|bad) / P(favor|good)</th>
                    <th>Sharpness</th>
                    <th>Plateau CV</th>
                    <th>推荐 Quantile Bins</th>
                </tr>
            </thead>
            <tbody>{features_html}
            </tbody>
        </table>
    </div>

    <div class="hint">
        <strong>💡 KPI 解读：</strong><br>
        <strong>Bad Suppression (主 KPI)</strong> = P(score &lt; 0.3 | bad) - P(score &lt; 0.3 | good) — Bad 被选择性压制<br>
        <strong>Good Amplification (双重约束)</strong> = P(score &gt; 0.7 | good) - P(score &gt; 0.7 | bad) — Good 被选择性放大<br>
        ❗ Evidence 必须同时满足双重约束，否则只是"无效过滤器"。<br><br>
        <strong>Plateau 判断</strong> = CV &lt; 0.5 (数值稳定) AND bins 距离 &lt; 0.15 (空间连续)<br>
        同时满足才认为是稳定平台，否则是 knife-edge。<br><br>
        <strong>❗ 多 Symbol 模式</strong>: Score 基于 within-gate relative rank，而非绝对分位数<br>
        这确保不同 symbol 的 evidence 可比较。
    </div>

    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence Score Plateau Optimization")
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
        "--min-sharpness",
        type=float,
        default=0.05,
        help="Minimum sharpness requirement",
    )
    parser.add_argument(
        "--features-store-root",
        default="feature_store",
        help="FeatureStore root directory (default: feature_store)",
    )
    parser.add_argument(
        "--features-store-layer",
        default=None,
        help="FeatureStore layer for missing features (auto-detect if omitted)",
    )
    parser.add_argument(
        "--timeframe",
        default="240T",
        help="Timeframe (default: 240T)",
    )
    args = parser.parse_args()

    # Load strategy archetype
    try:
        arch = load_strategy_archetype(args.strategy, args.strategies_root)
    except Exception as e:
        print(f"❌ Failed to load strategy '{args.strategy}': {e}")
        return 1

    print(f"✅ Loaded strategy: {arch.name}")
    print(f"   Evidence features: {len(arch.evidence.features)}")

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
            # ❗ Evidence优化基于RR分层: Good = Q4-Q5 (高RR), Bad = Q1-Q2 (低RR)
            # 参考: return_tree_kpi_framework.md 预测值分位数一致性
            q20 = df[rr_col].quantile(0.2)
            q80 = df[rr_col].quantile(0.8)
            # Good: RR > Q80, Bad: RR < Q20, Middle: 排除
            df_good = df[df[rr_col] > q80].copy()
            df_bad = df[df[rr_col] < q20].copy()
            df_good[args.label_col] = 1
            df_bad[args.label_col] = 0
            df = pd.concat([df_good, df_bad], ignore_index=True)
            print(f"ℹ️ Auto-generated '{args.label_col}' based on RR stratification")
            print(f"   Q20={q20:.2f}, Q80={q80:.2f}")
            print(f"   Good (RR > Q80): {len(df_good)}, Bad (RR < Q20): {len(df_bad)}")
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

    # ================================================================
    # FeatureStore: 自动补全 evidence 候选特征中缺失的列
    # ================================================================
    needed_features = {ef.feature for ef in arch.evidence.features}
    missing = [f for f in needed_features if f not in df.columns]
    if missing:
        print(f"\n   ⚠️  缺失 {len(missing)} 个 evidence 特征: {missing}")
        fs_layer = args.features_store_layer
        if not fs_layer:
            try:
                from src.feature_store.layer_naming import detect_layer_for_strategy

                fs_layer = detect_layer_for_strategy(
                    args.strategy, args.features_store_root
                )
            except Exception:
                pass
        if fs_layer:
            print(f"   📦 从 FeatureStore 补全 (layer={fs_layer})...")
            df = _load_missing_features_from_store(
                df, missing, args.features_store_root, fs_layer, args.timeframe
            )
            still_missing = [
                f for f in missing if f not in df.columns or df[f].isna().all()
            ]
            if still_missing:
                print(f"   ⚠️  仍缺失 (将被 skip): {still_missing}")
        else:
            print(
                "   💡 提示: 使用 --features-store-layer 指定 FeatureStore layer 以补全"
            )

    # Create config
    config = EvidenceOptimizationConfig(
        min_sharpness=args.min_sharpness,
    )

    # Optimize each evidence feature
    results = {}

    print("\n📋 Optimizing Evidence Features:")
    for ef in arch.evidence.features:
        feature_col = ef.feature
        # ❗ Bug 1 修复: 从 EvidenceFeature 获取 direction (如果有)
        # 默认 "higher_is_better"，但 volatility/risk 类特征应该配置为 "lower_is_better"
        direction = getattr(ef, "direction", "higher_is_better")
        print(f"  Processing: {feature_col} (direction: {direction})")

        result = optimize_evidence_feature(
            df, feature_col, args.label_col, config, direction=direction
        )
        results[ef.id] = result

        if result.get("status") == "optimized":
            print(f"    ✅ Optimized (Stable Plateau):")
            print(f"       Bins: {result['recommended_bins']}")
            print(f"       Bad Suppression: {result.get('bad_suppression', 0):.3f} ⭐")
            print(
                f"       Good Amplification: {result.get('good_amplification', 0):.3f} ⭐"
            )
            print(f"       Sharpness: {result.get('sharpness', 0):.3f}")
            print(
                f"       Neighbors: {result.get('num_neighbors', 0)} | Semantic drift: {result.get('max_semantic_drift', 0):.3f}"
            )
        elif result.get("status") == "rejected":
            # ❗ 简化设计: knife-edge 直接移除
            cv_ok = "✅" if result.get("cv_stable", False) else "❌"
            bins_ok = "✅" if result.get("bins_continuous", False) else "❌"
            nbr_ok = "✅" if result.get("neighbor_plateau", False) else "❌"
            sem_ok = "✅" if result.get("semantic_stable", False) else "❌"
            print(f"    ❌ Rejected (No Plateau):")
            print(f"       Bad Suppression: {result.get('bad_suppression', 0):.3f}")
            print(
                f"       CV {cv_ok} | Bins {bins_ok} | Neighbors {nbr_ok} | Semantic {sem_ok}"
            )
        else:
            print(f"    ❌ {result.get('status')}: {result.get('reason', 'N/A')}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    print(f"\n✅ Results saved to: {output_path}")

    # Summary
    n_optimized = sum(1 for r in results.values() if r.get("status") == "optimized")
    n_rejected = sum(1 for r in results.values() if r.get("status") == "rejected")
    n_skip = sum(1 for r in results.values() if r.get("status") == "skip")
    n_no_valid = sum(1 for r in results.values() if r.get("status") == "no_valid_bins")

    print(f"\n📊 Summary:")
    print(f"   ✅ Optimized (Stable Plateau): {n_optimized}")
    print(f"   ❌ Rejected (No Plateau): {n_rejected}")
    print(f"   ❌ No valid bins: {n_no_valid}")
    print(f"   ⏭ Skipped: {n_skip}")

    # Generate updated YAML config
    if n_optimized > 0:
        print("\n📝 Suggested evidence.yaml updates:")
        for ef_id, result in results.items():
            status = result.get("status", "")
            if status == "optimized":  # 只输出 plateau 的 evidence
                print(f"  ✅ {ef_id}:")
                print(f"    quantile_bins: {result['recommended_bins']}")
                print(f"    bad_suppression: {result.get('bad_suppression', 0):.3f}")

    # 生成美化的 HTML 报告
    html_path = output_path.with_suffix(".html")
    _generate_html_report(df, results, html_path, args.label_col)
    print(f"✅ HTML report saved to: {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
