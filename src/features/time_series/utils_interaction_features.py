"""
特征组合：交互特征和衍生特征

包含两类组合特征：
1. **交互特征**（Interaction）：两个特征的乘积（状态 × 动量）
   - 如：vpin × compression_energy = vpin_x_compression
   - 参考：docs/时序模型/高级特征：特征组合交互.md

2. **衍生特征**（Derived）：单个特征的变换或两个特征的其他运算
   - 如：dist_to_nearest_sr / atr = sr_distance_normalized（归一化）
   - 如：abs(close - zz_high_value) = dist_to_zz_high（差值）
   - 如：cvd 的滚动斜率 = cvd_slope_5（变换）

所有特征都是独立的计算函数，可以在 feature_dependencies.yaml 中单独定义
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature


@register_feature("compute_liquidity_void_x_wpt_risk", category="interaction")
def compute_liquidity_void_x_wpt_risk(
    df: pd.DataFrame,
    liquidity_void_col: str = "liquidity_void_detected",
    wpt_risk_col: str = "wpt_false_breakout_risk",
) -> pd.Series:
    """
    计算流动性真空 × WPT 假突破风险交互项
    
    Args:
        df: DataFrame with base features
        liquidity_void_col: Liquidity void detection column
        wpt_risk_col: WPT false breakout risk column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(liquidity_void_col, pd.Series(0.0, index=df.index))
    momentum = df.get(wpt_risk_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("liquidity_void_x_wpt_risk")


@register_feature("compute_liquidity_void_x_wpt_risk_from_series", category="interaction")
def compute_liquidity_void_x_wpt_risk_from_series(
    *,
    liquidity_void_detected: pd.Series,
    wpt_false_breakout_risk: pd.Series,
) -> pd.DataFrame:
    lv = pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0.0).astype(float)
    risk = pd.to_numeric(wpt_false_breakout_risk, errors="coerce").fillna(0.0).astype(float)
    return (lv * risk).rename("liquidity_void_x_wpt_risk").to_frame()


@register_feature("compute_liquidity_void_x_vpin_from_series", category="interaction")
def compute_liquidity_void_x_vpin_from_series(
    *,
    liquidity_void_detected: pd.Series,
    vpin: pd.Series,
    clip_vpin: float = 5.0,
) -> pd.DataFrame:
    """
    Heavy gate feature: liquidity void state × VPIN signal.

    This is intentionally a separate *heavy* node (depends on order-flow features) so the
    original `liquidity_void` can remain a cheap proxy.

    Args:
        liquidity_void_detected: 0/1 state series
        vpin: VPIN signal series (recommended: z-score like vpin_zscore_50)
        clip_vpin: clip VPIN signal to [-clip_vpin, clip_vpin] to avoid outliers dominating

    Returns:
        DataFrame with one column: liquidity_void_x_vpin
    """
    lv = pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0.0).astype(float)
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    if clip_vpin is not None and float(clip_vpin) > 0:
        vp = vp.clip(lower=-float(clip_vpin), upper=float(clip_vpin))
    out = (lv * vp).rename("liquidity_void_x_vpin")
    return out.to_frame()


@register_feature("compute_exhaustion_at_liquidity_void_from_series", category="interaction")
def compute_exhaustion_at_liquidity_void_from_series(
    *,
    trade_cluster_exhaustion_score: pd.Series,
    liquidity_void_detected: pd.Series,
    liquidity_void_false_breakout_risk: Optional[pd.Series] = None,
    use_risk: bool = True,
    clip_score: float = 1.0,
) -> pd.DataFrame:
    """
    Composite semantic: Exhaustion-at-Liquidity-Void

    Intuition:
    - liquidity_void_detected indicates a "low resistance path" / sweep-like episode (proxy without L2)
    - trade_cluster_exhaustion_score indicates "effort without progress" (reversal-friendly)
    - optional: weight by liquidity_void_false_breakout_risk to focus on quick-reversal voids

    Returns:
      DataFrame with one column: exhaustion_at_liquidity_void (0..~1 after clipping)
    """
    ex = pd.to_numeric(trade_cluster_exhaustion_score, errors="coerce").fillna(0.0).astype(float)
    lv = pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0.0).astype(float)
    out = ex * lv
    if use_risk and liquidity_void_false_breakout_risk is not None:
        risk = (
            pd.to_numeric(liquidity_void_false_breakout_risk, errors="coerce")
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0, upper=1.0)
        )
        out = out * risk
    if clip_score is not None and float(clip_score) > 0:
        out = out.clip(lower=0.0, upper=float(clip_score)) / float(clip_score)
    return out.rename("exhaustion_at_liquidity_void").to_frame()


@register_feature("compute_compression_energy_x_ofi_short", category="interaction")
def compute_compression_energy_x_ofi_short(
    df: pd.DataFrame,
    compression_col: str = "compression_energy",
    ofi_col: str = "ofi_short",
) -> pd.Series:
    """
    计算压缩能量 × 订单流强度交互项
    
    Args:
        df: DataFrame with base features
        compression_col: Compression energy column
        ofi_col: Order flow imbalance column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(compression_col, pd.Series(0.0, index=df.index))
    momentum = df.get(ofi_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("compression_energy_x_ofi_short")


@register_feature("compute_compression_energy_x_ofi_short_from_series", category="interaction")
def compute_compression_energy_x_ofi_short_from_series(
    *,
    compression_energy: pd.Series,
    ofi_short: pd.Series,
) -> pd.DataFrame:
    ce = pd.to_numeric(compression_energy, errors="coerce").fillna(0.0).astype(float)
    ofi = pd.to_numeric(ofi_short, errors="coerce").fillna(0.0).astype(float)
    return (ce * ofi).rename("compression_energy_x_ofi_short").to_frame()


@register_feature("compute_hurst_x_trend_r2", category="interaction")
def compute_hurst_x_trend_r2(
    df: pd.DataFrame,
    hurst_col: str = "hurst_close_rolling",
    trend_r2_col: str = "trend_r2_20",
) -> pd.Series:
    """
    计算 Hurst 指数 × 趋势 R² 交互项
    
    Args:
        df: DataFrame with base features
        hurst_col: Hurst exponent column
        trend_r2_col: Trend R² column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(hurst_col, pd.Series(0.5, index=df.index))
    momentum = df.get(trend_r2_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0.5) * momentum.fillna(0)).rename("hurst_x_trend_r2")


@register_feature("compute_hurst_x_trend_r2_from_series", category="interaction")
def compute_hurst_x_trend_r2_from_series(
    *,
    hurst_close_rolling: pd.Series,
    trend_r2_20: pd.Series,
) -> pd.DataFrame:
    h = pd.to_numeric(hurst_close_rolling, errors="coerce").fillna(0.5).astype(float)
    r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float)
    return (h * r2).rename("hurst_x_trend_r2").to_frame()


@register_feature("compute_evt_x_trend_r2", category="interaction")
def compute_evt_x_trend_r2(
    df: pd.DataFrame,
    evt_col: str = "evt_tail_shape",
    trend_r2_col: str = "trend_r2_20",
) -> pd.Series:
    """
    计算 EVT 尾部风险 × 趋势强度交互项
    
    Args:
        df: DataFrame with base features
        evt_col: EVT tail shape column
        trend_r2_col: Trend R² column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(evt_col, pd.Series(0.3, index=df.index))
    momentum = df.get(trend_r2_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0.3) * momentum.fillna(0)).rename("evt_x_trend_r2")


@register_feature("compute_evt_x_trend_r2_from_series", category="interaction")
def compute_evt_x_trend_r2_from_series(
    *,
    evt_tail_shape: pd.Series,
    trend_r2_20: pd.Series,
) -> pd.DataFrame:
    evt = pd.to_numeric(evt_tail_shape, errors="coerce").fillna(0.3).astype(float)
    r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float)
    return (evt * r2).rename("evt_x_trend_r2").to_frame()


@register_feature("compute_vpin_x_compression", category="interaction")
def compute_vpin_x_compression(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    compression_col: str = "compression_energy",
) -> pd.Series:
    """
    计算 VPIN × 压缩能量交互项
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        compression_col: Compression energy column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(compression_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_compression")


@register_feature("compute_vpin_x_compression_from_series", category="interaction")
def compute_vpin_x_compression_from_series(
    *,
    vpin: pd.Series,
    compression_energy: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    ce = pd.to_numeric(compression_energy, errors="coerce").fillna(0.0).astype(float)
    return (vp * ce).rename("vpin_x_compression").to_frame()


@register_feature("compute_vpin_x_trade_cluster_max_buy_run", category="interaction")
def compute_vpin_x_trade_cluster_max_buy_run(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    cluster_col: str = "trade_cluster_max_buy_run",
) -> pd.Series:
    """
    计算 VPIN × 最大连续买入长度交互项
    
    捕捉"高订单流不平衡 + 连续买入聚集"的组合信号
    可能表示知情交易者的策略性连续买入
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        cluster_col: Trade cluster max buy run column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(cluster_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_trade_cluster_max_buy_run")


@register_feature("compute_vpin_x_trade_cluster_max_buy_run_from_series", category="interaction")
def compute_vpin_x_trade_cluster_max_buy_run_from_series(
    *,
    vpin: pd.Series,
    trade_cluster_max_buy_run: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    run = pd.to_numeric(trade_cluster_max_buy_run, errors="coerce").fillna(0.0).astype(float)
    return (vp * run).rename("vpin_x_trade_cluster_max_buy_run").to_frame()


@register_feature("compute_vpin_zscore_x_trade_cluster_max_buy_run", category="interaction")
def compute_vpin_zscore_x_trade_cluster_max_buy_run(
    df: pd.DataFrame,
    vpin_zscore_col: str = "vpin_zscore_20",
    cluster_col: str = "trade_cluster_max_buy_run",
) -> pd.Series:
    """
    计算 VPIN Z-score × 最大连续买入长度交互项
    
    捕捉"异常高的订单流不平衡 + 连续买入聚集"的组合信号
    这是用户建议的交叉项，可能有超加成效应
    
    Args:
        df: DataFrame with base features
        vpin_zscore_col: VPIN Z-score column
        cluster_col: Trade cluster max buy run column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_zscore_col, pd.Series(0.0, index=df.index))
    momentum = df.get(cluster_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_zscore_x_trade_cluster_max_buy_run")


@register_feature("compute_vpin_zscore_x_trade_cluster_max_buy_run_from_series", category="interaction")
def compute_vpin_zscore_x_trade_cluster_max_buy_run_from_series(
    *,
    vpin_zscore_20: pd.Series,
    trade_cluster_max_buy_run: pd.Series,
) -> pd.DataFrame:
    vz = pd.to_numeric(vpin_zscore_20, errors="coerce").fillna(0.0).astype(float)
    run = pd.to_numeric(trade_cluster_max_buy_run, errors="coerce").fillna(0.0).astype(float)
    return (vz * run).rename("vpin_zscore_x_trade_cluster_max_buy_run").to_frame()


@register_feature("compute_vpin_signed_imbalance_x_trade_cluster_imbalance", category="interaction")
def compute_vpin_signed_imbalance_x_trade_cluster_imbalance(
    df: pd.DataFrame,
    vpin_signed_col: str = "vpin_signed_imbalance",
    cluster_imbalance_col: str = "trade_cluster_imbalance_ratio",
) -> pd.Series:
    """
    计算 VPIN Signed Imbalance × Trade Clustering Imbalance 交互项
    
    捕捉"订单流方向性 + 成交聚集方向性"的一致性
    两者方向一致时，信号更可靠
    
    Args:
        df: DataFrame with base features
        vpin_signed_col: VPIN signed imbalance column
        cluster_imbalance_col: Trade cluster imbalance ratio column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_signed_col, pd.Series(0.0, index=df.index))
    momentum = df.get(cluster_imbalance_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_signed_imbalance_x_trade_cluster_imbalance")


@register_feature("compute_vpin_signed_imbalance_x_trade_cluster_imbalance_from_series", category="interaction")
def compute_vpin_signed_imbalance_x_trade_cluster_imbalance_from_series(
    *,
    vpin_signed_imbalance: pd.Series,
    trade_cluster_imbalance_ratio: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin_signed_imbalance, errors="coerce").fillna(0.0).astype(float)
    imb = pd.to_numeric(trade_cluster_imbalance_ratio, errors="coerce").fillna(0.0).astype(float)
    return (vp * imb).rename("vpin_signed_imbalance_x_trade_cluster_imbalance").to_frame()


@register_feature("compute_vpin_x_trade_cluster_entropy", category="interaction")
def compute_vpin_x_trade_cluster_entropy(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    entropy_col: str = "trade_cluster_directional_entropy",
) -> pd.Series:
    """
    计算 VPIN × 方向熵交互项
    
    捕捉"订单流不平衡 + 成交混乱度"的组合
    高 VPIN + 低熵 = 大单主导且有序（知情交易）
    高 VPIN + 高熵 = 大单主导但混乱（可能假突破）
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        entropy_col: Trade cluster directional entropy column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(entropy_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_trade_cluster_entropy")


@register_feature("compute_vpin_x_trade_cluster_entropy_from_series", category="interaction")
def compute_vpin_x_trade_cluster_entropy_from_series(
    *,
    vpin: pd.Series,
    trade_cluster_directional_entropy: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    ent = pd.to_numeric(trade_cluster_directional_entropy, errors="coerce").fillna(0.0).astype(float)
    return (vp * ent).rename("vpin_x_trade_cluster_entropy").to_frame()


@register_feature("compute_sma_slope_x_price_pos", category="interaction")
def compute_sma_slope_x_price_pos(
    df: pd.DataFrame,
    sma_slope_col: str = "sma_200_slope",
    sma_col: str = "sma_200",
    close_col: str = "close",
) -> pd.Series:
    """
    计算均线斜率 × 价格位置交互项
    
    Args:
        df: DataFrame with base features
        sma_slope_col: SMA slope column
        sma_col: SMA column
        close_col: Close price column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(sma_slope_col, pd.Series(0.0, index=df.index))
    # 计算价格位置
    if sma_col in df.columns and close_col in df.columns:
        price_pos = (df[close_col] / df[sma_col].replace(0, np.nan)).fillna(1.0)
    else:
        price_pos = pd.Series(1.0, index=df.index)
    return (state.fillna(0) * price_pos).rename("sma_slope_x_price_pos")


@register_feature("compute_sma_slope_x_price_pos_from_series", category="interaction")
def compute_sma_slope_x_price_pos_from_series(
    *,
    sma_200_slope: pd.Series,
    sma_200: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    slope = pd.to_numeric(sma_200_slope, errors="coerce").fillna(0.0).astype(float)
    sma = pd.to_numeric(sma_200, errors="coerce").astype(float)
    cl = pd.to_numeric(close, errors="coerce").astype(float)
    price_pos = (cl / sma.replace(0, np.nan)).fillna(1.0)
    return (slope * price_pos).rename("sma_slope_x_price_pos").to_frame()


@register_feature("compute_vpin_x_wick_upper", category="interaction")
def compute_vpin_x_wick_upper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    wick_col: str = "wick_upper_ratio",
) -> pd.Series:
    """
    计算 VPIN × 上影线占比交互项（反转策略专用）
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        wick_col: Upper wick ratio column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(wick_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_wick_upper")


@register_feature("compute_vpin_x_wick_upper_from_series", category="interaction")
def compute_vpin_x_wick_upper_from_series(
    *,
    vpin: pd.Series,
    wick_upper_ratio: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    wr = pd.to_numeric(wick_upper_ratio, errors="coerce").fillna(0.0).astype(float)
    return (vp * wr).rename("vpin_x_wick_upper").to_frame()


@register_feature("compute_vpin_x_wick_lower", category="interaction")
def compute_vpin_x_wick_lower(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    wick_col: str = "wick_lower_ratio",
) -> pd.Series:
    """
    计算 VPIN × 下影线占比交互项（反转策略专用）
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        wick_col: Lower wick ratio column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(wick_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_wick_lower")


@register_feature("compute_vpin_x_wick_lower_from_series", category="interaction")
def compute_vpin_x_wick_lower_from_series(
    *,
    vpin: pd.Series,
    wick_lower_ratio: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    wr = pd.to_numeric(wick_lower_ratio, errors="coerce").fillna(0.0).astype(float)
    return (vp * wr).rename("vpin_x_wick_lower").to_frame()


@register_feature("apply_rank_transform_to_interaction", category="interaction")
def apply_rank_transform_to_interaction(
    df: pd.DataFrame,
    interaction_col: str,
    groupby_col: Optional[str] = None,
) -> pd.Series:
    """
    对单个交互项做 rank transform
    
    Args:
        df: DataFrame with interaction feature
        interaction_col: Interaction column name
        groupby_col: Optional column for cross-sectional rank
    
    Returns:
        Series with rank-transformed interaction feature
    """
    if interaction_col not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    
    if groupby_col and groupby_col in df.columns:
        # 横截面 rank（多标的场景）
        rank_series = df.groupby(groupby_col)[interaction_col].rank(pct=True, method="average")
    else:
        # 全局 rank（单标的时序场景）
        rank_series = df[interaction_col].rank(pct=True, method="average")
    
    return rank_series.fillna(0.5).rename(f"{interaction_col}_rank")


@register_feature("apply_rank_transform_to_interaction_from_series", category="interaction")
def apply_rank_transform_to_interaction_from_series(
    *,
    interaction: pd.Series,
) -> pd.DataFrame:
    """
    Narrow-IO rank transform for a single interaction series.
    Uses global rank(pct=True) and fills missing with 0.5, matching legacy behavior.
    """
    s = pd.to_numeric(interaction, errors="coerce").astype(float)
    ranked = s.rank(pct=True, method="average").fillna(0.5)
    return ranked.rename(f"{interaction.name or 'interaction'}_rank").to_frame()


# ========================================================================
# 衍生特征（Derived Features）：单个特征的变换或两个特征的其他运算
# ========================================================================

@register_feature("compute_sr_strength_combined", category="derived")
def compute_sr_strength_combined(
    df: pd.DataFrame,
    sqs_col: str = "sqs",
) -> pd.Series:
    """
    计算 SR 强度组合特征（单个特征的简单映射）
    
    Args:
        df: DataFrame with base features
        sqs_col: SQS column name
    
    Returns:
        Series with sr_strength_combined
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if sqs_col not in df.columns:
        raise ValueError(
            f"Required column '{sqs_col}' not found for sr_strength_combined. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return df[sqs_col].fillna(0.0).rename("sr_strength_combined")


@register_feature("compute_sr_strength_combined_from_series", category="derived")
def compute_sr_strength_combined_from_series(*, sqs: pd.Series) -> pd.DataFrame:
    """Narrow-IO entrypoint for sr_strength_combined."""
    s = pd.to_numeric(sqs, errors="coerce").fillna(0.0).astype(float)
    return s.rename("sr_strength_combined").to_frame()


@register_feature(
    "compute_sr_strength_combined_from_hal_sqs_from_series", category="derived"
)
def compute_sr_strength_combined_from_hal_sqs_from_series(
    *, sqs_hal_high: pd.Series, sqs_hal_low: pd.Series
) -> pd.DataFrame:
    """
    Robust Narrow-IO entrypoint for sr_strength_combined.

    Some pipelines may not materialize an intermediate `sqs` column (combined SQS),
    but do produce `sqs_hal_high` and `sqs_hal_low`. This function derives the
    combined strength directly as max(high, low), avoiding hard dependency on `sqs`.
    """
    h = pd.to_numeric(sqs_hal_high, errors="coerce").fillna(0.0).astype(float)
    l = pd.to_numeric(sqs_hal_low, errors="coerce").fillna(0.0).astype(float)
    out = pd.Series(np.maximum(h.values, l.values), index=h.index, name="sr_strength_combined")
    return out.to_frame()


@register_feature("compute_sr_distance_normalized", category="derived")
def compute_sr_distance_normalized(
    df: pd.DataFrame,
    dist_col: str = "dist_to_nearest_sr",
    atr_col: str = "atr",
) -> pd.Series:
    """
    计算 SR 距离归一化（两个特征的比值：dist / atr）
    
    Args:
        df: DataFrame with base features
        dist_col: Distance to nearest SR column
        atr_col: ATR column
    
    Returns:
        Series with sr_distance_normalized
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [dist_col, atr_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for sr_distance_normalized. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[dist_col] / df[atr_col].replace(0, np.nan)
    ).fillna(0.0).rename("sr_distance_normalized")


@register_feature("compute_sr_distance_normalized_from_series", category="derived")
def compute_sr_distance_normalized_from_series(
    *, dist_to_nearest_sr: pd.Series, atr: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for sr_distance_normalized."""
    dist = pd.to_numeric(dist_to_nearest_sr, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).replace(0, np.nan)
    out = (dist / atr_s).fillna(0.0).rename("sr_distance_normalized")
    return out.to_frame()


@register_feature("compute_dist_to_zz_high", category="derived")
def compute_dist_to_zz_high(
    df: pd.DataFrame,
    price_col: str = "close",
    zz_high_col: str = "zz_high_value",
) -> pd.Series:
    """
    计算到 ZigZag 高点的距离（两个特征的差值：abs(price - zz_high)）
    
    Args:
        df: DataFrame with base features
        price_col: Price column
        zz_high_col: ZigZag high value column
    
    Returns:
        Series with dist_to_zz_high
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [price_col, zz_high_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_high. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        (df[price_col] - df[zz_high_col]).abs()
    ).fillna(0.0).rename("dist_to_zz_high")


@register_feature("compute_dist_to_zz_high_from_series", category="derived")
def compute_dist_to_zz_high_from_series(
    *, close: pd.Series, zz_high_value: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_high."""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    z = pd.to_numeric(zz_high_value, errors="coerce").astype(float)
    return (c - z).abs().fillna(0.0).rename("dist_to_zz_high").to_frame()


@register_feature("compute_dist_to_zz_low", category="derived")
def compute_dist_to_zz_low(
    df: pd.DataFrame,
    price_col: str = "close",
    zz_low_col: str = "zz_low_value",
) -> pd.Series:
    """
    计算到 ZigZag 低点的距离（两个特征的差值：abs(price - zz_low)）
    
    Args:
        df: DataFrame with base features
        price_col: Price column
        zz_low_col: ZigZag low value column
    
    Returns:
        Series with dist_to_zz_low
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [price_col, zz_low_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_low. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        (df[price_col] - df[zz_low_col]).abs()
    ).fillna(0.0).rename("dist_to_zz_low")


@register_feature("compute_dist_to_zz_low_from_series", category="derived")
def compute_dist_to_zz_low_from_series(
    *, close: pd.Series, zz_low_value: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_low."""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    z = pd.to_numeric(zz_low_value, errors="coerce").astype(float)
    return (c - z).abs().fillna(0.0).rename("dist_to_zz_low").to_frame()


@register_feature("compute_dist_to_zz_high_atr", category="derived")
def compute_dist_to_zz_high_atr(
    df: pd.DataFrame,
    dist_col: str = "dist_to_zz_high",
    atr_col: str = "atr",
) -> pd.Series:
    """
    计算到 ZigZag 高点的距离（归一化到 ATR：dist / atr）
    
    Args:
        df: DataFrame with base features
        dist_col: Distance to ZZ high column
        atr_col: ATR column
    
    Returns:
        Series with dist_to_zz_high_atr
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [dist_col, atr_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_high_atr. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[dist_col] / df[atr_col].replace(0, np.nan)
    ).fillna(0.0).rename("dist_to_zz_high_atr")


@register_feature("compute_dist_to_zz_high_atr_from_series", category="derived")
def compute_dist_to_zz_high_atr_from_series(
    *, dist_to_zz_high: pd.Series, atr: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_high_atr."""
    dist = pd.to_numeric(dist_to_zz_high, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).replace(0, np.nan)
    return (dist / atr_s).fillna(0.0).rename("dist_to_zz_high_atr").to_frame()


@register_feature("compute_dist_to_zz_low_atr", category="derived")
def compute_dist_to_zz_low_atr(
    df: pd.DataFrame,
    dist_col: str = "dist_to_zz_low",
    atr_col: str = "atr",
) -> pd.Series:
    """
    计算到 ZigZag 低点的距离（归一化到 ATR：dist / atr）
    
    Args:
        df: DataFrame with base features
        dist_col: Distance to ZZ low column
        atr_col: ATR column
    
    Returns:
        Series with dist_to_zz_low_atr
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [dist_col, atr_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_low_atr. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[dist_col] / df[atr_col].replace(0, np.nan)
    ).fillna(0.0).rename("dist_to_zz_low_atr")


@register_feature("compute_dist_to_zz_low_atr_from_series", category="derived")
def compute_dist_to_zz_low_atr_from_series(
    *, dist_to_zz_low: pd.Series, atr: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_low_atr."""
    dist = pd.to_numeric(dist_to_zz_low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).replace(0, np.nan)
    return (dist / atr_s).fillna(0.0).rename("dist_to_zz_low_atr").to_frame()


@register_feature("compute_cvd_slope", category="derived")
def compute_cvd_slope(
    df: pd.DataFrame,
    cvd_col: str = "cvd",
    window: int = 5,
) -> pd.Series:
    """
    计算 CVD 斜率（单个特征的滚动变换）
    
    Args:
        df: DataFrame with base features
        cvd_col: CVD column
        window: Rolling window size
    
    Returns:
        Series with cvd_slope
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if cvd_col not in df.columns:
        raise ValueError(
            f"Required column '{cvd_col}' not found for cvd_slope_{window}. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    if len(df) <= window:
        raise ValueError(
            f"DataFrame length ({len(df)}) must be greater than window ({window}) for cvd_slope_{window}"
        )
    
    def _compute_slope(x):
        if len(x) > 1:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0.0
    
    return (
        df[cvd_col]
        .rolling(window=window, min_periods=1)
        .apply(_compute_slope)
        .fillna(0.0)
        .rename(f"cvd_slope_{window}")
    )


@register_feature("compute_cvd_slope_from_series", category="derived")
def compute_cvd_slope_from_series(*, cvd: pd.Series, window: int = 5) -> pd.DataFrame:
    """Narrow-IO entrypoint for cvd_slope_{window}."""
    s = pd.to_numeric(cvd, errors="coerce").astype(float)
    if len(s) <= window:
        # keep legacy behavior of raising in df version, but return zeros for short series to keep pipeline robust
        out = pd.Series(0.0, index=s.index, name=f"cvd_slope_{window}")
        return out.to_frame()

    def _compute_slope(x):
        if len(x) > 1:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0.0

    out = (
        s.rolling(window=window, min_periods=1)
        .apply(_compute_slope)
        .fillna(0.0)
        .rename(f"cvd_slope_{window}")
    )
    return out.to_frame()


@register_feature("compute_atr_ratio", category="derived")
def compute_atr_ratio(
    df: pd.DataFrame,
    atr_col: str = "atr",
    price_col: str = "close",
) -> pd.Series:
    """
    计算 ATR 比率（两个特征的比值：atr / price）
    
    Args:
        df: DataFrame with base features
        atr_col: ATR column
        price_col: Price column
    
    Returns:
        Series with atr_ratio
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [atr_col, price_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for atr_ratio. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[atr_col] / df[price_col].replace(0, np.nan)
    ).fillna(0.0).rename("atr_ratio")


@register_feature("compute_atr_ratio_from_series", category="derived")
def compute_atr_ratio_from_series(
    *,
    atr: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Narrow-input ATR ratio: atr / close."""
    atr = pd.to_numeric(atr, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    return (atr / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).rename(
        "atr_ratio"
    )


@register_feature("compute_bb_width_ratio", category="derived")
def compute_bb_width_ratio(
    df: pd.DataFrame,
    bb_upper_col: str = "bb_upper",
    bb_lower_col: str = "bb_lower",
    bb_middle_col: str = "bb_middle",
) -> pd.Series:
    """
    计算 Bollinger Band 宽度比率（多个特征的组合：(upper - lower) / middle）
    
    Args:
        df: DataFrame with base features
        bb_upper_col: BB upper column
        bb_lower_col: BB lower column
        bb_middle_col: BB middle column
    
    Returns:
        Series with bb_width_ratio
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    required_cols = [bb_upper_col, bb_lower_col, bb_middle_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for bb_width_ratio. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    bb_width = (
        (df[bb_upper_col] - df[bb_lower_col]) / df[bb_middle_col].replace(0, np.nan)
    ).fillna(0.0)
    return bb_width.rename("bb_width_ratio")


@register_feature("compute_bb_width_ratio_from_series", category="derived")
def compute_bb_width_ratio_from_series(
    *,
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    bb_middle: pd.Series,
) -> pd.Series:
    """Narrow-input BB width ratio: (upper - lower) / middle."""
    bb_upper = pd.to_numeric(bb_upper, errors="coerce").astype(float)
    bb_lower = pd.to_numeric(bb_lower, errors="coerce").astype(float)
    bb_middle = pd.to_numeric(bb_middle, errors="coerce").astype(float)
    out = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("bb_width_ratio")


@register_feature("compute_compression_score", category="derived")
def compute_compression_score(
    df: pd.DataFrame,
    bb_width_ratio_col: str = "bb_width_ratio",
) -> pd.Series:
    """
    计算压缩度分数（单个特征的变换：1 / (1 + bb_width_ratio)）
    
    Args:
        df: DataFrame with base features
        bb_width_ratio_col: BB width ratio column
    
    Returns:
        Series with compression_score
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if bb_width_ratio_col in df.columns:
        return (
            1.0 / (1.0 + df[bb_width_ratio_col])
        ).fillna(0.0).rename("compression_score")
    # 如果没有 bb_width_ratio，尝试从 BB 列计算
    bb_cols = ["bb_upper", "bb_lower", "bb_middle"]
    if all(col in df.columns for col in bb_cols):
        bb_width = (
            (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)
        ).fillna(0.0)
        return (1.0 / (1.0 + bb_width)).fillna(0.0).rename("compression_score")
    
    # 如果都不存在，报错
    raise ValueError(
        f"Required column '{bb_width_ratio_col}' or BB columns {bb_cols} not found for compression_score. "
        f"Available columns: {list(df.columns)[:20]}..."
    )


@register_feature("compute_compression_score_from_series", category="derived")
def compute_compression_score_from_series(*, bb_width_ratio: pd.Series) -> pd.Series:
    """Narrow-input compression_score: 1 / (1 + bb_width_ratio)."""
    bb_width_ratio = pd.to_numeric(bb_width_ratio, errors="coerce").astype(float)
    out = 1.0 / (1.0 + bb_width_ratio)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("compression_score")


@register_feature("compute_tbr_ma", category="derived")
def compute_tbr_ma(
    df: pd.DataFrame,
    tbr_col: str = "taker_buy_ratio",
    window: int = 5,
) -> pd.Series:
    """
    计算 TBR 移动平均（单个特征的滚动变换）
    
    Args:
        df: DataFrame with base features
        tbr_col: TBR column
        window: Moving average window
    
    Returns:
        Series with tbr_ma
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if tbr_col not in df.columns:
        raise ValueError(
            f"Required column '{tbr_col}' not found for tbr_ma_{window}. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[tbr_col].rolling(window=window, min_periods=1).mean()
    ).fillna(0.5).rename(f"tbr_ma_{window}")


@register_feature("compute_tbr_ma_from_series", category="derived")
def compute_tbr_ma_from_series(*, taker_buy_ratio: pd.Series, window: int = 5) -> pd.Series:
    """Narrow-input TBR moving average."""
    tbr = pd.to_numeric(taker_buy_ratio, errors="coerce").astype(float)
    out = tbr.rolling(window=window, min_periods=1).mean()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5).rename(f"tbr_ma_{window}")


@register_feature("compute_tbr_spike", category="derived")
def compute_tbr_spike(
    df: pd.DataFrame,
    tbr_col: str = "taker_buy_ratio",
    tbr_ma_col: str = "tbr_ma_5",
    spike_threshold: float = 1.5,
) -> pd.Series:
    """
    计算 TBR 突增信号（两个特征的比较：tbr > tbr_ma * threshold）
    
    Args:
        df: DataFrame with base features
        tbr_col: TBR column
        tbr_ma_col: TBR moving average column
        spike_threshold: Spike threshold multiplier
    
    Returns:
        Series with tbr_spike
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if tbr_col not in df.columns:
        raise ValueError(
            f"Required column '{tbr_col}' not found for tbr_spike. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    
    if tbr_ma_col in df.columns:
        tbr_ma = df[tbr_ma_col]
    else:
        # 如果没有 tbr_ma，尝试计算
        tbr_ma = df[tbr_col].rolling(window=5, min_periods=1).mean()
    
    spike = (df[tbr_col] > tbr_ma * spike_threshold).astype(float)
    return spike.rename("tbr_spike")


@register_feature("compute_tbr_spike_from_series", category="derived")
def compute_tbr_spike_from_series(
    *,
    taker_buy_ratio: pd.Series,
    tbr_ma_5: pd.Series,
    spike_threshold: float = 1.5,
) -> pd.Series:
    """Narrow-input TBR spike: taker_buy_ratio > tbr_ma_5 * threshold."""
    tbr = pd.to_numeric(taker_buy_ratio, errors="coerce").astype(float)
    ma = pd.to_numeric(tbr_ma_5, errors="coerce").astype(float)
    out = (tbr > ma * float(spike_threshold)).astype(float)
    out.name = "tbr_spike"
    return out


# ========================================================================
# 向后兼容：保留旧的批量函数（但推荐使用独立函数）
# ========================================================================


# 向后兼容：保留旧的批量函数（但推荐使用独立函数）
def build_interaction_features(
    df: pd.DataFrame,
    interaction_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    构建特征交互项（批量版本，向后兼容）
    
    推荐：使用独立的计算函数，在 feature_dependencies.yaml 中单独定义
    """
    df = df.copy()
    
    # 使用独立函数计算
    if "liquidity_void_detected" in df.columns or "wpt_false_breakout_risk" in df.columns:
        df["liquidity_void_x_wpt_risk"] = compute_liquidity_void_x_wpt_risk(df)
    
    if "compression_energy" in df.columns or "ofi_short" in df.columns:
        df["compression_energy_x_ofi_short"] = compute_compression_energy_x_ofi_short(df)
    
    if "hurst_close_rolling" in df.columns or "trend_r2_20" in df.columns:
        df["hurst_x_trend_r2"] = compute_hurst_x_trend_r2(df)
    
    if "evt_tail_shape" in df.columns or "trend_r2_20" in df.columns:
        df["evt_x_trend_r2"] = compute_evt_x_trend_r2(df)
    
    if "vpin" in df.columns:
        if "compression_energy" in df.columns:
            df["vpin_x_compression"] = compute_vpin_x_compression(df)
        if "wick_upper_ratio" in df.columns:
            df["vpin_x_wick_upper"] = compute_vpin_x_wick_upper(df)
        if "wick_lower_ratio" in df.columns:
            df["vpin_x_wick_lower"] = compute_vpin_x_wick_lower(df)
    
    if "sma_200_slope" in df.columns or "sma_200" in df.columns:
        df["sma_slope_x_price_pos"] = compute_sma_slope_x_price_pos(df)
    
    return df


def extract_interaction_features(
    df: pd.DataFrame,
    interaction_config: Optional[dict] = None,
    apply_rank: bool = True,
    groupby_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    提取特征交互项（完整流程，向后兼容）
    
    推荐：使用独立的计算函数，在 feature_dependencies.yaml 中单独定义
    """
    df = build_interaction_features(df, interaction_config)
    
    if apply_rank:
        interaction_cols = [col for col in df.columns if "_x_" in col]
        for col in interaction_cols:
            df[f"{col}_rank"] = apply_rank_transform_to_interaction(df, col, groupby_col)
    
    return df


@register_feature("compute_is_near_sr", category="derived")
def compute_is_near_sr(
    df: pd.DataFrame,
    dist_col: str = "dist_to_nearest_sr",
    atr_col: str = "atr",
    price_col: str = "close",
    dist_atr_mult: float = 1.5,
) -> pd.Series:
    """
    计算是否在SR附近的布尔列。
    
    基于 dist_to_nearest_sr 和 ATR，判断当前价格是否在SR附近（距离 <= dist_atr_mult * ATR）。
    
    注意：dist_to_nearest_sr 是相对百分比（如 0.05 表示 5%），需要转换为绝对价格距离后再与 ATR 比较。
    
    Args:
        df: DataFrame with base features
        dist_col: Distance to nearest SR column (default: "dist_to_nearest_sr")
        atr_col: ATR column (default: "atr")
        price_col: Price column for converting percentage to absolute distance (default: "close")
        dist_atr_mult: Distance threshold in ATR multiples (default: 1.5)
    
    Returns:
        Series with is_near_sr (boolean)
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [dist_col, atr_col, price_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for is_near_sr. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    
    # dist_to_nearest_sr 是相对百分比（如 0.05 表示 5%）
    dist_to_sr_pct = df[dist_col].abs()
    atr = df[atr_col].fillna(df[atr_col].median())
    price = df[price_col]
    
    # 将百分比距离转换为绝对价格距离
    # 例如：dist_to_sr = 0.05 (5%), price = 100 -> abs_distance = 5
    abs_distance = dist_to_sr_pct * price
    
    # 计算归一化距离（单位：ATR）
    # 例如：abs_distance = 5, atr = 10 -> dist_normalized = 0.5 ATR
    dist_normalized = abs_distance / (atr + 1e-8)
    
    # 判断是否在SR附近
    is_near = dist_normalized <= dist_atr_mult
    
    return is_near.fillna(False).astype(bool).rename("is_near_sr")


@register_feature("compute_is_near_sr_from_series", category="derived")
def compute_is_near_sr_from_series(
    *,
    dist_to_nearest_sr: pd.Series,
    atr: pd.Series,
    close: pd.Series,
    dist_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Narrow-IO entrypoint for is_near_sr.
    
    注意：dist_to_nearest_sr 是相对百分比，需要转换为绝对价格距离后再与 ATR 比较。
    """
    dist_pct = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs()
    atr_s = pd.to_numeric(atr, errors="coerce").fillna(atr.median())
    price = pd.to_numeric(close, errors="coerce")
    
    # 将百分比距离转换为绝对价格距离
    abs_distance = dist_pct * price
    
    # 计算归一化距离（单位：ATR）
    dist_normalized = abs_distance / (atr_s + 1e-8)
    
    # 判断是否在SR附近
    is_near = (dist_normalized <= dist_atr_mult).fillna(False).astype(bool)
    return is_near.rename("is_near_sr").to_frame()
