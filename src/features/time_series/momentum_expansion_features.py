"""
MomentumExpansion Archetype 专用特征模块

设计理念：
- 动能扩张：波动率和成交量同时放大
- 区间突破：布林带/ATR 扩张
- 订单流加速：连续方向性成交 cluster
- "钱在加速流入"的信号

核心输出：
1. me_score_expansion: 扩张强度 [0-1]
2. me_score_acceleration: 加速度 [0-1]
3. me_score_orderflow: 订单流质量 [0-1]

规范遵循：
- 特征工程鲁棒性设计规范
- Archetype 语义化特征建模规范
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature


# =============================================================================
# 📌 常量定义
# =============================================================================

DEFAULT_LOOKBACK = 20
ATR_EXPANSION_THRESHOLD = 0.7     # ATR 百分位阈值（高于此为扩张）
BB_WIDTH_EXPANSION_THRESHOLD = 0.7
VOL_SURGE_THRESHOLD = 1.5         # 成交量放大阈值
VPIN_ACTIVE_THRESHOLD = 0.55      # VPIN 活跃阈值

FEATURE_VERSION = "1.0"


def _stream_safe_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    流式安全的百分位计算
    
    确保流式和批量计算一致：
    - 使用固定窗口 window
    - min_periods = window（确保窗口内数据足够）
    - 窗口不足时返回 0.5（中性值）
    """
    # 使用当前值在窗口内的分位数
    # 注意：min_periods=window 确保只有窗口完全填充时才计算
    result = series.rolling(window, min_periods=window).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x) if len(x) == window else 0.5,
        raw=False
    )
    # 填充窗口不足的部分
    result = result.fillna(0.5)
    return result


# =============================================================================
# 🎯 主函数：Momentum Expansion 软分数
# =============================================================================

@register_feature(
    "compute_momentum_expansion_soft_phase_from_series",
    category="momentum_expansion",
    description="MomentumExpansion soft phase scores",
    outputs=[
        # === ATOMIC: Expansion 原子信号 ===
        "me_atr_expansion",
        "me_bb_width_expansion",
        "me_range_expansion",
        "me_vol_surge",
        # === ATOMIC: Acceleration 原子信号 ===
        "me_price_acceleration",
        "me_vol_acceleration",
        "me_momentum_slope",
        # === ATOMIC: OrderFlow 原子信号 ===
        "me_vpin_active",
        "me_cvd_directional",
        "me_cluster_intensity",
        # === COMPOSITE: 组合分数 ===
        "me_score_expansion",
        "me_score_acceleration",
        "me_score_orderflow",
        "me_score_total",
        # === CONTEXTUAL: 状态信号 ===
        "me_direction",
        "me_is_expanding",
        "me_vol_ratio",
    ],
)
def compute_momentum_expansion_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    # 可选特征
    bb_width_normalized: pd.Series = None,
    cvd_change_5: pd.Series = None,
    vpin: pd.Series = None,
    trade_cluster_flow_intensity: pd.Series = None,
    # 参数
    lookback: int = 20,
    pct_window: int = 100,
) -> pd.DataFrame:
    """
    MomentumExpansion 软阶段分数
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        atr: ATR 序列
        bb_width_normalized: 布林带宽度归一化（可选）
        cvd_change_5: CVD 5周期变化（可选）
        vpin: VPIN 指标（可选）
        trade_cluster_flow_intensity: 交易聚集强度（可选）
        lookback: 检测窗口
        pct_window: 百分位计算窗口
    
    Returns:
        DataFrame with momentum expansion soft phase scores
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    n = len(close)
    eps = 1e-8
    
    # ========== 1️⃣ Expansion: 波动率/区间扩张 ==========
    # ATR 扩张（使用流式安全的百分位计算）
    atr_pct = _stream_safe_percentile(atr_s, pct_window)
    me_atr_expansion = atr_pct.clip(0, 1)
    
    # 布林带宽度扩张
    if bb_width_normalized is not None:
        bb_width = pd.to_numeric(bb_width_normalized, errors="coerce").fillna(0.5).clip(0, 1)
    else:
        # 计算简化版布林带宽度
        rolling_std = close.rolling(lookback).std()
        rolling_mean = close.rolling(lookback).mean().clip(lower=eps)
        bb_width_raw = (rolling_std * 2 / rolling_mean).fillna(0)
        bb_width = _stream_safe_percentile(bb_width_raw, pct_window)
    me_bb_width_expansion = bb_width.clip(0, 1)
    
    # 区间扩张
    bar_range = high - low
    range_pct = _stream_safe_percentile(bar_range, pct_window)
    me_range_expansion = range_pct.clip(0, 1)
    
    # 成交量放大
    vol_ma = volume.rolling(lookback, min_periods=1).mean()
    vol_ratio = volume / vol_ma.clip(lower=eps)
    me_vol_surge = (vol_ratio / VOL_SURGE_THRESHOLD).clip(0, 1)
    
    # 综合扩张分数
    me_score_expansion = (
        me_atr_expansion * 0.3 +
        me_bb_width_expansion * 0.25 +
        me_range_expansion * 0.25 +
        me_vol_surge * 0.2
    ).clip(0, 1)
    
    # ========== 2️⃣ Acceleration: 加速度 ==========
    # 价格加速度
    returns = close.pct_change()
    returns_ma = returns.rolling(lookback, min_periods=1).mean()
    returns_prev_ma = returns.shift(lookback).rolling(lookback, min_periods=1).mean()
    price_accel = (returns_ma - returns_prev_ma.fillna(0)).clip(-0.1, 0.1) / 0.1
    me_price_acceleration = price_accel.abs().clip(0, 1)
    
    # 成交量加速度
    vol_ma_short = volume.rolling(lookback // 2, min_periods=1).mean()
    vol_ma_long = volume.rolling(lookback, min_periods=1).mean()
    vol_accel = (vol_ma_short / vol_ma_long.clip(lower=eps) - 1).clip(-1, 1)
    me_vol_acceleration = (vol_accel + 1) / 2  # 归一化到 0-1
    
    # 动量斜率
    momentum = close - close.shift(lookback)
    momentum_slope = momentum.diff(lookback // 2) / atr_s.clip(lower=eps)
    me_momentum_slope = momentum_slope.clip(-2, 2).abs() / 2
    
    # 综合加速度分数
    me_score_acceleration = (
        me_price_acceleration * 0.35 +
        me_vol_acceleration * 0.35 +
        me_momentum_slope * 0.3
    ).clip(0, 1)
    
    # ========== 3️⃣ OrderFlow: 订单流质量 ==========
    # VPIN 活跃度
    if vpin is not None:
        vpin_s = pd.to_numeric(vpin, errors="coerce").fillna(0.5).clip(0, 1)
        me_vpin_active = (vpin_s / VPIN_ACTIVE_THRESHOLD).clip(0, 1)
    else:
        me_vpin_active = pd.Series(0.5, index=close.index)
    
    # CVD 方向性
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        cvd_std = cvd.rolling(lookback, min_periods=1).std().clip(lower=eps)
        cvd_z = (cvd / cvd_std).clip(-3, 3)
        me_cvd_directional = cvd_z.abs() / 3
    else:
        me_cvd_directional = pd.Series(0.5, index=close.index)
    
    # Cluster 强度
    if trade_cluster_flow_intensity is not None:
        cluster = pd.to_numeric(trade_cluster_flow_intensity, errors="coerce").fillna(0.5).clip(0, 1)
        me_cluster_intensity = cluster
    else:
        me_cluster_intensity = pd.Series(0.5, index=close.index)
    
    # 综合订单流分数
    me_score_orderflow = (
        me_vpin_active * 0.4 +
        me_cvd_directional * 0.35 +
        me_cluster_intensity * 0.25
    ).clip(0, 1)
    
    # ========== 综合分数 ==========
    me_score_total = (
        me_score_expansion * 0.4 +
        me_score_acceleration * 0.3 +
        me_score_orderflow * 0.3
    ).clip(0, 1)
    
    # ========== 状态信号 ==========
    me_direction = np.sign(close - close.shift(lookback)).fillna(0).astype(int)
    me_is_expanding = (me_score_expansion > ATR_EXPANSION_THRESHOLD).astype(float)
    
    # ========== 输出 ==========
    result = pd.DataFrame({
        # === ATOMIC: Expansion ===
        "me_atr_expansion": me_atr_expansion,
        "me_bb_width_expansion": me_bb_width_expansion,
        "me_range_expansion": me_range_expansion,
        "me_vol_surge": me_vol_surge,
        # === ATOMIC: Acceleration ===
        "me_price_acceleration": me_price_acceleration,
        "me_vol_acceleration": me_vol_acceleration,
        "me_momentum_slope": me_momentum_slope,
        # === ATOMIC: OrderFlow ===
        "me_vpin_active": me_vpin_active,
        "me_cvd_directional": me_cvd_directional,
        "me_cluster_intensity": me_cluster_intensity,
        # === COMPOSITE ===
        "me_score_expansion": me_score_expansion,
        "me_score_acceleration": me_score_acceleration,
        "me_score_orderflow": me_score_orderflow,
        "me_score_total": me_score_total,
        # === CONTEXTUAL ===
        "me_direction": me_direction,
        "me_is_expanding": me_is_expanding,
        "me_vol_ratio": vol_ratio,
    }, index=close.index)
    
    result.attrs['feature_version'] = FEATURE_VERSION
    return result


# =============================================================================
# 🧩 失败信号特征
# =============================================================================

@register_feature(
    "compute_momentum_expansion_failure_from_series",
    category="momentum_expansion",
    description="MomentumExpansion failure signals for tree model discovery",
    outputs=[
        "me_false_expansion",
        "me_vol_divergence",
        "me_orderflow_exhaustion",
        "me_failure_score",
    ],
)
def compute_momentum_expansion_failure_from_series(
    *,
    close: pd.Series,
    me_score_expansion: pd.Series,
    me_score_acceleration: pd.Series,
    me_score_orderflow: pd.Series,
    volume: pd.Series,
    cvd_change_5: pd.Series = None,
    shd_pct: pd.Series = None,
    lookback: int = 10,
) -> pd.DataFrame:
    """
    MomentumExpansion 失败信号：供树模型发现语义不成立的条件
    
    - 假扩张：波动扩张但成交量不配合
    - 成交量背离：价格加速但成交量萎缩
    - 订单流力竭：订单流强度下降
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    expansion = pd.to_numeric(me_score_expansion, errors="coerce").fillna(0.5).clip(0, 1)
    accel = pd.to_numeric(me_score_acceleration, errors="coerce").fillna(0.5).clip(0, 1)
    orderflow = pd.to_numeric(me_score_orderflow, errors="coerce").fillna(0.5).clip(0, 1)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    
    # 假扩张：波动扩张但成交量不配合
    vol_ma = volume.rolling(lookback, min_periods=1).mean()
    vol_weak = (volume / vol_ma.clip(lower=1e-8)) < 0.8
    me_false_expansion = ((expansion > 0.6) & vol_weak).astype(float)
    
    # 成交量背离：价格加速但成交量萎缩
    price_rising = close > close.shift(lookback)
    vol_declining = volume < volume.shift(lookback)
    me_vol_divergence = ((price_rising & vol_declining) | (~price_rising & ~vol_declining)).astype(float)
    me_vol_divergence = me_vol_divergence * accel  # 加权：加速度越高，背离越严重
    
    # 订单流力竭
    orderflow_declining = orderflow < orderflow.shift(lookback // 2)
    me_orderflow_exhaustion = (orderflow_declining & (accel > 0.5)).astype(float)
    
    # 综合失败分数
    me_failure_score = (
        me_false_expansion * 0.35 +
        me_vol_divergence * 0.35 +
        me_orderflow_exhaustion * 0.3
    ).clip(0, 1)
    
    return pd.DataFrame({
        "me_false_expansion": me_false_expansion,
        "me_vol_divergence": me_vol_divergence,
        "me_orderflow_exhaustion": me_orderflow_exhaustion,
        "me_failure_score": me_failure_score,
    }, index=close.index)


@register_feature(
    "compute_momentum_expansion_context_from_series",
    category="momentum_expansion",
    description="MomentumExpansion context: jump risk + reflexivity",
    outputs=[
        "me_jump_risk_suitable",
        "me_reflex_risk",
        "me_regime_suitable",
    ],
)
def compute_momentum_expansion_context_from_series(
    *,
    close: pd.Series,
    jump_risk_pct: pd.Series = None,
    shd_pct: pd.Series = None,
    ofci_pct: pd.Series = None,
) -> pd.DataFrame:
    """
    MomentumExpansion 上下文特征：跳空风险 + 反身性风险
    
    ME 需要：
    - 跳空风险适中 (0.6-0.9)：太低说明没动能，太高风险过大
    - 反身性风险不能过高：过度拥挤会导致反转
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    
    # 跳空风险适合度
    if jump_risk_pct is not None:
        jump = pd.to_numeric(jump_risk_pct, errors="coerce").fillna(0.5).clip(0, 1)
        # ME 需要跳空风险在 0.6-0.9 之间
        in_range = ((jump >= 0.6) & (jump <= 0.9)).astype(float)
        distance_to_center = 1 - 2 * np.abs(jump - 0.75)
        me_jump_risk_suitable = (in_range * distance_to_center.clip(0, 1)).clip(0, 1)
    else:
        me_jump_risk_suitable = pd.Series(0.5, index=close.index)
    
    # 反身性风险
    if shd_pct is not None and ofci_pct is not None:
        shd = pd.to_numeric(shd_pct, errors="coerce").fillna(0.5).clip(0, 1)
        ofci = pd.to_numeric(ofci_pct, errors="coerce").fillna(0.5).clip(0, 1)
        me_reflex_risk = np.maximum(shd, ofci)
    elif shd_pct is not None:
        me_reflex_risk = pd.to_numeric(shd_pct, errors="coerce").fillna(0.5).clip(0, 1)
    else:
        me_reflex_risk = pd.Series(0.5, index=close.index)
    
    # Regime 适合度
    me_regime_suitable = (me_jump_risk_suitable * (1 - me_reflex_risk)).clip(0, 1)
    
    return pd.DataFrame({
        "me_jump_risk_suitable": me_jump_risk_suitable,
        "me_reflex_risk": me_reflex_risk,
        "me_regime_suitable": me_regime_suitable,
    }, index=close.index)
