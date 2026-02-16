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


# =============================================================================
# 🚀 ME Gate 专属特征：Breakout 结构语义
# =============================================================================

@register_feature(
    "compute_me_gate_features_from_series",
    category="momentum_expansion",
    description="ME Gate features: breakout structure quality for failure detection",
    outputs=[
        # === 扩张质量类 ===
        "me_impulse_ratio",           # breakout_bar_range / ATR (真突破 > 1.5)
        "me_cps",                     # close position strength (收盘强度)
        "me_multi_bar_acceleration",  # 多根K线加速度
        # === 区间破坏强度 ===
        "me_escape_dist",             # (close - prior_range_high) / prior_range_height
        "me_compression_depth",       # rolling_volatility_percentile (压缩程度)
        # === 延续潜力类 ===
        "me_air_pocket_score",        # distance_to_next_resistance / ATR
        # === 订单流一致性 ===
        "me_flow_alignment",          # sign(delta_cvd) == breakout_dir
        "me_aggression_ratio",        # buy_aggression vs sell_aggression
        # === 动量质量 ===
        "me_slope_consistency",       # momentum slope consistency
        "me_pullback_depth_after_break",  # 突破后回撤深度
        # === 组合分数 ===
        "me_gate_breakout_quality",   # 综合突破质量分数
        "me_gate_structure_score",    # 综合结构分数
    ],
)
def compute_me_gate_features_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series = None,
    volume: pd.Series,
    atr: pd.Series,
    # 可选订单流特征
    cvd_change_5: pd.Series = None,
    delta: pd.Series = None,
    # 可选 SR 特征
    hal_high: pd.Series = None,
    hal_low: pd.Series = None,
    # 参数
    lookback: int = 20,
    compression_window: int = 40,
) -> pd.DataFrame:
    """
    ME Gate 专属特征：用于识别突破结构质量，检测 failure
    
    设计理念：
    - BPC 有 pullback 结构特征，ME 需要 breakout 结构特征
    - 核心问题：ME 的失败是"结构错误"，不是"方向错误"
    - 需要捕捉的语义：扩张质量、区间破坏强度、延续潜力、订单流一致性
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        open_: 开盘价序列（可选，用于CPS）
        volume: 成交量序列
        atr: ATR 序列
        cvd_change_5: CVD 5周期变化（可选）
        delta: Delta（买卖差值，可选）
        hal_high: HAL 高点（可选，用于空间计算）
        hal_low: HAL 低点（可选，用于空间计算）
        lookback: 基础回看窗口
        compression_window: 压缩计算窗口
    
    Returns:
        DataFrame with ME Gate features
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    if open_ is not None:
        open_s = pd.to_numeric(open_, errors="coerce").astype(float)
    else:
        open_s = close.shift(1).fillna(close)
    
    n = len(close)
    eps = 1e-8
    
    # ========== 1️⃣ 扩张质量类 ==========
    
    # 1.1 Impulse Ratio: breakout_bar_range / ATR
    # 真突破通常 > 1.5 ATR，假突破 < 1.0 ATR
    bar_range = high - low
    me_impulse_ratio = (bar_range / atr_s.clip(lower=eps)).clip(0, 5)
    # 归一化到 0-1：使用 sigmoid 映射，1.5 ATR 对应 0.5
    me_impulse_ratio_norm = 1 / (1 + np.exp(-2 * (me_impulse_ratio - 1.5)))
    
    # 1.2 CPS (Close Position Strength): (close - low) / (high - low)
    # 参考 sr_strength_max_f 的设计：有 sr 位置和 sr 质量
    # CPS = 收盘在当日区间的位置，1.0 = 强势收盘（收在最高点）
    bar_range_safe = bar_range.clip(lower=eps)
    me_cps = ((close - low) / bar_range_safe).clip(0, 1)
    
    # 1.3 Multi-bar Acceleration: 多根K线加速度
    # 比较最近 lookback/2 根K线的平均涨幅 vs 前 lookback/2 根K线
    returns = close.pct_change().fillna(0)
    recent_returns = returns.rolling(lookback // 2, min_periods=1).mean()
    prior_returns = returns.shift(lookback // 2).rolling(lookback // 2, min_periods=1).mean().fillna(0)
    accel_raw = recent_returns - prior_returns
    # 使用 ATR 归一化的百分位，保持方差
    accel_atr_normalized = accel_raw / atr_s.rolling(lookback).mean().clip(lower=eps)
    accel_percentile = _stream_safe_percentile(accel_atr_normalized.fillna(0), compression_window)
    me_multi_bar_acceleration = accel_percentile.clip(0, 1)
    
    # ========== 2️⃣ 区间破坏强度 ==========
    
    # 2.1 Escape Distance: (close - prior_range_high) / prior_range_height
    # 衡量价格"逃离"之前区间的距离
    rolling_high = high.rolling(lookback, min_periods=1).max().shift(1)
    rolling_low = low.rolling(lookback, min_periods=1).min().shift(1)
    prior_range_height = (rolling_high - rolling_low).clip(lower=eps)
    
    # 做多方向的逃离距离（以 ATR 为单位）
    escape_up = (close - rolling_high) / atr_s.clip(lower=eps)
    # 做空方向的逃离距离
    escape_down = (rolling_low - close) / atr_s.clip(lower=eps)
    # 取绝对方向的最大逃离距离，保留原始值（以 ATR 为单位）
    escape_dist_raw = pd.Series(np.where(
        escape_up > escape_down,
        escape_up.clip(-2, 5),  # 允许负值（未突破）
        -escape_down.clip(-5, 2)
    ), index=close.index)
    # 使用百分位保持分布，而不是线性归一化
    me_escape_dist_norm = _stream_safe_percentile(escape_dist_raw, compression_window)
    
    # 2.2 Compression Depth: rolling_volatility_percentile
    # 突破前的压缩程度，压缩越深，突破后动能越强
    # 使用 ATR 归一化的波动率，而不是 rank
    rolling_vol = bar_range.rolling(lookback, min_periods=1).std()
    rolling_vol_normalized = rolling_vol / atr_s.clip(lower=eps)
    # 使用百分位，但窗口更长，保持历史趋势
    rolling_vol_pct = _stream_safe_percentile(rolling_vol_normalized, compression_window)
    me_compression_depth = 1 - rolling_vol_pct  # 反转：压缩程度越高，分数越高
    
    # ========== 3️⃣ 延续潜力类 ==========
    
    # 3.1 Air Pocket Score: distance_to_next_resistance / ATR
    # 衡量突破后的"空间"，即到下一个阻力/支撑的距离
    if hal_high is not None and hal_low is not None:
        hal_high_s = pd.to_numeric(hal_high, errors="coerce").fillna(0)
        hal_low_s = pd.to_numeric(hal_low, errors="coerce").fillna(0)
        # 做多：到上方阻力的距离
        dist_to_resistance = (hal_high_s - close).clip(lower=0) / atr_s.clip(lower=eps)
        # 做空：到下方支撑的距离
        dist_to_support = (close - hal_low_s).clip(lower=0) / atr_s.clip(lower=eps)
        # 取较小的距离（更近的 SR）
        me_air_pocket_score = np.minimum(dist_to_resistance, dist_to_support).clip(0, 5)
    else:
        # 无 SR 信息时，使用 rolling high/low 估算
        dist_to_rolling_high = (rolling_high - close).abs() / atr_s.clip(lower=eps)
        me_air_pocket_score = dist_to_rolling_high.clip(0, 5)
    # 归一化：2 ATR 距离 = 1.0（充足空间）
    me_air_pocket_score_norm = (me_air_pocket_score / 2).clip(0, 1)
    
    # ========== 4️⃣ 订单流一致性 ==========
    
    # 4.1 Flow Alignment: sign(delta_cvd) == breakout_dir
    # 订单流方向是否与突破方向一致
    price_direction = np.sign(close - close.shift(1)).fillna(0)
    
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        cvd_direction = np.sign(cvd)
        # 1 = 完全一致，0 = 不一致，0.5 = 中性（CVD 为 0）
        alignment_raw = (price_direction * cvd_direction)
        me_flow_alignment = (alignment_raw + 1) / 2  # 归一化到 0-1
    else:
        me_flow_alignment = pd.Series(0.5, index=close.index)
    
    # 4.2 Aggression Ratio: buy vs sell aggression
    if delta is not None:
        delta_s = pd.to_numeric(delta, errors="coerce").fillna(0)
        # 使用 delta 作为 aggression 的代理
        delta_ma = delta_s.rolling(lookback, min_periods=1).mean()
        delta_std = delta_s.rolling(lookback, min_periods=1).std().clip(lower=eps)
        me_aggression_ratio = ((delta_s - delta_ma) / delta_std).clip(-3, 3)
        me_aggression_ratio = (me_aggression_ratio / 3 + 1) / 2  # 归一化到 0-1
    else:
        # 无 delta 时，使用 volume 和 price change 估算
        vol_pct_change = volume.pct_change().fillna(0)
        price_pct_change = close.pct_change().fillna(0)
        # 价格上涨 + 放量 = 买入进攻，价格下跌 + 放量 = 卖出进攻
        aggression_raw = vol_pct_change * np.sign(price_pct_change)
        me_aggression_ratio = (aggression_raw.clip(-0.5, 0.5) + 0.5).clip(0, 1)
    
    # ========== 5️⃣ 动量质量 ==========
    
    # 5.1 Slope Consistency: momentum slope 的一致性
    # 连续上涨/下跌的斜率是否稳定
    returns_sign = np.sign(returns)
    sign_consistency = returns_sign.rolling(lookback // 2, min_periods=1).mean().abs()
    me_slope_consistency = sign_consistency.clip(0, 1)
    
    # 5.2 Pullback Depth After Break: 突破后的回撤深度
    # 健康的突破应该回撤浅（< 50%）
    max_high_5 = high.rolling(5, min_periods=1).max()
    min_low_5 = low.rolling(5, min_periods=1).min()
    recent_range = max_high_5 - min_low_5
    # 突破后回撤 = (recent_high - current_close) / recent_range
    pullback_from_high = (max_high_5 - close) / recent_range.clip(lower=eps)
    me_pullback_depth_after_break = pullback_from_high.clip(0, 1)
    # 反转：回撤越浅，分数越高
    me_pullback_depth_after_break = 1 - me_pullback_depth_after_break
    
    # ========== 6️⃣ 组合分数 ==========
    
    # 6.1 Breakout Quality: 综合突破质量
    # 高 impulse_ratio + 高 CPS + 高 flow_alignment = 高质量突破
    me_gate_breakout_quality = (
        me_impulse_ratio_norm * 0.35 +
        me_cps * 0.30 +
        me_flow_alignment * 0.35
    ).clip(0, 1)
    
    # 6.2 Structure Score: 综合结构分数
    # 考虑压缩深度、逃离距离、空间、动量一致性
    me_gate_structure_score = (
        me_compression_depth * 0.20 +
        me_escape_dist_norm * 0.25 +
        pd.Series(me_air_pocket_score_norm, index=close.index) * 0.20 +
        me_slope_consistency * 0.15 +
        me_pullback_depth_after_break * 0.20
    ).clip(0, 1)
    
    # ========== 输出 ==========
    return pd.DataFrame({
        # === 扩张质量类 ===
        "me_impulse_ratio": me_impulse_ratio_norm,
        "me_cps": me_cps,
        "me_multi_bar_acceleration": me_multi_bar_acceleration,
        # === 区间破坏强度 ===
        "me_escape_dist": me_escape_dist_norm,
        "me_compression_depth": me_compression_depth,
        # === 延续潜力类 ===
        "me_air_pocket_score": pd.Series(me_air_pocket_score_norm, index=close.index),
        # === 订单流一致性 ===
        "me_flow_alignment": me_flow_alignment,
        "me_aggression_ratio": me_aggression_ratio,
        # === 动量质量 ===
        "me_slope_consistency": me_slope_consistency,
        "me_pullback_depth_after_break": me_pullback_depth_after_break,
        # === 组合分数 ===
        "me_gate_breakout_quality": me_gate_breakout_quality,
        "me_gate_structure_score": me_gate_structure_score,
    }, index=close.index)
