"""
FailedBreakoutFade Archetype 专用特征模块

设计理念：
- 假突破检测：突破尝试失败，价格快速回到 SR 附近
- Wick 拒绝：长影线表示价格被市场拒绝
- Fade 入场：反向入场捕捉假突破后的回归

核心输出：
1. fbf_score_false_breakout: 假突破强度 [0-1]
2. fbf_score_rejection: 拒绝信号强度 [0-1]
3. fbf_score_fade: Fade 入场质量 [0-1]

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
WICK_REJECTION_THRESHOLD = 0.4    # Wick 占比阈值
SR_PROXIMITY_THRESHOLD = 0.3      # SR 距离阈值（百分位）
FALSE_BREAKOUT_ATR_MULT = 0.5     # 假突破判定 ATR 倍数

FEATURE_VERSION = "1.0"


# =============================================================================
# 🎯 主函数：Failed Breakout Fade 软分数
# =============================================================================

@register_feature(
    "compute_failed_breakout_fade_soft_phase_from_series",
    category="failed_breakout",
    description="FailedBreakoutFade soft phase scores",
    outputs=[
        # === ATOMIC: False Breakout 原子信号 ===
        "fbf_breakout_attempt",
        "fbf_breakout_failed",
        "fbf_price_return_speed",
        "fbf_liquidity_void_risk",
        # === ATOMIC: Rejection 原子信号 ===
        "fbf_wick_upper_ratio",
        "fbf_wick_lower_ratio",
        "fbf_wick_exhaustion",
        "fbf_vol_rejection",
        # === ATOMIC: Fade 原子信号 ===
        "fbf_reversal_momentum",
        "fbf_cvd_reversal",
        "fbf_sr_proximity",
        # === COMPOSITE: 组合分数 ===
        "fbf_score_false_breakout",
        "fbf_score_rejection",
        "fbf_score_fade",
        "fbf_score_total",
        # === CONTEXTUAL: 状态信号 ===
        "fbf_direction",
        "fbf_is_failed_breakout",
        "fbf_near_sr",
    ],
)
def compute_failed_breakout_fade_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    # 可选特征
    dist_to_nearest_sr: pd.Series = None,
    liquidity_void_false_breakout_risk: pd.Series = None,
    cvd_change_5: pd.Series = None,
    sr_strength_max: pd.Series = None,
    wick_exhaustion_score: pd.Series = None,
    # 参数
    lookback: int = 20,
) -> pd.DataFrame:
    """
    FailedBreakoutFade 软阶段分数
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        open_: 开盘价序列
        volume: 成交量序列
        atr: ATR 序列
        dist_to_nearest_sr: 到最近 SR 的距离（可选）
        liquidity_void_false_breakout_risk: 假突破风险（可选）
        cvd_change_5: CVD 5周期变化（可选）
        sr_strength_max: SR 强度最大值（可选）
        wick_exhaustion_score: Wick 力竭分数（可选）
        lookback: 检测窗口
    
    Returns:
        DataFrame with failed breakout fade soft phase scores
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    open_s = pd.to_numeric(open_, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    n = len(close)
    eps = 1e-8
    
    # ========== 1️⃣ False Breakout: 假突破检测 ==========
    # 突破尝试：价格突破近期高点/低点
    rolling_high = high.rolling(lookback, min_periods=1).max().shift(1)
    rolling_low = low.rolling(lookback, min_periods=1).min().shift(1)
    
    breakout_up_attempt = (high > rolling_high).astype(float)
    breakout_down_attempt = (low < rolling_low).astype(float)
    fbf_breakout_attempt = breakout_up_attempt + breakout_down_attempt
    
    # 突破失败：价格回到区间内
    close_below_high = close < rolling_high
    close_above_low = close > rolling_low
    failed_up = (breakout_up_attempt == 1) & close_below_high
    failed_down = (breakout_down_attempt == 1) & close_above_low
    fbf_breakout_failed = (failed_up | failed_down).astype(float)
    
    # 价格回归速度
    distance_from_extreme = np.where(
        high > rolling_high,
        high - close,
        np.where(low < rolling_low, close - low, 0)
    )
    fbf_price_return_speed = (distance_from_extreme / atr_s.clip(lower=eps)).clip(0, 2) / 2
    fbf_price_return_speed = pd.Series(fbf_price_return_speed, index=close.index)
    
    # 流动性真空假突破风险
    if liquidity_void_false_breakout_risk is not None:
        lv_risk = pd.to_numeric(liquidity_void_false_breakout_risk, errors="coerce").fillna(0).clip(0, 1)
    else:
        lv_risk = pd.Series(0.5, index=close.index)
    fbf_liquidity_void_risk = lv_risk
    
    # 综合假突破分数
    fbf_score_false_breakout = (
        fbf_breakout_failed * 0.3 +
        fbf_price_return_speed * 0.25 +
        fbf_liquidity_void_risk * 0.25 +
        fbf_breakout_attempt * 0.2
    ).clip(0, 1)
    
    # ========== 2️⃣ Rejection: 拒绝信号 ==========
    # Wick 比例
    full_range = (high - low).clip(lower=eps)
    body_top = np.maximum(close, open_s)
    body_bottom = np.minimum(close, open_s)
    
    upper_wick = high - body_top
    lower_wick = body_bottom - low
    
    fbf_wick_upper_ratio = (upper_wick / full_range).clip(0, 1)
    fbf_wick_lower_ratio = (lower_wick / full_range).clip(0, 1)
    
    # Wick 力竭（使用提供的分数或计算）
    if wick_exhaustion_score is not None:
        fbf_wick_exhaustion = pd.to_numeric(wick_exhaustion_score, errors="coerce").fillna(0).clip(0, 1)
    else:
        # 简化计算：wick 比例高 + 收盘在 wick 反方向
        upper_exhaustion = fbf_wick_upper_ratio * (close < open_s).astype(float)
        lower_exhaustion = fbf_wick_lower_ratio * (close > open_s).astype(float)
        fbf_wick_exhaustion = (upper_exhaustion + lower_exhaustion).clip(0, 1)
    
    # 成交量拒绝（放量但价格不涨/跌）
    vol_ma = volume.rolling(lookback, min_periods=1).mean()
    vol_ratio = volume / vol_ma.clip(lower=eps)
    price_change = (close - open_s).abs() / atr_s.clip(lower=eps)
    vol_rejection = ((vol_ratio > 1.2) & (price_change < 0.5)).astype(float)
    fbf_vol_rejection = vol_rejection * vol_ratio.clip(0, 2) / 2
    
    # 综合拒绝分数
    fbf_score_rejection = (
        fbf_wick_exhaustion * 0.4 +
        (fbf_wick_upper_ratio + fbf_wick_lower_ratio) / 2 * 0.3 +
        fbf_vol_rejection * 0.3
    ).clip(0, 1)
    
    # ========== 3️⃣ Fade: 反向入场信号 ==========
    # 反转动量
    reversal = -np.sign(close - close.shift(1)) * np.sign(close.shift(1) - close.shift(2))
    reversal_strength = (close - close.shift(1)).abs() / atr_s.clip(lower=eps)
    fbf_reversal_momentum = (reversal > 0).astype(float) * reversal_strength.clip(0, 2) / 2
    
    # CVD 反转
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        cvd_reversal = -np.sign(cvd) * np.sign(close - close.shift(1))
        fbf_cvd_reversal = (cvd_reversal > 0).astype(float) * (cvd.abs() / cvd.abs().rolling(20).mean().clip(lower=eps)).clip(0, 2) / 2
    else:
        fbf_cvd_reversal = pd.Series(0.5, index=close.index)
    
    # SR 接近度
    if dist_to_nearest_sr is not None:
        dist = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().fillna(1)
        fbf_sr_proximity = (1 - dist.clip(0, 0.05) / 0.05).clip(0, 1)
    else:
        fbf_sr_proximity = pd.Series(0.5, index=close.index)
    
    # 综合 Fade 分数
    fbf_score_fade = (
        fbf_reversal_momentum * 0.35 +
        fbf_cvd_reversal * 0.35 +
        fbf_sr_proximity * 0.3
    ).clip(0, 1)
    
    # ========== 综合分数 ==========
    fbf_score_total = (
        fbf_score_false_breakout * 0.4 +
        fbf_score_rejection * 0.35 +
        fbf_score_fade * 0.25
    ).clip(0, 1)
    
    # ========== 状态信号 ==========
    # 假突破方向（1=做空机会，-1=做多机会）
    fbf_direction = np.where(
        failed_up, -1,
        np.where(failed_down, 1, 0)
    )
    fbf_direction = pd.Series(fbf_direction, index=close.index)
    
    fbf_is_failed_breakout = (fbf_score_false_breakout > 0.5).astype(float)
    fbf_near_sr = (fbf_sr_proximity > 0.5).astype(float)
    
    # ========== 输出 ==========
    result = pd.DataFrame({
        # === ATOMIC: False Breakout ===
        "fbf_breakout_attempt": fbf_breakout_attempt,
        "fbf_breakout_failed": fbf_breakout_failed,
        "fbf_price_return_speed": fbf_price_return_speed,
        "fbf_liquidity_void_risk": fbf_liquidity_void_risk,
        # === ATOMIC: Rejection ===
        "fbf_wick_upper_ratio": fbf_wick_upper_ratio,
        "fbf_wick_lower_ratio": fbf_wick_lower_ratio,
        "fbf_wick_exhaustion": fbf_wick_exhaustion,
        "fbf_vol_rejection": fbf_vol_rejection,
        # === ATOMIC: Fade ===
        "fbf_reversal_momentum": fbf_reversal_momentum,
        "fbf_cvd_reversal": fbf_cvd_reversal,
        "fbf_sr_proximity": fbf_sr_proximity,
        # === COMPOSITE ===
        "fbf_score_false_breakout": fbf_score_false_breakout,
        "fbf_score_rejection": fbf_score_rejection,
        "fbf_score_fade": fbf_score_fade,
        "fbf_score_total": fbf_score_total,
        # === CONTEXTUAL ===
        "fbf_direction": fbf_direction,
        "fbf_is_failed_breakout": fbf_is_failed_breakout,
        "fbf_near_sr": fbf_near_sr,
    }, index=close.index)
    
    result.attrs['feature_version'] = FEATURE_VERSION
    return result


# =============================================================================
# 🧩 失败信号特征
# =============================================================================

@register_feature(
    "compute_failed_breakout_fade_failure_from_series",
    category="failed_breakout",
    description="FailedBreakoutFade failure signals for tree model discovery",
    outputs=[
        "fbf_true_breakout_risk",
        "fbf_weak_rejection",
        "fbf_sr_not_holding",
        "fbf_failure_score",
    ],
)
def compute_failed_breakout_fade_failure_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    fbf_score_false_breakout: pd.Series,
    fbf_score_rejection: pd.Series,
    fbf_sr_proximity: pd.Series,
    sr_strength_max: pd.Series = None,
    lookback: int = 10,
) -> pd.DataFrame:
    """
    FailedBreakoutFade 失败信号：供树模型发现语义不成立的条件
    
    - 真突破风险：突破可能是真的，不是假的
    - 弱拒绝：拒绝信号不够强
    - SR 不守：SR 被有效突破
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    fb_score = pd.to_numeric(fbf_score_false_breakout, errors="coerce").fillna(0.5).clip(0, 1)
    rej_score = pd.to_numeric(fbf_score_rejection, errors="coerce").fillna(0.5).clip(0, 1)
    sr_prox = pd.to_numeric(fbf_sr_proximity, errors="coerce").fillna(0.5).clip(0, 1)
    
    # 真突破风险：价格持续远离，没有回归
    rolling_high = high.rolling(lookback, min_periods=1).max()
    rolling_low = low.rolling(lookback, min_periods=1).min()
    distance_from_range = np.maximum(
        (high - rolling_high.shift(1)).clip(lower=0),
        (rolling_low.shift(1) - low).clip(lower=0)
    )
    distance_atr = distance_from_range / atr_s.clip(lower=1e-8)
    fbf_true_breakout_risk = distance_atr.rolling(lookback // 2).mean().clip(0, 2) / 2
    
    # 弱拒绝：拒绝分数低
    fbf_weak_rejection = (1 - rej_score).clip(0, 1)
    
    # SR 不守：SR 被有效突破
    if sr_strength_max is not None:
        sr_strength = pd.to_numeric(sr_strength_max, errors="coerce").fillna(0.5).clip(0, 1)
        sr_weak = sr_strength < 0.5
        sr_far = sr_prox < 0.3
        fbf_sr_not_holding = (sr_weak | sr_far).astype(float)
    else:
        fbf_sr_not_holding = (sr_prox < 0.3).astype(float)
    
    # 综合失败分数
    fbf_failure_score = (
        fbf_true_breakout_risk * 0.4 +
        fbf_weak_rejection * 0.35 +
        fbf_sr_not_holding * 0.25
    ).clip(0, 1)
    
    return pd.DataFrame({
        "fbf_true_breakout_risk": fbf_true_breakout_risk,
        "fbf_weak_rejection": fbf_weak_rejection,
        "fbf_sr_not_holding": fbf_sr_not_holding,
        "fbf_failure_score": fbf_failure_score,
    }, index=close.index)
