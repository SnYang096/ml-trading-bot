"""
BPC (Breakout-Pullback-Continuation) Archetype 专用特征模块

设计理念：
- 领域约束的失败发现
- Price tells you WHAT, Volume tells you IF, Order flow tells you WHO
- 软阶段概率代替硬阶段标签
- 每个阶段 = 价格信号 × 成交量确认 × 订单流验证

核心输出：
1. bpc_score_breakout: 突破强度 [0-1]
2. bpc_score_pullback: 回踩质量 [0-1]
3. bpc_score_continuation: 延续动能 [0-1]
4. bpc_score_neutral: 中性/蓄势 [0-1]

规范遵循：
- BPC三阶段语义化特征建模规范
- BPC阶段建模函数状态安全规范
- 特征工程鲁棒性设计规范
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from src.features.registry import register_feature
from src.features.time_series.utils_garch_features import (
    compute_ewma_vol,
    compute_ewma_vol_percentile,
)


# =============================================================================
# 📌 常量定义（便于调参和维护）
# =============================================================================

# --- P1: 行业标准参数（固定值） ---
DEFAULT_LOOKBACK_BREAKOUT = 20  # ATR 标准周期
DEFAULT_VOL_MA_WINDOW = 20  # 成交量均线窗口
DEFAULT_BREAKOUT_ATR_MULT = 1.0  # Donchian 标准

# --- P2: 关键动态参数（可自适应） ---
DEFAULT_PULLBACK_DECAY = 0.3  # 回踩质量衰减系数
PULLBACK_DECAY_MIN = 0.2  # 高波动时的快速衰减
PULLBACK_DECAY_MAX = 0.5  # 低波动时的慢速衰减
VOL_ADAPTIVE_WINDOW = 40  # 自适应波动率计算窗口

# --- P3: 业务阈值（可配置） ---
VOL_BREAKOUT_THRESHOLD = 1.5  # 突破放量阈值（均量倍数）
VOL_CONTINUATION_THRESHOLD = 1.2  # 续行放量阈值
VPIN_ACTIVE_THRESHOLD = 0.6  # VPIN 活跃阈值
CVD_ABSORPTION_THRESHOLD = 0.5  # CVD 吸收阈值（标准差）
BREAKOUT_TRIGGER_THRESHOLD = 0.3  # 突破触发阈值
PULLBACK_END_RATIO = 0.6  # Pullback 结束比例（峰值 * 0.6）
WEAK_BREAKOUT_THRESHOLD = 0.1  # 弱突破阈值

# --- 特征版本（用于元数据追溯） ---
FEATURE_VERSION = "2.1"


# =============================================================================
# 🎩 自适应参数函数
# =============================================================================


def _compute_adaptive_pullback_decay(volatility_pct: np.ndarray) -> np.ndarray:
    """
    根据波动率百分位计算自适应的 pullback_decay

    高波动 → 快速衰减（0.2）：市场敏感，回踩需要快速恢复
    低波动 → 慢速衰减（0.5）：市场稳定，允许慢回踩

    Args:
        volatility_pct: 波动率百分位数组 [0, 1]

    Returns:
        自适应的 pullback_decay 数组
    """
    # decay = 0.2 + 0.3 * (1 - volatility_pct)
    # 当 volatility_pct = 1.0 (高波动) → decay = 0.2
    # 当 volatility_pct = 0.0 (低波动) → decay = 0.5
    decay_range = PULLBACK_DECAY_MAX - PULLBACK_DECAY_MIN
    return PULLBACK_DECAY_MIN + decay_range * (1 - volatility_pct)


# =============================================================================
# 🎯 主函数：BPC 软阶段分数（完整版）
# =============================================================================


def _compute_soft_phase_core(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    volume: pd.Series,
    cvd_change_5: pd.Series = None,
    vpin: pd.Series = None,
    ofci_pct: pd.Series = None,
    bb_width_normalized: pd.Series = None,
    lookback_breakout: int = 20,
    breakout_atr_mult: float = 1.0,
    pullback_decay: float = 0.3,
    vol_ma_window: int = 20,
    gate_pullback_on_breakout: bool = True,
    ema_position: pd.Series = None,
    macro_vwap_position: pd.Series | None = None,
    tpc_semantic_reweight: bool = False,
    prefix: str = "bpc",
) -> pd.DataFrame:
    """Shared core for BPC and TPC soft phase scores.

    When ``gate_pullback_on_breakout=True`` (default, BPC behaviour):
      - ``is_after_breakout`` gates pullback/continuation scores
      - ``recent_direction`` is derived from Donchian breakout direction

    When ``gate_pullback_on_breakout=False`` (TPC behaviour):
      - ``is_after_breakout`` is always 1.0 (no breakout gate)
      - ``recent_direction`` comes from sign of ``ema_position``
      - If ``tpc_semantic_reweight`` and ``macro_vwap_position`` are set: rescales
        pullback vs continuation toward deeper VWAP discount and away from chop /
        VWAP stretch (see ``tpc_semantic_*`` outputs).
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)

    n = len(close)
    eps = 1e-8

    rolling_high = high.rolling(lookback_breakout, min_periods=1).max().shift(1)
    rolling_low = low.rolling(lookback_breakout, min_periods=1).min().shift(1)
    rolling_range = (rolling_high - rolling_low).clip(lower=eps)

    vol_ma = volume.rolling(vol_ma_window, min_periods=1).mean()
    vol_ratio = (volume / vol_ma.clip(lower=eps)).fillna(1.0)
    vol_pct = (
        volume.rolling(vol_ma_window * 2, min_periods=20).rank(pct=True).fillna(0.5)
    )

    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0.0)
        cvd_ma = cvd.rolling(vol_ma_window, min_periods=1).mean()
        cvd_std = cvd.rolling(vol_ma_window, min_periods=1).std().clip(lower=eps)
        cvd_z = ((cvd - cvd_ma) / cvd_std).fillna(0.0).clip(-3, 3)
    else:
        cvd_z = pd.Series(0.0, index=close.index)

    if vpin is not None:
        vpin_s = pd.to_numeric(vpin, errors="coerce").fillna(0.5).clip(0, 1)
    else:
        vpin_s = pd.Series(0.5, index=close.index)

    if ofci_pct is not None:
        ofci = pd.to_numeric(ofci_pct, errors="coerce").fillna(0.5)
    else:
        ofci = pd.Series(0.5, index=close.index)

    # ── 1. Breakout scores ──
    breakout_long_raw = (close - rolling_high) / (atr_s * breakout_atr_mult + eps)
    breakout_short_raw = (rolling_low - close) / (atr_s * breakout_atr_mult + eps)

    breakout_strength = np.maximum(
        breakout_long_raw.fillna(0).values, breakout_short_raw.fillna(0).values
    )
    breakout_strength = np.clip(breakout_strength, 0, 1)
    breakout_direction = np.where(
        breakout_long_raw.fillna(0).values > breakout_short_raw.fillna(0).values, 1, -1
    )

    vol_breakout_confirm = (vol_ratio / VOL_BREAKOUT_THRESHOLD).clip(0, 1).values

    cvd_z_values = cvd_z.values if hasattr(cvd_z, "values") else np.full(n, 0.0)
    vpin_values = vpin_s.values if hasattr(vpin_s, "values") else np.full(n, 0.5)

    cvd_breakout_confirm = np.where(
        breakout_direction > 0,
        (cvd_z_values > 0).astype(float) * (np.abs(cvd_z_values) / 2).clip(0, 1),
        (cvd_z_values < 0).astype(float) * (np.abs(cvd_z_values) / 2).clip(0, 1),
    )
    vpin_breakout_confirm = (vpin_values / VPIN_ACTIVE_THRESHOLD).clip(0, 1)

    bpc_score_breakout = (
        breakout_strength * 0.4
        + breakout_strength * vol_breakout_confirm * 0.3
        + breakout_strength * cvd_breakout_confirm * 0.2
        + breakout_strength * vpin_breakout_confirm * 0.1
    )
    bpc_score_breakout = np.clip(bpc_score_breakout, 0, 1)

    # ── 2. Pullback scores ──
    if gate_pullback_on_breakout:
        # BPC: require a recent Donchian breakout
        bpc_score_breakout_series = pd.Series(bpc_score_breakout, index=close.index)
        recent_breakout_strength = bpc_score_breakout_series.rolling(
            lookback_breakout, min_periods=1
        ).max()
        is_after_breakout = (
            (recent_breakout_strength > BREAKOUT_TRIGGER_THRESHOLD).astype(float).values
        )
        # Direction from Donchian breakout, forward-filled
        recent_direction = breakout_direction.copy()
        recent_direction_series = pd.Series(recent_direction, index=close.index)
        weak_mask = breakout_strength < WEAK_BREAKOUT_THRESHOLD
        recent_direction_series[weak_mask] = np.nan
        recent_direction = recent_direction_series.ffill().fillna(0).values.astype(int)
    else:
        # TPC: no breakout gate; direction from EMA position sign
        is_after_breakout = np.ones(n, dtype=float)
        if ema_position is not None:
            ema_pos = pd.to_numeric(ema_position, errors="coerce").fillna(0.0)
            recent_direction = np.sign(ema_pos.values).astype(int)
        else:
            recent_direction = np.sign(close.diff(lookback_breakout).fillna(0).values).astype(int)

    pullback_depth_long = ((rolling_high - close) / rolling_range).clip(0, 1).values
    pullback_depth_short = ((close - rolling_low) / rolling_range).clip(0, 1).values

    pullback_depth = np.where(
        recent_direction > 0, pullback_depth_long, pullback_depth_short
    )

    vol_pct_adaptive = (
        atr_s.rolling(VOL_ADAPTIVE_WINDOW, min_periods=20)
        .rank(pct=True)
        .fillna(0.5)
        .values
    )
    pullback_decay_adaptive = _compute_adaptive_pullback_decay(vol_pct_adaptive)
    pullback_quality = np.exp(-pullback_depth / pullback_decay_adaptive)

    vol_pct_values = vol_pct.values if hasattr(vol_pct, "values") else np.full(n, 0.5)
    vol_pullback_confirm = (1 - vol_pct_values).clip(0, 1)

    cvd_z_series = pd.Series(cvd_z_values, index=close.index)
    cvd_rolling_min = cvd_z_series.rolling(10, min_periods=1).min().values
    cvd_rolling_max = cvd_z_series.rolling(10, min_periods=1).max().values
    cvd_absorption = np.where(
        recent_direction > 0,
        ((cvd_z_values - cvd_rolling_min) > CVD_ABSORPTION_THRESHOLD).astype(float),
        ((cvd_rolling_max - cvd_z_values) > CVD_ABSORPTION_THRESHOLD).astype(float),
    )

    score_pullback = (
        is_after_breakout * pullback_quality * 0.3
        + is_after_breakout * pullback_quality * vol_pullback_confirm * 0.3
        + is_after_breakout * pullback_quality * cvd_absorption * 0.4
    )
    score_pullback = np.clip(score_pullback, 0, 1)

    # ── 3. Continuation scores ──
    pullback_low_series = (
        low.rolling(lookback_breakout // 2, min_periods=1).min().values
    )
    pullback_high_series = (
        high.rolling(lookback_breakout // 2, min_periods=1).max().values
    )

    atr_values = atr_s.values
    close_values = close.values
    recovery_long = ((close_values - pullback_low_series) / atr_values).clip(0, 2) / 2
    recovery_short = ((pullback_high_series - close_values) / atr_values).clip(0, 2) / 2
    recovery_strength = np.where(recent_direction > 0, recovery_long, recovery_short)

    momentum_dir = np.sign(close.diff(3).values)
    momentum_confirm = (momentum_dir == recent_direction).astype(float)

    vol_ratio_values = vol_ratio.values
    vol_continuation_confirm = (vol_ratio_values / VOL_CONTINUATION_THRESHOLD).clip(
        0, 1
    )

    cvd_z_shifted = pd.Series(cvd_z_values, index=close.index).shift(3).fillna(0).values
    cvd_momentum = np.where(
        recent_direction > 0,
        (cvd_z_values > cvd_z_shifted).astype(float),
        (cvd_z_values < cvd_z_shifted).astype(float),
    )
    vpin_shifted = pd.Series(vpin_values, index=close.index).shift(3).fillna(0.5).values
    vpin_rising = (vpin_values > vpin_shifted).astype(float)

    score_pullback_series = pd.Series(score_pullback, index=close.index)
    pullback_recent_high = (
        score_pullback_series.rolling(10, min_periods=1).max().values
    )
    pullback_current = score_pullback
    was_in_pullback = (pullback_recent_high > BREAKOUT_TRIGGER_THRESHOLD) & (
        pullback_current < pullback_recent_high * PULLBACK_END_RATIO
    )

    score_continuation = (
        was_in_pullback * recovery_strength * momentum_confirm * 0.3
        + was_in_pullback * recovery_strength * vol_continuation_confirm * 0.3
        + was_in_pullback * recovery_strength * cvd_momentum * 0.25
        + was_in_pullback * recovery_strength * vpin_rising * 0.15
    )
    score_continuation = np.clip(score_continuation, 0, 1)

    # ── 4. Neutral scores ──
    if bb_width_normalized is not None:
        bb_compression = (
            1
            - pd.to_numeric(bb_width_normalized, errors="coerce")
            .fillna(0.5)
            .clip(0, 1)
            .values
        )
    else:
        atr_pct = (
            atr_s.rolling(vol_ma_window * 2, min_periods=20).rank(pct=True).fillna(0.5)
        )
        bb_compression = (1 - atr_pct).values

    vol_compression = 1 - vol_pct_values

    long_strength = breakout_long_raw.fillna(0).clip(0, 1).values
    short_strength = breakout_short_raw.fillna(0).clip(0, 1).values
    direction_separation = np.abs(long_strength - short_strength)
    direction_confidence = (
        np.maximum(long_strength, short_strength) * direction_separation
    )
    direction_confidence = np.clip(direction_confidence, 0, 1)

    score_neutral = (
        bb_compression * 0.4 + vol_compression * 0.4 + (1 - direction_confidence) * 0.2
    )
    score_neutral = np.clip(score_neutral, 0, 1)

    # ── TPC: 语义对齐（更深回踩、弱化延续、压震荡区与动量末端追涨杀跌）──
    semantic_chop = np.zeros(n, dtype=float)
    semantic_extension = np.zeros(n, dtype=float)
    semantic_vwap_discount = np.full(n, 0.5, dtype=float)
    if tpc_semantic_reweight and (not gate_pullback_on_breakout) and macro_vwap_position is not None:
        mv_s = pd.to_numeric(macro_vwap_position, errors="coerce").fillna(0.0)
        mv = mv_s.values
        # 震荡区：BB 偏窄 + 方向置信低 → 类似「波动区来回扫」
        chop_raw = bb_compression * (1.0 - direction_confidence)
        semantic_chop = np.clip(chop_raw * 2.0, 0.0, 1.0)
        # 动量末端：价相对 VWAP1200 过度拉伸（多头在正侧、空头在负侧）
        ext_long = np.clip((mv - 0.03) / 0.10, 0.0, 1.0)
        ext_short = np.clip((-mv - 0.03) / 0.10, 0.0, 1.0)
        semantic_extension = np.where(recent_direction > 0, ext_long, ext_short).astype(
            float
        )
        # 相对 VWAP 的「折让」深度：多头希望价为负（在 VWAP 下方回踩更深）
        semantic_vwap_discount = np.where(
            recent_direction > 0,
            np.clip((-mv) / 0.14, 0.0, 1.0),
            np.clip(mv / 0.14, 0.0, 1.0),
        ).astype(float)
        ema_arr = (
            pd.to_numeric(ema_position, errors="coerce").fillna(0.0).values
            if ema_position is not None
            else np.zeros(n, dtype=float)
        )
        ema_slope = pd.Series(ema_arr, index=close.index).diff(24).fillna(0.0).values
        mv_slope = mv_s.diff(12).fillna(0.0).values
        # 长期锚上行且仍偏多头语境：仍在 VWAP 上方、且 macro 位置在变差（更贵）→ 加重末端惩罚
        long_uptrend = (recent_direction > 0) & (ema_arr > 0.02) & (ema_slope > 0)
        chase_risk = long_uptrend & (mv > -0.015) & (mv_slope > 0)
        semantic_extension = np.clip(
            semantic_extension + chase_risk.astype(float) * 0.35, 0.0, 1.0
        )
        short_downtrend = (recent_direction < 0) & (ema_arr < -0.02) & (ema_slope < 0)
        chase_risk_s = short_downtrend & (mv < 0.015) & (mv_slope < 0)
        semantic_extension = np.clip(
            semantic_extension + chase_risk_s.astype(float) * 0.35, 0.0, 1.0
        )
        # 弱化延续、强化「深折让」与 Donchian 回踩深度
        depth_w = pullback_depth.astype(float)
        p_scale = (
            (0.42 + 0.58 * semantic_vwap_discount)
            * (0.72 + 0.28 * depth_w)
            * (1.0 - 0.48 * semantic_chop)
        )
        c_scale = (1.0 - 0.62 * semantic_extension) * (1.0 - 0.52 * semantic_chop)
        score_pullback = np.clip(score_pullback * p_scale, 0.0, 1.0)
        score_continuation = np.clip(score_continuation * c_scale, 0.0, 1.0)

    # ── Output ──
    p = prefix
    out_cols: Dict[str, Any] = {
        f"{p}_price_breakout_strength": breakout_strength,
        f"{p}_vol_breakout_confirm": vol_breakout_confirm,
        f"{p}_cvd_breakout_confirm": cvd_breakout_confirm,
        f"{p}_vpin_breakout_confirm": vpin_breakout_confirm,
        f"{p}_pullback_depth": pullback_depth,
        f"{p}_pullback_quality": pullback_quality,
        f"{p}_vol_pullback_confirm": vol_pullback_confirm,
        f"{p}_cvd_absorption": cvd_absorption,
        f"{p}_recovery_strength": recovery_strength,
        f"{p}_momentum_confirm": momentum_confirm,
        f"{p}_vol_continuation_confirm": vol_continuation_confirm,
        f"{p}_cvd_momentum": cvd_momentum,
        f"{p}_vpin_rising": vpin_rising,
        f"{p}_bb_compression": bb_compression,
        f"{p}_vol_compression": vol_compression,
        f"{p}_score_breakout": bpc_score_breakout,
        f"{p}_score_pullback": score_pullback,
        f"{p}_score_continuation": score_continuation,
        f"{p}_score_neutral": score_neutral,
        f"{p}_breakout_direction": recent_direction,
        f"{p}_direction_confidence": direction_confidence,
        f"{p}_is_after_breakout": is_after_breakout,
        f"{p}_was_in_pullback": was_in_pullback.astype(float),
        f"{p}_vol_ratio": vol_ratio_values,
        f"{p}_cvd_z": cvd_z_values,
    }
    if p == "tpc":
        out_cols["tpc_semantic_chop"] = semantic_chop
        out_cols["tpc_semantic_extension"] = semantic_extension
        out_cols["tpc_semantic_vwap_discount"] = semantic_vwap_discount

    result = pd.DataFrame(out_cols, index=close.index)

    result.attrs["feature_version"] = FEATURE_VERSION
    result.attrs["param_lookback_breakout"] = lookback_breakout
    result.attrs["param_breakout_atr_mult"] = breakout_atr_mult
    result.attrs["param_vol_ma_window"] = vol_ma_window
    result.attrs["param_pullback_decay"] = "adaptive"
    result.attrs["thresholds"] = {
        "vol_breakout": VOL_BREAKOUT_THRESHOLD,
        "vol_continuation": VOL_CONTINUATION_THRESHOLD,
        "vpin_active": VPIN_ACTIVE_THRESHOLD,
        "cvd_absorption": CVD_ABSORPTION_THRESHOLD,
        "breakout_trigger": BREAKOUT_TRIGGER_THRESHOLD,
        "pullback_end_ratio": PULLBACK_END_RATIO,
    }

    return result


# -- BPC registered wrapper (backward-compatible) --

@register_feature(
    "compute_bpc_soft_phase_from_series",
    category="bpc",
    description="BPC soft phase scores with volume and orderflow confirmation",
    outputs=[
        "bpc_price_breakout_strength",
        "bpc_vol_breakout_confirm",
        "bpc_cvd_breakout_confirm",
        "bpc_vpin_breakout_confirm",
        "bpc_pullback_depth",
        "bpc_pullback_quality",
        "bpc_vol_pullback_confirm",
        "bpc_cvd_absorption",
        "bpc_recovery_strength",
        "bpc_momentum_confirm",
        "bpc_vol_continuation_confirm",
        "bpc_cvd_momentum",
        "bpc_vpin_rising",
        "bpc_bb_compression",
        "bpc_vol_compression",
        "bpc_score_breakout",
        "bpc_score_pullback",
        "bpc_score_continuation",
        "bpc_score_neutral",
        "bpc_breakout_direction",
        "bpc_direction_confidence",
        "bpc_is_after_breakout",
        "bpc_was_in_pullback",
        "bpc_vol_ratio",
        "bpc_cvd_z",
    ],
)
def compute_bpc_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    volume: pd.Series,
    cvd_change_5: pd.Series = None,
    vpin: pd.Series = None,
    ofci_pct: pd.Series = None,
    bb_width_normalized: pd.Series = None,
    lookback_breakout: int = 20,
    breakout_atr_mult: float = 1.0,
    pullback_decay: float = 0.3,
    vol_ma_window: int = 20,
) -> pd.DataFrame:
    """BPC soft phase scores: breakout-gated pullback/continuation."""
    return _compute_soft_phase_core(
        close=close,
        high=high,
        low=low,
        atr=atr,
        volume=volume,
        cvd_change_5=cvd_change_5,
        vpin=vpin,
        ofci_pct=ofci_pct,
        bb_width_normalized=bb_width_normalized,
        lookback_breakout=lookback_breakout,
        breakout_atr_mult=breakout_atr_mult,
        pullback_decay=pullback_decay,
        vol_ma_window=vol_ma_window,
        gate_pullback_on_breakout=True,
        prefix="bpc",
    )


# -- TPC registered wrapper (trend-based, no breakout gate) --

@register_feature(
    "compute_tpc_soft_phase_from_series",
    category="tpc",
    description="TPC soft phase scores: trend-based pullback/continuation (no breakout gate)",
    outputs=[
        "tpc_price_breakout_strength",
        "tpc_vol_breakout_confirm",
        "tpc_cvd_breakout_confirm",
        "tpc_vpin_breakout_confirm",
        "tpc_pullback_depth",
        "tpc_pullback_quality",
        "tpc_vol_pullback_confirm",
        "tpc_cvd_absorption",
        "tpc_recovery_strength",
        "tpc_momentum_confirm",
        "tpc_vol_continuation_confirm",
        "tpc_cvd_momentum",
        "tpc_vpin_rising",
        "tpc_bb_compression",
        "tpc_vol_compression",
        "tpc_score_breakout",
        "tpc_score_pullback",
        "tpc_score_continuation",
        "tpc_score_neutral",
        "tpc_breakout_direction",
        "tpc_direction_confidence",
        "tpc_is_after_breakout",
        "tpc_was_in_pullback",
        "tpc_vol_ratio",
        "tpc_cvd_z",
        "tpc_semantic_chop",
        "tpc_semantic_extension",
        "tpc_semantic_vwap_discount",
    ],
)
def compute_tpc_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    volume: pd.Series,
    cvd_change_5: pd.Series = None,
    vpin: pd.Series = None,
    ofci_pct: pd.Series = None,
    bb_width_normalized: pd.Series = None,
    ema_1200_position: pd.Series = None,
    macro_tp_vwap_1200_position: pd.Series = None,
    lookback_breakout: int = 20,
    breakout_atr_mult: float = 1.0,
    pullback_decay: float = 0.3,
    vol_ma_window: int = 20,
) -> pd.DataFrame:
    """TPC soft phase scores: trend direction from EMA, no breakout gate."""
    return _compute_soft_phase_core(
        close=close,
        high=high,
        low=low,
        atr=atr,
        volume=volume,
        cvd_change_5=cvd_change_5,
        vpin=vpin,
        ofci_pct=ofci_pct,
        bb_width_normalized=bb_width_normalized,
        lookback_breakout=lookback_breakout,
        breakout_atr_mult=breakout_atr_mult,
        pullback_decay=pullback_decay,
        vol_ma_window=vol_ma_window,
        gate_pullback_on_breakout=False,
        ema_position=ema_1200_position,
        macro_vwap_position=macro_tp_vwap_1200_position,
        tpc_semantic_reweight=macro_tp_vwap_1200_position is not None,
        prefix="tpc",
    )


# =============================================================================
# 🧩 辅助特征函数
# =============================================================================


@register_feature(
    "compute_bpc_pullback_depth_pct_from_series",
    category="bpc",
    description="BPC pullback depth as percentage of range, side-aware",
    outputs=[
        "bpc_pullback_depth_long",
        "bpc_pullback_depth_short",
        "bpc_pullback_depth_pct",
    ],
)
def compute_bpc_pullback_depth_pct_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    trend_sign: pd.Series = None,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    回踩深度百分比：当前价格相对近期高点/低点的回撤程度

    支持 side-aware：
    - 多头回踩深度：(rolling_high - close) / range
    - 空头回踩深度：(close - rolling_low) / range
    - 自适应深度：根据 trend_sign 自动选择

    语义：
    - 0.0 = 在高点/低点附近（没有回踩）
    - 0.3-0.5 = 健康回踩区间
    - 0.7+ = 深度回踩（结构可能被破坏）
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)

    rolling_high = high.rolling(window=lookback, min_periods=1).max()
    rolling_low = low.rolling(window=lookback, min_periods=1).min()
    range_size = (rolling_high - rolling_low).replace(0, np.nan)

    # 多头回踩深度
    long_depth = ((rolling_high - close) / range_size).fillna(0.0).clip(0.0, 1.0)
    # 空头回踩深度
    short_depth = ((close - rolling_low) / range_size).fillna(0.0).clip(0.0, 1.0)

    # 自适应深度
    if trend_sign is not None:
        trend = pd.to_numeric(trend_sign, errors="coerce").fillna(0)
        adaptive_depth = np.where(
            trend > 0, long_depth, np.where(trend < 0, short_depth, 0.5)
        )
        adaptive_depth = pd.Series(adaptive_depth, index=close.index)
    else:
        adaptive_depth = long_depth  # 默认多头

    return pd.DataFrame(
        {
            "bpc_pullback_depth_long": long_depth,
            "bpc_pullback_depth_short": short_depth,
            "bpc_pullback_depth_pct": adaptive_depth,
        }
    )


@register_feature(
    "compute_bpc_pullback_duration_from_series",
    category="bpc",
    description="BPC pullback duration in consecutive bars",
    outputs=["bpc_pullback_duration"],
)
def compute_bpc_pullback_duration_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    lookback: int = 20,
    threshold_pct: float = 0.1,
) -> pd.DataFrame:
    """
    回踩持续时间：连续低于近期高点阈值的 bars 数

    语义：
    - 1-5 bars: 短暂回踩（健康）
    - 6-15 bars: 中等回踩（需要确认）
    - 15+ bars: 长时间回踩（市场可能换剧本了）
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)

    rolling_high = high.rolling(window=lookback, min_periods=1).max()
    threshold = rolling_high * (1 - threshold_pct)

    is_pullback = close < threshold

    # 计算连续回踩 bars 数（run-length encoding）
    duration = np.zeros(len(close), dtype=int)
    cnt = 0
    for i, v in enumerate(is_pullback.values):
        if v:
            cnt += 1
        else:
            cnt = 0
        duration[i] = cnt

    # 归一化到 [0, 1]
    max_duration = float(lookback * 2)
    out = pd.Series(duration, index=close.index) / max_duration
    out = out.clip(0.0, 1.0)

    return out.rename("bpc_pullback_duration").to_frame()


@register_feature(
    "compute_bpc_pullback_speed_from_series",
    category="bpc",
    description="BPC pullback speed = depth / (duration + 1)",
    outputs=["bpc_pullback_speed"],
)
def compute_bpc_pullback_speed_from_series(
    *,
    bpc_pullback_depth_pct: pd.Series,
    bpc_pullback_duration: pd.Series,
) -> pd.DataFrame:
    """
    回踩速度：depth / (duration + 1)

    使用 (duration + 1) 防止除零问题

    语义：
    - 高速回踩（>0.5）: 可能是恐慌性抛售，结构不稳
    - 中速回踩（0.2-0.5）: 正常获利了结
    - 低速回踩（<0.2）: 惜售，结构健康
    """
    depth = pd.to_numeric(bpc_pullback_depth_pct, errors="coerce").fillna(0.0)
    duration = pd.to_numeric(bpc_pullback_duration, errors="coerce").fillna(0.0)

    # 使用 (duration + 1) 防止除零
    speed = (depth / (duration + 1)).clip(0.0, 1.0)

    return speed.rename("bpc_pullback_speed").to_frame()


@register_feature(
    "compute_bpc_impulse_return_atr_from_series",
    category="bpc",
    description="BPC impulse return normalized by ATR, with direction",
    outputs=["bpc_impulse_return_atr", "bpc_impulse_direction_match"],
)
def compute_bpc_impulse_return_atr_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Impulse 收益 / ATR：标准化的趋势收益，保留方向信息

    语义：
    - 正值：上涨 impulse
    - 负值：下跌 impulse
    - 绝对值 < 1 ATR: 弱 impulse
    - 绝对值 1-3 ATR: 正常 impulse
    - 绝对值 > 3 ATR: 强 impulse
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)

    returns = close - close.shift(lookback)
    # 保留符号
    ratio_signed = (returns / atr_s).fillna(0.0).clip(-5.0, 5.0)

    # 方向匹配特征（与近期趋势方向是否一致）
    trend_dir = np.sign(close.rolling(lookback * 2).mean().diff())
    impulse_dir = np.sign(returns)
    direction_match = (trend_dir == impulse_dir).astype(float)

    return pd.DataFrame(
        {
            "bpc_impulse_return_atr": ratio_signed / 5.0,  # 归一化到 [-1, 1]
            "bpc_impulse_direction_match": direction_match,
        }
    )


@register_feature(
    "compute_bpc_dir_consistency_multi_from_series",
    category="bpc",
    description="BPC multi-scale direction consistency",
    outputs=[
        "bpc_dir_consistency_short",
        "bpc_dir_consistency_mid",
        "bpc_dir_consistency_long",
    ],
)
def compute_bpc_dir_consistency_multi_from_series(
    *,
    close: pd.Series,
    window_short: int = 5,
    window_mid: int = 20,
    window_long: int = 50,
) -> pd.DataFrame:
    """
    多尺度方向一致性：短/中/长期的方向一致性

    语义：
    - 三者都高：强趋势
    - 短高、长低：趋势刚开始
    - 短低、长高：趋势可能结束
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)

    def _dir_consistency(window: int) -> pd.Series:
        direction = np.sign(close.diff())
        # 计算窗口内方向与最后一根方向一致的比例
        consistency = direction.rolling(window=window, min_periods=1).apply(
            lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
        )
        return consistency.fillna(0.5)

    return pd.DataFrame(
        {
            "bpc_dir_consistency_short": _dir_consistency(window_short),
            "bpc_dir_consistency_mid": _dir_consistency(window_mid),
            "bpc_dir_consistency_long": _dir_consistency(window_long),
        }
    )


@register_feature(
    "compute_bpc_dir_flip_count_from_series",
    category="bpc",
    description="BPC direction flip count in recent bars",
    outputs=["bpc_dir_flip_count"],
)
def compute_bpc_dir_flip_count_from_series(
    *,
    close: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    方向翻转次数：最近 N bars 内价格方向的翻转次数

    语义：
    - 低翻转（< 5）: 趋势稳定
    - 高翻转（> 10）: 震荡市场，BPC 结构不稳
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    direction = np.sign(close.diff())

    # 检测方向变化
    flips = (direction != direction.shift(1)).astype(int)
    flip_count = flips.rolling(window=lookback, min_periods=1).sum()

    # 归一化
    out = (flip_count / float(lookback)).clip(0.0, 1.0)

    return out.rename("bpc_dir_flip_count").to_frame()


@register_feature(
    "compute_bpc_volume_compression_pct_from_series",
    category="bpc",
    description="BPC volume compression percentile",
    outputs=["bpc_volume_compression_pct"],
)
def compute_bpc_volume_compression_pct_from_series(
    *,
    volume: pd.Series,
    window: int = 20,
    percentile_window: int = 540,
) -> pd.DataFrame:
    """
    成交量压缩百分位：当前成交量相对历史的百分位

    语义：
    - 低（< 0.3）: 成交量压缩，蓄势中
    - 高（> 0.7）: 放量，可能是突破或反转
    """
    volume = pd.to_numeric(volume, errors="coerce").astype(float)
    vol_ma = volume.rolling(window=window, min_periods=1).mean()

    def _percentile(arr: np.ndarray) -> float:
        if len(arr) <= 1:
            return 0.5
        current = arr[-1]
        return float(np.mean(arr[:-1] <= current))

    pct = vol_ma.rolling(window=percentile_window, min_periods=50).apply(
        _percentile, raw=True
    )

    return (
        pct.fillna(0.5).clip(0.0, 1.0).rename("bpc_volume_compression_pct").to_frame()
    )


@register_feature(
    "compute_bpc_pullback_delta_absorption_from_series",
    category="bpc",
    description="BPC pullback delta absorption using z-score",
    outputs=["bpc_pullback_delta_absorption"],
)
def compute_bpc_pullback_delta_absorption_from_series(
    *,
    close: pd.Series,
    cvd_change_5: pd.Series,
    atr: pd.Series,
    lookback: int = 50,
) -> pd.DataFrame:
    """
    回踩期间的 Delta 吸收（使用 z-score 标准化）

    吸收定义：CVD 与 price 方向相反，且 CVD 绝对值大

    语义：
    - 高吸收（> 0.7）: 大量反向订单流但价格不动 → 有人在吸筹
    - 低吸收（< 0.3）: 订单流与价格同向 → 正常回踩
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0.0)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)

    price_change = close.diff(5)

    # 使用滚动 z-score 标准化 CVD
    cvd_mean = cvd.rolling(lookback, min_periods=10).mean()
    cvd_std = cvd.rolling(lookback, min_periods=10).std().replace(0, np.nan)
    cvd_z = ((cvd - cvd_mean) / cvd_std).fillna(0.0).clip(-3, 3)

    # 吸收定义：CVD 与 price 方向相反，且 CVD 绝对值大
    price_dir = np.sign(price_change)
    cvd_dir = np.sign(cvd)
    is_counter = (price_dir != cvd_dir) & (price_dir != 0)

    # 吸收强度 = 反向 CVD z-score 的绝对值（仅在反向时计算）
    absorption = np.where(is_counter, cvd_z.abs() / 3.0, 0.0)

    out = pd.Series(absorption, index=close.index).fillna(0.0).clip(0.0, 1.0)

    return out.rename("bpc_pullback_delta_absorption").to_frame()


# =============================================================================
# 🎯 上下文特征：Volume Profile + Liquidity + Reflexivity
# =============================================================================


@register_feature(
    "compute_bpc_breakout_context_from_series",
    category="bpc",
    description="BPC breakout context: VP position + liquidity void + reflexivity",
    outputs=[
        "bpc_breakout_above_poc",
        "bpc_breakout_above_hal",
        "bpc_liquidity_void_ahead",
        "bpc_false_breakout_risk",
        "bpc_reflex_confirm",
    ],
)
def compute_bpc_breakout_context_from_series(
    *,
    close: pd.Series,
    bpc_breakout_direction: pd.Series,
    vp_poc: pd.Series = None,
    vp_hal_high: pd.Series = None,
    vp_hal_low: pd.Series = None,
    liquidity_void_detected: pd.Series = None,
    wpt_false_breakout_risk: pd.Series = None,
    ofci_pct: pd.Series = None,
) -> pd.DataFrame:
    """
    突破上下文特征：VP位置 + 流动性真空 + 反身性确认

    用途：树模型发现“在什么情况下突破语义不成立”
    - POC 上方突破更有力
    - 流动性真空区域突破阻力小
    - 高反身性 = 风险（过度拥挤）
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    direction = (
        pd.to_numeric(bpc_breakout_direction, errors="coerce").fillna(0).astype(int)
    )
    n = len(close)

    # 1. 突破是否在 POC 上方（多头）或下方（空头）
    if vp_poc is not None:
        poc = pd.to_numeric(vp_poc, errors="coerce").fillna(close)
        above_poc_long = ((close > poc) & (direction > 0)).astype(float)
        below_poc_short = ((close < poc) & (direction < 0)).astype(float)
        breakout_above_poc = above_poc_long + below_poc_short
    else:
        breakout_above_poc = pd.Series(0.5, index=close.index)

    # 2. 突破是否超越 HAL 边界
    if vp_hal_high is not None and vp_hal_low is not None:
        hal_h = pd.to_numeric(vp_hal_high, errors="coerce").fillna(close)
        hal_l = pd.to_numeric(vp_hal_low, errors="coerce").fillna(close)
        above_hal_long = ((close > hal_h) & (direction > 0)).astype(float)
        below_hal_short = ((close < hal_l) & (direction < 0)).astype(float)
        breakout_above_hal = above_hal_long + below_hal_short
    else:
        breakout_above_hal = pd.Series(0.5, index=close.index)

    # 3. 突破方向是否有流动性真空（阻力小）
    if liquidity_void_detected is not None:
        lv = (
            pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0).clip(0, 1)
        )
    else:
        lv = pd.Series(0.0, index=close.index)

    # 4. 假突破风险
    if wpt_false_breakout_risk is not None:
        fb_risk = (
            pd.to_numeric(wpt_false_breakout_risk, errors="coerce").fillna(0).clip(0, 1)
        )
    else:
        fb_risk = pd.Series(0.0, index=close.index)

    # 5. 反身性确认（OFCI 适中 = 突破确认，OFCI 极端 = 风险）
    if ofci_pct is not None:
        ofci = pd.to_numeric(ofci_pct, errors="coerce").fillna(0.5).clip(0, 1)
        # OFCI 0.3-0.7 是健康区间，超过这个范围表示风险
        reflex_confirm = 1 - 2 * np.abs(ofci - 0.5)
    else:
        reflex_confirm = pd.Series(0.5, index=close.index)

    return pd.DataFrame(
        {
            "bpc_breakout_above_poc": breakout_above_poc,
            "bpc_breakout_above_hal": breakout_above_hal,
            "bpc_liquidity_void_ahead": lv,
            "bpc_false_breakout_risk": fb_risk,
            "bpc_reflex_confirm": reflex_confirm,
        },
        index=close.index,
    )


@register_feature(
    "compute_bpc_pullback_structure_from_series",
    category="bpc",
    description="BPC pullback structure: Fib levels + VP support + volume density",
    outputs=[
        "bpc_pullback_fib_382",
        "bpc_pullback_fib_500",
        "bpc_pullback_fib_618",
        "bpc_pullback_to_poc",
        "bpc_pullback_in_hal",
        "bpc_pullback_volume_support",
    ],
)
def compute_bpc_pullback_structure_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bpc_breakout_direction: pd.Series = None,
    vp_poc: pd.Series = None,
    vp_hal_high: pd.Series = None,
    vp_hal_low: pd.Series = None,
    vpvr_volume_density: pd.Series = None,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    回踩结构特征：Fib水平 + VP支撑 + 成交密度

    用途：树模型发现“什么情况下回踩语义不成立”
    - 回踩到 0.382 = 健康
    - 回踩到 0.618 = 深度回踩，结构可能被破坏
    - 回踩到 POC = 关键支撑
    - 回踩在 HAL 区间内 = 健康
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)

    # 计算近期高低点
    rolling_high = high.rolling(lookback, min_periods=1).max()
    rolling_low = low.rolling(lookback, min_periods=1).min()
    swing_range = (rolling_high - rolling_low).clip(lower=1e-8)

    # 方向处理
    if bpc_breakout_direction is not None:
        direction = (
            pd.to_numeric(bpc_breakout_direction, errors="coerce").fillna(0).astype(int)
        )
    else:
        direction = np.sign(close.diff(5).fillna(0)).astype(int)

    # 计算 Fib 回踩水平（多头：从高点回踩；空头：从低点反弹）
    # 多头回踩水平 = (high - close) / range
    pullback_long = (rolling_high - close) / swing_range
    # 空头反弹水平 = (close - low) / range
    pullback_short = (close - rolling_low) / swing_range

    # 根据方向选择
    pullback_ratio = np.where(
        direction >= 0, pullback_long.values, pullback_short.values
    )
    pullback_ratio = pd.Series(pullback_ratio, index=close.index).clip(0, 1)

    # Fib 水平特征（接近该水平时为 1）
    fib_382 = (1 - np.abs(pullback_ratio - 0.382) / 0.15).clip(0, 1)
    fib_500 = (1 - np.abs(pullback_ratio - 0.500) / 0.15).clip(0, 1)
    fib_618 = (1 - np.abs(pullback_ratio - 0.618) / 0.15).clip(0, 1)

    # 回踩是否到达 POC
    if vp_poc is not None:
        poc = pd.to_numeric(vp_poc, errors="coerce").fillna(close)
        atr_proxy = swing_range / 4  # 粗略 ATR 估计
        dist_to_poc = np.abs(close - poc) / atr_proxy.clip(lower=1e-8)
        pullback_to_poc = (1 - dist_to_poc / 2).clip(0, 1)  # 2 ATR 内为接近
    else:
        pullback_to_poc = pd.Series(0.5, index=close.index)

    # 回踩是否在 HAL 区间内
    if vp_hal_high is not None and vp_hal_low is not None:
        hal_h = pd.to_numeric(vp_hal_high, errors="coerce").fillna(close)
        hal_l = pd.to_numeric(vp_hal_low, errors="coerce").fillna(close)
        in_hal = ((close >= hal_l) & (close <= hal_h)).astype(float)
    else:
        in_hal = pd.Series(0.5, index=close.index)

    # 回踩位置的成交密度（高密度 = 强支撑）
    if vpvr_volume_density is not None:
        vol_density = (
            pd.to_numeric(vpvr_volume_density, errors="coerce").fillna(0.5).clip(0, 1)
        )
    else:
        vol_density = pd.Series(0.5, index=close.index)

    return pd.DataFrame(
        {
            "bpc_pullback_fib_382": fib_382,
            "bpc_pullback_fib_500": fib_500,
            "bpc_pullback_fib_618": fib_618,
            "bpc_pullback_to_poc": pullback_to_poc,
            "bpc_pullback_in_hal": in_hal,
            "bpc_pullback_volume_support": vol_density,
        },
        index=close.index,
    )


@register_feature(
    "compute_bpc_continuation_target_from_series",
    category="bpc",
    description="BPC continuation target: LVN distance + momentum divergence",
    outputs=[
        "bpc_target_lvn_distance",
        "bpc_target_lvn_count",
        "bpc_momentum_divergence",
        "bpc_reflex_momentum",
    ],
)
def compute_bpc_continuation_target_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    bpc_breakout_direction: pd.Series = None,
    vpvr_lvn_distance: pd.Series = None,
    vpvr_lvn_count: pd.Series = None,
    cvd_change_5: pd.Series = None,
    shd_pct: pd.Series = None,
    lookback: int = 10,
) -> pd.DataFrame:
    """
    延续目标特征：LVN距离 + 动量背离 + 反身性动量

    用途：树模型发现“什么情况下延续语义不成立”
    - LVN 距离近 = 延续目标清晰
    - 动量背离 = 趋势衰竭预警
    - SHD 过高 = 结构不健康
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)

    # LVN 距离（归一化）
    if vpvr_lvn_distance is not None:
        lvn_dist = (
            pd.to_numeric(vpvr_lvn_distance, errors="coerce").fillna(0.5).clip(0, 1)
        )
    else:
        lvn_dist = pd.Series(0.5, index=close.index)

    # LVN 数量（归一化到 0-1）
    if vpvr_lvn_count is not None:
        lvn_cnt = pd.to_numeric(vpvr_lvn_count, errors="coerce").fillna(0)
        lvn_cnt_norm = (lvn_cnt / 5).clip(0, 1)  # 假设最多 5 个 LVN
    else:
        lvn_cnt_norm = pd.Series(0.5, index=close.index)

    # 动量背离检测（价格创新高/低但 CVD 没有）
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        price_change = close.diff(lookback)

        # 价格创新高但 CVD 没有 = 看空背离
        # 价格创新低但 CVD 没有 = 看多背离
        price_high = close >= close.rolling(lookback, min_periods=1).max()
        price_low = close <= close.rolling(lookback, min_periods=1).min()
        cvd_high = cvd >= cvd.rolling(lookback, min_periods=1).max()
        cvd_low = cvd <= cvd.rolling(lookback, min_periods=1).min()

        bearish_div = (price_high & ~cvd_high).astype(float)
        bullish_div = (price_low & ~cvd_low).astype(float)
        momentum_div = bearish_div + bullish_div
    else:
        momentum_div = pd.Series(0.0, index=close.index)

    # 反身性动量（SHD 健康度）
    if shd_pct is not None:
        shd = pd.to_numeric(shd_pct, errors="coerce").fillna(0.5).clip(0, 1)
        # SHD 低 = 健康，SHD 高 = 风险
        reflex_momentum = 1 - shd
    else:
        reflex_momentum = pd.Series(0.5, index=close.index)

    return pd.DataFrame(
        {
            "bpc_target_lvn_distance": lvn_dist,
            "bpc_target_lvn_count": lvn_cnt_norm,
            "bpc_momentum_divergence": momentum_div,
            "bpc_reflex_momentum": reflex_momentum,
        },
        index=close.index,
    )


@register_feature(
    "compute_bpc_compression_state_from_series",
    category="bpc",
    description="BPC compression state: volatility + volume + energy compression",
    outputs=[
        "bpc_vol_compression_state",
        "bpc_bb_compression_state",
        "bpc_garch_compression",
        "bpc_wpt_energy_low",
        "bpc_pre_breakout_score",
    ],
)
def compute_bpc_compression_state_from_series(
    *,
    close: pd.Series,
    volume: pd.Series,
    bb_width_normalized: pd.Series = None,
    vol_window: int = 20,
    pct_window: int = 100,
) -> pd.DataFrame:
    """
    蓄势状态特征：波动率 + 成交量 + 能量压缩

    用途：树模型发现“什么情况下蓄势语义不成立”
    - 成交量压缩 = 待爆发
    - 波动率压缩 = 突破前兆
    - WPT 能量低 = 待释放

    注意: bpc_garch_compression 已改为 BB 二阶压缩（BB 宽度的滚动百分位压缩），
    语义相同（波动率收窄的历史分位数），去掉了 GARCH 依赖（370s/周期）。
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)

    # 成交量压缩（百分位低 = 压缩）
    vol_ma = volume.rolling(vol_window, min_periods=1).mean()
    vol_pct = vol_ma.rolling(pct_window, min_periods=20).rank(pct=True).fillna(0.5)
    vol_compression = 1 - vol_pct

    # 布林带压缩（一阶）
    if bb_width_normalized is not None:
        bb_width = (
            pd.to_numeric(bb_width_normalized, errors="coerce").fillna(0.5).clip(0, 1)
        )
        bb_compression = 1 - bb_width
    else:
        bb_compression = pd.Series(0.5, index=close.index)

    # EWMA 波动率压缩：替代 GARCH（高相关 GARCH 近似，无需 arch 库）
    # 语义：当前指数加权波动率在历史分布中处于偏低分位 = 波动率压缩
    # 复用 utils_garch_features 中的工具函数，避免重复实现
    ewma_pct = compute_ewma_vol_percentile(close, ewma_span=20, pct_window=pct_window)
    ewma_compression = 1 - ewma_pct

    # WPT 能量低频分量：用 BB 宿度序列商位数实现
    # 语义： BB 压缩状态局5K屈1 = BB 持续收缩，能量聚集尚未释放
    if bb_width_normalized is not None:
        bb_w = (
            pd.to_numeric(bb_width_normalized, errors="coerce").fillna(0.5).clip(0, 1)
        )
        # 5块K线内 bb_width 持续低于中位数的比例（越高=持续压缩越久）
        bb_median = bb_w.rolling(pct_window, min_periods=20).median().fillna(0.5)
        bb_below_median = (bb_w < bb_median).astype(float)
        wpt_low = bb_below_median.rolling(5, min_periods=1).mean().clip(0, 1)
    else:
        wpt_low = pd.Series(0.5, index=close.index)

    # 预突破综合分（压缩程度加权平均）
    pre_breakout_score = (
        vol_compression * 0.25
        + bb_compression * 0.45
        + ewma_compression * 0.15
        + wpt_low * 0.15
    ).clip(0, 1)

    return pd.DataFrame(
        {
            "bpc_vol_compression_state": vol_compression,
            "bpc_bb_compression_state": bb_compression,
            "bpc_garch_compression": ewma_compression,  # EWMA波动率压缩，替代 GARCH，列名保持兼容
            "bpc_wpt_energy_low": wpt_low,
            "bpc_pre_breakout_score": pre_breakout_score,
        },
        index=close.index,
    )


@register_feature(
    "compute_bpc_phase_transition_from_series",
    category="bpc",
    description="BPC phase transition: transition probability + speed + direction",
    outputs=[
        "bpc_transition_b_to_p",
        "bpc_transition_p_to_c",
        "bpc_transition_speed",
        "bpc_structure_health",
    ],
)
def compute_bpc_phase_transition_from_series(
    *,
    bpc_score_breakout: pd.Series,
    bpc_score_pullback: pd.Series,
    bpc_score_continuation: pd.Series,
    bpc_score_neutral: pd.Series,
    shd_pct: pd.Series = None,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    阶段转换特征：转换概率 + 转换速度 + 结构健康度

    用途：树模型发现“什么情况下 BPC 循环不成立”
    - B→P 转换概率 = 突破后回踩的可能性
    - P→C 转换概率 = 回踩后延续的可能性
    - 转换速度 = 阶段变化的快慢
    """
    b = pd.to_numeric(bpc_score_breakout, errors="coerce").fillna(0).clip(0, 1)
    p = pd.to_numeric(bpc_score_pullback, errors="coerce").fillna(0).clip(0, 1)
    c = pd.to_numeric(bpc_score_continuation, errors="coerce").fillna(0).clip(0, 1)
    n = pd.to_numeric(bpc_score_neutral, errors="coerce").fillna(0).clip(0, 1)

    # 主导阶段（0=neutral, 1=breakout, 2=pullback, 3=continuation）
    scores = pd.DataFrame({"n": n, "b": b, "p": p, "c": c})
    phase_dominant = (
        scores.idxmax(axis=1).map({"n": 0, "b": 1, "p": 2, "c": 3}).fillna(0)
    )
    phase_confidence = scores.max(axis=1)

    # B→P 转换概率：breakout 分数下降且 pullback 上升
    b_falling = (b < b.shift(1)).astype(float)
    p_rising = (p > p.shift(1)).astype(float)
    trans_b_to_p = (b_falling * p_rising * b.shift(1)).fillna(0).clip(0, 1)

    # P→C 转换概率：pullback 分数下降且 continuation 上升
    p_falling = (p < p.shift(1)).astype(float)
    c_rising = (c > c.shift(1)).astype(float)
    trans_p_to_c = (p_falling * c_rising * p.shift(1)).fillna(0).clip(0, 1)

    # 转换速度：阶段分数变化的绝对值和
    score_changes = (
        b.diff().abs() + p.diff().abs() + c.diff().abs() + n.diff().abs()
    ).fillna(0)
    trans_speed = score_changes.rolling(lookback, min_periods=1).mean().clip(0, 1)

    # 结构健康度（基于 SHD 或简化估计）
    if shd_pct is not None:
        shd = pd.to_numeric(shd_pct, errors="coerce").fillna(0.5).clip(0, 1)
        structure_health = 1 - shd
    else:
        # 简化估计：主导阶段置信度高 = 结构清晰
        structure_health = phase_confidence

    return pd.DataFrame(
        {
            "bpc_transition_b_to_p": trans_b_to_p,
            "bpc_transition_p_to_c": trans_p_to_c,
            "bpc_transition_speed": trans_speed,
            "bpc_structure_health": structure_health,
        },
        index=b.index,
    )
