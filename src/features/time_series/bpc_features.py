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


# =============================================================================
# 📌 常量定义（便于调参和维护）
# =============================================================================

# --- P1: 行业标准参数（固定值） ---
DEFAULT_LOOKBACK_BREAKOUT = 20    # ATR 标准周期
DEFAULT_VOL_MA_WINDOW = 20        # 成交量均线窗口
DEFAULT_BREAKOUT_ATR_MULT = 1.0   # Donchian 标准

# --- P2: 关键动态参数（可自适应） ---
DEFAULT_PULLBACK_DECAY = 0.3      # 回踩质量衰减系数
PULLBACK_DECAY_MIN = 0.2          # 高波动时的快速衰减
PULLBACK_DECAY_MAX = 0.5          # 低波动时的慢速衰减
VOL_ADAPTIVE_WINDOW = 40          # 自适应波动率计算窗口

# --- P3: 业务阈值（可配置） ---
VOL_BREAKOUT_THRESHOLD = 1.5      # 突破放量阈值（均量倍数）
VOL_CONTINUATION_THRESHOLD = 1.2  # 续行放量阈值
VPIN_ACTIVE_THRESHOLD = 0.6       # VPIN 活跃阈值
CVD_ABSORPTION_THRESHOLD = 0.5    # CVD 吸收阈值（标准差）
BREAKOUT_TRIGGER_THRESHOLD = 0.3  # 突破触发阈值
PULLBACK_END_RATIO = 0.6          # Pullback 结束比例（峰值 * 0.6）
WEAK_BREAKOUT_THRESHOLD = 0.1     # 弱突破阈值

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

@register_feature(
    "compute_bpc_soft_phase_from_series",
    category="bpc",
    description="BPC soft phase scores with volume and orderflow confirmation",
    outputs=[
        # === ATOMIC: Breakout 原子信号 ===
        "bpc_price_breakout_strength",
        "bpc_vol_breakout_confirm",
        "bpc_cvd_breakout_confirm",
        "bpc_vpin_breakout_confirm",
        # === ATOMIC: Pullback 原子信号 ===
        "bpc_pullback_depth",
        "bpc_pullback_quality",
        "bpc_vol_pullback_confirm",
        "bpc_cvd_absorption",
        # === ATOMIC: Continuation 原子信号 ===
        "bpc_recovery_strength",
        "bpc_momentum_confirm",
        "bpc_vol_continuation_confirm",
        "bpc_cvd_momentum",
        "bpc_vpin_rising",
        # === ATOMIC: Neutral 原子信号 ===
        "bpc_bb_compression",
        "bpc_vol_compression",
        # === COMPOSITE: 组合分数（领域知识加权） ===
        "bpc_score_breakout",
        "bpc_score_pullback",
        "bpc_score_continuation",
        "bpc_score_neutral",
        # === CONTEXTUAL: 状态信号 ===
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
    # 订单流特征（可选但推荐）
    cvd_change_5: pd.Series = None,
    vpin: pd.Series = None,
    ofci_pct: pd.Series = None,
    # 波动率压缩（可选）
    bb_width_normalized: pd.Series = None,
    # 参数
    lookback_breakout: int = 20,
    breakout_atr_mult: float = 1.0,
    pullback_decay: float = 0.3,
    vol_ma_window: int = 20,
) -> pd.DataFrame:
    """
    BPC 软阶段分数：输出连续概率而非离散标签
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        atr: ATR 序列（价格单位）
        volume: 成交量序列
        cvd_change_5: CVD 5周期变化（可选）
        vpin: VPIN 指标（可选）
        ofci_pct: OFCI 百分位（可选）
        bb_width_normalized: 布林带宽度归一化（可选）
        lookback_breakout: Breakout 检测回看窗口
        breakout_atr_mult: Breakout ATR 倍数阈值
        pullback_decay: Pullback 质量衰减系数
        vol_ma_window: 成交量均线窗口
    
    Returns:
        DataFrame with soft phase scores and auxiliary features
    
    设计理念：
        - 软阶段概率代替硬阶段标签（连续、可微、无跳变）
        - 每个阶段 = 价格信号 × 成交量确认 × 订单流验证
        - Price tells you WHAT, Volume tells you IF, Order flow tells you WHO
    
    ⚠️ 使用注意：
        - 必须按 instrument 单独调用，不要拼接多个 instrument 的数据
        - 所有状态在函数内部初始化，不会跨调用残留
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    
    n = len(close)
    eps = 1e-8
    
    # ========== 预计算基础指标 ==========
    rolling_high = high.rolling(lookback_breakout, min_periods=1).max().shift(1)
    rolling_low = low.rolling(lookback_breakout, min_periods=1).min().shift(1)
    rolling_range = (rolling_high - rolling_low).clip(lower=eps)
    
    # 成交量基准
    vol_ma = volume.rolling(vol_ma_window, min_periods=1).mean()
    vol_ratio = (volume / vol_ma.clip(lower=eps)).fillna(1.0)
    vol_pct = volume.rolling(vol_ma_window * 2, min_periods=20).rank(pct=True).fillna(0.5)
    
    # 订单流处理（如果提供）
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
    
    # ========== 1️⃣ Breakout 分数：价格突破 × 放量 × CVD确认 ==========
    # 价格强度：距离前高的 ATR 倍数（带符号，正=多头突破，负=空头突破）
    breakout_long_raw = (close - rolling_high) / (atr_s * breakout_atr_mult + eps)
    breakout_short_raw = (rolling_low - close) / (atr_s * breakout_atr_mult + eps)
    
    # 取较强的方向
    breakout_strength = np.maximum(
        breakout_long_raw.fillna(0).values, 
        breakout_short_raw.fillna(0).values
    )
    breakout_strength = np.clip(breakout_strength, 0, 1)
    breakout_direction = np.where(
        breakout_long_raw.fillna(0).values > breakout_short_raw.fillna(0).values, 
        1, 
        -1
    )
    
    # 成交量确认：放量 > VOL_BREAKOUT_THRESHOLD 倍均量
    vol_breakout_confirm = (vol_ratio / VOL_BREAKOUT_THRESHOLD).clip(0, 1).values
    
    # 订单流确认：CVD 同向 + VPIN 活跃
    cvd_z_values = cvd_z.values if hasattr(cvd_z, 'values') else np.full(n, 0.0)
    vpin_values = vpin_s.values if hasattr(vpin_s, 'values') else np.full(n, 0.5)
    
    cvd_breakout_confirm = np.where(
        breakout_direction > 0,
        (cvd_z_values > 0).astype(float) * (np.abs(cvd_z_values) / 2).clip(0, 1),
        (cvd_z_values < 0).astype(float) * (np.abs(cvd_z_values) / 2).clip(0, 1)
    )
    vpin_breakout_confirm = (vpin_values / VPIN_ACTIVE_THRESHOLD).clip(0, 1)
    
    # 综合 Breakout 分数
    bpc_score_breakout = (
        breakout_strength * 0.4 +
        breakout_strength * vol_breakout_confirm * 0.3 +
        breakout_strength * cvd_breakout_confirm * 0.2 +
        breakout_strength * vpin_breakout_confirm * 0.1
    )
    bpc_score_breakout = np.clip(bpc_score_breakout, 0, 1)
    
    # ========== 2️⃣ Pullback 分数：回踩深度 × 缩量 × 订单流吸收 ==========
    # 是否近期有 breakout（滑动窗口）
    bpc_score_breakout_series = pd.Series(bpc_score_breakout, index=close.index)
    recent_breakout_strength = bpc_score_breakout_series.rolling(
        lookback_breakout, min_periods=1
    ).max()
    is_after_breakout = (recent_breakout_strength > BREAKOUT_TRIGGER_THRESHOLD).astype(float).values
    
    # 回踩深度（多头：从高点下来；空头：从低点上来）
    pullback_depth_long = ((rolling_high - close) / rolling_range).clip(0, 1).values
    pullback_depth_short = ((close - rolling_low) / rolling_range).clip(0, 1).values
    
    # 使用 breakout 方向决定用哪个 depth
    # 向前填充最近的方向（使用 ffill 替代 Python 循环，性能优化）
    recent_direction = breakout_direction.copy()
    # 将弱 breakout 位置设为 NaN，然后 ffill
    recent_direction_series = pd.Series(recent_direction, index=close.index)
    weak_mask = breakout_strength < WEAK_BREAKOUT_THRESHOLD
    recent_direction_series[weak_mask] = np.nan
    recent_direction = recent_direction_series.ffill().fillna(0).values.astype(int)
    
    pullback_depth = np.where(
        recent_direction > 0,
        pullback_depth_long,
        pullback_depth_short
    )
    
    # 回踩质量：深度越浅越好（指数衰减）
    # 使用自适应 pullback_decay（逐 bar 计算）
    vol_pct_adaptive = atr_s.rolling(VOL_ADAPTIVE_WINDOW, min_periods=20).rank(pct=True).fillna(0.5).values
    pullback_decay_adaptive = _compute_adaptive_pullback_decay(vol_pct_adaptive)
    pullback_quality = np.exp(-pullback_depth / pullback_decay_adaptive)
    
    # 成交量确认：缩量 < 0.8倍均量
    vol_pct_values = vol_pct.values if hasattr(vol_pct, 'values') else np.full(n, 0.5)
    vol_pullback_confirm = (1 - vol_pct_values).clip(0, 1)
    
    # 订单流确认：CVD 从谷底/顶部回升 = 吸收（使用相对变化替代硬编码阈值）
    cvd_z_series = pd.Series(cvd_z_values, index=close.index)
    cvd_rolling_min = cvd_z_series.rolling(10, min_periods=1).min().values
    cvd_rolling_max = cvd_z_series.rolling(10, min_periods=1).max().values
    # 多头：CVD 从近期低点回升 > CVD_ABSORPTION_THRESHOLD 个标准差
    # 空头：CVD 从近期高点下降 > CVD_ABSORPTION_THRESHOLD 个标准差
    cvd_absorption = np.where(
        recent_direction > 0,
        ((cvd_z_values - cvd_rolling_min) > CVD_ABSORPTION_THRESHOLD).astype(float),
        ((cvd_rolling_max - cvd_z_values) > CVD_ABSORPTION_THRESHOLD).astype(float)
    )
    
    # 综合 Pullback 分数
    bpc_score_pullback = (
        is_after_breakout * pullback_quality * 0.3 +
        is_after_breakout * pullback_quality * vol_pullback_confirm * 0.3 +
        is_after_breakout * pullback_quality * cvd_absorption * 0.4
    )
    bpc_score_pullback = np.clip(bpc_score_pullback, 0, 1)
    
    # ========== 3️⃣ Continuation 分数：价格恢复 × 动量 × 再次放量 + CVD ==========
    # 计算 pullback 期间的极值（滑动窗口近似）
    pullback_low_series = low.rolling(lookback_breakout // 2, min_periods=1).min().values
    pullback_high_series = high.rolling(lookback_breakout // 2, min_periods=1).max().values
    
    # 价格恢复强度
    atr_values = atr_s.values
    close_values = close.values
    recovery_long = ((close_values - pullback_low_series) / atr_values).clip(0, 2) / 2
    recovery_short = ((pullback_high_series - close_values) / atr_values).clip(0, 2) / 2
    recovery_strength = np.where(recent_direction > 0, recovery_long, recovery_short)
    
    # 动量确认：短期方向与 breakout 方向一致
    momentum_dir = np.sign(close.diff(3).values)
    momentum_confirm = (momentum_dir == recent_direction).astype(float)
    
    # 成交量确认：再次放量
    vol_ratio_values = vol_ratio.values
    vol_continuation_confirm = (vol_ratio_values / VOL_CONTINUATION_THRESHOLD).clip(0, 1)
    
    # 订单流确认：CVD 恢复 + VPIN 上升（修复因果性：使用 shift 替代 np.roll）
    cvd_z_shifted = pd.Series(cvd_z_values, index=close.index).shift(3).fillna(0).values
    cvd_momentum = np.where(
        recent_direction > 0,
        (cvd_z_values > cvd_z_shifted).astype(float),
        (cvd_z_values < cvd_z_shifted).astype(float)
    )
    vpin_shifted = pd.Series(vpin_values, index=close.index).shift(3).fillna(0.5).values
    vpin_rising = (vpin_values > vpin_shifted).astype(float)
    
    # 需要先经过 pullback 阶段（改进：检查 pullback 已结束而非仅发生过）
    bpc_score_pullback_series = pd.Series(bpc_score_pullback, index=close.index)
    pullback_recent_high = bpc_score_pullback_series.rolling(10, min_periods=1).max().values
    pullback_current = bpc_score_pullback
    # 条件：最近有 pullback（>BREAKOUT_TRIGGER_THRESHOLD）且当前已从峰值衰减（<峰值的 PULLBACK_END_RATIO）
    was_in_pullback = (pullback_recent_high > BREAKOUT_TRIGGER_THRESHOLD) & (pullback_current < pullback_recent_high * PULLBACK_END_RATIO)
    
    # 综合 Continuation 分数
    bpc_score_continuation = (
        was_in_pullback * recovery_strength * momentum_confirm * 0.3 +
        was_in_pullback * recovery_strength * vol_continuation_confirm * 0.3 +
        was_in_pullback * recovery_strength * cvd_momentum * 0.25 +
        was_in_pullback * recovery_strength * vpin_rising * 0.15
    )
    bpc_score_continuation = np.clip(bpc_score_continuation, 0, 1)
    
    # ========== 4️⃣ Neutral 分数：波动率压缩 × 成交量压缩 ==========
    if bb_width_normalized is not None:
        bb_compression = 1 - pd.to_numeric(
            bb_width_normalized, errors="coerce"
        ).fillna(0.5).clip(0, 1).values
    else:
        # 用 ATR percentile 近似
        atr_pct = atr_s.rolling(vol_ma_window * 2, min_periods=20).rank(pct=True).fillna(0.5)
        bb_compression = (1 - atr_pct).values
    
    vol_compression = 1 - vol_pct_values
    
    # ========== 计算方向置信度（修复：使用方向分离度替代 sign 差） ==========
    # 方向置信度 = 最强突破强度 × 方向分离度
    long_strength = breakout_long_raw.fillna(0).clip(0, 1).values
    short_strength = breakout_short_raw.fillna(0).clip(0, 1).values
    direction_separation = np.abs(long_strength - short_strength)
    direction_confidence = np.maximum(long_strength, short_strength) * direction_separation
    direction_confidence = np.clip(direction_confidence, 0, 1)
    
    # Neutral = 低波动 + 低成交量 + 方向模糊（不依赖其他阶段，避免循环依赖）
    bpc_score_neutral = (
        bb_compression * 0.4 +
        vol_compression * 0.4 +
        (1 - direction_confidence) * 0.2
    )
    bpc_score_neutral = np.clip(bpc_score_neutral, 0, 1)
    
    # ========== 输出（三层分类） ==========
    result = pd.DataFrame({
        # === ATOMIC: Breakout 原子信号（供树模型自由组合） ===
        "bpc_price_breakout_strength": breakout_strength,
        "bpc_vol_breakout_confirm": vol_breakout_confirm,
        "bpc_cvd_breakout_confirm": cvd_breakout_confirm,
        "bpc_vpin_breakout_confirm": vpin_breakout_confirm,
        # === ATOMIC: Pullback 原子信号 ===
        "bpc_pullback_depth": pullback_depth,
        "bpc_pullback_quality": pullback_quality,
        "bpc_vol_pullback_confirm": vol_pullback_confirm,
        "bpc_cvd_absorption": cvd_absorption,
        # === ATOMIC: Continuation 原子信号 ===
        "bpc_recovery_strength": recovery_strength,
        "bpc_momentum_confirm": momentum_confirm,
        "bpc_vol_continuation_confirm": vol_continuation_confirm,
        "bpc_cvd_momentum": cvd_momentum,
        "bpc_vpin_rising": vpin_rising,
        # === ATOMIC: Neutral 原子信号 ===
        "bpc_bb_compression": bb_compression,
        "bpc_vol_compression": vol_compression,
        # === COMPOSITE: 组合分数（领域知识加权，作为强先验） ===
        "bpc_score_breakout": bpc_score_breakout,
        "bpc_score_pullback": bpc_score_pullback,
        "bpc_score_continuation": bpc_score_continuation,
        "bpc_score_neutral": bpc_score_neutral,
        # === CONTEXTUAL: 状态信号（提供上下文） ===
        "bpc_breakout_direction": recent_direction,
        "bpc_direction_confidence": direction_confidence,
        "bpc_is_after_breakout": is_after_breakout,
        "bpc_was_in_pullback": was_in_pullback.astype(float),
        "bpc_vol_ratio": vol_ratio_values,
        "bpc_cvd_z": cvd_z_values,
    }, index=close.index)
    
    # ========== 添加元数据（用于 Outcome-Based Audit 追溯） ==========
    result.attrs['feature_version'] = FEATURE_VERSION
    result.attrs['param_lookback_breakout'] = lookback_breakout
    result.attrs['param_breakout_atr_mult'] = breakout_atr_mult
    result.attrs['param_vol_ma_window'] = vol_ma_window
    result.attrs['param_pullback_decay'] = 'adaptive'  # 标记为自适应
    result.attrs['thresholds'] = {
        'vol_breakout': VOL_BREAKOUT_THRESHOLD,
        'vol_continuation': VOL_CONTINUATION_THRESHOLD,
        'vpin_active': VPIN_ACTIVE_THRESHOLD,
        'cvd_absorption': CVD_ABSORPTION_THRESHOLD,
        'breakout_trigger': BREAKOUT_TRIGGER_THRESHOLD,
        'pullback_end_ratio': PULLBACK_END_RATIO,
    }
    
    return result


# =============================================================================
# 🧩 辅助特征函数
# =============================================================================

@register_feature(
    "compute_bpc_pullback_depth_pct_from_series",
    category="bpc",
    description="BPC pullback depth as percentage of range, side-aware",
    outputs=["bpc_pullback_depth_long", "bpc_pullback_depth_short", "bpc_pullback_depth_pct"],
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
        adaptive_depth = np.where(trend > 0, long_depth, np.where(trend < 0, short_depth, 0.5))
        adaptive_depth = pd.Series(adaptive_depth, index=close.index)
    else:
        adaptive_depth = long_depth  # 默认多头
    
    return pd.DataFrame({
        "bpc_pullback_depth_long": long_depth,
        "bpc_pullback_depth_short": short_depth,
        "bpc_pullback_depth_pct": adaptive_depth,
    })


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
    
    return pd.DataFrame({
        "bpc_impulse_return_atr": ratio_signed / 5.0,  # 归一化到 [-1, 1]
        "bpc_impulse_direction_match": direction_match,
    })


@register_feature(
    "compute_bpc_dir_consistency_multi_from_series",
    category="bpc",
    description="BPC multi-scale direction consistency",
    outputs=["bpc_dir_consistency_short", "bpc_dir_consistency_mid", "bpc_dir_consistency_long"],
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
            lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5,
            raw=False
        )
        return consistency.fillna(0.5)
    
    return pd.DataFrame({
        "bpc_dir_consistency_short": _dir_consistency(window_short),
        "bpc_dir_consistency_mid": _dir_consistency(window_mid),
        "bpc_dir_consistency_long": _dir_consistency(window_long),
    })


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
    percentile_window: int = 288,
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
    
    return pct.fillna(0.5).clip(0.0, 1.0).rename("bpc_volume_compression_pct").to_frame()


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
