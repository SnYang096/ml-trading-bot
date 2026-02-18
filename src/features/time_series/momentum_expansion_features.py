"""
MomentumExpansion Archetype 专用特征模块 v3.0

核心因果逻辑：
ME = Energy × Acceleration × Participation

三因子定义：
1. Energy（能量环境）：市场是否允许扩张？
   - ATR percentile, Volatility regime
2. Acceleration（方向加速）：价格速度是否在增强？
   - 2K/5K 二阶导数, 持续性, 多周期对齐
3. Participation（参与确认）：是真实资金推动？
   - CVD alignment/strength, Volume surge/accel, Delta net flow

设计原则：
- 不依赖结构位置（SR/HAL/Fib/LVN/Pullback/Breakout）
- 语义与BPC正交：BPC看位置，ME看速度
- 所有计算流式安全，无未来函数
- 双窗口加速度：2K用于Entry，5K用于Evidence

层级结构：
- Gate: Energy > threshold + Flow consistency > threshold（不看acceleration）
- Evidence: Energy × |Acceleration_5K| × Participation（乘法）
- Entry: Micro acceleration burst(2K) + Orderflow spike

参考：z实验_003_me/ME特征语义分析.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature


# =============================================================================
# 📌 常量定义
# =============================================================================
FEATURE_VERSION = "3.0"  # Energy × Acceleration × Participation


def _stream_safe_percentile(series: pd.Series, window: int) -> pd.Series:
    """流式安全的百分位计算

    - 仅使用历史数据（无未来泄露）
    - 窗口不足时返回 0.5（中性值）
    - 支持增量计算
    """
    result = series.rolling(window, min_periods=window).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x) if len(x) == window else 0.5,
        raw=False
    )
    result = result.fillna(0.5)
    return result


# =============================================================================
# 🎯 主函数：ME 三因子模型
# =============================================================================

@register_feature(
    "compute_momentum_expansion_soft_phase_from_series",
    category="momentum_expansion",
    description="ME Energy × Acceleration × Participation: 11 core features",
    outputs=[
        # === Energy (能量环境) ===
        "me_atr_pct",               # ATR百分位 [0-1]
        "me_vol_regime",            # 波动率扩张方向百分位 [0-1]
        # === Acceleration (方向加速) ===
        "me_accel_2k",              # 2根K线二阶导数/ATR [-3,3] → Entry
        "me_accel_5k",              # 短期vs中期均速差/ATR [-3,3] → Evidence
        "me_accel_persistence",     # 近5根加速方向一致性 [0-1]
        "me_multi_tf_alignment",    # 3/5/10bar动量方向对齐 [0-1]
        # === Participation (参与确认) ===
        "me_cvd_alignment",         # CVD方向对齐度 [0-1]
        "me_cvd_strength",          # CVD相对强度 [0-1]
        "me_volume_surge",          # 成交量爆发百分位 [0-1]
        "me_volume_accel",          # 成交量加速百分位 [0-1]
        "me_delta_net_flow",        # Delta净流×方向 [-1,1]
    ],
)
def compute_momentum_expansion_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    # 可选订单流
    cvd_change_5: pd.Series = None,
    delta: pd.Series = None,
    # 参数
    lookback: int = 20,
    pct_window: int = 100,
) -> pd.DataFrame:
    """
    ME 三因子模型：Energy × Acceleration × Participation

    所有计算严格因果（只用历史数据），支持流式增量。

    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        atr: ATR 序列
        cvd_change_5: CVD 5周期变化（可选）
        delta: 买卖差值（可选，来自 footprint）
        lookback: 基础窗口
        pct_window: 百分位计算窗口

    Returns:
        DataFrame with 11 core features
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)

    eps = 1e-8

    # ========== 1️⃣ Energy（能量环境）==========

    # E1. ATR百分位：市场是否在"能动"的状态？
    me_atr_pct = _stream_safe_percentile(atr_s, pct_window).clip(0, 1)

    # E2. 波动率regime：ATR是否在扩张？
    #     5根K线ATR变化率的百分位
    atr_change = atr_s / atr_s.shift(5).clip(lower=eps) - 1
    me_vol_regime = _stream_safe_percentile(
        atr_change.fillna(0), pct_window
    ).clip(0, 1)

    # ========== 2️⃣ Acceleration（方向加速）==========

    # A1. 2K加速度：2根K线二阶导数 / ATR → Entry微观触发
    #     物理意义：速度的瞬时变化率
    price_change_1 = close.diff(1)
    me_accel_2k = (
        (price_change_1 - price_change_1.shift(1)) / atr_s.clip(lower=eps)
    ).clip(-3, 3)

    # A2. 5K加速度：短期均速 vs 中期均速之差 / ATR → Evidence
    #     物理意义：速度是否在增强（不是趋势本身）
    #     constant-speed trend → short ≈ medium → accel ≈ 0（不触发）
    short_avg_speed = price_change_1.rolling(3, min_periods=1).mean()
    medium_avg_speed = price_change_1.rolling(8, min_periods=1).mean()
    me_accel_5k = (
        (short_avg_speed - medium_avg_speed) / atr_s.clip(lower=eps)
    ).clip(-3, 3)

    # A3. 加速持续性：近5根K线中正加速度的比例
    accel_positive = (me_accel_2k > 0).astype(float)
    me_accel_persistence = accel_positive.rolling(5, min_periods=1).mean().clip(0, 1)

    # A4. 多周期动量方向对齐：3/5/10 bar 收益率方向一致性
    #     全对齐=1.0, 分裂=0.0~0.33
    ret_3 = close.pct_change(3).fillna(0)
    ret_5 = close.pct_change(5).fillna(0)
    ret_10 = close.pct_change(10).fillna(0)
    direction_sum = np.sign(ret_3) + np.sign(ret_5) + np.sign(ret_10)
    me_multi_tf_alignment = (direction_sum.abs() / 3).clip(0, 1)

    # ========== 3️⃣ Participation（参与确认）==========

    price_dir = np.sign(close.diff(1)).fillna(0)

    # P1. CVD对齐度：价格方向与CVD方向一致性
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        cvd_dir = np.sign(cvd)
        me_cvd_alignment = ((price_dir * cvd_dir + 1) / 2).clip(0, 1)

        # P2. CVD强度：CVD相对历史波动的强度
        cvd_std = cvd.rolling(lookback, min_periods=1).std().clip(lower=eps)
        me_cvd_strength = (cvd.abs() / cvd_std).clip(0, 3) / 3
    else:
        me_cvd_alignment = pd.Series(0.5, index=close.index)
        me_cvd_strength = pd.Series(0.5, index=close.index)

    # P3. 成交量爆发：volume / MA 的百分位
    vol_ma = volume.rolling(lookback, min_periods=1).mean().clip(lower=eps)
    vol_ratio = volume / vol_ma
    me_volume_surge = _stream_safe_percentile(vol_ratio, pct_window).clip(0, 1)

    # P4. 成交量加速：短期成交量均值 vs 中期均值
    vol_short = volume.rolling(3, min_periods=1).mean()
    vol_medium = volume.rolling(10, min_periods=1).mean().clip(lower=eps)
    vol_accel_ratio = vol_short / vol_medium - 1
    me_volume_accel = _stream_safe_percentile(
        vol_accel_ratio.fillna(0), pct_window
    ).clip(0, 1)

    # P5. Delta净流：delta z-score × 方向一致性
    if delta is not None:
        delta_s = pd.to_numeric(delta, errors="coerce").fillna(0)
        delta_std = delta_s.rolling(lookback, min_periods=1).std().clip(lower=eps)
        delta_zscore = (delta_s / delta_std).clip(-3, 3)
        # 方向对齐：delta正且价格涨→正信号；delta正但价格跌→负信号
        me_delta_net_flow = (delta_zscore * price_dir / 3).clip(-1, 1)
    else:
        me_delta_net_flow = pd.Series(0.0, index=close.index)

    # ========== 输出 ==========
    result = pd.DataFrame({
        # Energy
        "me_atr_pct": me_atr_pct,
        "me_vol_regime": me_vol_regime,
        # Acceleration
        "me_accel_2k": me_accel_2k,
        "me_accel_5k": me_accel_5k,
        "me_accel_persistence": me_accel_persistence,
        "me_multi_tf_alignment": me_multi_tf_alignment,
        # Participation
        "me_cvd_alignment": me_cvd_alignment,
        "me_cvd_strength": me_cvd_strength,
        "me_volume_surge": me_volume_surge,
        "me_volume_accel": me_volume_accel,
        "me_delta_net_flow": me_delta_net_flow,
    }, index=close.index)

    result.attrs['feature_version'] = FEATURE_VERSION
    return result


# =============================================================================
# 🚀 ME 分层逻辑：Gate / Evidence / Entry
# =============================================================================

@register_feature(
    "compute_me_gate_from_series",
    category="momentum_expansion",
    description="ME Gate: energy + flow consistency + volume surge (no acceleration)",
    outputs=[
        "me_gate_expansion_ok",
        "me_gate_flow_ok",
        "me_gate_volume_ok",
        "me_gate_pass",
    ],
)
def compute_me_gate_from_series(
    *,
    me_atr_pct: pd.Series,
    me_cvd_alignment: pd.Series,
    me_volume_surge: pd.Series,
    expansion_threshold: float = 0.6,
    flow_threshold: float = 0.6,
    volume_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    ME Gate 层：Energy + Flow consistency + Volume surge

    不看 acceleration。Gate 只回答"环境允许吗？"
    """
    expansion_ok = (me_atr_pct >= expansion_threshold).astype(float)
    flow_ok = (me_cvd_alignment >= flow_threshold).astype(float)
    volume_ok = (me_volume_surge >= volume_threshold).astype(float)

    gate_pass = (expansion_ok * flow_ok * volume_ok).clip(0, 1)

    return pd.DataFrame({
        "me_gate_expansion_ok": expansion_ok,
        "me_gate_flow_ok": flow_ok,
        "me_gate_volume_ok": volume_ok,
        "me_gate_pass": gate_pass,
    }, index=me_atr_pct.index)


@register_feature(
    "compute_me_evidence_from_series",
    category="momentum_expansion",
    description="ME Evidence: Energy × |Acceleration_5K| × Participation",
    outputs=["me_evidence_signal"],
)
def compute_me_evidence_from_series(
    *,
    me_atr_pct: pd.Series,
    me_accel_5k: pd.Series,
    me_cvd_alignment: pd.Series,
    me_cvd_strength: pd.Series,
    me_volume_surge: pd.Series,
) -> pd.DataFrame:
    """
    ME Evidence 层：Energy × |Acceleration| × Participation

    乘法组合：三者共振才有信号。
    使用 5K 加速度（比 2K 更稳定，避免噪声）。
    """
    # Acceleration 强度归一化到 [0, 1]
    accel_strength = me_accel_5k.abs().clip(0, 2) / 2

    # Participation = CVD信号 × 成交量确认
    participation = (
        me_cvd_alignment *
        np.maximum(me_cvd_strength, me_volume_surge)  # 取 CVD 强度和量能中更强的
    ).clip(0, 1)

    # 乘法组合
    me_signal = (me_atr_pct * accel_strength * participation).clip(0, 1)

    return pd.DataFrame({
        "me_evidence_signal": me_signal,
    }, index=me_atr_pct.index)


@register_feature(
    "compute_me_entry_from_series",
    category="momentum_expansion",
    description="ME Entry: micro accel burst(2K) + orderflow/volume spike",
    outputs=[
        "me_entry_micro_accel",
        "me_entry_flow_burst",
        "me_entry_confirm",
    ],
)
def compute_me_entry_from_series(
    *,
    me_accel_2k: pd.Series,
    me_cvd_strength: pd.Series,
    me_volume_surge: pd.Series,
    micro_accel_threshold: float = 0.5,
    flow_burst_threshold: float = 0.7,
    volume_burst_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    ME Entry 层：微观触发确认

    - 2K加速度瞬时爆发
    - 订单流瞬时爆发 OR 成交量瞬时爆发
    """
    micro_accel_ok = (me_accel_2k.abs() >= micro_accel_threshold).astype(float)

    # 订单流 OR 成交量爆发（任一满足）
    flow_burst_ok = (
        (me_cvd_strength >= flow_burst_threshold) |
        (me_volume_surge >= volume_burst_threshold)
    ).astype(float)

    entry_confirm = (micro_accel_ok * flow_burst_ok).clip(0, 1)

    return pd.DataFrame({
        "me_entry_micro_accel": micro_accel_ok,
        "me_entry_flow_burst": flow_burst_ok,
        "me_entry_confirm": entry_confirm,
    }, index=me_accel_2k.index)


# =============================================================================
# 🧩 失败信号特征
# =============================================================================

@register_feature(
    "compute_momentum_expansion_failure_from_series",
    category="momentum_expansion",
    description="MomentumExpansion failure signals: expansion without follow-through",
    outputs=[
        "me_false_expansion",
        "me_vol_divergence",
        "me_flow_exhaustion",
        "me_failure_score",
    ],
)
def compute_momentum_expansion_failure_from_series(
    *,
    close: pd.Series,
    me_atr_pct: pd.Series,
    me_accel_5k: pd.Series,
    me_cvd_alignment: pd.Series,
    volume: pd.Series,
    lookback: int = 10,
) -> pd.DataFrame:
    """
    MomentumExpansion 失败信号：供树模型发现"动能衰竭"条件

    ME 失败 = 动能衰竭（不是结构错误）
    - 假扩张：ATR扩张但成交量不配合
    - 成交量背离：价格加速但成交量萎缩
    - 订单流力竭：CVD对齐度下降
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    expansion = pd.to_numeric(me_atr_pct, errors="coerce").fillna(0.5).clip(0, 1)
    accel = pd.to_numeric(me_accel_5k, errors="coerce").fillna(0).clip(-3, 3)
    flow = pd.to_numeric(me_cvd_alignment, errors="coerce").fillna(0.5).clip(0, 1)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)

    # 假扩张：ATR高百分位但成交量不配合
    vol_ma = volume.rolling(lookback, min_periods=1).mean()
    vol_weak = (volume / vol_ma.clip(lower=1e-8)) < 0.8
    me_false_expansion = ((expansion > 0.6) & vol_weak).astype(float)

    # 成交量背离：价格加速但成交量萎缩
    price_rising = close > close.shift(lookback)
    vol_declining = volume < volume.shift(lookback)
    me_vol_divergence = (
        (price_rising & vol_declining) | (~price_rising & ~vol_declining)
    ).astype(float)
    me_vol_divergence = me_vol_divergence * accel.abs().clip(0, 1)

    # 订单流力竭：CVD对齐度从高位下降
    flow_declining = flow < flow.shift(lookback // 2)
    me_flow_exhaustion = (flow_declining & (accel.abs() > 0.5)).astype(float)

    # 综合失败分数（独立子信号，让树模型自学权重）
    me_failure_score = (
        me_false_expansion * 0.35 +
        me_vol_divergence * 0.35 +
        me_flow_exhaustion * 0.3
    ).clip(0, 1)

    return pd.DataFrame({
        "me_false_expansion": me_false_expansion,
        "me_vol_divergence": me_vol_divergence,
        "me_flow_exhaustion": me_flow_exhaustion,
        "me_failure_score": me_failure_score,
    }, index=close.index)


# =============================================================================
# 🌍 上下文特征
# =============================================================================

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
        in_range = ((jump >= 0.6) & (jump <= 0.9)).astype(float)
        distance_to_center = 1 - 2 * np.abs(jump - 0.75)
        me_jump_risk_suitable = (
            in_range * distance_to_center.clip(0, 1)
        ).clip(0, 1)
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
