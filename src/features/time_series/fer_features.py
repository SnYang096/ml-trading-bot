"""
FER (FailureExhaustionReversal) Archetype 专用特征模块

设计理念：
- 核心语义：单边博弈失败 → 反向清算
- "资金强度没有下降，但价格推进已经死亡"

因果结构（5步）：
1. 单边 impulse 已存在
2. 参与者继续冲（aggressive flow 仍强）
3. 价格推进效率下降
4. 出现 trapped / absorption
5. 结构被打穿

每个子信号单独输出，不做 composite 加权。
由 evidence 层/NN 学习"什么时候共振足够高"。
FER 是"语义原子"，不是"半决策器"。

规范遵循：
- 无未来函数
- 支持流式计算
- 向量化实现（无 Python for-loop）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature


# =============================================================================
# 📌 常量定义
# =============================================================================

DEFAULT_EFFICIENCY_WINDOW = 20  # 推进效率滚动窗口
DEFAULT_ABSORPTION_WINDOW = 10  # 吸收检测窗口
DEFAULT_TRAPPED_LOOKBACK = 20  # Trapped 检测回溯
DEFAULT_FAILURE_WINDOW = 10  # Impulse 失败检测窗口
DEFAULT_DECAY_WINDOW = 20  # 衰减计算窗口
DEFAULT_DIVERGENCE_WINDOW = 10  # 背离检测窗口

EPS = 1e-9
FEATURE_VERSION = "2.2"

# CVD 活跃度检测参数
CVD_ACTIVITY_QUANTILE = 0.05  # 低于历史 5th percentile 视为"无流动性"
CVD_ACTIVITY_WINDOW = 80  # 活跃度评估的滚动窗口


# =============================================================================
# 🔧 内部辅助函数
# =============================================================================


def _rolling_abs_diff(series: pd.Series, window: int) -> pd.Series:
    """滚动绝对差 |s[i] - s[i-window]|"""
    return (series - series.shift(window)).abs()


def _rolling_diff(series: pd.Series, window: int) -> pd.Series:
    """滚动差 s[i] - s[i-window]"""
    return series - series.shift(window)


def _safe_divide(numerator, denominator, fill: float = 0.0):
    """安全除法，分母为零时填充"""
    denom = np.where(np.abs(denominator) > EPS, denominator, np.nan)
    result = numerator / denom
    return np.where(np.isnan(result), fill, result)


# =============================================================================
# 🎯 主函数：FER 失败反转信号
# =============================================================================


@register_feature(
    "compute_fer_failure_signals_from_series",
    category="fer",
    description="FER failure-exhaustion-reversal signals: signed efficiency, efficiency flip, absorption, trapped with CVD, impulse failure",
    outputs=[
        # === 方向化推进效率 (升级1) ===
        "fer_signed_efficiency",
        "fer_signed_efficiency_pct",
        # === 效率翻转点 (升级2) ===
        "fer_efficiency_flip",
        "fer_efficiency_flip_strength",
        # === 吸收 ===
        "fer_aggressor_absorption",
        "fer_absorption_streak",
        # === Trapped (带 CVD 方向) ===
        "fer_trapped_longs_score",
        "fer_trapped_shorts_score",
        # === Impulse 失败 ===
        "fer_impulse_failure_score",
        "fer_impulse_failure_direction",
        "fer_impulse_failure_direction_signed",
        # === SR 子语义：边界失败突破 ===
        "fer_sr_failed_breakout_score",
        "fer_sr_failed_breakout_score_pct",
        "fer_sr_failed_breakout_direction_signed",
        # === 衰减 ===
        "fer_momentum_efficiency_decay",
        # === 背离 ===
        "fer_volume_price_divergence",
    ],
)
def compute_fer_failure_signals_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    # 订单流（可选但强烈推荐）
    cvd: pd.Series = None,
    cvd_change_5: pd.Series = None,
    # SR 上下文（可选；无则 SR 子语义置 0）
    dist_to_nearest_sr: Optional[pd.Series] = None,
    direction_to_nearest_sr: Optional[pd.Series] = None,
    fake_breakout: Optional[pd.Series] = None,
    # 参数
    efficiency_window: int = DEFAULT_EFFICIENCY_WINDOW,
    absorption_window: int = DEFAULT_ABSORPTION_WINDOW,
    trapped_lookback: int = DEFAULT_TRAPPED_LOOKBACK,
    failure_window: int = DEFAULT_FAILURE_WINDOW,
    decay_window: int = DEFAULT_DECAY_WINDOW,
    divergence_window: int = DEFAULT_DIVERGENCE_WINDOW,
    sr_near_atr: float = 1.2,
) -> pd.DataFrame:
    """
    FER 失败反转特征 v2：语义原子输出，不做 composite 加权。

    三个关键升级：
    1. 方向化效率 signed_eff = ΔPrice / ΔCVD（带符号）
    2. 效率翻转点检测 signed_eff 从正→负
    3. Trapped 加入高位 CVD 强度判断
    4. 去掉 failure_strength composite，每个信号独立输出

    Args:
        close, high, low, volume, atr: 基础价格数据
        cvd: 累积成交量差 (可选)
        cvd_change_5: CVD 5周期变化 (可选)
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=EPS)

    n = len(close)
    idx = close.index

    # CVD 处理 + 活跃度掩码
    # 核心原则: |ΔCVD| 过小时输出 0.0 (中性值)，而非 NaN
    #   - NaN 在实盘 pd.isna() 过滤中被丢弃 → 模型收不到特征
    #   - 0.0 = 语义中性 (无效率/无翻转/无吸收) → 模型正常接收"无信号"
    has_cvd = cvd is not None
    if has_cvd:
        cvd_s = pd.to_numeric(cvd, errors="coerce").fillna(0.0)
    else:
        cvd_s = pd.Series(0.0, index=idx)

    if cvd_change_5 is not None:
        cvd5 = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0.0)
    else:
        cvd5 = _rolling_diff(cvd_s, 5)

    # CVD 活跃度掩码: 当 |ΔCVD| 低于历史 5th percentile → 标记为不活跃
    if has_cvd:
        cvd_abs_change = _rolling_diff(cvd_s, efficiency_window).abs()
        cvd_threshold = (
            cvd_abs_change.rolling(CVD_ACTIVITY_WINDOW, min_periods=efficiency_window)
            .quantile(CVD_ACTIVITY_QUANTILE)
            .fillna(0.0)
        )
        cvd_active = cvd_abs_change > cvd_threshold
        # 额外: CVD 全零段也标记为不活跃
        cvd_rolling_std = (
            cvd_s.rolling(efficiency_window, min_periods=1).std().fillna(0.0)
        )
        cvd_active = cvd_active & (cvd_rolling_std > EPS)
    else:
        # 完全没有 CVD 数据 → 全部不活跃
        cvd_active = pd.Series(False, index=idx)

    # ================================================================
    # 1️⃣ 方向化推进效率 (升级1)
    # signed_eff = ΔPrice / ΔCVD
    # 正 = 顺势推进, 负 = 反向吸收, ~0 = 推进停滞
    # ================================================================
    price_change = _rolling_diff(close, efficiency_window)
    cvd_change = _rolling_diff(cvd_s, efficiency_window)

    fer_signed_efficiency = pd.Series(
        _safe_divide(price_change.values, cvd_change.values, fill=np.nan),
        index=idx,
    )
    # CVD 不活跃时 → 0.0（中性值，不是 NaN，避免实盘特征被丢弃）
    fer_signed_efficiency = fer_signed_efficiency.where(cvd_active, 0.0)
    # 百分位：当前 signed_eff 在近期窗口的排名
    fer_signed_efficiency_pct = (
        fer_signed_efficiency.rolling(
            efficiency_window * 2, min_periods=efficiency_window
        )
        .rank(pct=True)
        .fillna(0.5)
    )  # 无数据 → 0.5 (中性百分位)

    # ================================================================
    # 2️⃣ 效率翻转点 (升级2)
    # 检测 signed_eff 从正→负（推进→吸收的转换点）
    # 这比"效率衰减"更有反转爆发力
    # ================================================================
    # 短窗口平滑避免噪声
    smooth_w = max(3, efficiency_window // 4)
    eff_smooth = fer_signed_efficiency.rolling(smooth_w, min_periods=1).mean()
    eff_smooth_prev = eff_smooth.shift(1)

    # flip: 前一刻正 → 当前负 = 多头推进→吸收 (做空信号)
    #        前一刻负 → 当前正 = 空头推进→吸收 (做多信号)
    flip_to_negative = (eff_smooth_prev > 0) & (eff_smooth <= 0)  # 多头失败
    flip_to_positive = (eff_smooth_prev < 0) & (eff_smooth >= 0)  # 空头失败

    fer_efficiency_flip = pd.Series(
        np.where(flip_to_negative, -1.0, np.where(flip_to_positive, 1.0, 0.0)),
        index=idx,
    )
    # efficiency flip 转化为连续衰减信号 [0,1]：
    # 发生 flip 事件时=1，此后指数衰减 (EWM span=smooth_w)
    # 语义: 最近是否发生过效率翻转 (direction-agnostic, prefilter 准确语义)
    _flip_event = pd.Series(np.abs(fer_efficiency_flip.values), index=idx).astype(float)
    fer_efficiency_flip = (
        _flip_event.ewm(span=smooth_w * 3, adjust=False).mean().clip(0.0, 1.0)
    )
    # 翻转强度 = |前值 - 后值| 的大小，衡量翻转的剧烈程度
    flip_magnitude = (eff_smooth_prev - eff_smooth).abs()
    # 归一化为百分位
    fer_efficiency_flip_strength = (
        flip_magnitude.rolling(efficiency_window * 4, min_periods=efficiency_window)
        .rank(pct=True)
        .fillna(0.0)
    )
    # 只在 flip 发生时保留强度
    fer_efficiency_flip_strength = fer_efficiency_flip_strength.where(
        fer_efficiency_flip != 0, 0.0
    )

    # ================================================================
    # 3️⃣ 吸收比率: aggressor 买入但价格下跌
    # CVD↑ + price↓ = 多头被吸收
    # CVD↓ + price↑ = 空头被吸收
    # ================================================================
    price_change_w = _rolling_diff(close, absorption_window)
    cvd_change_w = _rolling_diff(cvd_s, absorption_window)

    # 多头吸收: CVD↑ 但 price↓
    long_absorb = np.where(
        (cvd_change_w.values > EPS) & (price_change_w.values < -EPS),
        _safe_divide(np.abs(price_change_w.values), cvd_change_w.values, fill=0.0),
        0.0,
    )
    # 空头吸收: CVD↓ 但 price↑
    short_absorb = np.where(
        (cvd_change_w.values < -EPS) & (price_change_w.values > EPS),
        _safe_divide(
            np.abs(price_change_w.values), np.abs(cvd_change_w.values), fill=0.0
        ),
        0.0,
    )
    fer_aggressor_absorption_raw = pd.Series(
        np.maximum(long_absorb, short_absorb), index=idx
    )

    # 连续吸收 bar 数：必须从原始市价中判断（归一化后几乎所有值>0，会导致streak永不重置）
    is_absorbing = (fer_aggressor_absorption_raw > 0).astype(float)
    not_absorbing = ~(fer_aggressor_absorption_raw > 0)
    groups = not_absorbing.cumsum()
    fer_absorption_streak = is_absorbing.groupby(groups).cumsum()
    # 归一化到 [0,1]：rolling percentile，避免原始 count 跨 symbol 不可比
    fer_absorption_streak = (
        fer_absorption_streak.rolling(
            absorption_window * 10, min_periods=absorption_window
        )
        .rank(pct=True)
        .fillna(0.0)
    )
    # fer_aggressor_absorption 归一化到 [0,1]：rolling percentile，跨 symbol 可比
    fer_aggressor_absorption = (
        fer_aggressor_absorption_raw.rolling(
            absorption_window * 10, min_periods=absorption_window
        )
        .rank(pct=True)
        .fillna(0.0)
    )

    # ================================================================
    # 4️⃣ Trapped (带 CVD 方向 — 升级核心)
    # 真正的 trapped 语义：
    #   多头 trapped = 高位期间 CVD 强正 + 价格没继续创新高 + 回撤
    #   空头 trapped = 低位期间 CVD 强负 + 价格没继续创新低 + 反弹
    # ================================================================
    rolling_high = high.rolling(trapped_lookback, min_periods=1).max().shift(1)
    rolling_low = low.rolling(trapped_lookback, min_periods=1).min().shift(1)

    # 从最高价回撤 (ATR 归一化)
    drawdown_from_high = pd.Series(
        _safe_divide(
            (rolling_high - close).clip(lower=0).values,
            atr_s.values,
            fill=0.0,
        ),
        index=idx,
    )
    # 从最低价反弹
    bounce_from_low = pd.Series(
        _safe_divide(
            (close - rolling_low).clip(lower=0).values,
            atr_s.values,
            fill=0.0,
        ),
        index=idx,
    )

    # 高位区间 CVD 强度: lookback 窗口内 CVD 变化的百分位
    # 强正 = 多头在冲; 强负 = 空头在冲
    cvd_lookback_change = _rolling_diff(cvd_s, trapped_lookback)
    cvd_lookback_pct = (
        cvd_lookback_change.rolling(trapped_lookback * 4, min_periods=trapped_lookback)
        .rank(pct=True)
        .fillna(0.5)
    )  # trapped 已被 cvd_active 掩码保护，此处 0.5 不会泄漏

    # 多头 trapped:
    #   drawdown > 0 (价格从高位回撤)
    #   + CVD 在高位是强正 (多头在冲)
    #   = 多头被套
    fer_trapped_longs_score = pd.Series(
        np.clip(
            drawdown_from_high.values * cvd_lookback_pct.values * 2,
            0,
            5,
        ),
        index=idx,
    )
    # 归一化到 [0,1]：原始 clip [0,5]，除以 5 得到 bounded 连续值
    fer_trapped_longs_score = (fer_trapped_longs_score / 5.0).clip(0.0, 1.0)

    # 空头 trapped:
    #   bounce > 0 (价格从低位反弹)
    #   + CVD 在低位是强负 (空头在冲)
    #   = 空头被套
    fer_trapped_shorts_score = pd.Series(
        np.clip(
            bounce_from_low.values * (1 - cvd_lookback_pct.values) * 2,
            0,
            5,
        ),
        index=idx,
    )
    # 归一化到 [0,1]：原始 clip [0,5]，除以 5 得到 bounded 连续值
    fer_trapped_shorts_score = (fer_trapped_shorts_score / 5.0).clip(0.0, 1.0)

    # ================================================================
    # 5️⃣ Impulse 失败得分
    # 核心: 动量仍强 (CVD 绝对值高百分位)
    #       × 推进效率塌陷 (signed_eff 从正→低/负)
    # ================================================================
    cvd5_abs = cvd5.abs()
    cvd5_pct = (
        cvd5_abs.rolling(failure_window * 4, min_periods=failure_window)
        .rank(pct=True)
        .fillna(0.5)
    )  # impulse 已被 cvd_active 掩码保护

    # 效率塌陷 = signed_eff 的下降: 过去窗口均值 vs 当前窗口均值
    eff_current = fer_signed_efficiency.rolling(failure_window, min_periods=1).mean()
    eff_past = (
        fer_signed_efficiency.shift(failure_window)
        .rolling(failure_window, min_periods=1)
        .mean()
    )

    # 使用绝对值下降 (signed_eff 方向翻转也算塌陷)
    eff_decline = pd.Series(
        _safe_divide(
            (eff_past.abs() - eff_current.abs()).clip(lower=0).values,
            eff_past.abs().clip(lower=EPS).values,
            fill=0.0,
        ),
        index=idx,
    )

    fer_impulse_failure_score = pd.Series(
        np.clip(cvd5_pct.values * eff_decline.values * 2, 0, 1),
        index=idx,
    )

    # 失败方向
    cvd5_val = cvd5.values
    price_chg_5 = _rolling_diff(close, 5).values
    # 离散交易方向语义（供 direction.yaml 规则1，与 decay 列分离）:
    #   -1 = 多头 impulse 失败 → 交易方向 SHORT
    #   +1 = 空头 impulse 失败 → 交易方向 LONG
    #    0 = 无事件 → 交由 CVD/ROC 规则
    fer_impulse_failure_direction_signed = pd.Series(
        np.where(
            (cvd5_val > EPS) & (price_chg_5 < -EPS),
            -1.0,
            np.where(
                (cvd5_val < -EPS) & (price_chg_5 > EPS),
                1.0,
                0.0,
            ),
        ),
        index=idx,
    )
    # fer_impulse_failure_direction: 连续衰减 [0,1]，语义「最近失败事件强度」
    # （Gate/prefilter 等仍可用；勿再用于 negate_sign 方向）
    _impulse_failure_event = pd.Series(
        np.abs(fer_impulse_failure_direction_signed.values), index=idx
    ).astype(float)
    fer_impulse_failure_direction = (
        _impulse_failure_event.ewm(span=failure_window, adjust=False)
        .mean()
        .clip(0.0, 1.0)
    )

    # ================================================================
    # 5b️⃣ SR 子语义：边界失败突破（FER-SR）
    # 语义：在 SR 邻域内，出现 impulse failure 且方向与 SR 相对位置一致
    # ================================================================
    if dist_to_nearest_sr is not None:
        dist_sr = (
            pd.to_numeric(dist_to_nearest_sr, errors="coerce")
            .reindex(idx)
            .astype(float)
        )
    else:
        dist_sr = pd.Series(np.nan, index=idx, dtype=float)

    if direction_to_nearest_sr is not None:
        dir_to_sr = (
            pd.to_numeric(direction_to_nearest_sr, errors="coerce")
            .reindex(idx)
            .fillna(0.0)
            .astype(float)
        )
    else:
        # fallback: direction_to_nearest_sr 未给时由 dist 符号推断（与 baseline 定义一致）
        dir_to_sr = pd.Series(-np.sign(dist_sr.fillna(0.0).values), index=idx, dtype=float)

    near_sr_score = pd.Series(
        np.clip(
            1.0 - (dist_sr.abs().fillna(np.inf).values / max(float(sr_near_atr), EPS)),
            0.0,
            1.0,
        ),
        index=idx,
    )

    # 方向对齐：trade_dir == -direction_to_nearest_sr
    sr_alignment = pd.Series(
        np.where(
            (fer_impulse_failure_direction_signed.values != 0.0)
            & (np.sign(dir_to_sr.values) != 0.0)
            & (
                fer_impulse_failure_direction_signed.values
                == -np.sign(dir_to_sr.values).astype(float)
            ),
            1.0,
            0.0,
        ),
        index=idx,
    )

    fer_sr_failed_breakout_score = pd.Series(
        np.clip(
            near_sr_score.values
            * sr_alignment.values
            * (
                0.55 * _impulse_failure_event.values
                + 0.25 * fer_impulse_failure_score.values
                + 0.20 * fer_efficiency_flip_strength.values
            ),
            0.0,
            1.0,
        ),
        index=idx,
    )

    if fake_breakout is not None:
        fake_breakout_s = (
            pd.to_numeric(fake_breakout, errors="coerce")
            .reindex(idx)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        fer_sr_failed_breakout_score = (
            fer_sr_failed_breakout_score
            + 0.20 * fake_breakout_s * near_sr_score * _impulse_failure_event
        ).clip(0.0, 1.0)

    sr_rank_window = max(failure_window * 8, 40)
    fer_sr_failed_breakout_score_pct = (
        fer_sr_failed_breakout_score.rolling(
            sr_rank_window, min_periods=max(failure_window * 2, 10)
        )
        .rank(pct=True)
        .fillna(0.0)
    )
    fer_sr_failed_breakout_direction_signed = pd.Series(
        np.where(
            (fer_sr_failed_breakout_score.values >= 0.12)
            & (near_sr_score.values > 0.0)
            & (_impulse_failure_event.values > 0.0),
            fer_impulse_failure_direction_signed.values,
            0.0,
        ),
        index=idx,
    )

    # ================================================================
    # 6️⃣ 动量效率衰减
    # ================================================================
    price_chg_current = _rolling_abs_diff(close, decay_window)
    price_chg_past = _rolling_abs_diff(close.shift(decay_window), decay_window)
    vol_current = volume.rolling(decay_window, min_periods=1).sum()
    vol_past = volume.shift(decay_window).rolling(decay_window, min_periods=1).sum()

    eff_now = _safe_divide(price_chg_current.values, vol_current.values, fill=0.0)
    eff_ago = _safe_divide(price_chg_past.values, vol_past.values, fill=0.0)

    fer_momentum_efficiency_decay = pd.Series(
        np.clip(
            _safe_divide((eff_ago - eff_now), np.maximum(eff_ago, EPS), fill=0.0),
            0,
            1,
        ),
        index=idx,
    )

    # ================================================================
    # 7️⃣ 量价背离
    # ================================================================
    vol_change = _rolling_diff(volume, divergence_window)
    price_change_div = _rolling_diff(close, divergence_window)

    sign_opposite = (vol_change.values * price_change_div.values) < 0
    divergence_raw = pd.Series(
        np.where(
            sign_opposite,
            _safe_divide(
                np.abs(vol_change.values),
                np.abs(price_change_div.values) + EPS,
                fill=0.0,
            ),
            0.0,
        ),
        index=idx,
    )
    fer_volume_price_divergence = (
        divergence_raw.rolling(divergence_window * 4, min_periods=divergence_window)
        .rank(pct=True)
        .fillna(0.0)
    )

    # ========== 组装输出 (无 composite, 每个信号独立) ==========
    result = pd.DataFrame(
        {
            # 方向化效率
            "fer_signed_efficiency": fer_signed_efficiency,
            "fer_signed_efficiency_pct": fer_signed_efficiency_pct,
            # 效率翻转点
            "fer_efficiency_flip": fer_efficiency_flip,
            "fer_efficiency_flip_strength": fer_efficiency_flip_strength,
            # 吸收
            "fer_aggressor_absorption": fer_aggressor_absorption,
            "fer_absorption_streak": fer_absorption_streak,
            # Trapped (带 CVD)
            "fer_trapped_longs_score": fer_trapped_longs_score,
            "fer_trapped_shorts_score": fer_trapped_shorts_score,
            # Impulse 失败
            "fer_impulse_failure_score": fer_impulse_failure_score,
            "fer_impulse_failure_direction": fer_impulse_failure_direction,
            "fer_impulse_failure_direction_signed": fer_impulse_failure_direction_signed,
            "fer_sr_failed_breakout_score": fer_sr_failed_breakout_score,
            "fer_sr_failed_breakout_score_pct": fer_sr_failed_breakout_score_pct,
            "fer_sr_failed_breakout_direction_signed": fer_sr_failed_breakout_direction_signed,
            # 衰减
            "fer_momentum_efficiency_decay": fer_momentum_efficiency_decay,
            # 背离
            "fer_volume_price_divergence": fer_volume_price_divergence,
        },
        index=idx,
    )

    # ========== CVD 活跃度掩码：CVD 不活跃时，CVD 相关特征 → 0.0 (中性值) ==========
    # 0.0 = "无信号"，比 NaN 更安全:
    #   - NaN 在实盘特征提取中被 pd.isna() 丢弃 → 模型收不到特征 (bug)
    #   - 0.0 = 语义中性值 (无效率/无翻转/无吸收/无trapped) → 模型正常收到"无信号"
    # 不受影响的: momentum_efficiency_decay, volume_price_divergence (纯价量)
    cvd_dependent_cols = [
        "fer_signed_efficiency",
        "fer_signed_efficiency_pct",
        "fer_efficiency_flip",
        "fer_efficiency_flip_strength",
        "fer_aggressor_absorption",
        "fer_absorption_streak",
        "fer_trapped_longs_score",
        "fer_trapped_shorts_score",
        "fer_impulse_failure_score",
        "fer_impulse_failure_direction",
        "fer_impulse_failure_direction_signed",
        "fer_sr_failed_breakout_score",
        "fer_sr_failed_breakout_score_pct",
        "fer_sr_failed_breakout_direction_signed",
    ]
    for col in cvd_dependent_cols:
        result[col] = result[col].where(cvd_active, 0.0).fillna(0.0)

    return result
