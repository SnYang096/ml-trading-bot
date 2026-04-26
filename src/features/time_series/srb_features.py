"""
SRB (Structural breakout success near SR) 特征包

集中存放 SRB 族语义特征（与 FER 失败/假突破侧对偶）。
当前实现：近 SR + CVD–价格同向推进 + 效率稳定性，输出 ``srb_sr_success_breakout_*``。

规范：无未来函数、仅因果 rolling/shift、支持按窗口重算（流式与全量一致）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature
from src.features.time_series.fer_features import (
    CVD_ACTIVITY_QUANTILE,
    CVD_ACTIVITY_WINDOW,
    DEFAULT_EFFICIENCY_WINDOW,
    DEFAULT_FAILURE_WINDOW,
    EPS,
    _rolling_diff,
    _safe_divide,
)

FEATURE_VERSION = "1.0"


@register_feature(
    "compute_srb_sr_success_breakout_from_series",
    category="srb",
    description=(
        "SRB / structural breakout success near SR: dual of FER fer_sr_failed_breakout_* — "
        "near SR × impulse continuation aligned with crossing side (direction_to_nearest_sr), "
        "CVD-price agreement, efficiency gain vs prior window, low efficiency-flip noise."
    ),
    outputs=[
        "srb_sr_success_breakout_score",
        "srb_sr_success_breakout_score_pct",
        "srb_sr_success_breakout_direction_signed",
    ],
)
def compute_srb_sr_success_breakout_from_series(
    *,
    close: pd.Series,
    cvd: pd.Series = None,
    cvd_change_5: pd.Series = None,
    dist_to_nearest_sr: Optional[pd.Series] = None,
    direction_to_nearest_sr: Optional[pd.Series] = None,
    fake_breakout: Optional[pd.Series] = None,
    efficiency_window: int = DEFAULT_EFFICIENCY_WINDOW,
    failure_window: int = DEFAULT_FAILURE_WINDOW,
    sr_near_atr: float = 1.2,
) -> pd.DataFrame:
    """
    与 ``fer_features.compute_fer_failure_signals_from_series`` 中 SR 子块语义对偶：

    - FER: 近 SR + impulse **失败** 方向与 ``-direction_to_nearest_sr`` 一致。
    - SRB: 近 SR + aggressive flow 与价格**同向推进**，且与 ``direction_to_nearest_sr`` 同号。

    ``dist_to_nearest_sr`` 与 ``sr_strength_max_f`` 一致（ATR 倍数刻度）。
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    idx = close.index

    has_cvd = cvd is not None
    if has_cvd:
        cvd_s = pd.to_numeric(cvd, errors="coerce").fillna(0.0)
    else:
        cvd_s = pd.Series(0.0, index=idx)

    if cvd_change_5 is not None:
        cvd5 = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0.0)
    else:
        cvd5 = _rolling_diff(cvd_s, 5)

    if has_cvd:
        cvd_abs_change = _rolling_diff(cvd_s, efficiency_window).abs()
        cvd_threshold = (
            cvd_abs_change.rolling(CVD_ACTIVITY_WINDOW, min_periods=efficiency_window)
            .quantile(CVD_ACTIVITY_QUANTILE)
            .fillna(0.0)
        )
        cvd_active = cvd_abs_change > cvd_threshold
        cvd_rolling_std = (
            cvd_s.rolling(efficiency_window, min_periods=1).std().fillna(0.0)
        )
        cvd_active = cvd_active & (cvd_rolling_std > EPS)
    else:
        cvd_active = pd.Series(False, index=idx)

    price_change = _rolling_diff(close, efficiency_window)
    cvd_change = _rolling_diff(cvd_s, efficiency_window)
    signed_eff = pd.Series(
        _safe_divide(price_change.values, cvd_change.values, fill=np.nan),
        index=idx,
    )
    signed_eff = signed_eff.where(cvd_active, 0.0)

    smooth_w = max(3, efficiency_window // 4)
    eff_smooth = signed_eff.rolling(smooth_w, min_periods=1).mean()
    eff_smooth_prev = eff_smooth.shift(1)
    flip_to_negative = (eff_smooth_prev > 0) & (eff_smooth <= 0)
    flip_to_positive = (eff_smooth_prev < 0) & (eff_smooth >= 0)
    flip_event_signed = pd.Series(
        np.where(flip_to_negative, -1.0, np.where(flip_to_positive, 1.0, 0.0)),
        index=idx,
    )
    flip_event_abs = pd.Series(np.abs(flip_event_signed.values), index=idx).astype(float)
    flip_ewm = flip_event_abs.ewm(span=smooth_w * 3, adjust=False).mean().clip(0.0, 1.0)
    flip_magnitude = (eff_smooth_prev - eff_smooth).abs()
    flip_strength = (
        flip_magnitude.rolling(efficiency_window * 4, min_periods=efficiency_window)
        .rank(pct=True)
        .fillna(0.0)
    )
    flip_strength = flip_strength.where(flip_ewm != 0, 0.0)
    flip_stability = (1.0 - flip_strength).clip(0.0, 1.0)

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
        dir_to_sr = pd.Series(-np.sign(dist_sr.fillna(0.0).values), index=idx, dtype=float)

    near_sr_score = pd.Series(
        np.clip(
            1.0 - (dist_sr.abs().fillna(np.inf).values / max(float(sr_near_atr), EPS)),
            0.0,
            1.0,
        ),
        index=idx,
    )

    price_chg_5 = _rolling_diff(close, 5).values
    cvd5_val = cvd5.values
    impulse_success_signed = pd.Series(
        np.where(
            (cvd5_val > EPS) & (price_chg_5 > EPS),
            1.0,
            np.where((cvd5_val < -EPS) & (price_chg_5 < -EPS), -1.0, 0.0),
        ),
        index=idx,
    )
    impulse_success_event = pd.Series(
        np.abs(impulse_success_signed.values), index=idx
    ).astype(float)

    dir_sign = np.sign(dir_to_sr.values).astype(float)
    sr_alignment_success = pd.Series(
        np.where(
            (impulse_success_signed.values != 0.0)
            & (dir_sign != 0.0)
            & (impulse_success_signed.values == dir_sign),
            1.0,
            0.0,
        ),
        index=idx,
    )

    cvd5_abs = cvd5.abs()
    cvd5_pct = (
        cvd5_abs.rolling(failure_window * 4, min_periods=failure_window)
        .rank(pct=True)
        .fillna(0.5)
    )

    eff_current = signed_eff.rolling(failure_window, min_periods=1).mean()
    eff_past = (
        signed_eff.shift(failure_window)
        .rolling(failure_window, min_periods=1)
        .mean()
    )
    eff_gain = pd.Series(
        _safe_divide(
            (eff_current.abs() - eff_past.abs()).clip(lower=0).values,
            eff_past.abs().clip(lower=EPS).values,
            fill=0.0,
        ),
        index=idx,
    )
    impulse_continuation_strength = pd.Series(
        np.clip(cvd5_pct.values * eff_gain.values * 2.0, 0.0, 1.0),
        index=idx,
    )

    core = (
        0.50 * impulse_success_event.values
        + 0.30 * impulse_continuation_strength.values
        + 0.20 * flip_stability.values
    )
    srb_sr_success_breakout_score = pd.Series(
        np.clip(near_sr_score.values * sr_alignment_success.values * core, 0.0, 1.0),
        index=idx,
    )

    if fake_breakout is not None:
        fake_breakout_s = (
            pd.to_numeric(fake_breakout, errors="coerce")
            .reindex(idx)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        srb_sr_success_breakout_score = (
            srb_sr_success_breakout_score
            * (1.0 - 0.30 * fake_breakout_s * near_sr_score)
        ).clip(0.0, 1.0)

    sr_rank_window = max(failure_window * 8, 40)
    srb_sr_success_breakout_score_pct = (
        srb_sr_success_breakout_score.rolling(
            sr_rank_window, min_periods=max(failure_window * 2, 10)
        )
        .rank(pct=True)
        .fillna(0.0)
    )

    srb_sr_success_breakout_direction_signed = pd.Series(
        np.where(
            (srb_sr_success_breakout_score.values >= 0.12)
            & (near_sr_score.values > 0.0)
            & (impulse_success_event.values > 0.0),
            impulse_success_signed.values,
            0.0,
        ),
        index=idx,
    )

    result = pd.DataFrame(
        {
            "srb_sr_success_breakout_score": srb_sr_success_breakout_score,
            "srb_sr_success_breakout_score_pct": srb_sr_success_breakout_score_pct,
            "srb_sr_success_breakout_direction_signed": srb_sr_success_breakout_direction_signed,
        },
        index=idx,
    )

    for col in result.columns:
        result[col] = result[col].where(cvd_active, 0.0).fillna(0.0)

    return result


@register_feature(
    "compute_srb_l3_breakout_window_from_series",
    category="srb",
    description=(
        "SRB L3 structural breakout window: causal wide-SR cross side, age decay, "
        "hold state, and EMA1200 2b alignment as ordinary features."
    ),
    outputs=[
        "srb_l3_breakout_side",
        "srb_l3_breakout_age_bars",
        "srb_l3_breakout_age_decay",
        "srb_l3_breakout_hold",
        "srb_l3_breakout_ema_pos_align",
        "srb_l3_breakout_ema_slope_align",
        "srb_l3_breakout_2b_score",
    ],
)
def compute_srb_l3_breakout_window_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    wide_sr_upper_px: pd.Series,
    wide_sr_lower_px: pd.Series,
    ema_1200_position: Optional[pd.Series] = None,
    max_age_bars: int = 24,
    cross_buffer_atr: float = 0.0,
    hold_buffer_atr: float = 0.0,
    ema_slope_bars: int = 3,
    ema_pos_min: float = 0.008,
    ema_slope_min: float = 0.004,
) -> pd.DataFrame:
    """Causal L3 SR breakout window used to replace SRB's runtime 2a/2b state machine.

    2a is a close crossing beyond the shifted wide SR boundary. The breakout remains
    armed while price holds outside that boundary, with a linear age decay. 2b is
    expressed as EMA1200 position/slope alignment to the same breakout side.
    """
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    idx = close_s.index
    atr_s = pd.to_numeric(atr, errors="coerce").reindex(idx).astype(float)
    upper = pd.to_numeric(wide_sr_upper_px, errors="coerce").reindex(idx).astype(float)
    lower = pd.to_numeric(wide_sr_lower_px, errors="coerce").reindex(idx).astype(float)
    atr_safe = atr_s.where(atr_s > EPS, np.nan)

    cross_buf = float(cross_buffer_atr) * atr_safe
    hold_buf = float(hold_buffer_atr) * atr_safe
    up_level = upper + cross_buf
    dn_level = lower - cross_buf
    up_hold_level = upper - hold_buf
    dn_hold_level = lower + hold_buf

    prev_close = close_s.shift(1)
    prev_up = up_level.shift(1)
    prev_dn = dn_level.shift(1)
    cross_up = (close_s > up_level) & (prev_close <= prev_up)
    cross_down = (close_s < dn_level) & (prev_close >= prev_dn)
    outside_up = close_s > up_hold_level
    outside_down = close_s < dn_hold_level

    max_age = max(1, int(max_age_bars))
    sides: list[float] = []
    ages: list[float] = []
    holds: list[float] = []
    active_side = 0
    active_age = 0

    for i in range(len(idx)):
        if bool(cross_up.iloc[i]):
            active_side = 1
            active_age = 0
        elif bool(cross_down.iloc[i]):
            active_side = -1
            active_age = 0
        elif active_side == 1 and bool(outside_up.iloc[i]) and active_age < max_age:
            active_age += 1
        elif active_side == -1 and bool(outside_down.iloc[i]) and active_age < max_age:
            active_age += 1
        else:
            active_side = 0
            active_age = 0

        sides.append(float(active_side))
        ages.append(float(active_age) if active_side else 0.0)
        holds.append(1.0 if active_side else 0.0)

    side_s = pd.Series(sides, index=idx, dtype=float)
    age_s = pd.Series(ages, index=idx, dtype=float)
    hold_s = pd.Series(holds, index=idx, dtype=float)
    decay_s = (1.0 - (age_s / float(max_age))).clip(0.0, 1.0) * hold_s

    if ema_1200_position is not None:
        ema_pos = (
            pd.to_numeric(ema_1200_position, errors="coerce")
            .reindex(idx)
            .astype(float)
        )
    else:
        ema_pos = pd.Series(0.0, index=idx, dtype=float)
    slope_bars = max(1, int(ema_slope_bars))
    ema_slope = ema_pos - ema_pos.shift(slope_bars)

    ema_pos_align = pd.Series(
        np.where(
            side_s > 0,
            ema_pos >= float(ema_pos_min),
            np.where(side_s < 0, ema_pos <= -float(ema_pos_min), False),
        ),
        index=idx,
        dtype=float,
    )
    ema_slope_align = pd.Series(
        np.where(
            side_s > 0,
            ema_slope >= float(ema_slope_min),
            np.where(side_s < 0, ema_slope <= -float(ema_slope_min), False),
        ),
        index=idx,
        dtype=float,
    )
    ema_pos_align = ema_pos_align.where(hold_s > 0, 0.0).fillna(0.0)
    ema_slope_align = ema_slope_align.where(hold_s > 0, 0.0).fillna(0.0)
    two_b_score = (0.5 * ema_pos_align + 0.5 * ema_slope_align).where(
        hold_s > 0, 0.0
    )

    return pd.DataFrame(
        {
            "srb_l3_breakout_side": side_s.fillna(0.0),
            "srb_l3_breakout_age_bars": age_s.fillna(0.0),
            "srb_l3_breakout_age_decay": decay_s.fillna(0.0),
            "srb_l3_breakout_hold": hold_s.fillna(0.0),
            "srb_l3_breakout_ema_pos_align": ema_pos_align,
            "srb_l3_breakout_ema_slope_align": ema_slope_align,
            "srb_l3_breakout_2b_score": two_b_score.fillna(0.0),
        },
        index=idx,
    )
