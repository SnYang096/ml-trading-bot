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
