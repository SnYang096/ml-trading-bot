"""
Box Structure Features - causal consolidation-box detection.

设计理念（与既往 archetype features 风格对齐，不看未来）：

- 在三个档位上维护滚动 hi/lo（small=60, mid=120, big=240 根）。
- 输出以下原子信号，供多个策略按需消费（srb/bpc/me prefilter 升级、CRF 策略入场）：
    * ``box_hi_{N}``, ``box_lo_{N}`` - 过去 N 根（含当前）最高 high / 最低 low。
    * ``box_width_pct_{N}`` - ``(hi - lo) / mid``。
    * ``box_pos_{N}`` - ``(close - lo) / (hi - lo) ∈ [0, 1]`` 位置指示。
    * ``box_stability_{N}`` - 过去 N 根里 close 保持在 (lo+tol, hi-tol) 内的比例，
      其中 ``tol = max(1 × atr, 0.015 × mid)``。stability → 1 表示真盘整。
    * ``box_touches_hi_{N}`` / ``box_touches_lo_{N}`` - 过去 N 根触及上下沿的次数。
    * ``box_compression_score`` - ``box_width_pct_60 / box_width_pct_240``，<1 表短期压缩。
    * ``box_regime_label`` - 字符串分类：``small/mid/big/none``（由 stability × width 决定）。
    * ``box_breakout_up`` / ``box_breakout_down`` - 上一根刚从 mid 档 box 上/下破（±1/0）。
    * ``box_prior_trend_sign`` - box 形成前 60 根的 trend_r2 × sign(close change)。

规范：
- 仅 causal rolling；``box_hi/lo`` 用 ``high.rolling(N).max().shift(0)`` 即可；
  触发器（breakout）使用 ``.shift(1)`` 保证决策只能基于上一根信息。
- 所有输出 NaN-safe；warm-up 期填默认值（width=NaN, pos=0.5, stability=0.0, label='none'）。
- 无外部 CVD/OI 依赖；只需 OHLC + ATR。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register_feature

FEATURE_VERSION = "1.0"

# 档位窗口
BOX_WINDOWS = (60, 120, 240)

# tol 计算参数：tol = max(TOL_ATR_MULT * atr, TOL_PCT * mid)
TOL_ATR_MULT = 1.0
TOL_PCT = 0.015

# regime 判定阈值
REGIME_STABILITY_MIN = 0.70
REGIME_WIDTH_MAX_SMALL = 0.04  # <=4% 宽度 -> small
REGIME_WIDTH_MAX_MID = 0.08  # <=8% -> mid；>8% 且稳定 -> big

# 突破判定：close 越过上一根的 hi/lo（含 tol 松弛）
BREAKOUT_TOL_FRAC = 0.0  # 0 = 严格越过


def _nan_safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Simple rolling ATR; lighter than baseline.compute_atr but identical semantics.

    Used as a fallback when caller does not pass ``atr``.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _trend_r2_signed(close: pd.Series, n: int = 60) -> pd.Series:
    """Signed trend strength: R^2 of linear fit × sign of net change over N bars.

    Causal: uses only past N closes inclusive of the current bar.
    """
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _one(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return 0.0
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        y_var = ((y - y_mean) ** 2).sum()
        if y_var <= 0 or x_var <= 0:
            return 0.0
        r2 = (cov * cov) / (x_var * y_var)
        sgn = 1.0 if y[-1] >= y[0] else -1.0
        return r2 * sgn

    return close.rolling(n, min_periods=n).apply(_one, raw=True)


def _classify_regime(stability: pd.Series, width: pd.Series) -> pd.Series:
    """Return a string Series in {small, mid, big, none}.

    - stability < REGIME_STABILITY_MIN -> 'none'
    - width <= REGIME_WIDTH_MAX_SMALL -> 'small'
    - width <= REGIME_WIDTH_MAX_MID -> 'mid'
    - else -> 'big'
    """
    label = pd.Series("none", index=stability.index, dtype=object)
    stable = stability.fillna(0.0) >= REGIME_STABILITY_MIN
    w = width.fillna(np.inf)
    label[stable & (w <= REGIME_WIDTH_MAX_SMALL)] = "small"
    label[stable & (w > REGIME_WIDTH_MAX_SMALL) & (w <= REGIME_WIDTH_MAX_MID)] = "mid"
    label[stable & (w > REGIME_WIDTH_MAX_MID)] = "big"
    return label


def _compute_one_window(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    n: int,
) -> pd.DataFrame:
    """Compute box_* columns for a single window size n (causal)."""
    idx = close.index

    # past-N high/low (inclusive of current bar: window ends at t)
    box_hi = high.rolling(n, min_periods=n).max()
    box_lo = low.rolling(n, min_periods=n).min()
    mid = (box_hi + box_lo) * 0.5
    width = (box_hi - box_lo).clip(lower=1e-12)
    width_pct = (width / mid.replace(0, np.nan)).clip(lower=0.0)

    pos = ((close - box_lo) / width).clip(0.0, 1.0)

    # tol in price units
    tol = np.maximum(TOL_ATR_MULT * atr.fillna(0.0), TOL_PCT * mid.abs())

    # stability: fraction of past-N bars where [low, high] lies inside box ± tol.
    # Cast bool→float and explicitly zero-fill NaN so warm-up NaNs don't bleed
    # into the rolling mean and artificially depress stability post-warm-up.
    inside = (
        (low >= (box_lo - tol)) & (high <= (box_hi + tol))
    ).astype(float)
    # Where box boundaries are NaN (warm-up), inside is NaN from the comparison
    # above; convert to 0 to keep the rolling mean well-defined.
    inside = inside.where(inside.notna(), 0.0)
    stability = (
        inside.rolling(2 * n, min_periods=n).mean()
    )

    # touches within tol of the edge (again past-N window)
    near_hi = (high >= (box_hi - tol)).astype(float).fillna(0.0)
    near_lo = (low <= (box_lo + tol)).astype(float).fillna(0.0)
    touches_hi = near_hi.rolling(n, min_periods=n).sum()
    touches_lo = near_lo.rolling(n, min_periods=n).sum()

    out = pd.DataFrame(
        {
            f"box_hi_{n}": box_hi,
            f"box_lo_{n}": box_lo,
            f"box_width_pct_{n}": width_pct,
            f"box_pos_{n}": pos,
            f"box_stability_{n}": stability,
            f"box_touches_hi_{n}": touches_hi,
            f"box_touches_lo_{n}": touches_lo,
        },
        index=idx,
    )
    return out


@register_feature(
    "compute_box_structure_from_series",
    category="box_structure",
    description=(
        "Causal consolidation-box detector: rolling hi/lo + stability/touches on three "
        "scales (60/120/240 2H bars), plus compression score, regime label, and "
        "breakout / prior-trend-direction signals."
    ),
    outputs=[
        # 60
        "box_hi_60",
        "box_lo_60",
        "box_width_pct_60",
        "box_pos_60",
        "box_stability_60",
        "box_touches_hi_60",
        "box_touches_lo_60",
        # 120
        "box_hi_120",
        "box_lo_120",
        "box_width_pct_120",
        "box_pos_120",
        "box_stability_120",
        "box_touches_hi_120",
        "box_touches_lo_120",
        # 240
        "box_hi_240",
        "box_lo_240",
        "box_width_pct_240",
        "box_pos_240",
        "box_stability_240",
        "box_touches_hi_240",
        "box_touches_lo_240",
        # derived
        "box_compression_score",
        "box_regime_label",
        "box_breakout_up",
        "box_breakout_down",
        "box_prior_trend_sign",
    ],
)
def compute_box_structure_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series = None,
    atr_period: int = 14,
    trend_window: int = 60,
) -> pd.DataFrame:
    """Compute causal box-structure features on (close, high, low, [atr]).

    Args:
        close, high, low: OHLC series (DatetimeIndex aligned).
        atr: optional precomputed ATR in price units; if omitted, a rolling
            14-bar ATR is computed in-line.
        atr_period: fallback ATR period when ``atr`` is not provided.
        trend_window: window for ``box_prior_trend_sign``.

    Returns:
        DataFrame indexed by ``close.index`` with all registered ``box_*`` columns.
    """
    close = _nan_safe_series(close)
    high = _nan_safe_series(high)
    low = _nan_safe_series(low)

    if atr is None:
        atr = _rolling_atr(high, low, close, n=atr_period)
    else:
        atr = _nan_safe_series(atr)

    frames = [_compute_one_window(high, low, close, atr, n) for n in BOX_WINDOWS]
    out = pd.concat(frames, axis=1)

    # ── compression score: short-term width / long-term width ──
    wp60 = out["box_width_pct_60"]
    wp240 = out["box_width_pct_240"]
    compression = (wp60 / wp240.replace(0, np.nan)).clip(lower=0.0, upper=10.0)
    out["box_compression_score"] = compression

    # ── regime label (based on 120 window as the "default" timeframe) ──
    out["box_regime_label"] = _classify_regime(
        out["box_stability_120"], out["box_width_pct_120"]
    )

    # ── breakout triggers (use shift(1) so we compare current close to the PREVIOUS
    #    bar's box boundaries; this keeps the trigger strictly causal w.r.t. the
    #    decision at time t). ──
    prev_hi = out["box_hi_120"].shift(1)
    prev_lo = out["box_lo_120"].shift(1)
    breakout_up = (close > prev_hi * (1.0 + BREAKOUT_TOL_FRAC)).astype(int)
    breakout_down = (close < prev_lo * (1.0 - BREAKOUT_TOL_FRAC)).astype(int)
    # only count breakouts that emerge from a real box
    prev_regime = out["box_regime_label"].shift(1)
    in_box = prev_regime.isin(["small", "mid", "big"]).astype(int)
    out["box_breakout_up"] = (breakout_up * in_box).fillna(0).astype(int)
    out["box_breakout_down"] = (breakout_down * in_box).fillna(0).astype(int)

    # ── prior trend sign (signed R^2 over trend_window bars) ──
    out["box_prior_trend_sign"] = _trend_r2_signed(close, n=trend_window)

    # default-fill for downstream consumers
    for n in BOX_WINDOWS:
        out[f"box_pos_{n}"] = out[f"box_pos_{n}"].fillna(0.5)
        out[f"box_stability_{n}"] = out[f"box_stability_{n}"].fillna(0.0)
        out[f"box_touches_hi_{n}"] = out[f"box_touches_hi_{n}"].fillna(0.0)
        out[f"box_touches_lo_{n}"] = out[f"box_touches_lo_{n}"].fillna(0.0)
    out["box_compression_score"] = out["box_compression_score"].fillna(1.0)
    out["box_prior_trend_sign"] = out["box_prior_trend_sign"].fillna(0.0)

    return out
