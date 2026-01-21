"""
Physics / Regime Classifier

This module implements a minimal Physics/Regime classifier that determines
which execution regime the market is currently in, based on statistical
constraints on price path feasibility.

Regime types:
- TC_REGIME: Trend Continuation regime (low noise, stable)
- TE_REGIME: Trend Expansion regime (volatility expansion, range expansion)
- MEAN_REGIME: Extreme Mean Reversion regime (extreme dislocations only) [PROXY V0 - NOT production-ready]
- NO_TRADE: No viable execution regime (microstructure unmodelable zone)

Key principle: Regime determines "feasibility", not "direction" or "profitability".

⚠️ IMPORTANT WARNINGS:
1. dir_conf is only a weak signal in Regime classification, not the main axis
2. MEAN_REGIME is PROXY V0 and should NOT be enabled in production execution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd


RegimeType = Literal["TC_REGIME", "TE_REGIME", "MEAN_REGIME", "NO_TRADE"]


@dataclass(frozen=True)
class PhysicsRegimeConfig:
    """
    Configuration for Physics/Regime classification.

    Uses existing features + head outputs to determine execution feasibility.
    """

    # Input columns
    pred_dir_prob_col: str = "pred_dir_prob"
    atr_col: str = "atr"
    atr_percentile_col: str = "atr_percentile"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    # Feature columns (optional, computed if not present)
    atr_slope_window: int = 20
    range_expansion_window: int = 10
    jump_risk_window: int = 10

    # TC Regime constraints
    # ⚠️ IMPORTANT: dir_conf is only a weak signal, not the main axis
    # Main conditions should be: atr_slope, jump_risk_percentile, path_length, dir_conf stability
    tc_dir_conf_min: float = 0.0  # Do not require high dir_conf (stability > magnitude)
    tc_dir_conf_std_max_pct: float = 0.6  # Percentile for stability (lower std)
    tc_dir_sign_consistency_min_pct: float = 0.6  # Percentile for direction consistency
    tc_atr_slope_max_pct: float = 0.6  # Percentile max for low vol expansion
    tc_path_length_min_pct: float = 0.4  # Percentile for path length (longer paths)

    # TE Regime constraints
    # dir_conf is only for stability, not primary
    te_dir_conf_std_max_pct: float = 0.7  # Allow more direction variability
    te_dir_sign_consistency_min_pct: float = 0.5
    te_atr_slope_min_pct: float = 0.6  # Percentile min for vol expanding
    te_range_expansion_min_pct: float = 0.6  # Percentile min for range expansion

    # Physics feasibility score threshold (percentile-based)
    physics_score_min_pct: float = 0.9  # Top 10% of physics_score

    # MEAN Regime constraints (extreme only)
    # ⚠️ PROXY V0: Current definition is not fully physical
    # True MEAN_REGIME requires: price path statistically "unsustainable"
    # TODO: Add distance-to-anchor (z-score), path length > limit, liquidity vacuum
    # NOTE: Thresholds relaxed to generate FR/ET data (2026-01-21)
    # Original: 2.5, 0.8, 0.4, 0.9 → Relaxed: 0.85 (percentile), 0.7, 0.5, 0.8
    # ⚠️ IMPORTANT: All thresholds are now percentile-based for consistency
    mean_deviation_window: int = 200
    mean_deviation_z_abs_min_pct: float = (
        0.85  # Percentile threshold (was hard threshold 2.0)
    )
    mean_path_length_min_pct: float = (
        0.7  # Relaxed from 0.8 (Extreme path length percentile)
    )
    mean_dir_sign_consistency_max_pct: float = (
        0.5  # Relaxed from 0.4 (Direction instability)
    )
    mean_atr_percentile_min: float = 0.8  # Relaxed from 0.9 (Vol spike proxy)
    mean_dir_conf_max: float = 0.4  # Raised (weaker constraint, not primary)
    mean_vol_spike_min: float = 2.0  # Simplified proxy

    # Jump risk percentile bands (relative, not absolute)
    # These define the regime layers instead of an absolute veto.
    jump_risk_no_trade_pct: float = 0.9  # Top 10% -> NO_TRADE
    jump_risk_te_min_pct: float = 0.6
    jump_risk_te_max_pct: float = 0.9
    jump_risk_tc_min_pct: float = 0.3
    jump_risk_tc_max_pct: float = 0.6
    jump_risk_mean_max_pct: float = 0.3  # Bottom 30% -> MEAN/IDLE candidate

    # Physics v2 (Recall-first) hard veto
    hard_jump_risk_pct: float = 0.98  # Extreme jump-only veto
    hard_atr_percentile_min: float = 0.2  # Exclude dead/illiquid regime

    # Regime strategy
    # - "score_shape": use physics_score + shape constraints (v1.x)
    # - "simple_band": use jump_risk bands only (v2.1 recall-first)
    regime_strategy: str = "simple_band"

    # Missing handling
    missing_default_false: bool = True


def _dir_conf(p: np.ndarray) -> np.ndarray:
    """Convert pred_dir_prob [0,1] to dir_conf [0,1]."""
    return np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)


def _compute_atr_slope(
    atr: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Compute ATR slope (rate of change)."""
    if len(atr) < window:
        return np.full(len(atr), np.nan)

    # Simple linear regression slope over window
    slopes = np.full(len(atr), np.nan)
    x = np.arange(window)

    for i in range(window - 1, len(atr)):
        y = atr[i - window + 1 : i + 1]
        if not np.all(np.isnan(y)):
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() >= 3:
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                # Linear regression slope
                n = len(x_valid)
                sum_x = x_valid.sum()
                sum_y = y_valid.sum()
                sum_xy = (x_valid * y_valid).sum()
                sum_x2 = (x_valid**2).sum()

                denominator = n * sum_x2 - sum_x**2
                if abs(denominator) > 1e-9:
                    slope = (n * sum_xy - sum_x * sum_y) / denominator
                    slopes[i] = slope

    return slopes / (atr + 1e-9)  # Normalize by ATR level


def _compute_range_expansion(
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Compute range expansion ratio (current range / ATR)."""
    if len(high) < window:
        return np.full(len(high), np.nan)

    ranges = (high - low) / (atr + 1e-9)
    expansion = np.full(len(ranges), np.nan)

    for i in range(window - 1, len(ranges)):
        window_ranges = ranges[i - window + 1 : i + 1]
        if not np.all(np.isnan(window_ranges)):
            recent = window_ranges[-3:]  # Last 3 bars
            historical = window_ranges[:-3]  # Earlier bars
            if len(historical) > 0 and not np.all(np.isnan(historical)):
                recent_mean = np.nanmean(recent)
                historical_mean = np.nanmean(historical)
                if historical_mean > 0:
                    expansion[i] = recent_mean / historical_mean

    return expansion


def _compute_jump_risk(
    close: np.ndarray,
    atr: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Compute jump risk (max abs return / std of returns)."""
    if len(close) < window:
        return np.full(len(close), np.nan)

    returns = np.diff(close) / (close[:-1] + 1e-9)
    jump_risk = np.full(len(close), np.nan)

    for i in range(window, len(close)):
        window_returns = returns[i - window : i]
        if not np.all(np.isnan(window_returns)):
            max_abs_ret = np.nanmax(np.abs(window_returns))
            std_ret = np.nanstd(window_returns)
            if std_ret > 1e-9:
                jump_risk[i] = max_abs_ret / std_ret

    return jump_risk


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        return np.full(len(arr), np.nan)
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).std().to_numpy()


def _rolling_dir_sign_consistency(p: np.ndarray, window: int) -> np.ndarray:
    """Rolling consistency of directional sign in [0,1]."""
    if len(p) < window:
        return np.full(len(p), np.nan)
    sign = np.sign(p - 0.5)
    s = pd.Series(sign)
    # abs(mean(sign)) = 1 if stable, ~0 if flipping
    return s.rolling(window=window, min_periods=window).mean().abs().to_numpy()


def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1], ignoring NaNs."""
    s = pd.Series(arr)
    return s.rank(pct=True).to_numpy()


def _compute_path_length(
    close: np.ndarray,
    atr: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Rolling path length in ATR units (sum of abs returns / ATR)."""
    if len(close) < window:
        return np.full(len(close), np.nan)
    diffs = np.abs(np.diff(close, prepend=close[0]))
    atr_safe = atr + 1e-9
    path = diffs / atr_safe
    s = pd.Series(path)
    return s.rolling(window=window, min_periods=window).sum().to_numpy()


def _compute_deviation_z(
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling z-score of close relative to local mean/std."""
    if len(close) < window:
        return np.full(len(close), np.nan)
    s = pd.Series(close)
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std()
    z = (s - mean) / (std + 1e-9)
    return z.to_numpy()


def classify_regime(
    df: pd.DataFrame,
    *,
    cfg: PhysicsRegimeConfig = PhysicsRegimeConfig(),
) -> pd.DataFrame:
    """
    Classify Physics/Regime for each timestamp.

    Returns DataFrame with 'regime' column containing:
    - TC_REGIME: Trend Continuation regime
    - TE_REGIME: Trend Expansion regime
    - MEAN_REGIME: Extreme Mean Reversion regime
    - NO_TRADE: No viable execution regime

    Args:
        df: DataFrame with required features and head outputs
        cfg: Configuration for regime classification

    Returns:
        DataFrame with 'regime' column added
    """
    out = pd.DataFrame(index=df.index)

    # Extract required columns
    def _get_col(col: str, default: Optional[float] = None) -> np.ndarray:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if cfg.missing_default_false and default is None:
            return np.full(len(df), np.nan)
        return np.full(len(df), default if default is not None else 0.0)

    pred_dir_prob = _get_col(cfg.pred_dir_prob_col)
    atr = _get_col(cfg.atr_col, default=1.0)
    atr_percentile = _get_col(cfg.atr_percentile_col)
    high = _get_col(cfg.high_col)
    low = _get_col(cfg.low_col)
    close = _get_col(cfg.close_col)

    # Compute dir_conf from pred_dir_prob
    pred_dir_prob_clipped = np.clip(pred_dir_prob, 0.0, 1.0)
    dir_conf = _dir_conf(pred_dir_prob_clipped)

    # Compute physics features
    atr_slope = _compute_atr_slope(atr, window=cfg.atr_slope_window)
    atr_slope_pct = _percentile_rank(atr_slope)
    range_expansion = _compute_range_expansion(
        high, low, atr, window=cfg.range_expansion_window
    )
    range_expansion_pct = _percentile_rank(range_expansion)
    jump_risk = _compute_jump_risk(close, atr, window=cfg.jump_risk_window)
    jump_risk_pct = _percentile_rank(jump_risk)
    dir_conf_std = _rolling_std(dir_conf, window=cfg.atr_slope_window)
    dir_conf_std_pct = _percentile_rank(dir_conf_std)
    dir_sign_consistency = _rolling_dir_sign_consistency(
        pred_dir_prob_clipped, window=cfg.atr_slope_window
    )
    dir_sign_consistency_pct = _percentile_rank(dir_sign_consistency)
    path_length = _compute_path_length(close, atr, window=cfg.atr_slope_window)
    path_length_pct = _percentile_rank(path_length)
    deviation_z = _compute_deviation_z(close, window=cfg.mean_deviation_window)
    deviation_z_abs = np.abs(deviation_z)
    deviation_z_abs_pct = _percentile_rank(
        deviation_z_abs
    )  # Convert to percentile for consistency

    # Initialize regime as NO_TRADE
    regime = np.full(len(df), "NO_TRADE", dtype=object)

    # Physics v2 hard veto (recall-first: only kill extremes)
    hard_veto = (
        ~np.isnan(jump_risk_pct) & (jump_risk_pct >= cfg.hard_jump_risk_pct)
    ) | (~np.isnan(atr_percentile) & (atr_percentile < cfg.hard_atr_percentile_min))

    # Relative jump risk bands (percentile-based)
    no_trade_band = ~np.isnan(jump_risk_pct) & (
        jump_risk_pct >= cfg.jump_risk_no_trade_pct
    )
    te_band = (
        ~np.isnan(jump_risk_pct)
        & (jump_risk_pct >= cfg.jump_risk_te_min_pct)
        & (jump_risk_pct < cfg.jump_risk_te_max_pct)
    )
    tc_band = (
        ~np.isnan(jump_risk_pct)
        & (jump_risk_pct >= cfg.jump_risk_tc_min_pct)
        & (jump_risk_pct < cfg.jump_risk_tc_max_pct)
    )
    mean_band = ~np.isnan(jump_risk_pct) & (jump_risk_pct < cfg.jump_risk_mean_max_pct)

    # Physics feasibility score (soft-min proxy)
    physics_inputs = np.stack(
        [
            1.0 - jump_risk_pct,
            1.0 - atr_slope_pct,
            path_length_pct,
            1.0 - dir_conf_std_pct,
            dir_sign_consistency_pct,
        ],
        axis=0,
    )
    physics_score = np.full(len(df), np.nan)
    valid_mask = ~np.all(np.isnan(physics_inputs), axis=0)
    if valid_mask.any():
        physics_score[valid_mask] = np.nanmin(physics_inputs[:, valid_mask], axis=0)
    physics_score_pct = _percentile_rank(physics_score)

    if cfg.regime_strategy == "simple_band":
        # v2.1 recall-first: regime by jump_risk bands only
        regime[tc_band & (~hard_veto)] = "TC_REGIME"
        regime[te_band & (~hard_veto)] = "TE_REGIME"
    else:
        # score_shape strategy (v1.x)
        tc_shape_ok = (
            ~np.isnan(atr_slope_pct) & (atr_slope_pct < cfg.tc_atr_slope_max_pct)
        ) & (
            ~np.isnan(path_length_pct) & (path_length_pct >= cfg.tc_path_length_min_pct)
        )
        tc_mask = (
            (
                ~np.isnan(physics_score_pct)
                & (physics_score_pct >= cfg.physics_score_min_pct)
            )
            & tc_shape_ok
            & tc_band
            & (regime == "NO_TRADE")
            & (~hard_veto)
        )
        regime[tc_mask] = "TC_REGIME"

        te_feasible_inputs = np.stack(
            [
                1.0 - atr_slope_pct,
                range_expansion_pct,
                path_length_pct,
                1.0 - dir_conf_std_pct,
                dir_sign_consistency_pct,
            ],
            axis=0,
        )
        te_score = np.full(len(df), np.nan)
        te_valid_mask = ~np.all(np.isnan(te_feasible_inputs), axis=0)
        if te_valid_mask.any():
            te_score[te_valid_mask] = np.nanmin(
                te_feasible_inputs[:, te_valid_mask], axis=0
            )
        te_score_pct = _percentile_rank(te_score)

        te_shape_ok = (
            ~np.isnan(atr_slope_pct) & (atr_slope_pct >= cfg.te_atr_slope_min_pct)
        ) & (
            ~np.isnan(range_expansion_pct)
            & (range_expansion_pct >= cfg.te_range_expansion_min_pct)
        )
        te_mask = (
            (~np.isnan(te_score_pct) & (te_score_pct >= cfg.physics_score_min_pct))
            & te_shape_ok
            & te_band
            & (regime == "NO_TRADE")
            & (~hard_veto)
        )
        regime[te_mask] = "TE_REGIME"

    # MEAN Regime: Extreme dislocations only
    # ⚠️ PROXY V0: Current definition is not fully physical
    # True MEAN_REGIME requires:
    # - distance-to-anchor extreme (z-score) - TODO
    # - unidirectional path_length > reasonable limit - TODO
    # - local liquidity vacuum + quick refill - TODO
    # Current implementation is a simplified proxy
    # NOTE: This should NOT be enabled in production execution
    # ⚠️ IMPORTANT: All thresholds are percentile-based for consistency with TC/TE
    mean_physical_ok = (
        (
            ~np.isnan(deviation_z_abs_pct)
            & (deviation_z_abs_pct >= cfg.mean_deviation_z_abs_min_pct)
        )
        & (
            ~np.isnan(path_length_pct)
            & (path_length_pct >= cfg.mean_path_length_min_pct)
        )
        & (
            ~np.isnan(dir_sign_consistency_pct)
            & (dir_sign_consistency_pct <= cfg.mean_dir_sign_consistency_max_pct)
        )
        & (~np.isnan(atr_percentile) & (atr_percentile >= cfg.mean_atr_percentile_min))
    )
    mean_mask = mean_physical_ok & mean_band & (regime == "NO_TRADE") & (~hard_veto)
    regime[mean_mask] = "MEAN_REGIME"

    out["regime"] = regime

    # Also output intermediate features for diagnostics
    out["dir_conf"] = dir_conf
    out["dir_conf_std"] = dir_conf_std
    out["dir_conf_std_pct"] = dir_conf_std_pct
    out["dir_sign_consistency"] = dir_sign_consistency
    out["dir_sign_consistency_pct"] = dir_sign_consistency_pct
    out["atr_slope"] = atr_slope
    out["atr_slope_pct"] = atr_slope_pct
    out["range_expansion"] = range_expansion
    out["range_expansion_pct"] = range_expansion_pct
    out["jump_risk"] = jump_risk
    out["jump_risk_pct"] = jump_risk_pct
    out["path_length"] = path_length
    out["path_length_pct"] = path_length_pct
    out["physics_score"] = physics_score
    out["physics_score_pct"] = physics_score_pct
    if cfg.regime_strategy == "simple_band":
        out["te_score"] = np.full(len(df), np.nan)
        out["te_score_pct"] = np.full(len(df), np.nan)
    else:
        out["te_score"] = te_score
        out["te_score_pct"] = te_score_pct
    out["deviation_z"] = deviation_z
    out["deviation_z_abs"] = deviation_z_abs
    out["deviation_z_abs_pct"] = deviation_z_abs_pct
    out["hard_veto"] = hard_veto

    # Semantic scores (do NOT affect regime assignment; consumed by Gate)
    tc_semantic_inputs = np.stack(
        [
            1.0 - atr_slope_pct,
            path_length_pct,
            1.0 - dir_conf_std_pct,
            dir_sign_consistency_pct,
        ],
        axis=0,
    )
    tc_semantic_score = np.full(len(df), np.nan)
    tc_sem_valid = ~np.all(np.isnan(tc_semantic_inputs), axis=0)
    if tc_sem_valid.any():
        tc_semantic_score[tc_sem_valid] = np.nanmin(
            tc_semantic_inputs[:, tc_sem_valid], axis=0
        )

    te_semantic_inputs = np.stack(
        [
            atr_slope_pct,
            range_expansion_pct,
            path_length_pct,
            1.0 - dir_conf_std_pct,
            dir_sign_consistency_pct,
        ],
        axis=0,
    )
    te_semantic_score = np.full(len(df), np.nan)
    te_sem_valid = ~np.all(np.isnan(te_semantic_inputs), axis=0)
    if te_sem_valid.any():
        te_semantic_score[te_sem_valid] = np.nanmin(
            te_semantic_inputs[:, te_sem_valid], axis=0
        )

    out["tc_semantic_score"] = tc_semantic_score
    out["te_semantic_score"] = te_semantic_score

    # FR/ET semantic scores (for MEAN_REGIME archetypes)
    # FR: Failure Reversion - extreme deviation + unstable direction + overextended path
    fr_semantic_inputs = np.stack(
        [
            np.clip(deviation_z_abs / 5.0, 0.0, 1.0),  # Normalize z-score to [0,1]
            1.0
            - dir_sign_consistency_pct,  # Direction instability (lower consistency = better for FR)
            path_length_pct,
            atr_percentile,
        ],
        axis=0,
    )
    fr_semantic_score = np.full(len(df), np.nan)
    fr_sem_valid = ~np.all(np.isnan(fr_semantic_inputs), axis=0)
    if fr_sem_valid.any():
        fr_semantic_score[fr_sem_valid] = np.nanmin(
            fr_semantic_inputs[:, fr_sem_valid], axis=0
        )

    # ET: Exhaustion Turn - volatility spike + overextended path + unstable direction
    et_semantic_inputs = np.stack(
        [
            atr_percentile,
            path_length_pct,
            1.0 - dir_sign_consistency_pct,  # Direction instability
            np.clip(deviation_z_abs / 5.0, 0.0, 1.0),  # Normalize z-score to [0,1]
        ],
        axis=0,
    )
    et_semantic_score = np.full(len(df), np.nan)
    et_sem_valid = ~np.all(np.isnan(et_semantic_inputs), axis=0)
    if et_sem_valid.any():
        et_semantic_score[et_sem_valid] = np.nanmin(
            et_semantic_inputs[:, et_sem_valid], axis=0
        )

    out["fr_semantic_score"] = fr_semantic_score
    out["et_semantic_score"] = et_semantic_score

    return out
