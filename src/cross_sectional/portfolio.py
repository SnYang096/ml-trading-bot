from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PortfolioConstraints:
    max_weight_per_asset: float = 0.1
    long_short: bool = True
    gross_leverage: float = 1.0  # sum(|w|) <= gross_leverage
    turnover_penalty: float = 0.0  # lambda for |w - w_prev|
    cost_per_unit: float = 0.0  # linear cost per weight unit moved
    target_vol: Optional[float] = None  # optional scaling target


def _normalize_weights(weights: pd.Series, gross: float) -> pd.Series:
    s = weights.abs().sum()
    if s <= 0:
        return weights * 0.0
    return weights * (gross / s)


def _apply_caps(weights: pd.Series, cap: float) -> pd.Series:
    if cap is None or cap <= 0:
        return weights
    capped = weights.clip(lower=-cap, upper=cap)
    # re-normalize to original gross if any clipping occurred
    gross = weights.abs().sum()
    if gross <= 0:
        return capped
    new_gross = capped.abs().sum()
    if new_gross > 0:
        capped = capped * (gross / new_gross)
    return capped


def construct_portfolio(
    panel: pd.DataFrame,
    score_col: str,
    *,
    timestamp_level: int = 0,
    constraints: Optional[PortfolioConstraints] = None,
    prev_weights: Optional[pd.Series] = None,
    cost_col: Optional[str] = None,
) -> pd.Series:
    """
    Construct cross-sectional weights from scores with basic constraints and optional costs.
    Returns weights for the last timestamp slice in the panel.
    """
    if constraints is None:
        constraints = PortfolioConstraints()
    if score_col not in panel.columns:
        raise ValueError(f"score_col '{score_col}' missing in panel")
    last_ts = panel.index.get_level_values(timestamp_level).max()
    cs = panel.xs(last_ts, level=timestamp_level).copy()
    if cs.empty:
        return pd.Series(dtype=float)

    scores = cs[score_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Long-short or long-only transform
    if constraints.long_short:
        # Centered scores -> allow positive and negative exposures
        centered = scores - scores.mean()
        weights = centered
    else:
        # Long-only via percentile rank in [0,1]
        ranks = scores.rank(pct=True)
        weights = ranks.clip(lower=0.0)
        weights[weights < 0] = 0.0

    # Apply turnover and linear cost penalty (shrink weights toward prev)
    if prev_weights is not None and (constraints.turnover_penalty > 0.0 or constraints.cost_per_unit > 0.0):
        prev = prev_weights.reindex(weights.index).fillna(0.0)
        delta = weights - prev
        shrink = constraints.turnover_penalty + constraints.cost_per_unit
        weights = weights - shrink * np.sign(delta)

    # Cap per-asset weight and normalize to gross leverage
    weights = _apply_caps(weights, constraints.max_weight_per_asset)
    weights = _normalize_weights(weights, constraints.gross_leverage)

    # Optional volatility targeting (if per-asset vol provided in panel)
    if constraints.target_vol is not None and "asset_vol" in cs.columns:
        asset_vol = cs["asset_vol"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(asset_vol.median())
        # Inverse-vol scaling
        inv_vol = 1.0 / asset_vol.replace(0.0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
        weights = weights * inv_vol
        weights = _normalize_weights(weights, constraints.gross_leverage)

    return weights.astype(float)


def overlay_regime_weights(
    weights: pd.Series, regime_state: str, *,
    trend_gain: float = 1.2, range_gain: float = 0.9, collapse_gain: float = 0.6
) -> pd.Series:
    """
    Overlay regime-aware multiplicative gains to the entire cross-sectional book.
    """
    gain = 1.0
    state = (regime_state or "").lower()
    if "trend" in state:
        gain = trend_gain
    elif "range" in state:
        gain = range_gain
    elif "collapse" in state or "bear" in state:
        gain = collapse_gain
    w = weights * gain
    # Keep gross the same after overlay
    return _normalize_weights(w, weights.abs().sum())


