"""
Shared direction rule helpers — single implementation for live, validation, backtest.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

METHOD_DUAL_POSITION_AGREE_DEADBAND = "dual_position_agree_deadband"
METHOD_SINGLE_POSITION_BAND = "single_position_band"


def dual_position_agree_deadband_scalar(v1: Any, v2: Any, epsilon: float) -> int:
    """
    Strong long: v1 > eps and v2 > eps -> +1
    Strong short: v1 < -eps and v2 < -eps -> -1
    Else -> 0 (including NaN, missing, disagree, or either in [-eps, eps]).
    """
    eps = float(epsilon)
    if eps < 0:
        eps = 0.0
    try:
        p1 = float(v1)
        p2 = float(v2)
    except (TypeError, ValueError):
        return 0
    if np.isnan(p1) or np.isnan(p2):
        return 0
    if p1 > eps and p2 > eps:
        return 1
    if p1 < -eps and p2 < -eps:
        return -1
    return 0


def dual_position_agree_deadband_series(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    epsilon: float,
) -> pd.Series:
    """Per-row {-1,0,+1} aligned to df.index; missing columns -> all zeros."""
    if col_a not in df.columns or col_b not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)

    eps = float(epsilon)
    if eps < 0:
        eps = 0.0

    s1 = pd.to_numeric(df[col_a], errors="coerce").astype(float)
    s2 = pd.to_numeric(df[col_b], errors="coerce").astype(float)

    out = pd.Series(0.0, index=df.index, dtype=float)
    valid = s1.notna() & s2.notna()
    long_m = valid & (s1 > eps) & (s2 > eps)
    short_m = valid & (s1 < -eps) & (s2 < -eps)
    out.loc[long_m] = 1.0
    out.loc[short_m] = -1.0
    return out


def single_position_band_scalar(pos: Any, inner_abs: float, outer_abs: float) -> int:
    """Band-pass on normalized position (e.g. macro_tp_vwap_1200_position).

    Long: inner < pos < outer. Short: -outer < pos < -inner.
    Else (too near VWAP, overextended, NaN): 0.
    Requires 0 <= inner < outer.
    """
    try:
        inner = float(inner_abs)
        outer = float(outer_abs)
    except (TypeError, ValueError):
        return 0
    if inner < 0 or outer <= inner:
        return 0
    try:
        p = float(pos)
    except (TypeError, ValueError):
        return 0
    if np.isnan(p):
        return 0
    if inner < p < outer:
        return 1
    if -outer < p < -inner:
        return -1
    return 0


def single_position_band_series(
    df: pd.DataFrame,
    col: str,
    inner_abs: float,
    outer_abs: float,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    try:
        inner = float(inner_abs)
        outer = float(outer_abs)
    except (TypeError, ValueError):
        return pd.Series(0.0, index=df.index, dtype=float)
    if inner < 0 or outer <= inner:
        return pd.Series(0.0, index=df.index, dtype=float)

    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    out = pd.Series(0.0, index=df.index, dtype=float)
    valid = s.notna()
    long_m = valid & (s > inner) & (s < outer)
    short_m = valid & (s > -outer) & (s < -inner)
    out.loc[long_m] = 1.0
    out.loc[short_m] = -1.0
    return out


def parse_single_position_band_rule(
    rule: dict,
) -> Optional[tuple[str, float, float]]:
    """Return (feature_col, inner_abs, outer_abs) for single_position_band."""
    if not isinstance(rule, dict):
        return None
    if rule.get("method") != METHOD_SINGLE_POSITION_BAND:
        return None
    feat = rule.get("feature")
    if not isinstance(feat, str) or not feat.strip():
        return None
    try:
        inner = float(rule.get("inner_abs", rule.get("inner", 0.0)))
        outer = float(rule.get("outer_abs", rule.get("outer", 0.0)))
    except (TypeError, ValueError):
        return None
    return (feat.strip(), inner, outer)


def parse_dual_rule(rule: dict) -> Optional[tuple[str, str, float]]:
    """Return (col_a, col_b, epsilon) if rule is a valid dual deadband rule."""
    if not isinstance(rule, dict):
        return None
    if rule.get("method") != METHOD_DUAL_POSITION_AGREE_DEADBAND:
        return None
    feats = rule.get("features")
    if not isinstance(feats, list) or len(feats) != 2:
        return None
    a, b = str(feats[0]).strip(), str(feats[1]).strip()
    if not a or not b:
        return None
    try:
        eps = float(rule.get("epsilon", 0.0))
    except (TypeError, ValueError):
        eps = 0.0
    return (a, b, eps)


def direction_rule_ft_key(rule: dict) -> Tuple[Any, ...]:
    """Dedup key for direction rules (dual: method+cols+eps; else feature+transform)."""
    if not isinstance(rule, dict):
        return (None, None)
    parsed = parse_dual_rule(rule)
    if parsed is not None:
        a, b, e = parsed
        return ("dual_position_agree_deadband", a, b, float(e))
    band = parse_single_position_band_rule(rule)
    if band is not None:
        f, inn, out = band
        return ("single_position_band", f, float(inn), float(out))
    return (rule.get("feature"), rule.get("transform"))


def is_direction_rule_enabled(rule: dict) -> bool:
    """False only when enabled is explicitly false (same idea as entry_filter.enabled)."""
    if not isinstance(rule, dict):
        return True
    return rule.get("enabled", True) is not False
