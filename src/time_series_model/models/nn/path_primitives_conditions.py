from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SRFuseConditionConfig:
    """
    SR-near condition aligned with VectorBTBacktest._apply_sr_fuse semantics.

    dist_to_nearest_sr is typically a relative distance (pct of price). We normalize:
      norm_dist_atr = (abs(dist_pct) * price) / atr
    and gate by max_dist_atr.
    """

    dist_col: str = "dist_to_nearest_sr"
    atr_col: str = "atr"
    price_col: str = "close"
    max_dist_atr: float = 3.0
    dist_is_pct: Optional[bool] = True  # if None -> auto heuristic


def compute_near_sr_mask(df: pd.DataFrame, *, cfg: SRFuseConditionConfig) -> pd.Series:
    """
    Compute a boolean mask indicating whether each bar is 'near SR' (normalized by ATR),
    consistent with the backtest sr_fuse.
    """
    if cfg.dist_col not in df.columns or cfg.price_col not in df.columns:
        return pd.Series(False, index=df.index)

    dist_raw = pd.to_numeric(df[cfg.dist_col], errors="coerce").abs().astype(float)
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)

    if cfg.atr_col in df.columns:
        atr = pd.to_numeric(df[cfg.atr_col], errors="coerce").astype(float)
    else:
        atr = pd.Series(np.nan, index=df.index, dtype=float)

    if cfg.dist_is_pct is None:
        q95 = float(dist_raw.dropna().quantile(0.95)) if dist_raw.notna().any() else 0.0
        dist_is_pct = bool(q95 <= 2.0)
    else:
        dist_is_pct = bool(cfg.dist_is_pct)

    abs_dist = dist_raw * price if dist_is_pct else dist_raw
    norm_dist_atr = abs_dist / (atr + 1e-8)
    return (norm_dist_atr <= float(cfg.max_dist_atr)).fillna(False).astype(bool)
