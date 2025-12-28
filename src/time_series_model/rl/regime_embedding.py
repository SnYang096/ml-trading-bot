from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeEmbeddingConfig:
    """
    A minimal, production-safe "regime embedding" implemented as one-hot features.

    Design constraints:
    - Must not use future information (no rolling quantiles over the full series).
    - Must be derived from already-available Router-level observables (heads/drawdown).
    - Intended for A/B evaluation only; keep rules simple and auditable.
    """

    n_buckets: int = 4
    eps: float = 1e-9

    # input columns (already in logs)
    head_dir_score_col: str = "head_dir_score"
    head_mfe_col: str = "head_mfe_atr"
    head_mae_col: str = "head_mae_atr"
    head_ttm_col: str = "head_t_to_mfe"
    drawdown_col: str = "drawdown"

    out_bucket_col: str = "regime_bucket"
    out_prefix: str = "regime_"


def add_regime_onehot(
    df_logs: pd.DataFrame, *, cfg: RegimeEmbeddingConfig = RegimeEmbeddingConfig()
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add:
      - cfg.out_bucket_col: integer bucket in [0, n_buckets-1]
      - one-hot columns: f"{cfg.out_prefix}{i}" for i in 0..n_buckets-1

    Returns (df_out, onehot_cols).
    """
    df = df_logs.copy()

    # Basic proxies derived from heads (current-step only)
    dir_s = (
        pd.to_numeric(df.get(cfg.head_dir_score_col), errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    mfe = (
        pd.to_numeric(df.get(cfg.head_mfe_col), errors="coerce")
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    mae = (
        pd.to_numeric(df.get(cfg.head_mae_col), errors="coerce")
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    ttm = (
        pd.to_numeric(df.get(cfg.head_ttm_col), errors="coerce")
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    dd = (
        pd.to_numeric(df.get(cfg.drawdown_col), errors="coerce")
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )

    eff = mfe / (mae + float(cfg.eps))
    abs_dir = dir_s.abs()

    # Deterministic buckets (auditable)
    # 0: low opportunity / low signal strength
    # 1: mean-ish (good efficiency but weak direction, shorter ttm)
    # 2: trend-ish (good efficiency + strong direction, longer ttm)
    # 3: risk-off (drawdown elevated)
    bucket = np.zeros(len(df), dtype=int)
    bucket[(mfe < 0.4) | (eff < 1.05)] = 0
    bucket[(dd >= 0.12)] = 3
    bucket[(bucket == 0) & (eff >= 1.2) & (abs_dir < 0.25) & (ttm <= 12.0)] = 1
    bucket[(bucket == 0) & (eff >= 1.2) & (abs_dir >= 0.25) & (ttm >= 12.0)] = 2

    # Clamp to range
    bucket = bucket.clip(0, int(cfg.n_buckets) - 1)
    df[cfg.out_bucket_col] = bucket.astype(int)

    onehot_cols: List[str] = []
    for i in range(int(cfg.n_buckets)):
        c = f"{cfg.out_prefix}{i}"
        onehot_cols.append(c)
        df[c] = (df[cfg.out_bucket_col].to_numpy(dtype=int) == i).astype(float)

    return df, onehot_cols
