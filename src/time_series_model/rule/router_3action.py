from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule3ActionConfig:
    """
    Pure rule router based only on nnmultihead heads (no extra strategy features).

    It maps each timestamp into one of:
      - NO_TRADE
      - MEAN
      - TREND

    Assumptions:
    - pred_mfe_atr/pred_mae_atr/pred_t_to_mfe are in "training space" by default (log1p),
      unless preds_in_log1p=False.
    """

    # Input columns
    pred_dir_prob_col: str = "pred_dir_prob"
    pred_mfe_col: str = "pred_mfe_atr"
    pred_mae_col: str = "pred_mae_atr"
    pred_ttm_col: str = "pred_t_to_mfe"

    # Decision thresholds (in ATR units after inverse-transform)
    mfe_min: float = 0.4
    eff_min: float = 1.05  # mfe/(mae+eps) minimum to trade

    # Trend conditions
    dir_conf_trend_min: float = 0.25  # abs(p-0.5)*2
    mfe_trend_min: float = 0.8
    ttm_trend_min: float = 8.0

    # Mean conditions
    eff_mean_min: float = 1.15
    ttm_mean_max: float = 12.0

    eps: float = 1e-9


def _dir_conf(p: np.ndarray) -> np.ndarray:
    # p in [0,1] -> confidence in [0,1]
    return np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)


def _maybe_expm1(x: np.ndarray, *, preds_in_log1p: bool) -> np.ndarray:
    if not preds_in_log1p:
        return x
    # model heads are strictly positive; expm1 keeps 0->0
    # Clip to avoid overflow when preds are extreme (e.g. bad model weights / untrained run).
    # expm1(15) ~= 3.27e6 which is already "effectively infinite" for our thresholds.
    return np.expm1(np.clip(x, 0.0, 15.0))


def compute_mode_3action(
    df: pd.DataFrame,
    *,
    cfg: Rule3ActionConfig = Rule3ActionConfig(),
    preds_in_log1p: bool = True,
    out_col: str = "mode",
) -> pd.DataFrame:
    """
    Return a dataframe with columns:
      - mode (str): NO_TRADE/MEAN/TREND
      - mode_action (int): 0/1/2
      - optional derived diagnostics: mfe, mae, ttm, eff, dir_conf
    """
    missing = [
        c
        for c in [
            cfg.pred_dir_prob_col,
            cfg.pred_mfe_col,
            cfg.pred_mae_col,
            cfg.pred_ttm_col,
        ]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required pred columns for rule router: {missing}")

    p = pd.to_numeric(df[cfg.pred_dir_prob_col], errors="coerce").to_numpy(dtype=float)
    mfe_p = pd.to_numeric(df[cfg.pred_mfe_col], errors="coerce").to_numpy(dtype=float)
    mae_p = pd.to_numeric(df[cfg.pred_mae_col], errors="coerce").to_numpy(dtype=float)
    ttm_p = pd.to_numeric(df[cfg.pred_ttm_col], errors="coerce").to_numpy(dtype=float)

    mfe = _maybe_expm1(mfe_p, preds_in_log1p=preds_in_log1p)
    mae = _maybe_expm1(mae_p, preds_in_log1p=preds_in_log1p)
    ttm = _maybe_expm1(ttm_p, preds_in_log1p=preds_in_log1p)

    dconf = _dir_conf(np.clip(p, 0.0, 1.0))
    # Avoid invalid divisions from NaN/Inf.
    eff = np.where(
        np.isfinite(mfe) & np.isfinite(mae),
        mfe / (mae + float(cfg.eps)),
        0.0,
    )

    # Default: NO_TRADE
    mode = np.full(len(df), "NO_TRADE", dtype=object)
    action = np.zeros(len(df), dtype=int)

    tradable = (
        np.isfinite(mfe)
        & np.isfinite(mae)
        & (mfe >= float(cfg.mfe_min))
        & (eff >= float(cfg.eff_min))
    )

    # Trend: strong directional confidence, larger mfe, slower ttm
    trend = (
        tradable
        & (dconf >= float(cfg.dir_conf_trend_min))
        & (mfe >= float(cfg.mfe_trend_min))
        & (ttm >= float(cfg.ttm_trend_min))
    )

    # Mean: better efficiency and quicker ttm (wins fast)
    mean = (
        tradable
        & ~trend
        & (eff >= float(cfg.eff_mean_min))
        & (ttm <= float(cfg.ttm_mean_max))
    )

    mode[mean] = "MEAN"
    action[mean] = 1
    mode[trend] = "TREND"
    action[trend] = 2

    out = pd.DataFrame(index=df.index)
    out[out_col] = mode.astype(str)
    out["mode_action"] = action.astype(int)
    out["mfe_atr"] = mfe
    out["mae_atr"] = mae
    out["t_to_mfe"] = ttm
    out["eff"] = eff
    out["dir_conf"] = dconf
    return out
