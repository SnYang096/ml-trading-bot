from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PathPrimitivesLabelConfig:
    """
    Label config for path primitives.

    Important:
    - Use entry = open[t + entry_offset] by default to align with backtests/execution.
    - Use high/low scanning inside the future window to match intra-bar semantics.
    - Use ATR(t) for normalization; NaN/zero ATR will invalidate samples.
    """

    horizon_bars: int
    entry_offset: int = 1
    entry_price_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    atr_col: str = "atr"
    atr_eps: float = 1e-8

    # Clip to prevent extreme tails from dominating regression losses
    cap_mfe_atr: float = 10.0
    cap_mae_atr: float = 10.0

    # Optional: 3-class direction with neutral band in ATR units (|net| < band => neutral)
    use_3class_dir: bool = False
    neutral_band_atr: float = 0.2

    # Optional: right-censoring flag when MFE happens near window end (slow trend)
    compute_censor_flag: bool = True
    censor_last_k: int = 2


def _compute_path_primitives_labels_single(
    df: pd.DataFrame,
    *,
    cfg: PathPrimitivesLabelConfig,
    out_prefix: str = "",
) -> pd.DataFrame:
    """
    Compute counterfactual path primitives labels aligned to timestamp t (feature time).

    Returns a DataFrame with columns (prefixed by out_prefix):
    - dir_y: 0/1 (or -1/0/1 if use_3class_dir=True)
    - mfe_atr: >=0
    - mae_atr: >=0
    - t_to_mfe: integer bars in [0, H]
    - mfe_valid: 0/1 (whether max_up > 0)
    - mfe_censored: 0/1 (optional)

    Notes:
    - Labels are defined relative to entry=open[t+entry_offset], not close[t].
    - Uses high/low within future window to measure excursions.
    """

    H = int(cfg.horizon_bars)
    if H <= 0:
        raise ValueError(f"horizon_bars must be positive, got {H}")

    required = {
        cfg.entry_price_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.atr_col,
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for path labels: {missing}")

    n = len(df)
    out = pd.DataFrame(index=df.index)

    # Output arrays
    dir_y = np.full(n, np.nan, dtype=float)
    mfe_atr = np.full(n, np.nan, dtype=float)
    mae_atr = np.full(n, np.nan, dtype=float)
    t_to_mfe = np.full(n, np.nan, dtype=float)
    mfe_valid = np.full(n, np.nan, dtype=float)
    mfe_censored = np.full(n, np.nan, dtype=float) if cfg.compute_censor_flag else None

    entry_price = pd.to_numeric(df[cfg.entry_price_col], errors="coerce").to_numpy(
        dtype=float
    )
    high = pd.to_numeric(df[cfg.high_col], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df[cfg.low_col], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df[cfg.close_col], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(df[cfg.atr_col], errors="coerce").to_numpy(dtype=float)

    # We need: t + entry_offset + horizon_bars within bounds, and close at window end exists.
    last_i = n - (cfg.entry_offset + H) - 1
    if last_i <= 0:
        # Not enough rows to compute any labels; return empty NaNs with correct columns.
        pass
    else:
        for i in range(last_i):
            entry_i = i + cfg.entry_offset
            e_px = entry_price[entry_i]
            a = atr[i]

            if not np.isfinite(e_px) or not np.isfinite(a) or a <= cfg.atr_eps:
                continue

            # Future window inclusive of entry bar: [entry_i, entry_i+H]
            s = entry_i
            e = entry_i + H

            win_high = high[s : e + 1]
            win_low = low[s : e + 1]
            if win_high.size == 0 or win_low.size == 0:
                continue

            max_up = np.nanmax(win_high - e_px)
            max_dn = np.nanmax(e_px - win_low)  # drawdown magnitude (>=0)
            if not np.isfinite(max_up) or not np.isfinite(max_dn):
                continue

            # Normalize by ATR(t)
            mfe = max_up / (a + cfg.atr_eps)
            mae = max_dn / (a + cfg.atr_eps)

            mfe = float(np.clip(mfe, 0.0, cfg.cap_mfe_atr))
            mae = float(np.clip(mae, 0.0, cfg.cap_mae_atr))

            mfe_atr[i] = mfe
            mae_atr[i] = mae

            # Direction label: excursion dominance (path fact, not strategy)
            if cfg.use_3class_dir:
                net = (close[e] - e_px) / (a + cfg.atr_eps)
                if not np.isfinite(net):
                    continue
                if abs(net) < cfg.neutral_band_atr:
                    d = 0.0
                else:
                    d = 1.0 if mfe > mae else -1.0
            else:
                d = 1.0 if mfe > mae else 0.0
            dir_y[i] = d

            # t_to_mfe: index of max favorable excursion (based on high)
            # Always define t_to_mfe, but we'll mask in training when mfe_valid==0.
            t_idx = int(np.nanargmax(win_high - e_px)) if np.isfinite(max_up) else 0
            t_to_mfe[i] = float(t_idx)

            # mfe_valid: whether there exists any upside above entry in the window
            valid = 1.0 if max_up > 0 else 0.0
            mfe_valid[i] = valid

            if cfg.compute_censor_flag and mfe_censored is not None:
                mfe_censored[i] = 1.0 if t_idx >= (H - int(cfg.censor_last_k)) else 0.0

    out[f"{out_prefix}dir_y"] = dir_y
    out[f"{out_prefix}mfe_atr"] = mfe_atr
    out[f"{out_prefix}mae_atr"] = mae_atr
    out[f"{out_prefix}t_to_mfe"] = t_to_mfe
    out[f"{out_prefix}mfe_valid"] = mfe_valid
    if cfg.compute_censor_flag and mfe_censored is not None:
        out[f"{out_prefix}mfe_censored"] = mfe_censored
    return out


def compute_path_primitives_labels(
    df: pd.DataFrame,
    *,
    cfg: PathPrimitivesLabelConfig,
    out_prefix: str = "",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Group-safe wrapper around label computation.

    When group_col is provided (e.g., "symbol"), labels are computed independently
    per group to avoid horizon leakage across symbol boundaries.
    """
    if group_col is None or group_col not in df.columns:
        return _compute_path_primitives_labels_single(
            df, cfg=cfg, out_prefix=out_prefix
        )

    parts = []
    for _, g in df.groupby(group_col, sort=False):
        parts.append(
            _compute_path_primitives_labels_single(g, cfg=cfg, out_prefix=out_prefix)
        )
    if not parts:
        return _compute_path_primitives_labels_single(
            df.iloc[:0], cfg=cfg, out_prefix=out_prefix
        )
    out = pd.concat(parts, axis=0).reindex(df.index)
    return out
