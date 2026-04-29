"""Time-series quantile of semantic_chop within one symbol (causal rolling window).

Used for gate thresholds in [0, 1]: high = current chop is elevated vs recent
history for the same series. Constant window → 0.0 (do not force-deny).

Default window 1200 bars @ 2H ≈ 100 calendar days (~3 months of 2H bars);
``min_periods`` scales with window (~12.5% coverage, same ratio as legacy 60/480).

Implementation uses a NumPy loop (same semantics as former SciPy
``percentileofscore(..., kind='mean')`` on each window) — avoids
``Series.rolling().apply`` Python callback overhead on long series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Production defaults for BPC/TPC/ME soft-phase chop_ts_q (single source of truth).
DEFAULT_CHOP_TS_WINDOW = 1200
DEFAULT_CHOP_TS_MIN_PERIODS = 150


def _percentileofscore_mean_last(xv: np.ndarray) -> float:
    """``percentileofscore(xv, xv[-1], kind='mean') / 100`` for finite 1d ``xv``."""
    if xv.size < 2:
        return float("nan")
    if float(np.std(xv)) < 1e-12:
        return 0.0
    last = float(xv[-1])
    weak = np.count_nonzero(xv <= last) / xv.size * 100.0
    strict = np.count_nonzero(xv < last) / xv.size * 100.0
    return (weak + strict) * 0.5 / 100.0


def semantic_chop_ts_quantile(
    chop: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    window: int = DEFAULT_CHOP_TS_WINDOW,
    min_periods: int = DEFAULT_CHOP_TS_MIN_PERIODS,
) -> np.ndarray:
    x = np.asarray(chop, dtype=float)
    n = int(x.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if len(index) != n:
        raise ValueError("chop length must match len(index)")
    out = np.full(n, np.nan, dtype=float)
    win = int(window)
    min_p = int(min_periods)
    for i in range(n):
        lo = max(0, i - win + 1)
        xv_raw = x[lo : i + 1]
        if xv_raw.size < min_p:
            continue
        xv = xv_raw[np.isfinite(xv_raw)]
        if xv.size < 2:
            continue
        out[i] = _percentileofscore_mean_last(xv)
    return out
