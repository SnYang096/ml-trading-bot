"""Signal timeframe vs execution timeframe replay for multi-leg research.

Segments and regime masks are computed on a **signal** OHLCV series (e.g. 2h).
Inventory simulation runs on a **finer** OHLCV series (e.g. 1min, 100ms) with
signal features asof-joined onto each execution bar.

Live-aligned segment clock (matches ``FeatureStoreBarProvider`` +
``ChopGridLiveEngine``):

- Regime features from a left-labelled signal bar ``k`` are only knowable after
  that bar closes: ``t_confirm = signal_index[k] + signal_bar_delta``.
- Grid inventory simulation for signal segment ``[s, e]`` runs on execution bars
  in ``[t_enter, t_exit)`` where:

  - ``t_enter = signal_index[s] + signal_bar_delta`` (entry bar confirmed)
  - ``t_exit = signal_index[e] + signal_bar_delta`` (exit bar confirmed)

Grid **pricing** (center / spacing) stays anchored to signal bar ``s`` close/ATR;
only **fills and TP** use the execution path inside ``[t_enter, t_exit)``.

This mirrors ``event_backtest`` / ``_iter_update_bars_1min``: decision clock vs
fill clock are separated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Parse a pandas-compatible bar length (e.g. ``2h``, ``120T``, ``1min``)."""
    tf = str(timeframe or "").strip()
    if not tf:
        raise ValueError("empty timeframe")
    if tf.lower() in {"100ms", "100MS"}:
        return pd.Timedelta(milliseconds=100)
    return pd.to_timedelta(tf)


def segment_execution_bounds(
    signal_index: pd.DatetimeIndex,
    s: int,
    e: int,
    signal_bar_delta: pd.Timedelta,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return live-aligned ``(t_enter, t_exit)`` for signal segment ``[s, e]``.

    Execution bars use half-open interval ``[t_enter, t_exit)``.
    """
    t_enter = pd.Timestamp(signal_index[s]) + signal_bar_delta
    t_exit = pd.Timestamp(signal_index[e]) + signal_bar_delta
    return t_enter, t_exit


def merge_signal_features_onto_execution_bars(
    ohlc_exec: pd.DataFrame,
    df_signal: pd.DataFrame,
    signal_bar_delta: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Attach signal columns to each execution bar via backward ``merge_asof``.

    OHLCV on the execution frame wins; overlapping price columns on ``df_signal``
    are dropped before the join so we keep exchange micro-path from execution bars.
    If ``signal_bar_delta`` is provided, left-labelled signal rows become
    available only at the bar's right edge, avoiding use of a 2h close inside
    the same unfinished 2h bar.
    """
    if ohlc_exec.empty:
        return ohlc_exec
    drop_price = {"open", "high", "low", "close", "volume"}
    sig_cols = [c for c in df_signal.columns if c not in drop_price]
    sig = df_signal[sig_cols].sort_index().reset_index(names="sig_ts")
    if signal_bar_delta is not None:
        sig["sig_ts"] = sig["sig_ts"] + signal_bar_delta
    left = ohlc_exec.sort_index().reset_index(names="ts")
    merged = pd.merge_asof(
        left,
        sig,
        left_on="ts",
        right_on="sig_ts",
        direction="backward",
    )
    merged = merged.set_index("ts").sort_index()
    if "sig_ts" in merged.columns:
        merged = merged.drop(columns=["sig_ts"])
    return merged


def slice_execution_window(
    df_exec: pd.DataFrame,
    signal_index: pd.DatetimeIndex,
    s: int,
    e: int,
    signal_bar_delta: pd.Timedelta,
) -> pd.DataFrame:
    """Return execution bars for live-aligned segment window ``[t_enter, t_exit)``.

    Signal bars use **left-labelled** indices spanning
    ``[signal_index[k], signal_index[k] + delta)`` (same convention as
    ``_resample_ohlcv(..., label='left', closed='left')``).
    """
    t_enter, t_exit = segment_execution_bounds(signal_index, s, e, signal_bar_delta)
    sub = df_exec.loc[(df_exec.index >= t_enter) & (df_exec.index < t_exit)]
    return sub.copy()
