from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Alpha101CSConfig:
    """
    Minimal configuration for Alpha101 cross-sectional factor computation.

    Notes:
    - No ticks required.
    - `vwap` is approximated from OHLC as typical price.
    """

    vwap_mode: str = "typical"  # typical | ohlc4


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = (
            df.index.tz_localize("UTC")
            if df.index.tz is None
            else df.index.tz_convert("UTC")
        )
        out = df.copy()
        out.index = idx
        return out
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        out = df.copy()
        out["timestamp"] = ts
        out = out.dropna(subset=["timestamp"]).set_index("timestamp")
        return out
    raise ValueError("Expected DatetimeIndex or 'timestamp' column")


def _build_wide(frames: Mapping[str, pd.DataFrame], col: str) -> pd.DataFrame:
    parts = []
    for sym, df in frames.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        parts.append(s.rename(sym))
    if not parts:
        return pd.DataFrame()
    wide = pd.concat(parts, axis=1)
    wide.index = pd.to_datetime(wide.index, utc=True, errors="coerce")
    wide = wide.sort_index()
    return wide


def _infer_returns(close: pd.DataFrame) -> pd.DataFrame:
    # Avoid deprecated implicit fill_method='pad' to keep behavior explicit.
    return close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def _infer_adv20(volume: pd.DataFrame) -> pd.DataFrame:
    return volume.rolling(20, min_periods=1).mean()


def _infer_vwap(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    mode = str(mode).strip().lower()
    if mode == "ohlc4":
        return (open_ + high + low + close) / 4.0
    # typical price (H+L+C)/3
    return (high + low + close) / 3.0


def _series_to_panel(s: pd.Series, *, colname: str) -> pd.DataFrame:
    """
    Alpha functions return a Series with MultiIndex after stack/swaplevel.
    Normalize to MultiIndex (timestamp, symbol) and single column.
    """
    if (
        not isinstance(s, pd.Series)
        or not isinstance(s.index, pd.MultiIndex)
        or s.index.nlevels != 2
    ):
        raise TypeError("Expected alpha output as Series with 2-level MultiIndex")

    idx_names = list(s.index.names)
    # Common case: (ticker, date) due to swaplevel; make it (date, ticker)
    a0 = s.index.get_level_values(0)
    a1 = s.index.get_level_values(1)
    # Heuristic: timestamp-like level is datetime
    if pd.api.types.is_datetime64_any_dtype(
        a0
    ) and not pd.api.types.is_datetime64_any_dtype(a1):
        ts = pd.to_datetime(a0, utc=True, errors="coerce")
        sym = a1.astype(str)
        mi = pd.MultiIndex.from_arrays([ts, sym], names=["timestamp", "symbol"])
    elif pd.api.types.is_datetime64_any_dtype(a1):
        ts = pd.to_datetime(a1, utc=True, errors="coerce")
        sym = a0.astype(str)
        mi = pd.MultiIndex.from_arrays([ts, sym], names=["timestamp", "symbol"])
    else:
        # fallback: assume level1 is timestamp
        ts = pd.to_datetime(a1, utc=True, errors="coerce")
        sym = a0.astype(str)
        mi = pd.MultiIndex.from_arrays([ts, sym], names=["timestamp", "symbol"])

    out = s.copy()
    out.index = mi
    return out.rename(colname).to_frame()


def compute_alpha101_cs_rank_panel(
    frames: Mapping[str, pd.DataFrame],
    *,
    alpha_ids: Sequence[int],
    cfg: Alpha101CSConfig = Alpha101CSConfig(),
) -> pd.DataFrame:
    """
    Compute selected Alpha101 factors (original cross-sectional variants) on multi-asset wide data.

    Args:
        frames: symbol -> OHLCV dataframe (DatetimeIndex). Must include open/high/low/close/volume
        alpha_ids: e.g. [2,3,6,...,101]
    Returns:
        MultiIndex (timestamp, symbol) with columns: alpha101_cs_<id>
    """
    if not frames:
        raise ValueError("frames is empty")
    symbols = [str(s).strip().upper() for s in frames.keys()]
    # normalize frames
    frames_norm: Dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        if df is None or df.empty:
            continue
        frames_norm[str(sym).strip().upper()] = _to_utc_index(df)
    if not frames_norm:
        raise ValueError("No non-empty frames")

    o = _build_wide(frames_norm, "open")
    h = _build_wide(frames_norm, "high")
    l = _build_wide(frames_norm, "low")
    c = _build_wide(frames_norm, "close")
    v = _build_wide(frames_norm, "volume")
    if o.empty or h.empty or l.empty or c.empty:
        raise ValueError("Missing OHLC columns in frames")

    r = _infer_returns(c)
    adv20 = (
        _infer_adv20(v)
        if not v.empty
        else pd.DataFrame(index=c.index, columns=c.columns)
    )
    vwap = _infer_vwap(o, h, l, c, mode=cfg.vwap_mode)

    # Alpha library expects the column axis to be named "ticker" for stack("ticker")
    for df_wide in (o, h, l, c, v, r, adv20, vwap):
        if isinstance(df_wide, pd.DataFrame):
            df_wide.columns.name = "ticker"

    # Import alpha functions lazily; module uses local imports
    from cross_sectional.factors import alpha_functions as af  # type: ignore

    base_args = {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "v": v,
        "r": r,
        "adv20": adv20,
        "vwap": vwap,
    }

    panels: List[pd.DataFrame] = []
    for aid in alpha_ids:
        name = f"alpha{aid:03d}" if aid != 101 else "alpha101"
        func = getattr(af, name, None)
        if func is None or not callable(func):
            continue

        sig = inspect.signature(func)
        kwargs = {}
        ok = True
        for p in sig.parameters.values():
            k = p.name
            if k not in base_args:
                ok = False
                break
            kwargs[k] = base_args[k]
        if not ok:
            continue

        out = func(**kwargs)
        colname = f"alpha101_cs_{aid:03d}" if aid != 101 else "alpha101_cs_101"
        panel_col = _series_to_panel(out, colname=colname)
        panels.append(panel_col)

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, axis=1)
    panel.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(panel.index.get_level_values(0), utc=True, errors="coerce"),
            panel.index.get_level_values(1).astype(str),
        ],
        names=["timestamp", "symbol"],
    )
    panel = panel.sort_index()
    return panel
