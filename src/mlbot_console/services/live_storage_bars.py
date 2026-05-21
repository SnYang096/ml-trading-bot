"""1m OHLCV from live_storage daily parquets (``<root>/<SYMBOL>/YYYY-MM-DD.parquet``)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)
    out = df.copy()
    if "timestamp" not in out.columns:
        return pd.DataFrame(columns=_OHLCV_COLS)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            if col == "volume":
                continue
            if "close" in out.columns:
                out[col] = out["close"]
            else:
                out[col] = float("nan")
    keep = [c for c in _OHLCV_COLS if c in out.columns]
    return out[keep].sort_values("timestamp").reset_index(drop=True)


def load_live_storage_bars_1min(
    bars_root: Path,
    symbol: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load 1m bars from live_storage ``bars/<SYMBOL>/<date>.parquet`` files."""
    sym = symbol.upper()
    sym_dir = Path(bars_root) / sym
    if not sym_dir.is_dir():
        return pd.DataFrame(columns=_OHLCV_COLS)

    if end is not None:
        end_ts = _utc_ts(end)
    else:
        end_ts = pd.Timestamp.now(tz="UTC")
    if start is not None:
        start_ts = _utc_ts(start)
    else:
        start_ts = end_ts - pd.Timedelta(days=30)

    start_d = start_ts.floor("D")
    end_d = end_ts.floor("D")
    frames: list[pd.DataFrame] = []
    current = start_d
    while current <= end_d:
        path = sym_dir / f"{current.strftime('%Y-%m-%d')}.parquet"
        if path.is_file():
            try:
                df = pd.read_parquet(path)
                frames.append(_normalize_bars(df))
            except (OSError, ValueError):
                pass
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=_OHLCV_COLS)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    out = out[(out["timestamp"] >= start_ts) & (out["timestamp"] <= end_ts)]
    return out.reset_index(drop=True)
