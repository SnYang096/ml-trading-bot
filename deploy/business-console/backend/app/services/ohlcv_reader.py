"""OHLCV from Feature Bus bars_1min parquet with timeframe resampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# UI key -> pandas resample rule (canonical source: bars_1min)
class OhlcvWindowError(ValueError):
    """Requested time range exceeds configured max window."""


TIMEFRAME_RULES: Dict[str, str] = {
    "1min": "1min",
    "15min": "15min",
    "2h": "2h",
    "120T": "2h",
    "1d": "1D",
}


def _utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _bars_path(feature_bus_root: Path, symbol: str) -> Path:
    return feature_bus_root / "bars_1min" / f"{symbol.upper()}.parquet"


def latest_bar_meta(feature_bus_root: Path, symbol: str) -> Optional[Dict[str, Any]]:
    meta = (
        feature_bus_root
        / "latest"
        / "bars_1min"
        / f"{symbol.upper()}.json"
    )
    if not meta.is_file():
        feat_meta = (
            feature_bus_root
            / "latest"
            / "features/120T"
            / f"{symbol.upper()}.json"
        )
        if not feat_meta.is_file():
            return None
        meta = feat_meta
    try:
        raw = json.loads(meta.read_text(encoding="utf-8"))
        ts = _utc_ts(raw.get("timestamp"))
        return {
            "symbol": symbol.upper(),
            "timestamp": ts.isoformat(),
            "kind": raw.get("kind"),
            "path": raw.get("path"),
            "rows": raw.get("rows"),
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def bars_1min_bounds(path: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
    """Min/max timestamp and row count without loading full OHLC columns."""
    if not path.is_file():
        return None, None, 0
    try:
        df = pd.read_parquet(path, columns=["timestamp"])
    except (OSError, ValueError, KeyError):
        return None, None, 0
    if df.empty or "timestamp" not in df.columns:
        return None, None, 0
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return _utc_ts(ts.min()), _utc_ts(ts.max()), int(len(df))


def load_bars_1min(
    feature_bus_root: Path,
    symbol: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    path = _bars_path(feature_bus_root, symbol)
    if not path.is_file():
        return pd.DataFrame()
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    try:
        df = pd.read_parquet(path, columns=cols)
    except (OSError, ValueError, KeyError):
        df = pd.read_parquet(path)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    if start is not None:
        df = df[df["timestamp"] >= _utc_ts(start)]
    if end is not None:
        df = df[df["timestamp"] <= _utc_ts(end)]
    return df.reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, bool]:
    """Return OHLCV dataframe indexed by timestamp; degraded=True if OHLC incomplete."""
    if df.empty:
        return df, False

    rule = TIMEFRAME_RULES.get(str(timeframe).strip(), "2h")
    if rule == "1min":
        out = df.copy()
    else:
        cols = {c for c in df.columns}
        agg: Dict[str, str] = {}
        if "open" in cols:
            agg["open"] = "first"
        if "high" in cols:
            agg["high"] = "max"
        if "low" in cols:
            agg["low"] = "min"
        if "close" in cols:
            agg["close"] = "last"
        if "volume" in cols:
            agg["volume"] = "sum"
        if not agg and "close" in cols:
            agg["close"] = "last"
        indexed = df.set_index("timestamp")
        out = indexed.resample(rule).agg(agg).dropna(how="all").reset_index()

    degraded = False
    for field in ("open", "high", "low"):
        if field not in out.columns or out[field].isna().all():
            if "close" in out.columns:
                out[field] = out["close"]
                degraded = True
            else:
                degraded = True
    if "close" not in out.columns:
        degraded = True
    return out, degraded


def ohlcv_to_candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        ts = _utc_ts(r["timestamp"])
        t_sec = int(ts.timestamp())
        try:
            close = float(r["close"])
        except (TypeError, ValueError):
            continue
        o = float(r.get("open", close) or close)
        h = float(r.get("high", close) or close)
        l = float(r.get("low", close) or close)
        vol = r.get("volume")
        candle: Dict[str, Any] = {
            "time": t_sec,
            "open": o,
            "high": h,
            "low": l,
            "close": close,
        }
        if vol is not None and vol == vol:
            candle["volume"] = float(vol)
        rows.append(candle)
    return rows


def _resolve_window(
    path: Path,
    *,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    max_days: int,
    full_range: bool,
) -> Tuple[pd.Timestamp, pd.Timestamp, bool, int]:
    """Return (start_ts, end_ts, clipped_to_max_days, bars_1min_rows_in_file)."""
    file_start, file_end, row_count = bars_1min_bounds(path)
    explicit = start is not None or end is not None
    if end is not None:
        end_ts = _utc_ts(end)
    elif file_end is not None:
        end_ts = file_end
    else:
        end_ts = pd.Timestamp.now(tz="UTC")

    clipped = False
    if start is not None:
        start_ts = _utc_ts(start)
    elif full_range or not explicit:
        start_ts = file_start if file_start is not None else end_ts - pd.Timedelta(days=max_days)
        span_days = (end_ts - start_ts).total_seconds() / 86400.0
        if span_days > float(max_days):
            start_ts = end_ts - pd.Timedelta(days=float(max_days))
            clipped = True
    else:
        start_ts = end_ts - pd.Timedelta(days=max_days)

    span_days = (end_ts - start_ts).total_seconds() / 86400.0
    if explicit and span_days > float(max_days) + 1e-6:
        raise OhlcvWindowError(
            f"range {span_days:.1f}d exceeds max_ohlcv_days={max_days}"
        )
    return start_ts, end_ts, clipped, row_count


def fetch_ohlcv(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    max_days: int = 180,
    full_range: bool = True,
) -> Dict[str, Any]:
    path = _bars_path(feature_bus_root, symbol)
    start_ts, end_ts, clipped, bars_1min_rows = _resolve_window(
        path,
        start=start,
        end=end,
        max_days=max_days,
        full_range=full_range and start is None and end is None,
    )
    raw = load_bars_1min(feature_bus_root, symbol, start=start_ts, end=end_ts)
    resampled, degraded = resample_ohlcv(raw, timeframe)
    mtime = path.stat().st_mtime if path.is_file() else None
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "candles": ohlcv_to_candles(resampled),
        "degraded_ohlc": degraded,
        "source": "bars_1min",
        "source_mtime": mtime,
        "row_count": len(resampled),
        "bars_1min_rows": bars_1min_rows,
        "range_start": start_ts.isoformat(),
        "range_end": end_ts.isoformat(),
        "range_clipped": clipped,
        "max_ohlcv_days": max_days,
    }
