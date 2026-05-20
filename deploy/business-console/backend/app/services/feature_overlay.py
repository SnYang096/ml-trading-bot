"""Feature-bus overlays for Trade Map sub-charts (multi-column)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# UI timeframe -> candidate feature bus directories (first existing wins)
FEATURE_DIRS: Dict[str, List[str]] = {
    "2h": ["120T", "2h"],
    "120T": ["120T", "2h"],
    "15min": ["15min"],
    "1min": [],
    "1d": ["240T", "1d"],
}

# Not useful as standalone sub-chart series
SKIP_COLUMNS: Set[str] = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
}

# Default sub-chart when client sends no feature_columns
DEFAULT_SUBCHART_COLUMNS: List[str] = ["weekly_ema_200_position"]

# Horizontal reference lines (e.g. Spot prefilter threshold)
REFERENCE_Y_BY_COLUMN: Dict[str, float] = {
    "weekly_ema_200_position": 0.0,
}


def _resolve_feature_path(feature_bus_root: Path, symbol: str, timeframe: str) -> Optional[Path]:
    sym = symbol.upper()
    for sub in FEATURE_DIRS.get(str(timeframe).strip(), []):
        path = feature_bus_root / "features" / sub / f"{sym}.parquet"
        if path.is_file():
            return path
    return None


def _utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for name in df.columns:
        if name in SKIP_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[name]):
            cols.append(str(name))
    return sorted(cols)


def list_feature_columns(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    path = _resolve_feature_path(feature_bus_root, symbol, timeframe)
    if path is None:
        return {
            "available": False,
            "columns": [],
            "defaults": list(DEFAULT_SUBCHART_COLUMNS),
            "path": None,
            "timeframe_dir": None,
        }
    df = pd.read_parquet(path)
    if df.empty or "timestamp" not in df.columns:
        return {
            "available": False,
            "columns": [],
            "defaults": list(DEFAULT_SUBCHART_COLUMNS),
            "path": str(path),
            "timeframe_dir": path.parent.name,
        }
    columns = _numeric_columns(df)
    defaults = [c for c in DEFAULT_SUBCHART_COLUMNS if c in columns]
    if not defaults and columns:
        defaults = [columns[0]]
    return {
        "available": True,
        "columns": columns,
        "defaults": defaults,
        "path": str(path),
        "timeframe_dir": path.parent.name,
    }


def load_feature_overlay(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    column: str = "weekly_ema_200_position",
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    return load_feature_overlays(
        feature_bus_root,
        symbol,
        timeframe,
        [column],
        start=start,
        end=end,
    ).get(column) or {
        "available": False,
        "column": column,
        "points": [],
        "reference_y": REFERENCE_Y_BY_COLUMN.get(column),
    }


def load_feature_overlays(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    columns: List[str],
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Dict[str, Any]]:
    path = _resolve_feature_path(feature_bus_root, symbol, timeframe)
    out: Dict[str, Dict[str, Any]] = {}
    requested = [c.strip() for c in columns if c and c.strip()]
    if not requested:
        return out
    if path is None:
        for col in requested:
            out[col] = {
                "available": False,
                "column": col,
                "points": [],
                "reference_y": REFERENCE_Y_BY_COLUMN.get(col),
            }
        return out

    df = pd.read_parquet(path)
    if df.empty or "timestamp" not in df.columns:
        for col in requested:
            out[col] = {
                "available": False,
                "column": col,
                "points": [],
                "reference_y": REFERENCE_Y_BY_COLUMN.get(col),
                "path": str(path),
            }
        return out

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    if start is not None:
        df = df[df["timestamp"] >= _utc_ts(start)]
    if end is not None:
        df = df[df["timestamp"] <= _utc_ts(end)]

    for col in requested:
        ref_y = REFERENCE_Y_BY_COLUMN.get(col)
        if col not in df.columns:
            out[col] = {
                "available": False,
                "column": col,
                "points": [],
                "reference_y": ref_y,
                "path": str(path),
            }
            continue
        points: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            val = row.get(col)
            if val is None or (isinstance(val, float) and val != val):
                continue
            ts = _utc_ts(row["timestamp"])
            points.append({"time": int(ts.timestamp()), "value": float(val)})
        latest_val = points[-1]["value"] if points else None
        out[col] = {
            "available": True,
            "column": col,
            "points": points,
            "reference_y": ref_y,
            "latest": latest_val,
            "path": str(path),
        }
    return out
