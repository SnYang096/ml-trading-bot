"""Optional feature-bus overlays (e.g. weekly_ema_200_position) for Trade Map."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# UI timeframe -> candidate feature bus directories (first existing wins)
FEATURE_DIRS: Dict[str, List[str]] = {
    "2h": ["120T", "2h"],
    "120T": ["120T", "2h"],
    "15min": ["15min"],
    "1min": [],
    "1d": ["240T", "1d"],
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


def load_feature_overlay(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    column: str = "weekly_ema_200_position",
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    path = _resolve_feature_path(feature_bus_root, symbol, timeframe)
    if path is None:
        return {
            "available": False,
            "column": column,
            "points": [],
            "reference_y": 0.0,
        }
    df = pd.read_parquet(path)
    if df.empty or "timestamp" not in df.columns:
        return {"available": False, "column": column, "points": [], "reference_y": 0.0}
    if column not in df.columns:
        return {
            "available": False,
            "column": column,
            "points": [],
            "reference_y": 0.0,
            "path": str(path),
        }
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    if start is not None:
        df = df[df["timestamp"] >= _utc_ts(start)]
    if end is not None:
        df = df[df["timestamp"] <= _utc_ts(end)]
    points: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        val = row.get(column)
        if val is None or (isinstance(val, float) and val != val):
            continue
        ts = _utc_ts(row["timestamp"])
        points.append({"time": int(ts.timestamp()), "value": float(val)})
    latest_val = points[-1]["value"] if points else None
    return {
        "available": True,
        "column": column,
        "points": points,
        "reference_y": 0.0,
        "latest": latest_val,
        "path": str(path),
    }
