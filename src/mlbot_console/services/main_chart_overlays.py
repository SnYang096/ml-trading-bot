"""Slow-scale MA lines on Trade Map main chart (2h EMA1200, weekly EMA200)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from mlbot_console.services.feature_overlay import (
    COLUMN_ALIASES,
    _resolve_feature_path,
    _resolve_parquet_column,
    _utc_ts,
)

# Always sourced from 2h feature bus (120T), overlaid on any chart timeframe.
SOURCE_FEATURE_TF = "2h"

MAIN_OVERLAY_KEYS = frozenset({"ema_1200", "weekly_ema_200"})

_OVERLAY_SPECS: Dict[str, Dict[str, Any]] = {
    "ema_1200": {
        "label": "EMA1200 (2h)",
        "position_columns": ["ema_1200_position", "ema_1200_position_f"],
        "color": "#ffb74d",
    },
    "weekly_ema_200": {
        "label": "周线 EMA200",
        "position_columns": ["weekly_ema_200_position", "weekly_ema_200_position_f"],
        "color": "#64b5f6",
    },
}


def parse_main_overlay_keys(raw: Optional[str]) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    out: List[str] = []
    for part in str(raw).split(","):
        key = part.strip().lower()
        if key in MAIN_OVERLAY_KEYS and key not in out:
            out.append(key)
    return out


def _position_to_ma_price(close: pd.Series, position: pd.Series) -> pd.Series:
    """position = (close - ma) / close  =>  ma = close * (1 - position)."""
    c = pd.to_numeric(close, errors="coerce")
    p = pd.to_numeric(position, errors="coerce")
    return c * (1.0 - p)


def _utc_datetime64ns(series: pd.Series) -> pd.Series:
    """Normalize timestamps for merge_asof (parquet ns vs candle unit=s)."""
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


def _resolve_position_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        col = _resolve_parquet_column(df, name)
        if col is not None:
            return col
    return None


def _parquet_columns_to_read(path: Any, want: List[str]) -> List[str]:
    try:
        import pyarrow.parquet as pq

        names = set(pq.read_schema(path).names)
    except (ImportError, OSError, ValueError):
        return want
    cols = ["timestamp"]
    for w in want:
        if w in names:
            cols.append(w)
    return list(dict.fromkeys(cols))


def _load_source_features(
    feature_bus_root: Any,
    symbol: str,
    *,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    position_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    path = _resolve_feature_path(feature_bus_root, symbol, SOURCE_FEATURE_TF)
    if path is None:
        return pd.DataFrame()
    want = ["timestamp", "close"]
    if position_columns:
        want.extend(position_columns)
    read_cols = _parquet_columns_to_read(path, want)
    try:
        df = pd.read_parquet(path, columns=read_cols)
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


def _align_ma_to_candles(
    feat: pd.DataFrame,
    pos_col: str,
    candles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if feat.empty or "close" not in feat.columns or not candles:
        return []
    ma = _position_to_ma_price(feat["close"], feat[pos_col])
    src = pd.DataFrame(
        {
            "timestamp": _utc_datetime64ns(feat["timestamp"]),
            "value": ma,
        }
    ).dropna(subset=["value"])
    if src.empty:
        return []
    tgt = pd.DataFrame(
        {
            "timestamp": _utc_datetime64ns(
                pd.to_datetime([int(c["time"]) for c in candles], unit="s", utc=True)
            ),
            "ord": range(len(candles)),
        }
    )
    merged = pd.merge_asof(
        tgt.sort_values("timestamp"),
        src.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).sort_values("ord")
    points: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        val = row.get("value")
        if val is None or (isinstance(val, float) and val != val):
            continue
        points.append(
            {
                "time": int(row["timestamp"].timestamp()),
                "value": float(val),
            }
        )
    return points


def load_main_chart_overlays(
    feature_bus_root: Any,
    symbol: str,
    candles: List[Dict[str, Any]],
    overlay_keys: List[str],
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build price-level MA series aligned to chart candles."""
    requested = [k for k in overlay_keys if k in MAIN_OVERLAY_KEYS]
    out: Dict[str, Dict[str, Any]] = {}
    for key in requested:
        spec = _OVERLAY_SPECS[key]
        out[key] = {
            "available": False,
            "key": key,
            "label": spec["label"],
            "color": spec["color"],
            "source_timeframe": SOURCE_FEATURE_TF,
            "points": [],
        }
    if not requested or not candles:
        return out

    all_pos_cols: List[str] = []
    for key in requested:
        all_pos_cols.extend(_OVERLAY_SPECS[key]["position_columns"])
        for alias in _OVERLAY_SPECS[key]["position_columns"]:
            all_pos_cols.extend(COLUMN_ALIASES.get(alias, []))
    feat = _load_source_features(
        feature_bus_root,
        symbol,
        start=start,
        end=end,
        position_columns=all_pos_cols,
    )
    path = _resolve_feature_path(feature_bus_root, symbol, SOURCE_FEATURE_TF)
    for key in requested:
        spec = _OVERLAY_SPECS[key]
        candidates: List[str] = []
        for name in spec["position_columns"]:
            candidates.append(name)
            candidates.extend(COLUMN_ALIASES.get(name, []))
        pos_col = _resolve_position_column(feat, candidates)
        entry = out[key]
        entry["path"] = str(path) if path else None
        if feat.empty or pos_col is None:
            continue
        points = _align_ma_to_candles(feat, pos_col, candles)
        entry["points"] = points
        entry["available"] = bool(points)
        entry["parquet_column"] = pos_col
        if points:
            entry["latest"] = points[-1]["value"]
    return out
