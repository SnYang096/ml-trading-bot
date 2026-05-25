"""Feature-bus overlays for Trade Map sub-charts (multi-column)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# UI timeframe -> candidate feature bus directories (first existing wins)
FEATURE_DIRS: Dict[str, List[str]] = {
    "2h": ["120T", "2h", "primary"],
    "120T": ["120T", "2h", "primary"],
    "15min": ["15min", "120T", "primary"],
    "1min": ["15min", "120T", "primary"],
    "1d": ["240T", "1d", "120T", "primary"],
    "1w": ["240T", "1d", "120T", "primary"],
}

# UI / policy name -> parquet column candidates (first hit wins)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "weekly_ema_200_position": [
        "weekly_ema_200_position",
        "weekly_ema_200_position_f",
    ],
    "ema_1200_position": [
        "ema_1200_position",
        "ema_1200_position_f",
    ],
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

# Default sub-chart when client sends no feature_columns (strategy uses position, not raw EMA price)
DEFAULT_SUBCHART_COLUMNS: List[str] = [
    "weekly_ema_200_position",
    "ema_1200_position",
]

# Horizontal reference lines (e.g. Spot prefilter threshold)
REFERENCE_Y_BY_COLUMN: Dict[str, float] = {
    "weekly_ema_200_position": 0.0,
    "ema_1200_position": 0.0,
}


def _utc_datetime64ns(series: pd.Series) -> pd.Series:
    """Normalize timestamps for merge_asof (parquet ns vs candle unit=s)."""
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


def _align_points_to_candles(
    points: List[Dict[str, Any]],
    candles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """merge_asof feature values onto OHLCV bar times (no future leak).

      Uses backward as-of per bar, then forward-fills so candles after the last
      feature row still show the latest known value (OHLC often extends past a
    lagging feature bus).
    """
    if not points or not candles:
        return list(points or [])
    src = pd.DataFrame(
        {
            "timestamp": _utc_datetime64ns(
                pd.to_datetime([int(p["time"]) for p in points], unit="s", utc=True)
            ),
            "value": [p["value"] for p in points],
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
    merged["value"] = (
        pd.to_numeric(merged["value"], errors="coerce").ffill().bfill()
    )
    out: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        val = row.get("value")
        if val is None or (isinstance(val, float) and val != val):
            continue
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        out.append(
            {
                "time": int(ts.timestamp()),
                "value": float(val),
            }
        )
    return out


def _reference_lines_for_column(column: str) -> List[Dict[str, Any]]:
    from mlbot_console.config import SETTINGS
    from mlbot_console.services.feature_thresholds import (
        build_reference_lines_by_column,
    )

    ref_map = build_reference_lines_by_column(SETTINGS.strategies_root)
    lines = ref_map.get(column) or []
    if not lines and column in REFERENCE_Y_BY_COLUMN:
        y = REFERENCE_Y_BY_COLUMN[column]
        return [{"y": y, "label": f"{column}={y:g}", "operator": ""}]
    return [dict(x) for x in lines]


def _semantic_hint(column: str, latest: Optional[float]) -> str:
    from mlbot_console.services.feature_thresholds import semantic_hint_for_column

    return semantic_hint_for_column(column, latest)


def _resolve_feature_path(
    feature_bus_root: Path, symbol: str, timeframe: str
) -> Optional[Path]:
    sym = symbol.upper()
    for sub in FEATURE_DIRS.get(str(timeframe).strip(), []):
        path = feature_bus_root / "features" / sub / f"{sym}.parquet"
        if path.is_file():
            return path
    feat_root = feature_bus_root / "features"
    if feat_root.is_dir():
        for child in sorted(feat_root.iterdir()):
            if not child.is_dir():
                continue
            path = child / f"{sym}.parquet"
            if path.is_file():
                return path
    return None


def _resolve_parquet_column(df: pd.DataFrame, column: str) -> Optional[str]:
    if column in df.columns:
        return column
    for alt in COLUMN_ALIASES.get(column, []):
        if alt in df.columns:
            return alt
    suffixed = f"{column}_f"
    if suffixed in df.columns:
        return suffixed
    return None


def _default_columns_for_parquet(columns: List[str]) -> List[str]:
    colset = set(columns)
    defaults: List[str] = []
    for want in DEFAULT_SUBCHART_COLUMNS:
        if want in colset:
            defaults.append(want)
            continue
        for alt in COLUMN_ALIASES.get(want, [f"{want}_f"]):
            if alt in colset:
                defaults.append(alt)
                break
    if not defaults and columns:
        defaults = [columns[0]]
    return defaults


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
    *,
    include_taxonomy: bool = True,
) -> Dict[str, Any]:
    path = _resolve_feature_path(feature_bus_root, symbol, timeframe)
    if path is None:
        out: Dict[str, Any] = {
            "available": False,
            "columns": [],
            "defaults": list(DEFAULT_SUBCHART_COLUMNS),
            "path": None,
            "timeframe_dir": None,
        }
        if include_taxonomy:
            from mlbot_console.services.feature_taxonomy import (
                enrich_columns_with_taxonomy,
            )

            out.update(enrich_columns_with_taxonomy([]))
        return out
    df = pd.read_parquet(path)
    if df.empty or "timestamp" not in df.columns:
        out = {
            "available": False,
            "columns": [],
            "defaults": list(DEFAULT_SUBCHART_COLUMNS),
            "path": str(path),
            "timeframe_dir": path.parent.name,
        }
        if include_taxonomy:
            from mlbot_console.services.feature_taxonomy import (
                enrich_columns_with_taxonomy,
            )

            out.update(enrich_columns_with_taxonomy([]))
        return out
    columns = _numeric_columns(df)
    defaults = _default_columns_for_parquet(columns)
    out = {
        "available": True,
        "columns": columns,
        "defaults": defaults,
        "path": str(path),
        "timeframe_dir": path.parent.name,
    }
    if include_taxonomy:
        from mlbot_console.services.feature_taxonomy import enrich_columns_with_taxonomy

        out.update(enrich_columns_with_taxonomy(columns))
    return out


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
    candles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    path = _resolve_feature_path(feature_bus_root, symbol, timeframe)
    out: Dict[str, Dict[str, Any]] = {}
    requested = [c.strip() for c in columns if c and c.strip()]
    if not requested:
        return out
    if path is None:
        for col in requested:
            ref_lines = _reference_lines_for_column(col)
            out[col] = {
                "available": False,
                "column": col,
                "points": [],
                "reference_y": REFERENCE_Y_BY_COLUMN.get(col),
                "reference_lines": ref_lines,
            }
        return out

    df = pd.read_parquet(path)
    if df.empty or "timestamp" not in df.columns:
        for col in requested:
            ref_lines = _reference_lines_for_column(col)
            out[col] = {
                "available": False,
                "column": col,
                "points": [],
                "reference_y": REFERENCE_Y_BY_COLUMN.get(col),
                "reference_lines": ref_lines,
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
        ref_lines = _reference_lines_for_column(col)
        parquet_col = _resolve_parquet_column(df, col)
        if parquet_col is None:
            out[col] = {
                "available": False,
                "column": col,
                "points": [],
                "point_count": 0,
                "reference_y": ref_y,
                "reference_lines": ref_lines,
                "path": str(path),
            }
            continue
        points: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            val = row.get(parquet_col)
            if val is None or (isinstance(val, float) and val != val):
                continue
            ts = _utc_ts(row["timestamp"])
            points.append({"time": int(ts.timestamp()), "value": float(val)})
        aligned = bool(candles)
        if aligned:
            points = _align_points_to_candles(points, candles)
        latest_val = points[-1]["value"] if points else None
        feat_end = None
        if not df.empty:
            feat_end = _utc_ts(df["timestamp"].max()).isoformat()
        out[col] = {
            "available": bool(points),
            "column": col,
            "parquet_column": parquet_col,
            "points": points,
            "point_count": len(points),
            "reference_y": ref_y,
            "reference_lines": ref_lines,
            "latest": latest_val,
            "semantic_hint": _semantic_hint(col, latest_val),
            "aligned": aligned,
            "path": str(path),
            "timeframe_dir": path.parent.name,
            "feature_range_end": feat_end,
        }
    return out
