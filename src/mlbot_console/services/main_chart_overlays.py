"""Slow-scale MA lines on Trade Map main chart (2h EMA1200, weekly EMA200)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from mlbot_console.services.feature_overlay import (
    COLUMN_ALIASES,
    _align_points_to_candles,
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
        "price_columns": ["ema_1200"],
        "position_columns": ["ema_1200_position", "ema_1200_position_f"],
        "color": "#ffb74d",
        "use_macro_seed": False,
    },
    "weekly_ema_200": {
        "label": "周线 EMA200",
        "position_columns": ["weekly_ema_200_position", "weekly_ema_200_position_f"],
        "color": "#64b5f6",
        "use_macro_seed": True,
        "seed_ema_column": "weekly_ema_200",
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


def _resolve_price_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
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
    price_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    path = _resolve_feature_path(feature_bus_root, symbol, SOURCE_FEATURE_TF)
    if path is None:
        return pd.DataFrame()
    want = ["timestamp", "close"]
    if price_columns:
        want.extend(price_columns)
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


def _candle_time_bounds(
    candles: List[Dict[str, Any]],
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if not candles:
        return None, None
    ts = pd.to_datetime([int(c["time"]) for c in candles], unit="s", utc=True)
    return _utc_ts(ts.min()), _utc_ts(ts.max())


def _native_points_from_series(
    timestamps: pd.Series,
    values: pd.Series,
) -> List[Dict[str, Any]]:
    """One point per source timestamp (full EMA history, not a single level)."""
    points: List[Dict[str, Any]] = []
    for t, v in zip(timestamps, values):
        val = pd.to_numeric(v, errors="coerce")
        if val is None or (isinstance(val, float) and val != val):
            continue
        ts = _utc_ts(t)
        points.append({"time": int(ts.timestamp()), "value": float(val)})
    points.sort(key=lambda p: p["time"])
    return points


def _overlay_points_for_chart(
    native_points: List[Dict[str, Any]],
    candles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not native_points:
        return []
    if not candles:
        return native_points
    return _align_points_to_candles(native_points, candles)


def _align_weekly_ema_seed_to_candles(
    macro_seed_root: Any,
    symbol: str,
    candles: List[Dict[str, Any]],
    *,
    ema_column: str = "weekly_ema_200",
) -> List[Dict[str, Any]]:
    """Plot weekly EMA200 price from Vision spot macro seed (authoritative)."""
    try:
        from src.live_data_stream.spot_weekly_ema_seed import load_weekly_ema_seed
    except ImportError:
        return []
    seed = load_weekly_ema_seed(macro_seed_root, symbol)
    if seed is None or seed.empty or ema_column not in seed.columns:
        return []
    ema = pd.to_numeric(seed[ema_column], errors="coerce").dropna()
    if ema.empty:
        return []
    if isinstance(ema.index, pd.DatetimeIndex):
        ts = ema.index
    elif "week_ts" in seed.columns:
        ts = pd.to_datetime(seed["week_ts"], utc=True, errors="coerce")
    else:
        return []
    c_start, c_end = _candle_time_bounds(candles)
    if c_start is not None and c_end is not None:
        mask = (ts >= c_start) & (ts <= c_end)
        if mask.any():
            ema = ema[mask]
            ts = ema.index
    native = _native_points_from_series(ts, ema)
    return _overlay_points_for_chart(native, candles)


def _align_ma_to_candles(
    feat: pd.DataFrame,
    pos_col: str,
    candles: List[Dict[str, Any]],
    *,
    use_candle_close: bool = False,
) -> List[Dict[str, Any]]:
    if feat.empty or pos_col not in feat.columns or not candles:
        return []
    pos = pd.to_numeric(feat[pos_col], errors="coerce")
    if use_candle_close:
        tgt = pd.DataFrame(
            {
                "timestamp": _utc_datetime64ns(
                    pd.to_datetime([int(c["time"]) for c in candles], unit="s", utc=True)
                ),
                "close": [float(c.get("close") or 0) for c in candles],
                "ord": range(len(candles)),
            }
        )
        src = pd.DataFrame(
            {
                "timestamp": _utc_datetime64ns(feat["timestamp"]),
                "position": pos,
            }
        ).dropna(subset=["position"])
        if src.empty:
            return []
        merged = pd.merge_asof(
            tgt.sort_values("timestamp"),
            src.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        ).sort_values("ord")
        points: List[Dict[str, Any]] = []
        for _, row in merged.iterrows():
            c_close = float(row.get("close") or 0)
            p = float(row.get("position") or 0)
            if c_close <= 0 or not (p == p):
                continue
            val = c_close * (1.0 - p)
            points.append(
                {
                    "time": int(row["timestamp"].timestamp()),
                    "value": val,
                }
            )
        return points
    if "close" not in feat.columns:
        return []
    ma = _position_to_ma_price(feat["close"], pos)
    native = _native_points_from_series(feat["timestamp"], ma)
    return _overlay_points_for_chart(native, candles)


def load_main_chart_overlays(
    feature_bus_root: Any,
    symbol: str,
    candles: List[Dict[str, Any]],
    overlay_keys: List[str],
    *,
    macro_seed_root: Any = None,
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

    candle_start, candle_end = _candle_time_bounds(candles)
    feat_start = start
    feat_end = end
    if candle_start is not None:
        feat_start = (
            min(_utc_ts(feat_start), candle_start)
            if feat_start is not None
            else candle_start
        )
    if candle_end is not None:
        feat_end = max(_utc_ts(feat_end), candle_end) if feat_end is not None else candle_end

    all_pos_cols: List[str] = []
    all_price_cols: List[str] = []
    for key in requested:
        all_pos_cols.extend(_OVERLAY_SPECS[key]["position_columns"])
        for alias in _OVERLAY_SPECS[key]["position_columns"]:
            all_pos_cols.extend(COLUMN_ALIASES.get(alias, []))
        all_price_cols.extend(_OVERLAY_SPECS[key].get("price_columns") or [])
    feat = _load_source_features(
        feature_bus_root,
        symbol,
        start=feat_start,
        end=feat_end,
        position_columns=all_pos_cols,
        price_columns=all_price_cols,
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
        points: List[Dict[str, Any]] = []
        if spec.get("use_macro_seed") and macro_seed_root:
            points = _align_weekly_ema_seed_to_candles(
                macro_seed_root,
                symbol,
                candles,
                ema_column=str(spec.get("seed_ema_column") or "weekly_ema_200"),
            )
            if points:
                entry["source"] = "macro_seed"
        if not points and not feat.empty:
            price_col = _resolve_price_column(
                feat, list(spec.get("price_columns") or [])
            )
            if price_col:
                native = _native_points_from_series(feat["timestamp"], feat[price_col])
                points = _overlay_points_for_chart(native, candles)
                if points:
                    entry["source"] = "feature_bus_price"
                    entry["parquet_column"] = price_col
        if not points:
            if feat.empty or pos_col is None:
                continue
            # Weekly position uses spot seed EMA vs chart close; EMA1200 position
            # is defined on the same 2h bus close as the overlay source.
            use_candle_close = bool(spec.get("use_macro_seed"))
            points = _align_ma_to_candles(
                feat,
                pos_col,
                candles,
                use_candle_close=use_candle_close,
            )
            entry["parquet_column"] = pos_col
            entry["source"] = "position_inverted"
        entry["points"] = points
        entry["available"] = bool(points)
        if points:
            entry["latest"] = points[-1]["value"]
    return out
