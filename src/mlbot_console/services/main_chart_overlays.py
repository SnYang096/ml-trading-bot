"""Slow-scale MA lines on Trade Map main chart (2h EMA1200, weekly EMA200)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from mlbot_console.services.macro_spot_daily import MacroSpotDailyLoader

from mlbot_console.services.feature_overlay import (
    COLUMN_ALIASES,
    _align_points_to_candles,
    _resolve_feature_path,
    _resolve_parquet_column,
    _utc_ts,
)

# Always sourced from 2h feature bus (120T), overlaid on any chart timeframe.
SOURCE_FEATURE_TF = "2h"
EMA1200_SPAN_BARS = 1200
# Seed parquet older than chart end by this much is treated stale (flat 374 bug).
STALE_WEEKLY_SEED_LAG = pd.Timedelta(days=21)

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


def _seed_last_timestamp(seed: pd.DataFrame) -> Optional[pd.Timestamp]:
    if seed is None or seed.empty:
        return None
    if isinstance(seed.index, pd.DatetimeIndex):
        return _utc_ts(seed.index.max())
    if "week_ts" in seed.columns:
        return _utc_ts(pd.to_datetime(seed["week_ts"], utc=True, errors="coerce").max())
    return None


def _seed_is_stale_for_chart(seed: pd.DataFrame, candles: List[Dict[str, Any]]) -> bool:
    """Stale seed (e.g. 2025-01) must not ffilled across 2026 candles as a flat line."""
    last = _seed_last_timestamp(seed)
    _, c_end = _candle_time_bounds(candles)
    if last is None or c_end is None:
        return False
    return last < c_end - STALE_WEEKLY_SEED_LAG


def _weekly_ema_from_spot_daily(
    macro_kline_root: Any,
    symbol: str,
    candles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Recompute weekly EMA200 from Vision spot 1d ZIPs (curve, ~565 on BNB)."""
    from src.live_data_stream.spot_weekly_ema_seed import compute_weekly_ema_table

    c_start, c_end = _candle_time_bounds(candles)
    if c_end is None:
        return []
    load_start = (c_start or c_end) - pd.Timedelta(days=200 * 7 + 60)
    loader = MacroSpotDailyLoader(Path(macro_kline_root))
    daily = loader.load_symbol_daily(
        symbol,
        start_date=load_start.date(),
        end_date=c_end.date(),
    )
    if daily.empty or "close" not in daily.columns:
        return []
    weekly = compute_weekly_ema_table(daily["close"], ema_span_weeks=200)
    if weekly.empty or "weekly_ema_200" not in weekly.columns:
        return []
    ts = pd.to_datetime(weekly["week_ts"], utc=True, errors="coerce")
    ema = pd.to_numeric(weekly["weekly_ema_200"], errors="coerce")
    native = _native_points_from_series(ts, ema)
    return _overlay_points_for_chart(native, candles)


def _ema1200_from_candle_closes(
    candles: List[Dict[str, Any]],
    *,
    span: int = EMA1200_SPAN_BARS,
) -> List[Dict[str, Any]]:
    """EMA(span) on chart OHLC closes — full-width curve when feature bus is short."""
    rows: List[tuple[int, float]] = []
    for c in candles:
        close = c.get("close")
        t = c.get("time")
        if t is None or close is None:
            continue
        try:
            px = float(close)
            ti = int(t)
        except (TypeError, ValueError):
            continue
        if px > 0 and ti > 0:
            rows.append((ti, px))
    if not rows:
        return []
    rows.sort(key=lambda x: x[0])
    idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
    closes = pd.Series([r[1] for r in rows], index=idx, dtype=float)
    ema = closes.ewm(span=max(2, int(span)), adjust=False).mean()
    return _native_points_from_series(ema.index, ema.values)


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
    if _seed_is_stale_for_chart(seed, candles):
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
    macro_spot_kline_root: Any = None,
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
        if spec.get("use_macro_seed"):
            if macro_seed_root:
                points = _align_weekly_ema_seed_to_candles(
                    macro_seed_root,
                    symbol,
                    candles,
                    ema_column=str(spec.get("seed_ema_column") or "weekly_ema_200"),
                )
                if points:
                    entry["source"] = "macro_seed"
            if not points and macro_spot_kline_root:
                points = _weekly_ema_from_spot_daily(
                    macro_spot_kline_root, symbol, candles
                )
                if points:
                    entry["source"] = "spot_daily_weekly"
        elif key == "ema_1200":
            price_col = _resolve_price_column(
                feat, list(spec.get("price_columns") or [])
            )
            if price_col and not feat.empty:
                native = _native_points_from_series(
                    feat["timestamp"], feat[price_col]
                )
                points = _overlay_points_for_chart(native, candles)
                entry["source"] = "feature_bus_price"
                entry["parquet_column"] = price_col
            if not points:
                points = _ema1200_from_candle_closes(candles)
                entry["source"] = "candle_ewm"
            if not points and not feat.empty and pos_col:
                points = _align_ma_to_candles(
                    feat, pos_col, candles, use_candle_close=False
                )
                entry["parquet_column"] = pos_col
                entry["source"] = "position_inverted"
        if not points and spec.get("use_macro_seed"):
            if feat.empty or pos_col is None:
                continue
            points = _align_ma_to_candles(
                feat, pos_col, candles, use_candle_close=True
            )
            entry["parquet_column"] = pos_col
            entry["source"] = "position_inverted"
        entry["points"] = points
        entry["available"] = bool(points)
        if points:
            entry["latest"] = points[-1]["value"]
            vals = [p["value"] for p in points]
            entry["point_count"] = len(points)
            entry["value_range"] = {
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
            entry["coverage_from"] = int(points[0]["time"])
    return out
