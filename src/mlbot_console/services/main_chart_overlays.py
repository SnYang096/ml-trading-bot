"""Slow-scale MA lines on Trade Map main chart (CMS-local, no feature-bus columns)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from mlbot_console.services.feature_overlay import _align_points_to_candles, _utc_ts
from mlbot_console.services.macro_spot_daily import MacroSpotDailyLoader

EMA1200_SPAN_BARS = 1200
SOURCE_FEATURE_TF = "2h"
# Seed parquet older than chart end by this much is treated stale (flat 374 bug).
STALE_WEEKLY_SEED_LAG = pd.Timedelta(days=21)

MAIN_OVERLAY_KEYS = frozenset({"ema_1200", "weekly_ema_200"})

_OVERLAY_SPECS: Dict[str, Dict[str, Any]] = {
    "ema_1200": {
        "label": "EMA1200 (2h)",
        "color": "#ffb74d",
    },
    "weekly_ema_200": {
        "label": "周线 EMA200",
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


def _utc_datetime64ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


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


def _resample_candles_to_weekly_close(
    candles: List[Dict[str, Any]],
) -> pd.Series:
    """Build a weekly-anchored close series from arbitrary chart candles.

    Returns an empty series when there is no usable input. The resample anchor
    matches Binance's Mon 00:00 UTC weekly bar so the EMA aligns with macro
    seed weeks.
    """
    rows: List[tuple[pd.Timestamp, float]] = []
    for c in candles or []:
        t = c.get("time")
        close = c.get("close")
        if t is None or close is None:
            continue
        try:
            ts = _utc_ts(pd.Timestamp(int(t), unit="s", tz="UTC"))
            px = float(close)
        except (TypeError, ValueError):
            continue
        if px > 0:
            rows.append((ts, px))
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([r[0] for r in rows])
    series = pd.Series([r[1] for r in rows], index=idx, dtype=float).sort_index()
    weekly = series.resample("W-MON", label="left", closed="left").last().dropna()
    return weekly


def _weekly_ema_from_chart_candles(
    candles: List[Dict[str, Any]],
    *,
    span: int = 200,
    min_weeks: int = 52,
) -> List[Dict[str, Any]]:
    """Last-resort EMA(200) directly from on-chart closes (resampled to weekly).

    ``min_weeks`` guards against EWM warmup bias: a 1y minimum keeps the line
    meaningful when macro seed / macro_kline parquet are stale (e.g. macro
    refresher hasn't run for weeks). Returns ``[]`` if not enough weekly bars
    are available — typical for 2h/15m chart windows that only span months.
    """
    weekly = _resample_candles_to_weekly_close(candles)
    if weekly.empty or len(weekly) < int(min_weeks):
        return []
    ema = weekly.ewm(span=max(2, int(span)), adjust=False).mean().dropna()
    if ema.empty:
        return []
    native = _native_points_from_series(ema.index, ema.values)
    return _overlay_points_for_chart(native, candles)


def _ema1200_from_candle_closes(
    candles: List[Dict[str, Any]],
    *,
    span: int = EMA1200_SPAN_BARS,
) -> List[Dict[str, Any]]:
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


def _resample_candles_to_2h(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[tuple[pd.Timestamp, float]] = []
    for c in candles:
        t = c.get("time")
        close = c.get("close")
        if t is None or close is None:
            continue
        try:
            rows.append(
                (_utc_ts(pd.Timestamp(int(t), unit="s", tz="UTC")), float(close))
            )
        except (TypeError, ValueError):
            continue
    if not rows:
        return []
    idx = pd.DatetimeIndex([r[0] for r in rows]).sort_values()
    closes = pd.Series([r[1] for r in rows], index=idx, dtype=float)
    bars = closes.resample("2h", label="right", closed="right").last().dropna()
    return [
        {"time": int(ts.timestamp()), "close": float(v)}
        for ts, v in bars.items()
        if float(v) > 0
    ]


def _fetch_2h_candles(
    feature_bus_root: Path,
    symbol: str,
    candles: List[Dict[str, Any]],
    *,
    live_storage_bars_root: Optional[Path] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> List[Dict[str, Any]]:
    from mlbot_console.services.ohlcv_reader import fetch_ohlcv

    c_start, c_end = _candle_time_bounds(candles)
    win_start = c_start or start
    win_end = c_end or end
    if win_start is not None:
        win_start = win_start - pd.Timedelta(hours=EMA1200_SPAN_BARS * 2)
    pack = fetch_ohlcv(
        feature_bus_root,
        symbol,
        SOURCE_FEATURE_TF,
        start=win_start,
        end=win_end,
        full_range=False,
        live_storage_bars_root=live_storage_bars_root,
    )
    return list(pack.get("candles") or [])


def _ema1200_points_local(
    symbol: str,
    chart_candles: List[Dict[str, Any]],
    *,
    chart_timeframe: str,
    feature_bus_root: Any = None,
    live_storage_bars_root: Any = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> tuple[List[Dict[str, Any]], str]:
    """EMA(1200) on 2h closes, aligned to the visible chart."""
    tf = str(chart_timeframe or "2h").strip()
    src_candles: List[Dict[str, Any]]
    source = "candle_ewm_2h"
    if feature_bus_root and Path(feature_bus_root).is_dir():
        fetched = _fetch_2h_candles(
            Path(feature_bus_root),
            symbol,
            chart_candles,
            live_storage_bars_root=(
                Path(live_storage_bars_root) if live_storage_bars_root else None
            ),
            start=start,
            end=end,
        )
        if fetched:
            src_candles = fetched
            source = "bars_1min_2h"
        elif tf in ("2h", "120T"):
            src_candles = chart_candles
        else:
            src_candles = _resample_candles_to_2h(chart_candles)
            source = "chart_resample_2h"
    elif tf in ("2h", "120T"):
        src_candles = chart_candles
    else:
        src_candles = _resample_candles_to_2h(chart_candles)
        source = "chart_resample_2h"
    native = _ema1200_from_candle_closes(src_candles)
    return _overlay_points_for_chart(native, chart_candles), source


def _position_to_ma_price(close: pd.Series, position: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    p = pd.to_numeric(position, errors="coerce")
    return c * (1.0 - p)


def _align_ma_to_candles(
    feat: pd.DataFrame,
    pos_col: str,
    candles: List[Dict[str, Any]],
    *,
    use_candle_close: bool = False,
) -> List[Dict[str, Any]]:
    """Legacy helper (tests); live overlays do not invert feature-bus position."""
    if feat.empty or pos_col not in feat.columns or not candles:
        return []
    pos = pd.to_numeric(feat[pos_col], errors="coerce")
    if use_candle_close:
        tgt = pd.DataFrame(
            {
                "timestamp": _utc_datetime64ns(
                    pd.to_datetime(
                        [int(c["time"]) for c in candles], unit="s", utc=True
                    )
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
            points.append(
                {
                    "time": int(row["timestamp"].timestamp()),
                    "value": c_close * (1.0 - p),
                }
            )
        return points
    if "close" not in feat.columns:
        return []
    ma = _position_to_ma_price(feat["close"], pos)
    native = _native_points_from_series(feat["timestamp"], ma)
    return _overlay_points_for_chart(native, candles)


def _seed_ema_plausible(
    seed: pd.DataFrame,
    candles: List[Dict[str, Any]],
    *,
    ema_column: str = "weekly_ema_200",
    max_rel_gap: float = 0.35,
) -> bool:
    """Reject macro seed when EMA is far from spot (stale flat-line bug)."""
    if seed is None or seed.empty or ema_column not in seed.columns or not candles:
        return False
    ema = pd.to_numeric(seed[ema_column], errors="coerce").dropna()
    if ema.empty:
        return False
    closes = [
        float(c.get("close") or 0)
        for c in candles
        if c.get("close") is not None and float(c.get("close") or 0) > 0
    ]
    if not closes:
        return True
    ref_close = closes[-1]
    ref_ema = float(ema.iloc[-1])
    if ref_close <= 0 or ref_ema <= 0:
        return False
    return abs(ref_ema - ref_close) / ref_close <= float(max_rel_gap)


def _align_weekly_ema_seed_to_candles(
    macro_seed_root: Any,
    symbol: str,
    candles: List[Dict[str, Any]],
    *,
    ema_column: str = "weekly_ema_200",
) -> List[Dict[str, Any]]:
    try:
        from src.live_data_stream.spot_weekly_ema_seed import load_weekly_ema_seed
    except ImportError:
        return []
    seed = load_weekly_ema_seed(macro_seed_root, symbol)
    if seed is None or seed.empty or ema_column not in seed.columns:
        return []
    if _seed_is_stale_for_chart(seed, candles):
        return []
    if not _seed_ema_plausible(seed, candles, ema_column=ema_column):
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
    _, c_end = _candle_time_bounds(candles)
    if c_end is not None:
        keep = ts <= c_end
        if keep.any():
            ema = ema[keep]
            ts = ema.index
    native = _native_points_from_series(ts, ema)
    return _overlay_points_for_chart(native, candles)


def load_main_chart_overlays(
    symbol: str,
    candles: List[Dict[str, Any]],
    overlay_keys: List[str],
    *,
    chart_timeframe: str = "2h",
    macro_seed_root: Any = None,
    macro_spot_kline_root: Any = None,
    feature_bus_root: Any = None,
    live_storage_bars_root: Any = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build MA price series from CMS-local OHLC / macro (never feature-bus columns)."""
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

    for key in requested:
        spec = _OVERLAY_SPECS[key]
        entry = out[key]
        points: List[Dict[str, Any]] = []
        if key == "ema_1200":
            points, entry["source"] = _ema1200_points_local(
                symbol,
                candles,
                chart_timeframe=chart_timeframe,
                feature_bus_root=feature_bus_root,
                live_storage_bars_root=live_storage_bars_root,
                start=start,
                end=end,
            )
        elif spec.get("use_macro_seed"):
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
            if not points:
                points = _weekly_ema_from_chart_candles(candles)
                if points:
                    entry["source"] = "chart_resample_weekly"
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
