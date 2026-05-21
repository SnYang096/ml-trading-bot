"""OHLCV from Feature Bus bars_1min parquet with timeframe resampling."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mlbot_console.services.live_storage_bars import load_live_storage_bars_1min
from mlbot_console.services.macro_spot_daily import MacroSpotDailyLoader

# UI key -> pandas resample rule (canonical source: bars_1min)
class OhlcvWindowError(ValueError):
    """Requested time range exceeds configured max window."""


# Trade Map UI timeframes (no 1min — too heavy on bars_1min resample).
TRADE_MAP_TIMEFRAMES = frozenset({"2h", "120T", "15min", "1d", "1w"})

# Default visible window when client does not send from/to (keep payloads small).
TRADE_MAP_INITIAL_DAYS: Dict[str, int] = {
    "15min": 14,
    "2h": 60,
    "120T": 60,
    "1d": 120,
    "1w": 365,
}

# Extra history loaded when user pans chart toward the past.
TRADE_MAP_HISTORY_CHUNK_DAYS: Dict[str, int] = {
    "15min": 7,
    "2h": 30,
    "120T": 30,
    "1d": 90,
    "1w": 180,
}


def assert_trade_map_timeframe(timeframe: str) -> str:
    tf = str(timeframe or "").strip()
    if tf not in TRADE_MAP_TIMEFRAMES:
        raise OhlcvWindowError(
            f"unsupported timeframe {timeframe!r}; "
            f"Trade Map supports {', '.join(sorted(TRADE_MAP_TIMEFRAMES))}"
        )
    return tf


def resolve_trade_map_window(
    timeframe: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    full_range: bool = False,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool]:
    """Return (start, end, full_range_for_fetch). Windowed load when start omitted."""
    assert_trade_map_timeframe(timeframe)
    if start is not None:
        end_ts = _utc_ts(end) if end is not None else pd.Timestamp.now(tz="UTC")
        return _utc_ts(start), end_ts, False
    if full_range:
        # end=None → fetch_ohlcv uses parquet file_end (not wall-clock now).
        return None, _utc_ts(end) if end is not None else None, True
    tf = str(timeframe).strip()
    end_ts = _utc_ts(end) if end is not None else pd.Timestamp.now(tz="UTC")
    days = TRADE_MAP_INITIAL_DAYS.get(tf, TRADE_MAP_INITIAL_DAYS["2h"])
    return end_ts - pd.Timedelta(days=float(days)), end_ts, False


def resolve_macro_kline_root(
    primary: Path,
    *,
    live_data_root: Optional[Path] = None,
    live_root: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """First existing macro spot_klines directory (Vision 1d ZIP cache)."""
    candidates: List[Path] = [Path(primary)]
    if live_data_root is not None:
        candidates.append(Path(live_data_root) / "macro" / "spot_klines")
    if live_root is not None:
        candidates.append(Path(live_root) / "data" / "macro" / "spot_klines")
        candidates.append(Path(live_root) / "macro" / "spot_klines")
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_dir():
            return cand, True
    return Path(primary), False


def cap_window_to_max_days(
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    max_days: int,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Clip start so explicit windows do not exceed max_ohlcv_days."""
    if start is None or end is None:
        return start, end
    end_ts = _utc_ts(end)
    start_ts = _utc_ts(start)
    min_start = end_ts - pd.Timedelta(days=float(max_days))
    if start_ts < min_start:
        return min_start, end_ts
    return start_ts, end_ts


TIMEFRAME_RULES: Dict[str, str] = {
    "1min": "1min",
    "15min": "15min",
    "2h": "2h",
    "120T": "2h",
    "1d": "1D",
    "1w": "1W",
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


def _daily_index_to_ohlcv(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    out = daily.copy()
    if "timestamp" in out.columns:
        out = out.set_index("timestamp")
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def _merge_recent_daily_from_bus(
    macro_df: pd.DataFrame,
    feature_bus_root: Path,
    symbol: str,
    *,
    end_ts: pd.Timestamp,
    merge_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Append/overwrite tail daily bars from bars_1min when Vision cache lags."""
    bus_start = merge_start if merge_start is not None else end_ts - pd.Timedelta(days=120)
    bus = load_bars_1min(feature_bus_root, symbol, start=bus_start, end=end_ts)
    if bus.empty:
        return macro_df
    bus_daily, _ = resample_ohlcv(bus, "1d")
    if bus_daily.empty:
        return macro_df
    bus_daily = bus_daily.set_index("timestamp").sort_index()
    if macro_df.empty:
        return _daily_index_to_ohlcv(bus_daily)
    macro_idx = macro_df.set_index("timestamp").sort_index()
    combined = pd.concat([macro_idx, bus_daily])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return _daily_index_to_ohlcv(combined.reset_index())


def _resolve_daily_window(
    *,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    daily_start: date,
    max_daily_days: int,
    full_range: bool,
) -> Tuple[pd.Timestamp, pd.Timestamp, bool]:
    if end is not None:
        end_ts = _utc_ts(end)
    else:
        end_ts = pd.Timestamp.now(tz="UTC")
    clipped = False
    if start is not None:
        start_ts = _utc_ts(start)
    elif full_range:
        start_ts = pd.Timestamp(daily_start, tz="UTC")
        span = (end_ts - start_ts).total_seconds() / 86400.0
        if span > float(max_daily_days):
            start_ts = end_ts - pd.Timedelta(days=float(max_daily_days))
            clipped = True
    else:
        start_ts = end_ts - pd.Timedelta(days=float(max_daily_days))
    span = (end_ts - start_ts).total_seconds() / 86400.0
    if start is not None and end is not None and span > float(max_daily_days) + 1e-6:
        raise OhlcvWindowError(
            f"range {span:.1f}d exceeds max_daily_ohlcv_days={max_daily_days}"
        )
    return start_ts, end_ts, clipped


def fetch_ohlcv_daily_macro(
    feature_bus_root: Path,
    macro_kline_root: Path,
    symbol: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    daily_start: date = date(2017, 1, 1),
    max_daily_days: int = 3650,
    full_range: bool = True,
    live_data_root: Optional[Path] = None,
    live_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """1d OHLCV from Vision spot_klines (years), tail merged from bars_1min."""
    path = _bars_path(feature_bus_root, symbol)
    _, _, bars_1min_rows = bars_1min_bounds(path)
    start_ts, end_ts, clipped = _resolve_daily_window(
        start=start,
        end=end,
        daily_start=daily_start,
        max_daily_days=max_daily_days,
        full_range=full_range and start is None and end is None,
    )
    macro_root, macro_ok = resolve_macro_kline_root(
        macro_kline_root,
        live_data_root=live_data_root,
        live_root=live_root,
    )
    macro_rows = 0
    macro_df = pd.DataFrame()
    if macro_ok:
        loader = MacroSpotDailyLoader(macro_root)
        daily = loader.load_symbol_daily(
            symbol,
            start_date=start_ts.date(),
            end_date=end_ts.date(),
        )
        macro_df = _daily_index_to_ohlcv(daily)
        macro_rows = len(macro_df)
    merged = _merge_recent_daily_from_bus(
        macro_df, feature_bus_root, symbol, end_ts=end_ts, merge_start=start_ts
    )
    if merged.empty:
        fallback = fetch_ohlcv(
            feature_bus_root,
            symbol,
            "1d",
            start=start_ts,
            end=end_ts,
            max_days=max(180, int(max_daily_days)),
            full_range=False,
        )
        fallback["macro_kline_root"] = str(macro_root)
        fallback["macro_available"] = macro_ok
        fallback["macro_rows"] = macro_rows
        return fallback
    merged = merged[
        (merged["timestamp"] >= start_ts) & (merged["timestamp"] <= end_ts)
    ]
    payload = _macro_daily_payload(
        symbol=symbol,
        timeframe="1d",
        merged=merged,
        start_ts=start_ts,
        end_ts=end_ts,
        clipped=clipped,
        bars_1min_rows=bars_1min_rows,
        max_daily_days=max_daily_days,
        daily_start=daily_start,
    )
    payload["macro_kline_root"] = str(macro_root)
    payload["macro_available"] = macro_ok
    payload["macro_rows"] = macro_rows
    return payload


def _macro_daily_payload(
    *,
    symbol: str,
    timeframe: str,
    merged: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    clipped: bool,
    bars_1min_rows: int,
    max_daily_days: int,
    daily_start: date,
) -> Dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "candles": ohlcv_to_candles(merged),
        "degraded_ohlc": False,
        "source": "macro_spot_klines",
        "source_mtime": None,
        "row_count": len(merged),
        "bars_1min_rows": bars_1min_rows,
        "range_start": start_ts.isoformat(),
        "range_end": end_ts.isoformat(),
        "range_clipped": clipped,
        "max_ohlcv_days": max_daily_days,
        "daily_ohlcv_start": daily_start.isoformat(),
    }


def fetch_ohlcv_weekly_macro(
    feature_bus_root: Path,
    macro_kline_root: Path,
    symbol: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    daily_start: date = date(2017, 1, 1),
    max_daily_days: int = 3650,
    full_range: bool = True,
    live_data_root: Optional[Path] = None,
    live_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """1w OHLCV: Vision daily macro resampled to calendar weeks."""
    daily = fetch_ohlcv_daily_macro(
        feature_bus_root,
        macro_kline_root,
        symbol,
        start=start,
        end=end,
        daily_start=daily_start,
        max_daily_days=max_daily_days,
        full_range=full_range,
        live_data_root=live_data_root,
        live_root=live_root,
    )
    if not daily.get("candles"):
        out = dict(daily)
        out["timeframe"] = "1w"
        return out
    df = pd.DataFrame(daily["candles"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    indexed = df.set_index("timestamp")
    agg: Dict[str, str] = {}
    for col, how in (
        ("open", "first"),
        ("high", "max"),
        ("low", "min"),
        ("close", "last"),
        ("volume", "sum"),
    ):
        if col in indexed.columns:
            agg[col] = how
    weekly = indexed.resample("1W").agg(agg).dropna(how="all").reset_index()
    out = dict(daily)
    out["timeframe"] = "1w"
    out["candles"] = ohlcv_to_candles(weekly)
    out["row_count"] = len(weekly)
    return out


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


def stitch_live_storage_and_bus(
    history: pd.DataFrame,
    bus: pd.DataFrame,
) -> pd.DataFrame:
    """Merge archive 1m with bus tail; bus wins on duplicate timestamps."""
    if history.empty:
        return bus.reset_index(drop=True) if not bus.empty else history
    if bus.empty:
        return history.reset_index(drop=True)
    out = pd.concat([history, bus], ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = (
        out.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out


def _resolve_window(
    path: Path,
    *,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    max_days: int,
    full_range: bool,
    calendar_span: bool = False,
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
        if calendar_span:
            start_ts = end_ts - pd.Timedelta(days=float(max_days))
        else:
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
    live_storage_bars_root: Optional[Path] = None,
    stitch_live_storage: bool = True,
    macro_kline_root: Optional[Path] = None,
    daily_ohlcv_start: Optional[date] = None,
    max_daily_ohlcv_days: int = 3650,
    live_data_root: Optional[Path] = None,
    live_root: Optional[Path] = None,
) -> Dict[str, Any]:
    tf = str(timeframe).strip()
    if tf in ("1d", "1w"):
        macro_primary = (
            Path(macro_kline_root) if macro_kline_root is not None else Path(".")
        )
        if tf == "1w":
            return fetch_ohlcv_weekly_macro(
                feature_bus_root,
                macro_primary,
                symbol,
                start=start,
                end=end,
                daily_start=daily_ohlcv_start or date(2017, 1, 1),
                max_daily_days=max_daily_ohlcv_days,
                full_range=full_range,
                live_data_root=live_data_root,
                live_root=live_root,
            )
        return fetch_ohlcv_daily_macro(
            feature_bus_root,
            macro_primary,
            symbol,
            start=start,
            end=end,
            daily_start=daily_ohlcv_start or date(2017, 1, 1),
            max_daily_days=max_daily_ohlcv_days,
            full_range=full_range,
            live_data_root=live_data_root,
            live_root=live_root,
        )
    path = _bars_path(feature_bus_root, symbol)
    bars_root = Path(live_storage_bars_root) if live_storage_bars_root else None
    do_stitch = bool(
        stitch_live_storage
        and bars_root is not None
        and bars_root.is_dir()
        and start is None
        and end is None
    )
    start_ts, end_ts, clipped, bars_1min_rows = _resolve_window(
        path,
        start=start,
        end=end,
        max_days=max_days,
        full_range=full_range and start is None and end is None,
        calendar_span=do_stitch,
    )
    bus_df = load_bars_1min(feature_bus_root, symbol, start=start_ts, end=end_ts)
    live_storage_rows = 0
    if do_stitch:
        hist = load_live_storage_bars_1min(
            bars_root, symbol, start=start_ts, end=end_ts
        )
        live_storage_rows = len(hist)
        raw = stitch_live_storage_and_bus(hist, bus_df)
        source = "live_storage+bars_1min"
    else:
        raw = bus_df
        source = "bars_1min"
    resampled, degraded = resample_ohlcv(raw, timeframe)
    mtime = path.stat().st_mtime if path.is_file() else None
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "candles": ohlcv_to_candles(resampled),
        "degraded_ohlc": degraded,
        "source": source,
        "source_mtime": mtime,
        "row_count": len(resampled),
        "bars_1min_rows": bars_1min_rows,
        "live_storage_1m_rows": live_storage_rows,
        "range_start": start_ts.isoformat(),
        "range_end": end_ts.isoformat(),
        "range_clipped": clipped,
        "max_ohlcv_days": max_days,
    }
