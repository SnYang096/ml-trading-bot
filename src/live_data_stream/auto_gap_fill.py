"""Background gap repair for live 1m bars/ticks.

The feature-bus owns market data writes, so it is the safest place to repair
large storage gaps without relying on a separate operator command.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import pandas as pd

from src.live_data_stream.feature_storage import StorageManager
from src.live_data_stream.gap_filler import GapFiller

if TYPE_CHECKING:
    from src.live_data_stream.feature_bus import FeatureBusWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarGap:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    minutes: float
    kind: str = "internal"


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _load_recent_bars(
    storage: StorageManager,
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    df = storage.bar_1min.load_range(
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)]
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _load_recent_ticks(
    storage: StorageManager,
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    df = storage.ticks.load_range(
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)]
    return out.sort_values("timestamp")


def _gaps_from_minute_series(
    symbol: str,
    minute_ts: pd.Series,
    *,
    end: pd.Timestamp,
    min_gap: float,
    kind_prefix: str,
) -> List[BarGap]:
    minutes = (
        pd.to_datetime(minute_ts, utc=True)
        .dt.floor("min")
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if minutes.empty:
        return []

    gaps: List[BarGap] = []
    diffs = minutes.diff().dt.total_seconds().div(60.0)
    for idx, gap_minutes in diffs[diffs > min_gap + 1.0].items():
        prev_ts = minutes.iloc[int(idx) - 1]
        next_ts = minutes.iloc[int(idx)]
        gaps.append(
            BarGap(
                symbol=symbol,
                start=prev_ts + pd.Timedelta(minutes=1),
                end=next_ts - pd.Timedelta(minutes=1),
                minutes=float(gap_minutes - 1.0),
                kind=f"{kind_prefix}_internal",
            )
        )

    last_ts = minutes.iloc[-1]
    tail_minutes = (end - last_ts).total_seconds() / 60.0
    if tail_minutes > min_gap:
        gaps.append(
            BarGap(
                symbol=symbol,
                start=last_ts + pd.Timedelta(minutes=1),
                end=end.floor("min"),
                minutes=float(tail_minutes),
                kind=f"{kind_prefix}_tail",
            )
        )
    return gaps


def detect_large_bar_gaps(
    storage: StorageManager,
    symbols: Iterable[str],
    *,
    lookback_hours: float = 48.0,
    min_gap_minutes: float = 60.0,
    ignore_recent_minutes: float = 5.0,
    now: Optional[pd.Timestamp] = None,
) -> List[BarGap]:
    """Find large gaps in persisted 1m bars.

    ``ignore_recent_minutes`` avoids racing the currently forming bar.
    """
    now_ts = _utc_timestamp(now or pd.Timestamp.now(tz="UTC"))
    end = now_ts - pd.Timedelta(minutes=float(ignore_recent_minutes))
    start = end - pd.Timedelta(hours=float(lookback_hours))
    min_gap = float(min_gap_minutes)
    gaps: List[BarGap] = []

    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        try:
            bars = _load_recent_bars(storage, symbol, start=start, end=end)
        except Exception:
            logger.warning(
                "auto-gap-fill: bar scan failed for %s in last %.1fh",
                symbol,
                lookback_hours,
                exc_info=True,
            )
            continue
        if bars.empty:
            logger.warning(
                "auto-gap-fill: no recent bars for %s in last %.1fh",
                symbol,
                lookback_hours,
            )
            continue

        gaps.extend(
            _gaps_from_minute_series(
                symbol,
                bars["timestamp"],
                end=end,
                min_gap=min_gap,
                kind_prefix="bars",
            )
        )

    return gaps


def detect_large_tick_gaps(
    storage: StorageManager,
    symbols: Iterable[str],
    *,
    lookback_hours: float = 48.0,
    min_gap_minutes: float = 60.0,
    ignore_recent_minutes: float = 5.0,
    now: Optional[pd.Timestamp] = None,
) -> List[BarGap]:
    """Find large gaps in persisted tick minutes."""
    now_ts = _utc_timestamp(now or pd.Timestamp.now(tz="UTC"))
    end = now_ts - pd.Timedelta(minutes=float(ignore_recent_minutes))
    start = end - pd.Timedelta(hours=float(lookback_hours))
    min_gap = float(min_gap_minutes)
    gaps: List[BarGap] = []

    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        try:
            ticks = _load_recent_ticks(storage, symbol, start=start, end=end)
        except Exception:
            logger.warning(
                "auto-gap-fill: tick scan failed for %s in last %.1fh",
                symbol,
                lookback_hours,
                exc_info=True,
            )
            continue
        if ticks.empty:
            logger.warning(
                "auto-gap-fill: no recent ticks for %s in last %.1fh",
                symbol,
                lookback_hours,
            )
            continue

        gaps.extend(
            _gaps_from_minute_series(
                symbol,
                ticks["timestamp"],
                end=end,
                min_gap=min_gap,
                kind_prefix="ticks",
            )
        )

    return gaps


def _dedupe_gaps(gaps: Iterable[BarGap]) -> List[BarGap]:
    seen: set[tuple[str, int, int]] = set()
    out: List[BarGap] = []
    for gap in sorted(gaps, key=lambda g: (g.symbol, g.start, g.end, g.kind)):
        key = (gap.symbol, int(gap.start.timestamp()), int(gap.end.timestamp()))
        if key in seen:
            continue
        seen.add(key)
        out.append(gap)
    return out


def _save_bars_by_day(storage: StorageManager, symbol: str, bars: pd.DataFrame) -> int:
    if bars.empty or "timestamp" not in bars.columns:
        return 0
    out = bars.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    out["_date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    saved = 0
    for date_str, day_bars in out.groupby("_date"):
        day_data = day_bars.drop(columns=["_date"])
        storage.bar_1min.append(symbol, date_str, day_data, include_incomplete=False)
        saved += len(day_data)
    return saved


def _save_ticks_by_day(
    storage: StorageManager, symbol: str, ticks: pd.DataFrame
) -> int:
    if ticks.empty or "timestamp" not in ticks.columns:
        return 0
    out = ticks.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp")
    dedupe_cols = [
        c for c in ("timestamp", "price", "volume", "side") if c in out.columns
    ]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last")
    out["_date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    saved = 0
    for date_str, day_ticks in out.groupby("_date"):
        day_data = day_ticks.drop(columns=["_date"])
        storage.ticks.append(symbol, date_str, day_data)
        saved += len(day_data)
    return saved


def sync_filled_bars_to_feature_bus(
    writer: FeatureBusWriter,
    filled_by_symbol: Dict[str, pd.DataFrame],
) -> int:
    """Push gap-fill output into rolling feature-bus bar snapshots."""
    synced = 0
    for symbol, bars in filled_by_symbol.items():
        if bars is None or bars.empty:
            continue
        try:
            synced += writer.merge_bars_1m(symbol, bars)
        except Exception:
            logger.exception(
                "auto-gap-fill: feature-bus sync failed for %s", symbol
            )
    if synced:
        logger.warning(
            "auto-gap-fill: synced %d filled bar rows to feature bus", synced
        )
    return synced


def sync_archive_bars_to_feature_bus(
    storage: StorageManager,
    writer: FeatureBusWriter,
    symbols: Iterable[str],
    *,
    lookback_hours: float = 168.0,
    now: Optional[pd.Timestamp] = None,
) -> int:
    """Merge recent archive 1m bars into feature bus (archive wins on overlap)."""
    now_ts = _utc_timestamp(now or pd.Timestamp.now(tz="UTC"))
    start = now_ts - pd.Timedelta(hours=float(lookback_hours))
    synced = 0
    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        try:
            bars = _load_recent_bars(storage, symbol, start=start, end=now_ts)
        except Exception:
            logger.exception(
                "feature-bus sync: archive load failed for %s", symbol
            )
            continue
        if bars.empty:
            logger.warning(
                "feature-bus sync: no archive bars for %s in last %.1fh",
                symbol,
                lookback_hours,
            )
            continue
        try:
            n = writer.merge_bars_1m(symbol, bars)
            synced += n
            logger.warning(
                "feature-bus sync: merged %d archive rows for %s", n, symbol
            )
        except Exception:
            logger.exception(
                "feature-bus sync: merge failed for %s", symbol
            )
    return synced


def fill_large_bar_gaps(
    storage: StorageManager,
    gap_filler: GapFiller,
    gaps: Iterable[BarGap],
    *,
    max_gaps_per_run: int = 24,
    now: Optional[pd.Timestamp] = None,
    feature_bus_writer: Optional[FeatureBusWriter] = None,
) -> int:
    """Fill detected gaps and return written 1m bar count."""
    now_ts = _utc_timestamp(now or pd.Timestamp.now(tz="UTC"))
    today = now_ts.normalize()
    written = 0
    filled_by_symbol: Dict[str, List[pd.DataFrame]] = {}

    def _record_filled(sym: str, bars: pd.DataFrame) -> None:
        if bars is None or bars.empty:
            return
        filled_by_symbol.setdefault(str(sym).upper(), []).append(bars)

    for gap in list(gaps)[: int(max_gaps_per_run)]:
        logger.warning(
            "auto-gap-fill: detected %s gap %s %s -> %s (%.1f min)",
            gap.kind,
            gap.symbol,
            gap.start,
            gap.end,
            gap.minutes,
        )
        try:
            if gap.end.normalize() < today and gap_filler.data_gap_filler is not None:
                bars, raw_ticks = (
                    gap_filler.data_gap_filler.fill_gap_with_binance_vision(
                        gap_filler._convert_symbol(gap.symbol),
                        gap.start,
                        gap.end,
                    )
                )
                if not bars.empty:
                    gap_filler._save_filled_data(gap.symbol, bars, raw_ticks)
                    written += len(bars)
                    _record_filled(gap.symbol, bars)
                    logger.warning(
                        "auto-gap-fill: filled %s via Vision (%d bars, %d raw ticks)",
                        gap.symbol,
                        len(bars),
                        len(raw_ticks) if raw_ticks is not None else 0,
                    )
                else:
                    logger.warning("auto-gap-fill: Vision returned no rows for %s", gap)
                continue

            raw_ticks = gap_filler.fill_missing_ticks(gap.symbol, gap.start, gap.end)
            if (
                not raw_ticks.empty
                and gap_filler.data_gap_filler is not None
                and hasattr(gap_filler.data_gap_filler, "_aggregate_trades_to_1min")
            ):
                bars = gap_filler.data_gap_filler._aggregate_trades_to_1min(raw_ticks)
                saved_bars = _save_bars_by_day(storage, gap.symbol, bars)
                saved_ticks = _save_ticks_by_day(storage, gap.symbol, raw_ticks)
                written += saved_bars
                _record_filled(gap.symbol, bars)
                logger.warning(
                    "auto-gap-fill: filled %s via aggTrades (%d bars, %d raw ticks)",
                    gap.symbol,
                    saved_bars,
                    saved_ticks,
                )
                continue

            bars = gap_filler.fill_from_binance_api(
                gap.symbol,
                gap.start,
                gap.end,
                timeframe="1m",
            )
            saved = _save_bars_by_day(storage, gap.symbol, bars)
            written += saved
            if saved:
                _record_filled(gap.symbol, bars)
                logger.warning(
                    "auto-gap-fill: filled %s via kline API fallback (%d bars; no raw ticks)",
                    gap.symbol,
                    saved,
                )
            else:
                logger.warning("auto-gap-fill: kline API returned no rows for %s", gap)
        except Exception:
            logger.exception("auto-gap-fill: failed filling gap %s", gap)

    if feature_bus_writer is not None and filled_by_symbol:
        merged: Dict[str, pd.DataFrame] = {}
        for sym, frames in filled_by_symbol.items():
            merged[sym] = pd.concat(frames, ignore_index=True)
        sync_filled_bars_to_feature_bus(feature_bus_writer, merged)

    return written


def run_auto_gap_fill_once(
    storage: StorageManager,
    gap_filler: GapFiller,
    symbols: Iterable[str],
    *,
    lookback_hours: float = 48.0,
    min_gap_minutes: float = 60.0,
    max_gaps_per_run: int = 24,
    now: Optional[pd.Timestamp] = None,
    feature_bus_writer: Optional["FeatureBusWriter"] = None,
) -> int:
    """Run one repair pass for pending Vision gaps plus scanned bar/tick gaps."""
    symbol_list = [str(s).upper() for s in symbols]
    try:
        return _run_auto_gap_fill_once_impl(
            storage,
            gap_filler,
            symbol_list,
            lookback_hours=lookback_hours,
            min_gap_minutes=min_gap_minutes,
            max_gaps_per_run=max_gaps_per_run,
            now=now,
            feature_bus_writer=feature_bus_writer,
        )
    except Exception:
        logger.exception("auto-gap-fill: run failed")
        return 0


def _run_auto_gap_fill_once_impl(
    storage: StorageManager,
    gap_filler: GapFiller,
    symbol_list: List[str],
    *,
    lookback_hours: float,
    min_gap_minutes: float,
    max_gaps_per_run: int,
    now: Optional[pd.Timestamp],
    feature_bus_writer: Optional["FeatureBusWriter"] = None,
) -> int:
    pending_count = len(getattr(gap_filler, "_pending_vision_gaps", []))
    if pending_count:
        logger.warning(
            "auto-gap-fill: retrying %d queued Vision gaps before scan",
            pending_count,
        )
        gap_filler.retry_pending_gaps()

    gaps = detect_large_bar_gaps(
        storage,
        symbol_list,
        lookback_hours=lookback_hours,
        min_gap_minutes=min_gap_minutes,
        now=now,
    )
    gaps.extend(
        detect_large_tick_gaps(
            storage,
            symbol_list,
            lookback_hours=lookback_hours,
            min_gap_minutes=min_gap_minutes,
            now=now,
        )
    )
    gaps = _dedupe_gaps(gaps)
    if not gaps:
        logger.info("auto-gap-fill: no gaps >= %.1f min", min_gap_minutes)
        return 0

    written = fill_large_bar_gaps(
        storage,
        gap_filler,
        gaps,
        max_gaps_per_run=max_gaps_per_run,
        now=now,
        feature_bus_writer=feature_bus_writer,
    )
    logger.warning(
        "auto-gap-fill: run complete gaps=%d written_bars=%d",
        len(gaps),
        written,
    )
    return written


async def auto_gap_fill_loop(
    storage: StorageManager,
    gap_filler: GapFiller,
    symbols: Iterable[str],
    *,
    interval_seconds: float = 3600.0,
    lookback_hours: float = 48.0,
    startup_lookback_hours: Optional[float] = None,
    min_gap_minutes: float = 60.0,
    max_gaps_per_run: int = 24,
    initial_delay_seconds: float = 300.0,
    feature_bus_writer: Optional["FeatureBusWriter"] = None,
) -> None:
    if initial_delay_seconds > 0:
        await asyncio.sleep(float(initial_delay_seconds))

    loop = asyncio.get_running_loop()
    symbol_list = [str(s).upper() for s in symbols]
    first_run = True
    while True:
        try:
            scan_lookback = (
                float(startup_lookback_hours)
                if first_run and startup_lookback_hours is not None
                else float(lookback_hours)
            )

            def _run_once() -> int:
                return run_auto_gap_fill_once(
                    storage,
                    gap_filler,
                    symbol_list,
                    lookback_hours=scan_lookback,
                    min_gap_minutes=min_gap_minutes,
                    max_gaps_per_run=max_gaps_per_run,
                    feature_bus_writer=feature_bus_writer,
                )

            await loop.run_in_executor(None, _run_once)
            first_run = False
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("auto-gap-fill: background run failed")

        try:
            await asyncio.sleep(float(interval_seconds))
        except asyncio.CancelledError:
            break
