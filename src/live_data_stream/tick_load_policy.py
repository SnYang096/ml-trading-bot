"""Tick loading policy for live feature batch compute.

Order-flow / VPIN features need a rolling tick window (default 8 calendar days).
When the recent window is sparse, extend backward in small chunks instead of
loading the full historical span into memory at once.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.live_data_stream.feature_storage import StorageManager

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def load_ticks_for_feature_compute(
    storage: StorageManager,
    symbol: str,
    *,
    now: pd.Timestamp | None = None,
    bar_end: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load ticks for VPIN / order-flow batch compute.

    Returns ``(ticks_df, recent_window_count)`` where ``recent_window_count`` is
    the row count in the initial recent lookback (before any backward extension).
    """
    now_ts = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    end_str = bar_end or now_ts.strftime("%Y-%m-%d")
    base_days = max(1, _env_int("MLBOT_TICK_LOOKBACK_DAYS", 8))
    max_extended_days = max(base_days, _env_int("MLBOT_TICK_EXTENDED_MAX_DAYS", 30))
    chunk_days = max(1, _env_int("MLBOT_TICK_LOAD_CHUNK_DAYS", 7))
    min_required = max(1, _env_int("MLBOT_MIN_TICKS_REQUIRED", 15000))

    oldest_allowed = (now_ts - timedelta(days=max_extended_days)).normalize()

    recent_start = (now_ts - timedelta(days=base_days)).strftime("%Y-%m-%d")
    ticks = storage.ticks.load_range(symbol, recent_start, end_str)
    recent_count = len(ticks)

    if recent_count >= min_required:
        logger.debug(
            "[%s] tick load: recent %dd sufficient (%d rows)",
            symbol,
            base_days,
            recent_count,
        )
        return ticks, recent_count

    cursor_end = pd.Timestamp(recent_start, tz="UTC") - timedelta(days=1)
    while len(ticks) < min_required and cursor_end.normalize() >= oldest_allowed:
        chunk_start = max(oldest_allowed, cursor_end - timedelta(days=chunk_days - 1))
        chunk = storage.ticks.load_range(
            symbol,
            chunk_start.strftime("%Y-%m-%d"),
            cursor_end.strftime("%Y-%m-%d"),
        )
        if not chunk.empty:
            ticks = _merge_tick_frames(chunk, ticks)
            logger.info(
                "[%s] tick load chunk: %s ~ %s (+=%d, total=%d)",
                symbol,
                chunk_start.date(),
                cursor_end.date(),
                len(chunk),
                len(ticks),
            )
        cursor_end = chunk_start - timedelta(days=1)

    if len(ticks) < min_required:
        return ticks, recent_count

    return ticks, recent_count


def _merge_tick_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return right
    if right.empty:
        return left
    merged = pd.concat([left, right], ignore_index=True)
    dedupe_cols = [
        c for c in ("timestamp", "price", "volume", "side") if c in merged.columns
    ]
    if dedupe_cols:
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    if "timestamp" in merged.columns:
        merged = merged.sort_values("timestamp")
    return merged.reset_index(drop=True)
