"""Probe on-disk pipeline stages and export Prometheus freshness gauges."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from src.live_data_stream.feature_bus import FeatureBusReader, normalize_timeframe

logger = logging.getLogger(__name__)

# Max age (seconds) before a pipeline is considered stale.
DEFAULT_STALE_SECONDS = {
    "ticks_1m": 300,
    "bars_1m": 300,
    "bus_bars_1min": 300,
    "features_15min": 1200,
    "features_120T": 1200,
    "features_240T": 16200,
    "macro_seed": 172800,
}


def _stale_threshold(pipeline: str) -> float:
    raw = os.getenv(f"MLBOT_PIPELINE_STALE_{pipeline.upper()}", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(DEFAULT_STALE_SECONDS.get(pipeline, 3600))


def _age_from_latest_mtime(paths: Iterable[Path]) -> Optional[float]:
    latest: Optional[float] = None
    for path in paths:
        if not path.exists():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    if latest is None:
        return None
    return max(0.0, time.time() - latest)


def _glob_newest_parquet(root: Path, pattern: str = "**/*.parquet") -> Optional[Path]:
    if not root.is_dir():
        return None
    newest: Optional[Path] = None
    newest_mtime = -1.0
    try:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest = path
    except OSError:
        return None
    return newest


def _dated_storage_age_seconds(storage_root: Path, symbol: str) -> Optional[float]:
    sym_dir = storage_root / symbol.upper()
    if not sym_dir.is_dir():
        return None
    return _age_from_latest_mtime(sym_dir.glob("*.parquet"))


def collect_pipeline_ages(
    symbols: Sequence[str],
    *,
    storage_base: Path | str,
    bus_root: Path | str,
    seed_root: Optional[Path | str] = None,
    feature_timeframes: Optional[Sequence[str]] = None,
) -> dict[tuple[str, str], float]:
    """Return {(pipeline, symbol): age_seconds}. symbol may be ``_all``."""
    storage = Path(storage_base)
    bus = Path(bus_root)
    seed = Path(seed_root) if seed_root else None
    tfs = list(feature_timeframes or ("120T", "240T", "15min"))
    reader = FeatureBusReader(bus)
    out: dict[tuple[str, str], float] = {}

    for sym in symbols:
        su = str(sym).strip().upper()
        if not su:
            continue
        for pipeline, root in (
            ("ticks_1m", storage / "ticks"),
            ("bars_1m", storage / "bars"),
        ):
            age = _dated_storage_age_seconds(root, su)
            if age is not None:
                out[(pipeline, su)] = age

        bus_bar = bus / "bars_1min" / f"{su}.parquet"
        age = _age_from_latest_mtime([bus_bar])
        if age is not None:
            out[("bus_bars_1min", su)] = age

        for tf in tfs:
            tf_norm = normalize_timeframe(tf)
            key = f"features_{tf_norm}"
            age = reader.latest_snapshot_age_seconds(symbol=su, timeframe=tf_norm)
            if age is not None:
                out[(key, su)] = float(age)

    if seed and seed.is_dir():
        age = _age_from_latest_mtime(seed.glob("*.parquet"))
        if age is not None:
            out[("macro_seed", "_all")] = age

    return out


def update_pipeline_freshness_metrics(
    symbols: Sequence[str],
    *,
    storage_base: Path | str,
    bus_root: Path | str,
    seed_root: Optional[Path | str] = None,
    feature_timeframes: Optional[Sequence[str]] = None,
) -> None:
    """Export mlbot_pipeline_data_age_seconds / mlbot_pipeline_data_fresh on METRICS."""
    try:
        from src.time_series_model.live.metrics_exporter import METRICS
    except Exception:
        return

    if not getattr(METRICS, "pipeline_data_age_seconds", None):
        return

    ages = collect_pipeline_ages(
        symbols,
        storage_base=storage_base,
        bus_root=bus_root,
        seed_root=seed_root,
        feature_timeframes=feature_timeframes,
    )
    for (pipeline, symbol), age in ages.items():
        METRICS.pipeline_data_age_seconds.labels(
            pipeline=pipeline, symbol=symbol
        ).set(round(float(age), 2))
        fresh = 1.0 if float(age) <= _stale_threshold(pipeline) else 0.0
        METRICS.pipeline_data_fresh.labels(pipeline=pipeline, symbol=symbol).set(
            fresh
        )
