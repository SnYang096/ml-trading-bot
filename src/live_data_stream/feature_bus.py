"""Disk-backed feature bus for live market-data fanout.

The bus is intentionally boring: producers atomically rewrite small rolling
parquet snapshots, and consumers poll latest closed rows by timestamp. This is
the first step before introducing any IPC broker.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.live_data_stream.parquet_io import atomic_write_parquet


def normalize_timeframe(timeframe: str) -> str:
    """Return a filesystem-safe timeframe key."""
    return str(timeframe or "").strip().replace("/", "_").replace(" ", "")


def list_feature_bus_timeframe_dirs(feature_bus_root: str | Path) -> List[str]:
    """Return timeframe keys under ``features/`` that have at least one parquet file."""
    feat_root = Path(feature_bus_root) / "features"
    if not feat_root.is_dir():
        return []
    out: List[str] = []
    for child in sorted(feat_root.iterdir()):
        if child.is_dir() and any(child.glob("*.parquet")):
            out.append(child.name)
    return out


def resolve_disk_primary_timeframe(
    feature_bus_root: str | Path,
    strategy_timeframe: str,
) -> Tuple[str, bool]:
    """Pick reader primary key: ``120T``/``2h`` from meta, or legacy ``primary`` dir.

    Returns:
        (disk_timeframe_key, used_legacy_primary)
    """
    feat_root = Path(feature_bus_root) / "features"
    meta_tf = str(strategy_timeframe or "").strip() or "120T"
    mp_n = normalize_timeframe(meta_tf)
    preferred = feat_root / mp_n
    legacy_primary = feat_root / "primary"
    if preferred.is_dir() and any(preferred.glob("*.parquet")):
        return meta_tf, False
    if legacy_primary.is_dir() and any(legacy_primary.glob("*.parquet")):
        return "primary", True
    return meta_tf, False


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class FeatureBusWriter:
    """Append rolling bar/feature snapshots to a local disk bus."""

    def __init__(self, root: str | Path, *, max_rows: int = 5000) -> None:
        self.root = Path(root)
        self.max_rows = int(max_rows)

    def merge_bars_1m(
        self,
        symbol: str,
        bars: pd.DataFrame,
        *,
        preserve_history: bool = False,
    ) -> int:
        """Merge repaired/archive 1m bars into the rolling bus snapshot.

        ``preserve_history=True`` is the contract for one-shot/repair callers:
        the merged frame is never tailed below ``len(old)``, so a backfill
        script with a small ``max_rows`` cannot accidentally shrink a bus that
        was grown by the online publisher.
        """
        if bars.empty or "timestamp" not in bars.columns:
            return 0
        incoming = bars.copy()
        incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True)
        incoming = (
            incoming.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )
        path = self.root / "bars_1min" / f"{symbol.upper()}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
            df = pd.concat([old, incoming], ignore_index=True)
        else:
            df = incoming
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = (
            df.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if not preserve_history:
            df = df.tail(self.max_rows).reset_index(drop=True)
        atomic_write_parquet(df, path)
        if not df.empty:
            last_ts = _utc_timestamp(df["timestamp"].iloc[-1])
            self._write_latest(
                kind="bars_1min",
                symbol=symbol,
                timestamp=last_ts,
                path=path,
                rows=len(df),
            )
        return len(incoming)

    def append_bar_1m(self, symbol: str, bar: Dict[str, Any]) -> Path:
        row = dict(bar)
        if "timestamp" not in row:
            raise ValueError("bar snapshot requires timestamp")
        row["timestamp"] = _utc_timestamp(row["timestamp"])
        path = self.root / "bars_1min" / f"{symbol.upper()}.parquet"
        df = self._append_row(path, row)
        self._write_latest(
            kind="bars_1min",
            symbol=symbol,
            timestamp=row["timestamp"],
            path=path,
            rows=len(df),
        )
        return path

    def append_features(
        self,
        *,
        symbol: str,
        timeframe: str,
        features: Dict[str, Any],
        timestamp: Any,
    ) -> Path:
        tf = normalize_timeframe(timeframe)
        row = dict(features)
        row["timestamp"] = _utc_timestamp(row.get("timestamp", timestamp))
        path = self.root / "features" / tf / f"{symbol.upper()}.parquet"
        df = self._append_row(path, row)
        self._write_latest(
            kind=f"features/{tf}",
            symbol=symbol,
            timestamp=row["timestamp"],
            path=path,
            rows=len(df),
        )
        return path

    def _append_row(self, path: Path, row: Dict[str, Any]) -> pd.DataFrame:
        new_df = pd.DataFrame([row])
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, new_df], ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").tail(self.max_rows).reset_index(drop=True)
        else:
            df = new_df.sort_values("timestamp").reset_index(drop=True)
        atomic_write_parquet(df, path)
        return df

    def _write_latest(
        self,
        *,
        kind: str,
        symbol: str,
        timestamp: pd.Timestamp,
        path: Path,
        rows: int,
    ) -> None:
        meta_path = self.root / "latest" / kind / f"{symbol.upper()}.json"
        _atomic_write_json(
            {
                "kind": kind,
                "symbol": symbol.upper(),
                "timestamp": timestamp.isoformat(),
                "path": str(path),
                "rows": int(rows),
            },
            meta_path,
        )


class FeatureBusReader:
    """Poll rolling snapshots produced by :class:`FeatureBusWriter`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def latest_features(
        self,
        *,
        symbol: str,
        timeframe: str,
        after: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.Series]:
        tf = normalize_timeframe(timeframe)
        path = self.root / "features" / tf / f"{symbol.upper()}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty or "timestamp" not in df.columns:
            return None
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if after is not None:
            after_ts = _utc_timestamp(after)
            df = df[df["timestamp"] > after_ts]
        if df.empty:
            return None
        return df.sort_values("timestamp").iloc[-1]

    def latest_snapshot_age_seconds(
        self, *, symbol: str, timeframe: str
    ) -> Optional[float]:
        """Return seconds since the writer's ``latest/features/<tf>/`` JSON timestamp.

        Lightweight (reads one small JSON file); used for Prometheus bus health.
        """
        tf = normalize_timeframe(timeframe)
        meta = self.root / "latest" / f"features/{tf}" / f"{symbol.upper()}.json"
        if not meta.exists():
            return None
        try:
            raw = json.loads(meta.read_text(encoding="utf-8"))
            ts = _utc_timestamp(raw.get("timestamp"))
        except (OSError, ValueError, TypeError, KeyError):
            return None
        now = pd.Timestamp.now(tz="UTC")
        return max(0.0, float((now - ts).total_seconds()))

    def latest_bars_1m(
        self, *, symbol: str, after: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        path = self.root / "bars_1min" / f"{symbol.upper()}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if after is not None:
            df = df[df["timestamp"] > _utc_timestamp(after)]
        return df.sort_values("timestamp").reset_index(drop=True)

    def list_available_symbols(self, *, timeframe: str) -> Iterable[str]:
        tf = normalize_timeframe(timeframe)
        base = self.root / "features" / tf
        if not base.exists():
            return []
        return sorted(p.stem.upper() for p in base.glob("*.parquet"))
