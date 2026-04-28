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
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def normalize_timeframe(timeframe: str) -> str:
    """Return a filesystem-safe timeframe key."""
    return str(timeframe or "").strip().replace("/", "_").replace(" ", "")


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


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
        _atomic_write_parquet(df, path)
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
