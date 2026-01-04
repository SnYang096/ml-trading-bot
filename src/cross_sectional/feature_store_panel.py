from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureStorePanelConfig:
    root: str = "feature_store"
    layer: str = ""
    timeframe: str = "240T"
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"


def _month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = pd.Timestamp(start).normalize().replace(day=1)
    end = pd.Timestamp(end).normalize().replace(day=1)
    return list(pd.date_range(start=start, end=end, freq="MS"))


def _feature_store_month_path(
    *,
    root: Path,
    layer: str,
    symbol: str,
    timeframe: str,
    month_key: str,
) -> Path:
    # Convention used by `src/cross_sectional/scripts/rank_tokens.py`
    return root / layer / symbol / timeframe / f"{month_key}.parquet"


def load_feature_store_frames(
    *,
    symbols: Sequence[str],
    cfg: FeatureStorePanelConfig,
    start_date: str,
    end_date: str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load FeatureStore monthly partitions for multiple symbols, concatenate,
    and filter to [start_date, end_date] (inclusive on date, time-aware).

    Returns a flat table with columns: timestamp, symbol, <features...>
    """
    if not cfg.layer:
        raise ValueError("FeatureStorePanelConfig.layer is required")
    if not symbols:
        raise ValueError("symbols is empty")

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)

    root = Path(cfg.root)
    parts: List[pd.DataFrame] = []

    months = _month_starts(start_ts, end_ts)
    for sym in symbols:
        sym = str(sym).strip().upper()
        for ms in months:
            p = _feature_store_month_path(
                root=root,
                layer=str(cfg.layer),
                symbol=sym,
                timeframe=str(cfg.timeframe),
                month_key=_month_key(ms),
            )
            if not p.exists():
                continue
            df = pd.read_parquet(p)

            # FeatureStore parquet commonly has timestamp index.
            if isinstance(df.index, pd.DatetimeIndex):
                idx_name = df.index.name or cfg.timestamp_col
                if idx_name != cfg.timestamp_col:
                    df = df.copy()
                    df.index.name = cfg.timestamp_col
                df = df.copy()
                df[cfg.timestamp_col] = df.index

            if cfg.timestamp_col not in df.columns:
                raise ValueError(f"Missing '{cfg.timestamp_col}' in {p}")

            df[cfg.timestamp_col] = pd.to_datetime(
                df[cfg.timestamp_col], utc=True, errors="coerce"
            )
            df = df.dropna(subset=[cfg.timestamp_col])

            # Add symbol column if absent.
            if cfg.symbol_col not in df.columns:
                df = df.copy()
                df[cfg.symbol_col] = sym
            else:
                df[cfg.symbol_col] = df[cfg.symbol_col].astype(str)

            # Column selection (always keep timestamp+symbol)
            if columns:
                keep = [cfg.timestamp_col, cfg.symbol_col] + [
                    c for c in columns if c in df.columns
                ]
                df = df[keep]

            parts.append(df)

    if not parts:
        raise FileNotFoundError(
            f"No FeatureStore partitions found for layer={cfg.layer}, timeframe={cfg.timeframe}"
        )

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values([cfg.timestamp_col, cfg.symbol_col])
    mask = (out[cfg.timestamp_col] >= start_ts) & (out[cfg.timestamp_col] < end_ts)
    out = out.loc[mask].copy()
    return out


def available_month_partitions(
    *,
    symbol: str,
    cfg: FeatureStorePanelConfig,
) -> List[Path]:
    root = Path(cfg.root)
    base = root / str(cfg.layer) / str(symbol).upper() / str(cfg.timeframe)
    if not base.exists() or not base.is_dir():
        return []
    return sorted(base.glob("*.parquet"))
