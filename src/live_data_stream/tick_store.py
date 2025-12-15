from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .websocket_client import BinanceTick


@dataclass
class TickStorage:
    """
    Persist 100ms aggregated ticks to parquet per symbol per date.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, trading_date: str) -> Path:
        symbol_dir = self.root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{trading_date}.parquet"

    def append(self, symbol: str, trading_date: str, df: pd.DataFrame) -> Path:
        """
        Append aggregated ticks to parquet (create if missing).
        """
        target = self._path(symbol, trading_date)
        if target.exists():
            existing = pd.read_parquet(target)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_parquet(target, index=False)
        else:
            df.to_parquet(target, index=False)
        return target

    def load(self, symbol: str, trading_date: str) -> pd.DataFrame:
        target = self._path(symbol, trading_date)
        if not target.exists():
            raise FileNotFoundError(f"tick file missing: {target}")
        return pd.read_parquet(target)


def aggregate_ticks_100ms(ticks: List[BinanceTick]) -> pd.DataFrame:
    """
    Aggregate raw ticks into 100ms windows with order-flow fields.
    """
    if not ticks:
        return pd.DataFrame()

    rows = []
    for t in ticks:
        rows.append(
            {
                "ts": pd.to_datetime(t.timestamp_ms, unit="ms", utc=True),
                "ts_ms": t.timestamp_ms,
                "price": t.price,
                "volume": t.volume,
                "turnover": t.turnover,
                "side": "BUY" if t.side == 1 else "SELL" if t.side == -1 else "UNK",
                "is_buy": 1 if t.side == 1 else 0,
                "is_sell": 1 if t.side == -1 else 0,
            }
        )

    df = pd.DataFrame(rows).sort_values("ts_ms")
    df["window"] = (df["ts_ms"] // 100) * 100
    grouped = df.groupby("window")

    agg = grouped.agg(
        open_price=("price", "first"),
        high_price=("price", "max"),
        low_price=("price", "min"),
        close_price=("price", "last"),
        volume=("volume", "sum"),
        buy_volume=("volume", lambda x: (x * df.loc[x.index, "is_buy"]).sum()),
        sell_volume=("volume", lambda x: (x * df.loc[x.index, "is_sell"]).sum()),
        trade_count=("price", "count"),
        first_ts=("ts_ms", "min"),
        last_ts=("ts_ms", "max"),
    ).reset_index()

    agg["buy_ratio"] = (agg["buy_volume"] / agg["volume"]).fillna(0)
    agg["sell_ratio"] = (agg["sell_volume"] / agg["volume"]).fillna(0)
    agg["delta"] = (agg["buy_volume"] - agg["sell_volume"]).fillna(0)
    agg["ts"] = pd.to_datetime(agg["window"], unit="ms", utc=True)
    return agg.sort_values("window")

