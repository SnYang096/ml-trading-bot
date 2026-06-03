"""100ms OHLC from Binance aggTrades or research parquet ticks for chop_grid replay."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cur = pd.Timestamp(start.year, start.month, 1, tz=start.tz)
    last = pd.Timestamp(end.year, end.month, 1, tz=end.tz)
    while cur <= last:
        yield cur
        cur = cur + pd.DateOffset(months=1)


def _aggregate_tick_prices(
    prices: pd.Series,
    volumes: pd.Series,
    *,
    bar_delta: pd.Timedelta,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    bucket = idx.floor(bar_delta)
    frame = pd.DataFrame(
        {"price": prices.values, "volume": volumes.values}, index=bucket
    )
    return (
        frame.groupby(level=0)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
        )
        .sort_index()
    )


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_segment_tick_bars_from_parquet(
    *,
    symbol: str,
    parquet_data_dir: str | Path,
    t_enter: pd.Timestamp,
    t_exit: pd.Timestamp,
    bar_delta: pd.Timedelta,
) -> pd.DataFrame:
    """Build OHLCV for ``[t_enter, t_exit)`` from ``data/parquet_data`` tick parquet.

    Files are monthly ``{SYMBOL}_YYYY-MM.parquet`` with columns
    ``timestamp, price, volume, side``. Effective bar size depends on source
    timestamp resolution (research parquet is typically **1min**, not sub-second).
    """
    sym = str(symbol).upper()
    root = Path(parquet_data_dir)
    t_enter = _ensure_utc(t_enter)
    t_exit = _ensure_utc(t_exit)
    if t_exit <= t_enter:
        return pd.DataFrame()

    parts: List[pd.DataFrame] = []
    for month in _month_starts(t_enter, t_exit):
        path = root / f"{sym}_{month.strftime('%Y-%m')}.parquet"
        if not path.exists():
            continue
        chunk = pd.read_parquet(path, columns=["timestamp", "price", "volume"])
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["timestamp"], utc=True)
        mask = (ts >= t_enter) & (ts < t_exit)
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        sub.index = ts[mask]
        parts.append(
            _aggregate_tick_prices(sub["price"], sub["volume"], bar_delta=bar_delta)
        )

    if not parts:
        raise FileNotFoundError(
            f"no parquet ticks for {sym} in [{t_enter}, {t_exit}) under {root}"
        )

    out = (
        pd.concat(parts)
        .sort_index()
        .groupby(level=0)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    return out.sort_index()


def load_segment_100ms_bars(
    *,
    symbol: str,
    agg_data_dir: str | Path | None = None,
    parquet_data_dir: str | Path | None = None,
    t_enter: pd.Timestamp,
    t_exit: pd.Timestamp,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Build 100ms OHLCV for ``[t_enter, t_exit)``.

    Tries ``agg_data_dir`` aggTrades zips first, then ``parquet_data_dir`` tick
    parquet. Raises ``FileNotFoundError`` if neither source has ticks in window.
    """
    sym = str(symbol).upper()
    t_enter = _ensure_utc(t_enter)
    t_exit = _ensure_utc(t_exit)
    bar_delta = pd.Timedelta(milliseconds=100)

    if agg_data_dir is not None:
        root = Path(agg_data_dir)
        parts: List[pd.DataFrame] = []
        for month in _month_starts(t_enter, t_exit):
            zip_path = root / f"{sym}-aggTrades-{month.strftime('%Y-%m')}.zip"
            if not zip_path.exists():
                continue
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(zf.namelist()[0]) as fh:
                    for chunk in pd.read_csv(
                        fh,
                        usecols=["price", "quantity", "transact_time"],
                        chunksize=chunksize,
                    ):
                        ts = pd.to_datetime(chunk["transact_time"], unit="ms", utc=True)
                        mask = (ts >= t_enter) & (ts < t_exit)
                        if not mask.any():
                            continue
                        c = chunk.loc[mask, ["price", "quantity"]].copy()
                        c.index = ts[mask].dt.floor("100ms")
                        bars = c.groupby(level=0).agg(
                            open=("price", "first"),
                            high=("price", "max"),
                            low=("price", "min"),
                            close=("price", "last"),
                            volume=("quantity", "sum"),
                        )
                        parts.append(bars)
        if parts:
            out = (
                pd.concat(parts)
                .sort_index()
                .groupby(level=0)
                .agg(
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                )
            )
            return out.sort_index()

    if parquet_data_dir is not None:
        return load_segment_tick_bars_from_parquet(
            symbol=sym,
            parquet_data_dir=parquet_data_dir,
            t_enter=t_enter,
            t_exit=t_exit,
            bar_delta=bar_delta,
        )

    raise FileNotFoundError(
        f"no 100ms tick source for {sym} in [{t_enter}, {t_exit}) "
        f"(agg_data_dir={agg_data_dir!r}, parquet_data_dir={parquet_data_dir!r})"
    )
