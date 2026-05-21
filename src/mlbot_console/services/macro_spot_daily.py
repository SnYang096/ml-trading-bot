"""Binance Vision spot 1d ZIP cache (same layout as src/live_data_stream/spot_weekly_ema_seed)."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from typing import List, Tuple

import pandas as pd

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not s:
        raise ValueError("empty symbol")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def _month_list(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[Tuple[int, int]]:
    months: List[Tuple[int, int]] = []
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def parse_kline_zip_bytes(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=KLINE_COLUMNS,
                usecols=range(len(KLINE_COLUMNS)),
            )
    if df.empty:
        return df
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    bar_index = pd.DatetimeIndex(ts.to_numpy())
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(df["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(df["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(df["close"], errors="coerce").to_numpy(),
            "volume": pd.to_numeric(df["volume"], errors="coerce").to_numpy(),
        },
        index=bar_index,
    )
    out = out[~out.index.isna()].sort_index()
    return out[~out.index.duplicated(keep="last")]


class MacroSpotDailyLoader:
    def __init__(self, kline_root: Path) -> None:
        self.kline_root = Path(kline_root)

    def _monthly_zip_path(self, symbol: str, year: int, month: int) -> Path:
        sym = normalize_symbol(symbol)
        return (
            self.kline_root / sym / "monthly" / "1d" / f"{sym}-1d-{year}-{month:02d}.zip"
        )

    def _daily_zip_path(self, symbol: str, day: date) -> Path:
        sym = normalize_symbol(symbol)
        return self.kline_root / sym / "daily" / "1d" / f"{sym}-1d-{day.isoformat()}.zip"

    def load_symbol_daily(
        self,
        symbol: str,
        *,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Load Vision 1d ZIPs on disk (monthly + daily), then clip to [start, end]."""
        sym = normalize_symbol(symbol)
        frames: List[pd.DataFrame] = []
        monthly_dir = self.kline_root / sym / "monthly" / "1d"
        if monthly_dir.is_dir():
            for zp in sorted(monthly_dir.glob(f"{sym}-1d-*.zip")):
                if zp.stat().st_size <= 64:
                    continue
                try:
                    frames.append(parse_kline_zip_bytes(zp.read_bytes()))
                except (OSError, ValueError, zipfile.BadZipFile):
                    continue
        else:
            y0, m0 = start_date.year, start_date.month
            y1, m1 = end_date.year, end_date.month
            for year, month in _month_list(y0, m0, y1, m1):
                zp = self._monthly_zip_path(sym, year, month)
                if zp.is_file() and zp.stat().st_size > 64:
                    try:
                        frames.append(parse_kline_zip_bytes(zp.read_bytes()))
                    except (OSError, ValueError, zipfile.BadZipFile):
                        continue
        cur = start_date
        while cur <= end_date:
            zp = self._daily_zip_path(sym, cur)
            if zp.is_file() and zp.stat().st_size > 64:
                try:
                    frames.append(parse_kline_zip_bytes(zp.read_bytes()))
                except (OSError, ValueError, zipfile.BadZipFile):
                    pass
            cur = date.fromordinal(cur.toordinal() + 1)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        return out.loc[(out.index >= start_ts) & (out.index < end_ts)]
