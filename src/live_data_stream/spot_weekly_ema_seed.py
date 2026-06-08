"""Binance Vision spot daily klines → weekly EMA200 seed for live macro features."""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Shared with CMS weekly overlay: reject stale / flat-line seed (~412) vs live spot.
STALE_WEEKLY_SEED_LAG = pd.Timedelta(days=21)
SEED_VS_SPOT_MAX_REL_GAP = 0.15

VISION_SPOT_MONTHLY = "https://data.binance.vision/data/spot/monthly/klines"
VISION_SPOT_DAILY = "https://data.binance.vision/data/spot/daily/klines"

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


def _vision_open_time_unit(open_time: pd.Series) -> str:
    """Binance Vision spot klines: ms (13 digits) through 2024, us (16 digits) from 2025."""
    sample = pd.to_numeric(open_time, errors="coerce").dropna()
    if sample.empty:
        return "ms"
    # 1e14 ms ≈ year 5138; real 2025+ microsecond stamps are ~1.7e15.
    return "us" if float(sample.iloc[0]) > 1e14 else "ms"


def _parse_kline_zip_bytes(raw: bytes) -> pd.DataFrame:
    """Parse one Binance Vision kline ZIP into daily OHLC indexed by UTC open time."""
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
    unit = _vision_open_time_unit(df["open_time"])
    ts = pd.to_datetime(df["open_time"], unit=unit, utc=True, errors="coerce")
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
    out = out[~out.index.duplicated(keep="last")]
    return out


def compute_weekly_ema_table(
    daily_close: pd.Series,
    *,
    ema_span_weeks: int = 200,
) -> pd.DataFrame:
    """Resample daily close to W-SUN and compute weekly EMA."""
    c = pd.to_numeric(daily_close, errors="coerce").astype(float).dropna()
    if c.empty:
        return pd.DataFrame(columns=["week_ts", "weekly_close", "weekly_ema_200"])
    weekly = (
        pd.DataFrame({"close": c})
        .sort_index()
        .resample("W-SUN", label="right", closed="right")
        .agg({"close": "last"})
        .dropna(subset=["close"])
    )
    span = max(2, int(ema_span_weeks))
    # EMA200 needs ~200 weekly closes; span//5 allowed biased values (~400) on short history.
    min_periods = span
    weekly["weekly_ema_200"] = weekly["close"].ewm(
        span=span, adjust=False, min_periods=min_periods
    ).mean()
    weekly = weekly.rename(columns={"close": "weekly_close"})
    weekly.index.name = "week_ts"
    weekly = weekly.reset_index()
    return weekly


def weekly_ema_position_series(
    *,
    close: pd.Series,
    weekly_ema: pd.Series,
) -> pd.Series:
    """(current close - ffilled weekly EMA) / current close, clipped [-1, 1]."""
    idx = close.index
    c = pd.to_numeric(close, errors="coerce").astype(float)
    ema_on_bar = pd.to_numeric(weekly_ema, errors="coerce").astype(float).reindex(
        idx, method="ffill"
    )
    close_safe = c.replace(0, np.nan)
    out = ((c - ema_on_bar) / close_safe).replace([np.inf, -np.inf], np.nan)
    # Keep NaN when EMA unknown; do not masquerade missing macro history as 0.0.
    valid_ema = ema_on_bar.notna()
    out = out.where(valid_ema).clip(-1.0, 1.0)
    return out


def seed_parquet_path(seed_root: Path, symbol: str) -> Path:
    sym = normalize_symbol(symbol)
    return Path(seed_root) / f"{sym}.parquet"


def macro_seed_ready(
    symbol: str,
    seed_root: Path | str,
    *,
    min_valid_ema_rows: int = 1,
) -> bool:
    """True when seed parquet exists and has at least one valid weekly EMA row."""
    df = load_weekly_ema_seed(seed_root, symbol)
    if df is None or df.empty or "weekly_ema_200" not in df.columns:
        return False
    return int(df["weekly_ema_200"].notna().sum()) >= int(min_valid_ema_rows)


def macro_seeds_ready(
    symbols: Sequence[str],
    seed_root: Path | str,
    *,
    min_valid_ema_rows: int = 1,
) -> tuple[bool, list[str]]:
    """Return (all_ready, missing_symbols)."""
    missing: list[str] = []
    for raw in symbols:
        sym = normalize_symbol(raw)
        if not macro_seed_ready(sym, seed_root, min_valid_ema_rows=min_valid_ema_rows):
            missing.append(sym)
    return (len(missing) == 0, missing)


def seed_last_timestamp(seed: pd.DataFrame) -> Optional[pd.Timestamp]:
    if seed is None or seed.empty:
        return None
    if isinstance(seed.index, pd.DatetimeIndex):
        ts = seed.index.max()
    elif "week_ts" in seed.columns:
        ts = pd.to_datetime(seed["week_ts"], utc=True, errors="coerce").max()
    else:
        return None
    if ts is None or pd.isna(ts):
        return None
    out = pd.Timestamp(ts)
    if out.tz is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def seed_is_stale_for_bar(
    seed: pd.DataFrame,
    bar_ts: pd.Timestamp,
    *,
    max_lag: pd.Timedelta = STALE_WEEKLY_SEED_LAG,
) -> bool:
    """True when seed's last week is too old vs the decision bar (stale flat-line bug)."""
    last = seed_last_timestamp(seed)
    if last is None:
        return True
    ts = pd.Timestamp(bar_ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return last < ts - max_lag


def seed_ema_plausible_vs_close(
    ema: float,
    close: float,
    *,
    max_rel_gap: float = SEED_VS_SPOT_MAX_REL_GAP,
) -> bool:
    """Reject stale *low* seed EMA when price is far above it (frozen ~412 vs ~650).

    When close is below EMA (deep bear), a large gap is expected — do not reject.
    """
    if not np.isfinite(ema) or not np.isfinite(close) or close <= 0 or ema <= 0:
        return False
    c = float(close)
    e = float(ema)
    if c <= e:
        return True
    return (c - e) / c <= float(max_rel_gap)


def load_weekly_ema_seed(
    seed_root: Path | str,
    symbol: str,
) -> Optional[pd.DataFrame]:
    path = seed_parquet_path(Path(seed_root), symbol)
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    df = pd.read_parquet(path)
    if df.empty or "weekly_ema_200" not in df.columns:
        return None
    if "week_ts" in df.columns:
        ts = pd.to_datetime(df["week_ts"], utc=True, errors="coerce")
        df = df.set_index(ts).sort_index()
    return df


def weekly_ema_position_from_seed(
    *,
    close: float,
    bar_ts: pd.Timestamp,
    seed_root: Path | str,
    symbol: str,
) -> Optional[float]:
    """Single-bar position from macro seed; None if seed missing or EMA unavailable."""
    seed = load_weekly_ema_seed(seed_root, symbol)
    if seed is None or seed.empty:
        return None
    ts = pd.Timestamp(bar_ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if seed_is_stale_for_bar(seed, ts):
        logger.warning(
            "weekly EMA seed stale for %s at %s (last week in seed too old)",
            normalize_symbol(symbol),
            ts.isoformat(),
        )
        return None
    ema = seed["weekly_ema_200"].dropna()
    if ema.empty:
        return None
    # Last weekly EMA at or before bar time only (no bfill from future weeks).
    ema_at = ema.reindex(pd.DatetimeIndex([ts]), method="ffill")
    wk_val = float(ema_at.iloc[-1]) if not ema_at.isna().all() else float("nan")
    if not np.isfinite(wk_val) or not np.isfinite(close) or close == 0:
        return None
    if not seed_ema_plausible_vs_close(wk_val, float(close)):
        logger.warning(
            "weekly EMA seed implausible for %s: ema=%.2f close=%.2f",
            normalize_symbol(symbol),
            wk_val,
            float(close),
        )
        return None
    pos = (float(close) - wk_val) / float(close)
    if not np.isfinite(pos):
        return None
    return float(np.clip(pos, -1.0, 1.0))


@dataclass
class SpotDailyKlineDownloader:
    kline_root: Path
    retries: int = 4
    timeout_sec: int = 120
    sleep_sec: float = 0.15

    def __post_init__(self) -> None:
        self.kline_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _monthly_zip_path(self, symbol: str, year: int, month: int) -> Path:
        sym = normalize_symbol(symbol)
        d = self.kline_root / sym / "monthly" / "1d"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{sym}-1d-{year}-{month:02d}.zip"

    def _daily_zip_path(self, symbol: str, day: date) -> Path:
        sym = normalize_symbol(symbol)
        d = self.kline_root / sym / "daily" / "1d"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{sym}-1d-{day.isoformat()}.zip"

    def _download_url(self, url: str, dest: Path) -> bool:
        if dest.exists() and dest.stat().st_size > 64:
            return True
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, timeout=self.timeout_sec)
                if resp.status_code == 404:
                    return False
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                return True
            except Exception:
                time.sleep(min(2**attempt, 10))
        return False

    def download_monthly(
        self,
        symbol: str,
        year: int,
        month: int,
        *,
        force: bool = False,
    ) -> bool:
        sym = normalize_symbol(symbol)
        dest = self._monthly_zip_path(sym, year, month)
        if not force and dest.exists() and dest.stat().st_size > 64:
            return True
        url = f"{VISION_SPOT_MONTHLY}/{sym}/1d/{sym}-1d-{year}-{month:02d}.zip"
        return self._download_url(url, dest)

    def download_daily(
        self,
        symbol: str,
        day: date,
        *,
        force: bool = False,
    ) -> bool:
        sym = normalize_symbol(symbol)
        dest = self._daily_zip_path(sym, day)
        if not force and dest.exists() and dest.stat().st_size > 64:
            return True
        url = f"{VISION_SPOT_DAILY}/{sym}/1d/{sym}-1d-{day.isoformat()}.zip"
        return self._download_url(url, dest)

    def load_symbol_daily(
        self,
        symbol: str,
        *,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        sym = normalize_symbol(symbol)
        frames: List[pd.DataFrame] = []
        y0, m0 = start_date.year, start_date.month
        y1, m1 = end_date.year, end_date.month
        for year, month in _month_list(y0, m0, y1, m1):
            zp = self._monthly_zip_path(sym, year, month)
            if zp.exists() and zp.stat().st_size > 64:
                try:
                    frames.append(_parse_kline_zip_bytes(zp.read_bytes()))
                except Exception as exc:
                    logger.warning("spot seed: failed monthly %s: %s", zp.name, exc)
        # Recent daily files (current month gaps)
        cur = start_date
        while cur <= end_date:
            zp = self._daily_zip_path(sym, cur)
            if zp.exists() and zp.stat().st_size > 64:
                try:
                    frames.append(_parse_kline_zip_bytes(zp.read_bytes()))
                except Exception as exc:
                    logger.warning("spot seed: failed daily %s: %s", zp.name, exc)
            cur = date.fromordinal(cur.toordinal() + 1)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        return out.loc[(out.index >= start_ts) & (out.index < end_ts)]


def prepare_spot_weekly_ema_seed(
    symbols: Sequence[str],
    *,
    kline_root: Path | str,
    seed_root: Path | str,
    start_date: date = date(2017, 1, 1),
    end_date: Optional[date] = None,
    ema_span_weeks: int = 200,
    force_download: bool = False,
    refresh_recent_days: int = 45,
) -> dict[str, Path]:
    """Download spot 1d klines (idempotent) and write weekly EMA seed parquets."""
    kroot = Path(kline_root)
    sroot = Path(seed_root)
    sroot.mkdir(parents=True, exist_ok=True)
    end = end_date or datetime.now(timezone.utc).date()
    dl = SpotDailyKlineDownloader(kline_root=kroot)

    written: dict[str, Path] = {}
    for raw_sym in symbols:
        sym = normalize_symbol(raw_sym)
        logger.info(
            "spot weekly EMA seed: %s download %s → %s",
            sym,
            start_date,
            end,
        )
        y0, m0 = start_date.year, start_date.month
        y1, m1 = end.year, end.month
        for year, month in _month_list(y0, m0, y1, m1):
            dl.download_monthly(sym, year, month, force=force_download)
        refresh_start = (
            pd.Timestamp(end) - pd.Timedelta(days=refresh_recent_days)
        ).date()
        cur = max(start_date, refresh_start)
        while cur <= end:
            dl.download_daily(sym, cur, force=force_download)
            cur = date.fromordinal(cur.toordinal() + 1)

        daily = dl.load_symbol_daily(sym, start_date=start_date, end_date=end)
        if daily.empty:
            logger.warning("spot weekly EMA seed: no daily klines for %s", sym)
            continue
        weekly = compute_weekly_ema_table(
            daily["close"], ema_span_weeks=ema_span_weeks
        )
        out_path = seed_parquet_path(sroot, sym)
        weekly.to_parquet(out_path, index=False)
        written[sym] = out_path
        valid = int(weekly["weekly_ema_200"].notna().sum())
        logger.info(
            "spot weekly EMA seed: wrote %s rows=%d valid_ema=%d",
            out_path,
            len(weekly),
            valid,
        )
    return written
