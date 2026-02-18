#!/usr/bin/env python3
"""
Download historical Open Interest data from Binance Data Vision (S3).

Data source: https://data.binance.vision/
Path pattern: data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YYYY-MM-DD}.zip
Each ZIP contains a CSV with columns:
  create_time, symbol, sum_open_interest, sum_open_interest_value,
  count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
  count_long_short_ratio, sum_taker_long_short_vol_ratio

The CSV rows are at 5-minute intervals.

Output: Monthly parquet files compatible with the existing OI feature pipeline.
Format: data/open_interest/parquet/{SYMBOL}_{YYYY}-{MM}_oi_5m.parquet
Schema: DatetimeIndex('datetime', UTC), columns: oi_contracts, oi_usd, _symbol

Usage:
  python scripts/download_oi_from_data_vision.py \
    --symbols BTCUSDT ETHUSDT \
    --start-date 2023-01-01 \
    --parquet-dir data/open_interest/parquet

  # With universe config:
  python scripts/download_oi_from_data_vision.py \
    --universe-config config/download/crypto_4h_token_universe_groups.yaml \
    --universe-set starter_a \
    --start-date 2023-01-01
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
PERIOD = "5m"  # Data Vision metrics are always 5-min granularity


def _normalize_symbol(s: str) -> str:
    s = s.strip().upper().replace("-", "").replace("/", "")
    if not s.endswith("USDT"):
        s += "USDT"
    return s


def _date_range(start: date, end: date) -> List[date]:
    """Inclusive date range."""
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def _parquet_path(parquet_dir: Path, symbol: str, year: int, month: int) -> Path:
    return parquet_dir / f"{symbol}_{year}-{month:02d}_oi_{PERIOD}.parquet"


# ─────────────────────────────────────────────────────────────
# Download + parse one day
# ─────────────────────────────────────────────────────────────


def download_day(
    session: requests.Session,
    symbol: str,
    d: date,
    *,
    timeout: int = 30,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Download one day's metrics ZIP and return DataFrame with OI columns."""
    url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{d.isoformat()}.zip"

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None  # data not available for this date
            if resp.status_code == 403:
                return None  # forbidden / not exists
            if resp.status_code == 429:
                wait = min(2 ** (attempt + 2), 60)
                time.sleep(wait)
                continue
            resp.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)

            if df.empty:
                return None

            # Parse
            df["datetime"] = pd.to_datetime(df["create_time"], utc=True)
            df["oi_contracts"] = pd.to_numeric(df["sum_open_interest"], errors="coerce")
            df["oi_usd"] = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
            df["_symbol"] = symbol

            out = df[["datetime", "_symbol", "oi_contracts", "oi_usd"]].copy()
            out = out.sort_values("datetime").drop_duplicates(
                subset=["datetime"], keep="last"
            )
            out = out.set_index("datetime")
            return out

        except (requests.RequestException, zipfile.BadZipFile, KeyError):
            time.sleep(min(2**attempt, 10))
            continue

    return None


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def run_download(
    *,
    symbols: List[str],
    start_date: date,
    end_date: date,
    parquet_dir: Path,
    sleep_sec: float = 0.1,
    progress_every: int = 50,
    force: bool = False,
) -> None:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for sym_raw in symbols:
        sym = _normalize_symbol(sym_raw)
        print(f"\n{'='*60}")
        print(f"  {sym}: downloading OI from {start_date} to {end_date}")
        print(f"{'='*60}")

        # Collect data by month
        all_days = _date_range(start_date, end_date)

        # Group days by (year, month)
        month_days: dict[tuple[int, int], list[date]] = {}
        for d in all_days:
            key = (d.year, d.month)
            month_days.setdefault(key, []).append(d)

        for (year, month), days in sorted(month_days.items()):
            ppath = _parquet_path(parquet_dir, sym, year, month)
            if not force and ppath.exists() and ppath.stat().st_size > 0:
                print(f"  ⏩ skip {sym} {year}-{month:02d} (cached)")
                continue

            # Download all days for this month
            month_frames: list[pd.DataFrame] = []
            ok_days = 0
            empty_days = 0

            for i, d in enumerate(days):
                df = download_day(session, sym, d)
                if df is not None and not df.empty:
                    month_frames.append(df)
                    ok_days += 1
                else:
                    empty_days += 1

                if sleep_sec > 0:
                    time.sleep(sleep_sec)

                if progress_every and ((i + 1) % progress_every == 0):
                    print(
                        f"    [{i+1}/{len(days)}] {sym} {year}-{month:02d} "
                        f"ok={ok_days} empty={empty_days}"
                    )

            if not month_frames:
                print(f"  ⚪ empty {sym} {year}-{month:02d} (no data)")
                continue

            combined = pd.concat(month_frames)
            combined = combined.sort_index().loc[
                ~combined.index.duplicated(keep="last")
            ]

            ppath.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(ppath)
            print(
                f"  ✅ {sym} {year}-{month:02d}  "
                f"rows={len(combined)}  days={ok_days}/{len(days)}"
            )

    print("\n✅ All OI downloads complete.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download OI from Binance Data Vision (S3 metrics ZIP files)"
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols like BTCUSDT ETHUSDT",
    )
    ap.add_argument(
        "--universe-config",
        default=None,
        help="YAML universe config (overrides --symbols)",
    )
    ap.add_argument("--universe-set", default="starter_a")
    ap.add_argument("--universe-groups", default=None)
    ap.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--end-date",
        default=None,
        help="End date (YYYY-MM-DD). Default: yesterday.",
    )
    ap.add_argument(
        "--parquet-dir",
        default="data/open_interest/parquet",
        help="Output directory",
    )
    ap.add_argument(
        "--sleep-sec",
        type=float,
        default=0.1,
        help="Sleep between downloads (s)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N days (0 disables)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-download cached months",
    )
    args = ap.parse_args()

    # Resolve symbols
    symbols = args.symbols
    if args.universe_config:
        sys.path.insert(0, ".")
        from src.data_tools.universe_config import load_universe_config

        cfg = load_universe_config(args.universe_config)
        groups = (
            [g.strip() for g in str(args.universe_groups).split(",") if g.strip()]
            if args.universe_groups
            else None
        )
        symbols = cfg.resolve_symbols_usdt(
            universe_set=str(args.universe_set), groups=groups
        )

    if not symbols:
        print("ERROR: No symbols specified. Use --symbols or --universe-config.")
        sys.exit(1)

    start = date.fromisoformat(args.start_date)
    end = (
        date.fromisoformat(args.end_date)
        if args.end_date
        else (datetime.now(tz=timezone.utc) - timedelta(days=1)).date()
    )

    run_download(
        symbols=symbols,
        start_date=start,
        end_date=end,
        parquet_dir=Path(args.parquet_dir),
        sleep_sec=args.sleep_sec,
        progress_every=args.progress_every,
        force=args.force,
    )


if __name__ == "__main__":
    main()
