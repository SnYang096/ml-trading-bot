#!/usr/bin/env python3
"""
Check month coverage for Binance monthly aggTrades datasets (ZIP + Parquet).

Examples:
  python scripts/check_month_coverage.py --symbol BNBUSDT --start 2023-01 --end 2025-11
  python scripts/check_month_coverage.py --symbol BNBUSDT --start 2023-01 --end 2025-11 --show-missing
  # All symbols summary:
  python scripts/check_month_coverage.py --start 2023-01 --end 2025-11
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def _parse_ym(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{2})", s.strip())
    if not m:
        raise ValueError(f"Invalid YYYY-MM: {s!r}")
    y = int(m.group(1))
    mo = int(m.group(2))
    if mo < 1 or mo > 12:
        raise ValueError(f"Invalid month in YYYY-MM: {s!r}")
    return y, mo


def _month_range(start: Tuple[int, int], end: Tuple[int, int]) -> List[str]:
    (y, m) = start
    (ey, em) = end
    out: List[str] = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def _months_in_parquet(parquet_dir: Path, symbol: str) -> Set[str]:
    out: Set[str] = set()
    for p in parquet_dir.glob(f"{symbol}_????-??.parquet"):
        m = re.search(r"_(\d{4}-\d{2})\.parquet$", p.name)
        if m:
            out.add(m.group(1))
    return out


def _months_in_zip(zip_dir: Path, symbol: str) -> Set[str]:
    out: Set[str] = set()
    for z in zip_dir.glob(f"{symbol}-aggTrades-????-??.zip"):
        m = re.search(r"-(\d{4}-\d{2})\.zip$", z.name)
        if m:
            out.add(m.group(1))
    return out


def _list_symbols(zip_dir: Path, parquet_dir: Path) -> List[str]:
    syms: Set[str] = set()

    # Parquet: SYMBOL_YYYY-MM.parquet
    for p in parquet_dir.glob("*_????-??.parquet"):
        m = re.match(r"^([A-Z0-9]+)_\d{4}-\d{2}\.parquet$", p.name.upper())
        if m:
            syms.add(m.group(1))

    # ZIP: SYMBOL-aggTrades-YYYY-MM.zip
    for z in zip_dir.glob("*-aggTrades-????-??.zip"):
        m = re.match(r"^([A-Z0-9]+)-AGGTRADES-\d{4}-\d{2}\.ZIP$", z.name.upper())
        if m:
            syms.add(m.group(1))

    return sorted(syms)


@dataclass(frozen=True)
class Coverage:
    target: List[str]
    zip_months: Set[str]
    parquet_months: Set[str]

    def present_zip(self) -> List[str]:
        return [m for m in self.target if m in self.zip_months]

    def present_parquet(self) -> List[str]:
        return [m for m in self.target if m in self.parquet_months]

    def missing_zip(self) -> List[str]:
        return [m for m in self.target if m not in self.zip_months]

    def missing_parquet(self) -> List[str]:
        return [m for m in self.target if m not in self.parquet_months]

    def zip_not_parquet(self) -> List[str]:
        return [
            m
            for m in self.target
            if (m in self.zip_months and m not in self.parquet_months)
        ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check YYYY-MM coverage for ZIP + Parquet datasets."
    )
    ap.add_argument(
        "--symbol",
        default=None,
        help="Symbol like BNBUSDT (optional). If omitted, show all symbols summary.",
    )
    ap.add_argument("--start", required=True, help="Start YYYY-MM (inclusive)")
    ap.add_argument("--end", required=True, help="End YYYY-MM (inclusive)")
    ap.add_argument(
        "--zip-dir",
        default="data/agg_data",
        help="ZIP directory (default: data/agg_data)",
    )
    ap.add_argument(
        "--parquet-dir",
        default="data/parquet_data",
        help="Parquet directory (default: data/parquet_data)",
    )
    ap.add_argument(
        "--show-missing", action="store_true", help="Print missing month lists"
    )
    args = ap.parse_args()

    start = _parse_ym(args.start)
    end = _parse_ym(args.end)
    target = _month_range(start, end)

    zip_dir = Path(args.zip_dir)
    parquet_dir = Path(args.parquet_dir)

    if args.symbol:
        symbol = str(args.symbol).strip().upper().replace("-", "").replace("/", "")
        cov = Coverage(
            target=target,
            zip_months=_months_in_zip(zip_dir, symbol),
            parquet_months=_months_in_parquet(parquet_dir, symbol),
        )

        zip_in = cov.present_zip()
        parquet_in = cov.present_parquet()

        print(f"symbol: {symbol}")
        print(f"range:  {target[0]} .. {target[-1]} ({len(target)} months)")
        print()
        print(f"ZIP:     {len(zip_in)}/{len(target)} months present in {zip_dir}")
        print(
            f"Parquet: {len(parquet_in)}/{len(target)} months present in {parquet_dir}"
        )
        print(f"ZIP but no Parquet: {len(cov.zip_not_parquet())} months")

        if args.show_missing:
            print()
            print("missing ZIP:", ", ".join(cov.missing_zip()) or "(none)")
            print("missing Parquet:", ", ".join(cov.missing_parquet()) or "(none)")
            print(
                "ZIP present but Parquet missing:",
                ", ".join(cov.zip_not_parquet()) or "(none)",
            )
    else:
        symbols = _list_symbols(zip_dir=zip_dir, parquet_dir=parquet_dir)
        print(f"range: {target[0]} .. {target[-1]} ({len(target)} months)")
        print(f"symbols detected: {len(symbols)}")
        print()
        print("symbol\tzip\tparquet\tzip_no_parquet")
        for symbol in symbols:
            cov = Coverage(
                target=target,
                zip_months=_months_in_zip(zip_dir, symbol),
                parquet_months=_months_in_parquet(parquet_dir, symbol),
            )
            z = len(cov.present_zip())
            p = len(cov.present_parquet())
            znp = len(cov.zip_not_parquet())
            print(f"{symbol}\t{z}/{len(target)}\t{p}/{len(target)}\t{znp}")

            if args.show_missing and (
                cov.missing_zip() or cov.missing_parquet() or cov.zip_not_parquet()
            ):
                print(f"  missing ZIP: {', '.join(cov.missing_zip()) or '(none)'}")
                print(
                    f"  missing Parquet: {', '.join(cov.missing_parquet()) or '(none)'}"
                )
                print(
                    f"  ZIP present but Parquet missing: {', '.join(cov.zip_not_parquet()) or '(none)'}"
                )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Allow piping to tools like `head` without stack traces.
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)
