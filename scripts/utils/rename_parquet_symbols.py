#!/usr/bin/env python3
"""
Rename existing parquet files from legacy dash format (e.g. BTC-USD_2020-01.parquet)
to the unified symbol format (e.g. BTCUSDT_2020-01.parquet).

Usage:
    python scripts/utils/rename_parquet_symbols.py \
        --input-dir data/parquet_data \
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

SYMBOL_MAPPINGS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT",
    "XRP-USD": "XRPUSDT",
    "ADA-USD": "ADAUSDT",
    "DOGE-USD": "DOGEUSDT",
    "DOT-USD": "DOTUSDT",
    "MATIC-USD": "MATICUSDT",
    "SHIB-USD": "SHIBUSDT",
    "ATOM-USD": "ATOMUSDT",
    "AVAX-USD": "AVAXUSDT",
    "LTC-USD": "LTCUSDT",
    "LINK-USD": "LINKUSDT",
    "FIL-USD": "FILUSDT",
    "ICP-USD": "ICPUSDT",
    "APT-USD": "APTUSDT",
    "ARB-USD": "ARBUSDT",
    "OP-USD": "OPUSDT",
    "SUI-USD": "SUIUSDT",
}


def infer_symbol(stem: str) -> tuple[str | None, str | None]:
    """
    Return (legacy_symbol, date_part) for filenames like BTC-USD_2020-01.
    """
    match = re.match(r"(?P<symbol>[A-Z\-]+)_(?P<date>\d{4}-\d{2})$", stem)
    if not match:
        return None, None
    return match.group("symbol"), match.group("date")


def normalise_symbol(symbol: str) -> str:
    if symbol in SYMBOL_MAPPINGS:
        return SYMBOL_MAPPINGS[symbol]
    # remove hyphen/slash and append USDT if missing
    cleaned = symbol.replace("-", "").replace("/", "")
    if not cleaned.endswith("USDT"):
        cleaned = f"{cleaned}USDT"
    return cleaned


def rename_files(input_dir: Path,
                 dry_run: bool = False) -> list[tuple[Path, Path]]:
    renamed: list[tuple[Path, Path]] = []
    for file_path in input_dir.glob("*.parquet"):
        legacy_symbol, date_part = infer_symbol(file_path.stem)
        if not legacy_symbol or not date_part:
            continue

        new_symbol = normalise_symbol(legacy_symbol)
        if new_symbol == legacy_symbol.replace("-", ""):
            # nothing to do
            continue

        new_name = f"{new_symbol}_{date_part}.parquet"
        target_path = file_path.with_name(new_name)
        if target_path.exists():
            # Already renamed
            continue

        print(
            f"{'[DRY]' if dry_run else 'RENAME'} {file_path.name} -> {new_name}"
        )
        if not dry_run:
            file_path.rename(target_path)
        renamed.append((file_path, target_path))
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename legacy parquet symbol naming.")
    parser.add_argument("--input-dir",
                        type=str,
                        default="data/parquet_data",
                        help="Directory containing parquet files.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Show changes without renaming files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    renamed = rename_files(input_dir, dry_run=args.dry_run)
    if not renamed:
        print("No files needed renaming.")
    else:
        print(f"Renamed {len(renamed)} files.")


if __name__ == "__main__":
    main()
