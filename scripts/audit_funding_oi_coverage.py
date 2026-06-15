#!/usr/bin/env python3
"""Audit local OI + funding-rate parquet coverage for T5 / feature pipeline.

Usage:
    python scripts/audit_funding_oi_coverage.py
    python scripts/audit_funding_oi_coverage.py --symbols BTCUSDT,ETHUSDT,HYPEUSDT
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FUNDING_DIR = REPO / "data" / "funding_rate" / "parquet"
OI_DIR = REPO / "data" / "open_interest" / "parquet"


def _load_range(
    files: List[Path], value_col: str
) -> Optional[Tuple[int, pd.Timestamp, pd.Timestamp]]:
    if not files:
        return None
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df.index = pd.to_datetime(df["datetime"], utc=True)
            else:
                continue
        if value_col in df.columns:
            parts.append(df[[value_col]])
        else:
            parts.append(df.iloc[:, :1])
    if not parts:
        return None
    all_df = pd.concat(parts).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    return len(all_df), all_df.index.min(), all_df.index.max()


def audit_symbol(sym: str) -> dict:
    sym = sym.strip().upper()
    funding_files = sorted(FUNDING_DIR.glob(f"{sym}_*_funding_rate.parquet"))
    oi_files = sorted(OI_DIR.glob(f"{sym}_*_oi_5m.parquet"))
    fr = _load_range(funding_files, "funding_rate")
    oi = _load_range(oi_files, "oi_usd")
    return {
        "symbol": sym,
        "funding_files": len(funding_files),
        "funding_rows": fr[0] if fr else 0,
        "funding_start": fr[1] if fr else None,
        "funding_end": fr[2] if fr else None,
        "oi_files": len(oi_files),
        "oi_rows": oi[0] if oi else 0,
        "oi_start": oi[1] if oi else None,
        "oi_end": oi[2] if oi else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Audit funding + OI parquet coverage")
    p.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,HYPEUSDT",
    )
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"funding_dir: {FUNDING_DIR}")
    print(f"oi_dir:      {OI_DIR}")
    print()
    print(
        f"{'symbol':<10} {'fr_files':>8} {'fr_end':>28} {'oi_files':>8} {'oi_end':>28} status"
    )
    print("-" * 90)
    missing = 0
    for sym in symbols:
        r = audit_symbol(sym)
        fr_end = str(r["funding_end"])[:19] if r["funding_end"] else "-"
        oi_end = str(r["oi_end"])[:19] if r["oi_end"] else "-"
        if r["funding_files"] == 0 or r["oi_files"] == 0:
            status = "MISSING"
            missing += 1
        else:
            status = "ok"
        print(
            f"{r['symbol']:<10} {r['funding_files']:>8} {fr_end:>28} "
            f"{r['oi_files']:>8} {oi_end:>28} {status}"
        )
    print()
    if missing:
        print(
            f"⚠ {missing} symbol(s) missing data — run scripts/refresh_funding_oi_data.py"
        )
        return 1
    print("✓ all symbols have funding + OI parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
