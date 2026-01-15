#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColStats:
    total: int = 0
    nan: int = 0
    nonfinite: int = 0
    zeros: int = 0

    def as_dict(self) -> Dict[str, float]:
        total = max(1, self.total)
        return {
            "total": self.total,
            "nan": self.nan,
            "nonfinite": self.nonfinite,
            "zeros": self.zeros,
            "nan_rate": self.nan / total,
            "nonfinite_rate": self.nonfinite / total,
            "zero_rate": self.zeros / total,
        }


def _month_range(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date().replace(day=1)
    e = datetime.strptime(end, "%Y-%m-%d").date().replace(day=1)
    months: List[str] = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        year = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else (cur.month + 1)
        cur = cur.replace(year=year, month=month)
    return months


def _default_excludes() -> set[str]:
    return {"open", "high", "low", "close", "volume", "_symbol", "symbol"}


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)


def _iter_parquet_paths(
    root: Path,
    *,
    layer: str,
    symbols: Iterable[str],
    timeframe: str,
    months: Iterable[str],
) -> Iterable[Path]:
    for sym in symbols:
        for m in months:
            p = root / layer / sym / timeframe / f"{m}.parquet"
            if p.exists():
                yield p


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute missingness stats for a FeatureStore layer (monthly parquet)."
    )
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument("--layer", required=True)
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--exclude-base-cols",
        action="store_true",
        default=True,
        help="Exclude base OHLCV + symbol columns (default: true)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/diagnostics/feature_store_missingness/<timestamp>)",
    )
    args = ap.parse_args()

    root = Path(args.feature_store_root).resolve()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    months = _month_range(args.start_date, args.end_date)
    excludes = _default_excludes() if args.exclude_base_cols else set()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("results/diagnostics/feature_store_missingness")
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, ColStats] = defaultdict(ColStats)
    rows_total = 0
    files_n = 0

    for p in _iter_parquet_paths(
        root, layer=args.layer, symbols=symbols, timeframe=args.timeframe, months=months
    ):
        df = pd.read_parquet(p)
        files_n += 1
        rows_total += len(df)
        for col in df.columns:
            if col in excludes:
                continue
            s = df[col]
            if not _is_numeric_series(s):
                continue
            arr = s.to_numpy()
            n = arr.size
            if n == 0:
                continue
            st = stats[col]
            st.total += int(n)
            nan_mask = np.isnan(arr)
            st.nan += int(nan_mask.sum())
            finite_mask = np.isfinite(arr)
            st.nonfinite += int((~finite_mask).sum())
            st.zeros += int((arr == 0).sum())

    rows = []
    for col, st in stats.items():
        d = {"column": col}
        d.update(st.as_dict())
        rows.append(d)
    df_out = pd.DataFrame(rows).sort_values(
        ["nan_rate", "nonfinite_rate", "zero_rate"], ascending=False
    )

    summary = {
        "feature_store_root": str(root),
        "layer": args.layer,
        "symbols": symbols,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "months": months,
        "files_n": files_n,
        "rows_total": rows_total,
        "columns_n": int(df_out.shape[0]),
        "excluded_base_cols": sorted(excludes),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df_out.to_csv(out_dir / "missingness.csv", index=False)
    print(f"✅ Wrote: {out_dir / 'summary.json'}")
    print(f"✅ Wrote: {out_dir / 'missingness.csv'}")


if __name__ == "__main__":
    main()
