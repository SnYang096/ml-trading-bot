#!/usr/bin/env python3
"""Export tree holdout predictions → event_backtest score injection parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def export_scores(
    predictions: Path,
    output: Path,
    *,
    symbols: list[str] | None = None,
    split: str | None = "holdout",
    score_col: str = "pred",
    start_date: str | None = None,
    end_date: str | None = None,
    extra_cols: list[str] | None = None,
) -> Path:
    df = pd.read_parquet(predictions)
    if "timestamp" not in df.columns:
        raise ValueError(f"missing timestamp in {predictions}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    sym_col = "_symbol" if "_symbol" in df.columns else "symbol"
    if sym_col not in df.columns:
        raise ValueError(f"missing symbol column in {predictions}")
    if score_col not in df.columns:
        raise ValueError(f"missing {score_col} in {predictions}")
    if split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()].copy()
    if symbols:
        syms = {s.strip().upper() for s in symbols}
        df = df[df[sym_col].astype(str).str.upper().isin(syms)].copy()
    out = pd.DataFrame(
        {
            "symbol": df[sym_col].astype(str).str.upper(),
            "timestamp": pd.to_datetime(df["timestamp"], utc=True),
            "score": df[score_col].astype(float),
        }
    )
    # Carry gate / overlay feature columns so the event-time gate can read them
    # from the injected score parquet (aligned per symbol/timestamp).
    for col in extra_cols or []:
        if col in df.columns and col not in out.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    out = out.sort_values(["symbol", "timestamp"]).drop_duplicates(
        ["symbol", "timestamp"], keep="last"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    print(f"Wrote {output} rows={len(out)} symbols={sorted(out['symbol'].unique())}")
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--symbols", default=None, help="Comma-separated filter")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--score-col", default="pred")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument(
        "--extra-cols",
        default=None,
        help="Comma-separated extra feature columns to carry into the inject parquet",
    )
    args = ap.parse_args()
    syms = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )
    export_scores(
        Path(args.predictions),
        Path(args.output),
        symbols=syms,
        split=args.split or None,
        score_col=args.score_col,
        start_date=args.start_date,
        end_date=args.end_date,
        extra_cols=(
            [c.strip() for c in args.extra_cols.split(",") if c.strip()]
            if args.extra_cols
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
