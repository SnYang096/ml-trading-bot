#!/usr/bin/env python3
"""Merge long/short tree score exports into dual-head event inject parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _pick_score_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for cand in ("pred", "score"):
        if cand in df.columns:
            return cand
    raise ValueError(f"no score column in parquet (wanted {preferred})")


_OHLC_COLS = ("open", "high", "low", "close", "volume", "atr", "signal")


def _prep_side_frame(
    df: pd.DataFrame,
    *,
    score_name: str,
    preferred_col: str,
    keep_ohlc: bool = False,
) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("missing timestamp column")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "_symbol" in out.columns:
        out["symbol"] = out["_symbol"]
    elif "symbol" not in out.columns:
        raise ValueError("missing symbol column")
    score_col = _pick_score_col(out, preferred_col)
    cols = ["symbol", "timestamp", score_col]
    if "split" in out.columns:
        cols.append("split")
    if keep_ohlc:
        cols.extend(c for c in _OHLC_COLS if c in out.columns and c not in cols)
    out = out[cols].rename(columns={score_col: score_name})
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out


def merge_side_scores(
    *,
    long_parquet: Path,
    short_parquet: Path,
    output: Path,
    long_col: str = "pred",
    short_col: str = "pred",
    keep_ohlc: bool = False,
) -> Path:
    left = _prep_side_frame(
        pd.read_parquet(long_parquet),
        score_name="score_long",
        preferred_col=long_col,
        keep_ohlc=keep_ohlc,
    )
    right = _prep_side_frame(
        pd.read_parquet(short_parquet),
        score_name="score_short",
        preferred_col=short_col,
        keep_ohlc=False,
    )
    merged = left.merge(right, on=["symbol", "timestamp"], how="inner")
    merged["symbol"] = merged["symbol"].astype(str).str.upper()
    merged["_symbol"] = merged["symbol"]
    merged = merged.sort_values(["symbol", "timestamp"]).drop_duplicates(
        ["symbol", "timestamp"], keep="last"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    summary = {
        "n_rows": int(len(merged)),
        "score_long_mean": float(merged["score_long"].mean()),
        "score_short_mean": float(merged["score_short"].mean()),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-parquet", required=True)
    ap.add_argument("--short-parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--keep-ohlc",
        action="store_true",
        help="Retain OHLC/atr from long parquet for holdout τ-scan vectorbt",
    )
    args = ap.parse_args()
    merge_side_scores(
        long_parquet=Path(args.long_parquet),
        short_parquet=Path(args.short_parquet),
        output=Path(args.output),
        keep_ohlc=bool(args.keep_ohlc),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
