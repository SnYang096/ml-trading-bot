#!/usr/bin/env python3
"""Export 1min-path execution-aligned labels for holdout ablation (not default train)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.strategies.labels.execution_realized_r_label import (  # noqa: E402
    compute_realized_r_1min_ablation,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--exec-profile", choices=("g5_tight", "g10_wide_tight"), required=True
    )
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--symbols", default="")
    args = ap.parse_args()

    df = pd.read_parquet(args.features_parquet)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    block = compute_realized_r_1min_ablation(
        df,
        symbols=symbols,
        data_path=args.data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        exec_profile=args.exec_profile,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    block.to_parquet(out, index=True)
    print(f"wrote {out} rows={len(block)} cols={list(block.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
