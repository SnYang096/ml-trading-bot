#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute regime score floors from mode_3action file."
    )
    ap.add_argument("--mode", required=True, help="mode_3action parquet")
    ap.add_argument(
        "--q", type=float, default=0.05, help="Quantile for floor (e.g., 0.05)"
    )
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    df = pd.read_parquet(args.mode)
    if "regime" not in df.columns:
        raise ValueError("mode file must include 'regime' column")
    for col in ["tc_score", "te_score", "mean_score"]:
        if col not in df.columns:
            raise ValueError(f"mode file missing score column: {col}")

    floors = {}
    q = float(args.q)
    for regime, col, key in [
        ("TC", "tc_score", "tc_score_floor"),
        ("TE", "te_score", "te_score_floor"),
        ("MEAN", "mean_score", "mean_score_floor"),
    ]:
        s = pd.to_numeric(
            df.loc[df["regime"].astype(str) == regime, col], errors="coerce"
        ).dropna()
        floors[key] = float(s.quantile(q)) if not s.empty else None

    Path(args.out).write_text(json.dumps(floors, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
