#!/usr/bin/env python3
"""
Compute semantic score floors for TC/TE/FR/ET from physics_regime parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute semantic score floors (TC/TE/FR/ET)."
    )
    p.add_argument("--physics-regime", required=True, help="physics_regime parquet")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--tc-quantile", type=float, default=0.05)
    p.add_argument("--te-quantile", type=float, default=0.10)
    p.add_argument(
        "--fr-quantile", type=float, default=0.05, help="FR semantic score quantile"
    )
    p.add_argument(
        "--et-quantile", type=float, default=0.05, help="ET semantic score quantile"
    )
    args = p.parse_args()

    df = pd.read_parquet(args.physics_regime)
    tc_df = df[df["regime"] == "TC_REGIME"]
    te_df = df[df["regime"] == "TE_REGIME"]
    mean_df = df[df["regime"] == "MEAN_REGIME"]

    tc_score = pd.to_numeric(tc_df.get("tc_semantic_score"), errors="coerce")
    te_score = pd.to_numeric(te_df.get("te_semantic_score"), errors="coerce")
    fr_score = pd.to_numeric(mean_df.get("fr_semantic_score"), errors="coerce")
    et_score = pd.to_numeric(mean_df.get("et_semantic_score"), errors="coerce")

    out = {
        "tc_semantic_score_p05": (
            float(tc_score.quantile(args.tc_quantile))
            if tc_score.notna().any()
            else None
        ),
        "te_semantic_score_p10": (
            float(te_score.quantile(args.te_quantile))
            if te_score.notna().any()
            else None
        ),
        "fr_semantic_score_p05": (
            float(fr_score.quantile(args.fr_quantile))
            if fr_score.notna().any()
            else None
        ),
        "et_semantic_score_p05": (
            float(et_score.quantile(args.et_quantile))
            if et_score.notna().any()
            else None
        ),
        "counts": {
            "tc_rows": int(len(tc_df)),
            "te_rows": int(len(te_df)),
            "mean_rows": int(len(mean_df)),
        },
        "quantiles": {
            "tc": args.tc_quantile,
            "te": args.te_quantile,
            "fr": args.fr_quantile,
            "et": args.et_quantile,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
