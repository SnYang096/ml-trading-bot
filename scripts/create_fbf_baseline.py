#!/usr/bin/env python3
"""
Create baselines for FBF label evaluation:
- Random baseline
- Simple-rule baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _label_ratio(df: pd.DataFrame, label_col: str, label_value: str = "FBF") -> float:
    if label_col not in df.columns or len(df) == 0:
        return 0.0
    return float((df[label_col] == label_value).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create FBF baselines")
    parser.add_argument("--labels", required=True, help="labeled parquet")
    parser.add_argument("--out", required=True, help="output json path")
    parser.add_argument(
        "--sample-rate", type=float, default=0.1, help="random baseline sample rate"
    )
    parser.add_argument(
        "--simple-rule-sr-max",
        type=float,
        default=0.3,
        help="sr_distance_normalized max",
    )
    parser.add_argument(
        "--simple-rule-vpin-min", type=float, default=0.6, help="vpin min"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.labels)
    results = {"random_baseline": {}, "simple_rule_baseline": {}}

    # Random baseline
    rng = np.random.default_rng(42)
    sample_mask = rng.random(len(df)) < args.sample_rate
    random_df = df[sample_mask]
    results["random_baseline"] = {
        "sample_rate": args.sample_rate,
        "execution_fbf_ratio": _label_ratio(random_df, "execution_label"),
        "semantic_fbf_ratio": _label_ratio(random_df, "semantic_label"),
        "fbf_ratio": _label_ratio(random_df, "fbf_label"),
    }

    # Simple-rule baseline
    if {"sr_distance_normalized", "vpin"}.issubset(set(df.columns)):
        rule_mask = (df["sr_distance_normalized"] < args.simple_rule_sr_max) & (
            df["vpin"] > args.simple_rule_vpin_min
        )
    else:
        rule_mask = pd.Series(False, index=df.index)
    simple_df = df[rule_mask]
    results["simple_rule_baseline"] = {
        "rule": f"sr_distance_normalized < {args.simple_rule_sr_max} AND vpin > {args.simple_rule_vpin_min}",
        "execution_fbf_ratio": _label_ratio(simple_df, "execution_label"),
        "semantic_fbf_ratio": _label_ratio(simple_df, "semantic_label"),
        "fbf_ratio": _label_ratio(simple_df, "fbf_label"),
        "sample_rate": float(len(simple_df) / max(len(df), 1)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
