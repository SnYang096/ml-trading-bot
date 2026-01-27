#!/usr/bin/env python3
"""
Analyze near-miss FBF samples vs the rest of the labeled dataset.

Outputs:
  - JSON: feature deltas and summary stats
  - Markdown: human-readable report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


EXCLUDE_COLS = {
    "execution_label",
    "semantic_label",
    "fbf_label",
    "label_consistency",
    "structure_match",
    "execution_match",
    "direction_match",
    "execution_dir",
    "semantic_dir",
    "exit_reason",
    "holding_bars",
}


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _stats_for_col(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "std": 0.0}
    return {
        "mean": float(s.mean()),
        "p10": float(s.quantile(0.1)),
        "p50": float(s.quantile(0.5)),
        "p90": float(s.quantile(0.9)),
        "std": float(s.std()),
    }


def _feature_deltas(
    near_df: pd.DataFrame, other_df: pd.DataFrame, cols: List[str]
) -> List[Dict[str, object]]:
    out = []
    for col in cols:
        near_stats = _stats_for_col(near_df[col])
        other_stats = _stats_for_col(other_df[col])
        pooled_std = float(
            np.nanmean([near_stats.get("std", 0.0), other_stats.get("std", 0.0)])
        )
        z_delta = 0.0
        if pooled_std and pooled_std > 0:
            z_delta = (near_stats["mean"] - other_stats["mean"]) / pooled_std
        out.append(
            {
                "feature": col,
                "near": near_stats,
                "other": other_stats,
                "mean_delta": float(near_stats["mean"] - other_stats["mean"]),
                "z_delta": float(z_delta),
            }
        )
    return out


def _top_features(
    deltas: List[Dict[str, object]], top_n: int
) -> List[Dict[str, object]]:
    return sorted(deltas, key=lambda x: abs(x.get("z_delta", 0.0)), reverse=True)[
        :top_n
    ]


def _write_markdown(
    *,
    out_path: Path,
    summary: Dict[str, object],
    top_features: List[Dict[str, object]],
) -> None:
    lines = [
        "# FBF Near-Miss Analysis",
        "",
        "## Summary",
        "",
        f"- near_miss_rows: {summary['near_miss_rows']}",
        f"- other_rows: {summary['other_rows']}",
        f"- total_rows: {summary['total_rows']}",
        "",
        "## Top Feature Deltas (by |z_delta|)",
        "",
        "| feature | near_mean | other_mean | mean_delta | z_delta |",
        "|---|---|---|---|---|",
    ]
    for item in top_features:
        lines.append(
            "| {feature} | {near_mean:.6f} | {other_mean:.6f} | {mean_delta:.6f} | {z_delta:.3f} |".format(
                feature=item["feature"],
                near_mean=item["near"]["mean"],
                other_mean=item["other"]["mean"],
                mean_delta=item["mean_delta"],
                z_delta=item["z_delta"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze near-miss samples")
    parser.add_argument("--labels", required=True, help="labeled parquet")
    parser.add_argument("--near-miss", required=True, help="near-miss parquet")
    parser.add_argument("--out-json", required=True, help="output json")
    parser.add_argument("--out-md", required=True, help="output markdown")
    parser.add_argument("--top-n", type=int, default=20, help="top features to report")
    args = parser.parse_args()

    labels_df = pd.read_parquet(args.labels)
    near_df = pd.read_parquet(args.near_miss)
    if near_df.empty:
        raise ValueError("near-miss dataset is empty")

    # Align columns
    common_cols = [c for c in near_df.columns if c in labels_df.columns]
    labels_df = labels_df[common_cols]
    near_df = near_df[common_cols]

    other_df = labels_df.drop(index=near_df.index, errors="ignore")
    numeric_cols = _numeric_columns(labels_df)

    deltas = _feature_deltas(near_df, other_df, numeric_cols)
    top_features = _top_features(deltas, args.top_n)

    summary = {
        "near_miss_rows": int(len(near_df)),
        "other_rows": int(len(other_df)),
        "total_rows": int(len(labels_df)),
        "top_features": top_features,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(out_path=out_md, summary=summary, top_features=top_features)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
