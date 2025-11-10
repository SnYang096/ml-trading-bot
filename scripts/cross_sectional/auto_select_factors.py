#!/usr/bin/env python3
"""
Automatic factor selection pipeline for cross-sectional panels.

Usage example:
    python scripts/cross_sectional/auto_select_factors.py \
        --input results/feature_exports/cs_panel.parquet \
        --target future_return_12 \
        --min-assets 4 \
        --per-category-top 2 \
        --global-top 12 \
        --ic-threshold 0.01 \
        --ir-threshold 0.5 \
        --output results/cross_sectional/selected_factors.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from cross_sectional.factor_catalog import categorize_columns
from cross_sectional.factor_selection import (
    apply_factor_selection,
    compute_cross_sectional_ic,
    filter_panel_by_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically select cross-sectional factors by IC/IR."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Cross-sectional panel parquet/CSV (MultiIndex or columns timestamp,symbol).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target return column (default: auto-detect first 'future_return_' column).",
    )
    parser.add_argument(
        "--min-assets",
        type=int,
        default=4,
        help="Minimum assets per timestamp required to compute IC (default: 4).",
    )
    parser.add_argument(
        "--per-category-top",
        type=int,
        default=2,
        help="Select at most this many factors from each category (default: 2).",
    )
    parser.add_argument(
        "--global-top",
        type=int,
        default=12,
        help="Final global Top-K across all categories (default: 12).",
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=None,
        help="Absolute IC mean threshold (optional).",
    )
    parser.add_argument(
        "--ir-threshold",
        type=float,
        default=None,
        help="Absolute IC IR threshold (optional).",
    )
    parser.add_argument(
        "--ranking-stat",
        choices=["ic", "ir"],
        default="ic",
        help="Statistic used to rank factors when selecting Top-K (default: ic).",
    )
    parser.add_argument(
        "--include-categories",
        nargs="*",
        default=None,
        help="Only consider these categories (default: use all categories).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/cross_sectional/selected_factors.txt",
        help="Output text file for selected factors.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/cross_sectional/selection_summary.json",
        help="Output JSON file for metrics/diagnostics.",
    )
    return parser.parse_args()


def load_panel(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(file)
    if file.suffix.lower() == ".parquet":
        df = pd.read_parquet(file)
    else:
        df = pd.read_csv(file)

    if not isinstance(df.index, pd.MultiIndex):
        if {"timestamp", "symbol"}.issubset(df.columns):
            df = df.set_index(["timestamp", "symbol"])
        else:
            raise ValueError(
                "Input must either have MultiIndex (timestamp, symbol) or columns 'timestamp' & 'symbol'."
            )

    # Ensure timestamp index is datetime64[ns, UTC]
    ts = pd.to_datetime(df.index.get_level_values(0), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("NaT detected in timestamp index.")
    df.index = pd.MultiIndex.from_arrays(
        [ts, df.index.get_level_values(1)], names=["timestamp", "symbol"]
    )
    return df


def detect_target(panel: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in panel.columns:
            raise ValueError(f"Target column '{explicit}' not found.")
        return explicit
    candidates = [c for c in panel.columns if c.startswith("future_return")]
    if not candidates:
        raise ValueError("Unable to detect target column; please specify with --target.")
    return candidates[0]


def select_from_category(
    panel: pd.DataFrame,
    factors: List[str],
    target_col: str,
    *,
    min_assets: int,
    top_k: int,
    ic_threshold: Optional[float],
    ir_threshold: Optional[float],
    ranking_stat: str,
) -> Dict[str, object]:
    if not factors:
        return {"selected": [], "metrics": {}}

    metrics = compute_cross_sectional_ic(
        panel,
        factors,
        target_col,
        min_assets=min_assets,
    )
    selected = apply_factor_selection(
        metrics,
        factors,
        select_topk=top_k,
        ic_threshold=ic_threshold,
        ir_threshold=ir_threshold,
        ranking_stat=ranking_stat,
    )
    summary = metrics.loc[selected].to_dict(orient="index") if selected else {}
    return {"selected": selected, "metrics": summary}


def aggregate_selection(
    panel: pd.DataFrame,
    initial_selection: List[str],
    target_col: str,
    *,
    min_assets: int,
    global_top: int,
    ic_threshold: Optional[float],
    ir_threshold: Optional[float],
    ranking_stat: str,
) -> Dict[str, object]:
    metrics = compute_cross_sectional_ic(
        panel,
        initial_selection,
        target_col,
        min_assets=min_assets,
    )
    final_selected = apply_factor_selection(
        metrics,
        initial_selection,
        select_topk=global_top,
        ic_threshold=ic_threshold,
        ir_threshold=ir_threshold,
        ranking_stat=ranking_stat,
    )
    summary = metrics.loc[final_selected].to_dict(orient="index") if final_selected else {}
    return {"selected": final_selected, "metrics": summary}


def main() -> None:
    args = parse_args()
    panel = load_panel(args.input)
    target_col = detect_target(panel, args.target)

    # Filter by minimum assets upfront
    panel = filter_panel_by_assets(panel, min_assets=args.min_assets)

    categories = categorize_columns(panel.columns)
    if args.include_categories:
        include = set(args.include_categories)
        categories = {k: v for k, v in categories.items() if k in include}

    print("📂 Factor categories detected:")
    for name, cols in categories.items():
        print(f"  - {name}: {len(cols)} factors")

    per_category_top = max(0, args.per_category_top)
    if per_category_top == 0:
        raise ValueError("--per-category-top must be > 0")
    global_top = max(0, args.global_top)
    if global_top == 0:
        raise ValueError("--global-top must be > 0")

    category_results: Dict[str, Dict[str, object]] = {}
    selected_so_far: List[str] = []

    for name, cols in categories.items():
        print(f"\n🔎 Selecting from category '{name}' (candidates: {len(cols)})")
        result = select_from_category(
            panel,
            list(cols),
            target_col,
            min_assets=args.min_assets,
            top_k=per_category_top,
            ic_threshold=args.ic_threshold,
            ir_threshold=args.ir_threshold,
            ranking_stat=args.ranking_stat,
        )
        selected = result["selected"]
        category_results[name] = result
        if selected:
            print(f"   ✅ Selected {len(selected)} factors: {selected}")
            selected_so_far.extend(selected)
        else:
            print("   ⚠️  No factors passed thresholds in this category.")

    if not selected_so_far:
        raise RuntimeError("No factors selected from any category.")

    print("\n📊 Aggregating selections across categories...")
    aggregate = aggregate_selection(
        panel,
        selected_so_far,
        target_col,
        min_assets=args.min_assets,
        global_top=args.global_top,
        ic_threshold=args.ic_threshold,
        ir_threshold=args.ir_threshold,
        ranking_stat=args.ranking_stat,
    )
    final_factors = aggregate["selected"]
    if not final_factors:
        raise RuntimeError("Global selection removed all factors; try relaxing thresholds.")

    print(f"✅ Final factor list ({len(final_factors)}): {final_factors}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(final_factors), encoding="utf-8")
    print(f"📝 Saved selected factors to {out_path}")

    diagnostics = {
        "input": args.input,
        "target": target_col,
        "min_assets": args.min_assets,
        "per_category_top": args.per_category_top,
        "global_top": args.global_top,
        "ic_threshold": args.ic_threshold,
        "ir_threshold": args.ir_threshold,
        "ranking_stat": args.ranking_stat,
        "categories": {
            name: {
                "selected": result["selected"],
                "metrics": result["metrics"],
            }
            for name, result in category_results.items()
        },
        "final_selection": {
            "factors": final_factors,
            "metrics": aggregate["metrics"],
        },
    }

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"📊 Detailed diagnostics saved to {json_path}")


if __name__ == "__main__":
    main()

