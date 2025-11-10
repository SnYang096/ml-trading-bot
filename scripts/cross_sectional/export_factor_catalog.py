#!/usr/bin/env python3
"""
Export cross-sectional factor sets grouped by category.

Example:
    python scripts/cross_sectional/export_factor_catalog.py \
        --input results/feature_exports/cs_panel.parquet \
        --output-dir results/cross_sectional/factor_sets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from cross_sectional.factor_catalog import (
    categorize_columns,
    format_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group cross-sectional factors by heuristic categories."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cross-sectional panel parquet/CSV (MultiIndex with timestamp & symbol).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_sectional/factor_sets",
        help="Directory to store categorised factor lists.",
    )
    parser.add_argument(
        "--include-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write combined JSON catalogue alongside text files.",
    )
    parser.add_argument(
        "--override-exclude",
        nargs="*",
        default=None,
        help="Additional column names to exclude from categorisation.",
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
                "Input data must either have MultiIndex (timestamp, symbol) or columns 'timestamp' & 'symbol'."
            )
    return df


def write_factor_sets(
    categories: dict[str, list[str]],
    output_dir: Path,
    *,
    write_json: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, cols in categories.items():
        path = output_dir / f"{name}.txt"
        path.write_text("\n".join(cols), encoding="utf-8")
    if write_json:
        import json

        json_path = output_dir / "catalogue.json"
        json_path.write_text(json.dumps(categories, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = load_panel(args.input)

    categories = categorize_columns(
        df.columns,
        exclude_columns=args.override_exclude,
    )

    output_dir = Path(args.output_dir)
    write_factor_sets(categories, output_dir, write_json=args.include_json)

    print(format_summary(categories))
    print(f"✅ Factor sets saved under {output_dir}")


if __name__ == "__main__":
    main()

