#!/usr/bin/env python3
"""
Run Fama-MacBeth cross-sectional regression with Newey-West standard errors,
Information Coefficient diagnostics, and produce a Markdown report.

Example:
    python scripts/cross_sectional/run_famacbeth_report.py \
        --input "results/training/latest/features/*.parquet" \
        --symbols "BTCUSDT,ETHUSDT,SOLUSDT" \
        --horizon 12 \
        --output results/cross_sectional/fama_macbeth_report.md
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ml_trading.cross_sectional import (
    FactorPanelBuilder,
    PanelConfig,
    CrossSectionalRegressor,
    ReportContext,
    cross_sectional_zscore,
    winsorize_by_sigma,
    add_crypto_cross_sectional_factors,
    generate_markdown_report,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional Fama-MacBeth factor evaluation with Newey-West t-stats and IC/IR diagnostics."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input parquet/csv files or glob patterns containing engineered features with `timestamp` and `symbol` columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/cross_sectional/fama_macbeth_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbol list to filter (default: all symbols present).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forward return horizon in bars (used to pick/create `future_return_{horizon}` column).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Newey-West truncation lag.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualisation factor (e.g., 252 for daily, 17520 for 5-minute).",
    )
    parser.add_argument(
        "--winsor",
        type=float,
        default=3.0,
        help="Sigma threshold for cross-sectional winsorisation (set <= 0 to disable).",
    )
    parser.add_argument(
        "--zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply cross-sectional z-score normalisation per timestamp.",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Optional comma-separated feature list. When omitted, auto-detect features.",
    )
    parser.add_argument(
        "--skip-na-drop",
        action="store_true",
        help="Skip dropping timestamps with insufficient assets (retain partial panels).",
    )
    parser.add_argument(
        "--crypto-factors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment panel with built-in crypto cross-sectional factors.",
    )
    return parser.parse_args()


def collect_inputs(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded and Path(pattern).exists():
            expanded = [pattern]
        files.extend(expanded)
    unique = sorted({os.path.abspath(f) for f in files})
    if not unique:
        raise FileNotFoundError(f"No input files match the provided patterns: {patterns}")
    return unique


def load_frames(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")
        if df.empty:
            continue
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' column missing in {path}")
        if "symbol" not in df.columns:
            inferred = _infer_symbol_from_path(path)
            df["symbol"] = inferred
        frames.append(df)
    if not frames:
        raise ValueError("All input frames are empty.")
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    combined = combined.sort_values(["timestamp", "symbol"])
    return combined


def _infer_symbol_from_path(path: str) -> str:
    stem = Path(path).stem.upper()
    if "_" in stem:
        return stem.split("_")[0]
    if "-" in stem:
        return stem.split("-")[0]
    return stem


def filter_symbols(df: pd.DataFrame, symbols: Optional[str]) -> pd.DataFrame:
    if not symbols:
        return df
    symbol_list = [s.strip().upper() for s in symbols.replace(" ", ",").split(",") if s.strip()]
    return df[df["symbol"].str.upper().isin(symbol_list)].copy()


def ensure_future_return_column(
    df: pd.DataFrame, horizon: int, price_col: str = "close"
) -> tuple[pd.DataFrame, str]:
    col_name = f"future_return_{horizon}"
    if col_name in df.columns:
        return df, col_name
    if price_col not in df.columns:
        raise ValueError(f"{price_col} column missing; cannot compute forward returns.")
    df_sorted = df.sort_values(["symbol", "timestamp"]).copy()
    df_sorted[col_name] = (
        df_sorted.groupby("symbol")[price_col].apply(lambda x: x.shift(-horizon) / x - 1.0)
    )
    return df_sorted, col_name


def build_panel(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]],
    min_assets: int,
    horizon: int,
    dropna_after_fill: bool,
) -> tuple[pd.DataFrame, List[str]]:
    config = PanelConfig(
        feature_cols=feature_cols,
        target_col=target_col,
        forward_return_horizon=horizon,
        min_assets_per_ts=min_assets,
        fill_method="ffill",
        dropna_after_fill=dropna_after_fill,
        align_intersection_only=False,
    )
    builder = FactorPanelBuilder(config)
    panel = builder.from_concat_frame(df)
    factor_cols = [c for c in panel.columns if c != target_col]
    return panel, factor_cols


def preprocess_panel(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    winsor_sigma: float,
    apply_zscore: bool,
) -> pd.DataFrame:
    processed = panel.copy()
    if winsor_sigma and winsor_sigma > 0:
        processed = winsorize_by_sigma(processed, factor_cols, sigma=winsor_sigma)
    if apply_zscore:
        processed = cross_sectional_zscore(processed, factor_cols)
    return processed


def main() -> None:
    args = parse_args()
    input_paths = collect_inputs(args.input)
    raw_df = load_frames(input_paths)
    filtered_df = filter_symbols(raw_df, args.symbols)

    if filtered_df.empty:
        raise ValueError("No data available after symbol filtering.")

    filtered_df, target_col = ensure_future_return_column(filtered_df, args.horizon)
    feature_cols = (
        [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        if args.feature_cols
        else None
    )

    panel, detected_features = build_panel(
        filtered_df,
        target_col=target_col,
        feature_cols=feature_cols,
        min_assets=3,
        horizon=args.horizon,
        dropna_after_fill=not args.skip_na_drop,
    )
    factor_cols = list(feature_cols) if feature_cols else list(detected_features)

    if args.crypto_factors:
        panel = add_crypto_cross_sectional_factors(panel)
        crypto_cols = [
            col for col in panel.columns if col.startswith("cs_crypto_") and col != target_col
        ]
        factor_cols = list(dict.fromkeys(factor_cols + crypto_cols))

    processed_panel = preprocess_panel(panel, factor_cols, args.winsor, args.zscore)

    model = CrossSectionalRegressor(add_intercept=True, min_assets=3)
    result = model.fit(processed_panel, factor_cols=factor_cols, target_col=target_col)

    diagnostics = FactorPanelBuilder.describe_panel(processed_panel)

    context = ReportContext(
        title="Cross-Sectional Factor Efficacy Report",
        max_lag=args.max_lag,
        periods_per_year=args.periods_per_year,
        preprocessing=_describe_preprocessing(args.winsor, args.zscore),
        symbols=args.symbols or ", ".join(sorted(filtered_df["symbol"].unique())),
        horizon=args.horizon,
        observations=int(diagnostics.get("num_observations", 0)),
        timestamps=int(diagnostics.get("num_timestamps", 0)),
        assets_per_timestamp=float(diagnostics.get("mean_assets_per_timestamp", 0.0)),
    )
    markdown = generate_markdown_report(result, context)
    write_report(args.output, markdown)

    print(f"✅ Cross-sectional report generated at {args.output}")


def _describe_preprocessing(winsor_sigma: float, apply_zscore: bool) -> str:
    steps = []
    if winsor_sigma and winsor_sigma > 0:
        steps.append(f"winsorize |σ|<{winsor_sigma}")
    if apply_zscore:
        steps.append("z-score")
    if not steps:
        return "none"
    return " + ".join(steps)


if __name__ == "__main__":
    main()

