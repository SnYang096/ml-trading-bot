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

from cross_sectional import (
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
        type=str,
        default="auto",
        help="Annualisation factor (e.g., 252 for daily, 17520 for 5-minute). Use 'auto' to infer from data.",
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
        "--feature-file",
        type=str,
        default=None,
        help="Path to text file containing feature names (one per line). Overrides --feature-cols.",
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
        raise FileNotFoundError(
            f"No input files match the provided patterns: {patterns}"
        )
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
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
                df = df.reset_index()
            elif isinstance(df.index, pd.DatetimeIndex):
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
    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"], utc=True, errors="coerce"
    )
    combined = combined.dropna(subset=["timestamp", "symbol"])
    combined = combined.sort_values(["timestamp", "symbol"])
    combined = combined.set_index(["timestamp", "symbol"])
    combined["timestamp"] = combined.index.get_level_values("timestamp")
    combined["symbol"] = combined.index.get_level_values("symbol")
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
    symbol_list = [
        s.strip().upper() for s in symbols.replace(" ", ",").split(",") if s.strip()
    ]
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        mask = df.index.get_level_values("symbol").str.upper().isin(symbol_list)
        filtered = df[mask].copy()
        return filtered
    if "symbol" in df.columns:
        return df[df["symbol"].str.upper().isin(symbol_list)].copy()
    raise ValueError("Symbol filtering requested but 'symbol' column/level not found.")


def ensure_future_return_column(
    df: pd.DataFrame, horizon: int, price_col: str = "close"
) -> tuple[pd.DataFrame, str]:
    col_name = f"future_return_{horizon}"
    if col_name in df.columns:
        return df, col_name
    if price_col not in df.columns:
        raise ValueError(f"{price_col} column missing; cannot compute forward returns.")
    df_sorted = df.sort_values(["symbol", "timestamp"]).copy()
    df_sorted[col_name] = df_sorted.groupby("symbol")[price_col].apply(
        lambda x: x.shift(-horizon) / x - 1.0
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

    if isinstance(filtered_df.index, pd.MultiIndex):
        cols_to_drop = [
            col for col in ["timestamp", "symbol"] if col in filtered_df.columns
        ]
        if cols_to_drop:
            filtered_df = filtered_df.drop(columns=cols_to_drop)
        filtered_df = filtered_df.reset_index()

    filtered_df, target_col = ensure_future_return_column(filtered_df, args.horizon)
    feature_cols = None
    if args.feature_file:
        feature_path = Path(args.feature_file)
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        feature_cols = [
            line.strip()
            for line in feature_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        feature_cols = list(dict.fromkeys(feature_cols))
        print(f"   📄 Loaded {len(feature_cols)} features from {feature_path}")
    elif args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    if feature_cols:
        available = set(filtered_df.columns)
        missing = [c for c in feature_cols if c not in available]
        if missing:
            print(
                f"   ⚠️  Warning: {len(missing)} requested features not found in data: {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}"
            )
        feature_cols = [c for c in feature_cols if c in available]
        if not feature_cols:
            raise ValueError(
                "No valid features remaining after filtering against dataframe columns."
            )

    symbol_count = filtered_df["symbol"].nunique()
    min_assets_required = 3
    if symbol_count < min_assets_required:
        min_assets_required = max(1, symbol_count)

    print("   📥 Assembling panel...")
    panel, detected_features = build_panel(
        filtered_df,
        target_col=target_col,
        feature_cols=feature_cols,
        min_assets=min_assets_required,
        horizon=args.horizon,
        dropna_after_fill=not args.skip_na_drop,
    )
    factor_cols = list(feature_cols) if feature_cols else list(detected_features)
    print(f"   📦 Initial factor count: {len(factor_cols)}")

    if args.crypto_factors:
        print("   🔧 Adding crypto-specific factors...")
        panel = add_crypto_cross_sectional_factors(panel)
        crypto_cols = [
            col
            for col in panel.columns
            if col.startswith("cs_crypto_") and col != target_col
        ]
        factor_cols = list(dict.fromkeys(factor_cols + crypto_cols))
        print(f"   📦 Factor count after crypto enrichment: {len(factor_cols)}")

    processed_panel = preprocess_panel(panel, factor_cols, args.winsor, args.zscore)
    periods_per_year = resolve_periods_per_year(
        args.periods_per_year, processed_panel.index
    )
    print(
        f"   📊 Panel ready: {processed_panel.shape[0]} observations, "
        f"{len(factor_cols)} factors, periods_per_year={periods_per_year:.2f}"
    )

    model = CrossSectionalRegressor(add_intercept=True, min_assets=3)
    result = model.fit(processed_panel, factor_cols=factor_cols, target_col=target_col)

    diagnostics = FactorPanelBuilder.describe_panel(processed_panel)

    context = ReportContext(
        title="Cross-Sectional Factor Efficacy Report",
        max_lag=args.max_lag,
        periods_per_year=periods_per_year,
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


def resolve_periods_per_year(arg_value: str, index: pd.Index) -> float:
    value = (arg_value or "auto").strip().lower()
    if value != "auto":
        try:
            parsed = float(value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    timestamps = index
    if isinstance(timestamps, pd.MultiIndex):
        timestamps = timestamps.get_level_values(0)
    timestamps = pd.to_datetime(timestamps)
    timestamps = timestamps.sort_values().unique()
    if len(timestamps) < 2:
        return 252.0

    diffs_series = pd.Series(timestamps)
    diffs = diffs_series.diff().dropna()
    if diffs.empty:
        return 252.0
    if diffs.nunique() > 1:
        raise ValueError(
            "Detected multiple bar intervals in panel; please provide a single timeframe per run."
        )

    median_seconds = diffs.dt.total_seconds().iloc[0]
    if not median_seconds or median_seconds <= 0:
        return 252.0

    seconds_per_year = 365.0 * 24.0 * 3600.0
    inferred = seconds_per_year / median_seconds
    return float(inferred)


if __name__ == "__main__":
    main()
