#!/usr/bin/env python3
"""
Pre-train hyperparameter search for Q50 constraint compliance.

This script searches for optimal LightGBM parameters that satisfy:
    Q50 loss <= max(Q10, Q90) loss

The found parameters are saved to a JSON file and can be reused in training.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from ml_trading.models.lightgbm_model import LightGBMModel
from ml_trading.data_tools.baseline_feature_engineering import (
    BaselineFeatureEngineer,
    get_baseline_feature_columns,
)
from ml_trading.pipeline.training.preprocessing import clean_features_train_test


def load_data(
    data_dir: str,
    symbols: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: str = "5T",
    max_files: int = 10,
) -> pd.DataFrame:
    """Load and prepare data for parameter tuning."""
    from ml_trading.pipeline.training.train import _collect_files, _resample_ohlcv

    symbol_list = [s.strip() for s in symbols.split(",")]
    files = _collect_files([], data_dir, start_date, end_date, ",".join(symbol_list))

    if not files:
        raise ValueError(f"No data files found for symbols={symbols}")

    print(f"📁 Found {len(files)} data files")
    frames = []
    for file_path in files[:max_files]:
        df_raw = pd.read_parquet(file_path)
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            if "timestamp" in df_raw.columns:
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
                df_raw = df_raw.set_index("timestamp")
            else:
                raise ValueError(f"File {file_path} has no timestamp column or DatetimeIndex")
        
        df = _resample_ohlcv(df_raw, freq)
        if "symbol" not in df.columns:
            # Infer symbol from filename
            symbol = Path(file_path).stem.split("_")[0].split("-")[0]
            df["symbol"] = symbol
        frames.append(df)

    df = pd.concat(frames, axis=0).sort_index()
    print(f"📊 Loaded {len(df)} samples from {len(frames)} files")
    return df


def prepare_features_and_target(
    df: pd.DataFrame, forward_bars: int = 5
) -> tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """Engineer features and build target."""
    engineer = BaselineFeatureEngineer()
    feat_df = engineer.engineer_features(df, fit=True)
    feature_cols = get_baseline_feature_columns(feat_df)

    # Clean features (basic cleaning)
    feat_df_clean = clean_features_train_test(
        feat_df[feature_cols], feat_df[feature_cols], k=4.0
    )[0]

    # Build future return
    future_return = (
        feat_df["close"].shift(-forward_bars) / feat_df["close"] - 1
    ).rename("future_return")

    aligned = feat_df_clean.join(future_return, how="inner").dropna()

    if len(aligned) < 1000:
        raise ValueError(f"Insufficient aligned samples: {len(aligned)}")

    X = aligned[feature_cols]
    y = aligned["future_return"]

    groups = None
    if "symbol" in aligned.columns:
        groups = aligned["symbol"].values

    print(f"✅ Prepared {len(X)} samples with {len(feature_cols)} features")
    return X, y, groups


def search_parameters(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray],
    n_trials: int = 50,
    n_splits: int = 3,
    output_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Search for optimal Q50 parameters."""
    print(f"\n🔍 Starting Q50 parameter search ({n_trials} trials, {n_splits} CV folds)...")

    model = LightGBMModel(model_type="quantile", quantile_alpha=0.5)
    best_params = model.optimize_hyperparameters_for_q50_constraint(
        X, y, n_trials=n_trials, n_splits=n_splits, groups=groups
    )

    if best_params is None:
        print("❌ No valid parameters found that satisfy Q50 constraint")
        return None

    print(f"\n✅ Found optimal parameters:")
    for key, value in sorted(best_params.items()):
        print(f"   {key}: {value}")

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n💾 Saved parameters to: {output_path}")

    return best_params


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train hyperparameter search for Q50 constraint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing parquet data files",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Symbol(s), comma-separated (e.g., BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM (inclusive)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="5T",
        help="Bar timeframe (default: 5T)",
    )
    parser.add_argument(
        "--forward-bars",
        type=int,
        default=5,
        help="Forward bars for prediction (default: 5)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of CV splits (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results/params/q50_params_{symbol}_{freq}_{forward_bars}.json)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of data files to load (default: 10)",
    )

    args = parser.parse_args()

    # Generate default output path if not provided
    if args.output is None:
        symbol_tag = args.symbol.replace(",", "_")
        freq_tag = args.freq.replace("T", "min")
        output_path = f"results/params/q50_params_{symbol_tag}_{freq_tag}_{args.forward_bars}bars.json"
    else:
        output_path = args.output

    print("=" * 80)
    print("Q50 Parameter Search (Pre-training)")
    print("=" * 80)
    print(f"Symbols: {args.symbol}")
    print(f"Timeframe: {args.freq}")
    print(f"Forward Bars: {args.forward_bars}")
    print(f"Trials: {args.n_trials}, CV Folds: {args.n_splits}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Load data
    df = load_data(
        args.data_dir,
        args.symbol,
        args.start,
        args.end,
        args.freq,
        args.max_files,
    )

    # Prepare features and target
    X, y, groups = prepare_features_and_target(df, args.forward_bars)

    # Search parameters
    best_params = search_parameters(
        X, y, groups, args.n_trials, args.n_splits, output_path
    )

    if best_params:
        print("\n" + "=" * 80)
        print("✅ Parameter search completed successfully!")
        print(f"💾 Parameters saved to: {output_path}")
        print("\n💡 Usage in training:")
        print(f"   make train PARAMS_FILE={output_path} AUTO_TUNE=0")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ Parameter search failed - no valid parameters found")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())

