from __future__ import annotations

"""
Baseline single-run training using SR + compression features only.

Usage:
  python -m ml_trading.pipeline.baseline.train_baseline \
    --data /home/yin/trading/ml_trading_bot/data/parquet_data/BTC-USD_2024-05.parquet \
    --gpu

Optional:
  - multiple files: pass --data multiple times or a directory via --data-dir
  - label horizon: --forward-bars 3
"""

import os
import argparse
from typing import List
import pandas as pd

from ml_trading.data_tools.rolling_data import (
    load_parquet_file,
    create_labels,
)
from ml_trading.data_tools.baseline_feature_engineering import (
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from ml_trading.utils.training import (
    train_lightgbm_model,
    simple_backtest,
    print_backtest_results,
)


def _load_many(files: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for f in files:
        df = load_parquet_file(f) if f.endswith(".parquet") else None
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid data files loaded")
    return pd.concat(frames, axis=0).sort_index()


def _collect_files(data: List[str], data_dir: str | None) -> List[str]:
    files: List[str] = []
    files.extend(data)
    if data_dir and os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".parquet"):
                files.append(os.path.join(data_dir, name))
    files = [os.path.abspath(p) for p in files if os.path.exists(p)]
    if not files:
        raise FileNotFoundError("No parquet files found from inputs")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline training with SR+compression features")
    parser.add_argument("--data", type=str, action="append", default=[], help="Parquet file(s) to use")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing parquet files")
    parser.add_argument("--forward-bars", type=int, default=3, help="Bars ahead for label creation")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU for LightGBM")
    args = parser.parse_args()

    files = _collect_files(args.data, args.data_dir)
    print(f"📦 Loading {len(files)} parquet file(s)...")
    raw = _load_many(files)
    print(f"   ✓ Loaded {len(raw):,} bars")

    print("🧪 Engineering baseline features...")
    feat_df, _ = engineer_baseline_features(raw, None, fit=True)
    print(f"   ✓ Features ready: {feat_df.shape}")

    print(f"🏷️  Creating labels (forward_bars={args.forward_bars})...")
    feat_df = create_labels(feat_df, forward_bars=args.forward_bars)
    feat_df = feat_df.dropna()
    print(f"   ✓ Samples: {len(feat_df):,}")

    feature_cols = get_baseline_feature_columns(feat_df)
    X = feat_df[feature_cols].values
    y = feat_df["signal"].values

    print("🎯 Training LightGBM (baseline features only)...")
    model = train_lightgbm_model(X, y, use_gpu=args.gpu)
    print("   ✓ Model trained")

    print("🔮 Generating in-sample predictions (for quick sanity backtest)...")
    preds = model.predict(X)
    results = simple_backtest(feat_df, preds)
    print_backtest_results(results, label="In-sample Baseline Backtest")

    # Save model and columns next to results directory for convenience
    os.makedirs("results/baseline", exist_ok=True)
    model_path = os.path.join("results/baseline", "baseline_model.txt")
    model.save_model(model_path)
    with open(os.path.join("results/baseline", "baseline_features.txt"), "w") as f:
        f.write("\n".join(feature_cols))
    print(f"💾 Saved model and feature list to results/baseline/")


if __name__ == "__main__":
    main()


