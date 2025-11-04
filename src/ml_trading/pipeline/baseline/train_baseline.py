from __future__ import annotations
"""
Baseline single-run training using SR + compression features only.

Usage:
  python -m ml_trading.pipeline.baseline.train_baseline \
    --data /home/yin/trading/ml_trading_bot/data/parquet_data/BTC-USD_2024-05.parquet \
    --symbol BTCUSDT --freq 5T --start 2024-01 --end 2024-12 --gpu

Optional:
  - multiple files: pass --data multiple times or a directory via --data-dir
  - label horizon: --forward-bars 3
"""

import os
import argparse
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from ml_trading.data_tools.rolling_data import (
    load_parquet_file, )
from ml_trading.data_tools.baseline_feature_engineering import (
    engineer_baseline_features,
    get_baseline_feature_columns,
    create_binary_labels_baseline,
)
from ml_trading.models.lightgbm_model import LightGBMModel


def _load_many(files: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for f in files:
        df = load_parquet_file(f) if f.endswith(".parquet") else None
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid data files loaded")
    return pd.concat(frames, axis=0).sort_index()


def _collect_files(data: List[str],
                   data_dir: str | None,
                   start: str | None,
                   end: str | None,
                   symbol: str | None = None) -> List[str]:
    files: List[str] = []
    files.extend(data)
    if data_dir and os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".parquet"):
                files.append(os.path.join(data_dir, name))
    files = [os.path.abspath(p) for p in files if os.path.exists(p)]

    # Filter by symbol if provided (to avoid mixing different symbols)
    if symbol:
        # Symbol mapping: BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD, etc.
        symbol_mapping = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
            "BNBUSDT": "BNB-USD",
            "ADAUSDT": "ADA-USD",
            "SOLUSDT": "SOL-USD",
        }
        file_symbol = symbol_mapping.get(symbol,
                                         symbol.replace("USDT", "-USD"))

        # Filter files by symbol prefix
        filtered = []
        for p in files:
            filename = os.path.basename(p).upper()
            # Check if filename starts with symbol patterns
            if (filename.startswith(symbol.upper())
                    or filename.startswith(file_symbol.upper())
                    or filename.startswith(
                        file_symbol.replace("-", "_").upper())):
                filtered.append(p)
        files = filtered

    # Optional filter by YYYY-MM in filename
    if start or end:

        def _ym_from_name(n: str) -> str | None:
            import re
            m = re.search(r"(20\d{2})[-_](\d{2})", os.path.basename(n))
            return f"{m.group(1)}-{m.group(2)}" if m else None

        filtered = []
        for p in files:
            ym = _ym_from_name(p)
            if ym is None:
                continue
            if start and ym < start:
                continue
            if end and ym > end:
                continue
            filtered.append(p)
        files = filtered
    if not files:
        raise FileNotFoundError("No parquet files found from inputs")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline training with SR+compression features")
    parser.add_argument("--data",
                        type=str,
                        action="append",
                        default=[],
                        help="Parquet file(s) to use")
    parser.add_argument("--data-dir",
                        type=str,
                        default=None,
                        help="Directory containing parquet files")
    parser.add_argument("--symbol",
                        type=str,
                        default="BTCUSDT",
                        help="Symbol metadata for report")
    parser.add_argument(
        "--freq",
        type=str,
        default="5T",
        help="Bar timeframe(s), comma-separated: 5T,15T (or single: 5T)")
    parser.add_argument("--start",
                        type=str,
                        default=None,
                        help="Start YYYY-MM (inclusive)")
    parser.add_argument("--end",
                        type=str,
                        default=None,
                        help="End YYYY-MM (inclusive)")
    parser.add_argument(
        "--forward-bars",
        type=str,
        default="3",
        help=
        "Bars ahead for label creation, comma-separated: 1,5,10,15 (or single: 3)"
    )
    parser.add_argument("--cv-folds",
                        type=int,
                        default=0,
                        help="TimeSeries CV folds (0=disable)")
    parser.add_argument("--gpu",
                        action="store_true",
                        default=True,
                        help="Use GPU for LightGBM")
    args = parser.parse_args()

    # Parse multiple frequencies and forward bars
    freqs = [f.strip() for f in args.freq.split(",") if f.strip()]
    forward_bars_list = [
        int(fb.strip()) for fb in args.forward_bars.split(",") if fb.strip()
    ]

    print(f"\n" + "=" * 80)
    print(f"🧱 Baseline Training: {args.symbol}")
    print(f"   Timeframes: {freqs}")
    print(f"   Forward Bars: {forward_bars_list}")
    print("=" * 80)

    files = _collect_files(args.data,
                           args.data_dir,
                           args.start,
                           args.end,
                           symbol=args.symbol)
    print(f"\n📦 Loading {len(files)} parquet file(s) for {args.symbol}...")
    raw = _load_many(files)
    print(f"   ✓ Loaded {len(raw):,} bars")

    # Iterate over all combinations of freq and forward_bars
    for freq in freqs:
        for forward_bars in forward_bars_list:
            print(f"\n" + "-" * 80)
            print(f"🎯 Training: tf={freq}, fb={forward_bars}")
            print("-" * 80)

            # Resample data if needed (for now, use the same data for all timeframes)
            # In a real scenario, you might want to resample raw data to different timeframes
            feat_df = raw.copy()

            print("🧪 Engineering baseline features...")
            feat_df, baseline_engineer = engineer_baseline_features(feat_df,
                                                                    None,
                                                                    fit=True)
            print(f"   ✓ Features ready: {feat_df.shape}")

            print(f"🏷️  Creating labels (forward_bars={forward_bars})...")
            # Stage1: Binary classification labels (1=Long, 0=not Long)
            feat_df = create_binary_labels_baseline(feat_df,
                                                    forward_bars=forward_bars)
            # Stage2: Regression target (future return)
            feat_df["future_return"] = feat_df["close"].shift(
                -forward_bars) / feat_df["close"] - 1
            feat_df = feat_df.dropna()
            print(f"   ✓ Samples: {len(feat_df):,}")

            # Diagnostic: Check label distribution and data quality
            label_dist = feat_df["binary_signal"].value_counts()
            print(f"   📊 Label distribution: {dict(label_dist)}")
            print(
                f"      Long (1): {label_dist.get(1, 0)} ({100*label_dist.get(1, 0)/len(feat_df):.2f}%)"
            )
            print(
                f"      Not Long (0): {label_dist.get(0, 0)} ({100*label_dist.get(0, 0)/len(feat_df):.2f}%)"
            )
            future_ret_mean = feat_df["future_return"].mean()
            future_ret_std = feat_df["future_return"].std()
            print(
                f"   📈 Future return stats: mean={future_ret_mean:.6f}, std={future_ret_std:.6f}"
            )

            # Warn if label distribution is too imbalanced
            if label_dist.get(1, 0) / len(feat_df) < 0.1 or label_dist.get(
                    1, 0) / len(feat_df) > 0.9:
                print(
                    f"   ⚠️  Warning: Label distribution is highly imbalanced! This may affect model performance."
                )

            # Check for data leaks: ensure no forward-looking features
            feature_cols = get_baseline_feature_columns(feat_df)
            suspicious_cols = [
                c for c in feature_cols if 'future' in c.lower()
                or 'forward' in c.lower() or 'next' in c.lower()
            ]
            if suspicious_cols:
                print(
                    f"   ⚠️  Warning: Found suspicious feature columns that might contain future data: {suspicious_cols}"
                )

            feature_cols = get_baseline_feature_columns(feat_df)
            X = feat_df[feature_cols].values
            y_stage1 = feat_df[
                "binary_signal"].values  # Binary classification: 1=Long, 0=not Long
            y_stage2 = feat_df[
                "future_return"].values  # Regression: future return

            # Stage1: Binary classification (same as train command)
            print(
                "🎯 Stage1: Training LightGBM (binary classification, baseline features only)..."
            )
            model_stage1 = LightGBMModel(model_type="classification",
                                         use_gpu=args.gpu)
            # Override to use binary classification instead of multiclass
            model_stage1.params["objective"] = "binary"
            model_stage1.params["metric"] = "binary_logloss"
            if "num_class" in model_stage1.params:
                del model_stage1.params["num_class"]

            # Use TimeSeriesSplit CV (same as train command)
            n_splits = args.cv_folds if args.cv_folds > 0 else 5
            stage1_metrics = model_stage1.train(pd.DataFrame(
                X, columns=feature_cols),
                                                pd.Series(y_stage1),
                                                n_splits=n_splits,
                                                use_time_series_cv=True)

            cv_accuracy_mean = stage1_metrics.get("cv_accuracy", None)
            cv_accuracy_std = stage1_metrics.get("cv_accuracy_std", None)
            stage1_fold_details = stage1_metrics.get("fold_details", [])

            print("   ✓ Stage1 model trained")

            # Diagnostic: Warn if CV accuracy is suspiciously high
            if cv_accuracy_mean is not None and cv_accuracy_mean > 0.9:
                print(
                    f"   ⚠️  Warning: CV accuracy ({cv_accuracy_mean:.4f}) is very high (>0.9)."
                )
                print(f"      This may indicate:")
                print(
                    f"      1. Strong trends in the data (model is learning trends, not patterns)"
                )
                print(
                    f"      2. Label distribution imbalance (check label distribution above)"
                )
                print(
                    f"      3. Data leakage (features using future data) - check feature engineering"
                )
                print(f"      4. Overfitting (model memorizing training data)")
                print(
                    f"      Consider using out-of-sample testing or adjusting hyperparameters."
                )

            # Stage2: Regression (same as train command)
            print(
                "🎯 Stage2: Training LightGBM (regression, baseline features only)..."
            )
            model_stage2 = LightGBMModel(model_type="regression",
                                         use_gpu=args.gpu)

            stage2_metrics = model_stage2.train(pd.DataFrame(
                X, columns=feature_cols),
                                                pd.Series(y_stage2),
                                                n_splits=n_splits,
                                                use_time_series_cv=True)

            cv_rmse_mean = stage2_metrics.get("cv_rmse", None)
            cv_mse_mean = stage2_metrics.get("cv_mse", None)
            cv_mse_std = stage2_metrics.get("cv_mse_std", None)
            stage2_fold_details = stage2_metrics.get("fold_details", [])

            print("   ✓ Stage2 model trained")

            # Save models and generate training report similar to train command
            # Create output directory for this combination
            combo_dir = "results/baseline"
            if len(freqs) > 1 or len(forward_bars_list) > 1:
                combo_dir = os.path.join("results/baseline",
                                         f"fb{forward_bars}_tf{freq}")
            os.makedirs(combo_dir, exist_ok=True)

            model_stage1_path = os.path.join(combo_dir,
                                             "baseline_stage1_model.txt")
            model_stage2_path = os.path.join(combo_dir,
                                             "baseline_stage2_model.txt")
            model_stage1.model.save_model(
                model_stage1_path)  # model.model is the lgb.Booster object
            model_stage2.model.save_model(model_stage2_path)

            # Save scalers (fitted quantiles) for consistent train/test transformation
            scaler_path = os.path.join(combo_dir, "baseline_scalers.pkl")
            baseline_engineer.save_scalers(scaler_path)

            with open(os.path.join(combo_dir, "baseline_features.txt"),
                      "w") as f:
                f.write("\n".join(feature_cols))

            # Generate training info JSON and HTML report similar to train command
            from datetime import datetime
            import json
            actual_start = feat_df.index.min() if not feat_df.empty else None
            actual_end = feat_df.index.max() if not feat_df.empty else None

            info_path = os.path.join(combo_dir, "baseline_training_info.json")
            model_info = {
                "model_path":
                model_stage1_path,  # Keep stage1 as primary for backward compatibility
                "scaler_path":
                scaler_path,
                "training_date":
                datetime.now().isoformat(),
                "symbol":
                args.symbol,
                "actual_start":
                actual_start.isoformat() if actual_start else None,
                "actual_end":
                actual_end.isoformat() if actual_end else None,
                "total_bars":
                len(feat_df),
                "timeframes": {
                    freq: len(feat_df)
                },
                "price_range": [
                    float(feat_df["close"].min()) if not feat_df.empty else 0,
                    float(feat_df["close"].max()) if not feat_df.empty else 0,
                ],
                "metrics": {
                    "stage1": {
                        freq: {
                            "cv_accuracy": cv_accuracy_mean,
                            "cv_accuracy_std": cv_accuracy_std,
                            "fold_details": stage1_fold_details,
                        }
                    },
                    "stage2": {
                        freq: {
                            "cv_rmse": cv_rmse_mean,
                            "cv_mse": cv_mse_mean,
                            "cv_mse_std": cv_mse_std,
                            "fold_details": stage2_fold_details,
                        }
                    }
                },
                "feature_engineering":
                "BaselineFeatureEngineer",
                "forward_bars":
                forward_bars,
                "timeframe":
                freq,
                "data_files":
                files,
            }

            with open(info_path, "w") as f:
                json.dump(model_info, f, indent=2, default=str)

            # Generate HTML report
            try:
                from ml_trading.pipeline.dimensionality.report_generator import write_training_report
                report_path = os.path.join(combo_dir,
                                           "baseline_training_report.html")
                write_training_report(str(info_path), report_path)
                print(f"   ✓ Training report saved to {report_path}")
            except Exception as exc:
                print(f"   ⚠️  Failed to generate HTML report: {exc}")

            print(f"💾 Saved model and feature list to {combo_dir}/")

    print(f"\n" + "=" * 80)
    print(f"✅ Baseline training completed for all combinations")
    print("=" * 80)


if __name__ == "__main__":
    main()
