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
    parser.add_argument(
        "--oos-months",
        type=int,
        default=3,
        help=
        "Number of months after training end for OOS testing (default: 3, 0=disabled)"
    )
    parser.add_argument(
        "--oos-start",
        type=str,
        default=None,
        help=
        "OOS test start date (YYYY-MM-DD). If not specified, uses training end + oos-months"
    )
    parser.add_argument(
        "--oos-end",
        type=str,
        default=None,
        help=
        "OOS test end date (YYYY-MM-DD). If not specified, uses oos-start + 3 months"
    )
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

            # Determine OOS test period (before diagnostic to allow OOS data loading)
            from dateutil.relativedelta import relativedelta

            # Training data period
            train_end = feat_df.index.max() if not feat_df.empty else None

            # Determine OOS period
            oos_start_dt = None
            oos_end_dt = None
            oos_df = pd.DataFrame()

            # Check if OOS is enabled
            if args.oos_months > 0 or args.oos_start is not None:
                # Calculate OOS start date
                if args.oos_start:
                    try:
                        oos_start_dt = pd.to_datetime(args.oos_start)
                    except Exception:
                        print(
                            f"   ⚠️  Warning: Invalid --oos-start format: {args.oos_start}. Using default."
                        )
                        oos_start_dt = None

                # If not specified, use training end + oos_months
                if oos_start_dt is None and train_end is not None:
                    oos_start_dt = train_end + relativedelta(
                        months=args.oos_months)

                # Calculate OOS end date
                if args.oos_end:
                    try:
                        oos_end_dt = pd.to_datetime(args.oos_end)
                    except Exception:
                        print(
                            f"   ⚠️  Warning: Invalid --oos-end format: {args.oos_end}. Using default."
                        )
                        oos_end_dt = None

                # If not specified, use oos_start + 3 months
                if oos_end_dt is None and oos_start_dt is not None:
                    oos_end_dt = oos_start_dt + relativedelta(months=3)

                # Load OOS data if needed (may be outside training data range)
                if oos_start_dt is not None and oos_end_dt is not None:
                    # Try to get OOS data from existing data first
                    oos_mask = (feat_df.index >= oos_start_dt) & (
                        feat_df.index <= oos_end_dt)
                    oos_df = feat_df[oos_mask].copy()

                    # If OOS data is not in current dataset, try to load additional files
                    if len(oos_df) == 0:
                        print(
                            f"   ℹ️  OOS period ({oos_start_dt.date()} to {oos_end_dt.date()}) not in training data, trying to load additional files..."
                        )
                        # Load additional files for OOS period
                        oos_start_str = oos_start_dt.strftime("%Y-%m")
                        oos_end_str = oos_end_dt.strftime("%Y-%m")
                        oos_files = _collect_files(args.data,
                                                   args.data_dir,
                                                   oos_start_str,
                                                   oos_end_str,
                                                   symbol=args.symbol)
                        if oos_files:
                            oos_raw = _load_many(oos_files)
                            # Re-engineer features for OOS data (using fitted engineer)
                            oos_raw_feat, _ = engineer_baseline_features(
                                oos_raw, baseline_engineer, fit=False)
                            # Create labels
                            oos_raw_feat = create_binary_labels_baseline(
                                oos_raw_feat, forward_bars=forward_bars)
                            oos_raw_feat[
                                "future_return"] = oos_raw_feat["close"].shift(
                                    -forward_bars) / oos_raw_feat["close"] - 1
                            oos_raw_feat = oos_raw_feat.dropna()
                            # Filter by OOS period
                            oos_mask = (oos_raw_feat.index >= oos_start_dt) & (
                                oos_raw_feat.index <= oos_end_dt)
                            oos_df = oos_raw_feat[oos_mask].copy()

            # Split training data: exclude OOS period from training
            if len(oos_df) > 0 and oos_start_dt is not None:
                # Training data should be before OOS start
                train_mask = feat_df.index < oos_start_dt
                train_df = feat_df[train_mask].copy()
            else:
                # No OOS, use all data for training
                train_df = feat_df.copy()

            print(f"\n📊 Data Split:")
            print(
                f"   Training set: {len(train_df):,} bars ({train_df.index.min()} to {train_df.index.max()})"
            )
            if len(oos_df) > 0:
                print(
                    f"   OOS test set: {len(oos_df):,} bars ({oos_df.index.min()} to {oos_df.index.max()})"
                )
                print(
                    f"   OOS period: {oos_start_dt.date() if oos_start_dt else 'N/A'} to {oos_end_dt.date() if oos_end_dt else 'N/A'}"
                )
            else:
                print(f"   OOS test set: 0 bars (no OOS data available)")
                if args.oos_months > 0 or args.oos_start:
                    print(
                        f"   ⚠️  Warning: OOS period specified but no data found"
                    )

            # Diagnostic: Check label distribution and data quality (on training data)
            label_dist = train_df["binary_signal"].value_counts()
            print(f"\n   📊 Training Label distribution: {dict(label_dist)}")
            print(
                f"      Long (1): {label_dist.get(1, 0)} ({100*label_dist.get(1, 0)/len(train_df):.2f}%)"
            )
            print(
                f"      Not Long (0): {label_dist.get(0, 0)} ({100*label_dist.get(0, 0)/len(train_df):.2f}%)"
            )
            future_ret_mean = train_df["future_return"].mean()
            future_ret_std = train_df["future_return"].std()
            print(
                f"   📈 Training Future return stats: mean={future_ret_mean:.6f}, std={future_ret_std:.6f}"
            )

            # Warn if label distribution is too imbalanced
            if label_dist.get(1, 0) / len(train_df) < 0.1 or label_dist.get(
                    1, 0) / len(train_df) > 0.9:
                print(
                    f"   ⚠️  Warning: Label distribution is highly imbalanced! This may affect model performance."
                )

            # Check for data leaks: ensure no forward-looking features
            feature_cols = get_baseline_feature_columns(train_df)
            suspicious_cols = [
                c for c in feature_cols if 'future' in c.lower()
                or 'forward' in c.lower() or 'next' in c.lower()
            ]
            if suspicious_cols:
                print(
                    f"   ⚠️  Warning: Found suspicious feature columns that might contain future data: {suspicious_cols}"
                )

            feature_cols = get_baseline_feature_columns(train_df)

            # Prepare training data
            X_train = train_df[feature_cols].values
            y_stage1_train = train_df["binary_signal"].values
            y_stage2_train = train_df["future_return"].values

            X = X_train
            y_stage1 = y_stage1_train  # Binary classification: 1=Long, 0=not Long
            y_stage2 = y_stage2_train  # Regression: future return

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

            # Evaluate on OOS test set if available
            oos_metrics = {}
            if len(oos_df) > 0:
                print(
                    f"\n📊 Evaluating on OOS test set ({len(oos_df):,} bars)..."
                )
                X_oos = oos_df[feature_cols].values
                y_stage1_oos = oos_df["binary_signal"].values
                y_stage2_oos = oos_df["future_return"].values

                # Stage1 OOS evaluation
                from sklearn.metrics import accuracy_score, mean_squared_error
                y_pred_stage1 = model_stage1.model.predict(X_oos)
                y_pred_stage1_binary = (y_pred_stage1 > 0.5).astype(int)
                oos_accuracy = accuracy_score(y_stage1_oos,
                                              y_pred_stage1_binary)
                oos_metrics["stage1"] = {
                    "accuracy": float(oos_accuracy),
                    "samples": len(oos_df)
                }
                print(f"   Stage1 OOS Accuracy: {oos_accuracy:.4f}")

                # Stage2 OOS evaluation
                y_pred_stage2 = model_stage2.model.predict(X_oos)
                oos_mse = mean_squared_error(y_stage2_oos, y_pred_stage2)
                oos_rmse = np.sqrt(oos_mse)
                oos_metrics["stage2"] = {
                    "mse": float(oos_mse),
                    "rmse": float(oos_rmse),
                    "samples": len(oos_df)
                }
                print(
                    f"   Stage2 OOS RMSE: {oos_rmse:.6f}, MSE: {oos_mse:.6f}")

                # OOS period information
                oos_metrics["oos_period"] = {
                    "start":
                    oos_start_dt.isoformat()
                    if oos_start_dt is not None else None,
                    "end":
                    oos_end_dt.isoformat() if oos_end_dt is not None else None,
                    "months":
                    args.oos_months if args.oos_months > 0 else 3
                }
            else:
                print(f"\n📊 No OOS evaluation (OOS test set is empty)")

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
            train_start = train_df.index.min() if not train_df.empty else None
            train_end = train_df.index.max() if not train_df.empty else None

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
                "train_start":
                train_start.isoformat() if train_start else None,
                "train_end":
                train_end.isoformat() if train_end else None,
                "total_bars":
                len(feat_df),
                "train_bars":
                len(train_df),
                "oos_months":
                args.oos_months if len(oos_df) > 0 else 0,
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

            # Add OOS metrics if available
            if oos_metrics:
                model_info["oos_metrics"] = oos_metrics

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

    # Generate summary report if multiple configurations were trained
    if len(freqs) > 1 or len(forward_bars_list) > 1:
        print(f"\n📊 Generating summary report for all configurations...")
        try:
            from ml_trading.pipeline.baseline.generate_summary_report import generate_summary_report
            summary_path = generate_summary_report("results/baseline")
            if summary_path:
                print(f"   ✓ Summary report saved to {summary_path}")
        except Exception as exc:
            import traceback
            print(f"   ⚠️  Failed to generate summary report: {exc}")
            traceback.print_exc()

    print(f"\n" + "=" * 80)
    print(f"✅ Baseline training completed for all combinations")
    print("=" * 80)


if __name__ == "__main__":
    main()
