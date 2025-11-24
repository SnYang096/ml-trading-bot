"""Auto Rolling Update: Automatically detect available data and train/update models up to latest month.

This module:
1. Automatically finds all available monthly data files (across multiple years)
2. Trains initial model using early months
3. Rolls forward month by month up to the latest available data
4. Generates comprehensive HTML report

CLI usage (example):
    python -m time_series_model.pipeline.rolling.auto_rolling_update --symbol BTCUSDT --initial-train-months 6
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

from data_tools.rolling_data import (
    load_and_process_file,
    add_order_flow_features,
    engineer_features,
    get_feature_columns,
)
from time_series_model.pipeline.dimensionality.utils import (
    load_top_factors_list,
    filter_engineered_by_topk,
)
from time_series_model.pipeline.training.classification_model_trainer import (
    ClassificationModelTrainer,
)
from time_series_model.pipeline.training.label_utils import (
    log_return_magnitude,
    rolling_rms_volatility,
    future_volatility_label,
    rolling_quantile_classification_labels,
)
from src.time_series_model.strategies.classification_strategy_handler import (
    ClassificationStrategyHandler,
)
from time_series_model.pipeline.training.preprocessing import RobustWinsorizer
from time_series_model.models.quant_trading_model import TradingModelPipeline
from time_series_model.backtesting.vectorbot import (
    print_backtest_results,
    evaluate_signal_performance,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


class _SingleTimeframePipeline:
    """Stub pipeline to reuse ClassificationStrategyHandler with single timeframe."""

    def __init__(self, timeframe: str, cls_model, return_model, vol_model):
        self.classification_models = {timeframe: cls_model}
        self.return_models = {timeframe: return_model}
        self.volatility_models = {timeframe: vol_model}


def find_all_available_files(data_dir: str, symbol: str) -> List[Dict]:
    """Find all available monthly data files for a symbol across all years.

    Returns sorted list of files from earliest to latest.
    """
    files = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return files

    # Try multiple patterns
    patterns = [
        f"{symbol}-aggTrades-*.parquet",
        f"{symbol}-aggTrades-*.zip",
        f"{symbol}-*.parquet",
        f"{symbol}-*.zip",
    ]

    # Symbol mapping for file naming variations
    symbol_mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "BNBUSDT": "BNB-USD",
        "ADAUSDT": "ADA-USD",
        "SOLUSDT": "SOL-USD",
    }
    file_symbol = symbol_mapping.get(symbol, symbol)

    for pattern in patterns:
        for file_path in data_path.glob(pattern):
            # Try to extract year-month from filename
            stem = file_path.stem

            # Pattern extraction
            import re

            date_patterns = [
                rf"{re.escape(symbol)}-aggTrades-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
                rf"{re.escape(file_symbol)}_(?P<year>\d{{4}})-(?P<month>\d{{2}})",
                rf"{re.escape(file_symbol)}-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
                rf"(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            ]

            match = None
            for pattern_re in date_patterns:
                match = re.search(pattern_re, stem)
                if match:
                    break

            if match:
                try:
                    year = int(match.group("year"))
                    month = int(match.group("month"))

                    files.append(
                        {
                            "path": str(file_path),
                            "year": year,
                            "month": month,
                            "month_str": f"{year}-{month:02d}",
                            "timestamp": pd.Timestamp(year, month, 1),
                        }
                    )
                except (ValueError, KeyError):
                    continue

    # Sort by timestamp (earliest first)
    files.sort(key=lambda x: x["timestamp"])

    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto Rolling Update: Train and update models up to latest available data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DATA_DIR", "data/parquet_data"),
        help="Directory containing monthly data files",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol",
    )
    parser.add_argument(
        "--initial-train-months",
        type=int,
        default=6,
        help="Initial training months (e.g., 6 = first 6 months)",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=3,
        help="Minimum training months required (e.g., 3 = at least 3 months)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory name (default: auto_rolling_{symbol}_{timestamp})",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU for training",
    )
    parser.add_argument(
        "--add-order-flow",
        action="store_true",
        default=False,
        help="Add order flow features (CVD, taker_buy_ratio)",
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        default=False,
        help="Only update from last trained month, don't retrain all months",
    )
    parser.add_argument(
        "--use-top-factors",
        type=str,
        default=None,
        help="Path to top_factors JSON (from ts-dim-compare). If provided, filters engineered features to this Top-K list.",
    )
    parser.add_argument(
        "--forward-bars",
        type=int,
        default=3,
        help="Number of bars ahead for label prediction (default: 3). Use 1, 5, 10, or 15 for different horizons.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 Auto Rolling Update: Train and Update to Latest Data")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Symbol: {args.symbol}")
    print(f"   Data Directory: {args.data_dir}")
    print(f"   Initial Training: {args.initial_train_months} months")
    print(f"   Minimum Training: {args.min_train_months} months")
    print(f"   GPU: {args.gpu}")
    print(f"   Order Flow Features: {args.add_order_flow}")
    print(f"   Update Only: {args.update_only}")
    print(f"   Forward Bars (Horizon): {args.forward_bars}")
    if args.use_top_factors:
        print(f"   Top Factors: {args.use_top_factors}")

    # Find all available files
    print(f"\n🔍 Finding all available data files...")
    all_files = find_all_available_files(args.data_dir, args.symbol)

    if not all_files:
        print(f"❌ No data files found for {args.symbol} in {args.data_dir}!")
        print(
            f"   Looking for files matching: {args.symbol}-aggTrades-*.parquet or *.zip"
        )
        return

    print(f"   Found {len(all_files)} months of data:")
    print(f"   Earliest: {all_files[0]['month_str']}")
    print(f"   Latest: {all_files[-1]['month_str']}")

    # Determine output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"auto_rolling_{args.symbol.lower()}_{timestamp}"

    results_dir = f"results/{args.output}"
    os.makedirs(results_dir, exist_ok=True)

    # Check if we should resume from last run
    last_trained_month = None
    if args.update_only:
        # Try to find last trained month from summary.json
        summary_path = os.path.join(results_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
                last_results = summary.get("last_trained_month")
                if last_results:
                    last_trained_month = last_results
                    print(
                        f"\n📋 Resuming from last trained month: {last_trained_month}"
                    )

    # Filter files based on update-only mode
    if args.update_only and last_trained_month:
        # Find index of last trained month
        last_idx = None
        for idx, f in enumerate(all_files):
            if f["month_str"] == last_trained_month:
                last_idx = idx
                break

        if last_idx is not None and last_idx < len(all_files) - 1:
            # Start from next month after last trained
            all_files = all_files[last_idx + 1 :]
            print(f"   Continuing with {len(all_files)} remaining months")
        else:
            print(f"   ✅ Already up to date (last trained: {last_trained_month})")
            return
    else:
        # Load existing results if available
        existing_results = []
        results_csv_path = os.path.join(results_dir, "monthly_results.csv")
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path)
            existing_results = existing_df.to_dict("records")
            print(
                f"   Found {len(existing_results)} existing results, will append new ones"
            )

    # Ensure we have enough data
    if len(all_files) < args.min_train_months + 1:
        print(
            f"❌ Not enough data! Need at least {args.min_train_months + 1} months, found {len(all_files)}"
        )
        return

    # Rolling training
    all_results = []
    feature_engineer = None  # Will be created in first iteration

    print(f"\n" + "=" * 80)
    print(f"🔄 Starting Auto Rolling Update")
    print(f"=" * 80 + "\n")

    # Determine starting point
    start_idx = args.initial_train_months
    if args.update_only and existing_results:
        # Find the last trained month in existing results
        last_trained = existing_results[-1].get("test_month")
        for idx, f in enumerate(all_files):
            if f["month_str"] == last_trained:
                start_idx = idx + 1  # Start from next month
                break

    for i in range(start_idx, len(all_files)):
        train_files = all_files[:i]
        test_file = all_files[i]

        # Skip if not enough training data
        if len(train_files) < args.min_train_months:
            print(
                f"⚠️  Skipping {test_file['month_str']}: insufficient training data ({len(train_files)} < {args.min_train_months})"
            )
            continue

        print(f"\n{'=' * 80}")
        print(
            f"[{i - start_idx + 1}/{len(all_files) - start_idx}] {test_file['month_str']}"
        )
        print(f"{'=' * 80}")
        print(
            f"Train: {train_files[0]['month_str']} to {train_files[-1]['month_str']} ({len(train_files)} months)"
        )
        print(f"Test:  {test_file['month_str']}")

        # Load training data
        print(f"\n1. Loading training data...")
        train_data = []
        for file_info in train_files:
            print(f"   Loading {file_info['month_str']}")
            df = load_and_process_file(file_info["path"])
            if df is not None and len(df) > 0:
                if args.add_order_flow:
                    df = add_order_flow_features(file_info["path"], df)
                train_data.append(df)

        if not train_data:
            print("❌ No training data!")
            continue

        train_df = pd.concat(train_data, axis=0).sort_index()
        print(f"   ✓ Training data: {len(train_df):,} bars")

        # Load test data
        print(f"\n2. Loading test data...")
        test_df = load_and_process_file(test_file["path"])
        if test_df is None or len(test_df) == 0:
            print("❌ No test data!")
            continue

        if args.add_order_flow:
            test_df = add_order_flow_features(test_file["path"], test_df)

        print(f"   ✓ Test data: {len(test_df):,} bars")

        # Engineer features
        print(f"\n3. Engineering features...")
        train_df, feature_engineer = engineer_features(
            train_df, feature_engineer, fit=True
        )
        test_df, _ = engineer_features(test_df, feature_engineer, fit=False)
        print(
            f"   ✓ Features engineered: {len(get_feature_columns(train_df))} features"
        )

        # Optionally filter features by Top-K list
        if args.use_top_factors:
            try:
                top_list = load_top_factors_list(args.use_top_factors)
                if not top_list:
                    print("   ⚠️ Top factors list is empty; skipping filtering")
                else:
                    print(f"   🔎 Applying Top-K filter with {len(top_list)} factors")
                    feature_cols = get_feature_columns(train_df)
                    # Convert to dict format for filter function
                    engineered_data = {
                        "train": train_df[feature_cols],
                        "test": test_df[feature_cols],
                    }
                    filtered_data = filter_engineered_by_topk(engineered_data, top_list)
                    train_df_filtered = train_df.copy()
                    train_df_filtered = train_df_filtered.drop(columns=feature_cols)
                    train_df_filtered = pd.concat(
                        [train_df_filtered, filtered_data["train"]], axis=1
                    )
                    test_df_filtered = test_df.copy()
                    test_df_filtered = test_df_filtered.drop(columns=feature_cols)
                    test_df_filtered = pd.concat(
                        [test_df_filtered, filtered_data["test"]], axis=1
                    )
                    train_df = train_df_filtered
                    test_df = test_df_filtered
                    print(
                        f"   ✓ Applied Top-K filter: {len(get_feature_columns(train_df))} features remaining"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️ Failed to apply Top-K filter: {exc}")

        # Label construction following doc-aligned targets
        print(f"\n4. Creating labels (forward_bars={args.forward_bars})...")
        vol_window = max(5, args.forward_bars)

        def _apply_labeling(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # ⚠️  FIXED: Use close[t+1] as entry price to avoid current bar's close
            close_next = df["close"].shift(-1)  # Use next bar's close as entry
            df["future_return"] = close_next.shift(-args.forward_bars) / close_next - 1
            # ✅ Compute future volatility label: RMS of future single-period returns
            df["future_volatility"] = future_volatility_label(
                df["close"],
                horizon=args.forward_bars,
                min_periods=max(3, args.forward_bars // 2),
            )
            return df.dropna(subset=["future_return", "future_volatility"])

        train_df = _apply_labeling(train_df)
        test_df = _apply_labeling(test_df)
        print(f"   ✓ Train samples: {len(train_df):,}")
        print(f"   ✓ Test samples: {len(test_df):,}")

        # Prepare features/targets
        feature_cols = get_feature_columns(train_df)
        X_train_df = train_df[feature_cols]
        y_return_train = train_df["future_return"]
        y_vol_train = train_df["future_volatility"]

        X_test_df = test_df[feature_cols]
        y_return_test = test_df["future_return"]
        y_vol_test = test_df["future_volatility"]

        print(f"\n5. Training multi-model classifier (LightGBM ensemble)...")
        trainer_splits = max(2, min(5, len(X_train_df) // 5000 + 2))
        trainer = ClassificationModelTrainer(
            use_gpu=args.gpu,
            auto_tune_params=False,
            auto_tune_return=False,
            auto_tune_vol=False,
            use_quantile_labels=True,
            quantile_window=5000,
            quantile_lower=0.4,
            quantile_upper=0.6,
            quantile_min_periods=200,
        )

        models_dict, metrics_dict, preprocess_params_dict = trainer.train_models(
            X_df=X_train_df,
            y_return=y_return_train,
            y_vol=y_vol_train,
            train_df=train_df,
            n_splits=trainer_splits,
            groups=None,
            preprocess_fn=None,
            preprocess_kwargs={},
            feature_winsorize_k=4.0,
        )

        model_cls = models_dict.get("classification")
        model_return = models_dict.get("return")
        model_vol = models_dict.get("vol")
        classification_preprocess_params = preprocess_params_dict.get("classification")
        return_preprocess_params = preprocess_params_dict.get("return")
        vol_preprocess_params = preprocess_params_dict.get("vol")

        if model_cls is None or model_return is None or model_vol is None:
            print(
                "❌ Failed to train classification/return/volatility models; skipping month."
            )
            continue

        print("   ✓ Models trained successfully")

        # Predictions
        print(f"\n6. Generating predictions and signals...")
        class_proba = model_cls.predict(X_test_df)
        return_log_pred = model_return.predict(X_test_df)
        vol_pred = model_vol.predict(X_test_df)

        pipeline_stub = _SingleTimeframePipeline(
            timeframe="default",
            cls_model=model_cls,
            return_model=model_return,
            vol_model=model_vol,
        )
        strategy_handler = ClassificationStrategyHandler(
            pipeline_stub,
            signal_strength_threshold=0.0,
            confidence_threshold=0.0,
            base_position_size=1.0,
            classification_threshold=0.5,
        )
        test_slice = test_df.loc[X_test_df.index]
        signals_df = strategy_handler.generate_signals(
            X_test_df, test_slice, timeframe="default"
        )

        # Performance evaluation
        performance = evaluate_signal_performance(
            signals_df,
            y_return_test,
        )

        # Directional metrics using simple threshold
        y_true_dir = (y_return_test > 0).astype(int)
        y_pred_dir = (class_proba > 0.5).astype(int)
        cls_accuracy = float(accuracy_score(y_true_dir, y_pred_dir))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_dir, y_pred_dir, average="binary", zero_division=0
        )
        try:
            cls_auc = float(roc_auc_score(y_true_dir, class_proba))
        except Exception:
            cls_auc = None
        try:
            cls_pr_auc = float(average_precision_score(y_true_dir, class_proba))
        except Exception:
            cls_pr_auc = None

        y_true_mag = np.abs(y_return_test.to_numpy())
        y_pred_mag = signals_df["return_pred"].reindex(y_return_test.index).to_numpy()
        mask_mag = ~np.isnan(y_pred_mag)
        y_true_mag_valid = y_true_mag[mask_mag]
        y_pred_mag_valid = y_pred_mag[mask_mag]
        y_pred_log = return_log_pred[mask_mag]
        y_true_log = np.log1p(y_true_mag_valid)
        if len(y_true_mag_valid) > 0:
            ret_rmse = float(
                np.sqrt(np.mean((y_true_mag_valid - y_pred_mag_valid) ** 2))
            )
            ret_mae = float(np.mean(np.abs(y_true_mag_valid - y_pred_mag_valid)))
            ret_r2 = float(
                1
                - np.sum((y_true_mag_valid - y_pred_mag_valid) ** 2)
                / (np.sum((y_true_mag_valid - y_true_mag_valid.mean()) ** 2) + 1e-12)
            )
            ret_rmse_log = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))
            ret_mae_log = float(np.mean(np.abs(y_true_log - y_pred_log)))
            ret_r2_log = float(
                1
                - np.sum((y_true_log - y_pred_log) ** 2)
                / (np.sum((y_true_log - y_true_log.mean()) ** 2) + 1e-12)
            )
        else:
            ret_rmse = ret_mae = ret_r2 = ret_rmse_log = ret_mae_log = ret_r2_log = 0.0

        vol_array = y_vol_test.to_numpy()
        mask_vol = ~np.isnan(vol_pred)
        vol_true_valid = vol_array[mask_vol]
        vol_pred_valid = vol_pred[mask_vol]
        if len(vol_true_valid) > 0:
            vol_rmse = float(np.sqrt(np.mean((vol_true_valid - vol_pred_valid) ** 2)))
            vol_mae = float(np.mean(np.abs(vol_true_valid - vol_pred_valid)))
            vol_r2 = float(
                1
                - np.sum((vol_true_valid - vol_pred_valid) ** 2)
                / (np.sum((vol_true_valid - vol_true_valid.mean()) ** 2) + 1e-12)
            )
        else:
            vol_rmse = vol_mae = vol_r2 = 0.0

        performance.update(
            {
                "symbol": args.symbol,
                "timeframe": f"fb{args.forward_bars}",
                "forward_bars": args.forward_bars,
                "test_month": test_file["month_str"],
                "train_months": len(train_files),
                "train_samples": int(len(X_train_df)),
                "test_samples": int(len(X_test_df)),
                "num_features": len(feature_cols),
                "train_start": train_files[0]["month_str"],
                "train_end": train_files[-1]["month_str"],
                "feature_type": args.feature_type,
                "cls_accuracy": cls_accuracy,
                "cls_precision": float(precision),
                "cls_recall": float(recall),
                "cls_f1": float(f1),
                "cls_auc": cls_auc,
                "cls_pr_auc": cls_pr_auc,
                "avg_signal_strength": float(
                    np.mean(np.abs(signals_df["signal_strength"]))
                ),
                "test_rmse_return": ret_rmse,
                "test_mae_return": ret_mae,
                "test_r2_return": ret_r2,
                "test_rmse_return_log": ret_rmse_log,
                "test_mae_return_log": ret_mae_log,
                "test_r2_return_log": ret_r2_log,
                "test_rmse_vol": vol_rmse,
                "test_mae_vol": vol_mae,
                "test_r2_vol": vol_r2,
                "metrics": metrics_dict,
            }
        )

        all_results.append(performance)

        print_backtest_results(performance, f"{test_file['month_str']} Results")

        # Save pipelines for deployment
        models_dir = os.path.join(results_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        cls_pipeline = TradingModelPipeline(
            model_type="classification",
            forward_bars=args.forward_bars,
            feature_cols=feature_cols,
            preprocess_params=classification_preprocess_params,
            use_gpu=args.gpu,
        )
        cls_pipeline.model = model_cls.model
        if classification_preprocess_params:
            cls_pipeline.preprocessor = RobustWinsorizer.from_params(
                classification_preprocess_params, forward_bars=args.forward_bars
            )
        cls_path = os.path.join(
            models_dir, f"classification_pipeline_{test_file['month_str']}.pkl"
        )
        cls_pipeline.save(cls_path)

        return_pipeline = TradingModelPipeline(
            model_type="regression",
            forward_bars=args.forward_bars,
            feature_cols=feature_cols,
            preprocess_params=return_preprocess_params,
            use_gpu=args.gpu,
            target_transform="log1p_abs",
        )
        return_pipeline.model = model_return.model
        if return_preprocess_params:
            return_pipeline.preprocessor = RobustWinsorizer.from_params(
                return_preprocess_params, forward_bars=args.forward_bars
            )
        ret_path = os.path.join(
            models_dir, f"return_pipeline_{test_file['month_str']}.pkl"
        )
        return_pipeline.save(ret_path)

        vol_pipeline = TradingModelPipeline(
            model_type="regression",
            forward_bars=args.forward_bars,
            feature_cols=feature_cols,
            preprocess_params=vol_preprocess_params,
            use_gpu=args.gpu,
        )
        vol_pipeline.model = model_vol.model
        if vol_preprocess_params:
            vol_pipeline.preprocessor = RobustWinsorizer.from_params(
                vol_preprocess_params, forward_bars=args.forward_bars
            )
        vol_path = os.path.join(
            models_dir, f"vol_pipeline_{test_file['month_str']}.pkl"
        )
        vol_pipeline.save(vol_path)
        print(f"\n   💾 Pipelines saved under {models_dir}")

    # Combine with existing results if updating
    if args.update_only and existing_results:
        all_results = existing_results + all_results

    # Save all results
    print(f"\n" + "=" * 80)
    print(f"📊 SUMMARY")
    print(f"=" * 80 + "\n")

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "monthly_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    # Print summary table
    print(
        f"{'Month':<12} {'Trades':<8} {'Return':<10} {'Win%':<8} {'PF':<8} {'MaxDD':<10}"
    )
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(
            f"{row['test_month']:<12} {row['total_trades']:<8} "
            f"{row['total_return']:>8.2f}% {row['win_rate']:>6.1f}% "
            f"{row['profit_factor']:>6.2f} {row['max_drawdown']:>8.2f}%"
        )

    print("-" * 80)
    print(
        f"{'AVERAGE':<12} {results_df['total_trades'].mean():<8.1f} "
        f"{results_df['total_return'].mean():>8.2f}% "
        f"{results_df['win_rate'].mean():>6.1f}% "
        f"{results_df['profit_factor'].mean():>6.2f} "
        f"{results_df['max_drawdown'].mean():>8.2f}%"
    )

    # Save summary
    summary = {
        "symbol": args.symbol,
        "total_months_tested": len(results_df),
        "earliest_month": all_files[0]["month_str"] if all_files else "N/A",
        "latest_month": all_files[-1]["month_str"] if all_files else "N/A",
        "last_trained_month": all_files[-1]["month_str"] if all_files else "N/A",
        "avg_return": float(results_df["total_return"].mean()),
        "avg_win_rate": float(results_df["win_rate"].mean()),
        "avg_profit_factor": float(results_df["profit_factor"].mean()),
        "avg_max_drawdown": float(results_df["max_drawdown"].mean()),
        "total_trades": int(results_df["total_trades"].sum()),
        "feature_engineering": "EnhancedFeatureEngineer",
        "configuration": vars(args),
        "created_at": datetime.now().isoformat(),
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"   - monthly_results.csv")
    print(f"   - summary.json")
    print(
        f"   - classification_pipeline_*.pkl / return_pipeline_*.pkl / vol_pipeline_*.pkl"
    )

    # Generate HTML report
    try:
        from time_series_model.pipeline.dimensionality.report_generator import (
            write_rolling_report,
        )

        report_path = write_rolling_report(
            results_dir,
            summary_path=summary_path,
            results_csv_path=results_csv_path,
            report_type="monthly",
        )
        print(f"   - monthly_rolling_report.html")
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Failed to generate HTML report: {exc}")

    print("\n" + "=" * 80)
    print("✅ Auto rolling update completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
