"""Auto Rolling Update: Automatically detect available data and train/update models up to latest month.

This script:
1. Automatically finds all available monthly data files (across multiple years)
2. Trains initial model using early months
3. Rolls forward month by month up to the latest available data
4. Generates comprehensive HTML report

Usage:
    python auto_rolling_update.py --symbol BTCUSDT --initial-train-months 6
"""

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

from ml_trading.data_tools.rolling_data import (
    load_and_process_file,
    add_order_flow_features,
    engineer_features,
    create_labels,
    get_feature_columns,
)
from ml_trading.pipeline.dimensionality.utils import (
    load_top_factors_list,
    filter_engineered_by_topk,
)
from ml_trading.models.autoencoder import UnifiedAutoencoder
from ml_trading.utils.training import (
    train_lightgbm_model,
    simple_backtest,
    print_backtest_results,
)


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
            
            # Pattern 1: BTCUSDT-aggTrades-2024-10 or BTC-USD_2024-10
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
                    
                    files.append({
                        "path": str(file_path),
                        "year": year,
                        "month": month,
                        "month_str": f"{year}-{month:02d}",
                        "timestamp": pd.Timestamp(year, month, 1),
                    })
                except (ValueError, KeyError):
                    continue
    
    # Sort by timestamp (earliest first)
    files.sort(key=lambda x: x["timestamp"])
    
    return files


def main():
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
        help="Path to top_factors JSON (from dim-compare). If provided, filters engineered features to this Top-K list.",
    )
    parser.add_argument(
        "--use-autoencoder",
        type=str,
        default=None,
        help="Path to a trained autoencoder .pth (UnifiedAutoencoder). If provided, engineered features will be transformed to compressed embeddings before training.",
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=None,
        help="Encoding dimension of the provided autoencoder (required with --use-autoencoder)",
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
    if args.use_autoencoder:
        print(f"   Autoencoder: {args.use_autoencoder} (dim={args.encoding_dim})")
    
    # Find all available files
    print(f"\n🔍 Finding all available data files...")
    all_files = find_all_available_files(args.data_dir, args.symbol)
    
    if not all_files:
        print(f"❌ No data files found for {args.symbol} in {args.data_dir}!")
        print(f"   Looking for files matching: {args.symbol}-aggTrades-*.parquet or *.zip")
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
                    print(f"\n📋 Resuming from last trained month: {last_trained_month}")
    
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
            all_files = all_files[last_idx + 1:]
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
            print(f"   Found {len(existing_results)} existing results, will append new ones")
    
    # Ensure we have enough data
    if len(all_files) < args.min_train_months + 1:
        print(f"❌ Not enough data! Need at least {args.min_train_months + 1} months, found {len(all_files)}")
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
            print(f"⚠️  Skipping {test_file['month_str']}: insufficient training data ({len(train_files)} < {args.min_train_months})")
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
        train_df, feature_engineer = engineer_features(train_df, feature_engineer, fit=True)
        test_df, _ = engineer_features(test_df, feature_engineer, fit=False)
        print(f"   ✓ Features engineered: {len(get_feature_columns(train_df))} features")
        
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
                    engineered_data = {"train": train_df[feature_cols], "test": test_df[feature_cols]}
                    filtered_data = filter_engineered_by_topk(engineered_data, top_list)
                    train_df_filtered = train_df.copy()
                    train_df_filtered = train_df_filtered.drop(columns=feature_cols)
                    train_df_filtered = pd.concat([train_df_filtered, filtered_data["train"]], axis=1)
                    test_df_filtered = test_df.copy()
                    test_df_filtered = test_df_filtered.drop(columns=feature_cols)
                    test_df_filtered = pd.concat([test_df_filtered, filtered_data["test"]], axis=1)
                    train_df = train_df_filtered
                    test_df = test_df_filtered
                    print(f"   ✓ Applied Top-K filter: {len(get_feature_columns(train_df))} features remaining")
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️ Failed to apply Top-K filter: {exc}")
        
        # Optionally compress features using a provided autoencoder
        if args.use_autoencoder:
            if not args.encoding_dim:
                print(
                    "   ❌ --encoding-dim is required when --use-autoencoder is provided"
                )
                continue
            try:
                import torch
                feature_cols = get_feature_columns(train_df)
                input_dim = len(feature_cols)
                encoding_dim = int(args.encoding_dim)
                autoencoder = UnifiedAutoencoder(
                    input_dim,
                    encoding_dim,
                    architecture="production",
                )
                state = torch.load(args.use_autoencoder, map_location="cpu")
                autoencoder.load_state_dict(state)
                autoencoder.eval()
                
                def _transform_df(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
                    X = df[feature_cols].values
                    with torch.no_grad():
                        X_tensor = torch.as_tensor(X, dtype=torch.float32)
                        _, Z = autoencoder(X_tensor)
                        z_np = Z.numpy()
                    cols = [f"compressed_feature_{i}" for i in range(z_np.shape[1])]
                    df_transformed = df.drop(columns=feature_cols)
                    df_compressed = pd.DataFrame(z_np, index=df.index, columns=cols)
                    return pd.concat([df_transformed, df_compressed], axis=1)
                
                print(f"   🔄 Applying autoencoder compression ({input_dim} → {encoding_dim})...")
                train_df = _transform_df(train_df, feature_cols)
                test_df = _transform_df(test_df, feature_cols)
                print(f"   ✓ Applied autoencoder compression: {len(get_feature_columns(train_df))} compressed features")
            except Exception as exc:  # noqa: BLE001
                print(f"   ❌ Failed to apply autoencoder compression: {exc}")
                continue
        
        # Create labels
        print(f"\n4. Creating labels (forward_bars={args.forward_bars})...")
        train_df = create_labels(train_df, forward_bars=args.forward_bars)
        test_df = create_labels(test_df, forward_bars=args.forward_bars)
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        print(f"   ✓ Train samples: {len(train_df):,}")
        print(f"   ✓ Test samples: {len(test_df):,}")
        
        # Prepare features
        feature_cols = get_feature_columns(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df["signal"].values  # Use 3-class signal (0=Hold, 1=Long, 2=Short)
        X_test = test_df[feature_cols].values
        
        # Train model
        print(f"\n5. Training LightGBM model...")
        print(f"   Samples: {len(X_train):,}, Features: {len(feature_cols)}")
        model = train_lightgbm_model(X_train, y_train, use_gpu=args.gpu)
        print("   ✓ Training complete")
        
        # Predict
        print(f"\n6. Generating predictions...")
        predictions = model.predict(X_test)
        print(f"   ✓ Predictions generated")
        
        # Backtest
        print(f"\n7. Running backtest...")
        results = simple_backtest(test_df, predictions)
        results["test_month"] = test_file["month_str"]
        results["train_months"] = len(train_files)
        results["train_samples"] = len(X_train)
        results["test_samples"] = len(X_test)
        results["num_features"] = len(feature_cols)
        results["train_start"] = train_files[0]["month_str"]
        results["train_end"] = train_files[-1]["month_str"]
        
        all_results.append(results)
        
        print_backtest_results(results, f"{test_file['month_str']} Results")
        
        # Save model (optional - for production use)
        model_path = os.path.join(results_dir, f"model_{test_file['month_str']}.txt")
        model.save_model(model_path)
        print(f"\n   💾 Model saved: {model_path}")
    
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
    print(f"{'Month':<12} {'Trades':<8} {'Return':<10} {'Win%':<8} {'PF':<8} {'MaxDD':<10}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['test_month']:<12} {row['total_trades']:<8} "
              f"{row['total_return']:>8.2f}% {row['win_rate']:>6.1f}% "
              f"{row['profit_factor']:>6.2f} {row['max_drawdown']:>8.2f}%")
    
    print("-" * 80)
    print(f"{'AVERAGE':<12} {results_df['total_trades'].mean():<8.1f} "
          f"{results_df['total_return'].mean():>8.2f}% "
          f"{results_df['win_rate'].mean():>6.1f}% "
          f"{results_df['profit_factor'].mean():>6.2f} "
          f"{results_df['max_drawdown'].mean():>8.2f}%")
    
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
    print(f"   - model_*.txt (one per month)")
    
    # Generate HTML report
    try:
        from ml_trading.pipeline.dimensionality.report_generator import write_rolling_report
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

