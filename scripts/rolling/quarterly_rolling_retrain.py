"""Quarterly Rolling Re-training: Train on expanding window, test on next quarter.

使用 EnhancedFeatureEngineer 进行特征工程（WPT + Hurst + 高级特征）
"""

import os
import sys
import pandas as pd
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add common utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utils import (
    load_and_process_file,
    add_order_flow_features,
    engineer_features,
    create_labels,
    get_feature_columns,
)
from training_utils import train_lightgbm_model, simple_backtest, print_backtest_results


def find_data_files_by_quarter(data_dir, symbols, start_year, end_year):
    """Find and organize data files by quarter."""
    all_files = []
    for symbol in symbols:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                file_path = os.path.join(
                    data_dir, f"{symbol}-aggTrades-{year}-{month:02d}.zip"
                )
                if os.path.exists(file_path):
                    # Determine quarter
                    quarter = (month - 1) // 3 + 1
                    quarter_str = f"{year}Q{quarter}"
                    all_files.append(
                        {
                            "path": file_path,
                            "year": year,
                            "month": month,
                            "quarter": quarter_str,
                            "symbol": symbol,
                        }
                    )

    return sorted(all_files, key=lambda x: (x["year"], x["month"]))


def main():
    parser = argparse.ArgumentParser(description="Quarterly Rolling Re-training")
    parser.add_argument(
        "--data-dir", type=str, default=r"D:\GitHub\trading\rlbot\data\agg_data"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--initial-train-quarters",
        type=int,
        default=8,
        help="Initial training quarters (e.g., 8 = 2 years)",
    )
    parser.add_argument("--output", type=str, default="quarterly_rolling_btc")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument(
        "--add-order-flow",
        action="store_true",
        default=False,
        help="Add order flow features (CVD, taker_buy_ratio)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("📊 Quarterly Rolling Re-training (Enhanced Features)")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Symbols: {', '.join(args.symbols)}")
    print(f"   Period: {args.start_year}-{args.end_year}")
    print(f"   Initial training: {args.initial_train_quarters} quarters")
    print(f"   GPU: {args.gpu}")
    print(f"   Order Flow Features: {args.add_order_flow}")
    print(f"   Feature Engineering: EnhancedFeatureEngineer (WPT + Hurst + Advanced)")

    # Find all files
    print(f"\n🔍 Finding data files...")
    all_files = find_data_files_by_quarter(
        args.data_dir, args.symbols, args.start_year, args.end_year
    )

    if not all_files:
        print("❌ No data files found!")
        return

    # Group by quarter
    quarters_dict = {}
    for file_info in all_files:
        q = file_info["quarter"]
        if q not in quarters_dict:
            quarters_dict[q] = []
        quarters_dict[q].append(file_info)

    quarters = sorted(quarters_dict.keys())
    print(f"   Found {len(quarters)} quarters: {quarters[0]} to {quarters[-1]}")

    # Rolling training
    results_dir = f"results/{args.output}"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    feature_engineer = None  # Will be created in first iteration

    print(f"\n" + "=" * 80)
    print(f"🔄 Starting Quarterly Rolling Re-training")
    print(f"=" * 80 + "\n")

    for i in range(args.initial_train_quarters, len(quarters)):
        train_quarters = quarters[:i]
        test_quarter = quarters[i]

        print(f"\n{'=' * 80}")
        print(
            f"[{i - args.initial_train_quarters + 1}/{len(quarters) - args.initial_train_quarters}] {test_quarter}"
        )
        print(f"{'=' * 80}")
        print(
            f"Train: {train_quarters[0]} to {train_quarters[-1]} ({len(train_quarters)} quarters)"
        )
        print(f"Test:  {test_quarter}")

        # Load training data
        print(f"\n1. Loading training data...")
        train_files = []
        for q in train_quarters:
            train_files.extend(quarters_dict[q])

        train_data = []
        for file_info in train_files:
            print(
                f"   Loading {file_info['quarter']}: {os.path.basename(file_info['path'])}"
            )
            df = load_and_process_file(file_info["path"])
            if df is not None and len(df) > 0:
                # Add order flow features if requested
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
        test_files = quarters_dict[test_quarter]
        test_data = []
        for file_info in test_files:
            print(f"   Loading: {os.path.basename(file_info['path'])}")
            df = load_and_process_file(file_info["path"])
            if df is not None and len(df) > 0:
                # Add order flow features if requested
                if args.add_order_flow:
                    df = add_order_flow_features(file_info["path"], df)
                test_data.append(df)

        if not test_data:
            print("❌ No test data!")
            continue

        test_df = pd.concat(test_data, axis=0).sort_index()
        print(f"   ✓ Test data: {len(test_df):,} bars")

        # Engineer features using EnhancedFeatureEngineer
        print(f"\n3. Engineering enhanced features...")
        print(f"   Features: WPT + Hurst + Hilbert + Spectral + Advanced Derived")
        train_df, feature_engineer = engineer_features(
            train_df, feature_engineer, fit=True
        )
        test_df, _ = engineer_features(test_df, feature_engineer, fit=False)
        print(
            f"   ✓ Features engineered: {len(get_feature_columns(train_df))} features"
        )

        # Create labels
        print(f"\n4. Creating labels...")
        train_df = create_labels(train_df)
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        print(f"   ✓ Train samples: {len(train_df):,}")
        print(f"   ✓ Test samples: {len(test_df):,}")

        # Prepare features
        feature_cols = get_feature_columns(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df["binary_signal"].values
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
        results["quarter"] = test_quarter
        results["train_quarters"] = len(train_quarters)
        results["train_samples"] = len(X_train)
        results["test_samples"] = len(X_test)
        results["num_features"] = len(feature_cols)

        all_results.append(results)

        print_backtest_results(results, f"{test_quarter} Results")

        # Save model
        model_path = os.path.join(results_dir, f"model_{test_quarter}.txt")
        model.save_model(model_path)
        print(f"\n   💾 Model saved: {model_path}")

        # Save feature importance
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importance()}
        ).sort_values("importance", ascending=False)
        importance_path = os.path.join(
            results_dir, f"feature_importance_{test_quarter}.csv"
        )
        importance_df.to_csv(importance_path, index=False)
        print(f"   💾 Feature importance saved: {importance_path}")

    # Save all results
    print(f"\n" + "=" * 80)
    print(f"📊 SUMMARY")
    print(f"=" * 80 + "\n")

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "quarterly_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    # Print summary table
    print(
        f"{'Quarter':<10} {'Trades':<8} {'Return':<10} {'Win%':<8} {'PF':<8} {'MaxDD':<10}"
    )
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(
            f"{row['quarter']:<10} {row['total_trades']:<8} "
            f"{row['total_return']:>8.2f}% {row['win_rate']:>6.1f}% "
            f"{row['profit_factor']:>6.2f} {row['max_drawdown']:>8.2f}%"
        )

    print("-" * 80)
    print(
        f"{'AVERAGE':<10} {results_df['total_trades'].mean():<8.1f} "
        f"{results_df['total_return'].mean():>8.2f}% "
        f"{results_df['win_rate'].mean():>6.1f}% "
        f"{results_df['profit_factor'].mean():>6.2f} "
        f"{results_df['max_drawdown'].mean():>8.2f}%"
    )

    # Save summary
    summary = {
        "total_quarters_tested": len(results_df),
        "avg_return": float(results_df["total_return"].mean()),
        "avg_win_rate": float(results_df["win_rate"].mean()),
        "avg_profit_factor": float(results_df["profit_factor"].mean()),
        "avg_max_drawdown": float(results_df["max_drawdown"].mean()),
        "total_trades": int(results_df["total_trades"].sum()),
        "feature_engineering": "EnhancedFeatureEngineer",
        "configuration": vars(args),
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"   - quarterly_results.csv")
    print(f"   - summary.json")
    print(f"   - model_*.txt (one per quarter)")
    print(f"   - feature_importance_*.csv (one per quarter)")

    print("\n" + "=" * 80)
    print("✅ Quarterly rolling re-training completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
