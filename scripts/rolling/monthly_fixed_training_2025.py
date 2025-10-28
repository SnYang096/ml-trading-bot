"""Fixed Training for 2025: Train on 2024 Q4, test on 2025 months (no retraining).

使用固定训练方式：用2024年Q4训练一次，然后测试2025年1-6月
对比滚动训练的效果差异
"""

import os
import sys
import pandas as pd
import json
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


def main():
    data_dir = r"D:\GitHub\trading\rlbot\data\agg_data"

    print("\n" + "=" * 80)
    print("📊 Fixed Training (2025 - Enhanced Features)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Train: 2024 Q4 (Oct-Dec) - FIXED")
    print(f"   Test: 2025 Jan-Jun")
    print(f"   Feature Engineering: EnhancedFeatureEngineer (WPT + Hurst + OrderFlow)")
    print(f"   Training Strategy: FIXED (no retraining)")

    # Load 2024 Q4 as training data (Oct, Nov, Dec)
    print(f"\n🔍 Finding training data files...")
    train_files = []
    for month in [10, 11, 12]:
        file_path = os.path.join(data_dir, f"BTCUSDT-aggTrades-2024-{month:02d}.zip")
        if os.path.exists(file_path):
            train_files.append(file_path)

    print(f"   Found {len(train_files)} months for training (2024 Q4)")

    # Test on 2025 Jan-Jun
    test_months = [1, 2, 3, 4, 5, 6]

    results_dir = "results/monthly_fixed_training_2025"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # Load and prepare training data (ONCE)
    print(f"\n📥 Loading training data (2024 Q4)...")
    train_data = []
    for fp in train_files:
        print(f"   Loading {os.path.basename(fp)}")
        df = load_and_process_file(fp)
        if df is not None and len(df) > 0:
            # Add order flow features
            try:
                df = add_order_flow_features(fp, df)
                print(f"   ✓ Order flow features added")
            except Exception as e:
                print(f"   ⚠️  Order flow failed: {e}")
            train_data.append(df)

    # Combine training data
    train_df = pd.concat(train_data, axis=0).sort_index()
    print(f"   ✓ Combined training samples: {len(train_df):,} bars")

    # Engineer features for training data (ONCE)
    print(f"\n🔧 Engineering features for training data...")
    print(f"   Features: WPT + Hurst + Hilbert + Spectral + OrderFlow")
    train_df, feature_engineer = engineer_features(train_df, None, fit=True)
    print(f"   ✓ Features engineered: {len(get_feature_columns(train_df))} features")

    # Create labels for training data
    print(f"\n🏷️  Creating labels for training data...")
    train_df = create_labels(train_df)
    train_df = train_df.dropna()
    print(f"   ✓ Training samples: {len(train_df):,}")

    # Prepare training features
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df["binary_signal"].values

    # Train model (ONCE)
    print(f"\n🤖 Training LightGBM model (FIXED)...")
    print(f"   Samples: {len(X_train):,}, Features: {len(feature_cols)}")
    model = train_lightgbm_model(X_train, y_train, use_gpu=True, num_boost_round=200)
    print("   ✓ Training complete")

    # Save the trained model
    model_path = os.path.join(results_dir, "model_fixed_2024Q4.txt")
    model.save_model(model_path)
    print(f"   💾 Model saved: {model_path}")

    print(f"\n" + "=" * 80)
    print(f"🧪 Testing on 2025 months (FIXED MODEL)")
    print(f"=" * 80 + "\n")

    # Test on each month (using the SAME model)
    for i, test_month in enumerate(test_months, 1):
        test_month_str = f"2025-{test_month:02d}"

        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(test_months)}] {test_month_str} (FIXED MODEL)")
        print(f"{'=' * 80}")

        # Load test data
        print(f"\n1. Loading test data...")
        test_file = os.path.join(data_dir, f"BTCUSDT-aggTrades-{test_month_str}.zip")
        if not os.path.exists(test_file):
            print(f"   ⚠️  {test_month_str} not found")
            continue

        test_df = load_and_process_file(test_file)
        if test_df is None or len(test_df) == 0:
            print(f"   ❌ No test data for {test_month_str}")
            continue

        # Add order flow features for test data
        try:
            test_df = add_order_flow_features(test_file, test_df)
            print(f"   ✓ Order flow features added")
        except Exception as e:
            print(f"   ⚠️  Order flow failed: {e}")

        print(f"   ✓ Test samples: {len(test_df):,} bars")

        # Engineer features for test data (using SAME feature engineer)
        print(f"\n2. Engineering features for test data...")
        test_df, _ = engineer_features(test_df, feature_engineer, fit=False)
        print(f"   ✓ Features engineered: {len(get_feature_columns(test_df))} features")

        # Create labels for test data
        print(f"\n3. Creating labels for test data...")
        test_df = create_labels(test_df)
        test_df = test_df.dropna()
        print(f"   ✓ Test samples: {len(test_df):,}")

        # Prepare test features
        X_test = test_df[feature_cols].values

        # Predict using the SAME model
        print(f"\n4. Generating predictions (FIXED MODEL)...")
        predictions = model.predict(X_test)

        # Backtest
        print(f"\n5. Running backtest...")
        results = simple_backtest(test_df, predictions)
        results["test_month"] = test_month_str
        results["train_samples"] = len(X_train)
        results["test_samples"] = len(X_test)
        results["num_features"] = len(feature_cols)
        results["training_method"] = "FIXED"

        all_results.append(results)

        print_backtest_results(results, f"{test_month_str} Results (FIXED)")

    # Summary
    print(f"\n" + "=" * 80)
    print(f"📊 SUMMARY (FIXED TRAINING)")
    print(f"=" * 80 + "\n")

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "monthly_results_fixed_2025.csv")
    results_df.to_csv(results_csv_path, index=False)

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
        "total_months_tested": len(results_df),
        "avg_return": float(results_df["total_return"].mean()),
        "avg_win_rate": float(results_df["win_rate"].mean()),
        "avg_profit_factor": float(results_df["profit_factor"].mean()),
        "avg_max_drawdown": float(results_df["max_drawdown"].mean()),
        "total_trades": int(results_df["total_trades"].sum()),
        "feature_engineering": "EnhancedFeatureEngineer",
        "training_method": "FIXED",
        "train_period": "2024 Q4 (Oct-Dec)",
        "test_period": "2025 Jan-Jun",
        "num_features": len(feature_cols),
    }

    summary_path = os.path.join(results_dir, "summary_fixed.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"   - monthly_results_fixed_2025.csv")
    print(f"   - summary_fixed.json")
    print(f"   - model_fixed_2024Q4.txt")

    print("\n" + "=" * 80)
    print("✅ Fixed training complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
