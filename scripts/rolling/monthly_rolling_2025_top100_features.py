"""Monthly Rolling Training with Top 100 Features Selection.

选择最重要的100个特征进行训练，减少过拟合风险
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


def load_top_features():
    """Load the top 100 most important features from previous analysis."""
    importance_file = "results/monthly_rolling_2025/feature_importance_with_names.csv"

    if not os.path.exists(importance_file):
        print(
            "❌ Feature importance file not found. Please run feature importance analysis first."
        )
        return None

    # Load feature importance
    df = pd.read_csv(importance_file)

    # Select top 100 features
    top_features = df.head(100)["feature"].tolist()

    print(f"✓ Loaded top 100 features from importance analysis")
    print(f"  Top 5: {top_features[:5]}")

    return top_features


def filter_features(df, selected_features):
    """Filter DataFrame to only include selected features."""
    # Get all available features
    available_features = get_feature_columns(df)

    # Find which selected features are available
    features_to_keep = [f for f in selected_features if f in available_features]
    missing_features = [f for f in selected_features if f not in available_features]

    if missing_features:
        print(f"⚠️  Missing {len(missing_features)} features: {missing_features[:5]}...")

    print(
        f"✓ Using {len(features_to_keep)} out of {len(selected_features)} selected features"
    )

    return features_to_keep


def main():
    data_dir = r"D:\GitHub\trading\rlbot\data\agg_data"

    print("\n" + "=" * 80)
    print("📊 Monthly Rolling Training (Top 100 Features)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Initial Train: 2024 Q4 (Oct-Dec)")
    print(f"   Test: 2025 Jan-Jun")
    print(f"   Feature Selection: Top 100 Most Important Features")
    print(f"   Feature Engineering: EnhancedFeatureEngineer (WPT + Hurst + OrderFlow)")

    # Load top 100 features
    top_features = load_top_features()
    if top_features is None:
        return

    # Load 2024 Q4 as initial training (Oct, Nov, Dec)
    print(f"\n🔍 Finding data files...")
    train_files = []
    for month in [10, 11, 12]:
        file_path = os.path.join(data_dir, f"BTCUSDT-aggTrades-2024-{month:02d}.zip")
        if os.path.exists(file_path):
            train_files.append(file_path)

    print(f"   Found {len(train_files)} months for initial training (2024 Q4)")

    # Test on 2025 Jan-Jun
    test_months = [1, 2, 3, 4, 5, 6]

    results_dir = "results/monthly_rolling_2025_top100"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    feature_engineer = None  # Will be created in first iteration

    # Load initial training data
    print(f"\n📥 Loading initial training data (2024 Q4)...")
    train_data = []
    for fp in train_files:
        print(f"   Loading {os.path.basename(fp)}")
        df = load_and_process_file(fp)
        if df is not None and len(df) > 0:
            # Add order flow features
            try:
                df = add_order_flow_features(fp, df)
            except Exception as _:
                pass
            train_data.append(df)

    print(f"\n" + "=" * 80)
    print(f"🔄 Starting Monthly Rolling Training (Top 100 Features)")
    print(f"=" * 80 + "\n")

    for i, test_month in enumerate(test_months, 1):
        test_month_str = f"2025-{test_month:02d}"

        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(test_months)}] {test_month_str}")
        print(f"{'=' * 80}")

        # Prepare training data
        train_df = pd.concat(train_data, axis=0).sort_index()
        print(f"   Training samples: {len(train_df):,} bars")

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
        except Exception as _:
            pass

        print(f"   ✓ Test samples: {len(test_df):,} bars")

        # Engineer features using EnhancedFeatureEngineer
        print(f"\n2. Engineering enhanced features...")
        print(f"   Features: WPT + Hurst + Hilbert + Spectral + Advanced Derived")
        train_df, feature_engineer = engineer_features(
            train_df, feature_engineer, fit=True
        )
        test_df, _ = engineer_features(test_df, feature_engineer, fit=False)
        print(
            f"   ✓ Features engineered: {len(get_feature_columns(train_df))} features"
        )

        # Filter to top 100 features
        print(f"\n3. Selecting top 100 features...")
        selected_features = filter_features(train_df, top_features)

        # Create labels
        print(f"\n4. Creating labels...")
        train_df = create_labels(train_df)
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        print(f"   ✓ Train samples: {len(train_df):,}")
        print(f"   ✓ Test samples: {len(test_df):,}")

        # Prepare features (only selected ones)
        X_train = train_df[selected_features].values
        y_train = train_df["binary_signal"].values
        X_test = test_df[selected_features].values

        # Train model
        print(f"\n5. Training LightGBM model (Top 100 Features)...")
        print(f"   Samples: {len(X_train):,}, Features: {len(selected_features)}")
        model = train_lightgbm_model(
            X_train, y_train, use_gpu=True, num_boost_round=200
        )
        print("   ✓ Training complete")

        # Predict
        print(f"\n6. Generating predictions...")
        predictions = model.predict(X_test)

        # Backtest
        print(f"\n7. Running backtest...")
        results = simple_backtest(test_df, predictions)
        results["test_month"] = test_month_str
        results["train_samples"] = len(X_train)
        results["test_samples"] = len(X_test)
        results["num_features"] = len(selected_features)
        results["feature_selection"] = "Top100"

        all_results.append(results)

        print_backtest_results(results, f"{test_month_str} Results (Top 100)")

        # Save model
        model_path = os.path.join(results_dir, f"model_{test_month_str}.txt")
        model.save_model(model_path)
        print(f"\n   💾 Model saved: {model_path}")

        # Add this month to training data for next iteration (expanding window)
        train_data.append(test_df)
        print(f"   ✓ Added {test_month_str} to training set for next iteration")

    # Summary
    print(f"\n" + "=" * 80)
    print(f"📊 SUMMARY (Top 100 Features)")
    print(f"=" * 80 + "\n")

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "monthly_results_top100_2025.csv")
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
        "feature_selection": "Top100",
        "initial_train": "2024 Q4 (Oct-Dec)",
        "test_period": "2025 Jan-Jun",
        "num_features": 100,
    }

    summary_path = os.path.join(results_dir, "summary_top100.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"   - monthly_results_top100_2025.csv")
    print(f"   - summary_top100.json")
    print(f"   - model_*.txt (one per month)")

    print("\n" + "=" * 80)
    print("✅ Top 100 features training complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
