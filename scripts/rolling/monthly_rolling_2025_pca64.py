"""Monthly Rolling Training with PCA Dimensionality Reduction.

使用PCA将特征降维到64维，减少过拟合风险
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def apply_pca_reduction(X_train, X_test, n_components=64):
    """Apply PCA dimensionality reduction to features."""
    print(
        f"   Applying PCA reduction to {X_train.shape[1]} features -> {n_components} components"
    )

    # Standardize features before PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"   ✓ PCA applied successfully")
    print(
        f"   ✓ Explained variance: {explained_variance[:5].sum():.3f} (first 5 components)"
    )
    print(
        f"   ✓ Cumulative variance: {cumulative_variance[-1]:.3f} (all {n_components} components)"
    )

    return X_train_pca, X_test_pca, pca, scaler


def main():
    data_dir = r"D:\GitHub\trading\rlbot\data\agg_data"

    print("\n" + "=" * 80)
    print("📊 Monthly Rolling Training (PCA 64D)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Initial Train: 2024 Q4 (Oct-Dec)")
    print(f"   Test: 2025 Jan-Jun")
    print(f"   Dimensionality Reduction: PCA to 64 components")
    print(f"   Feature Engineering: EnhancedFeatureEngineer (WPT + Hurst + OrderFlow)")

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

    results_dir = "results/monthly_rolling_2025_pca64"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    feature_engineer = None  # Will be created in first iteration
    pca_model = None  # Will be created in first iteration
    scaler_model = None  # Will be created in first iteration

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
    print(f"🔄 Starting Monthly Rolling Training (PCA 64D)")
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

        # Create labels
        print(f"\n3. Creating labels...")
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

        # Apply PCA dimensionality reduction
        print(f"\n4. Applying PCA dimensionality reduction...")
        X_train_pca, X_test_pca, pca_model, scaler_model = apply_pca_reduction(
            X_train, X_test, n_components=64
        )

        # Train model
        print(f"\n5. Training LightGBM model (PCA 64D)...")
        print(f"   Samples: {len(X_train_pca):,}, Features: {X_train_pca.shape[1]}")
        model = train_lightgbm_model(
            X_train_pca, y_train, use_gpu=True, num_boost_round=200
        )
        print("   ✓ Training complete")

        # Predict
        print(f"\n6. Generating predictions...")
        predictions = model.predict(X_test_pca)

        # Backtest
        print(f"\n7. Running backtest...")
        results = simple_backtest(test_df, predictions)
        results["test_month"] = test_month_str
        results["train_samples"] = len(X_train_pca)
        results["test_samples"] = len(X_test_pca)
        results["num_features"] = X_train_pca.shape[1]
        results["dimensionality_reduction"] = "PCA64"
        results["original_features"] = len(feature_cols)

        all_results.append(results)

        print_backtest_results(results, f"{test_month_str} Results (PCA 64D)")

        # Save model
        model_path = os.path.join(results_dir, f"model_{test_month_str}.txt")
        model.save_model(model_path)
        print(f"\n   💾 Model saved: {model_path}")

        # Save PCA and scaler for this month
        pca_path = os.path.join(results_dir, f"pca_{test_month_str}.pkl")
        scaler_path = os.path.join(results_dir, f"scaler_{test_month_str}.pkl")

        import pickle

        with open(pca_path, "wb") as f:
            pickle.dump(pca_model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_model, f)

        print(f"   💾 PCA and Scaler saved: {pca_path}, {scaler_path}")

        # Add this month to training data for next iteration (expanding window)
        train_data.append(test_df)
        print(f"   ✓ Added {test_month_str} to training set for next iteration")

    # Summary
    print(f"\n" + "=" * 80)
    print(f"📊 SUMMARY (PCA 64D)")
    print(f"=" * 80 + "\n")

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "monthly_results_pca64_2025.csv")
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
        "dimensionality_reduction": "PCA64",
        "initial_train": "2024 Q4 (Oct-Dec)",
        "test_period": "2025 Jan-Jun",
        "num_features": 64,
        "original_features": int(results_df["original_features"].mean()),
    }

    summary_path = os.path.join(results_dir, "summary_pca64.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"   - monthly_results_pca64_2025.csv")
    print(f"   - summary_pca64.json")
    print(f"   - model_*.txt (one per month)")
    print(f"   - pca_*.pkl, scaler_*.pkl (PCA and scaler for each month)")

    print("\n" + "=" * 80)
    print("✅ PCA 64D training complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
