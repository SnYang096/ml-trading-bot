"""Optimized Monthly Rolling Training with Feature Selection, Incremental PCA, and Warm Start.

整合所有优化：
1. 重要性特征选择 + PCA降维
2. 增量PCA（不每月重新计算）
3. 使用Sharpe Ratio、Profit Factor、Max Drawdown评估
4. Warm Start滚动训练（保留旧知识）
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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


def load_top_features(n_features=100):
    """Load the top N most important features from previous analysis."""
    importance_file = "results/monthly_rolling_2025/feature_importance_with_names.csv"

    if not os.path.exists(importance_file):
        print(
            "❌ Feature importance file not found. Please run feature importance analysis first."
        )
        return None

    # Load feature importance
    df = pd.read_csv(importance_file)

    # Select top N features
    top_features = df.head(n_features)["feature"].tolist()

    print(f"✓ Loaded top {n_features} features from importance analysis")
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


def apply_incremental_pca(
    X_train, X_test, pca_model=None, scaler=None, n_components=64
):
    """Apply incremental PCA dimensionality reduction."""
    print(
        f"   Applying incremental PCA: {X_train.shape[1]} features -> {n_components} components"
    )

    # Standardize features before PCA
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Apply incremental PCA
    if pca_model is None:
        pca_model = IncrementalPCA(n_components=n_components, batch_size=1000)
        X_train_pca = pca_model.fit_transform(X_train_scaled)
    else:
        # Partial fit with new data
        pca_model.partial_fit(X_train_scaled)
        X_train_pca = pca_model.transform(X_train_scaled)

    X_test_pca = pca_model.transform(X_test_scaled)

    # Print explained variance
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"   ✓ Incremental PCA applied successfully")
    print(
        f"   ✓ Explained variance: {explained_variance[:5].sum():.3f} (first 5 components)"
    )
    print(
        f"   ✓ Cumulative variance: {cumulative_variance[-1]:.3f} (all {n_components} components)"
    )

    return X_train_pca, X_test_pca, pca_model, scaler


def calculate_strategy_metrics(results):
    """Calculate comprehensive strategy quality metrics."""
    returns = results["total_return"] / 100  # Convert percentage to decimal
    trades = results["total_trades"]
    win_rate = results["win_rate"] / 100
    profit_factor = results["profit_factor"]
    max_drawdown = abs(results["max_drawdown"]) / 100  # Convert to positive decimal

    # Sharpe Ratio (assuming monthly returns, risk-free rate = 0)
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(12)  # Annualized
    else:
        sharpe_ratio = 0

    # Calmar Ratio (return / max drawdown)
    if max_drawdown > 0:
        calmar_ratio = np.mean(returns) / max_drawdown
    else:
        calmar_ratio = 0

    # Win Rate Quality Score
    win_rate_score = win_rate if win_rate > 0.5 else 0

    # Profit Factor Quality Score
    pf_score = min(profit_factor, 3.0) / 3.0  # Cap at 3.0 for scoring

    # Drawdown Quality Score (lower is better)
    dd_score = max(0, 1 - max_drawdown)  # 1 - drawdown, higher is better

    # Overall Quality Score (weighted combination)
    quality_score = (
        0.3 * sharpe_ratio  # 30% Sharpe
        + 0.25 * pf_score  # 25% Profit Factor
        + 0.25 * dd_score  # 25% Drawdown
        + 0.2 * win_rate_score  # 20% Win Rate
    )

    return {
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "win_rate_score": win_rate_score,
        "pf_score": pf_score,
        "dd_score": dd_score,
        "quality_score": quality_score,
    }


def train_with_warm_start(X_train, y_train, prev_model=None, num_boost_round=50):
    """Train LightGBM with warm start from previous model."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }

    train_data = lgb.Dataset(X_train, label=y_train)

    if prev_model is not None:
        print(f"   🔥 Warm start: Using previous model as initialization")
        model = lgb.train(
            params,
            train_data,
            init_model=prev_model,
            num_boost_round=num_boost_round,
            keep_training_booster=True,
        )
    else:
        print(f"   🆕 Cold start: Training new model")
        model = lgb.train(params, train_data, num_boost_round=num_boost_round)

    return model


def main():
    data_dir = r"D:\GitHub\trading\rlbot\data\agg_data"

    print("\n" + "=" * 80)
    print("📊 Optimized Monthly Rolling Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Initial Train: 2024 Q4 (Oct-Dec)")
    print(f"   Test: 2025 Jan-Jun")
    print(f"   Feature Selection: Top 100 + PCA 64D")
    print(f"   Training: Warm Start (保留旧知识)")
    print(f"   Evaluation: Sharpe + PF + MaxDD + Quality Score")

    # Load top 100 features
    top_features = load_top_features(100)
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

    results_dir = "results/monthly_rolling_2025_optimized"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    feature_engineer = None
    pca_model = None
    scaler_model = None
    prev_model = None  # For warm start

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
    print(f"🔄 Starting Optimized Monthly Rolling Training")
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

        # Apply incremental PCA dimensionality reduction
        print(f"\n5. Applying incremental PCA dimensionality reduction...")
        X_train_pca, X_test_pca, pca_model, scaler_model = apply_incremental_pca(
            X_train, X_test, pca_model, scaler_model, n_components=64
        )

        # Train model with warm start
        print(f"\n6. Training LightGBM model (Warm Start)...")
        print(f"   Samples: {len(X_train_pca):,}, Features: {X_train_pca.shape[1]}")
        model = train_with_warm_start(
            X_train_pca, y_train, prev_model, num_boost_round=50
        )
        print("   ✓ Training complete")

        # Predict
        print(f"\n7. Generating predictions...")
        predictions = model.predict(X_test_pca)

        # Backtest
        print(f"\n8. Running backtest...")
        results = simple_backtest(test_df, predictions)
        results["test_month"] = test_month_str
        results["train_samples"] = len(X_train_pca)
        results["test_samples"] = len(X_test_pca)
        results["num_features"] = X_train_pca.shape[1]
        results["original_features"] = len(selected_features)

        # Calculate strategy quality metrics
        quality_metrics = calculate_strategy_metrics(results)
        results.update(quality_metrics)

        all_results.append(results)

        # Enhanced results display
        print(f"\n📊 {test_month_str} Results (Optimized)")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Return: {results['total_return']:.2f}%")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   Quality Score: {results['quality_score']:.3f}")

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

        # Update previous model for warm start
        prev_model = model

        # Add this month to training data for next iteration (expanding window)
        train_data.append(test_df)
        print(f"   ✓ Added {test_month_str} to training set for next iteration")

    # Enhanced Summary
    print(f"\n" + "=" * 80)
    print(f"📊 OPTIMIZED SUMMARY")
    print(f"=" * 80 + "\n")

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "monthly_results_optimized_2025.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(
        f"{'Month':<12} {'Trades':<8} {'Return':<10} {'Win%':<8} {'PF':<8} {'MaxDD':<10} {'Sharpe':<8} {'Quality':<8}"
    )
    print("-" * 100)
    for _, row in results_df.iterrows():
        print(
            f"{row['test_month']:<12} {row['total_trades']:<8} "
            f"{row['total_return']:>8.2f}% {row['win_rate']:>6.1f}% "
            f"{row['profit_factor']:>6.2f} {row['max_drawdown']:>8.2f}% "
            f"{row['sharpe_ratio']:>6.3f} {row['quality_score']:>6.3f}"
        )

    print("-" * 100)
    print(
        f"{'AVERAGE':<12} {results_df['total_trades'].mean():<8.1f} "
        f"{results_df['total_return'].mean():>8.2f}% "
        f"{results_df['win_rate'].mean():>6.1f}% "
        f"{results_df['profit_factor'].mean():>6.2f} "
        f"{results_df['max_drawdown'].mean():>8.2f}% "
        f"{results_df['sharpe_ratio'].mean():>6.3f} "
        f"{results_df['quality_score'].mean():>6.3f}"
    )

    # Save enhanced summary
    summary = {
        "total_months_tested": len(results_df),
        "avg_return": float(results_df["total_return"].mean()),
        "avg_win_rate": float(results_df["win_rate"].mean()),
        "avg_profit_factor": float(results_df["profit_factor"].mean()),
        "avg_max_drawdown": float(results_df["max_drawdown"].mean()),
        "avg_sharpe_ratio": float(results_df["sharpe_ratio"].mean()),
        "avg_quality_score": float(results_df["quality_score"].mean()),
        "total_trades": int(results_df["total_trades"].sum()),
        "feature_engineering": "EnhancedFeatureEngineer",
        "feature_selection": "Top100",
        "dimensionality_reduction": "IncrementalPCA64",
        "training_method": "WarmStart",
        "initial_train": "2024 Q4 (Oct-Dec)",
        "test_period": "2025 Jan-Jun",
        "num_features": 64,
        "original_features": int(results_df["original_features"].mean()),
    }

    summary_path = os.path.join(results_dir, "summary_optimized.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"   - monthly_results_optimized_2025.csv")
    print(f"   - summary_optimized.json")
    print(f"   - model_*.txt (one per month)")
    print(f"   - pca_*.pkl, scaler_*.pkl (PCA and scaler for each month)")

    print("\n" + "=" * 80)
    print("✅ Optimized training complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
