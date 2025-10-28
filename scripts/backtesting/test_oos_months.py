"""Test trained model on out-of-sample data (June-September 2025)."""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import zipfile
from datetime import datetime
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml_trading.data_tools.data_loader import MarketDataLoader


def extract_and_load_data(zip_path: str) -> pd.DataFrame:
    """Extract zip and load BTCUSDT data."""
    print(f"Loading data from {zip_path}...")

    # Create temp directory
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_oos_test")
    os.makedirs(temp_dir, exist_ok=True)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find CSV
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {zip_path}")

    csv_path = os.path.join(temp_dir, csv_files[0])

    # Load data
    df = pd.read_csv(csv_path)

    # Convert timestamp
    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("No timestamp column found")

    df.set_index("timestamp", inplace=True)

    # Convert to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Remove invalid data
    df = df.dropna(subset=["price", "quantity"])

    # Create OHLCV
    ohlc_dict = {"price": ["first", "max", "min", "last"], "quantity": "sum"}

    resampled = df.groupby(pd.Grouper(freq="1s")).agg(ohlc_dict)
    resampled.columns = ["open", "high", "low", "close", "volume"]
    resampled = resampled.dropna().ffill()

    # Add microstructure features
    try:
        agg = pd.read_csv(csv_path)
        if "transact_time" in agg.columns:
            agg["timestamp"] = pd.to_datetime(agg["transact_time"], unit="ms")
        else:
            agg["timestamp"] = pd.to_datetime(agg["timestamp"])
        agg["price"] = pd.to_numeric(agg["price"], errors="coerce")
        agg["quantity"] = pd.to_numeric(agg["quantity"], errors="coerce")
        agg = agg.dropna(subset=["price", "quantity"])

        if "is_buyer_maker" in agg.columns:
            agg["taker_buy"] = (~agg["is_buyer_maker"].astype(bool)).astype(int)
        else:
            agg["taker_buy"] = 0

        agg["buy_qty"] = np.where(agg["taker_buy"] == 1, agg["quantity"], 0.0)
        agg["sell_qty"] = np.where(agg["taker_buy"] == 1, 0.0, agg["quantity"])
        agg = agg.set_index("timestamp")

        per_sec = agg.groupby(pd.Grouper(freq="1s")).agg(
            {"buy_qty": "sum", "sell_qty": "sum"}
        )
        per_sec["taker_buy_ratio"] = per_sec["buy_qty"] / (
            per_sec["buy_qty"] + per_sec["sell_qty"]
        ).replace(0, np.nan)
        per_sec["taker_buy_ratio"] = per_sec["taker_buy_ratio"].fillna(0.5)
        per_sec["cvd"] = (per_sec["buy_qty"] - per_sec["sell_qty"]).cumsum()

        resampled = (
            resampled.join(
                per_sec[["buy_qty", "sell_qty", "taker_buy_ratio", "cvd"]], how="left"
            )
            .ffill()
            .fillna(0)
        )
    except Exception as e:
        print(f"Warning: failed to compute microstructure features: {e}")

    print(f"  Loaded {len(resampled)} bars")
    print(f"  Time range: {resampled.index[0]} to {resampled.index[-1]}")
    print(
        f"  Price range: {resampled['close'].min():.2f} to {resampled['close'].max():.2f}"
    )

    # Cleanup
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return resampled


def test_on_month(model_data: dict, month_zip: str, month_name: str) -> dict:
    """Test model on one month of data."""
    print(f"\n{'='*70}")
    print(f"Testing on {month_name}")
    print("=" * 70)

    # Load test data
    test_data = extract_and_load_data(month_zip)

    # Get strategy components
    strategy = model_data["strategy"]
    data_loader = model_data["data_loader"]
    feature_engineer = model_data["feature_engineer"]

    # Set test data
    data_loader.raw_data = test_data

    # Get multi-timeframe data
    print("\nCreating multi-timeframe data...")
    multi_tf_data = data_loader.get_multi_timeframe_data()

    # Engineer features (using fitted scalers from training)
    print("Engineering features (using training scalers)...")
    engineered_data = feature_engineer.engineer_features(multi_tf_data, fit=False)

    # Make predictions for each timeframe
    print("\nGenerating predictions...")
    results = {}

    for timeframe, data in engineered_data.items():
        print(f"\n  Timeframe: {timeframe}")

        # Get models
        stage1_model = strategy.pipeline.stage1_models.get(timeframe)
        stage2_model = strategy.pipeline.stage2_models.get(timeframe)

        if stage1_model is None or stage2_model is None:
            print(f"    No models found for {timeframe}, skipping...")
            continue

        # Prepare features
        feature_columns = [
            col
            for col in data.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X = data[feature_columns]

        # Prepare targets for evaluation
        y_stage1, y_stage2 = strategy.pipeline.prepare_targets(data)

        # Remove NaN
        valid_indices = ~(X.isna().any(axis=1) | y_stage1.isna() | y_stage2.isna())
        X_clean = X[valid_indices]
        y_stage1_clean = y_stage1[valid_indices]
        y_stage2_clean = y_stage2[valid_indices]

        if len(X_clean) == 0:
            print(f"    No valid data after cleaning")
            continue

        print(f"    Valid samples: {len(X_clean)}")

        # Predict
        try:
            pred_stage1 = stage1_model.predict(X_clean)
            pred_stage2 = stage2_model.predict(X_clean)

            # Evaluate Stage 1 (Classification)
            from sklearn.metrics import accuracy_score, classification_report

            pred_stage1_binary = (pred_stage1 > 0.5).astype(int)

            # Map to -1, 0, 1 like the target
            # The model predicts probability, so we need to map y_stage1 to binary first
            y_stage1_binary = (y_stage1_clean == 1).astype(
                int
            )  # 1 if long signal, 0 otherwise

            accuracy = accuracy_score(y_stage1_binary, pred_stage1_binary)

            # Evaluate Stage 2 (Regression)
            from sklearn.metrics import mean_squared_error, mean_absolute_error

            mse = mean_squared_error(y_stage2_clean, pred_stage2)
            mae = mean_absolute_error(y_stage2_clean, pred_stage2)
            rmse = np.sqrt(mse)

            # Calculate trading metrics
            # Simulate simple trading: long when pred_stage1 > 0.5
            returns = y_stage2_clean.values
            signals = pred_stage1_binary

            strategy_returns = returns * signals  # Returns only when we have a signal

            total_return = strategy_returns.sum()
            win_rate = (strategy_returns > 0).sum() / max((signals == 1).sum(), 1)
            avg_win = (
                strategy_returns[strategy_returns > 0].mean()
                if (strategy_returns > 0).any()
                else 0
            )
            avg_loss = (
                strategy_returns[strategy_returns < 0].mean()
                if (strategy_returns < 0).any()
                else 0
            )

            results[timeframe] = {
                "accuracy": accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "total_return": total_return,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "num_signals": (signals == 1).sum(),
                "num_samples": len(X_clean),
            }

            print(f"    Stage 1 Accuracy: {accuracy:.4f}")
            print(f"    Stage 2 RMSE: {rmse:.6f}")
            print(
                f"    Trading: {(signals == 1).sum()} signals, {win_rate:.2%} win rate"
            )
            print(f"    Total Return: {total_return:.6f}")

        except Exception as e:
            print(f"    Error during prediction: {e}")
            import traceback

            traceback.print_exc()

    return results


def main():
    """Test model on all out-of-sample months."""
    print("=" * 70)
    print("Testing Model on Out-of-Sample Data (June-September 2025)")
    print("=" * 70)

    # Load trained model
    model_path = "trained_model_wavelet_may_2025.pkl"
    print(f"\nLoading trained model from {model_path}...")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    print(f"✅ Model loaded successfully")
    print(f"   Training date: {model_data['training_date']}")
    print(f"   Data info: {model_data['data_info']['date_range']}")

    # Test on each month
    months = {
        "June 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-06.zip",
        "July 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-07.zip",
        "August 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-08.zip",
        "September 2025": "data/aggTrades/BTCUSDT-aggTrades-2025-09.zip",
    }

    all_results = {}

    for month_name, zip_path in months.items():
        if not os.path.exists(zip_path):
            print(f"\n⚠️  File not found: {zip_path}")
            continue

        try:
            results = test_on_month(model_data, zip_path, month_name)
            all_results[month_name] = results
        except Exception as e:
            print(f"\n❌ Error testing {month_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    print("\n" + "=" * 70)
    print("SUMMARY OF OUT-OF-SAMPLE TESTING")
    print("=" * 70)

    summary = {}
    for month_name, results in all_results.items():
        print(f"\n{month_name}:")
        summary[month_name] = {}

        for tf, metrics in results.items():
            print(f"\n  {tf}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    RMSE: {metrics['rmse']:.6f}")
            print(f"    Win Rate: {metrics['win_rate']:.2%}")
            print(f"    Total Return: {metrics['total_return']:.6f}")
            print(f"    Signals: {metrics['num_signals']}")

            summary[month_name][tf] = {
                "accuracy": float(metrics["accuracy"]),
                "rmse": float(metrics["rmse"]),
                "win_rate": float(metrics["win_rate"]),
                "total_return": float(metrics["total_return"]),
                "num_signals": int(metrics["num_signals"]),
            }

    # Save to JSON
    output_file = "oos_test_results_with_timeseries_cv.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print("\n🎉 Out-of-sample testing completed!")


if __name__ == "__main__":
    main()
