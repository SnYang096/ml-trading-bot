"""Backtest script for BTCUSDT data using the ML trading strategy."""

import sys
import os
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.ml_trading.strategies.ml_strategy import MLTradingStrategy
from src.ml_trading.data.data_loader import MarketDataLoader
from src.ml_trading.data.feature_engineering import FeatureEngineer


def extract_zip_data(zip_path: str) -> str:
    """Extract zip file and return path to CSV file."""
    print(f"Extracting {zip_path}...")

    # Create a temporary directory for extraction
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find the CSV file
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the zip archive")

    csv_path = os.path.join(temp_dir, csv_files[0])
    print(f"Extracted CSV file: {csv_path}")
    return csv_path


def load_btcusdt_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess BTCUSDT aggregate trade data."""
    print(f"Loading BTCUSDT data from {csv_path}...")

    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Display first few rows to understand the structure
    print("\nFirst 5 rows:")
    print(df.head())

    # Convert timestamp to datetime
    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("No timestamp column found")

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)

    # Convert price and quantity to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Remove rows with invalid data
    df = df.dropna(subset=["price", "quantity"])

    print(f"After cleaning: {len(df)} records")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")

    return df


def create_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert aggregate trade data to OHLCV format."""
    print("Converting aggregate trades to OHLCV...")

    # Resample to 1-second bars
    ohlc_dict = {"price": ["first", "max", "min", "last"], "quantity": "sum"}

    # Group by 1-second intervals
    resampled = df.groupby(pd.Grouper(freq="1s")).agg(ohlc_dict)

    # Flatten column names
    resampled.columns = ["open", "high", "low", "close", "volume"]

    # Remove rows with all NaN values
    resampled = resampled.dropna()

    # Forward fill any remaining NaN values
    resampled = resampled.ffill()

    print(f"Created {len(resampled)} 1-second OHLCV bars")
    print(f"Time range: {resampled.index[0]} to {resampled.index[-1]}")

    return resampled


def run_backtest():
    """Run the complete backtest with BTCUSDT data."""
    print("🚀 BTCUSDT ML Trading Strategy Backtest")
    print("=" * 50)

    # Path to the zip file
    zip_path = "/home/yin/trading/rlbot/ml_project/BTCUSDT-aggTrades-2025-05-05.zip"

    # Check if file exists
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return

    print(f"✅ Zip file found: {zip_path}")

    try:
        # Extract zip file
        csv_path = extract_zip_data(zip_path)

        # Load BTCUSDT data
        raw_data = load_btcusdt_data(csv_path)

        # Create OHLCV data
        ohlcv_data = create_ohlcv_data(raw_data)

        # Initialize data loader with the OHLCV data
        print("\n1. Initializing data loader...")
        data_loader = MarketDataLoader()
        data_loader.raw_data = ohlcv_data  # Set the processed data directly

        # Get multi-timeframe data
        print("\n2. Creating multi-timeframe data...")
        multi_tf_data = data_loader.get_multi_timeframe_data()

        print(f"   ✓ Created data for timeframes: {list(multi_tf_data.keys())}")
        for tf, data in multi_tf_data.items():
            print(f"     - {tf}: {len(data)} bars")
            if len(data) > 0:
                print(
                    f"       Price range: {data['close'].min():.2f} to {data['close'].max():.2f}"
                )

        # Feature engineering
        print("\n3. Engineering features...")
        feature_engineer = FeatureEngineer()
        engineered_data = feature_engineer.engineer_features(multi_tf_data)

        print(f"   ✓ Engineered features for all timeframes")
        for tf, data in engineered_data.items():
            print(f"     - {tf}: {data.shape[1]} features, {len(data)} rows")

        # Initialize and train strategy
        print("\n4. Training ML strategy...")
        strategy = MLTradingStrategy()
        strategy.data_loader = data_loader  # Use our data loader

        # Train the strategy
        metrics = strategy.train_strategy()
        print("   ✓ Strategy training completed")

        # Print training metrics
        print("\n   Training Metrics:")
        for stage, stage_metrics in metrics.items():
            print(f"     {stage.upper()}:")
            for timeframe, metrics in stage_metrics.items():
                print(f"       {timeframe}: {metrics}")

        # Generate signals for 5T timeframe
        print("\n5. Generating trading signals (5T timeframe)...")

        # Get 5T data
        data_5t = engineered_data["5T"]

        if len(data_5t) < 10:
            print("   ✗ Not enough data for signal generation")
            return

        # Prepare features
        feature_columns = [
            col
            for col in data_5t.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X_5t = data_5t[feature_columns]

        # Remove rows with any NaN values
        X_5t_clean = X_5t.dropna()
        if X_5t_clean.empty:
            print("   ✗ No valid data for prediction after cleaning")
            return

        print(f"   ✓ Using {len(X_5t_clean)} clean data points for prediction")

        # Generate predictions
        stage1_model = strategy.pipeline.stage1_models["5T"]
        stage2_model = strategy.pipeline.stage2_models["5T"]

        stage1_preds = stage1_model.predict(X_5t_clean)
        stage2_preds = stage2_model.predict(X_5t_clean)

        print(f"   ✓ Generated {len(stage1_preds)} stage 1 predictions")
        print(f"   ✓ Generated {len(stage2_preds)} stage 2 predictions")

        # Create signals DataFrame
        signals = pd.DataFrame(
            {
                "timestamp": X_5t_clean.index,
                "stage1_pred": stage1_preds,
                "stage2_pred": stage2_preds,
                "discrete_signal": 0,
            }
        )

        # Convert continuous signal to discrete (-1, 0, 1)
        signals.loc[stage1_preds > 0.6, "discrete_signal"] = 1  # Long
        signals.loc[stage1_preds < 0.4, "discrete_signal"] = -1  # Short

        # Add price information for analysis
        price_data = data_5t[["open", "high", "low", "close", "volume"]].loc[
            X_5t_clean.index
        ]
        signals = pd.concat([signals, price_data], axis=1)

        # Reset index to ensure proper alignment
        signals = signals.reset_index(drop=True)

        # Save signals
        signals.to_csv("btcusdt_backtest_signals.csv", index=False)
        print(f"\n   ✓ Saved signals to btcusdt_backtest_signals.csv")

        # Print signal statistics
        print(f"\n   Signal Statistics:")
        print(f"     Total signals: {len(signals)}")
        print(f"     Long signals (1): {len(signals[signals['discrete_signal'] == 1])}")
        print(
            f"     Short signals (-1): {len(signals[signals['discrete_signal'] == -1])}"
        )
        print(f"     Hold signals (0): {len(signals[signals['discrete_signal'] == 0])}")

        print(f"\n   Stage 1 Predictions:")
        print(f"     Min: {stage1_preds.min():.4f}")
        print(f"     Max: {stage1_preds.max():.4f}")
        print(f"     Mean: {stage1_preds.mean():.4f}")

        print(f"\n   Stage 2 Predictions:")
        print(f"     Min: {stage2_preds.min():.6f}")
        print(f"     Max: {stage2_preds.max():.6f}")
        print(f"     Mean: {stage2_preds.mean():.6f}")

        # Show sample signals
        print(f"\n   Sample signals:")
        print(
            signals[
                ["timestamp", "close", "stage1_pred", "stage2_pred", "discrete_signal"]
            ].head(10)
        )

        print("\n🎉 Backtest completed successfully!")
        print("\nNext steps:")
        print("1. Check btcusdt_backtest_signals.csv for the generated signals")
        print("2. Run python view_btcusdt_signals.py to analyze the results")
        print("3. Use these signals for further backtesting with Nautilus Trader")

    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up temporary files
        temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract")
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    run_backtest()
