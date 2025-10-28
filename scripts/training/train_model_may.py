"""Train ML model using May 2025 BTCUSDT data."""

import sys
import os
import zipfile
import pandas as pd
import numpy as np
import pickle
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


def train_model():
    """Train ML model using May 2025 BTCUSDT data."""
    print("🚀 Training ML Model with May 2025 BTCUSDT Data")
    print("=" * 60)

    # Path to the zip file
    zip_path = "/home/yin/trading/rlbot/ml_project/BTCUSDT-aggTrades-2025-05.zip"

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
        data_loader.raw_data = ohlcv_data

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
        strategy.data_loader = data_loader

        # Train the strategy
        metrics = strategy.train_strategy()
        print("   ✓ Strategy training completed")

        # Print training metrics
        print("\n   Training Metrics:")
        for stage, stage_metrics in metrics.items():
            print(f"     {stage.upper()}:")
            for timeframe, metrics in stage_metrics.items():
                print(f"       {timeframe}: {metrics}")

        # Save the trained model
        print("\n5. Saving trained model...")
        model_data = {
            "strategy": strategy,
            "data_loader": data_loader,
            "feature_engineer": feature_engineer,
            "engineered_data": engineered_data,
            "metrics": metrics,
            "training_date": datetime.now(),
            "data_info": {
                "total_bars": len(ohlcv_data),
                "timeframes": {tf: len(data) for tf, data in multi_tf_data.items()},
                "price_range": (ohlcv_data["close"].min(), ohlcv_data["close"].max()),
                "date_range": (ohlcv_data.index[0], ohlcv_data.index[-1]),
            },
        }

        # Save model to pickle file
        model_path = "trained_model_may_2025.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"   ✓ Model saved to {model_path}")

        # Save model info
        model_info = {
            "model_path": model_path,
            "training_date": datetime.now().isoformat(),
            "data_source": "BTCUSDT-aggTrades-2025-05.zip",
            "total_bars": len(ohlcv_data),
            "timeframes": {tf: len(data) for tf, data in multi_tf_data.items()},
            "price_range": (
                float(ohlcv_data["close"].min()),
                float(ohlcv_data["close"].max()),
            ),
            "date_range": (
                ohlcv_data.index[0].isoformat(),
                ohlcv_data.index[-1].isoformat(),
            ),
            "metrics": metrics,
        }

        import json

        with open("model_info_may_2025.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"   ✓ Model info saved to model_info_may_2025.json")

        print("\n🎉 Model training completed successfully!")
        print("\nNext steps:")
        print("1. Use the trained model for backtesting")
        print("2. Run vectorbot backtest with stop loss and take profit")
        print("3. Implement position sizing and scaling logic")

    except Exception as e:
        print(f"❌ Error during training: {e}")
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
    train_model()
