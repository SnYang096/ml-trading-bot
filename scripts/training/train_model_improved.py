"""Train ML model with improved feature engineering and normalization."""

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
from src.ml_trading.data.feature_engineering_improved import ImprovedFeatureEngineer


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


def train_improved_model():
    """Train ML model with improved feature engineering."""
    print("🚀 Training Improved ML Model with Normalization")
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

        # Feature engineering with normalization
        print("\n3. Engineering features with normalization...")
        feature_engineer = ImprovedFeatureEngineer(scaler_type="standard")
        engineered_data = feature_engineer.engineer_features(multi_tf_data, fit=True)

        print(f"   ✓ Engineered and normalized features for all timeframes")
        for tf, data in engineered_data.items():
            print(f"     - {tf}: {data.shape[1]} features, {len(data)} rows")

            # Show feature statistics
            feature_cols = [
                col
                for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            if feature_cols:
                print(f"       Feature columns: {len(feature_cols)}")
                print(f"       Sample feature stats:")
                for col in feature_cols[:5]:  # Show first 5 features
                    print(
                        f"         {col}: mean={data[col].mean():.4f}, std={data[col].std():.4f}"
                    )

        # Initialize and train strategy
        print("\n4. Training ML strategy...")
        strategy = MLTradingStrategy()
        strategy.data_loader = data_loader
        strategy.feature_engineer = feature_engineer  # Use improved feature engineer

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
        model_path = "trained_model_improved_may_2025.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"   ✓ Model saved to {model_path}")

        # Save scalers separately
        scaler_path = "feature_scalers_may_2025.pkl"
        feature_engineer.save_scalers(scaler_path)

        # Save model info
        model_info = {
            "model_path": model_path,
            "scaler_path": scaler_path,
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
            "feature_engineering": "improved_with_normalization",
            "scaler_type": "standard",
        }

        import json

        with open("model_info_improved_may_2025.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"   ✓ Model info saved to model_info_improved_may_2025.json")

        # Show feature importance info
        print(f"\n📊 Feature Engineering Info:")
        for tf in multi_tf_data.keys():
            feature_info = feature_engineer.get_feature_importance_info(tf)
            if feature_info:
                print(f"   {tf}: {len(feature_info['mean'])} features normalized")
                print(f"     Scaler type: {feature_info['scaler_type']}")

        print("\n🎉 Improved model training completed successfully!")
        print("\nKey improvements:")
        print("1. ✅ Feature normalization with StandardScaler")
        print(
            "2. ✅ Additional normalized features (BB position, RSI normalized, etc.)"
        )
        print("3. ✅ Price-relative indicators (ATR/price, MACD/price)")
        print("4. ✅ Momentum and moving average ratios")
        print("5. ✅ Proper handling of NaN values")

        print("\nNext steps:")
        print("1. Use the improved model for backtesting")
        print("2. Compare performance with original model")
        print("3. Fine-tune normalization parameters if needed")

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
    train_improved_model()
