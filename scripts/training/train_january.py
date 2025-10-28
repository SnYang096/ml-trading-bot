"""Train ML model with GPU using January 2025 data."""

import sys
import os
import zipfile
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml_trading.strategies.ml_strategy import MLTradingStrategy
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.data_tools.feature_engineering_wavelet import WaveletFeatureEngineer


def extract_zip_data(zip_path: str) -> str:
    """Extract zip file and return path to CSV file."""
    print(f"📦 Extracting {zip_path}...")

    # Create a temporary directory for extraction
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract_jan")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find the CSV file
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the zip archive")

    csv_path = os.path.join(temp_dir, csv_files[0])
    print(f"✅ Extracted CSV file: {csv_path}")
    return csv_path


def load_btcusdt_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess BTCUSDT aggregate trade data."""
    print(f"📊 Loading BTCUSDT data from {csv_path}...")

    # Load the data
    df = pd.read_csv(csv_path)
    print(f"   Raw data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

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

    print(f"   After cleaning: {len(df)} records")
    print(f"   Time range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")

    return df


def create_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert aggregate trade data to OHLCV format."""
    print("🔄 Converting aggregate trades to OHLCV...")

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

    print(f"   Created {len(resampled)} 1-second OHLCV bars")
    print(f"   Time range: {resampled.index[0]} to {resampled.index[-1]}")

    return resampled


def train_january_model():
    """Train ML model with wavelet transform features using January 2025 data."""
    print("\n" + "=" * 70)
    print("🚀 GPU Training: January 2025 → Model Generation")
    print("=" * 70 + "\n")

    # Path to the zip file in agg_data directory
    zip_path = r"D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-01.zip"

    # Check if file exists
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        print("\n💡 Please download the data first using:")
        print("   .\\download_to_agg_data.ps1")
        return

    print(f"✅ Zip file found: {zip_path}")
    print(f"   File size: {os.path.getsize(zip_path) / (1024**3):.2f} GB\n")

    try:
        # Extract zip file
        csv_path = extract_zip_data(zip_path)

        # Load BTCUSDT data
        raw_data = load_btcusdt_data(csv_path)

        # Create OHLCV data
        ohlcv_data = create_ohlcv_data(raw_data)

        # Derive microstructure series from agg trades for wavelet: CVD and taker_buy_ratio
        print("📈 Computing microstructure features (CVD, taker_buy_ratio)...")
        try:
            agg = pd.read_csv(csv_path)
            if "transact_time" in agg.columns:
                agg["timestamp"] = pd.to_datetime(agg["transact_time"], unit="ms")
            else:
                agg["timestamp"] = pd.to_datetime(agg["timestamp"])

            agg["price"] = pd.to_numeric(agg["price"], errors="coerce")
            agg["quantity"] = pd.to_numeric(agg["quantity"], errors="coerce")
            agg = agg.dropna(subset=["price", "quantity"])

            # Classify taker side
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

            # Align into ohlcv_data index
            ohlcv_data = (
                ohlcv_data.join(
                    per_sec[["buy_qty", "sell_qty", "taker_buy_ratio", "cvd"]],
                    how="left",
                )
                .ffill()
                .fillna(0)
            )

            print("   ✅ Microstructure features computed successfully")
        except Exception as e:
            print(f"   ⚠️ Warning: failed to compute microstructure series: {e}")

        # Initialize data loader with the OHLCV data
        print("\n🔧 Step 1: Initializing data loader...")
        data_loader = MarketDataLoader()
        data_loader.raw_data = ohlcv_data

        # Get multi-timeframe data
        print("\n🔧 Step 2: Creating multi-timeframe data...")
        multi_tf_data = data_loader.get_multi_timeframe_data()

        print(f"   ✓ Created data for timeframes: {list(multi_tf_data.keys())}")
        for tf, data in multi_tf_data.items():
            print(f"     - {tf}: {len(data)} bars")
            if len(data) > 0:
                print(
                    f"       Price range: {data['close'].min():.2f} to {data['close'].max():.2f}"
                )

        # Feature engineering with wavelet transform
        print("\n🔧 Step 3: Engineering features with wavelet transform...")
        print("   Using: db4 wavelet, 4 levels, StandardScaler")
        feature_engineer = WaveletFeatureEngineer(
            scaler_type="standard", wavelet="db4", wavelet_levels=4
        )
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

        # Initialize and train strategy
        print("\n🔧 Step 4: Training ML strategy with GPU acceleration...")
        print("   This may take several minutes depending on data size...")
        strategy = MLTradingStrategy()
        strategy.data_loader = data_loader
        strategy.feature_engineer = feature_engineer

        # Train the strategy
        metrics = strategy.train_strategy()
        print("   ✅ Strategy training completed!")

        # Print training metrics
        print("\n📊 Training Metrics:")
        for stage, stage_metrics in metrics.items():
            print(f"   {stage.upper()}:")
            for timeframe, tf_metrics in stage_metrics.items():
                print(f"     {timeframe}: {tf_metrics}")

        # Save the trained model
        print("\n💾 Step 5: Saving trained model...")

        # Create models directory if it doesn't exist
        models_dir = os.path.join("models")
        os.makedirs(models_dir, exist_ok=True)

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
        model_path = os.path.join(models_dir, "trained_model_january_2025.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"   ✓ Model saved to {model_path}")

        # Save scalers separately
        scaler_path = os.path.join(models_dir, "feature_scalers_january_2025.pkl")
        feature_engineer.save_scalers(scaler_path)

        # Save model info
        model_info = {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "training_date": datetime.now().isoformat(),
            "data_source": "BTCUSDT-aggTrades-2025-01.zip",
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
            "feature_engineering": "wavelet_transform_with_normalization",
            "scaler_type": "standard",
            "wavelet": "db4",
            "wavelet_levels": 4,
        }

        import json

        info_path = os.path.join(models_dir, "model_info_january_2025.json")
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"   ✓ Model info saved to {info_path}")

        print("\n" + "=" * 70)
        print("🎉 January 2025 Model Training Completed Successfully!")
        print("=" * 70)
        print("\n📋 Summary:")
        print(f"   ✅ Trained on: {len(ohlcv_data):,} 1-second bars")
        print(f"   ✅ Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
        print(f"   ✅ Features: Wavelet transform (db4, 4 levels)")
        print(f"   ✅ GPU acceleration: Enabled")
        print(f"   ✅ Model saved: {model_path}")

        print("\n📈 Next Steps:")
        print("   1. Run OOS test on February 2025 data:")
        print("      python scripts/oos_february.py")
        print("   2. Compare performance metrics")
        print("   3. Analyze feature importance\n")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up temporary files
        temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract_jan")
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary files\n")


if __name__ == "__main__":
    train_january_model()
