"""Train ML model with enhanced features: WPT + Entropy + Hurst."""

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
from ml_trading.data_tools.feature_engineering_enhanced import EnhancedFeatureEngineer


def extract_zip_data(zip_path: str) -> str:
    """Extract zip file and return path to CSV file."""
    print(f"Extracting {zip_path}...")

    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract_enhanced")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the zip archive")

    csv_path = os.path.join(temp_dir, csv_files[0])
    print(f"Extracted CSV file: {csv_path}")
    return csv_path


def load_btcusdt_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess BTCUSDT aggregate trade data."""
    print(f"Loading BTCUSDT data from {csv_path}...")

    df = pd.read_csv(csv_path)
    print(f"Raw data shape: {df.shape}")

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

    print(f"After cleaning: {len(df)} records")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")

    return df


def create_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert aggregate trade data to OHLCV format."""
    print("Converting aggregate trades to OHLCV...")

    ohlc_dict = {"price": ["first", "max", "min", "last"], "quantity": "sum"}

    resampled = df.groupby(pd.Grouper(freq="1s")).agg(ohlc_dict)
    resampled.columns = ["open", "high", "low", "close", "volume"]
    resampled = resampled.dropna().ffill()

    print(f"Created {len(resampled)} 1-second OHLCV bars")
    print(f"Time range: {resampled.index[0]} to {resampled.index[-1]}")

    return resampled


def train_enhanced_model():
    """Train ML model with enhanced WPT + Entropy + Hurst features."""
    print("🚀 Training Enhanced ML Model (WPT + Entropy + Hurst)")
    print("=" * 80)

    # Path to the zip file
    zip_path = "data/raw/BTCUSDT-aggTrades-2025-05.zip"

    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return

    print(f"✅ Zip file found: {zip_path}")

    try:
        # Extract and load data
        csv_path = extract_zip_data(zip_path)
        raw_data = load_btcusdt_data(csv_path)
        ohlcv_data = create_ohlcv_data(raw_data)

        # Derive microstructure series from agg trades for WPT: CVD and taker_buy_ratio
        print("\nAdding order flow features (CVD and taker_buy_ratio)...")
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
            print(f"   ✓ Added CVD and taker_buy_ratio to OHLCV data")
        except Exception as e:
            print(f"   ⚠️  Warning: failed to compute microstructure series: {e}")

        # Initialize data loader
        print("\n1. Initializing data loader...")
        data_loader = MarketDataLoader()
        data_loader.raw_data = ohlcv_data

        # Get multi-timeframe data
        print("\n2. Creating multi-timeframe data...")
        multi_tf_data = data_loader.get_multi_timeframe_data()

        print(f"   ✓ Created data for timeframes: {list(multi_tf_data.keys())}")
        for tf, data in multi_tf_data.items():
            print(f"     - {tf}: {len(data)} bars")

        # Enhanced feature engineering
        print("\n3. Engineering enhanced features (WPT + Entropy + Hurst)...")
        print("   This may take a few minutes...")

        feature_engineer = EnhancedFeatureEngineer(
            scaler_type="standard", wavelet="db4", wpt_level=3, hurst_window=100
        )

        engineered_data = feature_engineer.engineer_features(multi_tf_data, fit=True)

        print(f"\n   ✓ Enhanced features engineered for all timeframes")
        for tf, data in engineered_data.items():
            print(f"     - {tf}: {data.shape[1]} features, {len(data)} rows")

            # Show sample features
            feature_cols = [
                col
                for col in data.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            if feature_cols:
                print(f"       Total features: {len(feature_cols)}")
                # Show some WPT and Hurst features
                wpt_features = [col for col in feature_cols if "wpt_" in col]
                hurst_features = [col for col in feature_cols if "hurst" in col]
                print(f"       WPT features: {len(wpt_features)}")
                print(f"       Hurst features: {len(hurst_features)}")

        # Initialize and train strategy
        print("\n4. Training ML strategy with enhanced features...")
        strategy = MLTradingStrategy()
        strategy.data_loader = data_loader
        strategy.feature_engineer = feature_engineer

        # Train the strategy
        metrics = strategy.train_strategy()
        print("   ✓ Strategy training completed")

        # Print training metrics
        print("\n   Training Metrics:")
        for stage, stage_metrics in metrics.items():
            print(f"     {stage.upper()}:")
            for timeframe, tf_metrics in stage_metrics.items():
                print(f"       {timeframe}: {tf_metrics}")

        # Save the trained model
        print("\n5. Saving trained enhanced model...")
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

        # Save model
        model_path = "trained_model_enhanced_may_2025.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"   ✓ Model saved to {model_path}")

        # Save scalers
        scaler_path = "feature_scalers_enhanced_may_2025.pkl"
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
            "feature_engineering": "enhanced_with_wpt_entropy_hurst",
            "features": {
                "scaler_type": "standard",
                "wavelet": "db4",
                "wpt_level": 3,
                "hurst_window": 100,
            },
        }

        import json

        with open("model_info_enhanced_may_2025.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"   ✓ Model info saved to model_info_enhanced_may_2025.json")

        print("\n🎉 Enhanced model training completed successfully!")
        print("\nKey Features Added (对所有信号源):")
        print(
            "1. ✅ Wavelet Packet Transform (WPT) - 对close/open/volume/cvd/taker_buy_ratio"
        )
        print("2. ✅ Shannon Entropy - 每个信号源的能量分布混乱度")
        print("3. ✅ Hurst Exponent - 每个信号源的趋势持续性/均值回归")
        print("4. ✅ Energy Features - 各频带能量特征")
        print("5. ✅ Time Series CV - 正确的交叉验证")
        print(
            f"\n预期特征数: 5个信号源 × ~40个WPT特征 + 5×6个Hurst特征 + ~30基础特征 ≈ 260个"
        )

        print("\nNext steps:")
        print("1. Test on OOS data (June-September)")
        print("2. Compare with baseline model")
        print("3. Analyze feature importance")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up temporary files
        temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract_enhanced")
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    train_enhanced_model()
