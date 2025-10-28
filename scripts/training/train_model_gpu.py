"""GPU training script - supports any BTCUSDT aggTrades data."""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import pickle
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    print(f"✅ LightGBM version: {lgb.__version__}")
except ImportError:
    print("❌ LightGBM not installed. Please install: pip install lightgbm")
    sys.exit(1)


def extract_zip(zip_path):
    """Extract zip file."""
    print(f"\n📦 Extracting {os.path.basename(zip_path)}...")
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract_train")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV in zip")

    csv_path = os.path.join(temp_dir, csv_files[0])
    print(f"✅ Extracted: {csv_files[0]}")
    return csv_path, temp_dir


def load_and_resample(csv_path, freq="5T"):
    """Load aggTrades and create OHLCV bars."""
    print(f"\n📊 Loading data from CSV...")
    df = pd.read_csv(csv_path)

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])

    print(f"   Records: {len(df):,}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Create OHLCV
    print(f"\n🔄 Resampling to {freq} bars...")
    ohlc = df.groupby(pd.Grouper(freq=freq)).agg(
        {"price": ["first", "max", "min", "last"], "quantity": "sum"}
    )
    ohlc.columns = ["open", "high", "low", "close", "volume"]
    ohlc = ohlc.dropna().ffill()

    print(f"✅ Created {len(ohlc):,} bars")
    return ohlc


def add_features(df):
    """Add technical indicators."""
    print(f"\n🔧 Engineering features...")

    # Price features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"price_to_sma_{window}"] = df["close"] / df[f"sma_{window}"]

    # Volatility
    df["volatility_20"] = df["returns"].rolling(20).std()

    # RSI
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = calc_rsi(df["close"])

    # MACD
    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    # ATR
    df["hl"] = df["high"] - df["low"]
    df["hc"] = abs(df["high"] - df["close"].shift(1))
    df["lc"] = abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["hl", "hc", "lc"]].max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    # Volume features
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    # Price momentum
    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)
        df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / df[
            "close"
        ].shift(period)

    # High/Low features
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]

    feature_cols = [
        col
        for col in df.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]
    print(f"✅ Created {len(feature_cols)} features")

    return df


def create_labels(df, forward_bars=3, threshold=0.005):
    """Create trading labels."""
    print(f"\n🎯 Creating labels (forward={forward_bars}, threshold={threshold})...")

    # Future returns
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1

    # Classification labels
    df["signal"] = 0  # neutral
    df.loc[df["future_return"] > threshold, "signal"] = 1  # long
    df.loc[df["future_return"] < -threshold, "signal"] = -1  # short

    # Binary classification (long vs neutral+short)
    df["binary_signal"] = (df["signal"] == 1).astype(int)

    print(
        f"   Long signals: {(df['signal'] == 1).sum()} ({(df['signal'] == 1).sum() / len(df) * 100:.2f}%)"
    )
    print(
        f"   Short signals: {(df['signal'] == -1).sum()} ({(df['signal'] == -1).sum() / len(df) * 100:.2f}%)"
    )
    print(
        f"   Neutral: {(df['signal'] == 0).sum()} ({(df['signal'] == 0).sum() / len(df) * 100:.2f}%)"
    )

    return df


def train_model(X, y, use_gpu=True):
    """Train LightGBM model."""
    print(f"\n🚀 Training LightGBM model (GPU={use_gpu})...")

    # Split train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")

    # LightGBM parameters
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
        "force_col_wise": True,
    }

    if use_gpu:
        try:
            params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
            print("   🎮 GPU acceleration enabled")
        except:
            print("   ⚠️  GPU not available, using CPU")
            use_gpu = False

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    train_acc = accuracy_score(y_train, (y_pred_train > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (y_pred_val > 0.5).astype(int))
    val_precision = precision_score(
        y_val, (y_pred_val > 0.5).astype(int), zero_division=0
    )
    val_recall = recall_score(y_val, (y_pred_val > 0.5).astype(int), zero_division=0)
    val_f1 = f1_score(y_val, (y_pred_val > 0.5).astype(int), zero_division=0)

    print(f"\n📊 Model Performance:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Val Accuracy: {val_acc:.4f}")
    print(f"   Val Precision: {val_precision:.4f}")
    print(f"   Val Recall: {val_recall:.4f}")
    print(f"   Val F1: {val_f1:.4f}")

    metrics = {
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "val_f1": float(val_f1),
        "n_trees": model.num_trees(),
    }

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model with GPU")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data zip file"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5T",
        help="Timeframe for resampling (default: 5T)",
    )
    parser.add_argument(
        "--model-name", type=str, default="model_trained", help="Output model name"
    )
    parser.add_argument(
        "--gpu", action="store_true", default=True, help="Use GPU (default: True)"
    )
    parser.add_argument(
        "--no-gpu", dest="gpu", action="store_false", help="Disable GPU"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 GPU Training: LightGBM Model")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Data: {args.data}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Model name: {args.model_name}")
    print(f"   GPU: {args.gpu}")

    # Check file exists
    if not os.path.exists(args.data):
        print(f"\n❌ Data not found: {args.data}")
        print("\n💡 Please provide valid data path:")
        print(
            "   Example: --data D:\\GitHub\\trading\\rlbot\\data\\agg_data\\BTCUSDT-aggTrades-2025-01.zip"
        )
        return

    print(f"\n✅ Data file found ({os.path.getsize(args.data) / (1024**3):.2f} GB)")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    try:
        # Extract
        csv_path, temp_dir = extract_zip(args.data)

        # Load and resample
        df = load_and_resample(csv_path, freq=args.timeframe)

        # Add features
        df = add_features(df)

        # Create labels
        df = create_labels(df, forward_bars=3, threshold=0.005)

        # Prepare training data
        df_clean = df.dropna()
        print(f"\n📋 Final dataset: {len(df_clean)} samples")

        feature_cols = [
            col
            for col in df_clean.columns
            if col
            not in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "signal",
                "binary_signal",
                "future_return",
                "hl",
                "hc",
                "lc",
                "tr",
            ]
        ]

        X = df_clean[feature_cols].values
        y = df_clean["binary_signal"].values

        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X)}")

        # Train model
        model, metrics = train_model(X, y, use_gpu=args.gpu)

        # Save model
        model_path = os.path.join(models_dir, f"{args.model_name}.txt")
        model.save_model(model_path)
        print(f"\n💾 Model saved: {model_path}")

        # Save metadata
        metadata = {
            "model_path": model_path,
            "training_date": datetime.now().isoformat(),
            "data_source": os.path.basename(args.data),
            "timeframe": args.timeframe,
            "n_samples": len(df_clean),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "date_range": [str(df_clean.index[0]), str(df_clean.index[-1])],
            "metrics": metrics,
            "gpu_used": args.gpu,
        }

        metadata_path = os.path.join(models_dir, f"{args.model_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Metadata saved: {metadata_path}")

        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        print(f"\n📁 Saved files:")
        print(f"   - {model_path}")
        print(f"   - {metadata_path}")
        print(f"\n📈 Next step:")
        print(
            f"   python scripts/oos_test.py --model {args.model_name} --data <test_data.zip>"
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n🧹 Cleaned up temp files")


if __name__ == "__main__":
    main()
