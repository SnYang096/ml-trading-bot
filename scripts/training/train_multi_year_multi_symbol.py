"""Large-scale training: Multiple years, multiple symbols (BTC/ETH/SOL 2021-2023)."""

import os
import sys
import glob
import zipfile
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    print("❌ LightGBM not installed")
    sys.exit(1)


def find_data_files(data_dir, symbols, start_year, end_year):
    """Find all data files for given symbols and year range."""
    all_files = []
    for symbol in symbols:
        pattern = os.path.join(
            data_dir, f"{symbol}-aggTrades-{{{start_year}..{end_year}}}-*.zip"
        )
        # Manual glob for year range
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                file_path = os.path.join(
                    data_dir, f"{symbol}-aggTrades-{year}-{month:02d}.zip"
                )
                if os.path.exists(file_path):
                    all_files.append(file_path)

    return sorted(all_files)


def load_single_file(zip_path, freq="5T", sample_rate=1.0):
    """
    Load and process a single zip file.

    Args:
        zip_path: Path to zip file
        freq: Resampling frequency
        sample_rate: Fraction of data to use (0-1), for faster training
    """
    print(f"   Loading {os.path.basename(zip_path)}...", end=" ")

    # Extract
    temp_dir = os.path.join(os.path.dirname(zip_path), f"temp_{os.getpid()}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_dir)

        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        csv_path = os.path.join(temp_dir, csv_files[0])

        # Load CSV - handle both with and without headers
        try:
            df = pd.read_csv(csv_path)
            # Check if has proper headers
            if "transact_time" in df.columns or "timestamp" in df.columns:
                # Has headers
                if "transact_time" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                df.set_index("timestamp", inplace=True)
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
            else:
                # No headers - use Binance aggTrades format
                # agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
                df = pd.read_csv(
                    csv_path,
                    header=None,
                    names=[
                        "agg_trade_id",
                        "price",
                        "quantity",
                        "first_trade_id",
                        "last_trade_id",
                        "transact_time",
                        "is_buyer_maker",
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

            df = df.dropna(subset=["price", "quantity"])
        except Exception as e:
            print(f"Error loading: {e}")
            return None

        # Time-aware sampling (NOT random!) - preserve temporal order
        if sample_rate < 1.0:
            # Method: Take every Nth record to maintain time uniformity
            step = int(1 / sample_rate)
            df = df.iloc[::step]  # Keep time order, uniform sampling
            print(f"(sampled every {step}th record)", end=" ")

        # Resample to OHLCV
        ohlc = df.groupby(pd.Grouper(freq=freq)).agg(
            {"price": ["first", "max", "min", "last"], "quantity": "sum"}
        )
        ohlc.columns = ["open", "high", "low", "close", "volume"]
        ohlc = ohlc.dropna().ffill()

        print(f"{len(ohlc)} bars")

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        return ohlc

    except Exception as e:
        print(f"Error: {e}")
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def add_features(df):
    """Add technical indicators."""
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"price_to_sma_{window}"] = df["close"] / df[f"sma_{window}"]

    df["volatility_20"] = df["returns"].rolling(20).std()

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = calc_rsi(df["close"])

    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    df["hl"] = df["high"] - df["low"]
    df["hc"] = abs(df["high"] - df["close"].shift(1))
    df["lc"] = abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["hl", "hc", "lc"]].max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)
        df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / df[
            "close"
        ].shift(period)

    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]

    return df


def create_labels(df, forward_bars=3, threshold=0.005):
    """Create trading labels."""
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1
    df["signal"] = 0
    df.loc[df["future_return"] > threshold, "signal"] = 1
    df.loc[df["future_return"] < -threshold, "signal"] = -1
    df["binary_signal"] = (df["signal"] == 1).astype(int)
    return df


def train_model(X, y, use_gpu=True, n_boost_round=200, use_time_series_cv=True):
    """Train LightGBM model with proper time series validation."""
    print(
        f"\n🚀 Training LightGBM model (GPU={use_gpu}, TimeSeriesCV={use_time_series_cv})..."
    )

    if use_time_series_cv:
        # Use TimeSeriesSplit for proper time series validation
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=5)
        print(f"   Using TimeSeriesSplit with 5 folds")

        # Use the last split for final training
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]  # Last fold has most data

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"   Train: {len(X_train):,} samples (time: 0 -> {train_idx[-1]})")
        print(f"   Val: {len(X_val):,} samples (time: {val_idx[0]} -> {val_idx[-1]})")
        print(f"   ✅ No future leakage - val set is strictly after train set")
    else:
        # Simple time-based split (80/20) - still maintains time order
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"   Train: {len(X_train):,} samples")
        print(f"   Val: {len(X_val):,} samples")

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
            print("   ⚠️ GPU not available, using CPU")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=20),
        ],
    )

    # Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, (y_pred_val > 0.5).astype(int))
    val_precision = precision_score(
        y_val, (y_pred_val > 0.5).astype(int), zero_division=0
    )
    val_recall = recall_score(y_val, (y_pred_val > 0.5).astype(int), zero_division=0)
    val_f1 = f1_score(y_val, (y_pred_val > 0.5).astype(int), zero_division=0)

    print(f"\n📊 Model Performance:")
    print(f"   Val Accuracy: {val_acc:.4f}")
    print(f"   Val Precision: {val_precision:.4f}")
    print(f"   Val Recall: {val_recall:.4f}")
    print(f"   Val F1: {val_f1:.4f}")

    metrics = {
        "val_acc": float(val_acc),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "val_f1": float(val_f1),
        "n_trees": model.num_trees(),
    }

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-year, multi-symbol training")
    parser.add_argument(
        "--data-dir", type=str, default=r"D:\GitHub\trading\rlbot\data\agg_data"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--timeframe", type=str, default="5T")
    parser.add_argument("--model-name", type=str, default="model_2021_2023_multi")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.3,
        help="Fraction of data to use (0-1), lower for faster training",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Max files to process (for testing)"
    )
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 Large-Scale Training: Multi-Year, Multi-Symbol")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Symbols: {', '.join(args.symbols)}")
    print(f"   Years: {args.start_year}-{args.end_year}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Sample rate: {args.sample_rate*100}%")
    print(f"   GPU: {args.gpu}")

    if args.sample_rate < 1.0:
        print(f"\n⚠️  WARNING: Using sample_rate={args.sample_rate}")
        print(
            f"   - Sampling method: Time-uniform (every {int(1/args.sample_rate)}th record)"
        )
        print(f"   - This preserves temporal order (no random sampling)")
        print(f"   - For production, use --sample-rate 1.0")

    # Find files
    print(f"\n🔍 Finding data files...")
    data_files = find_data_files(
        args.data_dir, args.symbols, args.start_year, args.end_year
    )

    if args.max_files:
        data_files = data_files[: args.max_files]

    print(f"   Found {len(data_files)} files")

    if not data_files:
        print("❌ No data files found!")
        return

    # Load and combine data
    print(f"\n📊 Loading data (this may take a while)...")
    all_data = []

    for i, file_path in enumerate(data_files, 1):
        print(f"[{i}/{len(data_files)}]", end=" ")
        df = load_single_file(
            file_path, freq=args.timeframe, sample_rate=args.sample_rate
        )
        if df is not None and len(df) > 0:
            all_data.append(df)

    if not all_data:
        print("\n❌ No data loaded successfully!")
        return

    print(f"\n🔗 Combining {len(all_data)} dataframes...")
    combined_df = pd.concat(all_data, axis=0).sort_index()
    print(f"   Total bars: {len(combined_df):,}")
    print(f"   Date range: {combined_df.index[0]} to {combined_df.index[-1]}")

    # Add features
    print(f"\n🔧 Engineering features...")
    combined_df = add_features(combined_df)
    combined_df = create_labels(combined_df)
    combined_df = combined_df.dropna()

    print(f"   Final dataset: {len(combined_df):,} samples")

    # Prepare training data
    feature_cols = [
        col
        for col in combined_df.columns
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

    X = combined_df[feature_cols].values
    y = combined_df["binary_signal"].values

    print(f"   Features: {len(feature_cols)}")
    print(f"   Long signals: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")

    # Train with TimeSeriesSplit
    start_time = datetime.now()
    model, metrics = train_model(X, y, use_gpu=args.gpu, use_time_series_cv=True)
    train_duration = (datetime.now() - start_time).total_seconds()

    print(f"\n⏱️ Training time: {train_duration:.1f}s")

    # Save
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{args.model_name}.txt")
    model.save_model(model_path)
    print(f"\n💾 Model saved: {model_path}")

    metadata = {
        "model_path": model_path,
        "training_date": datetime.now().isoformat(),
        "symbols": args.symbols,
        "years": f"{args.start_year}-{args.end_year}",
        "timeframe": args.timeframe,
        "sample_rate": args.sample_rate,
        "n_files": len(data_files),
        "n_samples": len(combined_df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "date_range": [str(combined_df.index[0]), str(combined_df.index[-1])],
        "metrics": metrics,
        "gpu_used": args.gpu,
        "train_duration_seconds": train_duration,
    }

    metadata_path = os.path.join(models_dir, f"{args.model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved: {metadata_path}")

    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)
    print(f"\n📈 Next step:")
    print(f"   Test on 2024-2025 data:")
    print(
        f"   python scripts/oos_batch_test.py --model {args.model_name} --pattern '.*-202[45]-.*'"
    )


if __name__ == "__main__":
    main()
