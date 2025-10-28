"""Data loading and processing utilities for rolling training."""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import shutil
from typing import Optional, Dict, Tuple, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
)


def load_parquet_file(parquet_path: str) -> Optional[pd.DataFrame]:
    """
    Load a parquet file directly.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        # Ensure timestamp is the index
        if 'timestamp' in df.columns and df.index.name != 'timestamp':
            df.set_index('timestamp', inplace=True)
        
        # Verify required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️  Missing columns in parquet file: {missing_cols}")
            return None
        
        return df
        
    except Exception as e:
        print(f"   ⚠️  Error loading parquet {parquet_path}: {e}")
        return None


def load_and_process_file(zip_path: str, freq: str = "5T", parquet_dir: str = None) -> Optional[pd.DataFrame]:
    """
    Load a single data file (parquet or zip) and create OHLCV data.
    Parquet files are preferred if they exist.

    Args:
        zip_path: Path to the zip file (or base path for parquet)
        freq: Resampling frequency (default '5T' for 5 minutes)
        parquet_dir: Directory containing parquet files (default: data/parquet_data)

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    # Check for parquet file first
    if parquet_dir is None:
        # Default parquet directory - 处理多种路径情况
        if "ml_project" in zip_path:
            # 如果路径包含 ml_project，则数据在上一级的 rlbot/data
            project_root = zip_path.split("ml_project")[0].rstrip("/")
            parquet_dir = os.path.join(project_root, "data", "parquet_data")
        else:
            # 默认假设 data 目录与 zip_path 同级
            base_dir = os.path.dirname(os.path.dirname(zip_path))
            parquet_dir = os.path.join(base_dir, "parquet_data")
    
    # Try to find corresponding parquet file
    if os.path.exists(parquet_dir):
        zip_basename = os.path.basename(zip_path)
        # Convert zip filename to expected parquet filename
        # Example: BTCUSDT-aggTrades-2025-01.zip -> BTC-USD_2025-01.parquet
        if "BTCUSDT-aggTrades-" in zip_basename:
            date_part = zip_basename.replace("BTCUSDT-aggTrades-", "").replace(".zip", "")
            parquet_name = f"BTC-USD_{date_part}.parquet"
        elif "ETHUSDT-aggTrades-" in zip_basename:
            date_part = zip_basename.replace("ETHUSDT-aggTrades-", "").replace(".zip", "")
            parquet_name = f"ETH-USD_{date_part}.parquet"
        else:
            parquet_name = zip_basename.replace(".zip", ".parquet")
        
        parquet_path = os.path.join(parquet_dir, parquet_name)
        
        if os.path.exists(parquet_path):
            print(f"   📊 Loading from parquet: {os.path.basename(parquet_path)}")
            return load_parquet_file(parquet_path)
    
    # Fall back to zip file processing
    print(f"   📦 Loading from zip: {os.path.basename(zip_path)}")
    temp_dir = os.path.join(os.path.dirname(zip_path), f"temp_{os.getpid()}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_dir)

        # Find CSV file
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {zip_path}")

        csv_path = os.path.join(temp_dir, csv_files[0])

        # Load data
        df = pd.read_csv(csv_path)

        # Handle different timestamp formats
        if "transact_time" in df.columns or "timestamp" in df.columns:
            if "transact_time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            # No headers case
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

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Convert to numeric
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df = df.dropna(subset=["price", "quantity"])

        # Resample to OHLCV
        ohlc = df.groupby(pd.Grouper(freq=freq)).agg(
            {"price": ["first", "max", "min", "last"], "quantity": "sum"}
        )
        ohlc.columns = ["open", "high", "low", "close", "volume"]
        ohlc = ohlc.dropna().ffill()

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        return ohlc

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"   ⚠️  Error loading {zip_path}: {e}")
        return None


def add_order_flow_features(zip_path: str, ohlcv_df: pd.DataFrame, parquet_dir: str = None) -> pd.DataFrame:
    """
    Add order flow features (CVD, taker_buy_ratio) from aggregate trade data.
    Note: Currently only supports loading from zip files as parquet files are pre-aggregated OHLCV.

    Args:
        zip_path: Path to the zip file containing aggregate trades
        ohlcv_df: OHLCV DataFrame to add features to
        parquet_dir: Not used for order flow (parquet files don't contain tick-level data)

    Returns:
        DataFrame with added order flow features
    """
    # Check if parquet exists - if so, skip order flow features as they need tick data
    if parquet_dir is None:
        # 使用与 load_and_process_file 相同的逻辑
        if "ml_project" in zip_path:
            project_root = zip_path.split("ml_project")[0].rstrip("/")
            parquet_dir = os.path.join(project_root, "data", "parquet_data")
        else:
            base_dir = os.path.dirname(os.path.dirname(zip_path))
            parquet_dir = os.path.join(base_dir, "parquet_data")
    
    # Note: Parquet files are pre-aggregated and don't contain tick-level data needed for order flow
    # If using parquet, order flow features should be pre-computed during conversion
    
    temp_dir = os.path.join(os.path.dirname(zip_path), f"temp_of_{os.getpid()}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Extract and load aggregate trades from zip
        if not os.path.exists(zip_path):
            print(f"   ℹ️  Zip file not found for order flow features, skipping")
            return ohlcv_df
            
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_dir)

        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        csv_path = os.path.join(temp_dir, csv_files[0])

        agg = pd.read_csv(csv_path)

        # Handle timestamp
        if "transact_time" in agg.columns:
            agg["timestamp"] = pd.to_datetime(agg["transact_time"], unit="ms")
        else:
            agg["timestamp"] = pd.to_datetime(agg["timestamp"])

        # Convert to numeric
        agg["price"] = pd.to_numeric(agg["price"], errors="coerce")
        agg["quantity"] = pd.to_numeric(agg["quantity"], errors="coerce")
        agg = agg.dropna(subset=["price", "quantity"])

        # Classify taker side
        if "is_buyer_maker" in agg.columns:
            agg["taker_buy"] = (~agg["is_buyer_maker"].astype(bool)).astype(int)
        else:
            agg["taker_buy"] = 0

        # Calculate buy/sell quantities
        agg["buy_qty"] = np.where(agg["taker_buy"] == 1, agg["quantity"], 0.0)
        agg["sell_qty"] = np.where(agg["taker_buy"] == 1, 0.0, agg["quantity"])
        agg = agg.set_index("timestamp")

        # Resample to same frequency as OHLCV
        freq = pd.infer_freq(ohlcv_df.index[:10])
        if freq is None:
            freq = "5T"  # Default to 5 minutes

        per_interval = agg.groupby(pd.Grouper(freq=freq)).agg(
            {"buy_qty": "sum", "sell_qty": "sum"}
        )

        # Calculate features
        per_interval["taker_buy_ratio"] = per_interval["buy_qty"] / (
            per_interval["buy_qty"] + per_interval["sell_qty"]
        ).replace(0, np.nan)
        per_interval["taker_buy_ratio"] = per_interval["taker_buy_ratio"].fillna(0.5)

        # Delta (buy - sell)
        delta = per_interval["buy_qty"] - per_interval["sell_qty"]

        # CVD改进版本：不使用全局cumsum，而是使用多个时间窗口的滚动累计
        # 1. 短期CVD (20个周期，约100分钟)
        per_interval["cvd_short"] = delta.rolling(window=20, min_periods=1).sum()

        # 2. 中期CVD (60个周期，约5小时)
        per_interval["cvd_medium"] = delta.rolling(window=60, min_periods=1).sum()

        # 3. 长期CVD (288个周期，约24小时)
        per_interval["cvd_long"] = delta.rolling(window=288, min_periods=1).sum()

        # 4. CVD变化率（momentum）
        per_interval["cvd_change_1"] = delta  # 当前周期的delta
        per_interval["cvd_change_5"] = delta.rolling(window=5).sum()  # 5周期累计
        per_interval["cvd_change_20"] = delta.rolling(window=20).sum()  # 20周期累计

        # 5. CVD归一化（相对于成交量）
        total_volume = per_interval["buy_qty"] + per_interval["sell_qty"]
        per_interval["cvd_normalized"] = delta / total_volume.replace(0, np.nan)
        per_interval["cvd_normalized"] = per_interval["cvd_normalized"].fillna(0)

        # 保留原始CVD用于向后兼容（但建议使用滚动窗口版本）
        per_interval["cvd"] = delta.cumsum()

        # Join with OHLCV data
        result = (
            ohlcv_df.join(
                per_interval[
                    [
                        "buy_qty",
                        "sell_qty",
                        "taker_buy_ratio",
                        "cvd",
                        "cvd_short",
                        "cvd_medium",
                        "cvd_long",
                        "cvd_change_1",
                        "cvd_change_5",
                        "cvd_change_20",
                        "cvd_normalized",
                    ]
                ],
                how="left",
            )
            .ffill()
            .fillna(0)
        )

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        return result

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"   ⚠️  Warning: Failed to add order flow features: {e}")
        return ohlcv_df


def engineer_features(
    df: pd.DataFrame,
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, ComprehensiveFeatureEngineer]:
    """
    Engineer comprehensive features using ComprehensiveFeatureEngineer.

    Args:
        df: OHLCV DataFrame (with optional order flow features)
        feature_engineer: Existing feature engineer (for test data), or None to create new
        fit: Whether to fit the scaler (True for train, False for test)

    Returns:
        Tuple of (engineered DataFrame, feature engineer)
    """
    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer(
            scaler_type="standard", wavelet="db4", wpt_level=3, hurst_window=100
        )

    # Engineer features for single timeframe
    engineered_data = feature_engineer.engineer_features(df, fit=fit)

    return engineered_data, feature_engineer


def create_labels(
    df: pd.DataFrame, forward_bars: int = 3, threshold: float = 0.005
) -> pd.DataFrame:
    """
    Create trading labels for supervised learning.

    Args:
        df: DataFrame with OHLCV data
        forward_bars: Number of bars to look forward
        threshold: Return threshold for signal generation

    Returns:
        DataFrame with added label columns
    """
    df = df.copy()

    # Calculate future return
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1

    # Create signals
    df["signal"] = 0
    df.loc[df["future_return"] > threshold, "signal"] = 1
    df.loc[df["future_return"] < -threshold, "signal"] = -1

    # Binary signal for classification
    df["binary_signal"] = (df["signal"] == 1).astype(int)

    return df


def add_dl_time_series_features(
    df: pd.DataFrame,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    use_fp16: bool = True,
) -> pd.DataFrame:
    """
    Add Deep Learning sequence features (Mamba/Transformer) to DataFrame.

    Args:
        df: DataFrame with OHLCV data
        backend: 'mamba', 'flash_attention', 'transformer', or 'auto'
        seq_length: Sequence length (default: 120 bars = 10 hours)
        d_model: Output dimension (default: 64)
        use_fp16: Use FP16 mixed precision (default: True)

    Returns:
        DataFrame with DL sequence features added
    """
    try:
        # 使用综合特征工程中的深度学习特征
        from ml_trading.data_tools.dl_sequence_features import add_dl_sequence_features

        df_with_dl = add_dl_sequence_features(
            df,
            backend=backend,
            seq_length=seq_length,
            d_model=d_model,
            feature_columns=["open", "high", "low", "close", "volume"],
            use_fp16=use_fp16,
        )
        print(f"      ✅ DL sequence features added: {df_with_dl.shape}")
        return df_with_dl
    except Exception as e:
        print(f"   ⚠️  DL feature extraction failed: {e}")
        print(f"   ⚠️  Continuing without DL features")
        return df


# Alias for backward compatibility
add_transformer_time_series_features = add_dl_time_series_features


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get feature column names (excluding OHLCV and label columns).

    Args:
        df: DataFrame with all columns

    Returns:
        List of feature column names
    """
    exclude_cols = [
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

    return [col for col in df.columns if col not in exclude_cols]
