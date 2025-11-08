"""Data loading and processing utilities for rolling training workflows."""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer, )


def load_parquet_file(parquet_path: str) -> Optional[pd.DataFrame]:
    """Load a parquet file directly into an OHLCV DataFrame."""

    try:
        df = pd.read_parquet(parquet_path)

        if "timestamp" in df.columns and df.index.name != "timestamp":
            df.set_index("timestamp", inplace=True)

        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️  Missing columns in parquet file: {missing_cols}")
            return None

        return df

    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Error loading parquet {parquet_path}: {exc}")
        return None


def _infer_default_parquet_dir(zip_path: str) -> str:
    if "ml_project" in zip_path:
        project_root = zip_path.split("ml_project")[0].rstrip("/")
        return os.path.join(project_root, "data", "parquet_data")

    base_dir = os.path.dirname(os.path.dirname(zip_path))
    return os.path.join(base_dir, "parquet_data")


def _infer_parquet_name(zip_basename: str) -> str:
    if "BTCUSDT-aggTrades-" in zip_basename:
        date_part = zip_basename.replace("BTCUSDT-aggTrades-",
                                         "").replace(".zip", "")
        return f"BTC-USD_{date_part}.parquet"

    if "ETHUSDT-aggTrades-" in zip_basename:
        date_part = zip_basename.replace("ETHUSDT-aggTrades-",
                                         "").replace(".zip", "")
        return f"ETH-USD_{date_part}.parquet"

    return zip_basename.replace(".zip", ".parquet")


def load_and_process_file(
        zip_path: str,
        *,
        freq: str = "5T",
        parquet_dir: str | None = None) -> Optional[pd.DataFrame]:
    """Load a single aggregate-trade file (parquet preferred, fallback to zip)."""

    path = os.path.abspath(zip_path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        print(f"   📊 Loading from parquet: {os.path.basename(path)}")
        return load_parquet_file(path)

    parquet_dir = parquet_dir or _infer_default_parquet_dir(zip_path)

    if os.path.exists(parquet_dir):
        zip_basename = os.path.basename(zip_path)
        parquet_name = _infer_parquet_name(zip_basename)
        parquet_path = os.path.join(parquet_dir, parquet_name)

        if os.path.exists(parquet_path):
            print(
                f"   📊 Loading from parquet: {os.path.basename(parquet_path)}")
            return load_parquet_file(parquet_path)

    print(f"   📦 Loading from zip: {os.path.basename(zip_path)}")
    temp_dir = os.path.join(os.path.dirname(zip_path), f"temp_{os.getpid()}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_dir)

        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {zip_path}")

        csv_path = os.path.join(temp_dir, csv_files[0])
        df = pd.read_csv(csv_path)

        if "transact_time" in df.columns or "timestamp" in df.columns:
            if "transact_time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["transact_time"],
                                                 unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
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

        ohlc = df.groupby(pd.Grouper(freq=freq)).agg({
            "price": ["first", "max", "min", "last"],
            "quantity":
            "sum"
        })
        ohlc.columns = ["open", "high", "low", "close", "volume"]
        ohlc = ohlc.dropna().ffill()

        return ohlc

    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Error loading {zip_path}: {exc}")
        return None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def add_order_flow_features(zip_path: str,
                            ohlcv_df: pd.DataFrame,
                            parquet_dir: str | None = None) -> pd.DataFrame:
    """Add order-flow derived features (CVD, taker buy ratio, etc.)."""

    parquet_dir = parquet_dir or _infer_default_parquet_dir(zip_path)
    temp_dir = os.path.join(os.path.dirname(zip_path),
                            f"temp_of_{os.getpid()}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        if not os.path.exists(zip_path):
            print(
                "   ℹ️  Zip file not found for order flow features, skipping")
            return ohlcv_df

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_dir)

        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        csv_path = os.path.join(temp_dir, csv_files[0])
        agg = pd.read_csv(csv_path)

        if "transact_time" in agg.columns:
            agg["timestamp"] = pd.to_datetime(agg["transact_time"], unit="ms")
        else:
            agg["timestamp"] = pd.to_datetime(agg["timestamp"])

        agg["price"] = pd.to_numeric(agg["price"], errors="coerce")
        agg["quantity"] = pd.to_numeric(agg["quantity"], errors="coerce")
        agg = agg.dropna(subset=["price", "quantity"])

        if "is_buyer_maker" in agg.columns:
            agg["taker_buy"] = (
                ~agg["is_buyer_maker"].astype(bool)).astype(int)
        else:
            agg["taker_buy"] = 0

        agg["buy_qty"] = np.where(agg["taker_buy"] == 1, agg["quantity"], 0.0)
        agg["sell_qty"] = np.where(agg["taker_buy"] == 1, 0.0, agg["quantity"])
        agg = agg.set_index("timestamp")

        freq = pd.infer_freq(ohlcv_df.index[:10]) or "5T"
        per_interval = agg.groupby(pd.Grouper(freq=freq)).agg({
            "buy_qty": "sum",
            "sell_qty": "sum"
        })

        per_interval["taker_buy_ratio"] = per_interval["buy_qty"] / (
            per_interval["buy_qty"] + per_interval["sell_qty"]).replace(
                0, np.nan)
        per_interval["taker_buy_ratio"] = per_interval[
            "taker_buy_ratio"].fillna(0.5)

        delta = per_interval["buy_qty"] - per_interval["sell_qty"]
        per_interval["cvd_short"] = delta.rolling(window=20,
                                                  min_periods=1).sum()
        per_interval["cvd_medium"] = delta.rolling(window=60,
                                                   min_periods=1).sum()
        per_interval["cvd_long"] = delta.rolling(window=288,
                                                 min_periods=1).sum()
        per_interval["cvd_change_1"] = delta
        per_interval["cvd_change_5"] = delta.rolling(window=5).sum()
        per_interval["cvd_change_20"] = delta.rolling(window=20).sum()

        total_volume = per_interval["buy_qty"] + per_interval["sell_qty"]
        per_interval["cvd_normalized"] = (
            delta / total_volume.replace(0, np.nan)).fillna(0)
        per_interval["cvd"] = delta.cumsum()

        result = (ohlcv_df.join(
            per_interval[[
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
            ]],
            how="left",
        ).ffill().fillna(0))

        return result

    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Warning: Failed to add order flow features: {exc}")
        return ohlcv_df

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def engineer_features(
    df: pd.DataFrame,
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
    *,
    fit: bool = True,
) -> Tuple[pd.DataFrame, ComprehensiveFeatureEngineer]:
    """Engineer features using the reusable ComprehensiveFeatureEngineer."""

    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer(scaler_type="standard",
                                                        wavelet="db4",
                                                        wpt_level=3,
                                                        hurst_window=100)

    engineered_data = feature_engineer.engineer_features(df, fit=fit)
    return engineered_data, feature_engineer


def create_labels(df: pd.DataFrame,
                  *,
                  forward_bars: int = 3,
                  threshold: float = 0.005) -> pd.DataFrame:
    """Create future-return based classification labels (3-class: 0=Hold, 1=Long, 2=Short).

    Args:
        df: DataFrame with OHLCV data
        forward_bars: Number of bars ahead for prediction
        threshold: Threshold for signal classification
    
    Returns:
        DataFrame with 'signal' column containing 3-class labels:
        - 0: Hold (future_return between -threshold and threshold)
        - 1: Long (future_return > threshold)
        - 2: Short (future_return < -threshold)
    """

    df = df.copy()
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1

    # Create 3-class labels (0=Hold, 1=Long, 2=Short) for multiclass classification
    df["signal"] = 0  # Hold by default
    df.loc[df["future_return"] > threshold, "signal"] = 1   # Long
    df.loc[df["future_return"] < -threshold, "signal"] = 2  # Short
    
    # Keep backward compatibility: binary_signal for legacy code (1=Long, 0=not Long)
    df["binary_signal"] = (df["signal"] == 1).astype(int)

    return df


def create_labels_multi_horizon(
    df: pd.DataFrame,
    *,
    horizons: list[int] = [1, 5, 10, 15],
    threshold: float = 0.005
) -> pd.DataFrame:
    """Create future-return based labels for multiple horizons (3-class: 0=Hold, 1=Long, 2=Short).
    
    Args:
        df: DataFrame with OHLCV data
        horizons: List of forward bars to look ahead (e.g., [1, 5, 10, 15])
        threshold: Threshold for signal classification
    
    Returns:
        DataFrame with multiple label columns for each horizon:
        - signal_{horizon}: 3-class labels (0=Hold, 1=Long, 2=Short)
        - binary_signal_{horizon}: Backward compatibility (1=Long, 0=not Long)
        - Also creates backward-compatible 'signal' and 'binary_signal' using the first horizon
    """
    df = df.copy()
    
    for horizon in horizons:
        # Create future return for this horizon
        future_return_col = f"future_return_{horizon}"
        df[future_return_col] = df["close"].shift(-horizon) / df["close"] - 1
        
        # Create 3-class signal for this horizon (0=Hold, 1=Long, 2=Short)
        signal_col = f"signal_{horizon}"
        df[signal_col] = 0  # Hold by default
        df.loc[df[future_return_col] > threshold, signal_col] = 1   # Long
        df.loc[df[future_return_col] < -threshold, signal_col] = 2  # Short
        
        # Keep backward compatibility: binary_signal for legacy code
        binary_signal_col = f"binary_signal_{horizon}"
        df[binary_signal_col] = (df[signal_col] == 1).astype(int)
    
    # Create backward-compatible columns using the first horizon
    if horizons:
        df["signal"] = df[f"signal_{horizons[0]}"]
        df["binary_signal"] = df[f"binary_signal_{horizons[0]}"]
    
    return df


def add_dl_time_series_features(
    df: pd.DataFrame,
    *,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    use_fp16: bool = True,
) -> pd.DataFrame:
    """Add DL sequence features (Mamba/Transformer) to the dataset."""

    try:
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

    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  DL feature extraction failed: {exc}")
        print("   ⚠️  Continuing without DL features")
        return df


add_transformer_time_series_features = add_dl_time_series_features


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return model feature columns (excluding OHLCV + label columns)."""

    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "binary_signal",  # Keep for backward compatibility
        "future_return",
        "hl",
        "hc",
        "lc",
        "tr",
    }
    
    # Also exclude multi-horizon label columns (e.g., signal_1, binary_signal_5, future_return_10)
    exclude_cols.update([
        col for col in df.columns 
        if col.startswith("signal_") or 
           col.startswith("binary_signal_") or 
           col.startswith("future_return_")
    ])

    return [col for col in df.columns if col not in exclude_cols]


__all__ = [
    "load_parquet_file",
    "load_and_process_file",
    "add_order_flow_features",
    "engineer_features",
    "create_labels",
    "create_labels_multi_horizon",
    "add_dl_time_series_features",
    "add_transformer_time_series_features",
    "get_feature_columns",
]
