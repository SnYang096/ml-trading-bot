"""Data loading and processing utilities for rolling training workflows."""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer


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
    upper_name = zip_basename.upper()
    symbol_match = re.search(r"([A-Z]+USDT)", upper_name)
    symbol = symbol_match.group(1) if symbol_match else zip_basename.replace(
        ".zip", "")

    date_match = re.search(r"(\d{4})-(\d{2})", zip_basename)
    if date_match:
        date_part = f"{date_match.group(1)}-{date_match.group(2)}"
    else:
        date_part = "unknown"

    return f"{symbol}_{date_part}.parquet"


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
    df.loc[df["future_return"] > threshold, "signal"] = 1  # Long
    df.loc[df["future_return"] < -threshold, "signal"] = 2  # Short

    # Keep backward compatibility: binary_signal for legacy code (1=Long, 0=not Long)
    df["binary_signal"] = (df["signal"] == 1).astype(int)

    return df


def create_labels_multi_horizon(
    df: pd.DataFrame,
    *,
    horizons: list[int] = [1, 5, 10, 15],
    threshold: float = 0.005,
    use_risk_adjusted: bool = False,  # Changed default to False
    vol_window: int = 20,
    use_quantile_threshold: bool = False,
    quantile_window: int = 5000,
    lower_quantile: float = 0.4,
    upper_quantile: float = 0.6,
    quantile_min_periods: int = 200,
    use_rank_percentile:
    bool = True,  # NEW: Use rolling rank percentile (recommended)
    rank_window: int
    | None = None,  # If None, calculated as horizon * 20 (min 100)
    top_percentile: float = 0.7,  # Top 30% = Long
    bottom_percentile: float = 0.3
) -> pd.DataFrame:  # Bottom 30% = Short
    """Create future-return based labels for multiple horizons (3-class: 0=Hold, 1=Long, 2=Short).
    
    Improved version that addresses the issue of fixed threshold in low-volatility periods:
    - Option 0 (RECOMMENDED): Use rolling rank percentile (future_return rank in rolling window)
    - Option 1: Use risk-adjusted returns (Sharpe-like: return / volatility)
    - Option 2: Use rolling quantile thresholds (adaptive to market conditions)
    
    Args:
        df: DataFrame with OHLCV data
        horizons: List of forward bars to look ahead (e.g., [1, 5, 10, 15])
        threshold: Threshold for signal classification (used if all methods=False)
        use_risk_adjusted: If True, use risk-adjusted returns (return / rolling_volatility)
        vol_window: Window size for rolling volatility calculation (used if use_risk_adjusted=True)
        use_quantile_threshold: If True, use rolling quantile thresholds
        quantile_window: Window size for rolling quantile calculation (used if use_quantile_threshold=True)
        lower_quantile: Lower quantile threshold (e.g., 0.4 = 40th percentile)
        upper_quantile: Upper quantile threshold (e.g., 0.6 = 60th percentile)
        quantile_min_periods: Minimum periods required for quantile calculation
        use_rank_percentile: If True (RECOMMENDED), use rolling rank percentile instead of absolute threshold
        rank_window: Window size for rolling rank calculation. If None, will be calculated as horizon * multiplier (default: 20x horizon, min 100)
        top_percentile: Top percentile threshold for Long signal (e.g., 0.7 = top 30%)
        bottom_percentile: Bottom percentile threshold for Short signal (e.g., 0.3 = bottom 30%)
    
    Returns:
        DataFrame with multiple label columns for each horizon:
        - signal_{horizon}: 3-class labels (0=Hold, 1=Long, 2=Short)
        - binary_signal_{horizon}: Backward compatibility (1=Long, 0=not Long)
        - future_return_{horizon}: Original absolute returns
        - rank_percentile_{horizon}: Rank percentile in rolling window (if use_rank_percentile=True)
        - risk_adjusted_return_{horizon}: Risk-adjusted returns (if use_risk_adjusted=True)
        - Also creates backward-compatible 'signal' and 'binary_signal' using the first horizon
    """
    import numpy as np
    import pandas as pd

    df = df.copy()

    for horizon in horizons:
        # Create future return for this horizon (absolute return)
        future_return_col = f"future_return_{horizon}"
        df[future_return_col] = df["close"].shift(-horizon) / df["close"] - 1

        # Choose label generation method (priority: rank_percentile > quantile_threshold > risk_adjusted > fixed_threshold)
        if use_rank_percentile:
            # Method 0 (RECOMMENDED): Use rolling rank percentile
            # This avoids absolute thresholds and adapts to market conditions
            # Example: future_return rank in rolling window > 70th percentile → Long
            #          future_return rank in rolling window < 30th percentile → Short

            # Calculate rank_window based on horizon if not provided
            # Rule: rank_window = horizon * multiplier (default: 20x), minimum 100 bars
            # This ensures the ranking window is proportional to the prediction horizon
            current_rank_window = rank_window
            if current_rank_window is None:
                current_rank_window = max(
                    horizon * 20, 100)  # 20x horizon, but at least 100 bars
                print(
                    f"   📊 Auto-calculated rank_window for horizon={horizon}: {current_rank_window} (horizon * 20, min=100)"
                )

            # Calculate rolling rank percentile (using trailing window to avoid lookahead bias)
            # Shift by 1 to avoid using current value in its own ranking
            shifted_return = df[future_return_col].shift(1)

            # Calculate rank percentile in rolling window
            # Use pandas rolling().rank(pct=True) for efficient calculation
            rank_pct_col = f"rank_percentile_{horizon}"
            min_periods = max(10, current_rank_window //
                              10)  # At least 10% of window

            # More efficient: use pandas rolling rank
            # For each position, calculate rank percentile of current value within trailing window
            rank_pct_series = shifted_return.rolling(
                window=current_rank_window, min_periods=min_periods).apply(
                    lambda x: pd.Series(x).rank(pct=True, method='first').iloc[
                        -1]
                    if len(x) > 0 and not pd.isna(x.iloc[-1]) else np.nan,
                    raw=False)

            df[rank_pct_col] = rank_pct_series

            # Create 3-class signal using rank percentile
            signal_col = f"signal_{horizon}"
            df[signal_col] = 0  # Hold by default

            # Long: rank percentile > top_percentile (e.g., top 30% = rank > 70th percentile)
            df.loc[df[rank_pct_col] > top_percentile, signal_col] = 1  # Long

            # Short: rank percentile < bottom_percentile (e.g., bottom 30% = rank < 30th percentile)
            df.loc[df[rank_pct_col] < bottom_percentile,
                   signal_col] = 2  # Short

            # Invalid samples (NaN in rank percentile) remain as 0 (Hold)
            valid_mask = df[rank_pct_col].notna()

            print(
                f"   ✅ Horizon {horizon}: Using rolling rank percentile "
                f"(window={current_rank_window}, top={top_percentile:.0%}, bottom={bottom_percentile:.0%})"
            )
            print(f"      Valid samples: {valid_mask.sum()}/{len(valid_mask)} "
                  f"({valid_mask.sum()/len(valid_mask)*100:.1f}%)")

            # Check label distribution to prevent constant prediction
            signal_dist = df[signal_col].value_counts().to_dict()
            total_signals = df[signal_col].notna().sum()
            if total_signals > 0:
                long_rate = signal_dist.get(1, 0) / total_signals
                short_rate = signal_dist.get(2, 0) / total_signals
                hold_rate = signal_dist.get(0, 0) / total_signals
                print(
                    f"      Label distribution: Long={long_rate:.2%}, Short={short_rate:.2%}, Hold={hold_rate:.2%}"
                )

                # Rank percentile should give balanced labels (top 30% vs bottom 30% = ~30% each)
                expected_long_rate = 1.0 - top_percentile
                expected_short_rate = bottom_percentile
                if abs(long_rate - expected_long_rate) > 0.1:
                    print(
                        f"      ⚠️  WARNING: Long rate ({long_rate:.2%}) differs from expected ({expected_long_rate:.2%})"
                    )
                    print(
                        f"         → This may indicate insufficient data in rolling window"
                    )
                if abs(short_rate - expected_short_rate) > 0.1:
                    print(
                        f"      ⚠️  WARNING: Short rate ({short_rate:.2%}) differs from expected ({expected_short_rate:.2%})"
                    )
                else:
                    print(
                        f"      ✅ Label distribution matches expected (Long={expected_long_rate:.2%}, Short={expected_short_rate:.2%})"
                    )

        elif use_quantile_threshold:
            # Method 2: Use rolling quantile thresholds (adaptive to market conditions)
            # This avoids the problem of fixed threshold in low-volatility periods
            from time_series_model.pipeline.training.label_utils import rolling_quantile_classification_labels

            # Use shifted returns to avoid lookahead bias
            y_return = df[future_return_col]
            y_quantile_labels, valid_mask, upper_threshold, lower_threshold = rolling_quantile_classification_labels(
                y_return,
                window=quantile_window,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                min_periods=quantile_min_periods,
            )

            # Create 3-class signal: 1=Long, 0=Hold, 2=Short
            signal_col = f"signal_{horizon}"
            df[signal_col] = 0  # Hold by default
            df.loc[valid_mask & (y_quantile_labels == 1),
                   signal_col] = 1  # Long
            df.loc[valid_mask & (y_quantile_labels == 0),
                   signal_col] = 2  # Short
            # Invalid samples (NaN in quantile labels) remain as 0 (Hold)

            print(
                f"   ✅ Horizon {horizon}: Using rolling quantile thresholds "
                f"(window={quantile_window}, q_low={lower_quantile}, q_high={upper_quantile})"
            )
            print(f"      Valid samples: {valid_mask.sum()}/{len(valid_mask)} "
                  f"({valid_mask.sum()/len(valid_mask)*100:.1f}%)")

            # Check label distribution to prevent constant prediction
            signal_dist = df[signal_col].value_counts().to_dict()
            total_signals = df[signal_col].notna().sum()
            if total_signals > 0:
                long_rate = signal_dist.get(1, 0) / total_signals
                short_rate = signal_dist.get(2, 0) / total_signals
                hold_rate = signal_dist.get(0, 0) / total_signals
                print(
                    f"      Label distribution: Long={long_rate:.2%}, Short={short_rate:.2%}, Hold={hold_rate:.2%}"
                )

                # Quantile-based labels should be more balanced, but still check
                if long_rate < 0.01 or long_rate > 0.99:
                    print(
                        f"      ⚠️  WARNING: Extreme label imbalance! Long rate={long_rate:.2%} (should be 1%-99%)"
                    )
                    print(
                        f"         → This is unexpected for quantile-based labels"
                    )
                    print(
                        f"         → Check quantile thresholds (lower_quantile={lower_quantile}, upper_quantile={upper_quantile})"
                    )

        elif use_risk_adjusted:
            # Method 1: Use risk-adjusted returns (Sharpe-like: return / volatility)
            # Calculate rolling volatility (using trailing window to avoid lookahead bias)
            # Use abs(return) as volatility proxy for simplicity
            vol_col = f"volatility_{horizon}"
            df[vol_col] = df[future_return_col].abs().rolling(
                window=vol_window, min_periods=max(3, vol_window // 2)).std()

            # Fill NaN with a small value to avoid division by zero
            df[vol_col] = df[vol_col].fillna(
                df[future_return_col].abs().mean() + 1e-8)

            # Calculate risk-adjusted return (Sharpe-like)
            risk_adjusted_col = f"risk_adjusted_return_{horizon}"
            df[risk_adjusted_col] = df[future_return_col] / df[vol_col]

            # Create 3-class signal using risk-adjusted returns
            signal_col = f"signal_{horizon}"
            df[signal_col] = 0  # Hold by default
            df.loc[df[risk_adjusted_col] > threshold, signal_col] = 1  # Long
            df.loc[df[risk_adjusted_col] < -threshold, signal_col] = 2  # Short

            print(f"   ✅ Horizon {horizon}: Using risk-adjusted returns "
                  f"(threshold={threshold}, vol_window={vol_window})")
            print(f"      Risk-adjusted return stats: "
                  f"mean={df[risk_adjusted_col].mean():.4f}, "
                  f"std={df[risk_adjusted_col].std():.4f}, "
                  f"min={df[risk_adjusted_col].min():.4f}, "
                  f"max={df[risk_adjusted_col].max():.4f}")

            # Check label distribution to prevent constant prediction
            signal_dist = df[signal_col].value_counts().to_dict()
            total_signals = df[signal_col].notna().sum()
            if total_signals > 0:
                long_rate = signal_dist.get(1, 0) / total_signals
                short_rate = signal_dist.get(2, 0) / total_signals
                hold_rate = signal_dist.get(0, 0) / total_signals
                print(
                    f"      Label distribution: Long={long_rate:.2%}, Short={short_rate:.2%}, Hold={hold_rate:.2%}"
                )

                # Warn if extreme imbalance (can cause constant prediction)
                if long_rate < 0.01 or long_rate > 0.99:
                    print(
                        f"      ⚠️  WARNING: Extreme label imbalance! Long rate={long_rate:.2%} (should be 1%-99%)"
                    )
                    print(
                        f"         → Model may degenerate to constant prediction"
                    )
                    print(
                        f"         → Consider using quantile-based thresholds (use_quantile_threshold=True)"
                    )
                elif long_rate < 0.05 or long_rate > 0.95:
                    print(
                        f"      ⚠️  WARNING: Significant label imbalance! Long rate={long_rate:.2%} (recommended: 5%-95%)"
                    )

        else:
            # Method 0: Original fixed threshold (for backward compatibility)
            # This has the problem: in low-volatility periods, most values are near 0
            signal_col = f"signal_{horizon}"
            df[signal_col] = 0  # Hold by default
            df.loc[df[future_return_col] > threshold, signal_col] = 1  # Long
            df.loc[df[future_return_col] < -threshold, signal_col] = 2  # Short

            print(
                f"   ⚠️  Horizon {horizon}: Using fixed threshold ({threshold}) - "
                f"may have issues in low-volatility periods")

        # Keep backward compatibility: binary_signal for legacy code
        binary_signal_col = f"binary_signal_{horizon}"
        df[binary_signal_col] = (df[signal_col] == 1).astype(int)

    # Create backward-compatible columns using the first horizon
    if horizons:
        df["signal"] = df[f"signal_{horizons[0]}"]
        df["binary_signal"] = df[f"binary_signal_{horizons[0]}"]
        df["future_return"] = df[f"future_return_{horizons[0]}"]

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
        from data_tools.dl_sequence_features import add_dl_sequence_features

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
        "future_volatility",
        "classification_label",
        "hl",
        "hc",
        "lc",
        "tr",
    }

    # Also exclude multi-horizon label columns (e.g., signal_1, binary_signal_5, future_return_10)
    exclude_cols.update([
        col for col in df.columns if col.startswith("signal_")
        or col.startswith("binary_signal_") or col.startswith("future_return_")
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
