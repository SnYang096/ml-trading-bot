"""Data loading utilities for dimensionality comparison."""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from data_tools.data_loader import MarketDataLoader
from data_tools.rolling_data import create_labels_multi_horizon


def load_real_market_data(
    data_path: str,
    symbol: str = "ETH-USD",
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: list[int] | None = None,
    feature_type: str = "comprehensive",
    timeframe: str = "5T",
) -> Tuple[np.ndarray, np.ndarray, list, list[int], pd.DataFrame]:
    """Load real market data for one or multiple symbols.
    
    Args:
        symbol: Single symbol or comma-separated symbols (e.g., "ETH-USD" or "ETH-USD,BTC-USD,SOL-USD")
        timeframe: Timeframe for data resampling (e.g., "5T", "15T", "60T", "240T"). Default: "5T"
    """
    # Support multiple symbols (comma-separated)
    symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
    symbols_str = ",".join(symbol_list) if len(
        symbol_list) > 1 else symbol_list[0] if symbol_list else "UNKNOWN"
    print(f"📊 Loading real market data for {symbols_str}...")
    print(f"   Feature type: {feature_type}")
    if len(symbol_list) > 1:
        print(f"   Multi-asset training: {len(symbol_list)} assets")

    try:
        loader = MarketDataLoader(data_path)
        # Load and resample data for all symbols, then merge
        all_dfs = []
        for sym in symbol_list:
            # Create a new loader for each symbol to ensure proper resampling
            symbol_loader = MarketDataLoader(data_path)
            df_single = symbol_loader.load_data(symbol=sym,
                                                start_date=start_date,
                                                end_date=end_date)
            if df_single is not None and not df_single.empty:
                # Resample each symbol's data before merging
                if hasattr(symbol_loader, 'resample_data'):
                    df_single = symbol_loader.resample_data(timeframe)
                elif isinstance(df_single.index, pd.DatetimeIndex):
                    # Fallback: resample manually
                    df_single = df_single.resample(timeframe).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                if df_single is not None and not df_single.empty:
                    all_dfs.append(df_single)

        if not all_dfs:
            print(
                "⚠️ No real data found for any symbol, generating sample data..."
            )
            return create_enhanced_sample_data()

        # Merge all dataframes (already resampled)
        # For multi-asset training, all assets' data are merged together
        # Add symbol identifier for rank-based IC calculation
        all_dfs_with_symbol = []
        for sym, df_single in zip(symbol_list, all_dfs):
            df_with_symbol = df_single.copy()
            df_with_symbol['_symbol'] = sym  # Add symbol identifier
            all_dfs_with_symbol.append(df_with_symbol)

        df = pd.concat(all_dfs_with_symbol, axis=0).sort_index()
        if len(symbol_list) > 1:
            print(
                f"   Merged {len(all_dfs)} asset(s), total {len(df)} samples")
            print(f"   Added symbol identifier for rank-based IC calculation")

        # Store symbol info before feature engineering (in case it gets dropped)
        symbol_info = df['_symbol'].copy() if '_symbol' in df.columns else None

        comprehensive_engineer = ComprehensiveFeatureEngineer(
            feature_types=feature_type)
        df_features = comprehensive_engineer.engineer_all_features(df,
                                                                   fit=True)

        # Restore symbol info if it was dropped during feature engineering
        if symbol_info is not None and '_symbol' not in df_features.columns:
            df_features['_symbol'] = symbol_info.reindex(df_features.index)

        # Parse horizons
        if horizons and len(horizons) > 0:
            horizons_list = horizons
        else:
            horizons_list = [1]

        # Create multi-horizon labels
        print(
            f"   Creating multi-horizon labels for horizons: {horizons_list}")
        df_features = create_labels_multi_horizon(df_features,
                                                  horizons=horizons_list)

        # Store original df_features for multi-horizon label creation
        df_features_stored = df_features.copy()

        # Build safe feature columns (exclude targets/labels and future info)
        # Exclude raw OHLC price features - use derived features instead
        # Exclude raw volume/order flow features - use normalized/derived features instead
        exclude_exact = {
            "timestamp",
            "close",
            "open",  # Exclude raw OHLC prices - use derived features instead
            "high",  # Exclude raw OHLC prices - use derived features instead
            "low",  # Exclude raw OHLC prices - use derived features instead
            "volume",  # Exclude raw volume - use volume_percentile, volume_anomaly, etc.
            "cvd",  # Exclude raw CVD - use cvd_normalized, cvd_spectral_*, cvd_wpt_*, etc.
            "sell_qty",  # Exclude raw sell_qty - use normalized/derived features instead
            "buy_qty",  # Exclude raw buy_qty - use normalized/derived features instead
            "signal",
            "binary_signal",
            "future_return",
            "_symbol",  # Exclude symbol identifier (used for rank-based IC only)
        }
        exclude_prefixes = (
            "signal_",
            "binary_signal_",
            "future_return_",
        )
        feature_cols = [
            col for col in df_features.columns
            if (col not in exclude_exact) and (not any(
                col.startswith(pfx) for pfx in exclude_prefixes))
        ]

        # Debug: engineered feature summary
        try:
            print(
                f"[DEBUG] Engineered features: total={len(feature_cols)} | sample={feature_cols[:10]}"
            )
        except Exception:
            pass

        X = df_features[feature_cols].values

        # Use first horizon for backward compatibility
        default_horizon = horizons_list[0]
        y = df_features[f"binary_signal_{default_horizon}"].dropna(
        ).values  # Use binary signal (0=Short, 1=Long)

        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        print(f"✅ Real data loaded: {X.shape}, {y.shape}")
        print(
            f"   Using horizon: {default_horizon} bars (for backward compatibility)"
        )

        # Store horizons for multi-horizon training
        if len(horizons_list) > 1:
            print(f"   Multi-horizon mode enabled: {horizons_list}")

        return X, y, feature_cols, horizons_list, df_features_stored

    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Error loading real data: {exc}")
        print("📊 Generating sample data...")
        X, y, feature_cols = create_enhanced_sample_data()
        return X, y, feature_cols, [1], pd.DataFrame()


def create_enhanced_sample_data(
    n_samples: int = 10000,
    n_factors: int = 100,
) -> Tuple[np.ndarray, np.ndarray, list]:
    print(
        f"📊 Creating enhanced sample data: {n_samples} samples, {n_factors} features"
    )

    np.random.seed(42)

    factor_names = []
    categories = [
        "momentum",
        "volatility",
        "mean_reversion",
        "trend",
        "volume",
        "liquidity",
        "sentiment",
    ]

    for i in range(n_factors):
        category = categories[i % len(categories)]
        factor_names.append(f"{category}_{i+1}")

    X = np.random.randn(n_samples, n_factors)

    for i in range(0, n_factors, 10):
        if i + 5 < n_factors:
            X[:, i + 1:i + 5] = (X[:, i:i + 4] * 0.7 +
                                 np.random.randn(n_samples, 4) * 0.3)

    momentum_factors = [
        i for i, name in enumerate(factor_names) if "momentum" in name
    ]
    volatility_factors = [
        i for i, name in enumerate(factor_names) if "volatility" in name
    ]
    trend_factors = [
        i for i, name in enumerate(factor_names) if "trend" in name
    ]

    y = (np.tanh(X[:, momentum_factors].mean(axis=1)) * 0.4 +
         np.sin(X[:, volatility_factors].mean(axis=1)) * 0.3 +
         X[:, trend_factors].mean(axis=1) * 0.2 +
         np.random.randn(n_samples) * 0.1)

    return X, y, factor_names

