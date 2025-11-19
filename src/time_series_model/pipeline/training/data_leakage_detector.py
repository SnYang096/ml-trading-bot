"""
Data Leakage Detection Module

This module provides functions to detect data leakage in time series models:
1. Random walk test: Train model on random data, if Rank IC > 0.05, there's leakage
2. Feature-future correlation test: Check if features correlate with future returns
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import TimeSeriesSplit

from time_series_model.pipeline.training.rank_ic_trainer import (
    prepare_rank_ic_labels,
    train_rank_ic_model,
)
from time_series_model.pipeline.training.rank_ic_utils import compute_rank_ic


def generate_random_walk_data(
    n_samples: int = 1000,
    n_features: int = 50,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate random walk data for leakage detection.

    This creates synthetic data where:
    - close: Random walk (cumulative sum of random normal)
    - features: Random normal or derived from close (but no future information)

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with random walk data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random walk for close price
    returns = np.random.randn(n_samples) * 0.01  # 1% volatility
    close = 100 * (1 + returns).cumprod()  # Start at 100

    # Generate OHLCV data
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.abs(np.random.randn(n_samples) * 1000)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Generate ONLY completely random features (no patterns, no correlation with future)
    # These features have zero predictive power by design
    for i in range(n_features):
        # Completely random feature (no correlation with future, no patterns)
        df[f"feature_{i}"] = np.random.randn(n_samples)

    # DO NOT add any technical features derived from price data
    # This ensures the test is truly on random data with no predictive patterns
    # Technical features like MA, volatility, etc. might have spurious correlations

    # Create datetime index
    df.index = pd.date_range(start="2020-01-01", periods=n_samples, freq="15T")

    return df


def _simple_series_random_walk_test(
    n_samples: int = 1500,
    n_features: int = 20,
    hold_period: int = 5,
    n_splits: int = 5,
    seed: Optional[int] = 7,
    threshold: float = 0.03,
) -> Dict:
    """
    Minimal random walk leakage test using pure pandas/numpy pipeline.

    The goal is to eliminate any complex feature engineering, label post-processing,
    or model training side effects. We:
        1. Generate a random walk price series.
        2. Compute future_return via direct shift.
        3. Generate fully independent random features.
        4. Fit a simple OLS model per TSCV fold (no LightGBM, no weights).
        5. Measure Rank IC on validation folds only.
    """

    print("\n" + "-" * 60)
    print("🧪 Simple Random Walk Baseline Test")
    print("-" * 60)

    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n_samples + hold_period)
    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({"close": close})

    for i in range(n_features):
        df[f"baseline_feature_{i}"] = rng.normal(0.0, 1.0, size=len(df))

    df["future_return"] = df["close"].pct_change(hold_period).shift(-hold_period)
    df = df.iloc[:-hold_period].copy()
    df = df.dropna(subset=["future_return"])

    feature_cols = [c for c in df.columns if c.startswith("baseline_feature_")]
    if len(df) < 200 or len(feature_cols) == 0:
        return {
            "test": "random_walk_simple",
            "status": "insufficient_data",
            "message": "Not enough samples or features for baseline test",
        }

    X = df[feature_cols].to_numpy()
    y = df["future_return"].to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    ic_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if len(np.unique(y_train)) < 2:
            print(f"   ⚠️  Fold {fold}: insufficient variation in y, skipping")
            continue

        try:
            coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(X_train) @ y_train

        preds = X_val @ coef
        ic = compute_rank_ic(preds, y_val)
        ic_scores.append(ic if not np.isnan(ic) else 0.0)
        print(f"   Fold {fold}: Rank IC = {ic:.4f}")

    if not ic_scores:
        return {
            "test": "random_walk_simple",
            "status": "insufficient_folds",
            "message": "All folds skipped due to insufficient variation",
        }

    avg_ic = float(np.mean(ic_scores))
    std_ic = float(np.std(ic_scores))
    has_leakage = abs(avg_ic) > threshold

    print(
        f"\n   Avg Rank IC: {avg_ic:.4f} (std={std_ic:.4f}, threshold={threshold:.4f})"
    )
    if has_leakage:
        print("   ⚠️  Baseline test indicates potential leakage")
    else:
        print("   ✅ Baseline test shows no obvious leakage")

    return {
        "test": "random_walk_simple",
        "status": "completed",
        "avg_rank_ic": avg_ic,
        "std_rank_ic": std_ic,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "n_splits": n_splits,
        "threshold": threshold,
        "has_leakage": has_leakage,
    }


def test_random_walk_leakage(
    feature_cols: List[str],
    n_samples: int = 1000,
    n_features: int = 50,
    hold_period: int = 5,
    n_splits: int = 5,
    seed: Optional[int] = 42,
    threshold: float = 0.03,  # Stricter threshold: 0.03 instead of 0.05
    mode: str = "full",
) -> Dict:
    """
    Test for data leakage using random walk data.

    If model achieves Rank IC > threshold on random data, there's likely leakage.

    Args:
        feature_cols: List of feature column names (for compatibility)
        n_samples: Number of random samples to generate
        n_features: Number of random features
        hold_period: Holding period for labels
        n_splits: Number of CV folds
        seed: Random seed
        threshold: Rank IC threshold (default: 0.05)

    Returns:
        Dictionary with test results
    """
    if mode == "simple":
        return _simple_series_random_walk_test(
            n_samples=n_samples,
            n_features=n_features,
            hold_period=hold_period,
            n_splits=n_splits,
            seed=seed,
            threshold=threshold,
        )

    print("\n" + "=" * 60)
    print("🔍 Data Leakage Test: Random Walk")
    print("=" * 60)
    print(f"Generating {n_samples} random walk samples...")

    # Generate random walk data
    df_random = generate_random_walk_data(
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
    )

    # Prepare labels (should be random, no signal)
    print("Preparing labels on random data...")
    df_with_labels = prepare_rank_ic_labels(
        df_random,
        price_col="close",
        asset_col=None,
        date_col=None,
        hold_period=hold_period,
        lookback_window=60,
        ensure_volatility=True,
    )

    valid_samples = df_with_labels["volatility_normalized_target"].notna().sum()
    print(f"   ✅ {valid_samples} valid samples prepared")

    if valid_samples < 100:
        return {
            "test": "random_walk",
            "status": "insufficient_data",
            "valid_samples": valid_samples,
            "message": "Not enough valid samples for testing",
        }

    # Get feature columns (use all numeric columns except price/volume/label columns)
    # IMPORTANT: Only use truly random features, exclude any features that might have
    # been computed from the price data (like technical indicators)
    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "returns",
        "future_return",
        "volatility_normalized_target",
        "return_quantile",
        "tradable",
        "rolling_vol",
        "momentum",
        "trend_strength",
        # Exclude any technical indicators that might have patterns
        "ma_5",
        "ma_20",
        "volatility",
    }

    # Only use features that start with "feature_" (truly random features)
    # This ensures we're testing on completely random data with no patterns
    random_feature_cols = [
        col
        for col in df_with_labels.columns
        if col.startswith("feature_")  # Only use explicitly random features
        and col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df_with_labels[col])
        and df_with_labels[col].notna().sum() > 10
    ]

    if len(random_feature_cols) < 10:
        return {
            "test": "random_walk",
            "status": "insufficient_features",
            "n_features": len(random_feature_cols),
            "message": "Not enough features for testing",
        }

    print(f"Using {len(random_feature_cols)} features for testing...")

    # Train model on random data
    print("Training model on random walk data...")
    try:
        # Ensure we have a proper integer index for TSCV
        # TSCV uses integer indices, so we need to reset if index is not integer
        if not isinstance(df_with_labels.index, pd.RangeIndex):
            df_with_labels_reset = df_with_labels.reset_index(drop=True)
        else:
            df_with_labels_reset = df_with_labels.copy()

        # Ensure all feature columns exist
        available_features = [
            f for f in random_feature_cols if f in df_with_labels_reset.columns
        ]
        if len(available_features) < 10:
            return {
                "test": "random_walk",
                "status": "insufficient_features",
                "n_features": len(available_features),
                "message": f"Only {len(available_features)} features available after index reset",
            }

        # CRITICAL: For random walk test, we need to ensure tradable mask doesn't filter
        # based on future information. Set all samples as tradable for the test.
        # Also, we should not use trend_strength as weights (it might have patterns)
        df_with_labels_reset["tradable"] = True  # Make all samples tradable for test
        if "trend_strength" in df_with_labels_reset.columns:
            df_with_labels_reset["trend_strength"] = 1.0  # Set uniform weights

        models, avg_rank_ic, cv_results = train_rank_ic_model(
            df_with_labels_reset,
            feature_cols=available_features,
            target_col="volatility_normalized_target",
            date_col=None,
            n_splits=n_splits,
            use_gpu=False,
            filter_high_confidence=False,
            min_trend_strength=0.0,
            smooth_target=False,
            weight_col=None,  # Don't use weights for random test
        )

        print(f"\n📊 Random Walk Test Results:")
        print(f"   Average Rank IC: {avg_rank_ic:.4f}")
        print(f"   Threshold: {threshold:.4f}")

        # Calculate standard deviation for additional check
        ic_std = 0.0
        if cv_results is not None and len(cv_results) > 0:
            ic_std = cv_results["rank_ic"].std()
            print(f"   Standard Deviation: {ic_std:.4f}")

        # Check for leakage
        # On completely random data, Rank IC should be close to 0
        # Primary check: if |Rank IC| > threshold, there's likely leakage
        has_leakage = abs(avg_rank_ic) > threshold

        # Additional check: if std is very large relative to mean, might indicate issues
        # This suggests unstable features that might have patterns
        if ic_std > 0.15:  # Very large variance might indicate instability
            print(f"   ⚠️  High variance in Rank IC across folds (std={ic_std:.4f})")
            print(f"   This might indicate unstable features")
            # Don't automatically mark as leakage, but warn

        # Statistical significance check (only as warning, not as leakage marker)
        # If Rank IC is statistically significant AND close to threshold, it's suspicious
        n_folds = len(cv_results) if cv_results is not None else n_splits
        if n_folds > 1 and ic_std > 0:
            se = ic_std / np.sqrt(n_folds)
            # Only warn if it's statistically significant AND above 0.02 (half of threshold)
            if abs(avg_rank_ic) > 2 * se and abs(avg_rank_ic) > 0.02:
                print(
                    f"   ⚠️  Rank IC is statistically significant (|{avg_rank_ic:.4f}| > 2*SE={2*se:.4f})"
                )
                print(
                    f"   This is suspicious but not necessarily leakage (threshold: {threshold:.4f})"
                )
                # Only mark as leakage if it's also above threshold
                if abs(avg_rank_ic) > threshold:
                    has_leakage = True

        if has_leakage:
            print(
                f"   ⚠️  WARNING: Rank IC ({avg_rank_ic:.4f}) > threshold ({threshold:.4f})"
            )
            print(f"   This suggests possible data leakage!")
            print(f"   On completely random data, Rank IC should be close to 0.")
        else:
            print(f"   ✅ Rank IC ({avg_rank_ic:.4f}) <= threshold ({threshold:.4f})")
            print(f"   No obvious leakage detected in this test.")

        return {
            "test": "random_walk",
            "status": "completed",
            "avg_rank_ic": float(avg_rank_ic),
            "threshold": threshold,
            "has_leakage": has_leakage,
            "n_samples": n_samples,
            "n_features": len(random_feature_cols),
            "cv_results": (
                cv_results.to_dict("records") if cv_results is not None else None
            ),
        }
    except Exception as e:
        return {
            "test": "random_walk",
            "status": "error",
            "error": str(e),
            "message": f"Error during random walk test: {e}",
        }


def check_feature_future_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    future_return_col: str = "future_return",
    correlation_threshold: float = 0.1,
    min_samples: int = 100,
) -> Dict:
    """
    Check correlation between features and future returns.

    If many features have high correlation with future returns, there may be leakage.

    Args:
        df: DataFrame with features and future returns
        feature_cols: List of feature column names
        future_return_col: Name of future return column
        correlation_threshold: Threshold for suspicious correlation (default: 0.1)
        min_samples: Minimum samples required for correlation calculation

    Returns:
        Dictionary with correlation analysis results
    """
    print("\n" + "=" * 60)
    print("🔍 Data Leakage Test: Feature-Future Correlation")
    print("=" * 60)

    if future_return_col not in df.columns:
        return {
            "test": "feature_future_correlation",
            "status": "missing_column",
            "message": f"Future return column '{future_return_col}' not found",
        }

    # Align data
    valid_mask = df[future_return_col].notna() & df[feature_cols].notna().all(axis=1)

    if valid_mask.sum() < min_samples:
        return {
            "test": "feature_future_correlation",
            "status": "insufficient_data",
            "valid_samples": int(valid_mask.sum()),
            "min_samples": min_samples,
            "message": f"Not enough valid samples ({valid_mask.sum()} < {min_samples})",
        }

    df_valid = df.loc[valid_mask].copy()
    future_returns = df_valid[future_return_col].values

    print(f"Analyzing {len(feature_cols)} features on {len(df_valid)} samples...")

    correlations = []
    suspicious_features = []

    for feat_col in feature_cols:
        if feat_col not in df_valid.columns:
            continue

        feat_values = df_valid[feat_col].values

        # Skip if feature has no variance
        if np.std(feat_values) < 1e-10:
            continue

        # Calculate Spearman correlation (rank-based, more robust)
        try:
            corr, p_value = spearmanr(feat_values, future_returns)

            if np.isnan(corr):
                continue

            correlations.append(
                {
                    "feature": feat_col,
                    "correlation": float(corr),
                    "p_value": float(p_value),
                    "abs_correlation": float(abs(corr)),
                }
            )

            # Check if suspicious
            if abs(corr) > correlation_threshold:
                suspicious_features.append(
                    {
                        "feature": feat_col,
                        "correlation": float(corr),
                        "p_value": float(p_value),
                    }
                )
        except Exception as e:
            continue

    if not correlations:
        return {
            "test": "feature_future_correlation",
            "status": "no_correlations",
            "message": "Could not calculate any correlations",
        }

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)

    # Statistics
    abs_corrs = [c["abs_correlation"] for c in correlations]
    n_suspicious = len(suspicious_features)
    pct_suspicious = n_suspicious / len(correlations) * 100 if correlations else 0

    print(f"\n📊 Correlation Analysis Results:")
    print(f"   Total features analyzed: {len(correlations)}")
    print(
        f"   Suspicious features (|corr| > {correlation_threshold}): {n_suspicious} ({pct_suspicious:.1f}%)"
    )
    print(f"   Mean |correlation|: {np.mean(abs_corrs):.4f}")
    print(f"   Max |correlation|: {np.max(abs_corrs):.4f}")

    # Check for leakage
    # If > 10% of features have high correlation, or max correlation is very high, suspect leakage
    has_leakage = (
        pct_suspicious > 10.0  # More than 10% of features are suspicious
        or np.max(abs_corrs) > 0.3  # At least one feature has very high correlation
    )

    if has_leakage:
        print(f"   ⚠️  WARNING: Suspicious correlations detected!")
        print(f"   Top 10 features by |correlation|:")
        for i, corr_info in enumerate(correlations[:10], 1):
            print(
                f"      {i}. {corr_info['feature']}: {corr_info['correlation']:.4f} (p={corr_info['p_value']:.4f})"
            )
    else:
        print(f"   ✅ No obvious leakage detected in feature-future correlations.")

    return {
        "test": "feature_future_correlation",
        "status": "completed",
        "n_features_analyzed": len(correlations),
        "n_suspicious": n_suspicious,
        "pct_suspicious": pct_suspicious,
        "mean_abs_correlation": float(np.mean(abs_corrs)),
        "max_abs_correlation": float(np.max(abs_corrs)),
        "threshold": correlation_threshold,
        "has_leakage": has_leakage,
        "top_correlations": correlations[:20],  # Top 20
        "suspicious_features": suspicious_features,
    }


def detect_data_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    future_return_col: str = "future_return",
    run_random_walk_test: bool = True,
    run_correlation_test: bool = True,
    random_walk_params: Optional[Dict] = None,
    correlation_params: Optional[Dict] = None,
) -> Dict:
    """
    Comprehensive data leakage detection.

    Runs both random walk test and feature-future correlation test.

    Args:
        df: DataFrame with features and future returns
        feature_cols: List of feature column names
        future_return_col: Name of future return column
        run_random_walk_test: Whether to run random walk test
        run_correlation_test: Whether to run correlation test
        random_walk_params: Parameters for random walk test
        correlation_params: Parameters for correlation test

    Returns:
        Dictionary with all test results
    """
    results = {
        "random_walk_test": None,
        "correlation_test": None,
        "overall_status": "unknown",
    }

    # Random walk test
    if run_random_walk_test:
        params = random_walk_params or {}
        results["random_walk_test"] = test_random_walk_leakage(
            feature_cols=feature_cols,
            **params,
        )

    # Correlation test
    if run_correlation_test:
        params = correlation_params or {}
        results["correlation_test"] = check_feature_future_correlation(
            df=df,
            feature_cols=feature_cols,
            future_return_col=future_return_col,
            **params,
        )

    # Overall assessment
    has_leakage = False
    if results["random_walk_test"] and results["random_walk_test"].get("has_leakage"):
        has_leakage = True
    if results["correlation_test"] and results["correlation_test"].get("has_leakage"):
        has_leakage = True

    results["overall_status"] = "leakage_detected" if has_leakage else "no_leakage"

    print("\n" + "=" * 60)
    print("📋 Data Leakage Detection Summary")
    print("=" * 60)
    if has_leakage:
        print("⚠️  WARNING: Data leakage may be present!")
    else:
        print("✅ No obvious data leakage detected.")
    print("=" * 60)

    return results
