"""
Rank IC-optimized trainer for time series regression models.

This module implements the best practices from the documentation:
- Volatility-normalized targets (Sharpe-like)
- Historical quantile labels for evaluation
- Tradable mask for sample filtering
- Trend strength as sample weights
- Rank IC (Spearman correlation) as core evaluation metric
- Time series cross-validation
- Confidence-based signal generation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from time_series_model.pipeline.training.label_utils import (
    volatility_normalized_target,
    historical_quantile_label,
    tradable_mask,
    trend_strength_weight,
    compute_momentum,
    rolling_rms_volatility,
)
from time_series_model.pipeline.training.rank_ic_utils import (
    compute_rank_ic,
    prediction_quantile,
    confidence_score,
    generate_trading_signals,
)
from time_series_model.pipeline.training.evaluation_utils import (
    analyze_quantile_distribution,
    compute_confidence_statistics,
    ensure_volatility_feature,
    print_evaluation_summary,
)


def prepare_rank_ic_labels(
    df: pd.DataFrame,
    price_col: str = "close",
    asset_col: Optional[str] = None,
    date_col: Optional[str] = None,
    hold_period: int = 5,
    lookback_window: int = 60,
    vol_mult: float = 0.5,
    min_samples: int = 30,
    ensure_volatility: bool = True,
    use_risk_reward_label: bool = False,  # NEW: Enable R/R label
    rr_ratio_threshold: float = 2.0,  # NEW: Target R/R
    max_holding_bars: int = 24,  # NEW: Max holding period for R/R label
    signal_col: Optional[str] = None,  # NEW: Signal column for R/R label
    use_continuous_rr_label: bool = False,  # NEW: Use continuous R/R or binary
    split_by_reaction_type: bool = False,  # NEW: Split by reversal vs breakout
) -> pd.DataFrame:
    """
    Prepare labels for Rank IC-optimized training.

    This function implements the complete label preparation pipeline:
    1. Compute future_return
    2. Compute rolling volatility
    3. Create volatility-normalized target
    4. Compute historical quantile labels
    5. Create tradable mask
    6. Compute trend strength (if momentum available)

    Args:
        df: DataFrame with price data
        price_col: Name of price column
        asset_col: Optional asset identifier column (for multi-asset)
        date_col: Optional date column (for sorting)
        hold_period: Holding period for future return
        lookback_window: Window for historical quantile calculation
        vol_mult: Multiplier for volatility threshold in tradable mask
        min_samples: Minimum samples for quantile calculation

    Returns:
        DataFrame with added columns:
        - future_return: Raw future return
        - rolling_vol: Rolling volatility
        - volatility_normalized_target: Sharpe-like target
        - return_quantile: Historical quantile label
        - tradable: Tradable mask
        - trend_strength: Sample weight (if momentum available)
    """
    df = df.copy()

    # Ensure volatility feature exists
    if ensure_volatility:
        df = ensure_volatility_feature(
            df,
            price_col=price_col,
            volatility_col="rolling_vol",
            window=lookback_window,
            asset_col=asset_col,
        )

    # Sort by date if date column is provided
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    # Group columns for multi-asset support
    group_cols = [asset_col] if asset_col and asset_col in df.columns else []

    # 0. Compute R/R label if enabled (before computing future_return)
    if use_risk_reward_label:
        from time_series_model.pipeline.training.label_utils import (
            compute_rr_label,
            classify_sr_reaction,
            compute_rr_label_by_reaction,
        )

        # Determine signal column
        if signal_col is None:
            # Try to find signal column
            for col in ["signal", "sr_signal", "trading_signal"]:
                if col in df.columns:
                    signal_col = col
                    break

        if signal_col and signal_col in df.columns:
            # Classify SR reaction type if split_by_reaction_type is enabled
            if split_by_reaction_type:
                print(f"   📊 Classifying SR reaction types...")
                df["sr_reaction"] = classify_sr_reaction(
                    df,
                    signal_col=signal_col,
                    price_col=price_col,
                    atr_col="atr",
                    atr_window=14,
                    lookback_window=5,
                )

                reversal_count = (df["sr_reaction"] == "reversal").sum()
                breakout_count = (df["sr_reaction"] == "breakout").sum()
                print(f"      ✅ Reaction types classified:")
                print(f"         - Reversal: {reversal_count} samples")
                print(f"         - Breakout: {breakout_count} samples")

            print(
                f"   📊 Computing R/R labels (signal_col={signal_col}, rr_ratio={rr_ratio_threshold})"
            )

            if split_by_reaction_type:
                # Compute separate labels for reversal and breakout
                print(f"      Computing R/R labels for reversal opportunities...")
                rr_reversal = compute_rr_label_by_reaction(
                    df,
                    signal_col=signal_col,
                    reaction_col="sr_reaction",
                    price_col=price_col,
                    atr_col="atr",
                    atr_window=14,
                    rr_ratio=rr_ratio_threshold,
                    max_holding_bars=max_holding_bars,
                    stop_loss_r=1.0,
                    take_profit_r=rr_ratio_threshold,
                    reaction_type="reversal",
                    use_continuous_label=use_continuous_rr_label,
                )
                df["rr_reversal_achieved"] = rr_reversal

                print(f"      Computing R/R labels for breakout opportunities...")
                rr_breakout = compute_rr_label_by_reaction(
                    df,
                    signal_col=signal_col,
                    reaction_col="sr_reaction",
                    price_col=price_col,
                    atr_col="atr",
                    atr_window=14,
                    rr_ratio=rr_ratio_threshold,
                    max_holding_bars=max_holding_bars,
                    stop_loss_r=1.0,
                    take_profit_r=rr_ratio_threshold,
                    reaction_type="breakout",
                    use_continuous_label=use_continuous_rr_label,
                )
                df["rr_breakout_achieved"] = rr_breakout

                # For SR reversal strategy, use reversal labels as primary target
                # Breakout labels can be used for filtering or separate model
                df["rr_achieved"] = (
                    rr_reversal  # Primary target for SR reversal strategy
                )
                df["volatility_normalized_target"] = rr_reversal.fillna(0.0)

                valid_reversal = rr_reversal.notna().sum()
                valid_breakout = rr_breakout.notna().sum()
                if valid_reversal > 0:
                    reversal_success = (rr_reversal == 1.0).sum() / valid_reversal * 100
                    print(
                        f"      ✅ Reversal R/R labels: {valid_reversal} samples, {reversal_success:.2f}% success"
                    )
                if valid_breakout > 0:
                    breakout_success = (rr_breakout == 1.0).sum() / valid_breakout * 100
                    print(
                        f"      ✅ Breakout R/R labels: {valid_breakout} samples, {breakout_success:.2f}% success"
                    )
            else:
                # Compute unified R/R labels (no split)
                rr_labels = compute_rr_label(
                    df,
                    signal_col=signal_col,
                    price_col=price_col,
                    atr_col="atr",
                    atr_window=14,
                    rr_ratio=rr_ratio_threshold,
                    max_holding_bars=max_holding_bars,
                    stop_loss_r=1.0,
                    take_profit_r=rr_ratio_threshold,
                    use_continuous_label=use_continuous_rr_label,
                )
                df["rr_achieved"] = rr_labels

                # Use R/R label as the target
                if use_continuous_rr_label:
                    # Continuous: use realized R/R directly
                    df["volatility_normalized_target"] = rr_labels.fillna(0.0)
                else:
                    # Binary: use as-is (0 or 1)
                    df["volatility_normalized_target"] = rr_labels.fillna(0.0)

                valid_rr = rr_labels.notna().sum()
                if valid_rr > 0:
                    success_rate = (rr_labels == 1.0).sum() / valid_rr * 100
                    print(f"   ✅ R/R labels computed: {valid_rr} valid samples")
                    print(f"      Success rate: {success_rate:.2f}%")
                else:
                    print(f"   ⚠️  Warning: No valid R/R labels computed")
        else:
            print(
                f"   ⚠️  Warning: signal_col '{signal_col}' not found, falling back to future_return"
            )
            use_risk_reward_label = False

    # 1. Compute future return (only if not using R/R label)
    if not use_risk_reward_label:
        # ⚠️  FIXED: Avoid using current bar's close (which may not be finalized in real-time)
        # Strategy: Use close[t+1] as entry price (assume trade at t+1 open)
        # future_return[t] = (close[t+1+horizon] - close[t+1]) / close[t+1]
        if group_cols:
            # Compute future_return per group using transform for better alignment
            def _compute_future_return(group: pd.Series) -> pd.Series:
                # Entry price: close[t+1]
                price_entry = group.shift(-1)
                # Exit price: close[t+1+hold_period]
                price_exit = group.shift(-1 - hold_period)
                # Future return: (exit - entry) / entry
                return (price_exit - price_entry) / price_entry

            # Use transform to ensure proper alignment with original dataframe
            future_return = df.groupby(group_cols, group_keys=False)[
                price_col
            ].transform(_compute_future_return)
            # Ensure future_return is a Series (transform should return Series, but check anyway)
            if isinstance(future_return, pd.DataFrame):
                future_return = future_return.iloc[:, 0]
            df["future_return"] = future_return

            # Debug: Check if calculation produced any valid values
            valid_future_return = df["future_return"].notna().sum()
            if valid_future_return == 0 and len(df) > 0:
                print(
                    f"   ⚠️  Warning: future_return calculation (grouped) produced 0 valid values"
                )
                print(f"      Total samples: {len(df)}")
                print(f"      Group columns: {group_cols}")
                print(f"      hold_period: {hold_period}")
                # Check a sample group
                if group_cols and group_cols[0] in df.columns:
                    sample_group_val = df[group_cols[0]].iloc[0]
                    sample_group = df[df[group_cols[0]] == sample_group_val]
                    print(
                        f"      Sample group '{group_cols[0]}={sample_group_val}': {len(sample_group)} samples"
                    )
                    if len(sample_group) > 0:
                        price_entry_sample = sample_group[price_col].shift(-1)
                        price_exit_sample = sample_group[price_col].shift(
                            -1 - hold_period
                        )
                        print(
                            f"      price_entry non-null in sample group: {price_entry_sample.notna().sum()}"
                        )
                        print(
                            f"      price_exit non-null in sample group: {price_exit_sample.notna().sum()}"
                        )
        else:
            # Entry price: close[t+1]
            price_series = df[price_col]
            # Ensure price_series is a Series, not DataFrame
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]  # Take first column if DataFrame
            price_entry = price_series.shift(-1)
            # Exit price: close[t+1+hold_period]
            price_exit = price_series.shift(-1 - hold_period)
            # Future return: (exit - entry) / entry
            future_return = (price_exit - price_entry) / price_entry
            # Ensure future_return is a Series
            if isinstance(future_return, pd.DataFrame):
                future_return = future_return.iloc[:, 0]
            df["future_return"] = future_return

            # Debug: Check if calculation produced any valid values
            valid_future_return = df["future_return"].notna().sum()
            if valid_future_return == 0 and len(df) > 0:
                print(
                    f"   ⚠️  Warning: future_return calculation produced 0 valid values"
                )
                print(f"      Total samples: {len(df)}")
                print(f"      price_entry non-null: {price_entry.notna().sum()}")
                print(f"      price_exit non-null: {price_exit.notna().sum()}")
                print(f"      hold_period: {hold_period}")
                print(
                    f"      Sample price_entry values (first 5): {price_entry.head().tolist()}"
                )
                print(
                    f"      Sample price_exit values (first 5): {price_exit.head().tolist()}"
                )
    else:
        # If using R/R label, still compute future_return for compatibility
        # but it won't be used as the target
        if "future_return" not in df.columns:
            price_series = df[price_col]
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
            price_entry = price_series.shift(-1)
            price_exit = price_series.shift(-1 - hold_period)
            future_return = (price_exit - price_entry) / price_entry
            if isinstance(future_return, pd.DataFrame):
                future_return = future_return.iloc[:, 0]
            df["future_return"] = future_return

    # 2. Compute rolling volatility (if not already computed by ensure_volatility_feature)
    # Note: pct_change(hold_period) computes historical returns: (close[t] - close[t-hold_period]) / close[t-hold_period]
    # This is correct - rolling_vol[t] uses only historical information up to time t
    if "rolling_vol" not in df.columns or df["rolling_vol"].isna().all():

        def _rolling_vol(series: pd.Series) -> pd.Series:
            # Compute historical returns: (close[t] - close[t-hold_period]) / close[t-hold_period]
            rets = series.pct_change(hold_period)
            # rolling_vol[t] = std of historical returns over [t-window, t]
            # Note: rets has NaN for first hold_period values, which is correct
            rolling_vol = rets.rolling(
                window=lookback_window, min_periods=min_samples
            ).std()
            # Reindex to match original series index (preserve NaN positions)
            return rolling_vol.reindex(series.index)

        if group_cols:
            df["rolling_vol"] = df.groupby(group_cols)[price_col].transform(
                _rolling_vol
            )
        else:
            df["rolling_vol"] = _rolling_vol(df[price_col])

    # 3. Create volatility-normalized target
    if not use_risk_reward_label:
        # Use traditional volatility-normalized target
        df["volatility_normalized_target"] = volatility_normalized_target(
            df["future_return"], df["rolling_vol"]
        )
    # If using R/R label, volatility_normalized_target was already set above

    # 4. Compute historical quantile label
    asset_series = df[asset_col] if asset_col and asset_col in df.columns else None
    df["return_quantile"] = historical_quantile_label(
        df["future_return"],
        lookback_window=lookback_window,
        hold_period=hold_period,
        min_samples=min_samples,
        asset_col=asset_series,
    )

    # 5. Create tradable mask
    df["tradable"] = tradable_mask(
        df["future_return"],
        df["rolling_vol"],
        df["return_quantile"],
        vol_mult=vol_mult,
    )

    # 6. Compute trend strength (if momentum available or can be computed)
    if "momentum" not in df.columns:
        # Compute momentum from price
        df["momentum"] = compute_momentum(
            df[price_col],
            window=20,
            diff_period=5,
            asset_col=asset_series,
        )

    df["trend_strength"] = trend_strength_weight(
        df["momentum"],
        df["rolling_vol"],
    )

    return df


def evaluate_model_performance(
    df: pd.DataFrame,
    signals: pd.Series,
    return_quantile_col: str = "return_quantile",
    pred_quantile_col: str = "pred_quantile",
    confidence_col: str = "confidence_score",
    true_return_col: str = "future_return",
    confidence_threshold: float = 0.85,
    hold_period: Optional[int] = None,  # NEW: Hold period to prevent overlapping trades
) -> Dict:
    """
    Comprehensive model performance evaluation.

    This function combines quantile distribution analysis and confidence statistics
    to provide a complete evaluation of model performance.

    Args:
        df: DataFrame with predictions, signals, and labels
        signals: Trading signals (1=Long, -1=Short, 0=Hold)
        return_quantile_col: Name of return quantile column
        pred_quantile_col: Name of prediction quantile column
        confidence_col: Name of confidence score column
        true_return_col: Name of true return column
        confidence_threshold: Minimum confidence threshold

    Returns:
        Dictionary with complete evaluation statistics
    """
    results = {}

    # Quantile distribution analysis
    if return_quantile_col in df.columns:
        return_quantile = df[return_quantile_col]
        pred_quantile = (
            df[pred_quantile_col] if pred_quantile_col in df.columns else None
        )

        quantile_stats = analyze_quantile_distribution(
            return_quantile,
            pred_quantile=pred_quantile,
        )
        results["quantile_distribution"] = quantile_stats

    # Confidence statistics
    if confidence_col in df.columns and true_return_col in df.columns:
        # Pass price column if available for proper single-period return calculation
        price_series = None
        if "close" in df.columns:
            price_series = df["close"]
        elif "price" in df.columns:
            price_series = df["price"]

        confidence_stats = compute_confidence_statistics(
            signals,
            df[true_return_col],
            df[confidence_col],
            confidence_threshold=confidence_threshold,
            price_col=price_series,
            predictions=df["pred"] if "pred" in df.columns else None,
            use_multi_period_returns=True,  # FIXED: Use future_return directly
            hold_period=hold_period,  # FIXED: Prevent overlapping trades
        )
        results["confidence_statistics"] = confidence_stats

    # Print summary
    if "quantile_distribution" in results and "confidence_statistics" in results:
        print_evaluation_summary(
            results["quantile_distribution"],
            results["confidence_statistics"],
        )

    return results


def filter_high_confidence_samples(
    df: pd.DataFrame,
    trend_strength_col: str = "trend_strength",
    min_trend_strength: float = 1.0,
) -> pd.DataFrame:
    """
    Filter samples to only high-confidence periods (strong trends).

    This filters out choppy/consolidation periods where signals are weak,
    focusing training on periods with clear trends.

    Args:
        df: DataFrame with trend_strength column
        trend_strength_col: Name of trend strength column
        min_trend_strength: Minimum trend strength to keep

    Returns:
        DataFrame with filtered samples
    """
    if trend_strength_col not in df.columns:
        print(
            f"   ⚠️  Warning: {trend_strength_col} not found, skipping sample filtering"
        )
        return df

    high_confidence_mask = df[trend_strength_col] >= min_trend_strength
    n_filtered = high_confidence_mask.sum()
    n_total = len(df)

    print(
        f"   📊 Sample filtering: {n_filtered}/{n_total} ({n_filtered/n_total:.1%}) samples with trend_strength >= {min_trend_strength}"
    )

    return df[high_confidence_mask].copy()


def train_rank_ic_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "volatility_normalized_target",
    tradable_col: str = "tradable",
    weight_col: Optional[str] = "trend_strength",
    date_col: Optional[str] = None,
    n_splits: int = 5,
    tscv_gap: int = 0,
    lgbm_params: Optional[Dict] = None,
    use_gpu: bool = True,
    filter_high_confidence: bool = False,
    min_trend_strength: float = 1.0,
    smooth_target: bool = False,
    smooth_method: str = "moving_average",
    smooth_window: int = 3,
    hold_period: Optional[int] = None,
) -> Tuple[List[lgb.Booster], float, pd.DataFrame, List[str]]:
    """
    Train Rank IC-optimized model with time series cross-validation.

    This function:
    1. Uses TimeSeriesSplit to prevent time leakage
    2. Filters samples using tradable mask (optional)
    3. Uses sample weights (trend strength) for training
    4. Evaluates using Rank IC (Spearman correlation)
    5. Returns ensemble of models from all CV folds

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        target_col: Name of target column (volatility_normalized_target)
        tradable_col: Name of tradable mask column
        weight_col: Optional name of sample weight column
        date_col: Optional date column for sorting
        n_splits: Number of CV folds
        lgbm_params: LightGBM parameters
        use_gpu: Whether to use GPU
        filter_high_confidence: Whether to filter to high-confidence samples only
        min_trend_strength: Minimum trend strength for filtering
        smooth_target: Whether to smooth target variable
        smooth_method: Smoothing method ("moving_average", "ewm", or "quantile")
        smooth_window: Window size for smoothing
        hold_period: Optional holding period for adaptive anti-overfitting parameters

    Returns:
        Tuple of (models, avg_rank_ic, results_df, used_feature_cols):
        - models: List of trained models (one per fold)
        - avg_rank_ic: Average Rank IC across folds
        - results_df: DataFrame with predictions and metrics per fold
        - used_feature_cols: Final feature list actually used for training
    """
    # Prepare data
    # Diagnostic: Check NaN counts before dropping
    target_cols = [target_col, "future_return", "return_quantile"]

    # Validate feature columns exist in dataframe
    print(f"   📊 Checking feature columns: requested {len(feature_cols)} features")
    print(f"   📊 DataFrame has {len(df.columns)} columns")
    missing_feature_cols = [col for col in feature_cols if col not in df.columns]
    if missing_feature_cols:
        print(
            f"   ⚠️  Warning: {len(missing_feature_cols)} feature columns not found in dataframe:"
        )
        for col in missing_feature_cols[:20]:
            print(f"      - {col}")
        if len(missing_feature_cols) > 20:
            print(f"      ... and {len(missing_feature_cols) - 20} more")
        # Remove missing columns from feature list
        feature_cols = [col for col in feature_cols if col in df.columns]
        print(
            f"   ✅ Using {len(feature_cols)} available feature columns (removed {len(missing_feature_cols)} missing)"
        )
    else:
        print(f"   ✅ All {len(feature_cols)} requested features found in dataframe")

    if len(feature_cols) == 0:
        raise ValueError(
            f"No valid feature columns found! "
            f"Requested {len(feature_cols)} features, but none exist in dataframe. "
            f"Available columns: {list(df.columns)[:30]}..."
        )

    required_cols = feature_cols + target_cols
    print(f"   📊 Data preparation diagnostics:")
    print(f"      Initial samples: {len(df)}")
    print(f"      Feature columns: {len(feature_cols)}")

    # Check target columns first (these are critical)
    missing_target_cols = [col for col in target_cols if col not in df.columns]
    if missing_target_cols:
        raise ValueError(
            f"Missing required target columns: {missing_target_cols}. "
            f"Available columns: {list(df.columns)[:20]}..."
        )

    for col in target_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            print(
                f"      - {col}: {nan_count} NaN values ({nan_count/len(df)*100:.1f}%)"
            )

    # Check feature columns with most NaN values
    feature_nan_counts = []
    for col in feature_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                feature_nan_counts.append((col, nan_count, nan_count / len(df) * 100))

    # Sort by NaN count and show top 10
    feature_nan_counts.sort(key=lambda x: x[1], reverse=True)
    for col, nan_count, pct in feature_nan_counts[:10]:
        print(f"      - {col}: {nan_count} NaN values ({pct:.1f}%)")
    if len(feature_nan_counts) > 10:
        print(
            f"      ... and {len(feature_nan_counts) - 10} more features with NaN values"
        )

    # First, drop rows where target columns are NaN (these are required)
    df = df.dropna(subset=target_cols).copy()
    print(f"      After dropping rows with NaN targets: {len(df)} samples")

    # Then, handle feature NaN values by filling with 0 (or could use median/forward fill)
    # This is less aggressive than dropping all rows with any NaN feature
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            # Fill NaN with 0 for features (could also use median or forward fill)
            df[col] = df[col].fillna(0)

    print(f"      After filling feature NaN values: {len(df)} samples")

    if len(df) == 0:
        print(f"      ⚠️  All samples removed! This is likely due to:")
        print(
            f"         - Missing future_return (need {hold_period if hold_period else 5} future bars)"
        )
        print(f"         - Missing target or quantile labels")

    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    # Optional: Smooth target to reduce noise
    if smooth_target:
        from time_series_model.pipeline.training.label_utils import smooth_target

        # Note: asset_col parameter is not available in this function, would need to be added
        # For now, smoothing is done without asset grouping
        df[target_col] = smooth_target(
            pd.Series(df[target_col], index=df.index),
            method=smooth_method,
            window=smooth_window,
            asset_col=None,  # Could be enhanced to support asset_col parameter
        )
        print(f"   ✅ Target smoothed using {smooth_method} (window={smooth_window})")

    # Optional: Filter to high-confidence samples only
    if filter_high_confidence:
        df = filter_high_confidence_samples(
            df, weight_col or "trend_strength", min_trend_strength
        )

    # Adaptive parameters based on sample size and feature count
    # For small samples + long horizon, use stronger regularization to prevent overfitting
    n_samples = len(df)
    n_features = len(feature_cols)
    samples_per_feature = n_samples / max(n_features, 1)

    # Diagnostic: Print feature count
    print(f"   📊 Feature count: {n_features} features, {n_samples} samples")
    if n_features == 0:
        raise ValueError(
            f"Zero features available! "
            f"Feature columns list: {feature_cols[:20] if feature_cols else 'EMPTY'}..."
        )
    if n_features == 1:
        print(f"   ⚠️  WARNING: Only 1 feature available! This may indicate a problem.")
        print(f"      Feature: {feature_cols[0] if feature_cols else 'N/A'}")

    # Determine if we need anti-overfitting measures
    # Get hold_period from parameter or default
    if hold_period is None:
        hold_period = 5  # Default

    is_small_sample = n_samples < 3000
    is_long_horizon = hold_period >= 20
    is_high_dim = samples_per_feature < 10  # Less than 10 samples per feature

    # Feature selection for high-dimensional cases
    # DISABLED: Use all features from config, even if correlation is NaN
    # Original logic was:
    # if is_high_dim and n_features > 50:
    #     # Select top features based on correlation with target
    #     ...
    # Now we use all features from the configuration file regardless of correlation
    if False:  # Disabled: always use all features from config
        print(
            f"   ⚠️  High-dimensional case detected ({n_features} features, {samples_per_feature:.1f} samples/feature)"
        )
        print(
            f"      Applying feature selection to reduce to top {min(50, int(n_samples / 20))} features"
        )
        # Select top features based on correlation with target
        feature_importance_scores = []
        invalid_features = []  # Track features that couldn't be scored
        for col in feature_cols:
            try:
                corr = abs(df[col].corr(df[target_col]))
                if not np.isnan(corr):
                    feature_importance_scores.append((col, corr))
                else:
                    invalid_features.append((col, "correlation is NaN"))
            except Exception as e:
                invalid_features.append((col, f"error: {str(e)[:50]}"))

        # Log invalid features if any
        if invalid_features:
            print(
                f"      ⚠️  {len(invalid_features)} features could not be scored (skipped in selection):"
            )
            for feat, reason in invalid_features[:10]:
                print(f"         - {feat}: {reason}")
            if len(invalid_features) > 10:
                print(f"         ... and {len(invalid_features) - 10} more")

        # Sort by correlation and keep top features
        feature_importance_scores.sort(key=lambda x: x[1], reverse=True)
        max_features = min(50, int(n_samples / 20), len(feature_importance_scores))
        selected_features = [f[0] for f in feature_importance_scores[:max_features]]
        feature_cols = selected_features
        n_features = len(feature_cols)
        samples_per_feature = n_samples / max(n_features, 1)
        print(
            f"      Selected {n_features} features from {len(feature_importance_scores)} valid features "
            f"(max allowed: {max_features}, samples/feature: {samples_per_feature:.1f})"
        )

    # Use all features from config (even if correlation is NaN)
    print(
        f"   ✅ Using all {n_features} features from configuration "
        f"(samples/feature: {samples_per_feature:.1f})"
    )

    # Validate feature columns before creating X
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    if len(available_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols)
        print(
            f"   ⚠️  Warning: {len(missing)} feature columns missing after data preparation:"
        )
        for col in list(missing)[:10]:
            print(f"      - {col}")
        feature_cols = available_feature_cols

    if len(feature_cols) == 0:
        raise ValueError(
            f"No valid feature columns available! "
            f"DataFrame has {len(df.columns)} columns: {list(df.columns)[:30]}..."
        )

    print(f"   ✅ Using {len(feature_cols)} feature columns for training")

    X = df[feature_cols].values
    y = df[target_col].values
    y_true_return = df["future_return"].values

    # Validate X shape
    if X.shape[1] == 0:
        raise ValueError(
            f"Feature matrix X has 0 features! "
            f"Feature columns: {feature_cols[:10]}..."
        )
    print(f"   ✅ Feature matrix shape: {X.shape} (samples, features)")
    tradable = (
        df[tradable_col].values
        if tradable_col in df.columns
        else np.ones(len(df), dtype=bool)
    )
    weights = df[weight_col].values if weight_col and weight_col in df.columns else None

    # Check if we have enough samples for cross-validation
    # TimeSeriesSplit requires at least n_splits + 1 samples
    min_samples_required = n_splits + 1
    if len(X) < min_samples_required:
        error_msg = (
            f"Insufficient samples for cross-validation: "
            f"have {len(X)} samples, need at least {min_samples_required} samples "
            f"for {n_splits} folds. "
            f"This may be due to aggressive filtering or missing data."
        )
        print(f"   ❌ {error_msg}")
        raise ValueError(error_msg)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    ic_scores = []
    fold_results = []

    if tscv_gap > 0:
        print(
            f"   ℹ️  Applying TSCV gap of {tscv_gap} samples between train/validation folds"
        )

    if is_small_sample or is_long_horizon or is_high_dim:
        print(
            f"   ⚠️  Small sample ({n_samples}) + long horizon detected, using anti-overfitting parameters"
        )
        print(f"      Samples per feature: {samples_per_feature:.1f}")
        # Stronger regularization for small samples
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": min(
                15, max(7, int(n_samples / 100))
            ),  # Adaptive: smaller for small samples
            "learning_rate": 0.01,  # Lower learning rate (reduced from 0.02 to prevent overfitting)
            "feature_fraction": 0.5,  # More aggressive feature sampling (reduced from 0.7)
            "bagging_fraction": 0.7,  # More aggressive bagging
            "bagging_freq": 3,
            "min_data_in_leaf": max(
                50, int(n_samples / 30)
            ),  # More samples per leaf (increased from 20)
            "min_gain_to_split": 0.2,  # Higher threshold to split (increased from 0.1)
            "lambda_l1": 0.5,  # L1 regularization (increased from 0.1)
            "lambda_l2": 0.5,  # L2 regularization (increased from 0.1)
            "max_depth": 4,  # Limit tree depth (reduced from 5)
            "verbose": -1,
            "force_col_wise": True,
        }
    else:
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "force_col_wise": True,
        }

    if lgbm_params:
        default_params.update(lgbm_params)

    if use_gpu:
        default_params.update(
            {
                "device": "cuda",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            }
        )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if tscv_gap > 0:
            if len(train_idx) <= tscv_gap:
                print(
                    f"   ⚠️  Fold {fold+1}: Not enough training samples after applying gap ({len(train_idx)} <= gap {tscv_gap}), skipping fold"
                )
                continue
            train_idx = train_idx[:-tscv_gap]
        # Time alignment check: ensure validation set is after training set
        if date_col and date_col in df.columns:
            train_end_date = df.loc[train_idx, date_col].max()
            val_start_date = df.loc[val_idx, date_col].min()
            if val_start_date <= train_end_date:
                print(f"   ⚠️  WARNING: Fold {fold+1} time leakage detected!")
                print(f"      Train end: {train_end_date}, Val start: {val_start_date}")

        # Filter by tradable mask (optional)
        train_mask = tradable[train_idx]
        X_train = X[train_idx][train_mask]
        y_train = y[train_idx][train_mask]
        w_train = weights[train_idx][train_mask] if weights is not None else None

        X_val = X[val_idx]
        y_val_true_ret = y_true_return[val_idx]
        quantile_val = df.loc[val_idx, "return_quantile"].values

        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            feature_name=feature_cols,  # Preserve feature names
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val,
            label=y[val_idx],
            reference=train_data,
            feature_name=feature_cols,  # Preserve feature names
            free_raw_data=False,
        )

        # Adaptive early stopping based on sample size
        # For small samples, use more aggressive early stopping to prevent overfitting
        if is_small_sample or is_long_horizon or is_high_dim:
            stopping_rounds = max(
                10,
                min(
                    30, int(len(X_train) / 30)
                ),  # More aggressive (reduced from 20-50 to 10-30)
            )  # Adaptive stopping
            num_boost_round = max(
                50, min(200, int(len(X_train) / 10))  # Reduced max iterations
            )  # Limit iterations
        else:
            stopping_rounds = 30  # Reduced from 50
            num_boost_round = 300  # Reduced from 500

        # Train model
        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Predict and evaluate Rank IC
        pred_val = model.predict(X_val)
        # 【关键修复】：统一样本过滤方式，与 OOS 测试保持一致
        # 同时检查 pred、future_return 和 quantile_val（如果可用）
        # quantile_val 为 NaN 的样本通常是数据不足时的低质量样本，应该排除
        # 但也要确保 pred 和 future_return 有效，以与 OOS 测试保持一致
        valid_mask = (
            ~np.isnan(pred_val)
            & ~np.isnan(y_val_true_ret)
            & np.isfinite(pred_val)
            & np.isfinite(y_val_true_ret)
            & ~np.isnan(quantile_val)  # 保留 quantile_val 检查，排除低质量样本
        )

        if valid_mask.sum() > 10:
            ic = compute_rank_ic(pred_val[valid_mask], y_val_true_ret[valid_mask])
            ic_scores.append(ic if not np.isnan(ic) else 0.0)
            print(f"   Fold {fold+1}: Rank IC = {ic:.4f}")
        else:
            ic_scores.append(0.0)
            print(f"   Fold {fold+1}: Insufficient valid samples for Rank IC")

        models.append(model)

        # Store fold results
        fold_results.append(
            {
                "fold": fold + 1,
                "rank_ic": ic if valid_mask.sum() > 10 else 0.0,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_valid": valid_mask.sum(),
            }
        )

    avg_ic = np.mean(ic_scores)
    std_ic = np.std(ic_scores)

    print(f"\n   ✅ Average Rank IC: {avg_ic:.4f} ± {std_ic:.4f}")

    results_df = pd.DataFrame(fold_results)

    return models, avg_ic, results_df, feature_cols


def generate_ensemble_signals(
    df: pd.DataFrame,
    models: List[lgb.Booster],
    feature_cols: List[str],
    confidence_threshold: float = 0.85,
    long_threshold: float = 0.9,
    short_threshold: float = 0.1,
    asset_col: Optional[str] = None,
    signal_method: str = "quantile",  # "quantile", "sign", "hybrid", "optimized"
    calibrate_predictions: bool = False,
) -> pd.DataFrame:
    """
    Generate trading signals using ensemble of models.

    This function:
    1. Computes ensemble predictions (average of all models)
    2. Computes prediction quantile
    3. Computes confidence score
    4. Generates signals based on selected method

    Args:
        df: DataFrame with features
        models: List of trained models
        feature_cols: List of feature column names
        confidence_threshold: Minimum confidence to trade
        long_threshold: Quantile threshold for Long
        short_threshold: Quantile threshold for Short
        asset_col: Optional asset identifier for multi-asset
        signal_method: Signal generation method
            - "quantile": Use quantile (current method, default)
            - "sign": Use prediction sign directly
            - "hybrid": Combine sign and quantile
            - "optimized": Optimize threshold based on historical performance
        calibrate_predictions: Whether to calibrate predictions (requires future_return in df)

    Returns:
        DataFrame with added columns:
        - pred: Ensemble prediction
        - pred_quantile: Prediction quantile
        - confidence_score: Confidence score
        - signal: Trading signal (1=Long, -1=Short, 0=Hold)
    """
    df = df.copy()
    X = df[feature_cols].values

    # Ensemble prediction (average of all models)
    preds = np.array([model.predict(X) for model in models])
    df["pred"] = np.mean(preds, axis=0)

    # Optionally calibrate predictions
    if calibrate_predictions and "future_return" in df.columns:
        try:
            from time_series_model.pipeline.training.rank_ic_utils_improved import (
                calibrate_predictions as calibrate,
            )

            df["pred"] = calibrate(
                pd.Series(df["pred"], index=df.index),
                pd.Series(df["future_return"], index=df.index),
                method="sigmoid",
            )
            print(f"   ✅ Calibrated predictions using sigmoid scaling")
        except Exception as e:
            print(f"   ⚠️  Prediction calibration failed: {e}")

    # 【关键修复】：先计算分位数，再检测并反转负相关的预测值
    # 这样可以确保分位数计算基于原始预测值分布，而不是反转后的分布
    # Compute prediction quantile FIRST (before potential inversion)
    asset_series = df[asset_col] if asset_col and asset_col in df.columns else None
    df["pred_quantile"] = prediction_quantile(
        pd.Series(df["pred"], index=df.index),
        asset_col=asset_series,
    )

    # 【移除反转逻辑】：如果特征已经修复，Rank IC 应该是正的
    # 如果 Rank IC 是负的，说明特征或模型有问题，应该深入调查原因，而不是简单反转
    # 反转逻辑是"补丁"，用来纠正问题，而不是解决问题
    # 如果特征已经修复，就不需要反转逻辑了
    #
    # 如果确实需要保留（作为安全网），可以使用更严格的条件：
    # - 只在 Rank IC < -0.1 时才触发（而不是 -0.05）
    # - 或者完全移除，让问题暴露出来，便于调试
    #
    # 当前策略：完全移除反转逻辑，如果 Rank IC 是负的，应该检查：
    # 1. 目标变量计算是否正确
    # 2. 模型训练是否有问题
    # 3. 特征和目标变量的关系是否真的是负的
    if "future_return" in df.columns:
        valid_mask = df["pred"].notna() & df["future_return"].notna()
        if valid_mask.sum() > 10:
            from scipy.stats import spearmanr

            # 计算 Rank IC（Spearman 相关性）用于诊断
            rank_ic, _ = spearmanr(
                df.loc[valid_mask, "pred"].values,
                df.loc[valid_mask, "future_return"].values,
            )
            if not np.isnan(rank_ic) and rank_ic < -0.05:
                print(f"   ⚠️  WARNING: Negative Rank IC ({rank_ic:.4f}) detected!")
                print(f"   💡 This may indicate:")
                print(f"      - Target variable calculation issue")
                print(f"      - Model training problem")
                print(f"      - Feature-target relationship is truly negative (rare)")
                print(
                    f"   🔍 Please investigate the root cause instead of inverting predictions"
                )
            # 不再自动反转，让问题暴露出来

    # Compute confidence score
    df["confidence_score"] = confidence_score(df["pred_quantile"])

    # Generate signals using selected method
    if signal_method in ["sign", "hybrid", "optimized"]:
        try:
            from time_series_model.pipeline.training.rank_ic_utils_improved import (
                generate_trading_signals_improved,
            )

            true_returns = (
                df["future_return"] if "future_return" in df.columns else None
            )
            # 不再需要传递反转标记（因为已移除反转逻辑）
            df["signal"] = generate_trading_signals_improved(
                pd.Series(df["pred"], index=df.index),
                df["pred_quantile"],
                df["confidence_score"],
                true_returns=true_returns,
                method=signal_method,
                confidence_threshold=confidence_threshold,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                optimize_on_train=(
                    signal_method == "optimized" and true_returns is not None
                ),
                pred_inverted=False,  # 不再使用反转逻辑
            )
            print(f"   ✅ Generated signals using '{signal_method}' method")
        except Exception as e:
            print(f"   ⚠️  Improved signal method failed: {e}, falling back to quantile")
            df["signal"] = generate_trading_signals(
                df["pred_quantile"],
                df["confidence_score"],
                confidence_threshold=confidence_threshold,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
            )
    else:
        # Default: use quantile method
        df["signal"] = generate_trading_signals(
            df["pred_quantile"],
            df["confidence_score"],
            confidence_threshold=confidence_threshold,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )

    return df
