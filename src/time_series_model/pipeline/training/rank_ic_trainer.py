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

    # 1. Compute future return
    # ⚠️  FIXED: Avoid using current bar's close (which may not be finalized in real-time)
    # Strategy: Use close[t+1] as entry price (assume trade at t+1 open)
    # future_return[t] = (close[t+1+horizon] - close[t+1]) / close[t+1]
    if group_cols:

        def _future_return(group: pd.DataFrame) -> pd.Series:
            close_next = group[price_col].shift(-1)
            return close_next.pct_change(hold_period).shift(-hold_period)

        future_ret = df.groupby(group_cols, group_keys=False).apply(_future_return)
        # Ensure it's a Series, not DataFrame
        if isinstance(future_ret, pd.DataFrame):
            future_ret = future_ret.iloc[:, 0]
        df["future_return"] = future_ret
    else:
        # Shift price forward by 1 to use next bar's close as entry
        price_shifted = df[price_col].shift(-1)
        df["future_return"] = price_shifted.pct_change(hold_period).shift(-hold_period)

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
    df["volatility_normalized_target"] = volatility_normalized_target(
        df["future_return"], df["rolling_vol"]
    )

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
    df = df.dropna(
        subset=feature_cols + [target_col, "future_return", "return_quantile"]
    ).copy()

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

    # Determine if we need anti-overfitting measures
    # Get hold_period from parameter or default
    if hold_period is None:
        hold_period = 5  # Default

    is_small_sample = n_samples < 3000
    is_long_horizon = hold_period >= 20
    is_high_dim = samples_per_feature < 10  # Less than 10 samples per feature

    # Feature selection for high-dimensional cases
    if is_high_dim and n_features > 50:
        print(
            f"   ⚠️  High-dimensional case detected ({n_features} features, {samples_per_feature:.1f} samples/feature)"
        )
        print(
            f"      Applying feature selection to reduce to top {min(50, int(n_samples / 20))} features"
        )
        # Select top features based on correlation with target
        feature_importance_scores = []
        for col in feature_cols:
            try:
                corr = abs(df[col].corr(df[target_col]))
                if not np.isnan(corr):
                    feature_importance_scores.append((col, corr))
            except:
                pass

        # Sort by correlation and keep top features
        feature_importance_scores.sort(key=lambda x: x[1], reverse=True)
        max_features = min(50, int(n_samples / 20), len(feature_importance_scores))
        selected_features = [f[0] for f in feature_importance_scores[:max_features]]
        feature_cols = selected_features
        n_features = len(feature_cols)
        samples_per_feature = n_samples / max(n_features, 1)
        print(
            f"      Selected {n_features} features (samples/feature: {samples_per_feature:.1f})"
        )

    X = df[feature_cols].values
    y = df[target_col].values
    y_true_return = df["future_return"].values
    tradable = (
        df[tradable_col].values
        if tradable_col in df.columns
        else np.ones(len(df), dtype=bool)
    )
    weights = df[weight_col].values if weight_col and weight_col in df.columns else None

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
            "learning_rate": 0.02,  # Lower learning rate
            "feature_fraction": 0.7,  # More aggressive feature sampling
            "bagging_fraction": 0.7,  # More aggressive bagging
            "bagging_freq": 3,
            "min_data_in_leaf": max(20, int(n_samples / 50)),  # More samples per leaf
            "min_gain_to_split": 0.1,  # Higher threshold to split
            "lambda_l1": 0.1,  # L1 regularization
            "lambda_l2": 0.1,  # L2 regularization
            "max_depth": 5,  # Limit tree depth
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
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val,
            label=y[val_idx],
            reference=train_data,
            free_raw_data=False,
        )

        # Adaptive early stopping based on sample size
        # For small samples, use more aggressive early stopping
        if is_small_sample or is_long_horizon or is_high_dim:
            stopping_rounds = max(
                20, min(50, int(len(X_train) / 20))
            )  # Adaptive stopping
            num_boost_round = max(
                100, min(300, int(len(X_train) / 5))
            )  # Limit iterations
        else:
            stopping_rounds = 50
            num_boost_round = 500

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
        valid_mask = ~np.isnan(quantile_val)

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
) -> pd.DataFrame:
    """
    Generate trading signals using ensemble of models.

    This function:
    1. Computes ensemble predictions (average of all models)
    2. Computes prediction quantile
    3. Computes confidence score
    4. Generates signals based on quantile and confidence

    Args:
        df: DataFrame with features
        models: List of trained models
        feature_cols: List of feature column names
        confidence_threshold: Minimum confidence to trade
        long_threshold: Quantile threshold for Long
        short_threshold: Quantile threshold for Short
        asset_col: Optional asset identifier for multi-asset

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

    # Compute prediction quantile
    asset_series = df[asset_col] if asset_col and asset_col in df.columns else None
    df["pred_quantile"] = prediction_quantile(
        pd.Series(df["pred"], index=df.index),
        asset_col=asset_series,
    )

    # Compute confidence score
    df["confidence_score"] = confidence_score(df["pred_quantile"])

    # Generate signals
    df["signal"] = generate_trading_signals(
        df["pred_quantile"],
        df["confidence_score"],
        confidence_threshold=confidence_threshold,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    )

    return df
