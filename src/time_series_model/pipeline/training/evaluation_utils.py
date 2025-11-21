"""
Evaluation utilities for Rank IC-optimized models.

This module provides functions for:
- Quantile distribution analysis
- Confidence-based statistics (win rate, Sharpe ratio, etc.)
- Performance metrics for trading signals
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd


def analyze_quantile_distribution(
    return_quantile: pd.Series,
    pred_quantile: Optional[pd.Series] = None,
    bins: int = 10,
) -> Dict:
    """
    Analyze quantile distribution for return_quantile and pred_quantile.

    This helps assess label quality and prediction distribution:
    - Uniform distribution is ideal (no bias)
    - Skewed distribution may indicate label quality issues

    Args:
        return_quantile: Historical quantile labels (0-1)
        pred_quantile: Optional prediction quantile (0-1)
        bins: Number of bins for histogram

    Returns:
        Dictionary with distribution statistics
    """
    stats = {}

    # Filter NaN values
    return_quantile_clean = return_quantile.dropna()

    if len(return_quantile_clean) == 0:
        return {"error": "No valid return_quantile data"}

    # Return quantile statistics
    stats["return_quantile"] = {
        "count": len(return_quantile_clean),
        "mean": float(return_quantile_clean.mean()),
        "std": float(return_quantile_clean.std()),
        "min": float(return_quantile_clean.min()),
        "max": float(return_quantile_clean.max()),
        "median": float(return_quantile_clean.median()),
        "q25": float(return_quantile_clean.quantile(0.25)),
        "q75": float(return_quantile_clean.quantile(0.75)),
        "skewness": float(return_quantile_clean.skew()),
        "kurtosis": float(return_quantile_clean.kurtosis()),
    }

    # Histogram distribution
    hist, bin_edges = np.histogram(return_quantile_clean, bins=bins, range=(0, 1))
    stats["return_quantile"]["histogram"] = {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "uniformity_score": float(
            1.0 - (hist.std() / hist.mean()) if hist.mean() > 0 else 0.0
        ),  # Closer to 1 = more uniform
    }

    # Prediction quantile statistics (if provided)
    if pred_quantile is not None:
        pred_quantile_clean = pred_quantile.dropna()

        if len(pred_quantile_clean) > 0:
            stats["pred_quantile"] = {
                "count": len(pred_quantile_clean),
                "mean": float(pred_quantile_clean.mean()),
                "std": float(pred_quantile_clean.std()),
                "min": float(pred_quantile_clean.min()),
                "max": float(pred_quantile_clean.max()),
                "median": float(pred_quantile_clean.median()),
                "q25": float(pred_quantile_clean.quantile(0.25)),
                "q75": float(pred_quantile_clean.quantile(0.75)),
                "skewness": float(pred_quantile_clean.skew()),
                "kurtosis": float(pred_quantile_clean.kurtosis()),
            }

            # Correlation between return_quantile and pred_quantile
            if len(return_quantile_clean) == len(pred_quantile_clean):
                common_index = return_quantile_clean.index.intersection(
                    pred_quantile_clean.index
                )
                if len(common_index) > 10:
                    corr = return_quantile_clean.loc[common_index].corr(
                        pred_quantile_clean.loc[common_index]
                    )
                    stats["correlation"] = {
                        "return_vs_pred_quantile": (
                            float(corr) if not np.isnan(corr) else 0.0
                        ),
                    }

    return stats


def compute_confidence_statistics(
    signals: pd.Series,
    true_returns: pd.Series,
    confidence_score: pd.Series,
    confidence_threshold: float = 0.85,
    risk_free_rate: float = 0.0,
    fee_rate: float = 0.001,
    price_col: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None,
    use_multi_period_returns: bool = True,  # NEW: Use future_return directly if True
    hold_period: Optional[int] = None,  # NEW: Hold period to prevent overlapping trades
) -> Dict:
    """
    Compute confidence-based trading statistics with proper equity curve construction.

    This analyzes performance of high-confidence signals:
    - Win rate: Percentage of profitable trades
    - Sharpe ratio: Risk-adjusted returns
    - Total return: Cumulative return from continuous equity curve
    - Max drawdown: Maximum peak-to-trough decline

    Key fix: Builds continuous equity curve by tracking position changes,
    not assuming all signals are held simultaneously.

    Args:
        signals: Trading signals (1=Long, -1=Short, 0=Hold)
        true_returns: True future returns (can be multi-period or single-period)
        confidence_score: Confidence score series (0-1)
        confidence_threshold: Minimum confidence to consider (default: 0.85)
        risk_free_rate: Risk-free rate for Sharpe calculation
        fee_rate: Trading fee rate (default: 0.1% per trade)
        price_col: Optional price series for computing single-period returns

    Returns:
        Dictionary with confidence-based statistics
    """
    stats = {}

    # Align indices
    common_index = signals.index.intersection(true_returns.index).intersection(
        confidence_score.index
    )
    if len(common_index) == 0:
        return {"error": "No common index between signals, returns, and confidence"}

    # Ensure index is sorted and convert to list to avoid type mismatch issues
    # Then create a clean index (preserve original type if possible)
    try:
        common_index = common_index.sort_values()
    except (TypeError, ValueError):
        # If sorting fails (mixed types), just use as-is but ensure it's a proper Index
        common_index = pd.Index(common_index)

    signals_aligned = signals.loc[common_index].sort_index()
    returns_aligned = true_returns.loc[common_index].sort_index()
    confidence_aligned = confidence_score.loc[common_index].sort_index()

    # Filter to high-confidence signals only
    high_conf_mask = confidence_aligned >= confidence_threshold
    high_conf_signals = signals_aligned.copy()
    high_conf_signals[~high_conf_mask] = 0  # Set low-confidence signals to Hold

    # Overall statistics
    stats["overall"] = {
        "total_signals": len(signals_aligned),
        "high_confidence_signals": int(high_conf_mask.sum()),
        "high_confidence_ratio": float(high_conf_mask.mean()),
        "avg_confidence": float(confidence_aligned.mean()),
        "avg_confidence_high": (
            float(confidence_aligned[high_conf_mask].mean())
            if high_conf_mask.sum() > 0
            else 0.0
        ),
    }

    # Check direction accuracy (sign consistency between predictions and true returns)
    # This is critical: IC measures ranking, but trading needs correct direction
    # IMPORTANT: Use multi-period returns (future_return) for direction accuracy to match Win Rate
    # This ensures Direction Accuracy and Win Rate use the same time window (RANK_IC_HORIZON)
    direction_stats = {}
    if predictions is not None:
        # Align predictions with common index
        pred_aligned = (
            predictions.loc[common_index].sort_index()
            if predictions is not None
            else None
        )

        if pred_aligned is not None:
            # Use multi-period returns (returns_aligned, i.e., future_return) for direction accuracy
            # This matches the time window used for Win Rate calculation
            # returns_aligned is already the multi-period return (horizon=RANK_IC_HORIZON)
            period_returns_for_direction = returns_aligned.reindex(common_index).fillna(
                0
            )

            if period_returns_for_direction is not None:
                valid_direction_mask = (
                    pred_aligned.notna()
                    & period_returns_for_direction.notna()
                    & high_conf_mask
                )
                if valid_direction_mask.sum() > 0:
                    pred_sign = np.sign(pred_aligned.loc[valid_direction_mask])
                    # Use multi-period returns (future_return) for direction accuracy to match Win Rate
                    true_sign = np.sign(
                        period_returns_for_direction.loc[valid_direction_mask]
                    )

                    # Direction accuracy: same sign = correct direction
                    direction_accuracy = (pred_sign == true_sign).mean()
                    direction_stats["direction_accuracy"] = float(direction_accuracy)
                    direction_stats["n_samples"] = int(valid_direction_mask.sum())

                    # Check if predictions and returns are positively correlated
                    # If correlation is negative, signals might need to be inverted
                    pred_values = pred_aligned.loc[valid_direction_mask].values
                    true_values = period_returns_for_direction.loc[
                        valid_direction_mask
                    ].values
                    if len(pred_values) > 10:
                        from scipy.stats import pearsonr

                        pearson_corr, _ = pearsonr(pred_values, true_values)
                        direction_stats["pearson_correlation"] = float(pearson_corr)

                        # Check sign consistency for high-confidence signals
                        long_signals = high_conf_signals == 1
                        short_signals = high_conf_signals == -1

                        if long_signals.sum() > 0:
                            long_mask = valid_direction_mask & long_signals
                            if long_mask.sum() > 0:
                                long_direction_acc = (
                                    np.sign(pred_aligned.loc[long_mask])
                                    == np.sign(
                                        period_returns_for_direction.loc[long_mask]
                                    )
                                ).mean()
                                long_avg_return = period_returns_for_direction.loc[
                                    long_mask
                                ].mean()
                                direction_stats["long_direction_accuracy"] = float(
                                    long_direction_acc
                                )
                                direction_stats["long_avg_return"] = float(
                                    long_avg_return
                                )

                        if short_signals.sum() > 0:
                            short_mask = valid_direction_mask & short_signals
                            if short_mask.sum() > 0:
                                short_direction_acc = (
                                    np.sign(pred_aligned.loc[short_mask])
                                    == np.sign(
                                        period_returns_for_direction.loc[short_mask]
                                    )
                                ).mean()
                                short_avg_return = period_returns_for_direction.loc[
                                    short_mask
                                ].mean()
                                direction_stats["short_direction_accuracy"] = float(
                                    short_direction_acc
                                )
                                direction_stats["short_avg_return"] = float(
                                    short_avg_return
                                )

    stats["direction_analysis"] = direction_stats

    # Build continuous equity curve for high-confidence signals
    if high_conf_mask.sum() > 0:
        # Create position series (-1, 0, 1) with consistent index
        position = high_conf_signals.reindex(common_index).fillna(0)

        # FIXED: Use multi-period returns (future_return) directly if available
        # Signal at time t corresponds to future_return[t] (from t+1 to t+1+hold_period)
        if use_multi_period_returns:
            # Use future_return directly: signal[t] * future_return[t]
            # No shift needed: signal at t directly corresponds to future_return[t]
            trade_returns = returns_aligned.reindex(common_index).fillna(0)

            # Ensure trade_returns is a 1D Series
            if isinstance(trade_returns, pd.DataFrame):
                trade_returns = trade_returns.iloc[:, 0]

            # FIXED: Filter overlapping trades if hold_period is provided
            # Prevent opening new positions while holding an existing position
            position_filtered = position.copy()
            if hold_period is not None and hold_period > 0:
                # Create a mask to filter overlapping trades
                position_filtered = position.copy()
                last_trade_pos = None

                # Convert to list for easier indexing
                common_index_list = list(common_index)

                for i, idx in enumerate(common_index_list):
                    current_signal = position.loc[idx] if idx in position.index else 0

                    # If we have a valid signal
                    if current_signal != 0:
                        # Check if we're still within hold_period of last trade
                        if last_trade_pos is not None:
                            distance = i - last_trade_pos

                            # If within hold_period, filter out this signal (no overlapping trades)
                            if distance < hold_period:
                                position_filtered.loc[idx] = 0
                                continue

                        # This is a valid new trade
                        last_trade_pos = i
                    else:
                        # No signal, reset if we've passed hold_period
                        if last_trade_pos is not None:
                            distance = i - last_trade_pos

                            if distance >= hold_period:
                                last_trade_pos = None

            # Calculate PnL: signal[t] * future_return[t]
            position_values = position_filtered.reindex(common_index).fillna(0).values
            trade_returns_values = trade_returns.reindex(common_index).fillna(0).values

            # Ensure trade_returns_values is 1D
            if trade_returns_values.ndim > 1:
                trade_returns_values = (
                    trade_returns_values.flatten()
                    if trade_returns_values.shape[1] == 1
                    else trade_returns_values[:, 0]
                )

            # Calculate PnL: signal[t] * future_return[t] (no shift!)
            pnl_values = position_values * trade_returns_values

            # Convert back to Series with proper index
            pnl = pd.Series(pnl_values, index=common_index)

            # Deduct trading fees when position changes (only on entry/exit, not hold)
            position_changes = position_filtered.diff().abs()
            pnl = pnl - (position_changes * fee_rate)

            # Filter out NaN values (from future_return calculation)
            pnl = pnl[pnl.notna()]

            # Update position to filtered version for statistics
            position = position_filtered
        else:
            # OLD LOGIC: Use single-period returns (for backward compatibility)
            # Compute single-period returns for equity curve construction
            if price_col is not None:
                price_aligned = price_col.loc[common_index].sort_index()
                if isinstance(price_aligned, pd.DataFrame):
                    price_aligned = price_aligned.iloc[:, 0]
                period_returns = price_aligned.pct_change().fillna(0)
                if isinstance(period_returns, pd.DataFrame):
                    period_returns = period_returns.iloc[:, 0]
                period_returns = period_returns.reindex(common_index).fillna(0)
            else:
                period_returns = returns_aligned.copy()
                if period_returns.abs().max() > 0.5:
                    period_returns = period_returns / 5.0
                period_returns = period_returns.reindex(common_index).fillna(0)

            if isinstance(period_returns, pd.DataFrame):
                period_returns = period_returns.iloc[:, 0]

            position_values = position.reindex(common_index).fillna(0).values
            period_returns_values = (
                period_returns.reindex(common_index).fillna(0).values
            )

            if period_returns_values.ndim > 1:
                period_returns_values = (
                    period_returns_values.flatten()
                    if period_returns_values.shape[1] == 1
                    else period_returns_values[:, 0]
                )

            # OLD: Calculate PnL with shift (T-1 signal determines T position)
            position_shifted = np.concatenate([[0], position_values[:-1]])
            pnl_values = position_shifted * period_returns_values
            pnl = pd.Series(pnl_values, index=common_index)
            position_changes = position.diff().abs()
            pnl = pnl - (position_changes * fee_rate)
            pnl = pnl.iloc[1:]

        if len(pnl) > 0 and pnl.notna().sum() > 0:
            if use_multi_period_returns:
                # FIXED: For multi-period returns, calculate statistics per trade
                # Each signal corresponds to one trade with future_return

                # Filter to only non-zero signals (actual trades)
                trade_mask = position != 0
                trade_pnl = pnl[trade_mask]

                if len(trade_pnl) > 0:
                    # Win rate: percentage of profitable trades
                    win_rate = (trade_pnl > 0).mean()

                    # FIXED: For multi-period returns (future_return), each trade is independent
                    # future_return is already the complete return for the holding period
                    # If trades don't overlap, use product; if they overlap, we need continuous equity curve
                    # For simplicity, assume trades are sequential and non-overlapping
                    # Total return = product of (1 + return) for all trades - 1
                    total_return = (
                        float((1 + trade_pnl).prod() - 1.0)
                        if len(trade_pnl) > 0
                        else 0.0
                    )

                    # Build equity curve for drawdown calculation
                    # Use cumprod for visualization, but total_return is calculated above
                    equity_curve = (1 + trade_pnl).cumprod()

                    # Statistics
                    avg_return = trade_pnl.mean()
                    std_return = trade_pnl.std()

                    # FIXED: Calculate Sharpe Ratio with better method
                    # For multi-period returns, each trade_pnl value is the complete return for that trade
                    # Sharpe Ratio = (mean return - risk_free_rate) / std return
                    #
                    # Note: With few trades (e.g., 25), std may be unstable
                    # Consider using time-series approach or annualized version

                    if std_return > 0:
                        sharpe_ratio = (avg_return - risk_free_rate) / std_return
                    else:
                        sharpe_ratio = 0.0

                    # Additional diagnostic: Check if Sharpe Ratio is affected by outliers
                    # For small samples, median-based Sharpe might be more robust
                    if len(trade_pnl) < 30:
                        # Small sample: use median-based Sharpe for robustness
                        median_return = trade_pnl.median()
                        # Use MAD (Median Absolute Deviation) as robust std estimate
                        mad = (trade_pnl - median_return).abs().median()
                        robust_std = (
                            mad * 1.4826
                        )  # Convert MAD to std for normal distribution
                        if robust_std > 0:
                            robust_sharpe = (
                                median_return - risk_free_rate
                            ) / robust_std
                            # Use the more conservative estimate
                            sharpe_ratio = (
                                min(sharpe_ratio, robust_sharpe)
                                if sharpe_ratio > 0
                                else robust_sharpe
                            )

                    # Drawdown calculation
                    running_max = equity_curve.expanding().max()
                    drawdown = (equity_curve - running_max) / running_max
                    max_drawdown = drawdown.min()

                    # Count trades (only count actual trades with valid PnL)
                    trade_count = len(trade_pnl)
                    long_mask = position == 1
                    short_mask = position == -1

                    # Count long/short trades (only those with valid PnL)
                    long_trade_mask = long_mask[trade_mask]
                    short_trade_mask = short_mask[trade_mask]
                    long_count = int(long_trade_mask.sum())
                    short_count = int(short_trade_mask.sum())

                    # Separate long/short statistics
                    long_pnl = trade_pnl[long_trade_mask]
                    avg_return_long = (
                        float(long_pnl.mean()) if len(long_pnl) > 0 else 0.0
                    )
                    win_rate_long = (
                        float((long_pnl > 0).mean()) if len(long_pnl) > 0 else 0.0
                    )

                    short_pnl = trade_pnl[short_trade_mask]
                    avg_return_short = (
                        float(short_pnl.mean()) if len(short_pnl) > 0 else 0.0
                    )
                    win_rate_short = (
                        float((short_pnl > 0).mean()) if len(short_pnl) > 0 else 0.0
                    )
                else:
                    # No trades
                    trade_count = 0
                    win_rate = 0.0
                    total_return = 0.0
                    avg_return = 0.0
                    std_return = 0.0
                    sharpe_ratio = 0.0
                    max_drawdown = 0.0
                    long_count = 0
                    short_count = 0
                    avg_return_long = 0.0
                    win_rate_long = 0.0
                    avg_return_short = 0.0
                    win_rate_short = 0.0
            else:
                # OLD LOGIC: Build equity curve from period returns
                equity_curve = (1 + pnl).cumprod()
                total_return = (
                    equity_curve.iloc[-1] - 1.0 if len(equity_curve) > 0 else 0.0
                )
                period_returns = pnl[pnl.notna()]
                avg_return = period_returns.mean()
                std_return = period_returns.std()
                sharpe_ratio = (
                    (avg_return - risk_free_rate) / std_return
                    if std_return > 0
                    else 0.0
                )
                running_max = equity_curve.expanding().max()
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown = drawdown.min()
                win_rate = (pnl > 0).mean()
                long_count = int((position == 1).sum())
                short_count = int((position == -1).sum())
                long_mask = position == 1
                short_mask = position == -1
                long_pnl = pnl[long_mask.shift(1).fillna(False)]
                avg_return_long = float(long_pnl.mean()) if len(long_pnl) > 0 else 0.0
                win_rate_long = (
                    float((long_pnl > 0).mean()) if len(long_pnl) > 0 else 0.0
                )
                short_pnl = pnl[short_mask.shift(1).fillna(False)]
                avg_return_short = (
                    float(short_pnl.mean()) if len(short_pnl) > 0 else 0.0
                )
                win_rate_short = (
                    float((short_pnl > 0).mean()) if len(short_pnl) > 0 else 0.0
                )

            # Determine trade count based on logic used
            if use_multi_period_returns:
                # For multi-period returns, trade_count is already calculated
                trade_count_value = trade_count
            else:
                # For single-period returns, count position changes
                trade_count_value = int(position_changes.sum())

            # Add diagnostic information for Sharpe Ratio
            sharpe_diagnostic = {}
            if use_multi_period_returns and "trade_pnl" in locals():
                sharpe_diagnostic = {
                    "n_trades": len(trade_pnl),
                    "avg_return": float(avg_return),
                    "std_return": float(std_return),
                    "return_volatility_ratio": (
                        float(std_return / abs(avg_return))
                        if avg_return != 0
                        else float("inf")
                    ),
                    "min_return": float(trade_pnl.min()),
                    "max_return": float(trade_pnl.max()),
                    "median_return": float(trade_pnl.median()),
                }
                # Note about small sample size
                if len(trade_pnl) < 30:
                    sharpe_diagnostic["note"] = (
                        "Small sample size may cause unstable Sharpe Ratio"
                    )

            stats["high_confidence_trades"] = {
                "count": trade_count_value,  # Number of trades
                "long_count": long_count,
                "short_count": short_count,
                "win_rate": float(win_rate),
                "total_return": float(total_return),
                "avg_return": float(avg_return),
                "std_return": float(std_return),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "avg_return_long": avg_return_long,
                "avg_return_short": avg_return_short,
                "win_rate_long": win_rate_long,
                "win_rate_short": win_rate_short,
                "sharpe_diagnostic": sharpe_diagnostic,  # NEW: Diagnostic info
            }
        else:
            stats["high_confidence_trades"] = {"error": "No valid PnL data"}
    else:
        stats["high_confidence_trades"] = {"error": "No high-confidence signals"}

    # Compare with all signals (not filtered by confidence)
    all_signals = signals_aligned.copy()
    all_position = all_signals.reindex(common_index).fillna(0)

    if (all_position != 0).sum() > 0:
        # Ensure period_returns is available and aligned
        if price_col is not None:
            price_aligned = price_col.loc[common_index].sort_index()
            # Ensure price_aligned is a Series (not DataFrame)
            if isinstance(price_aligned, pd.DataFrame):
                price_aligned = price_aligned.iloc[:, 0]
            all_period_returns = price_aligned.pct_change().fillna(0)
            # Ensure all_period_returns is a Series
            if isinstance(all_period_returns, pd.DataFrame):
                all_period_returns = all_period_returns.iloc[:, 0]
            all_period_returns = all_period_returns.reindex(common_index).fillna(0)
        else:
            all_period_returns = returns_aligned.copy()
            if all_period_returns.abs().max() > 0.5:
                all_period_returns = all_period_returns / 5.0
            all_period_returns = all_period_returns.reindex(common_index).fillna(0)

        # Ensure all_period_returns is a 1D Series
        if isinstance(all_period_returns, pd.DataFrame):
            all_period_returns = all_period_returns.iloc[:, 0]

        # Build equity curve for all signals using numpy arrays to avoid index type issues
        all_position_values = all_position.values
        all_period_returns_values = all_period_returns.values

        # Ensure all_period_returns_values is 1D
        if all_period_returns_values.ndim > 1:
            all_period_returns_values = (
                all_period_returns_values.flatten()
                if all_period_returns_values.shape[1] == 1
                else all_period_returns_values[:, 0]
            )
        all_position_shifted = np.concatenate([[0], all_position_values[:-1]])
        all_pnl_values = all_position_shifted * all_period_returns_values
        all_pnl = pd.Series(all_pnl_values, index=common_index)
        all_position_changes = all_position.diff().abs()
        all_pnl = all_pnl - (all_position_changes * fee_rate)
        all_pnl = all_pnl.iloc[1:]

        if len(all_pnl) > 0 and all_pnl.notna().sum() > 0:
            all_period_returns = all_pnl[all_pnl.notna()]
            all_win_rate = (all_pnl > 0).mean()
            all_avg_return = all_period_returns.mean()
            all_std_return = all_period_returns.std()

            stats["all_signals"] = {
                "count": int(all_position_changes.sum()),
                "win_rate": float(all_win_rate),
                "avg_return": float(all_avg_return),
                "sharpe_ratio": float(
                    (all_avg_return - risk_free_rate) / all_std_return
                    if all_std_return > 0
                    else 0.0
                ),
            }

    return stats


def ensure_volatility_feature(
    df: pd.DataFrame,
    price_col: str = "close",
    volatility_col: str = "rolling_vol",
    window: int = 20,
    asset_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ensure DataFrame contains volatility feature for volatility normalization.

    If volatility column doesn't exist, compute it from price data.

    Args:
        df: DataFrame with price data
        price_col: Name of price column
        volatility_col: Name of volatility column to create/check
        window: Window size for rolling volatility calculation
        asset_col: Optional asset identifier for multi-asset data

    Returns:
        DataFrame with volatility column (added if missing)
    """
    df = df.copy()

    if volatility_col in df.columns:
        # Check if column has valid data
        if df[volatility_col].notna().sum() > 0:
            print(f"   ✅ Volatility feature '{volatility_col}' already exists")
            return df
        else:
            print(
                f"   ⚠️  Volatility feature '{volatility_col}' exists but is empty, recomputing..."
            )

    if price_col not in df.columns:
        print(
            f"   ⚠️  Warning: Price column '{price_col}' not found, cannot compute volatility"
        )
        return df

    # Compute rolling volatility (historical, for use as feature)
    from time_series_model.pipeline.training.label_utils import (
        historical_rolling_volatility,
    )

    if asset_col and asset_col in df.columns:
        # Multi-asset: compute volatility per asset
        vol_list = []
        for symbol in df[asset_col].unique():
            mask = df[asset_col] == symbol
            symbol_data = df.loc[mask, price_col].sort_index()
            # Ensure symbol_data is a Series, not DataFrame
            if isinstance(symbol_data, pd.DataFrame):
                symbol_data = symbol_data.iloc[:, 0]  # Take first column if DataFrame
            returns = symbol_data.pct_change()
            # Ensure returns is a Series
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]
            vol_series = historical_rolling_volatility(
                returns, window=window, min_periods=window // 2
            )
            # Ensure vol_series is a Series
            if isinstance(vol_series, pd.DataFrame):
                vol_series = vol_series.iloc[:, 0]
            vol_list.append(vol_series)
        # Concatenate and align - ensure it's a Series
        vol_combined = pd.concat(vol_list, axis=0).sort_index()
        if isinstance(vol_combined, pd.DataFrame):
            vol_combined = vol_combined.iloc[:, 0]  # Take first column if DataFrame
        df[volatility_col] = vol_combined.reindex(df.index)
    else:
        # Single asset
        price_series = df[price_col]
        # Ensure price_series is a Series, not DataFrame
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]  # Take first column if DataFrame
        returns = price_series.pct_change()
        # Ensure returns is a Series
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        vol_result = historical_rolling_volatility(
            returns, window=window, min_periods=window // 2
        )
        # Ensure vol_result is a Series
        if isinstance(vol_result, pd.DataFrame):
            vol_result = vol_result.iloc[:, 0]
        df[volatility_col] = vol_result

    print(f"   ✅ Computed volatility feature '{volatility_col}' (window={window})")

    return df


def print_evaluation_summary(
    quantile_stats: Dict,
    confidence_stats: Dict,
) -> None:
    """
    Print formatted evaluation summary.

    Args:
        quantile_stats: Output from analyze_quantile_distribution
        confidence_stats: Output from compute_confidence_statistics
    """
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)

    # Quantile distribution
    if "return_quantile" in quantile_stats:
        rq = quantile_stats["return_quantile"]
        print(f"\n📈 Return Quantile Distribution:")
        print(f"   Mean: {rq['mean']:.3f}, Std: {rq['std']:.3f}")
        print(f"   Range: [{rq['min']:.3f}, {rq['max']:.3f}]")
        print(f"   Skewness: {rq['skewness']:.3f}, Kurtosis: {rq['kurtosis']:.3f}")
        if "histogram" in rq:
            print(
                f"   Uniformity Score: {rq['histogram']['uniformity_score']:.3f} (1.0 = perfectly uniform)"
            )

    if "pred_quantile" in quantile_stats:
        pq = quantile_stats["pred_quantile"]
        print(f"\n📈 Prediction Quantile Distribution:")
        print(f"   Mean: {pq['mean']:.3f}, Std: {pq['std']:.3f}")
        print(f"   Range: [{pq['min']:.3f}, {pq['max']:.3f}]")

    if "correlation" in quantile_stats:
        corr = quantile_stats["correlation"]["return_vs_pred_quantile"]
        print(f"\n🔗 Correlation (Return vs Pred Quantile): {corr:.3f}")

    # Confidence statistics
    if "overall" in confidence_stats:
        overall = confidence_stats["overall"]
        print(f"\n🎯 Confidence Statistics:")
        print(f"   Total Signals: {overall['total_signals']}")
        print(
            f"   High-Confidence Signals: {overall['high_confidence_signals']} ({overall['high_confidence_ratio']:.1%})"
        )
        print(f"   Avg Confidence: {overall['avg_confidence']:.3f}")
        print(f"   Avg Confidence (High): {overall['avg_confidence_high']:.3f}")

    if (
        "high_confidence_trades" in confidence_stats
        and "error" not in confidence_stats["high_confidence_trades"]
    ):
        hct = confidence_stats["high_confidence_trades"]
        print(f"\n💰 High-Confidence Trade Performance:")
        print(
            f"   Total Trades: {hct['count']} (Long: {hct['long_count']}, Short: {hct['short_count']})"
        )
        print(f"   Win Rate: {hct['win_rate']:.1%}")
        print(f"   Total Return: {hct['total_return']:.2%}")
        print(f"   Sharpe Ratio: {hct['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {hct['max_drawdown']:.2%}")
        print(
            f"   Long Win Rate: {hct['win_rate_long']:.1%}, Avg Return: {hct['avg_return_long']:.4f}"
        )
        print(
            f"   Short Win Rate: {hct['win_rate_short']:.1%}, Avg Return: {hct['avg_return_short']:.4f}"
        )

        # Print direction accuracy analysis if available
        if (
            "direction_analysis" in confidence_stats
            and confidence_stats["direction_analysis"]
        ):
            dir_stats = confidence_stats["direction_analysis"]
            print(f"\n🔍 Direction Accuracy Analysis:")
            if "direction_accuracy" in dir_stats:
                print(
                    f"   Overall Direction Accuracy: {dir_stats['direction_accuracy']:.1%} (n={dir_stats.get('n_samples', 0)})"
                )
            if "pearson_correlation" in dir_stats:
                corr = dir_stats["pearson_correlation"]
                print(f"   Pearson Correlation (pred vs true): {corr:.4f}")
                if corr < 0:
                    print(
                        f"   ⚠️  WARNING: Negative correlation! Signals may need to be inverted."
                    )
            if "long_direction_accuracy" in dir_stats:
                print(
                    f"   Long Direction Accuracy: {dir_stats['long_direction_accuracy']:.1%}, Avg Return: {dir_stats.get('long_avg_return', 0):.6f}"
                )
            if "short_direction_accuracy" in dir_stats:
                print(
                    f"   Short Direction Accuracy: {dir_stats['short_direction_accuracy']:.1%}, Avg Return: {dir_stats.get('short_avg_return', 0):.6f}"
                )

    print("\n" + "=" * 60)
