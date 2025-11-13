"""Backtest evaluator for dimensionality comparison.

This module provides functions to calculate real financial metrics (Sharpe Ratio, Max Drawdown, etc.)
based on model predictions and actual price data.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_strategy_returns_from_predictions(
    predictions: np.ndarray,
    price_data: pd.DataFrame,
    horizon: int = 1,
) -> np.ndarray:
    """
    Calculate strategy returns based on classification predictions and actual price movements.
    
    Args:
        predictions: Model predictions (0=Hold, 1=Long, 2=Short)
        price_data: DataFrame with 'close' column containing price data
        horizon: Forward horizon for calculating returns (default: 1)
    
    Returns:
        Array of strategy returns
    """
    if 'close' not in price_data.columns:
        raise ValueError("price_data must contain 'close' column")
    
    if len(predictions) != len(price_data):
        raise ValueError(f"predictions length ({len(predictions)}) must match price_data length ({len(price_data)})")
    
    close_prices = price_data['close'].values
    
    # Calculate forward returns: (price[t+horizon] / price[t] - 1)
    forward_returns = np.zeros(len(close_prices))
    for i in range(len(close_prices) - horizon):
        if close_prices[i] > 0:
            forward_returns[i] = (close_prices[i + horizon] / close_prices[i]) - 1.0
    
    # Calculate strategy returns based on predictions
    strategy_returns = np.zeros(len(predictions))
    
    # Long positions (prediction = 1): use forward return
    long_mask = predictions == 1
    strategy_returns[long_mask] = forward_returns[long_mask]
    
    # Short positions (prediction = 2): use negative forward return
    short_mask = predictions == 2
    strategy_returns[short_mask] = -forward_returns[short_mask]
    
    # Hold positions (prediction = 0): return = 0
    # (already initialized to 0)
    
    return strategy_returns


def calculate_financial_metrics_from_returns(
    strategy_returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate financial metrics from strategy returns.
    
    Args:
        strategy_returns: Array of strategy returns
        risk_free_rate: Risk-free rate (annualized, default 0)
    
    Returns:
        Dictionary of financial metrics
    """
    metrics = {}
    
    if len(strategy_returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
        }
    
    # 1. Total return (cumulative)
    total_return = float(np.sum(strategy_returns))
    metrics["total_return"] = total_return
    
    # 2. Annualized return (assuming daily data)
    n_periods = len(strategy_returns)
    if n_periods > 0:
        # Simple annualization: multiply by ~252 trading days
        annualized_return = total_return * (252.0 / n_periods) if n_periods < 252 else total_return
        metrics["annualized_return"] = annualized_return
    else:
        metrics["annualized_return"] = 0.0
    
    # 3. Sharpe ratio
    returns_std = np.std(strategy_returns)
    if returns_std > 1e-8:
        # Annualized Sharpe: (mean_return - risk_free) / std_return * sqrt(252)
        daily_rf = risk_free_rate / 252.0
        sharpe_ratio = (np.mean(strategy_returns) - daily_rf) / returns_std * np.sqrt(252.0)
        metrics["sharpe_ratio"] = float(sharpe_ratio)
    else:
        metrics["sharpe_ratio"] = 0.0
    
    # 4. Maximum drawdown
    cumulative_returns = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    metrics["max_drawdown"] = max_drawdown
    metrics["max_drawdown_pct"] = max_drawdown / (1.0 + abs(running_max[-1])) if len(running_max) > 0 and running_max[-1] != 0 else 0.0
    
    # 5. Win rate
    winning_trades = (strategy_returns > 0).sum()
    total_trades = len(strategy_returns[strategy_returns != 0])  # Only count non-hold positions
    metrics["win_rate"] = float(winning_trades / total_trades) if total_trades > 0 else 0.0
    
    # 6. Volatility (annualized)
    volatility = np.std(strategy_returns) * np.sqrt(252.0)
    metrics["volatility"] = float(volatility)
    
    return metrics


def backtest_classification_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    price_data: pd.DataFrame,
    horizon: int = 1,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Run backtest for classification model and return financial metrics.
    
    Args:
        model: Trained classification model
        X_test: Test feature matrix
        y_test: Test labels (for alignment, not used in backtest)
        price_data: DataFrame with 'close' column containing price data
        horizon: Forward horizon for calculating returns (default: 1)
        risk_free_rate: Risk-free rate (annualized, default 0)
    
    Returns:
        Dictionary of financial metrics
    """
    # Get predictions
    predictions = model.predict(X_test)
    
    # Handle multiclass predictions
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        # Multiclass: convert probability array to class predictions
        predictions = np.argmax(predictions, axis=1)
    
    # Align price data with predictions
    if len(predictions) != len(price_data):
        # If lengths don't match, take the minimum
        min_len = min(len(predictions), len(price_data))
        predictions = predictions[:min_len]
        price_data = price_data.iloc[:min_len]
    
    # Calculate strategy returns
    strategy_returns = calculate_strategy_returns_from_predictions(
        predictions, price_data, horizon=horizon
    )
    
    # Calculate financial metrics
    metrics = calculate_financial_metrics_from_returns(
        strategy_returns, risk_free_rate=risk_free_rate
    )
    
    return metrics

