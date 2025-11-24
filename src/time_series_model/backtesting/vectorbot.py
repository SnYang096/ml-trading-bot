"""VectorBot backtest with stop loss, take profit, and position scaling."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Only rolling format (TradingModelPipeline) is supported

# ============================================================================
# Unified Backtest Functions
# ============================================================================
# All backtest-related functions are centralized here for consistency


def evaluate_signal_performance(
    signals_df: pd.DataFrame,
    future_returns: pd.Series,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """
    Evaluate risk-adjusted signals by simulating cumulative equity.

    This is a simplified backtest that uses signal_strength directly as position size.
    For more advanced backtesting with risk management, use VectorBotBacktest class.

    Args:
        signals_df: DataFrame with 'signal_strength' column
        future_returns: Series of future returns (aligned with signals_df index)
        initial_capital: Starting capital

    Returns:
        Dictionary with backtest metrics
    """
    df = signals_df.join(future_returns.rename("future_return"), how="inner").dropna(
        subset=["future_return"]
    )

    if df.empty:
        return {
            "total_trades": 0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "final_equity": initial_capital,
            "equity_curve": [],
        }

    df = df.copy()
    df["position"] = np.clip(df["signal_strength"], -1.0, 1.0)
    df["period_return"] = df["position"] * df["future_return"]

    equity_curve = (1.0 + df["period_return"]).cumprod() * initial_capital
    total_return = float(equity_curve.iloc[-1] / initial_capital - 1.0)

    trade_mask = df["position"].abs() > 1e-8
    total_trades = int(trade_mask.sum())

    if total_trades > 0:
        win_rate = float((df.loc[trade_mask, "period_return"] > 0).mean() * 100.0)
    else:
        win_rate = 0.0

    positives = df.loc[df["period_return"] > 0, "period_return"]
    negatives = df.loc[df["period_return"] < 0, "period_return"]

    if not negatives.empty and not positives.empty:
        profit_factor = float(positives.sum() / abs(negatives.sum()))
    elif negatives.empty and not positives.empty:
        profit_factor = float("inf")
    else:
        profit_factor = 1.0

    avg_win = float(positives.mean() * 100.0) if not positives.empty else 0.0
    avg_loss = float(negatives.mean() * 100.0) if not negatives.empty else 0.0

    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = float(drawdown.min() * 100.0)

    return {
        "total_trades": total_trades,
        "total_return": total_return * 100.0,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "final_equity": float(equity_curve.iloc[-1]),
        "equity_curve": [
            {"timestamp": idx, "equity": float(val)}
            for idx, val in equity_curve.items()
        ],
    }


def print_backtest_results(results: Dict[str, Any], label: str = "Results") -> None:
    """Print backtest results in a formatted way."""
    print(f"\n📊 {label}")
    print(f"   Trades: {results['total_trades']}")
    print(f"   Return: {results['total_return']:+.2f}%")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    print(f"   Avg Win: ${results['avg_win']:,.2f}")
    print(f"   Avg Loss: ${results['avg_loss']:,.2f}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   Final Equity: ${results['final_equity']:,.0f}")


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
    if "close" not in price_data.columns:
        raise ValueError("price_data must contain 'close' column")

    if len(predictions) != len(price_data):
        raise ValueError(
            f"predictions length ({len(predictions)}) must match price_data length ({len(price_data)})"
        )

    close_prices = price_data["close"].values

    # Calculate forward returns: (price[t+horizon] / price[t] - 1)
    forward_returns = np.zeros(len(close_prices))
    for i in range(len(close_prices) - horizon):
        if close_prices[i] > 0:
            forward_returns[i] = (close_prices[i + horizon] / close_prices[i]) - 1.0

    # Calculate strategy returns based on predictions
    strategy_returns = np.zeros(len(predictions))

    # Determine if this is binary classification (0/1) or multiclass (0/1/2)
    unique_preds = np.unique(predictions)
    is_binary = len(unique_preds) <= 2 and np.all(np.isin(unique_preds, [0, 1]))

    if is_binary:
        # Binary classification: 0 = Short, 1 = Long
        long_mask = predictions == 1
        short_mask = predictions == 0
        strategy_returns[long_mask] = forward_returns[long_mask]
        strategy_returns[short_mask] = -forward_returns[short_mask]
    else:
        # Multiclass: 0 = Hold, 1 = Long, 2 = Short
        long_mask = predictions == 1
        short_mask = predictions == 2
        strategy_returns[long_mask] = forward_returns[long_mask]
        strategy_returns[short_mask] = -forward_returns[short_mask]
        # Hold positions (prediction = 0): return = 0 (already initialized to 0)

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
        annualized_return = (
            total_return * (252.0 / n_periods) if n_periods < 252 else total_return
        )
        metrics["annualized_return"] = annualized_return
    else:
        metrics["annualized_return"] = 0.0

    # 3. Sharpe ratio
    returns_std = np.std(strategy_returns)
    if returns_std > 1e-8:
        # Annualized Sharpe: (mean_return - risk_free) / std_return * sqrt(252)
        daily_rf = risk_free_rate / 252.0
        sharpe_ratio = (
            (np.mean(strategy_returns) - daily_rf) / returns_std * np.sqrt(252.0)
        )
        metrics["sharpe_ratio"] = float(sharpe_ratio)
    else:
        metrics["sharpe_ratio"] = 0.0

    # 4. Maximum drawdown
    # Convert returns to cumulative equity (starting from 1.0)
    cumulative_equity = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_equity)
    drawdown = (cumulative_equity - running_max) / running_max
    max_drawdown_pct = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    # Also store absolute drawdown for backward compatibility
    max_drawdown_abs = (
        float(np.min(cumulative_equity - running_max)) if len(drawdown) > 0 else 0.0
    )
    metrics["max_drawdown"] = max_drawdown_pct  # Store as percentage
    metrics["max_drawdown_abs"] = max_drawdown_abs  # Store absolute value
    metrics["max_drawdown_pct"] = max_drawdown_pct

    # 5. Win rate
    winning_trades = (strategy_returns > 0).sum()
    total_trades = len(
        strategy_returns[strategy_returns != 0]
    )  # Only count non-hold positions
    metrics["win_rate"] = (
        float(winning_trades / total_trades) if total_trades > 0 else 0.0
    )

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


# ============================================================================
# VectorBotBacktest Class
# ============================================================================


class VectorBotBacktest:
    """VectorBot backtest with advanced risk management."""

    def __init__(
        self,
        model_path: str,
        symbol: Optional[str] = None,
        initial_capital: float = 100000,
    ):
        """
        Initialize VectorBot backtest.

        Args:
            model_path: Path to trained model
            symbol: Trading symbol (for logging/output naming)
            initial_capital: Starting capital
        """
        self.model_path = model_path
        self.symbol = symbol or "UNKNOWN"
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital

        # Risk management parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_position_size = 0.1  # 10% of capital per position
        self.scaling_factor = 0.5  # 50% scaling for additional positions
        self.max_positions = 3  # Maximum number of positions

        # Load trained model
        self.load_model()

    def load_model(self) -> None:
        """Load the trained model.

        Only supports rolling format: directory with TradingModelPipeline pipelines (from make rolling).
        Legacy format (pickle file) is no longer supported.
        """
        print(f"Loading model from {self.model_path}...")

        # Only support directory format (rolling training format)
        if not os.path.isdir(self.model_path):
            raise ValueError(
                f"Model path must be a directory (rolling format). "
                f"Legacy format (pickle file) is no longer supported. "
                f"Got: {self.model_path}"
            )

        # Try to load from latest directory or the directory itself
        latest_dir = os.path.join(self.model_path, "latest")
        if os.path.exists(latest_dir):
            model_dir = latest_dir
            print(f"   Found 'latest' directory, using models from: {model_dir}")
        else:
            model_dir = self.model_path
            print(f"   Loading models from directory: {model_dir}")

        # Load rolling training models
        self._load_rolling_models(model_dir)

    def _load_rolling_models(self, model_dir: str) -> None:
        """Load models from rolling training directory."""
        from time_series_model.models.quant_trading_model import TradingModelPipeline

        # Load classification pipeline
        cls_path = os.path.join(model_dir, "classification_pipeline.pkl")
        if not os.path.exists(cls_path):
            raise FileNotFoundError(
                f"Classification pipeline not found: {cls_path}\n"
                f"Expected files in {model_dir}:\n"
                f"  - classification_pipeline.pkl\n"
                f"  - return_pipeline.pkl\n"
                f"  - vol_pipeline.pkl\n"
                f"  - scalers.pkl (optional)"
            )

        print(f"   Loading classification pipeline: {cls_path}")
        self.cls_pipeline = TradingModelPipeline.load(cls_path)

        print(
            f"   Loading return pipeline: {os.path.join(model_dir, 'return_pipeline.pkl')}"
        )
        self.return_pipeline = TradingModelPipeline.load(
            os.path.join(model_dir, "return_pipeline.pkl")
        )

        print(
            f"   Loading volatility pipeline: {os.path.join(model_dir, 'vol_pipeline.pkl')}"
        )
        self.vol_pipeline = TradingModelPipeline.load(
            os.path.join(model_dir, "vol_pipeline.pkl")
        )

        # Load scalers if available
        scaler_path = os.path.join(model_dir, "scalers.pkl")
        if os.path.exists(scaler_path):
            print(f"   Loading scalers: {scaler_path}")
            import joblib

            self.scalers = joblib.load(scaler_path)
        else:
            self.scalers = None

        # Load features list if available
        # Try .txt first (most common format, or symlink to .txt)
        features_txt = os.path.join(model_dir, "features.txt")
        features_pkl = os.path.join(model_dir, "features.pkl")

        # Load base feature list
        feature_cols_loaded = False

        # Check if features.pkl is actually a symlink to .txt file
        if os.path.exists(features_pkl) and os.path.islink(features_pkl):
            link_target = os.readlink(features_pkl)
            # Resolve relative path
            if not os.path.isabs(link_target):
                link_target = os.path.join(model_dir, link_target)
            # If it points to a .txt file, read it as text
            if link_target.endswith(".txt") and os.path.exists(link_target):
                try:
                    with open(link_target, "r") as f:
                        self.feature_cols = [line.strip() for line in f if line.strip()]
                    print(
                        f"   Loaded {len(self.feature_cols)} features from {link_target}"
                    )
                    feature_cols_loaded = True
                except Exception as e:
                    print(f"   ⚠️ Failed to load features from symlink: {e}")
            else:
                # Try to load as pickle
                try:
                    import joblib

                    self.feature_cols = joblib.load(features_pkl)
                    if isinstance(self.feature_cols, (list, tuple)):
                        self.feature_cols = list(self.feature_cols)
                    print(
                        f"   Loaded {len(self.feature_cols)} features from features.pkl"
                    )
                    feature_cols_loaded = True
                except Exception as e:
                    print(f"   ⚠️ Failed to load features.pkl: {e}")

        if not feature_cols_loaded and os.path.exists(features_txt):
            try:
                with open(features_txt, "r") as f:
                    self.feature_cols = [line.strip() for line in f if line.strip()]
                print(f"   Loaded {len(self.feature_cols)} features from features.txt")
                feature_cols_loaded = True
            except Exception as e:
                print(f"   ⚠️ Failed to load features.txt: {e}")

        if not feature_cols_loaded and os.path.exists(features_pkl):
            try:
                import joblib

                self.feature_cols = joblib.load(features_pkl)
                if isinstance(self.feature_cols, (list, tuple)):
                    self.feature_cols = list(self.feature_cols)
                print(f"   Loaded {len(self.feature_cols)} features from features.pkl")
                feature_cols_loaded = True
            except Exception as e:
                print(f"   ⚠️ Failed to load features.pkl: {e}")
                print(f"      Error details: {type(e).__name__}: {e}")

        # Fallback to pipeline feature_cols
        if not feature_cols_loaded:
            self.feature_cols = self.cls_pipeline.feature_cols or []
            if self.feature_cols:
                print(
                    f"   Using {len(self.feature_cols)} features from pipeline metadata"
                )

        if not self.feature_cols:
            print(
                "   ⚠️ Warning: No feature columns found, will extract from model metadata"
            )

        # Mark as rolling format
        self.is_rolling_format = True
        self.model_dir = model_dir

        print("✅ Rolling models loaded successfully")
        print(f"   Feature count: {len(self.feature_cols)}")
        print(f"   Forward bars: {self.cls_pipeline.forward_bars}")

    def calculate_position_size(
        self, signal_strength: float, current_price: float
    ) -> float:
        """
        Calculate position size based on signal strength and risk management.

        Args:
            signal_strength: Strength of the signal (0-1)
            current_price: Current price

        Returns:
            Position size in units
        """
        # Base position size
        base_size = self.capital * self.max_position_size / current_price

        # Adjust for signal strength
        adjusted_size = base_size * signal_strength

        # Check if we can add more positions
        active_positions = len([p for p in self.positions if p["status"] == "active"])
        if active_positions >= self.max_positions:
            return 0.0

        # Scale down for additional positions
        if active_positions > 0:
            adjusted_size *= self.scaling_factor**active_positions

        return adjusted_size

    def update_positions(self, current_price: float, timestamp: pd.Timestamp) -> None:
        """Update all active positions with current price."""
        # Ensure timestamp is pd.Timestamp
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        for position in self.positions:
            if position["status"] != "active":
                continue

            # Calculate current P&L
            if position["side"] == "long":
                pnl = (current_price - position["entry_price"]) * position["size"]
            else:  # short
                pnl = (position["entry_price"] - current_price) * position["size"]

            position["current_pnl"] = pnl
            position["current_price"] = current_price
            position["timestamp"] = timestamp

            # Only check stop loss/take profit if enough time has passed
            # This prevents immediate exits within the same bar
            entry_time = position["entry_time"]
            if not isinstance(entry_time, pd.Timestamp):
                entry_time = pd.Timestamp(entry_time)

            time_diff_minutes = (timestamp - entry_time).total_seconds() / 60
            min_hold_time_minutes = 5.0  # Minimum 1 bar (5 minutes for 5T data)

            # Only check stop loss/take profit after minimum hold time
            if time_diff_minutes >= min_hold_time_minutes:
                # Check stop loss
                if position["side"] == "long":
                    stop_price = position["entry_price"] * (1 - self.stop_loss_pct)
                    if current_price <= stop_price:
                        self.close_position(
                            position, current_price, timestamp, "stop_loss"
                        )
                        continue
                else:  # short
                    stop_price = position["entry_price"] * (1 + self.stop_loss_pct)
                    if current_price >= stop_price:
                        self.close_position(
                            position, current_price, timestamp, "stop_loss"
                        )
                        continue

                # Check take profit
                if position["side"] == "long":
                    take_profit_price = position["entry_price"] * (
                        1 + self.take_profit_pct
                    )
                    if current_price >= take_profit_price:
                        self.close_position(
                            position, current_price, timestamp, "take_profit"
                        )
                        continue
                else:  # short
                    take_profit_price = position["entry_price"] * (
                        1 - self.take_profit_pct
                    )
                    if current_price <= take_profit_price:
                        self.close_position(
                            position, current_price, timestamp, "take_profit"
                        )
                        continue

    def close_position(
        self, position: Dict, exit_price: float, timestamp: pd.Timestamp, reason: str
    ) -> None:
        """Close a position and record the trade."""
        # Calculate final P&L
        if position["side"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["size"]
        else:  # short
            pnl = (position["entry_price"] - exit_price) * position["size"]

        # Update capital
        self.capital += pnl

        # Record trade
        # Ensure timestamps are pd.Timestamp for duration calculation
        entry_time = position["entry_time"]
        exit_time = timestamp
        if not isinstance(entry_time, pd.Timestamp):
            entry_time = pd.Timestamp(entry_time)
        if not isinstance(exit_time, pd.Timestamp):
            exit_time = pd.Timestamp(exit_time)

        trade = {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "size": position["size"],
            "pnl": pnl,
            "return_pct": pnl / (position["entry_price"] * position["size"]) * 100,
            "reason": reason,
            "duration": (exit_time - entry_time).total_seconds() / 60,  # minutes
        }

        self.trades.append(trade)
        position["status"] = "closed"
        position["exit_price"] = exit_price
        position["exit_time"] = timestamp
        position["pnl"] = pnl
        position["reason"] = reason

        print(f"   🔄 Closed {position['side']} position: {pnl:.2f} P&L ({reason})")

    def open_position(
        self,
        side: str,
        size: float,
        price: float,
        timestamp: pd.Timestamp,
        signal_strength: float,
    ) -> None:
        """Open a new position."""
        # Ensure timestamp is pd.Timestamp
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        position = {
            "id": len(self.positions),
            "side": side,
            "size": size,
            "entry_price": price,
            "entry_time": timestamp,
            "status": "active",
            "signal_strength": signal_strength,
            "current_pnl": 0.0,
            "current_price": price,
        }

        self.positions.append(position)
        print(f"   📈 Opened {side} position: {size:.4f} units at {price:.2f}")

    def run_backtest(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Run the backtest."""
        print("🚀 Starting VectorBot Backtest")
        print("=" * 50)

        # Only support rolling format
        if not hasattr(self, "is_rolling_format") or not self.is_rolling_format:
            raise ValueError(
                "Only rolling format (TradingModelPipeline) is supported. "
                "Legacy format (MLTradingStrategy) is no longer supported."
            )

        self._run_rolling_backtest(start_date, end_date, output_dir)

    def calculate_results(self) -> None:
        """Calculate backtest results."""
        if not self.trades:
            print("❌ No trades executed")
            # Set default results
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.capital,
                "initial_capital": self.initial_capital,
            }
            return

        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t["pnl"] > 0])
        losing_trades = len([t for t in self.trades if t["pnl"] < 0])

        win_rate = winning_trades / total_trades * 100.0

        total_pnl = sum([t["pnl"] for t in self.trades])
        avg_win = (
            np.mean([t["pnl"] for t in self.trades if t["pnl"] > 0])
            if winning_trades > 0
            else 0.0
        )
        avg_loss = (
            np.mean([t["pnl"] for t in self.trades if t["pnl"] < 0])
            if losing_trades > 0
            else 0.0
        )

        profit_factor = (
            abs(avg_win * winning_trades / (avg_loss * losing_trades))
            if losing_trades > 0
            else float("inf")
        )

        # Risk metrics
        returns = [t["return_pct"] for t in self.trades]
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0.0
        )

        # Final equity
        final_equity = self.capital
        total_return = (
            (final_equity - self.initial_capital) / self.initial_capital * 100.0
        )

        self.results = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self.max_drawdown * 100.0,
            "final_equity": final_equity,
            "initial_capital": self.initial_capital,
        }

    def _run_rolling_backtest(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Run backtest using rolling training models.

        This requires:
        1. Data files to be available in data/parquet_data
        2. Feature engineering to be performed (using the same feature engineering as training)
        """
        print("🔄 Running backtest with rolling training models")
        print("=" * 50)

        # Get configuration from model
        timeframe = getattr(self.cls_pipeline, "timeframe", "5T")
        forward_bars = getattr(self.cls_pipeline, "forward_bars", 3)

        print(f"   Timeframe: {timeframe}")
        print(f"   Forward bars: {forward_bars}")
        print(f"   Feature columns: {len(self.feature_cols)}")

        # Determine data directory (try multiple locations)
        data_dirs = [
            "data/parquet_data",
            "/workspace/data/parquet_data",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(self.model_dir))),
                "data",
                "parquet_data",
            ),
        ]
        data_dir = None
        for dd in data_dirs:
            if os.path.exists(dd):
                data_dir = dd
                break

        if not data_dir:
            print("❌ Cannot find data directory. Tried:")
            for dd in data_dirs:
                print(f"   - {dd}")
            print("\n💡 Rolling training already includes backtest results!")
            print(
                f"   Check: {os.path.dirname(os.path.dirname(self.model_dir))}/monthly_results.csv"
            )
            print(
                f"   Or view HTML report: {os.path.dirname(os.path.dirname(self.model_dir))}/monthly_rolling_report.html"
            )
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        print(f"   Data directory: {data_dir}")

        # Find data files for the symbol
        from time_series_model.pipeline.training.rolling import find_all_available_files

        all_files = find_all_available_files(data_dir, self.symbol)

        if not all_files:
            print(f"❌ No data files found for {self.symbol} in {data_dir}")
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        # Filter files by date range
        if start_date:
            start_ts = pd.Timestamp(start_date)
            all_files = [f for f in all_files if f["timestamp"] >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date)
            all_files = [f for f in all_files if f["timestamp"] <= end_ts]

        if not all_files:
            print(
                f"❌ No data files in date range {start_date or 'start'} to {end_date or 'end'}"
            )
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        print(f"   Found {len(all_files)} data files")
        print(
            f"   Date range: {all_files[0]['month_str']} to {all_files[-1]['month_str']}"
        )

        # Load and process all data files
        print("\n📊 Loading and processing data...")
        from data_tools.rolling_data import load_and_process_file
        from src.features.time_series.comprehensive_features import (
            ComprehensiveFeatureEngineer,
        )

        data_parts = []
        for file_info in all_files:
            print(f"   Loading {file_info['month_str']}...")
            df = load_and_process_file(file_info["path"], freq=timeframe)
            if df is not None and len(df) > 0:
                data_parts.append(df)

        if not data_parts:
            print("❌ No data loaded")
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        # Combine all data
        data_df = pd.concat(data_parts, axis=0).sort_index()
        print(f"   Loaded {len(data_df):,} bars")

        # Filter by date range if specified
        if start_date:
            data_df = data_df[data_df.index >= start_date]
        if end_date:
            data_df = data_df[data_df.index <= end_date]

        if data_df.empty:
            print("❌ No data in specified date range")
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        print(f"   Filtered to {len(data_df):,} bars in date range")
        print(f"   Date range: {data_df.index[0]} to {data_df.index[-1]}")

        # Feature engineering
        print("\n🔧 Engineering features...")
        try:
            # Determine feature type from model metadata or use comprehensive
            feature_type = getattr(self.cls_pipeline, "feature_type", "comprehensive")
            feature_engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)

            # Use feature_cols as required_features to only generate needed features
            # This is much more efficient than generating all features and then filtering
            required_features = None
            if self.feature_cols:
                required_features = set(self.feature_cols)
                print(
                    f"   🎯 Only generating {len(required_features)} required features (from features.pkl)"
                )
                print(
                    f"      This is more efficient than generating all features and filtering"
                )

            # Prepare data for feature engineering
            # Handle timestamp column conflict: if 'timestamp' already exists as a column,
            # we need to handle it before reset_index()
            data_df_work = data_df.copy()

            # Check if 'timestamp' column already exists
            has_timestamp_col = "timestamp" in data_df_work.columns

            # If index name is 'timestamp' and column 'timestamp' exists, rename the column first
            if has_timestamp_col and data_df_work.index.name == "timestamp":
                # Rename the existing column to avoid conflict
                data_df_work = data_df_work.rename(
                    columns={"timestamp": "timestamp_orig"}
                )

            # Reset index (this will create 'timestamp' column from index if index name is 'timestamp')
            data_df_reset = data_df_work.reset_index()

            # Ensure we have a 'timestamp' column for feature engineering
            if "timestamp" not in data_df_reset.columns:
                if has_timestamp_col and "timestamp_orig" in data_df_reset.columns:
                    # Use the original timestamp column
                    data_df_reset["timestamp"] = data_df_reset["timestamp_orig"]
                    data_df_reset = data_df_reset.drop(columns=["timestamp_orig"])
                elif isinstance(data_df.index, pd.DatetimeIndex):
                    # Use index as timestamp
                    data_df_reset["timestamp"] = data_df.index
                else:
                    # Try to find a datetime column
                    datetime_cols = [
                        col
                        for col in data_df_reset.columns
                        if pd.api.types.is_datetime64_any_dtype(data_df_reset[col])
                    ]
                    if datetime_cols:
                        data_df_reset["timestamp"] = data_df_reset[datetime_cols[0]]
                    else:
                        # Fallback: use index
                        data_df_reset["timestamp"] = data_df.index

            # Ensure timestamp is datetime type
            if "timestamp" in data_df_reset.columns:
                data_df_reset["timestamp"] = pd.to_datetime(data_df_reset["timestamp"])

            engineered_df = feature_engineer.engineer_all_features(
                data_df_reset, fit=False, required_features=required_features
            )

            # Set timestamp back as index
            if "timestamp" in engineered_df.columns:
                engineered_df = engineered_df.set_index("timestamp").sort_index()
            else:
                # Fallback: use original index
                engineered_df = engineered_df.set_index(
                    data_df.index[: len(engineered_df)]
                ).sort_index()

            print(f"   Generated {len(engineered_df.columns)} features")
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            import traceback

            traceback.print_exc()
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        # Filter to available features
        available_features = [
            f for f in self.feature_cols if f in engineered_df.columns
        ]
        missing_features = set(self.feature_cols) - set(available_features)

        if missing_features:
            print(
                f"   ⚠️  {len(missing_features)} features missing from data (will use available {len(available_features)})"
            )
            if len(missing_features) <= 10:
                print(f"      Missing: {', '.join(list(missing_features)[:10])}")

        if not available_features:
            print("❌ No matching features found between model and data")
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        # Prepare features for prediction
        feature_df = engineered_df[available_features].copy()

        # Ensure we have 'close' price for backtesting
        if "close" not in engineered_df.columns and "close" in data_df.columns:
            feature_df["close"] = data_df["close"]
        elif "close" in engineered_df.columns:
            feature_df["close"] = engineered_df["close"]

        # Drop rows with NaN features
        feature_df_clean = feature_df.dropna(subset=available_features)

        if feature_df_clean.empty:
            print("❌ No valid data after cleaning")
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        print(f"   Using {len(available_features)} features")
        print(f"   Clean data points: {len(feature_df_clean):,}")

        # Get predictions
        print("\n🤖 Generating predictions...")
        try:
            X = feature_df_clean[available_features].values

            # Get classification predictions (probabilities)
            cls_pred = self.cls_pipeline.predict_proba(X)
            if cls_pred.ndim == 2:
                if cls_pred.shape[1] >= 2:
                    long_prob = (
                        cls_pred[:, 1] if cls_pred.shape[1] > 1 else cls_pred[:, 0]
                    )
                    short_prob = (
                        cls_pred[:, 2] if cls_pred.shape[1] > 2 else (1.0 - long_prob)
                    )
                else:
                    long_prob = cls_pred[:, 0]
                    short_prob = 1.0 - long_prob
            else:
                long_prob = cls_pred
                short_prob = 1.0 - long_prob

            # Get return predictions
            return_pred = self.return_pipeline.predict(X)
            if return_pred.ndim > 1:
                return_pred = return_pred.ravel()

            # Get volatility predictions
            vol_pred = self.vol_pipeline.predict(X)
            if vol_pred.ndim > 1:
                vol_pred = vol_pred.ravel()

            print(f"   Generated predictions for {len(X):,} samples")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback

            traceback.print_exc()
            # Set empty results
            self.trades = []
            self.results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.initial_capital,
                "initial_capital": self.initial_capital,
            }
            if output_dir:
                self.save_results(output_dir, start_date=start_date, end_date=end_date)
            return

        # Create signals DataFrame
        # Ensure we use DatetimeIndex for proper timestamp handling
        if isinstance(feature_df_clean.index, pd.DatetimeIndex):
            # Use index directly as DatetimeIndex
            signals = pd.DataFrame(
                {
                    "long_prob": long_prob,
                    "short_prob": short_prob,
                    "return_pred": return_pred,
                    "vol_pred": vol_pred,
                    "close": feature_df_clean["close"].values,
                },
                index=feature_df_clean.index.copy(),
            )
        else:
            # Try to convert index to DatetimeIndex
            try:
                datetime_index = pd.to_datetime(feature_df_clean.index)
                signals = pd.DataFrame(
                    {
                        "long_prob": long_prob,
                        "short_prob": short_prob,
                        "return_pred": return_pred,
                        "vol_pred": vol_pred,
                        "close": feature_df_clean["close"].values,
                    },
                    index=datetime_index,
                )
            except Exception:
                # Fallback: create timestamp column and set as index
                signals = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(feature_df_clean.index),
                        "long_prob": long_prob,
                        "short_prob": short_prob,
                        "return_pred": return_pred,
                        "vol_pred": vol_pred,
                        "close": feature_df_clean["close"].values,
                    }
                )
                signals = signals.set_index("timestamp")

        # Ensure index is DatetimeIndex
        if not isinstance(signals.index, pd.DatetimeIndex):
            try:
                signals.index = pd.to_datetime(signals.index)
            except Exception as e:
                print(
                    f"   ⚠️  Warning: Could not convert signals index to DatetimeIndex: {e}"
                )

        # Generate discrete signals
        signals["discrete_signal"] = 0
        long_mask = (signals["long_prob"] > 0.55) & (
            signals["long_prob"] > signals["short_prob"]
        )
        short_mask = (signals["short_prob"] > 0.55) & (
            signals["short_prob"] > signals["long_prob"]
        )
        signals.loc[long_mask, "discrete_signal"] = 1
        signals.loc[short_mask, "discrete_signal"] = -1

        # Calculate signal strength
        signals["signal_strength"] = np.abs(
            signals["long_prob"] - signals["short_prob"]
        )

        print(f"\n📊 Signal Statistics:")
        print(f"   Total signals: {len(signals)}")
        print(f"   Long signals: {len(signals[signals['discrete_signal'] == 1])}")
        print(f"   Short signals: {len(signals[signals['discrete_signal'] == -1])}")
        print(f"   Hold signals: {len(signals[signals['discrete_signal'] == 0])}")

        # Debug: Show probability distributions
        print(f"\n📈 Probability Statistics:")
        print(
            f"   Long prob - min: {signals['long_prob'].min():.3f}, max: {signals['long_prob'].max():.3f}, mean: {signals['long_prob'].mean():.3f}"
        )
        print(
            f"   Short prob - min: {signals['short_prob'].min():.3f}, max: {signals['short_prob'].max():.3f}, mean: {signals['short_prob'].mean():.3f}"
        )
        print(
            f"   Signal strength - min: {signals['signal_strength'].min():.3f}, max: {signals['signal_strength'].max():.3f}, mean: {signals['signal_strength'].mean():.3f}"
        )

        # Count signals that pass different thresholds
        min_signal_strength = 0.1
        signals_above_threshold = len(
            signals[
                (signals["discrete_signal"] != 0)
                & (signals["signal_strength"] > min_signal_strength)
            ]
        )
        print(
            f"   Signals with strength > {min_signal_strength}: {signals_above_threshold}"
        )

        # Run backtest
        print("\n🔄 Running backtest...")
        signals_attempted = 0
        signals_rejected = 0
        for i, (timestamp, row) in enumerate(signals.iterrows()):
            # Ensure timestamp is pd.Timestamp
            if not isinstance(timestamp, pd.Timestamp):
                # Try to convert - handle various formats
                try:
                    if isinstance(timestamp, (int, float)):
                        # Might be Unix timestamp in nanoseconds
                        if timestamp > 1e12:  # Nanoseconds
                            timestamp = pd.Timestamp(timestamp, unit="ns")
                        elif timestamp > 1e9:  # Seconds
                            timestamp = pd.Timestamp(timestamp, unit="s")
                        else:
                            timestamp = pd.Timestamp(timestamp)
                    else:
                        timestamp = pd.Timestamp(timestamp)
                except Exception as e:
                    print(
                        f"   ⚠️  Warning: Failed to convert timestamp {timestamp}: {e}"
                    )
                    # Fallback: use row index or current time
                    if "timestamp" in row:
                        timestamp = pd.to_datetime(row["timestamp"])
                    else:
                        timestamp = pd.Timestamp.now()

            current_price = row["close"]
            signal = row["discrete_signal"]
            signal_strength = row["signal_strength"]

            # Update existing positions
            self.update_positions(current_price, timestamp)

            # Check for new signals
            min_signal_strength = 0.1
            if signal != 0 and signal_strength > min_signal_strength:
                signals_attempted += 1
                active_positions = len(
                    [p for p in self.positions if p["status"] == "active"]
                )

                if active_positions < self.max_positions:
                    position_size = self.calculate_position_size(
                        signal_strength, current_price
                    )

                    if position_size > 0:
                        side = "long" if signal == 1 else "short"
                        self.open_position(
                            side,
                            position_size,
                            current_price,
                            timestamp,
                            signal_strength,
                        )
                    else:
                        signals_rejected += 1
                        if signals_rejected <= 5:  # Print first few rejections
                            print(
                                f"   ⚠️  Rejected signal at {timestamp}: position_size={position_size:.4f}, signal_strength={signal_strength:.3f}, active_positions={active_positions}"
                            )
                else:
                    signals_rejected += 1
                    if signals_rejected <= 5:
                        print(
                            f"   ⚠️  Rejected signal at {timestamp}: max positions reached ({active_positions})"
                        )
            elif signal != 0:
                signals_rejected += 1
                if signals_rejected <= 5:
                    print(
                        f"   ⚠️  Rejected signal at {timestamp}: signal_strength={signal_strength:.3f} <= {min_signal_strength}"
                    )

            # Update equity curve
            total_pnl = sum(
                [p["current_pnl"] for p in self.positions if p["status"] == "active"]
            )
            current_equity = self.capital + total_pnl
            self.equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": current_equity,
                    "capital": self.capital,
                    "open_pnl": total_pnl,
                }
            )

            # Update drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity

            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

            # Progress update
            if i % 100 == 0:
                print(
                    f"   Processed {i+1}/{len(signals)} bars, Equity: {current_equity:.2f}, "
                    f"Attempted: {signals_attempted}, Rejected: {signals_rejected}, Open positions: {len([p for p in self.positions if p['status'] == 'active'])}"
                )

        # Final summary
        print(f"\n📊 Backtest Summary:")
        print(f"   Signals attempted: {signals_attempted}")
        print(f"   Signals rejected: {signals_rejected}")
        print(
            f"   Total positions opened: {len([p for p in self.positions if p['status'] != 'active'])}"
        )

        # Close any remaining positions
        for position in self.positions:
            if position["status"] == "active":
                last_price = signals["close"].iloc[-1]
                last_timestamp = signals.index[-1]
                self.close_position(position, last_price, last_timestamp, "end_of_data")

        # Calculate final results
        self.calculate_results()

        # Save results
        if output_dir:
            self.save_results(output_dir, start_date=start_date, end_date=end_date)

        print("\n🎉 Rolling backtest completed successfully!")

    def save_results(
        self, output_dir: str | None, *, start_date: str | None, end_date: str | None
    ) -> None:
        """Save backtest results."""
        base_dir = Path(output_dir or "results/vectorbot_backtests")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        start_tag = (start_date or "start").replace("-", "")
        end_tag = (end_date or "end").replace("-", "")
        run_dir = base_dir / f"{self.symbol}_{start_tag}_{end_tag}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        trades_path = run_dir / "vectorbot_trades.csv"
        equity_path = run_dir / "vectorbot_equity_curve.csv"
        results_path = run_dir / "vectorbot_results.json"
        report_path = run_dir / "vectorbot_report.html"

        # Save trades
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(trades_path, index=False)

        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(equity_path, index=False)

        # Save results
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Generate HTML report
        self._generate_html_report(
            trades_df, equity_df, report_path, start_date, end_date
        )

        print("\n💾 Results saved:")
        print(f"   - {trades_path}")
        print(f"   - {equity_path}")
        print(f"   - {results_path}")
        print(f"   - {report_path}")

    def _generate_html_report(
        self,
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        output_path: Path,
        start_date: str | None,
        end_date: str | None,
    ) -> None:
        """Generate HTML report with visualizations."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            has_plotly = True
        except ImportError:
            has_plotly = False

        # Convert timestamps if needed
        if not trades_df.empty and "entry_time" in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

        if not equity_df.empty and "timestamp" in equity_df.columns:
            equity_df = equity_df.copy()
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])

        # Prepare data for charts
        if has_plotly and not equity_df.empty:
            # Create interactive charts with Plotly
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Equity Curve",
                    "Drawdown",
                    "P&L Distribution",
                    "Monthly Returns",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # Equity curve
            fig.add_trace(
                go.Scatter(
                    x=equity_df["timestamp"],
                    y=equity_df["equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(color="#3498db", width=2),
                ),
                row=1,
                col=1,
            )
            fig.add_hline(
                y=self.initial_capital,
                line_dash="dash",
                line_color="red",
                annotation_text="Initial Capital",
                row=1,
                col=1,
            )

            # Drawdown
            peak = equity_df["equity"].expanding().max()
            drawdown = (equity_df["equity"] - peak) / peak * 100
            fig.add_trace(
                go.Scatter(
                    x=equity_df["timestamp"],
                    y=drawdown,
                    mode="lines",
                    fill="tozeroy",
                    name="Drawdown",
                    line=dict(color="#e74c3c", width=1),
                    fillcolor="rgba(231, 76, 60, 0.3)",
                ),
                row=1,
                col=2,
            )

            # P&L Distribution
            if not trades_df.empty and "pnl" in trades_df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=trades_df["pnl"],
                        nbinsx=20,
                        name="P&L Distribution",
                        marker_color="#27ae60",
                    ),
                    row=2,
                    col=1,
                )

                # Monthly returns
                trades_df["month"] = (
                    trades_df["entry_time"].dt.to_period("M").astype(str)
                )
                monthly_pnl = trades_df.groupby("month")["pnl"].sum().reset_index()
                fig.add_trace(
                    go.Bar(
                        x=monthly_pnl["month"],
                        y=monthly_pnl["pnl"],
                        name="Monthly P&L",
                        marker_color=[
                            "#27ae60" if x > 0 else "#e74c3c"
                            for x in monthly_pnl["pnl"]
                        ],
                    ),
                    row=2,
                    col=2,
                )

            fig.update_layout(
                height=800,
                showlegend=False,
                title_text=f"VectorBot Backtest Report - {self.symbol}",
                title_x=0.5,
            )
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_xaxes(title_text="P&L ($)", row=2, col=1)
            fig.update_xaxes(title_text="Month", row=2, col=2)
            fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_yaxes(title_text="P&L ($)", row=2, col=2)

            charts_html = fig.to_html(include_plotlyjs="cdn", div_id="backtest-charts")
        else:
            # Fallback: simple HTML without charts
            charts_html = (
                "<p>📊 Charts require plotly. Install with: pip install plotly</p>"
            )

        # Prepare summary metrics
        results = self.results
        metrics_html = f"""
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{results.get('total_trades', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {'positive' if results.get('win_rate', 0) > 50 else 'negative'}">
                    {results.get('win_rate', 0):.2f}%
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if results.get('total_return', 0) > 0 else 'negative'}">
                    {results.get('total_return', 0):.2f}%
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value {'positive' if results.get('total_pnl', 0) > 0 else 'negative'}">
                    ${results.get('total_pnl', 0):.2f}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{results.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{results.get('profit_factor', 0):.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{results.get('max_drawdown', 0):.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Final Equity</div>
                <div class="metric-value">${results.get('final_equity', 0):.2f}</div>
            </div>
        </div>
        """

        # Prepare trades table
        if not trades_df.empty:
            trades_table = trades_df.to_html(
                classes="trades-table",
                table_id="trades-table",
                index=False,
                escape=False,
            )
        else:
            trades_table = "<p>No trades executed during this backtest period.</p>"

        # Generate full HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VectorBot Backtest Report - {self.symbol}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .info {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        .info p {{
            margin: 5px 0;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value.positive {{
            color: #27ae60;
        }}
        .metric-value.negative {{
            color: #e74c3c;
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .trades-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .trades-table td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .trades-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 VectorBot Backtest Report</h1>
        
        <div class="info">
            <p><strong>Symbol:</strong> {self.symbol}</p>
            <p><strong>Start Date:</strong> {start_date or 'N/A'}</p>
            <p><strong>End Date:</strong> {end_date or 'N/A'}</p>
            <p><strong>Initial Capital:</strong> ${self.initial_capital:,.2f}</p>
        </div>

        <h2>📊 Performance Metrics</h2>
        {metrics_html}

        <h2>📈 Charts</h2>
        {charts_html}

        <h2>📋 Trade Details</h2>
        {trades_table}

        <div class="timestamp">
            Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
</body>
</html>
        """

        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


# Export all backtest functions for backward compatibility
__all__ = [
    "VectorBotBacktest",
    "evaluate_signal_performance",
    "print_backtest_results",
    "calculate_strategy_returns_from_predictions",
    "calculate_financial_metrics_from_returns",
    "backtest_classification_model",
]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="VectorBot backtest runner")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_PATH"),
        help="Path to trained model .pkl",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=os.environ.get("SYMBOL", "BTCUSDT"),
        help="Symbol (for logging)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=os.environ.get("START_DATE"),
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=os.environ.get("END_DATE"),
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("VECTORBOT_RESULTS_DIR", "results/vectorbot_backtests"),
        help="Directory to store backtest outputs",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=float(os.environ.get("INITIAL_CAPITAL", "100000")),
        help="Initial capital for backtest",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    """Main function to run VectorBot backtest."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model_path = args.model
    if not model_path:
        print("❌ No model path provided. Use --model or set MODEL_PATH env.")
        return

    # Handle Docker path mapping: if path starts with /home/yin/trading/ml_trading_bot, map to /workspace
    if model_path.startswith("/home/yin/trading/ml_trading_bot"):
        model_path = model_path.replace(
            "/home/yin/trading/ml_trading_bot", "/workspace", 1
        )
        print(f"   📍 Mapped model path to Docker: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run `make train` first to produce the model bundle (.pkl)")
        print("   Or use `make rolling` to produce rolling training models")
        return

    print(f"Symbol: {args.symbol}")
    if args.start or args.end:
        print(f"Backtest range: {args.start or '-∞'} → {args.end or '+∞'}")

    # Initialize backtest
    backtest = VectorBotBacktest(
        model_path, symbol=args.symbol, initial_capital=args.initial_capital
    )

    # Run backtest with optional date range
    backtest.run_backtest(
        start_date=args.start, end_date=args.end, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
