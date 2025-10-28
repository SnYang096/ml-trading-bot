"""Training and backtesting utilities for rolling training."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_gpu: bool = True,
    num_boost_round: int = 200,
    params: Dict[str, Any] = None,
) -> lgb.Booster:
    """
    Train a LightGBM model.

    Args:
        X_train: Training features
        y_train: Training labels
        use_gpu: Whether to use GPU acceleration
        num_boost_round: Number of boosting rounds
        params: Custom parameters (optional)

    Returns:
        Trained LightGBM model
    """
    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "force_col_wise": True,
    }

    if params:
        default_params.update(params)

    if use_gpu:
        default_params.update(
            {"device": "cuda", "gpu_platform_id": 0, "gpu_device_id": 0}
        )

    train_data = lgb.Dataset(X_train, label=y_train)

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=0)],  # Silent
    )

    return model


def simple_backtest(
    df: pd.DataFrame,
    predictions: np.ndarray,
    signal_threshold: float = 0.6,
    initial_capital: float = 100000.0,
    risk_per_trade: float = 0.01,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> Dict[str, Any]:
    """
    Simple backtest with stop loss and take profit.

    Args:
        df: DataFrame with OHLCV data
        predictions: Model predictions
        signal_threshold: Threshold for entry signal
        initial_capital: Starting capital
        risk_per_trade: Risk per trade as fraction of capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage

    Returns:
        Dictionary with backtest results
    """
    df = df.copy()
    df["prediction"] = predictions
    df["signal"] = (predictions > signal_threshold).astype(int)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        price = row["close"]

        # Check exit conditions
        if position is not None:
            if position["side"] == "long":
                # Stop loss
                if price <= position["stop"]:
                    pnl = (price - position["entry"]) * position["size"]
                    trades.append(
                        {
                            "entry_time": position["entry_time"],
                            "exit_time": idx,
                            "entry_price": position["entry"],
                            "exit_price": price,
                            "size": position["size"],
                            "pnl": pnl,
                            "reason": "stop_loss",
                        }
                    )
                    capital += pnl
                    position = None
                # Take profit
                elif price >= position["target"]:
                    pnl = (price - position["entry"]) * position["size"]
                    trades.append(
                        {
                            "entry_time": position["entry_time"],
                            "exit_time": idx,
                            "entry_price": position["entry"],
                            "exit_price": price,
                            "size": position["size"],
                            "pnl": pnl,
                            "reason": "take_profit",
                        }
                    )
                    capital += pnl
                    position = None

        # Check entry conditions
        if position is None and row["signal"] == 1:
            size = (capital * risk_per_trade) / (price * stop_loss_pct)
            position = {
                "side": "long",
                "entry": price,
                "entry_time": idx,
                "size": size,
                "stop": price * (1 - stop_loss_pct),
                "target": price * (1 + take_profit_pct),
            }

        # Record equity
        current_equity = capital
        if position is not None:
            unrealized_pnl = (price - position["entry"]) * position["size"]
            current_equity += unrealized_pnl
        equity_curve.append({"timestamp": idx, "equity": current_equity})

    # Close any remaining position
    if position is not None:
        last_price = df["close"].iloc[-1]
        pnl = (last_price - position["entry"]) * position["size"]
        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": df.index[-1],
                "entry_price": position["entry"],
                "exit_price": last_price,
                "size": position["size"],
                "pnl": pnl,
                "reason": "end_of_period",
            }
        )
        capital += pnl

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df["pnl"].sum()
        total_return = (capital - initial_capital) / initial_capital * 100
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        profit_factor = (
            abs(wins["pnl"].sum() / losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0
            else 0
        )

        # Calculate max drawdown
        equity_df = pd.DataFrame(equity_curve)
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (
            (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
        )
        max_drawdown = equity_df["drawdown"].min()
    else:
        total_pnl = 0
        total_return = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        max_drawdown = 0

    return {
        "total_trades": len(trades),
        "total_pnl": float(total_pnl),
        "total_return": float(total_return),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "final_equity": float(capital),
        "trades": trades if len(trades) > 0 else [],
        "equity_curve": equity_curve,
    }


def print_backtest_results(results: Dict[str, Any], label: str = "Results"):
    """
    Print formatted backtest results.

    Args:
        results: Results dictionary from simple_backtest
        label: Label for the results
    """
    print(f"\n📊 {label}")
    print(f"   Trades: {results['total_trades']}")
    print(f"   Return: {results['total_return']:+.2f}%")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    print(f"   Avg Win: ${results['avg_win']:,.2f}")
    print(f"   Avg Loss: ${results['avg_loss']:,.2f}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   Final Equity: ${results['final_equity']:,.0f}")
