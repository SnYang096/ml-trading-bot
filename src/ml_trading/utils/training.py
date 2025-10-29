"""Training utilities shared across dimensionality workflows."""

from __future__ import annotations

from typing import Any, Dict

import lightgbm as lgb
from lightgbm.basic import LightGBMError
import numpy as np
import pandas as pd


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_gpu: bool = True,
    num_boost_round: int = 200,
    params: Dict[str, Any] | None = None,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
    eval_period: int | None = 50,
    categorical_feature: Any | None = None,
) -> lgb.Booster:
    """Train a LightGBM model with optional validation support."""

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
        default_params.update({
            "device": "cuda",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        })

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )

    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            categorical_feature=categorical_feature,
            free_raw_data=False,
        )
        valid_sets.append(val_data)
        valid_names.append("valid")

    callbacks = []
    if eval_period is not None:
        callbacks.append(lgb.log_evaluation(period=eval_period))

    if early_stopping_rounds is not None and len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))

    try:
        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
    except LightGBMError as exc:
        if use_gpu and "CUDA" in str(exc).upper():
            print("⚠️  Falling back to CPU-based LightGBM training")
            return train_lightgbm_model(
                X_train,
                y_train,
                use_gpu=False,
                num_boost_round=num_boost_round,
                params=params,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=early_stopping_rounds,
                eval_period=eval_period,
                categorical_feature=categorical_feature,
            )
        raise

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
    df = df.copy()
    df["prediction"] = predictions
    df["signal"] = (predictions > signal_threshold).astype(int)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        price = row["close"]

        if position is not None:
            if position["side"] == "long":
                if price <= position["stop"]:
                    pnl = (price - position["entry"]) * position["size"]
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": idx,
                        "entry_price": position["entry"],
                        "exit_price": price,
                        "size": position["size"],
                        "pnl": pnl,
                        "reason": "stop_loss",
                    })
                    capital += pnl
                    position = None
                elif price >= position["target"]:
                    pnl = (price - position["entry"]) * position["size"]
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": idx,
                        "entry_price": position["entry"],
                        "exit_price": price,
                        "size": position["size"],
                        "pnl": pnl,
                        "reason": "take_profit",
                    })
                    capital += pnl
                    position = None

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

        current_equity = capital
        if position is not None:
            unrealized_pnl = (price - position["entry"]) * position["size"]
            current_equity += unrealized_pnl
        equity_curve.append({"timestamp": idx, "equity": current_equity})

    if position is not None:
        last_price = df["close"].iloc[-1]
        pnl = (last_price - position["entry"]) * position["size"]
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": df.index[-1],
            "entry_price": position["entry"],
            "exit_price": last_price,
            "size": position["size"],
            "pnl": pnl,
            "reason": "end_of_period",
        })
        capital += pnl

    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df["pnl"].sum()
        total_return = (capital - initial_capital) / initial_capital * 100
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        win_rate = len(wins) / len(trades_df) * 100 if len(
            trades_df) > 0 else 0

        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        profit_factor = (abs(wins["pnl"].sum() / losses["pnl"].sum()) if
                         len(losses) > 0 and losses["pnl"].sum() != 0 else 0)

        equity_df = pd.DataFrame(equity_curve)
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = ((equity_df["equity"] - equity_df["peak"]) /
                                 equity_df["peak"] * 100)
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
        "trades": trades if trades else [],
        "equity_curve": equity_curve,
    }


def print_backtest_results(results: Dict[str, Any],
                           label: str = "Results") -> None:
    print(f"\n📊 {label}")
    print(f"   Trades: {results['total_trades']}")
    print(f"   Return: {results['total_return']:+.2f}%")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    print(f"   Avg Win: ${results['avg_win']:,.2f}")
    print(f"   Avg Loss: ${results['avg_loss']:,.2f}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   Final Equity: ${results['final_equity']:,.0f}")


__all__ = [
    "train_lightgbm_model",
    "simple_backtest",
    "print_backtest_results",
]
