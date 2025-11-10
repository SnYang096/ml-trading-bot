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
    """Train a LightGBM model with optional validation support.
    
    Automatically detects whether to use binary, multiclass, or regression based on y_train:
    - If unique values <= 2: binary classification
    - If unique values > 2 and all integers: multiclass classification
    - Otherwise: regression
    """

    # Auto-detect task type based on labels
    # IMPORTANT: Classification tasks use 3-class (0=Hold, 1=Long, 2=Short)
    # Regression tasks (predicting returns) remain as regression - DO NOT CHANGE
    unique_labels = np.unique(y_train)
    num_unique = len(unique_labels)

    # Determine objective based on label characteristics
    # If labels are integers in [0, 2] → 3-class classification (signal prediction)
    # If labels are continuous values → regression (return prediction)
    if num_unique <= 3 and np.all(np.equal(
            np.mod(unique_labels, 1),
            0)) and np.all(unique_labels >= 0) and np.all(unique_labels <= 2):
        # 3-class classification for signal prediction (0=Hold, 1=Long, 2=Short)
        objective = "multiclass"
        metric = "multi_logloss"
        task_params = {"num_class": 3}
    elif num_unique == 2:
        # Binary classification (fallback for compatibility)
        objective = "binary"
        metric = "binary_logloss"
        task_params = {}
    else:
        # Regression for predicting continuous returns (DO NOT CHANGE - this is correct for return prediction)
        objective = "regression"
        metric = "rmse"
        task_params = {}

    default_params = {
        "objective": objective,
        "metric": metric,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "force_col_wise": True,
        **task_params,  # Add num_class for multiclass
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


def evaluate_signal_performance(
    signals_df: pd.DataFrame,
    future_returns: pd.Series,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """Evaluate risk-adjusted signals by simulating cumulative equity."""

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
    "evaluate_signal_performance",
    "print_backtest_results",
]
