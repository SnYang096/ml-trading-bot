#!/usr/bin/env python3
"""
Optuna joint optimization for SR Reversal: Model hyperparameters + Prediction thresholds.

This script simultaneously optimizes:
1. Model hyperparameters (XGBoost: max_depth, learning_rate, n_estimators, etc.)
2. Prediction thresholds (long_entry_threshold, short_entry_threshold, etc.)

This is an end-to-end optimization that directly targets the business objective (backtest performance).

Note: This is computationally expensive as each trial requires full model training.
Use this when you need to optimize both model and thresholds together.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import optuna
import numpy as np

from src.time_series_model.strategies.evaluation.strategy_feature_compare import (
    execute_single_run,
    StrategyConfigLoader,
    load_raw_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna joint optimization for SR reversal: model hyperparameters + prediction thresholds."
    )
    parser.add_argument(
        "--strategy-config", required=True, help="Path to strategy directory."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--test-warmup-bars", type=int, default=200)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--output-dir", default="results/sr_reversal_optuna_joint")
    parser.add_argument(
        "--optimize-model-only",
        action="store_true",
        help="Only optimize model hyperparameters, not thresholds",
    )
    parser.add_argument(
        "--optimize-thresholds-only",
        action="store_true",
        help="Only optimize thresholds, not model hyperparameters",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="sharpe",
        choices=["sharpe", "total_return", "cv_metric", "sharpe_with_cv_fallback"],
        help="Optimization objective. 'sharpe' (default) uses backtest Sharpe ratio, "
        "'total_return' uses backtest total return, 'cv_metric' uses CV metric, "
        "'sharpe_with_cv_fallback' uses Sharpe if available, otherwise CV metric.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum number of trades required (prune trials with too few trades).",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Minimum win rate required (0.0-1.0). Prune trials below this threshold.",
    )
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    split_idx = int(len(df_raw) * (1 - args.test_size))
    if split_idx <= 0 or split_idx >= len(df_raw):
        raise ValueError("Invalid test_size split; no train/test separation.")
    test_warmup = min(args.test_warmup_bars, split_idx)
    df_train_raw = df_raw.iloc[:split_idx].copy()
    df_test_raw = df_raw.iloc[split_idx - test_warmup :].copy()
    return df_train_raw, df_test_raw, test_warmup


def sample_model_params(trial: optuna.Trial, model_type: str = "xgboost") -> Dict:
    """
    Sample model hyperparameters for Optuna trial.

    Supports XGBoost and LightGBM.
    """
    if model_type.lower() == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    elif model_type.lower() == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 20, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "min_sum_hessian_in_leaf": trial.suggest_float(
                "min_sum_hessian_in_leaf", 1e-3, 10.0, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def sample_threshold_params(trial: optuna.Trial) -> Dict[str, float]:
    """
    Sample prediction thresholds for Optuna trial.

    These thresholds determine when to enter/exit trades based on model predictions.
    """
    long_entry = trial.suggest_float("long_entry_threshold", 0.4, 0.8)
    long_exit = trial.suggest_float("long_exit_threshold", 0.2, 0.5)
    short_entry = trial.suggest_float("short_entry_threshold", 0.2, 0.6)
    short_exit = trial.suggest_float("short_exit_threshold", 0.5, 0.8)

    # Ensure logical constraints
    if long_entry <= long_exit:
        raise optuna.TrialPruned(
            f"Invalid: long_entry_threshold ({long_entry:.4f}) <= long_exit_threshold ({long_exit:.4f})"
        )
    if short_exit <= short_entry:
        raise optuna.TrialPruned(
            f"Invalid: short_exit_threshold ({short_exit:.4f}) <= short_entry_threshold ({short_entry:.4f})"
        )

    return {
        "long_entry_threshold": long_entry,
        "long_exit_threshold": long_exit,
        "short_entry_threshold": short_entry,
        "short_exit_threshold": short_exit,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_loader = StrategyConfigLoader(Path(args.strategy_config))
    strategy_cfg = cfg_loader.load()
    df_train_raw, df_test_raw, warmup = build_dataset(args)

    # Determine model type from config
    model_type = strategy_cfg.model.trainer.params.get("model_type", "xgboost")

    def objective(trial: optuna.Trial) -> float:
        # Create a copy of strategy config
        trial_cfg = deepcopy(strategy_cfg)

        # 1. Optimize model hyperparameters (if enabled)
        if not args.optimize_thresholds_only:
            model_params = sample_model_params(trial, model_type=model_type)
            # Update model_params in trainer params
            if trial_cfg.model.trainer.params.get("model_params") is None:
                trial_cfg.model.trainer.params["model_params"] = {}
            trial_cfg.model.trainer.params["model_params"].update(model_params)
            trial.set_user_attr("model_params", model_params)
        else:
            # Use default model params
            trial.set_user_attr("model_params", {})

        # 2. Optimize prediction thresholds (if enabled)
        if not args.optimize_model_only:
            threshold_params = sample_threshold_params(trial)
            # Update thresholds in backtest config
            if trial_cfg.backtest.params is None:
                trial_cfg.backtest.params = {}
            trial_cfg.backtest.params.update(threshold_params)
            trial.set_user_attr("threshold_params", threshold_params)
        else:
            # Use default thresholds
            trial.set_user_attr("threshold_params", {})

        # 3. Run training and evaluation with these parameters
        result = execute_single_run(
            trial_cfg,
            df_train_raw.copy(),
            df_test_raw.copy(),
            test_warmup_bars=warmup,
            variant_name=f"optuna_joint_trial_{trial.number}",
        )

        if not result:
            raise optuna.TrialPruned("Insufficient samples.")

        # Store threshold params for analysis
        if not args.optimize_model_only:
            trial.set_user_attr("threshold_params", threshold_params)
        if not args.optimize_thresholds_only:
            trial.set_user_attr("model_params", model_params)

        # Get backtest results for business metrics
        backtest_results = result.get("backtest")
        if backtest_results:
            trial.set_user_attr("backtest_results", backtest_results)

            # Get number of trades (from debug payload if available, otherwise estimate)
            n_trades = 0
            if (
                "debug" in backtest_results
                and "trades_meta" in backtest_results["debug"]
            ):
                n_trades = backtest_results["debug"]["trades_meta"].get("n_trades", 0)
            # If no debug info, we can't check trades constraint, but continue

            # Check minimum trades constraint (important for imbalanced data)
            if n_trades > 0 and n_trades < args.min_trades:
                raise optuna.TrialPruned(
                    f"Insufficient trades: {n_trades} < {args.min_trades}"
                )

            # Check minimum win rate constraint
            # win_rate is stored as percentage (0-100) in backtest results
            win_rate_pct = backtest_results.get("win_rate", 0.0)
            win_rate = win_rate_pct / 100.0 if win_rate_pct > 1.0 else win_rate_pct
            if win_rate < args.min_win_rate:
                raise optuna.TrialPruned(
                    f"Win rate too low: {win_rate:.4f} < {args.min_win_rate:.4f}"
                )

        # Select objective based on args.objective
        if args.objective == "sharpe":
            if backtest_results and "sharpe" in backtest_results:
                sharpe = backtest_results["sharpe"]
                if sharpe is not None and not np.isnan(sharpe):
                    return float(sharpe)
            # Fallback to CV metric if no backtest Sharpe
            metric = result.get("avg_cv_metric")
            if metric is not None and not np.isnan(metric):
                return float(metric)
            raise optuna.TrialPruned("No valid metric available.")

        elif args.objective == "total_return":
            if backtest_results and "total_return_pct" in backtest_results:
                total_return = backtest_results["total_return_pct"]
                if total_return is not None and not np.isnan(total_return):
                    return float(total_return)
            # Fallback to CV metric
            metric = result.get("avg_cv_metric")
            if metric is not None and not np.isnan(metric):
                return float(metric)
            raise optuna.TrialPruned("No valid metric available.")

        elif args.objective == "sharpe_with_cv_fallback":
            # Prefer Sharpe, fallback to CV metric
            if backtest_results and "sharpe" in backtest_results:
                sharpe = backtest_results["sharpe"]
                if sharpe is not None and not np.isnan(sharpe):
                    return float(sharpe)
            # Fallback to CV metric
            metric = result.get("avg_cv_metric")
            if metric is not None and not np.isnan(metric):
                return float(metric)
            raise optuna.TrialPruned("No valid metric available.")

        else:  # args.objective == "cv_metric"
            # Use CV metric (original behavior)
            metric = result.get("avg_cv_metric")
            if metric is None or np.isnan(metric):
                raise optuna.TrialPruned("CV metric undefined.")
            return float(metric)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Extract best parameters
    best_model_params = study.best_trial.user_attrs.get("model_params", {})
    best_threshold_params = study.best_trial.user_attrs.get("threshold_params", {})
    best_backtest = study.best_trial.user_attrs.get("backtest_results", {})

    best = {
        "value": study.best_value,
        "params": study.best_trial.params,
        "model_params": best_model_params,
        "threshold_params": best_threshold_params,
        "backtest_results": best_backtest,
    }

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "trial_history.csv", index=False)

    print("✅ Optuna joint optimization completed. Best metric:", best["value"])

    if best_model_params:
        print("   Best model hyperparameters:")
        for key, value in best_model_params.items():
            if isinstance(value, (int, float)):
                print(
                    f"     {key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"     {key}: {value}"
                )

    if best_threshold_params:
        print("   Best thresholds:")
        for key, value in best_threshold_params.items():
            print(f"     {key}: {value:.4f}")

    if best_backtest:
        print("   Best backtest results:")
        for key, value in best_backtest.items():
            if isinstance(value, (int, float)):
                print(
                    f"     {key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"     {key}: {value}"
                )


if __name__ == "__main__":
    main()
