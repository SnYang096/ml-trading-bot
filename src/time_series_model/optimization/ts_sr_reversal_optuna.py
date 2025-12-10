#!/usr/bin/env python3
"""
Optuna search for SR Reversal signal configuration.

Each trial tweaks the SR_SIGNAL_* environment variables, runs a single training
pass identical to ts-strategy-feature-compare's execute_single_run, and uses
the average CV metric as the objective.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import optuna
import numpy as np

from scripts.strategy_management.strategy_feature_compare import (
    execute_single_run,
    StrategyConfigLoader,
    load_raw_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

ENV_KEYS = [
    "SR_SIGNAL_MIN_STRENGTH",
    "SR_SIGNAL_MIN_SUPPORT",
    "SR_SIGNAL_MIN_RESISTANCE",
    "SR_SIGNAL_TOLERANCE_MULT",
    "SR_SIGNAL_MIN_TOLERANCE_PCT",
    "SR_SIGNAL_REQUIRE_FIRST_TOUCH",
    "SR_SIGNAL_MAX_TOUCHES",
    "SR_SIGNAL_ZONE_PRECISION",
]


@contextmanager
def sr_signal_env(params: Dict[str, str]):
    """Temporarily override SR signal env vars for a trial."""

    backup = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        for key, value in params.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search for SR reversal signal parameters."
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
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--output-dir", default="results/sr_reversal_optuna")
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


def sample_params(trial: optuna.Trial):
    min_strength = trial.suggest_float("min_strength", 0.0, 0.4)
    min_support = trial.suggest_float("min_support", 0.0, 0.4)
    tolerance = trial.suggest_float("tolerance_mult", 0.6, 1.8)
    min_tol_pct = trial.suggest_float("min_tolerance_pct", 0.001, 0.01)
    require_first_touch = trial.suggest_categorical(
        "require_first_touch", [False, True]
    )
    max_touch_choice = trial.suggest_categorical("max_zone_touches", ["none", 3, 5, 8])
    zone_precision = trial.suggest_categorical("zone_price_precision", [2, 3])

    raw = {
        "min_strength": min_strength,
        "min_support": min_support,
        "tolerance_mult": tolerance,
        "min_tolerance_pct": min_tol_pct,
        "require_first_touch": require_first_touch,
        "max_zone_touches": None if max_touch_choice == "none" else max_touch_choice,
        "zone_price_precision": zone_precision,
    }

    env = {
        "SR_SIGNAL_MIN_STRENGTH": f"{min_strength:.6f}",
        "SR_SIGNAL_MIN_SUPPORT": f"{min_support:.6f}",
        "SR_SIGNAL_MIN_RESISTANCE": f"{min_support:.6f}",
        "SR_SIGNAL_TOLERANCE_MULT": f"{tolerance:.6f}",
        "SR_SIGNAL_MIN_TOLERANCE_PCT": f"{min_tol_pct:.6f}",
        "SR_SIGNAL_REQUIRE_FIRST_TOUCH": "1" if require_first_touch else "0",
        "SR_SIGNAL_MAX_TOUCHES": (
            "none" if max_touch_choice == "none" else str(max_touch_choice)
        ),
        "SR_SIGNAL_ZONE_PRECISION": str(zone_precision),
    }
    return raw, env


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_loader = StrategyConfigLoader(Path(args.strategy_config))
    strategy_cfg = cfg_loader.load()
    df_train_raw, df_test_raw, warmup = build_dataset(args)

    def objective(trial: optuna.Trial) -> float:
        raw_params, env_params = sample_params(trial)
        with sr_signal_env(env_params):
            result = execute_single_run(
                strategy_cfg,
                df_train_raw.copy(),
                df_test_raw.copy(),
                test_warmup_bars=warmup,
                variant_name=f"optuna_trial_{trial.number}",
            )
        if not result:
            raise optuna.TrialPruned("Insufficient samples.")
        metric = result.get("avg_cv_metric")
        if metric is None or np.isnan(metric):
            raise optuna.TrialPruned("Metric undefined.")
        trial.set_user_attr("sr_signal_env", env_params)
        return float(metric)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_env = study.best_trial.user_attrs.get("sr_signal_env", {})
    best = {
        "value": study.best_value,
        "params": study.best_trial.params,
        "sr_signal_env": best_env,
    }
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "trial_history.csv", index=False)

    print("✅ Optuna completed. Best metric:", best["value"])
    print("   SR signal params:", best["sr_signal_env"])


if __name__ == "__main__":
    main()
