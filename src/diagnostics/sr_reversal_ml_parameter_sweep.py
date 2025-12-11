#!/usr/bin/env python3
from __future__ import annotations
"""
SR Reversal ML Parameter Sweep: Generate parameter grid data for plateau analysis.

This script is specific to the sr_reversal strategy. It sweeps ML/ML+Volatility 
parameters (threshold + R/R combinations) to generate data for plateau heatmaps.

For other strategies, create similar sweep scripts (e.g., sr_breakout_ml_parameter_sweep.py).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader, )  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402
from src.time_series_model.strategies.labels.sr_reversal_label import (  # noqa: E402
    SRSignalConfig, _generate_sr_reversal_signals, _ensure_atr,
)
from src.time_series_model.pipeline.training.label_utils import (  # noqa: E402
    compute_rr_label, future_volatility_label,
)
from src.diagnostics.sr_reversal_model_comparison import (  # noqa: E402
    train_ml_model, train_volatility_model, evaluate_ml_model,
    evaluate_ml_volatility_model,
)


def parse_list(arg: str, cast_type) -> List:
    if not arg:
        return []
    return [cast_type(x.strip()) for x in arg.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ML/ML+Vol parameters to build plateau dataset")
    parser.add_argument("--strategy-config", required=True, type=str)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--timeframe", default="4H")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-dir",
                        type=str,
                        default="results/model_comparison")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.35,0.45,0.55,0.65",
        help="Comma-separated ML probability thresholds",
    )
    parser.add_argument(
        "--stop-loss-values",
        type=str,
        default="1.0,1.25",
        help="Comma-separated stop_loss_r values",
    )
    parser.add_argument(
        "--take-profit-values",
        type=str,
        default="2.0,2.5,3.0",
        help="Comma-separated take_profit_r values",
    )
    parser.add_argument(
        "--max-holding-values",
        type=str,
        default="48,60,72",
        help="Comma-separated max_holding_bars values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    thresholds = parse_list(args.thresholds, float)
    stop_losses = parse_list(args.stop_loss_values, float)
    take_profits = parse_list(args.take_profit_values, float)
    max_holdings = parse_list(args.max_holding_values, int)
    if not (thresholds and stop_losses and take_profits and max_holdings):
        raise ValueError("All parameter lists must be non-empty")

    cfg_dir = Path(args.strategy_config).resolve()
    cfg_loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = cfg_loader.load()

    print("📊 Loading data...")
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    print(f"   Loaded {len(df_raw)} bars")

    print("🔧 Loading features...")
    feature_loader = StrategyFeatureLoader()
    df_features = strategy_runner.run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )

    atr_series = _ensure_atr(df_features, "atr", "close", "high", "low", 14)
    split_idx = int(len(df_features) * (1 - args.test_size))
    df_train = df_features.iloc[:split_idx].copy()
    df_test = df_features.iloc[split_idx:].copy()
    atr_train = atr_series.iloc[:split_idx]
    atr_test = atr_series.iloc[split_idx:]

    # Base rule params (same defaults as comparison script)
    base_params = {
        "sr_strength_min": 0.3,
        "sqs_min": 0.7,
        "touch_distance_atr": 1.5,
        "stop_loss_r": 1.25,
        "take_profit_r": 3.0,
        "max_holding_bars": 72,
        "use_vpin_filter": False,
    }

    # Generate signals & labels for training
    sqs_min = base_params["sqs_min"]
    sr_cfg = SRSignalConfig(
        min_sr_strength=base_params["sr_strength_min"],
        min_support_score=sqs_min,
        min_resistance_score=sqs_min,
        tolerance_mult=base_params["touch_distance_atr"],
        use_vpin_filter=base_params["use_vpin_filter"],
    )
    train_signals = _generate_sr_reversal_signals(
        df_train,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_train,
        cfg=sr_cfg,
    )
    df_train["signal"] = train_signals
    train_labels = compute_rr_label(
        df_train.copy(),
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=base_params["max_holding_bars"],
        stop_loss_r=base_params["stop_loss_r"],
        take_profit_r=base_params["take_profit_r"],
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
        use_breakeven_stop=False,
    )
    if "future_volatility" not in df_features.columns:
        df_features["future_volatility"] = future_volatility_label(
            df_features["close"], horizon=10)
    train_vol_labels = df_features.loc[
        df_train.index,
        "future_volatility"].fillna(df_features["future_volatility"].median())

    # Prepare features
    feature_cols = [
        col for col in df_train.columns if col not in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signal",
            "label",
            "atr",
            "_symbol",
            "symbol",
            "timestamp",
            "datetime",
            "date",
        ]
    ]
    numeric_cols = (df_train[feature_cols].select_dtypes(
        include=[np.number]).columns.tolist())
    X_train = df_train[numeric_cols].fillna(0)
    X_train_valid = X_train[(train_signals != 0) & train_labels.notna()]
    y_train_valid = train_labels[(train_signals != 0)
                                 & train_labels.notna()].astype(int)
    y_vol_train_valid = train_vol_labels[(train_signals != 0)
                                         & train_labels.notna()]

    # Train models once
    print("🤖 Training ML model for sweep...")
    ml_model, _ = train_ml_model(
        X_train_valid,
        y_train_valid,
        X_train_valid,
        y_train_valid,
    )
    print("📈 Training volatility model for sweep...")
    vol_model, _ = train_volatility_model(
        X_train_valid,
        y_vol_train_valid,
        X_train_valid,
        y_vol_train_valid,
    )

    results: List[Dict[str, Any]] = []

    df_test_base = df_test.copy()
    atr_test_base = atr_test.copy()

    for threshold in thresholds:
        for sl in stop_losses:
            for tp in take_profits:
                for max_hold in max_holdings:
                    trial_params = base_params.copy()
                    trial_params.update({
                        "stop_loss_r": sl,
                        "take_profit_r": tp,
                        "max_holding_bars": max_hold,
                    })
                    ml_metrics = evaluate_ml_model(
                        df_test_base.copy(),
                        atr_test_base.copy(),
                        ml_model,
                        trial_params,
                        threshold=threshold,
                    )
                    ml_metrics.update({
                        "method": "ml_model",
                        "threshold": threshold,
                        "stop_loss_r": sl,
                        "take_profit_r": tp,
                        "max_holding_bars": max_hold,
                    })
                    results.append(ml_metrics)

                    ml_vol_metrics = evaluate_ml_volatility_model(
                        df_test_base.copy(),
                        atr_test_base.copy(),
                        ml_model,
                        vol_model,
                        trial_params,
                        threshold=threshold,
                    )
                    ml_vol_metrics.update({
                        "method": "ml_volatility",
                        "threshold": threshold,
                        "stop_loss_r": sl,
                        "take_profit_r": tp,
                        "max_holding_bars": max_hold,
                    })
                    results.append(ml_vol_metrics)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "ml_param_sweep.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ Saved ML parameter sweep results to {out_csv}")


if __name__ == "__main__":
    main()
