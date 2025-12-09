#!/usr/bin/env python3
"""
诊断 SR Reversal 模型为什么不开单：
1. 对比规则基线和模型预测在 signal != 0 点上的分布
2. 检查模型是否"聪明"（知道不赚钱就不开）还是bug
3. 分析pred阈值是否合理
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.strategy_config import StrategyConfigLoader  # noqa: E402
from src.time_series_model.pipeline.training.label_utils import (
    compute_rr_label,
)  # noqa: E402
from src.time_series_model.strategies.labels.sr_reversal_label import (  # noqa: E402
    SRSignalConfig,
    _ensure_atr,
    _generate_sr_reversal_signals,
    compute_sr_reversal_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why SR Reversal model doesn't open trades."
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        default="config/strategies/sr_reversal",
        help="Path to SR reversal strategy config directory.",
    )
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--timeframe", type=str, default="240T")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--long-entry-threshold",
        type=float,
        default=0.6,
        help="Long entry threshold (pred >= this).",
    )
    parser.add_argument(
        "--short-entry-threshold",
        type=float,
        default=0.4,
        help="Short entry threshold (pred <= this).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.strategy_config).resolve()
    loader = StrategyConfigLoader(config_dir)
    strategy_cfg = loader.load()

    print(f"📂 Strategy config: {config_dir}")
    print(f"📈 Loading data for {args.symbol} [{args.timeframe}]")

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    print(f"   ✅ Loaded {len(df_raw)} bars.")

    # Split train/test
    split_idx = int(len(df_raw) * (1 - args.test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    df_test_raw = df_raw.iloc[split_idx:].copy()
    print(f"   📊 Train: {len(df_train_raw)}, Test: {len(df_test_raw)}")

    # Run feature pipeline
    feature_loader = StrategyFeatureLoader()
    df_train_features = strategy_runner.run_feature_pipeline(
        df_train_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )
    df_test_features = strategy_runner.run_feature_pipeline(
        df_test_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=False,
    )

    # Generate signals BEFORE filtering (same as rule baseline)
    signal_col = strategy_cfg.labels.generator.params.get("signal_col", "signal")
    print(f"\n📊 Generating SR signals (before filters)...")

    def generate_signals_on_df(df: pd.DataFrame, name: str) -> pd.Series:
        atr_series = _ensure_atr(df, "atr", "close", "high", "low", 14)
        df["atr"] = atr_series
        auto_signals = _generate_sr_reversal_signals(
            df,
            price_col="close",
            high_col="high",
            low_col="low",
            atr_series=atr_series,
            cfg=SRSignalConfig(),
        )
        n_sig = (auto_signals != 0).sum()
        print(f"   {name}: {n_sig} signals ({100*n_sig/len(df):.1f}%)")
        return auto_signals

    train_signals_before = generate_signals_on_df(
        df_train_features.copy(), "Train (before filters)"
    )
    test_signals_before = generate_signals_on_df(
        df_test_features.copy(), "Test (before filters)"
    )

    # Store signals before filtering
    df_train_features[signal_col] = train_signals_before
    df_test_features[signal_col] = test_signals_before

    # Generate labels
    label_func = strategy_runner.import_callable(
        strategy_cfg.labels.generator.module, strategy_cfg.labels.generator.function
    )
    target_col = strategy_cfg.labels.target_column
    df_train_features[target_col] = label_func(
        df_train_features.copy(), **strategy_cfg.labels.generator.params
    )
    df_test_features[target_col] = label_func(
        df_test_features.copy(), **strategy_cfg.labels.generator.params
    )

    # Apply filters
    df_train_filtered = strategy_runner.apply_filters(
        df_train_features, strategy_cfg.labels.filters
    )
    df_test_filtered = strategy_runner.apply_filters(
        df_test_features, strategy_cfg.labels.filters
    )

    df_train_filtered = strategy_runner.apply_post_label_filters(
        df_train_filtered, strategy_cfg.labels.post_label_filters, []
    )
    df_test_filtered = strategy_runner.apply_post_label_filters(
        df_test_filtered, strategy_cfg.labels.post_label_filters, []
    )

    print(f"\n📊 After filters:")
    print(
        f"   Train: {len(df_train_filtered)} (labels: {df_train_filtered[target_col].notna().sum()})"
    )
    print(
        f"   Test: {len(df_test_filtered)} (labels: {df_test_filtered[target_col].notna().sum()})"
    )

    # Check signal retention after filtering
    if signal_col in df_train_filtered.columns:
        train_signals_after = (df_train_filtered[signal_col] != 0).sum()
        test_signals_after = (df_test_filtered[signal_col] != 0).sum()
        print(
            f"   Signals after filters - Train: {train_signals_after}, Test: {test_signals_after}"
        )

    # Train model
    trainer_func = strategy_runner.import_callable(
        strategy_cfg.model.trainer.module, strategy_cfg.model.trainer.function
    )
    trainer_params = dict(strategy_cfg.model.trainer.params)
    target_col_param = trainer_params.pop("target_col", target_col)
    model_type = trainer_params.get("model_type", "xgboost")
    task_type = trainer_params.get("task_type", "regression")

    feature_cols = strategy_runner.determine_feature_columns(
        df_train_filtered, strategy_cfg.features
    )

    print(f"\n🤖 Training model with {len(feature_cols)} features...")
    models, avg_metric, cv_results, used_features = trainer_func(
        df_train_filtered,
        feature_cols=feature_cols,
        target_col=target_col_param,
        **trainer_params,
    )
    print(f"   ✅ CV metric: {avg_metric:.4f}")

    # Generate predictions on test set
    X_test = df_test_filtered[used_features].values
    y_test = df_test_filtered[target_col].values
    preds = strategy_runner.generate_predictions(
        models=models,
        model_type=model_type,
        task_type=task_type,
        X=X_test,
    )
    preds_series = pd.Series(preds, index=df_test_filtered.index)

    # Get signal column - should already be in filtered dataframes
    signal_col = strategy_cfg.labels.generator.params.get("signal_col", "signal")

    # Check signal distribution in both sets (signals were generated before filtering)
    train_signal_series = (
        df_train_filtered.get(signal_col, pd.Series(0.0, index=df_train_filtered.index))
        .fillna(0)
        .astype(float)
    )
    test_signal_series = (
        df_test_filtered.get(signal_col, pd.Series(0.0, index=df_test_filtered.index))
        .fillna(0)
        .astype(float)
    )

    train_n_signals = (train_signal_series != 0).sum()
    test_n_signals = (test_signal_series != 0).sum()

    print(f"\n📊 Signal distribution:")
    print(
        f"   Train set: {len(df_train_filtered)} bars, {train_n_signals} signals ({100*train_n_signals/len(df_train_filtered):.1f}%)"
    )
    print(
        f"   Test set: {len(df_test_filtered)} bars, {test_n_signals} signals ({100*test_n_signals/len(df_test_filtered):.1f}%)"
    )

    # Use test set if available, otherwise fall back to train set
    if test_n_signals > 0:
        signal_series = test_signal_series
        df_analysis = df_test_filtered
        preds_series_analysis = preds_series
        analysis_set_name = "test"
    elif train_n_signals > 0:
        print(f"\n   ⚠️  No signals in test set, analyzing TRAIN set instead...")
        signal_series = train_signal_series
        df_analysis = df_train_filtered
        # Generate predictions on train set for comparison
        X_train = df_train_filtered[used_features].values
        preds_train = strategy_runner.generate_predictions(
            models=models,
            model_type=model_type,
            task_type=task_type,
            X=X_train,
        )
        preds_series_analysis = pd.Series(preds_train, index=df_train_filtered.index)
        analysis_set_name = "train"
    else:
        print("   ❌ No signals in both train and test sets!")
        print(
            "   💡 This suggests SR signal generation logic may be too strict or features are missing"
        )
        return

    # Focus on signal != 0 points
    signal_mask = signal_series != 0
    n_signals = signal_mask.sum()
    print(f"\n📊 Analysis on {analysis_set_name} set:")
    print(f"   Total {analysis_set_name} bars: {len(df_analysis)}")
    print(
        f"   Bars with signal != 0: {n_signals} ({100*n_signals/len(df_analysis):.1f}%)"
    )

    # Analyze predictions on signal points
    signal_preds = preds_series_analysis[signal_mask]
    signal_labels = df_analysis.loc[signal_mask, target_col]
    signal_directions = signal_series[signal_mask]

    print(f"\n📈 Prediction distribution on signal points:")
    print(f"   Pred mean: {signal_preds.mean():.4f}")
    print(f"   Pred std: {signal_preds.std():.4f}")
    print(f"   Pred min: {signal_preds.min():.4f}")
    print(f"   Pred max: {signal_preds.max():.4f}")
    print(f"   Pred median: {signal_preds.median():.4f}")

    # Check threshold crossings
    long_mask = signal_directions > 0
    short_mask = signal_directions < 0

    long_preds = signal_preds[long_mask]
    short_preds = signal_preds[short_mask]

    long_above_threshold = (
        (long_preds >= args.long_entry_threshold).sum() if len(long_preds) > 0 else 0
    )
    short_below_threshold = (
        (short_preds <= args.short_entry_threshold).sum() if len(short_preds) > 0 else 0
    )

    print(f"\n🎯 Entry threshold analysis:")
    print(f"   Long signals: {long_mask.sum()}")
    print(
        f"     Pred >= {args.long_entry_threshold}: {long_above_threshold} ({100*long_above_threshold/max(long_mask.sum(),1):.1f}%)"
    )
    if len(long_preds) > 0:
        print(
            f"     Long pred mean: {long_preds.mean():.4f}, median: {long_preds.median():.4f}"
        )
    print(f"   Short signals: {short_mask.sum()}")
    print(
        f"     Pred <= {args.short_entry_threshold}: {short_below_threshold} ({100*short_below_threshold/max(short_mask.sum(),1):.1f}%)"
    )
    if len(short_preds) > 0:
        print(
            f"     Short pred mean: {short_preds.mean():.4f}, median: {short_preds.median():.4f}"
        )

    # Compare with labels
    valid_labels = signal_labels.notna()
    if valid_labels.sum() > 0:
        label_1_mask = signal_labels == 1.0
        label_0_mask = signal_labels == 0.0

        print(f"\n📊 Label distribution on signal points:")
        print(f"   Valid labels: {valid_labels.sum()}")
        print(
            f"   Label=1 (success): {label_1_mask.sum()} ({100*label_1_mask.sum()/valid_labels.sum():.1f}%)"
        )
        print(
            f"   Label=0 (failure): {label_0_mask.sum()} ({100*label_0_mask.sum()/valid_labels.sum():.1f}%)"
        )

        # Check if model is "smart"
        if len(long_preds) > 0:
            long_label_1_preds = signal_preds[long_mask & label_1_mask]
            long_label_0_preds = signal_preds[long_mask & label_0_mask]
            if len(long_label_1_preds) > 0 and len(long_label_0_preds) > 0:
                print(f"\n🧠 Model 'intelligence' check (Long):")
                print(
                    f"   Pred on label=1: mean={long_label_1_preds.mean():.4f}, median={long_label_1_preds.median():.4f}"
                )
                print(
                    f"   Pred on label=0: mean={long_label_0_preds.mean():.4f}, median={long_label_0_preds.median():.4f}"
                )
                print(
                    f"   Difference: {long_label_1_preds.mean() - long_label_0_preds.mean():.4f}"
                )

        if len(short_preds) > 0:
            short_label_1_preds = signal_preds[short_mask & label_1_mask]
            short_label_0_preds = signal_preds[short_mask & label_0_mask]
            if len(short_label_1_preds) > 0 and len(short_label_0_preds) > 0:
                print(f"\n🧠 Model 'intelligence' check (Short):")
                print(
                    f"   Pred on label=1: mean={short_label_1_preds.mean():.4f}, median={short_label_1_preds.median():.4f}"
                )
                print(
                    f"   Pred on label=0: mean={short_label_0_preds.mean():.4f}, median={short_label_0_preds.median():.4f}"
                )
                print(
                    f"   Difference: {short_label_1_preds.mean() - short_label_0_preds.mean():.4f}"
                )

    # Summary
    total_entries = long_above_threshold + short_below_threshold
    print(f"\n✅ Summary:")
    print(f"   Total signals in test: {n_signals}")
    print(
        f"   Would enter (with current thresholds): {total_entries} ({100*total_entries/n_signals:.1f}%)"
    )
    print(f"   Rule baseline would enter: {n_signals} (100%)")
    print(f"\n💡 Diagnosis:")
    if total_entries == 0:
        print(
            f"   ❌ Model is TOO CONSERVATIVE: 0 entries with thresholds {args.long_entry_threshold}/{args.short_entry_threshold}"
        )
        print(
            f"   💡 Suggestion: Lower thresholds or check if pred distribution is shifted"
        )
    elif total_entries < n_signals * 0.1:
        print(
            f"   ⚠️  Model is VERY CONSERVATIVE: only {100*total_entries/n_signals:.1f}% of signals pass threshold"
        )
        print(
            f"   💡 Suggestion: Consider lowering thresholds or using rank-based selection"
        )
    else:
        print(
            f"   ✅ Model filters {100*(n_signals-total_entries)/n_signals:.1f}% of signals"
        )
        if valid_labels.sum() > 0:
            baseline_win_rate = label_1_mask.sum() / valid_labels.sum()
            print(f"   📊 Rule baseline win rate: {100*baseline_win_rate:.1f}%")
            print(
                f"   💡 If model is 'smart', filtered signals should have higher win rate"
            )


if __name__ == "__main__":
    main()
