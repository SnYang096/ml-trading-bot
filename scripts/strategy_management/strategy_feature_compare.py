#!/usr/bin/env python3
"""Compare strategy performance across different feature configurations."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.strategy_config import StrategyConfig, StrategyConfigLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strategy feature configurations."
    )
    parser.add_argument(
        "--strategy-config", required=True, help="Base strategy directory"
    )
    parser.add_argument(
        "--feature-overrides",
        nargs="*",
        default=[],
        help="List of variant definitions in the form name=path/to/features.yaml",
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-dir", default="results/strategy_compare")
    parser.add_argument("--run-rolling", action="store_true")
    parser.add_argument("--rolling-train-bars", type=int, default=5000)
    parser.add_argument("--rolling-test-bars", type=int, default=1000)
    parser.add_argument("--rolling-step-bars", type=int, default=1000)
    parser.add_argument("--rolling-max-windows", type=int, default=5)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def merge_features(base_path: Path, override_path: Path, variant_name: str) -> Dict:
    base_data = load_yaml(base_path)
    override_data = load_yaml(override_path)
    merged = dict(base_data)
    for key, value in override_data.items():
        if key == "feature_pipeline" and isinstance(value, dict):
            merged.setdefault("feature_pipeline", {})
            merged["feature_pipeline"].update(value)
        else:
            merged[key] = value
    merged["name"] = f"{base_data.get('name', 'strategy')}_{variant_name}"
    return merged


def update_meta(meta_path: Path, variant_name: str) -> None:
    data = load_yaml(meta_path)
    strategy_info = data.get("strategy", {})
    base_name = strategy_info.get("name", variant_name)
    strategy_info["name"] = f"{base_name}_{variant_name}"
    data["strategy"] = strategy_info
    dump_yaml(meta_path, data)


@dataclass
class VariantSpec:
    name: str
    config_dir: Path
    is_temp: bool = False


def build_variants(
    base_dir: Path, overrides: List[str]
) -> Tuple[List[VariantSpec], List[Path]]:
    variants = [VariantSpec(name="base", config_dir=base_dir, is_temp=False)]
    temp_dirs: List[Path] = []

    for entry in overrides:
        if "=" in entry:
            variant_name, override_path = entry.split("=", 1)
        else:
            variant_name = Path(entry).stem
            override_path = entry
        variant_name = variant_name.strip()
        override_path = Path(override_path).resolve()
        if not override_path.exists():
            raise FileNotFoundError(f"Override file not found: {override_path}")

        temp_dir = Path(tempfile.mkdtemp(prefix=f"strategy_variant_{variant_name}_"))
        shutil.copytree(base_dir, temp_dir, dirs_exist_ok=True)
        merged_features = merge_features(
            base_dir / "features.yaml", override_path, variant_name
        )
        dump_yaml(temp_dir / "features.yaml", merged_features)
        meta_path = temp_dir / "meta.yaml"
        if meta_path.exists():
            update_meta(meta_path, variant_name)
        variants.append(
            VariantSpec(name=variant_name, config_dir=temp_dir, is_temp=True)
        )
        temp_dirs.append(temp_dir)

    return variants, temp_dirs


def execute_single_run(
    strategy_cfg: StrategyConfig,
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
) -> Optional[Dict]:
    if df_train_raw.empty or df_test_raw.empty:
        return None

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

    feature_cols = strategy_runner.determine_feature_columns(
        df_train_features, strategy_cfg.features
    )
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

    df_train_filtered = strategy_runner.apply_filters(
        df_train_features, strategy_cfg.labels.filters
    )
    df_test_filtered = strategy_runner.apply_filters(
        df_test_features, strategy_cfg.labels.filters
    )

    df_train_filtered = strategy_runner.apply_post_label_filters(
        df_train_filtered, strategy_cfg.labels.post_label_filters, feature_cols
    )
    df_test_filtered = strategy_runner.apply_post_label_filters(
        df_test_filtered, strategy_cfg.labels.post_label_filters, feature_cols
    )

    if len(df_train_filtered) < 50 or len(df_test_filtered) < 10:
        return None

    trainer_func = strategy_runner.import_callable(
        strategy_cfg.model.trainer.module, strategy_cfg.model.trainer.function
    )
    trainer_params = dict(strategy_cfg.model.trainer.params)
    target_col = trainer_params.pop("target_col", target_col)
    model_type = trainer_params.get("model_type", "xgboost")
    task_type = trainer_params.get("task_type", "regression")

    models, avg_metric, cv_results, used_features = trainer_func(
        df_train_filtered,
        feature_cols=feature_cols,
        target_col=target_col,
        **trainer_params,
    )

    X_test = df_test_filtered[used_features].values
    y_test = df_test_filtered[target_col].values
    preds = strategy_runner.generate_predictions(
        models=models,
        model_type=model_type,
        task_type=task_type,
        X=X_test,
    )

    evaluation_results = strategy_runner.evaluate_predictions(
        preds, y_test, strategy_cfg.evaluation
    )
    backtest_results = strategy_runner.run_vectorbt_backtest(
        df_test_filtered, preds, strategy_cfg.backtest, task_type
    )

    return {
        "avg_cv_metric": float(avg_metric),
        "evaluation": evaluation_results,
        "backtest": backtest_results,
        "used_features": used_features,
        "n_train": int(len(df_train_filtered)),
        "n_test": int(len(df_test_filtered)),
    }


def run_rolling_evaluation(
    strategy_cfg: StrategyConfig,
    df_raw: pd.DataFrame,
    params: argparse.Namespace,
) -> Optional[Dict]:
    train_size = params.rolling_train_bars
    test_size = params.rolling_test_bars
    step = params.rolling_step_bars
    max_windows = params.rolling_max_windows

    windows: List[Dict] = []
    start = 0
    while start + train_size + test_size <= len(df_raw) and len(windows) < max_windows:
        train_raw = df_raw.iloc[start : start + train_size].copy()
        test_raw = df_raw.iloc[
            start + train_size : start + train_size + test_size
        ].copy()
        result = execute_single_run(strategy_cfg, train_raw, test_raw)
        if result:
            result["window_start"] = str(train_raw.index[0])
            result["window_end"] = str(test_raw.index[-1])
            windows.append(result)
        start += step

    if not windows:
        return None

    eval_keys = sorted({k for w in windows for k in w["evaluation"].keys()})
    aggregate_eval = {
        key: float(np.nanmean([w["evaluation"].get(key, np.nan) for w in windows]))
        for key in eval_keys
    }
    if any(w.get("backtest") for w in windows):
        bt_keys = sorted(
            {k for w in windows if w.get("backtest") for k in w["backtest"].keys()}
        )
        aggregate_bt = {
            key: float(
                np.nanmean(
                    [
                        w["backtest"].get(key, np.nan)
                        for w in windows
                        if w.get("backtest")
                    ]
                )
            )
            for key in bt_keys
        }
    else:
        aggregate_bt = None

    avg_cv = float(np.nanmean([w["avg_cv_metric"] for w in windows]))
    return {
        "windows": windows,
        "aggregate": {
            "avg_cv_metric": avg_cv,
            "evaluation": aggregate_eval,
            "backtest": aggregate_bt,
            "n_windows": len(windows),
        },
    }


def summarize_results(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        row = {
            "variant": item["variant"],
            "avg_cv_metric": (
                item["base"]["avg_cv_metric"] if item.get("base") else np.nan
            ),
            "n_train": item["base"].get("n_train", 0) if item.get("base") else 0,
            "n_test": item["base"].get("n_test", 0) if item.get("base") else 0,
        }
        evaluation = item["base"].get("evaluation", {}) if item.get("base") else {}
        for key, value in evaluation.items():
            row[f"eval_{key}"] = value
        backtest = item["base"].get("backtest") if item.get("base") else None
        if backtest:
            for key, value in backtest.items():
                row[f"bt_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.strategy_config).resolve()
    variants, temp_dirs = build_variants(base_dir, args.feature_overrides)

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=None,
        end_date=None,
        timeframe=args.timeframe,
    )
    split_idx = int(len(df_raw) * (1 - args.test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    df_test_raw = df_raw.iloc[split_idx:].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_results = []
    try:
        for variant in variants:
            loader = StrategyConfigLoader(variant.config_dir)
            strategy_cfg = loader.load()
            base_result = execute_single_run(strategy_cfg, df_train_raw, df_test_raw)
            rolling_result = None
            if args.run_rolling:
                rolling_result = run_rolling_evaluation(strategy_cfg, df_raw, args)
            comparison_results.append(
                {
                    "variant": variant.name,
                    "base": base_result or {},
                    "rolling": rolling_result or {},
                }
            )
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

    summary_df = summarize_results(comparison_results)
    summary_csv = output_dir / "strategy_feature_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    detailed_json = output_dir / "strategy_feature_compare_summary.json"
    with open(detailed_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "test_size": args.test_size,
                "results": comparison_results,
            },
            fh,
            indent=2,
            default=str,
        )

    print(f"✅ Saved summary CSV to {summary_csv}")
    print(f"✅ Saved summary JSON to {detailed_json}")


if __name__ == "__main__":
    main()
