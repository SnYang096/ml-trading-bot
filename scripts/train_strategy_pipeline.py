#!/usr/bin/env python3
"""
Unified strategy training script driven entirely by per-strategy configuration directories.

This script is the MAIN ENTRY POINT for strategy training. It orchestrates the entire pipeline:
- Loads raw data and strategy configuration
- Runs feature engineering pipeline
- Generates labels
- Calls the model trainer (strategy_trainer.py) for cross-validation
- Evaluates predictions and runs vectorbt backtests
- Saves results to disk

IMPORTANT: This is different from strategy_trainer.py:
- train_strategy_pipeline.py: Complete training pipeline orchestrator (THIS FILE)
- strategy_trainer.py: Low-level model training function (XGBoost/CatBoost/LightGBM CV only)

Usage:
    python scripts/train_strategy_pipeline.py --config config/strategies/sr_reversal --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd

from src.data_tools.data_utils import load_raw_data
from src.data_tools.tick_loader import list_tick_files, serialize_tick_loader_params
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.time_series_model.pipeline.training.label_utils import (
    simulate_rr_exits,
    future_volatility_label,
)
from src.time_series_model.pipeline.training.volatility_model_config import (
    load_volatility_model_config,
    prepare_volatility_model_data,
    get_volatility_model_params,
)
from src.time_series_model.strategies.backtesting.vectorbt_backtest import (
    VectorBTBacktest,
)

BASE_DATA_COLUMNS = {
    "timestamp",
    "datetime",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "_symbol",
    "trade_count",
    "buy_qty",
    "sell_qty",
    "delta",
    "taker_buy_ratio",
    "cvd",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified strategy trainer (config driven)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/strategies",
        help="Path to strategy config directory or root containing multiple strategies",
    )
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to train on")
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--timeframe", type=str, default="15T")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-root", type=str, default="results/strategies")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Optional specific strategy name (or comma separated) inside config root",
    )
    return parser.parse_args()


def discover_strategy_dirs(
    config_path: Path, selected: Optional[List[str]]
) -> List[Path]:
    if (config_path / "features.yaml").exists():
        # Single strategy directory
        if selected and config_path.name not in selected:
            return []
        return [config_path]

    strategies = []
    for subdir in sorted(p for p in config_path.iterdir() if p.is_dir()):
        if not (subdir / "features.yaml").exists():
            continue
        if selected and subdir.name not in selected:
            continue
        strategies.append(subdir)
    return strategies


def import_callable(module_path: str, function_name: str):
    module = import_module(module_path)
    return getattr(module, function_name)


def ensure_signal_column(
    df: pd.DataFrame, ensure_cfg: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    if not ensure_cfg:
        return df
    column = ensure_cfg.get("name", "signal")
    default_value = ensure_cfg.get("default_value", 0)
    if column not in df.columns:
        df[column] = default_value
    return df


def _ensure_ticks_configured(
    feature_loader: StrategyFeatureLoader,
    symbol: str,
    data_path: str | Path,
    start_ts: Optional[str],
    end_ts: Optional[str],
    requested_features: List[str],
) -> None:
    """
    确保所有需要 ticks 数据的特征都配置了 ticks_loader_json。

    如果找不到 ticks 数据，抛出 ValueError。

    Args:
        feature_loader: 特征加载器
        symbol: 交易对符号
        data_path: 数据路径
        start_ts: 开始时间戳
        end_ts: 结束时间戳
        requested_features: 请求的特征列表（用于检查哪些特征需要 ticks）

    Raises:
        ValueError: 如果找不到 ticks 数据文件
    """
    if not start_ts or not end_ts:
        raise ValueError("start_ts and end_ts are required for ticks configuration")

    # 确保 feature_deps 中有 "features" 键
    if "features" not in feature_loader.feature_deps:
        feature_loader.feature_deps["features"] = {}
    features_cfg = feature_loader.feature_deps["features"]

    # 关键：requested_features 里可能是 output columns（如 "vpin"），不是父特征名（如 "vpin_features"）。
    # compute_features_parallel 会把 output column 映射回父特征；这里也必须做同样的解析，否则会漏配 ticks。
    output_col_to_feature: dict[str, str] = {}
    for feat_name, feat_cfg in features_cfg.items():
        out_cols = feat_cfg.get("output_columns", [feat_name]) or [feat_name]
        for out_col in out_cols:
            output_col_to_feature[out_col] = feat_name

    actual_requested: list[str] = []
    for req in requested_features or []:
        if req in features_cfg:
            actual_requested.append(req)
        elif req in output_col_to_feature:
            parent = output_col_to_feature[req]
            if parent not in actual_requested:
                actual_requested.append(parent)
        else:
            # 保留未知项（后续可能由依赖解析或直接报错）
            actual_requested.append(req)

    # 基于 compute_func 签名判断是否需要 ticks（而不是硬编码）
    tick_required_features: list[str] = []
    try:
        import inspect
        from src.features.loader.feature_function_mapping import get_compute_func

        for feat_name in actual_requested:
            if feat_name not in features_cfg:
                continue
            compute_func_name = features_cfg[feat_name].get("compute_func")
            if not compute_func_name:
                continue
            compute_func = get_compute_func(compute_func_name)
            sig = inspect.signature(compute_func)
            if ("ticks" in sig.parameters) or ("ticks_loader_json" in sig.parameters):
                tick_required_features.append(feat_name)
    except Exception:
        # fallback：保守兼容旧逻辑
        tick_required_features = [
            f for f in actual_requested if f in ("vpin_features", "footprint_basic")
        ]

    tick_required_features = list(dict.fromkeys(tick_required_features))

    if not tick_required_features:
        return

    # 检查是否已经有 ticks_loader_json（从 vpin_features 或其他特征）
    ticks_loader_json = None
    for feature_name, feature_cfg in features_cfg.items():
        compute_params = feature_cfg.get("compute_params", {})
        if compute_params.get("ticks_loader_json"):
            ticks_loader_json = compute_params["ticks_loader_json"]
            break

    # 如果还没有，创建新的
    if not ticks_loader_json:
        tick_files = list_tick_files(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            ticks_dir=str(data_path),
            lookback_minutes=60,
        )

        if not tick_files:
            raise ValueError(
                f"Tick data files not found for {symbol} in time range {start_ts} to {end_ts}. "
                f"Required for features: {requested_need_ticks}. "
                f"Please ensure tick data files exist in {data_path}"
            )

        tick_params = {
            "symbol": symbol,
            "tick_files": [str(Path(f)) for f in tick_files],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "lookback_minutes": 60,
        }
        ticks_loader_json = serialize_tick_loader_params(tick_params)
        print(f"   ✅ Configured ticks_loader_json with {len(tick_files)} files")

    # 为所有需要 ticks 的特征设置 ticks_loader_json
    for feature_name in tick_required_features:
        if feature_name in features_cfg:
            # 确保 feature_cfg 有 compute_params 键
            if "compute_params" not in features_cfg[feature_name]:
                features_cfg[feature_name]["compute_params"] = {}
            compute_params = features_cfg[feature_name]["compute_params"]

            if not compute_params.get("ticks_loader_json"):
                compute_params["ticks_loader_json"] = ticks_loader_json
                print(f"   ✅ Set ticks_loader_json for {feature_name}")
                # 验证设置是否成功
                if features_cfg[feature_name]["compute_params"].get(
                    "ticks_loader_json"
                ):
                    print(
                        f"   ✅ Verified: {feature_name} now has ticks_loader_json in feature_deps"
                    )
                else:
                    print(
                        f"   ⚠️  Warning: Failed to set ticks_loader_json for {feature_name}"
                    )
            else:
                print(f"   ℹ️  {feature_name} already has ticks_loader_json")
        else:
            raise ValueError(
                f"Feature '{feature_name}' is requested but not found in feature_deps. "
                f"Available features: {list(features_cfg.keys())[:20]}"
            )

    # 最终验证：检查所有需要的特征是否都有 ticks_loader_json
    print(f"   🔍 Final verification of ticks_loader_json configuration:")
    for feature_name in tick_required_features:
        if feature_name in features_cfg:
            compute_params = features_cfg[feature_name].get("compute_params", {})
            if compute_params.get("ticks_loader_json"):
                print(f"   ✅ {feature_name}: ticks_loader_json is set")
            else:
                print(
                    f"   ❌ {feature_name}: ticks_loader_json is NOT set (keys: {list(compute_params.keys())})"
                )
                raise ValueError(
                    f"Failed to set ticks_loader_json for {feature_name}. "
                    f"This should not happen. Please check the code."
                )


def run_feature_pipeline(
    df: pd.DataFrame,
    feature_loader: StrategyFeatureLoader,
    pipeline_cfg,
    fit: bool,
) -> pd.DataFrame:
    df_features = feature_loader.load_features_from_requested(
        df, pipeline_cfg.requested_features, fit=fit
    )
    df_features = ensure_signal_column(df_features, pipeline_cfg.ensure_signal)

    for processor in pipeline_cfg.post_processors:
        func = import_callable(processor.module, processor.function)
        df_features = func(df_features, **processor.params)

    return df_features


def determine_feature_columns(
    df: pd.DataFrame,
    pipeline_cfg,
) -> List[str]:
    if pipeline_cfg.selector:
        selector_func = import_callable(
            pipeline_cfg.selector.module, pipeline_cfg.selector.function
        )
        try:
            return selector_func(df, list(df.columns), **pipeline_cfg.selector.params)
        except TypeError:
            return selector_func(df, **pipeline_cfg.selector.params)

    return [
        col
        for col in df.columns
        if col not in BASE_DATA_COLUMNS
        and not col.startswith(("signal", "binary_signal"))
    ]


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    result = df
    for filt in filters:
        column = filt.get("column")
        if not column or column not in result.columns:
            continue
        if filt.get("notna"):
            result = result[result[column].notna()]
        if "include" in filt:
            result = result[result[column].isin(filt["include"])]
        if "exclude" in filt:
            result = result[~result[column].isin(filt["exclude"])]
        if "min" in filt:
            result = result[result[column] >= filt["min"]]
        if "max" in filt:
            result = result[result[column] <= filt["max"]]
    return result


def apply_post_label_filters(
    df: pd.DataFrame,
    filters: List[Dict[str, Any]],
    feature_cols: List[str],
) -> pd.DataFrame:
    result = df
    for filt in filters:
        if filt.get("ensure_feature_non_null"):
            if feature_cols:
                result = result[result[feature_cols].notna().all(axis=1)]
            continue

        column = filt.get("column")
        if filt.get("type") == "map_values" and column and column in result.columns:
            mapping = filt.get("mapping", {})
            output_column = filt.get("output_column", column)
            result[output_column] = result[column].map(mapping)
            continue

        if column and column in result.columns and filt.get("notna"):
            result = result[result[column].notna()]
    return result


def train_volatility_model_in_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_loader: Any,
    vol_config,
) -> Tuple[Optional[Any], Optional[Dict[str, float]]]:
    """
    Train volatility model within the main training pipeline.

    Args:
        df_train: Training DataFrame with features
        df_test: Test DataFrame with features
        feature_loader: Feature loader for computing missing features
        vol_config: VolatilityModelConfig instance

    Returns:
        Tuple of (volatility_model, metrics_dict) or (None, None) if training fails
    """
    try:
        from src.time_series_model.strategies.models.lightgbm_model import (
            LightGBMTrainer,
        )

        # Load volatility model config
        config_path = vol_config.config_path
        config = (
            load_volatility_model_config(config_path)
            if config_path
            else load_volatility_model_config()
        )

        # Generate volatility labels
        target_col = vol_config.target_column
        if target_col not in df_train.columns:
            print(f"   📊 Generating {target_col} labels...")
            # Use future_volatility_label to generate labels
            horizon = config.get("prediction", {}).get("horizon", 10)
            df_train[target_col] = future_volatility_label(
                df_train["close"], horizon=horizon
            )
            df_test[target_col] = future_volatility_label(
                df_test["close"], horizon=horizon
            )

        # Prepare volatility model data
        X_train_vol, vol_features, categorical_features = prepare_volatility_model_data(
            df_train, config, feature_loader=feature_loader
        )
        X_test_vol, _, _ = prepare_volatility_model_data(
            df_test, config, feature_loader=feature_loader
        )

        y_vol_train = df_train[target_col]
        y_vol_test = df_test[target_col]

        # Filter to valid samples
        valid_train = y_vol_train.notna() & X_train_vol[vol_features].notna().all(
            axis=1
        )
        valid_test = y_vol_test.notna() & X_test_vol[vol_features].notna().all(axis=1)

        X_train_vol = X_train_vol[vol_features].loc[valid_train]
        y_vol_train = y_vol_train.loc[valid_train]
        X_test_vol = X_test_vol[vol_features].loc[valid_test]
        y_vol_test = y_vol_test.loc[valid_test]

        if len(X_train_vol) < 50:
            print(
                f"   ⚠️  Not enough samples for volatility model training: {len(X_train_vol)}"
            )
            return None, None

        # Get training parameters from config
        trainer_config = config.get("trainer", {})
        use_gpu = trainer_config.get("use_gpu", True)
        n_splits = trainer_config.get("n_splits", 5)
        auto_tune_params = trainer_config.get("auto_tune_params", False)
        model_params = get_volatility_model_params(config)

        # Train volatility model
        vol_model = LightGBMTrainer(model_type="regression", use_gpu=use_gpu)
        if model_params:
            vol_model.params = model_params

        metrics, _ = vol_model.train(
            X_train_vol,
            y_vol_train,
            n_splits=n_splits,
            use_time_series_cv=True,
            groups=None,
            auto_tune_params=auto_tune_params,
            categorical_features=categorical_features,
        )

        # Store feature list for prediction
        vol_model._volatility_features = vol_features
        if categorical_features:
            vol_model._categorical_features = categorical_features

        return vol_model, metrics

    except Exception as e:
        print(f"   ⚠️  Volatility model training failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def drop_inf_rows(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Remove rows containing inf/-inf in feature columns (NaN is kept)."""
    if not feature_cols:
        return df
    dedup_cols = list(dict.fromkeys(feature_cols))
    result = df.copy()
    # 只替换 inf，保留 NaN（NaN 可能是正常的，如数据不足）
    subset = result[dedup_cols].replace([np.inf, -np.inf], np.nan)
    for col in dedup_cols:
        result[col] = subset[col]
    # 只检查 inf，不检查 NaN（因为 NaN 可能是正常的）
    inf_mask = np.isinf(result[dedup_cols])
    has_inf = inf_mask.any(axis=1)
    finite_mask = ~has_inf
    dropped = len(result) - finite_mask.sum()
    result = result[finite_mask]
    if dropped > 0:
        print(f"   ⚠️  Dropped {dropped} rows due to inf/-inf in features")
    return result


def generate_predictions(
    models: List[Any],
    model_type: str,
    task_type: str,
    X: np.ndarray,
) -> np.ndarray:
    if not models:
        return np.zeros(len(X))

    is_multiclass = task_type == "multiclass"
    preds_sum: Optional[np.ndarray]
    preds_sum = None

    if model_type == "xgboost":
        import xgboost as xgb

        dtest = xgb.DMatrix(X)
        for model in models:
            pred = model.predict(dtest)
            if preds_sum is None:
                preds_sum = np.zeros_like(pred)
            preds_sum += pred
    elif model_type == "catboost":
        for model in models:
            if task_type == "binary":
                pred = model.predict_proba(X)[:, 1]
            elif task_type == "multiclass":
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            if preds_sum is None:
                preds_sum = np.zeros_like(pred)
            preds_sum += pred
    elif model_type == "lightgbm":
        for model in models:
            pred = model.predict(X)
            if preds_sum is None:
                preds_sum = np.zeros_like(pred)
            preds_sum += pred
    else:
        raise ValueError(f"Unsupported model_type for prediction: {model_type}")

    preds = preds_sum / len(models)

    # For binary catboost/lightgbm, ensure 1D vector
    if not is_multiclass and preds.ndim > 1:
        preds = preds[:, 1] if preds.shape[1] > 1 else preds.ravel()

    return preds


def evaluate_predictions(
    preds: np.ndarray,
    y_true: np.ndarray,
    evaluation_cfg,
) -> Dict[str, float]:
    metrics = evaluation_cfg.metrics or [
        {
            "name": "pearson_correlation",
            "type": "correlation",
            "params": {"method": "pearson"},
        }
    ]
    results: Dict[str, float] = {}

    for metric in metrics:
        name = metric.get("name", metric.get("type", "metric"))
        metric_type = metric.get("type", "correlation")
        params = metric.get("params", {}) or {}

        if metric_type == "correlation":
            method = params.get("method", "pearson")
            if method == "spearman":
                from scipy.stats import spearmanr

                score = float(
                    spearmanr(preds, y_true, nan_policy="omit").correlation or 0.0
                )
            else:
                score = float(np.corrcoef(preds, y_true)[0, 1])
        elif metric_type == "rank_ic":
            from scipy.stats import spearmanr

            score = float(
                spearmanr(preds, y_true, nan_policy="omit").correlation or 0.0
            )
        elif metric_type == "accuracy":
            if preds.ndim == 2:
                pred_class = np.argmax(preds, axis=1)
            else:
                threshold = params.get("threshold", 0.5)
                pred_class = (preds >= threshold).astype(int)
            score = float((pred_class == y_true).mean())
        else:
            raise ValueError(f"Unsupported evaluation metric type: {metric_type}")

        if np.isnan(score):
            score = 0.0
        results[name] = score

    return results


def run_vectorbt_backtest(
    df: pd.DataFrame,
    preds: np.ndarray,
    backtest_cfg,
    task_type: str,
    strategy_config=None,
) -> Optional[Dict[str, float]]:
    if not backtest_cfg.enabled:
        return None
    try:
        import vectorbt as vbt
    except ImportError:
        print("   ⚠️  vectorbt not installed. Skipping backtest.")
        return None

    params = backtest_cfg.params or {}
    price_col = params.get("price_col", "close")
    if price_col not in df.columns:
        print(f"   ⚠️  Price column '{price_col}' not found. Skipping backtest.")
        return None

    price = df[price_col].astype(float)
    fee = params.get("fee", 0.0004)
    slippage = params.get("slippage", 0.0)
    init_cash = params.get("initial_cash", 10000.0)

    index = df.index

    debug = bool(params.get("debug", False))
    use_signal_direction = bool(params.get("use_signal_direction", False))
    signal_col = params.get("signal_col", "signal")
    use_rr_exit = bool(params.get("use_rr_exit", False))

    # 确定策略方向：从配置或策略名称推断
    strategy_direction = params.get(
        "strategy_direction", None
    )  # long_only, short_only, both
    if strategy_direction is None and strategy_config is not None:
        # 从 label_generator.params 中读取 combine_mode
        label_params = strategy_config.labels.generator.params or {}
        combine_mode = label_params.get("combine_mode")
        if combine_mode == "long_only":
            strategy_direction = "long_only"
        elif combine_mode == "short_only":
            strategy_direction = "short_only"
        else:
            # 从策略名称推断
            strategy_name = strategy_config.name.lower()
            if "_long" in strategy_name or strategy_name.endswith("_long"):
                strategy_direction = "long_only"
            elif "_short" in strategy_name or strategy_name.endswith("_short"):
                strategy_direction = "short_only"
            else:
                strategy_direction = "both"  # 默认双向
    elif strategy_direction is None:
        strategy_direction = "both"  # 默认双向

    if task_type == "multiclass" and preds.ndim == 2:
        class_preds = np.argmax(preds, axis=1)
        multi_cfg = params.get("multiclass", {})
        long_class = multi_cfg.get("long_class", 2)
        short_class = multi_cfg.get("short_class", 0)
        neutral_class = multi_cfg.get("neutral_class", 1)
        long_entries = pd.Series(class_preds == long_class, index=index)
        long_exits = pd.Series(class_preds == neutral_class, index=index)
        short_entries = pd.Series(class_preds == short_class, index=index)
        short_exits = pd.Series(class_preds == neutral_class, index=index)
    else:
        long_entry = params.get("long_entry_threshold", 0.6)
        long_exit = params.get("long_exit_threshold", 0.4)
        short_entry = params.get("short_entry_threshold", 0.4)
        short_exit = params.get("short_exit_threshold", 0.6)

        preds_series = pd.Series(preds, index=index)

        if use_signal_direction and signal_col in df.columns:
            # SR reversal 等策略：方向由 signal 决定，preds 只控制"是否参与这笔 SR 反转交易"
            signal_series = df[signal_col].fillna(0).astype(float)

            base_long_entries = preds_series >= long_entry
            base_short_entries = preds_series <= short_entry

            # 根据策略方向过滤信号
            if strategy_direction == "long_only":
                long_entries = (signal_series > 0) & base_long_entries
                short_entries = pd.Series(False, index=index)  # 不做空
            elif strategy_direction == "short_only":
                long_entries = pd.Series(False, index=index)  # 不做多
                short_entries = (signal_series < 0) & base_short_entries
            else:  # both
                long_entries = (signal_series > 0) & base_long_entries
                short_entries = (signal_series < 0) & base_short_entries

            # 初始情形下仍保留概率退出，后续可被 RR 逻辑覆盖
            long_exits = preds_series <= long_exit
            short_exits = preds_series >= short_exit
        else:
            # 默认行为：仅根据预测得分构造多空信号
            if strategy_direction == "long_only":
                long_entries = preds_series >= long_entry
                long_exits = preds_series <= long_exit
                short_entries = pd.Series(False, index=index)  # 不做空
                short_exits = pd.Series(False, index=index)
            elif strategy_direction == "short_only":
                long_entries = pd.Series(False, index=index)  # 不做多
                long_exits = pd.Series(False, index=index)
                short_entries = preds_series <= short_entry
                short_exits = preds_series >= short_exit
            else:  # both
                long_entries = preds_series >= long_entry
                long_exits = preds_series <= long_exit
                short_entries = preds_series <= short_entry
                short_exits = preds_series >= short_exit

        if debug:
            debug_signals = pd.DataFrame(
                {
                    "price": price,
                    "pred": preds_series,
                    "long_entry": long_entries,
                    "long_exit": long_exits,
                    "short_entry": short_entries,
                    "short_exit": short_exits,
                }
            )

    # 如果启用 RR 驱动的平仓逻辑，则重写 exits/short_exits（与 compute_rr_label 保持一致）
    if use_rr_exit:
        if not use_signal_direction:
            raise ValueError(
                "use_rr_exit=True 要求 use_signal_direction=True，以确保方向由 signal 决定"
            )

        rr_params = params.get("rr", {})
        rr_max_holding_bars = int(rr_params.get("max_holding_bars", 24))
        rr_stop_loss_r = float(rr_params.get("stop_loss_r", 1.0))
        rr_take_profit_r = float(rr_params.get("take_profit_r", 2.0))
        rr_atr_window = int(rr_params.get("atr_window", 14))
        rr_entry_offset = int(rr_params.get("entry_offset", 1))
        rr_entry_price_col = rr_params.get("entry_price_col", None)

        # 构造仅包含“被模型选中的 SR 信号”的方向列：1=多，-1=空
        rr_signal = pd.Series(0.0, index=index)
        rr_signal[long_entries] = 1.0
        rr_signal[short_entries] = -1.0

        df_rr = df.copy()
        df_rr[signal_col] = rr_signal

        long_exits_rr, short_exits_rr = simulate_rr_exits(
            df_rr,
            signal_col=signal_col,
            price_col=price_col,
            atr_col=params.get("atr_col", "atr"),
            atr_window=rr_atr_window,
            max_holding_bars=rr_max_holding_bars,
            stop_loss_r=rr_stop_loss_r,
            take_profit_r=rr_take_profit_r,
            entry_price_col=rr_entry_price_col,
            entry_offset=rr_entry_offset,
        )

        # 用 RR 逻辑产生的 exits 覆盖概率退出
        long_exits = long_exits_rr.reindex(index).fillna(False)
        short_exits = short_exits_rr.reindex(index).fillna(False)

    # Determine frequency for vectorbt metrics (REQUIRED for proper metrics calculation)
    freq = params.get("freq", None)
    if freq is None:
        # Try to infer frequency from DatetimeIndex as fallback
        if isinstance(index, pd.DatetimeIndex):
            inferred_freq = index.inferred_freq
            if inferred_freq:
                freq = inferred_freq
            else:
                # Fallback: try to infer from common timeframes
                if len(index) > 1:
                    time_diff = index[1] - index[0]
                    # Convert to pandas frequency string
                    if time_diff.total_seconds() == 900:  # 15 minutes
                        freq = "15T"
                    elif time_diff.total_seconds() == 3600:  # 1 hour
                        freq = "1H"
                    elif time_diff.total_seconds() == 14400:  # 4 hours
                        freq = "4H"
                    elif time_diff.total_seconds() == 86400:  # 1 day
                        freq = "1D"

        # If still None, raise error - freq MUST be configured in backtest.yaml
        if freq is None:
            raise ValueError(
                "❌ 'freq' must be configured in backtest.yaml params. "
                "Example: freq: '4H' for 4-hour timeframe, '15T' for 15-minute. "
                "This is required for vectorbt to calculate Sharpe ratio and other frequency-dependent metrics."
            )

    try:
        portfolio = vbt.Portfolio.from_signals(
            price,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=init_cash,
            fees=fee,
            slippage=slippage,
            freq=freq,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Backtest failed: {exc}")
        return None

    stats = portfolio.stats()

    debug_payload: Dict[str, Any] | None = None
    if debug:
        debug_payload = {}
        try:
            trades = portfolio.trades.records_readable
        except Exception:
            trades = None

        # 组装 debug summary，供日志和 HTML 使用
        total_return_pct = float(stats.get("Total Return [%]", 0.0))
        sharpe_ratio = float(stats.get("Sharpe Ratio", 0.0))
        max_dd_pct = float(stats.get("Max Drawdown [%]", 0.0))
        win_rate_pct = float(stats.get("Win Rate [%]", 0.0))

        debug_payload["summary"] = {
            "total_return_pct": total_return_pct,
            "sharpe": sharpe_ratio,
            "max_drawdown_pct": max_dd_pct,
            "win_rate_pct": win_rate_pct,
        }

        if trades is not None and not trades.empty:
            n_trades = int(len(trades))
            n_win = int((trades["PnL"] > 0).sum())
            win_rate_manual = 100.0 * n_win / n_trades

            # 为 HTML 导出部分 trades（避免太大），按时间排序
            trades_sample = (
                trades.sort_values("Entry Timestamp").head(200).reset_index(drop=True)
            )
            debug_payload["trades"] = trades_sample.to_dict(orient="records")
            debug_payload["trades_meta"] = {
                "n_trades": n_trades,
                "n_win": n_win,
                "win_rate_manual": win_rate_manual,
            }

        # 存储部分信号行（仅非多分类情形下）
        if "debug_signals" in locals():
            entry_mask = long_entries | short_entries
            if strategy_direction == "long_only":
                debug_payload["strategy_direction"] = "long_only"
            elif strategy_direction == "short_only":
                debug_payload["strategy_direction"] = "short_only"
            else:
                debug_payload["strategy_direction"] = "both"
            signals_sample = (
                debug_signals[entry_mask]
                .head(200)
                .reset_index()
                .rename(columns={"index": "timestamp"})
            )
            debug_payload["signals"] = signals_sample.to_dict(orient="records")

        # returns 统计
        try:
            returns = portfolio.returns()
            mean_ret = float(returns.mean())
            std_ret = float(returns.std())
            debug_payload["returns_stats"] = {
                "mean": mean_ret,
                "std": std_ret,
            }
        except Exception:
            pass

    result: Dict[str, Any] = {
        "total_return_pct": float(stats.get("Total Return [%]", 0.0)),
        "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
        "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)),
        "win_rate": float(stats.get("Win Rate [%]", 0.0)),
    }

    if debug_payload is not None:
        result["debug"] = debug_payload

    return result


def run_backtest_with_strategy(
    df: pd.DataFrame,
    preds: np.ndarray,
    strategy_config,
    task_type: str,
    vol_model: Optional[Any] = None,
) -> Optional[Dict[str, float]]:
    """
    根据 backtest 配置动态选择回测类；若未指定 class 则回退到 VectorBTBacktest。
    """
    backtest_cfg = strategy_config.backtest
    params = backtest_cfg.params or {}
    params["enabled"] = backtest_cfg.enabled

    # 确定策略方向：从 label_generator.params 或策略名称推断
    strategy_direction = params.get("strategy_direction")
    if strategy_direction is None:
        # 从 label_generator.params 中读取 combine_mode
        label_params = strategy_config.labels.generator.params or {}
        combine_mode = label_params.get("combine_mode")
        if combine_mode == "long_only":
            strategy_direction = "long_only"
        elif combine_mode == "short_only":
            strategy_direction = "short_only"
        else:
            # 从策略名称推断
            strategy_name = strategy_config.name.lower()
            if "_long" in strategy_name or strategy_name.endswith("_long"):
                strategy_direction = "long_only"
            elif "_short" in strategy_name or strategy_name.endswith("_short"):
                strategy_direction = "short_only"
            else:
                strategy_direction = "both"  # 默认双向
        params["strategy_direction"] = strategy_direction
        params["strategy_name"] = strategy_config.name  # 也传递策略名称

    # 统一使用 VectorBTBacktest（训练阶段不切换到策略特定类）
    backtester = VectorBTBacktest()
    return backtester.run(df=df, predictions=preds, task_type=task_type, **params)


def train_strategy(
    config_dir: Path,
    args: argparse.Namespace,
    feature_loader: StrategyFeatureLoader,
) -> None:
    print("\n" + "=" * 80)
    print(f"📂 Loading strategy config from {config_dir}")
    loader = StrategyConfigLoader(config_dir)
    strategy_config = loader.load()

    output_dir = Path(args.output_root) / strategy_config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Strategy: {strategy_config.name}")

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    # 监控源数据质量
    from src.features.utils.data_monitor import check_source_data_quality

    source_quality = check_source_data_quality(df_raw, args.data_path)

    # Optional date cropping to align with available tick data or focus window
    start_override = os.getenv("TRAIN_START_DATE")
    end_override = os.getenv("TRAIN_END_DATE")
    if start_override or end_override:
        dt_idx = None
        if not df_raw.empty:
            for col in ("datetime", "timestamp", "date"):
                if col in df_raw.columns:
                    dt_idx = pd.to_datetime(df_raw[col])
                    break
            if dt_idx is None and isinstance(df_raw.index, pd.DatetimeIndex):
                dt_idx = df_raw.index

        if dt_idx is not None:
            mask = pd.Series(True, index=df_raw.index)
            if start_override:
                mask &= dt_idx >= pd.to_datetime(start_override)
            if end_override:
                mask &= dt_idx <= pd.to_datetime(end_override)
            df_raw = df_raw.loc[mask]
            print(
                f"   ℹ️  Cropped data to [{start_override or '-inf'}, {end_override or '+inf'}], rows={len(df_raw)}"
            )

    # Configure VPIN tick loader if tick data is available
    datetime_col = next(
        (col for col in ("datetime", "timestamp", "date") if col in df_raw.columns),
        None,
    )
    if not df_raw.empty:
        if datetime_col:
            dt_series = pd.to_datetime(df_raw[datetime_col])
        elif isinstance(df_raw.index, pd.DatetimeIndex):
            dt_series = df_raw.index
        else:
            dt_series = None

        if dt_series is not None and len(dt_series) > 0:
            start_ts = dt_series.min().strftime("%Y-%m-%d %H:%M:%S")
            end_ts = dt_series.max().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"   📅 Ensuring ticks configuration for time range: {start_ts} to {end_ts}"
            )
            # 获取请求的特征列表
            requested_features = strategy_config.features.requested_features
            _ensure_ticks_configured(
                feature_loader,
                symbol=args.symbol,
                data_path=args.data_path,
                start_ts=start_ts,
                end_ts=end_ts,
                requested_features=requested_features,
            )
        else:
            raise ValueError(
                "No datetime/timestamp found in dataframe; cannot configure ticks"
            )
    else:
        raise ValueError("Empty dataframe; cannot configure ticks")

    split_idx = int(len(df_raw) * (1 - args.test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    df_test_raw = df_raw.iloc[split_idx:].copy()

    print(f"   ✅ Samples - Train: {len(df_train_raw)}, " f"Test: {len(df_test_raw)}")

    # 打印测试集时间范围，用于验证 tick 数据可用性
    if not df_test_raw.empty:
        datetime_col = next(
            (
                col
                for col in ("datetime", "timestamp", "date")
                if col in df_test_raw.columns
            ),
            None,
        )
        if datetime_col:
            test_start = pd.to_datetime(df_test_raw[datetime_col]).min()
            test_end = pd.to_datetime(df_test_raw[datetime_col]).max()
            print(f"   📅 Test set time range: {test_start} to {test_end}")
        elif isinstance(df_test_raw.index, pd.DatetimeIndex):
            test_start = df_test_raw.index.min()
            test_end = df_test_raw.index.max()
            print(f"   📅 Test set time range: {test_start} to {test_end}")

    requested = strategy_config.features.requested_features
    print(f"\n   ▶️ Feature pipeline (train) start: {len(requested)} requested features")
    df_train_features = run_feature_pipeline(
        df_train_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=True,
    )
    print(
        f"   ✅ Feature pipeline (train) done: rows={len(df_train_features)}, cols={len(df_train_features.columns)}"
    )
    print(f"   ▶️ Feature pipeline (test) start")
    df_test_features = run_feature_pipeline(
        df_test_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=False,
    )
    print(
        f"   ✅ Feature pipeline (test) done: rows={len(df_test_features)}, cols={len(df_test_features.columns)}\n"
    )

    feature_cols = determine_feature_columns(
        df_train_features, strategy_config.features
    )
    print(f"   ✅ Candidate features: {len(feature_cols)}")

    # Label generation
    label_func = import_callable(
        strategy_config.labels.generator.module,
        strategy_config.labels.generator.function,
    )

    df_train_features[strategy_config.labels.target_column] = label_func(
        df_train_features.copy(), **strategy_config.labels.generator.params
    )
    df_test_features[strategy_config.labels.target_column] = label_func(
        df_test_features.copy(), **strategy_config.labels.generator.params
    )
    train_labels = df_train_features[strategy_config.labels.target_column]
    test_labels = df_test_features[strategy_config.labels.target_column]
    print(
        f"   ℹ️  Label stats before filtering - "
        f"Train non-null: {train_labels.notna().sum()}, "
        f"pos: {(train_labels==1).sum()}, neg: {(train_labels==0).sum()}; "
        f"Test non-null: {test_labels.notna().sum()}, "
        f"pos: {(test_labels==1).sum()}, neg: {(test_labels==0).sum()}"
    )

    df_train_filtered = apply_filters(df_train_features, strategy_config.labels.filters)
    df_test_filtered = apply_filters(df_test_features, strategy_config.labels.filters)

    df_train_filtered = apply_post_label_filters(
        df_train_filtered,
        strategy_config.labels.post_label_filters,
        feature_cols,
    )
    df_test_filtered = apply_post_label_filters(
        df_test_filtered,
        strategy_config.labels.post_label_filters,
        feature_cols,
    )

    def _debug_inf(df: pd.DataFrame, name: str):
        if not feature_cols:
            return
        # 正确区分 inf 和 NaN：只检查真正的 inf/-inf，不包括 NaN
        # 注意：np.isfinite() 对 NaN 也返回 False，所以不能用来检查 inf
        inf_mask = np.isinf(df[feature_cols])
        if inf_mask.any().any():
            # 统计每列 inf/-inf 数量
            col_counts = inf_mask.sum().sort_values(ascending=False)
            top_cols = col_counts[col_counts > 0].head(10)
            print(
                f"   ⚠️  {name}: found inf/-inf in {len(top_cols)} columns "
                f"(top): {top_cols.to_dict()}"
            )
            # 打印每个问题列的极值和示例索引，便于定位
            for col in top_cols.index:
                col_series = df[col]
                # 只获取真正的 inf 值，不包括 NaN
                inf_idx = col_series[np.isinf(col_series)].index[:5]
                # 分别计算有限值、inf 值和 NaN 的统计
                finite_vals = col_series[np.isfinite(col_series)]
                inf_vals = col_series[np.isinf(col_series)]
                nan_vals = col_series[col_series.isna()]
                finite_min = finite_vals.min() if len(finite_vals) > 0 else None
                finite_max = finite_vals.max() if len(finite_vals) > 0 else None
                inf_count = len(inf_vals)
                nan_count = len(nan_vals)
                # 检查 inf 值的实际值
                inf_actual = inf_vals.head(3).tolist() if len(inf_vals) > 0 else []
                print(
                    f"      ↳ {col}: finite_min={finite_min}, finite_max={finite_max}, "
                    f"inf_count={inf_count}, nan_count={nan_count}, inf_samples={inf_actual}, "
                    f"sample_idx={list(inf_idx)}"
                )

    _debug_inf(df_train_filtered, "Train before drop_inf_rows")
    _debug_inf(df_test_filtered, "Test before drop_inf_rows")

    df_train_filtered = drop_inf_rows(df_train_filtered, feature_cols)
    df_test_filtered = drop_inf_rows(df_test_filtered, feature_cols)

    print(
        f"   ✅ Valid samples after filtering - "
        f"Train: {len(df_train_filtered)}, Test: {len(df_test_filtered)}"
    )
    if len(df_train_filtered) < 50:
        print("   ⚠️  Not enough samples to train, skipping strategy.")
        return

    trainer_func = import_callable(
        strategy_config.model.trainer.module,
        strategy_config.model.trainer.function,
    )
    trainer_params = dict(strategy_config.model.trainer.params)
    target_col = trainer_params.pop("target_col", strategy_config.labels.target_column)
    model_type = trainer_params.get("model_type", "xgboost")
    task_type = trainer_params.get("task_type", "regression")

    print(
        f"\n   🚀 Training model ({model_type}, task={task_type}) "
        f"on {len(df_train_filtered)} samples, {len(feature_cols)} features"
    )
    models, avg_metric, cv_results, used_features = trainer_func(
        df_train_filtered,
        feature_cols=feature_cols,
        target_col=target_col,
        **trainer_params,
    )

    print(f"   ✅ Average CV Metric: {avg_metric:.4f}")

    # Train volatility model if enabled
    vol_model = None
    vol_metrics = None
    if (
        strategy_config.model.volatility_model
        and strategy_config.model.volatility_model.enabled
    ):
        print("\n" + "=" * 80)
        print("📊 Training Volatility Model")
        print("=" * 80)
        vol_model, vol_metrics = train_volatility_model_in_pipeline(
            df_train_filtered,
            df_test_filtered,
            feature_loader=feature_loader,
            vol_config=strategy_config.model.volatility_model,
        )
        if vol_model:
            print(f"   ✅ Volatility model trained successfully")
            if vol_metrics:
                for metric_name, score in vol_metrics.items():
                    print(f"   ✅ Vol {metric_name}: {score:.4f}")
        else:
            print("   ⚠️  Volatility model training failed or skipped")
        print("=" * 80 + "\n")

    X_test = df_test_filtered[used_features].values
    y_test = df_test_filtered[target_col].values

    print(f"   ▶️ Generating predictions on test set ({len(df_test_filtered)} samples)")
    preds = generate_predictions(
        models=models,
        model_type=model_type,
        task_type=task_type,
        X=X_test,
    )

    evaluation_results = evaluate_predictions(
        preds,
        y_test,
        strategy_config.evaluation,
    )

    for metric_name, score in evaluation_results.items():
        print(f"   ✅ {metric_name}: {score:.4f}")

    results = {
        "strategy": strategy_config.name,
        "model_type": model_type,
        "task_type": task_type,
        "avg_cv_metric": float(avg_metric),
        "n_features": len(used_features),
        "n_train_samples": len(df_train_filtered),
        "n_test_samples": len(df_test_filtered),
        "evaluation": evaluation_results,
    }

    # Add volatility model results if trained
    if vol_model and vol_metrics:
        results["volatility_model"] = {
            "trained": True,
            "metrics": {k: float(v) for k, v in vol_metrics.items()},
        }
    elif (
        strategy_config.model.volatility_model
        and strategy_config.model.volatility_model.enabled
    ):
        results["volatility_model"] = {"trained": False}

    print(f"\n   ▶️ Running backtest on test set")
    backtest_results = run_backtest_with_strategy(
        df_test_filtered,
        preds,
        strategy_config,
        task_type=task_type,
        vol_model=vol_model,  # Pass volatility model to backtest
    )
    if backtest_results:
        results["backtest"] = backtest_results
        print(f"   ✅ Backtest completed")

    output_cfg = strategy_config.model.output
    if output_cfg.get("save_results", True):
        filename = output_cfg.get("filename", "results.json")
        results_file = output_dir / filename
        with open(results_file, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"   💾 Results saved to {results_file}")

        # Save volatility model if trained
        if vol_model:
            import joblib

            vol_model_file = output_dir / "volatility_model.pkl"
            joblib.dump(vol_model, vol_model_file)
            print(f"   💾 Volatility model saved to {vol_model_file}")


def main():
    args = parse_args()
    config_path = Path(args.config)
    selected = (
        [s.strip() for s in args.strategy.split(",") if s.strip()]
        if args.strategy
        else None
    )
    strategy_dirs = discover_strategy_dirs(config_path, selected)

    if not strategy_dirs:
        raise FileNotFoundError(
            f"No strategy configs found in {config_path}. "
            "Ensure the directory contains strategy subdirectories with features.yaml."
        )

    feature_loader = StrategyFeatureLoader()
    for strategy_dir in strategy_dirs:
        train_strategy(strategy_dir, args, feature_loader)


if __name__ == "__main__":
    main()
