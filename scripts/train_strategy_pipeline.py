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
    python scripts/train_strategy_pipeline.py --config config/strategies/sr_reversal_long --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

# Allow running this script directly without installing the project package.
# (So `import src.*` works when executed from the repo root.)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import pandas as pd

from src.data_tools.data_handler import DataHandler
from src.data_tools.tick_loader import list_tick_files, serialize_tick_loader_params
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.feature_store.layer_naming import default_layer_from_config
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
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Symbol to train on. Supports comma-separated symbols for pooled multi-symbol training (e.g. BTCUSDT,ETHUSDT).",
    )
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--timeframe", type=str, default="15T")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-root", type=str, default="results/strategies")
    # FeatureStore is always enabled for tree training (read-first + auto materialize on miss).
    parser.add_argument(
        "--feature-store-dir",
        type=str,
        default="feature_store",
        help="FeatureStore root dir (default: feature_store).",
    )
    parser.add_argument(
        "--feature-store-layer",
        type=str,
        default="AUTO",
        help="FeatureStore layer (dataset id). Default=AUTO (derived from config content).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducible training/backtests.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force single-threaded deterministic training (slower but reproducible).",
    )
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
        # Skip deprecated strategies unless explicitly selected
        if not selected:
            meta_path = subdir / "meta.yaml"
            if meta_path.exists():
                try:
                    import yaml

                    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                    if isinstance(meta, dict) and meta.get("deprecated") is True:
                        continue
                except Exception:
                    # If meta.yaml can't be parsed, do not block discovery.
                    pass
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
        from src.features.registry import get_compute_func

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
                f"Required for features: {tick_required_features}. "
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
    *,
    feature_store_dir: str | None = None,
    feature_store_layer: str | None = None,
    feature_store_symbol: str | None = None,
    feature_store_timeframe: str | None = None,
) -> pd.DataFrame:
    df_features = feature_loader.load_features_from_requested(
        df,
        pipeline_cfg.requested_features,
        fit=fit,
        feature_store_dir=feature_store_dir,
        feature_store_layer=feature_store_layer,
        feature_store_symbol=feature_store_symbol,
        feature_store_timeframe=feature_store_timeframe,
    )
    df_features = ensure_signal_column(df_features, pipeline_cfg.ensure_signal)

    # Process post_processors if they exist
    if pipeline_cfg.post_processors:
        for processor in pipeline_cfg.post_processors:
            try:
                func = import_callable(processor.module, processor.function)
                df_features = func(df_features, **processor.params)
            except (ModuleNotFoundError, AttributeError) as e:
                print(
                    f"   ⚠️  Warning: Failed to load post-processor {processor.module}.{processor.function}: {e}"
                )
                print(
                    f"   ℹ️  Skipping post-processor. If this is intentional, remove it from the config."
                )
                # Continue without this post-processor

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
    # Only numeric columns can contain +/-inf in a meaningful way.
    # Some feature pipelines may include non-numeric columns (e.g. DTW match labels).
    numeric_cols = (
        result[dedup_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    if not numeric_cols:
        return result

    # 只替换 inf，保留 NaN（NaN 可能是正常的，如数据不足）
    subset = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        result[col] = subset[col]
    # 只检查 inf，不检查 NaN（因为 NaN 可能是正常的）
    inf_mask = np.isinf(result[numeric_cols])
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
        elif metric_type == "regression_mae":
            # Mean Absolute Error for regression tasks
            valid_mask = ~(np.isnan(preds) & ~np.isnan(y_true))
            if valid_mask.sum() > 0:
                score = float(np.mean(np.abs(preds[valid_mask] - y_true[valid_mask])))
            else:
                score = 0.0
        elif metric_type == "regression_mse":
            # Mean Squared Error for regression tasks
            valid_mask = ~(np.isnan(preds) & ~np.isnan(y_true))
            if valid_mask.sum() > 0:
                score = float(np.mean((preds[valid_mask] - y_true[valid_mask]) ** 2))
            else:
                score = 0.0
        elif metric_type == "regression_rmse":
            # Root Mean Squared Error for regression tasks
            valid_mask = ~(np.isnan(preds) & ~np.isnan(y_true))
            if valid_mask.sum() > 0:
                score = float(
                    np.sqrt(np.mean((preds[valid_mask] - y_true[valid_mask]) ** 2))
                )
            else:
                score = 0.0
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

    # Optional safety fuse: block entries when too far from SR (OOD/overtrade guard)
    # Implemented as a mask applied to entries before RR exits / vectorbt portfolio.
    sr_fuse_cfg = params.get("sr_fuse", {}) or {}
    sr_fuse_enabled = bool(sr_fuse_cfg.get("enabled", False))
    sr_fuse_mask = pd.Series(True, index=df.index)
    if sr_fuse_enabled:
        dist_col = sr_fuse_cfg.get("dist_col", "dist_to_nearest_sr")
        atr_col = sr_fuse_cfg.get("atr_col", params.get("atr_col", "atr"))
        max_dist_atr = float(sr_fuse_cfg.get("max_dist_atr", 6.0))
        on_missing = str(sr_fuse_cfg.get("on_missing", "skip")).lower()  # skip|block

        have_dist = dist_col in df.columns
        have_atr = atr_col in df.columns

        if not have_dist and on_missing == "block":
            sr_fuse_mask = pd.Series(False, index=df.index)
            if debug:
                print(
                    f"   ⚠️  SR fuse enabled but '{dist_col}' missing; blocking all entries (on_missing=block)"
                )
        else:
            # Ensure ATR if needed and possible (uses RR atr_window if provided)
            if not have_atr:
                try:
                    from src.time_series_model.strategies.labels.sr_reversal_label import (
                        _ensure_atr,
                    )

                    rr_atr_window = int(
                        (params.get("rr", {}) or {}).get("atr_window", 14)
                    )
                    atr_series = _ensure_atr(
                        df.copy(),
                        atr_col=atr_col,
                        price_col="close",
                        high_col="high",
                        low_col="low",
                        atr_window=rr_atr_window,
                    )
                    df = df.copy()
                    df[atr_col] = atr_series
                    have_atr = True
                    if debug:
                        print(f"   ℹ️  SR fuse: computed missing ATR column '{atr_col}'")
                except Exception as exc:  # noqa: BLE001
                    if on_missing == "block":
                        sr_fuse_mask = pd.Series(False, index=df.index)
                        if debug:
                            print(
                                f"   ⚠️  SR fuse enabled but cannot compute ATR; blocking all entries: {exc}"
                            )
                    else:
                        if debug:
                            print(
                                f"   ⚠️  SR fuse enabled but cannot compute ATR; skipping fuse: {exc}"
                            )
                        sr_fuse_enabled = False

            if sr_fuse_enabled and have_dist and have_atr:
                dist = pd.to_numeric(df[dist_col], errors="coerce").abs()
                atr = (
                    pd.to_numeric(df[atr_col], errors="coerce")
                    .replace(0.0, np.nan)
                    .abs()
                )
                dist_atr = dist / atr
                sr_fuse_mask = (dist_atr <= max_dist_atr).fillna(
                    False if on_missing == "block" else True
                )
                if debug:
                    blocked = int((~sr_fuse_mask).sum())
                    print(
                        f"   ℹ️  SR fuse active: max_dist_atr={max_dist_atr}, blocked={blocked}/{len(sr_fuse_mask)}"
                    )

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

    if task_type == "regression":
        # For regression tasks (e.g., continuous RR prediction), use top quantile selection
        preds_series = pd.Series(preds, index=index)
        top_quantile = params.get("top_quantile", 0.1)  # Default: top 10%
        quantile_threshold = preds_series.quantile(1 - top_quantile)

        if strategy_direction == "long_only":
            long_entries_raw = preds_series >= quantile_threshold
            entry_mode = str(params.get("entry_mode", "level")).lower()
            if entry_mode == "cross":
                long_entries = long_entries_raw & (
                    ~long_entries_raw.shift(1).fillna(False)
                )
            else:
                long_entries = long_entries_raw
            long_exits = pd.Series(False, index=index)  # Exits handled by RR logic
            short_entries = pd.Series(False, index=index)
            short_exits = pd.Series(False, index=index)
        elif strategy_direction == "short_only":
            long_entries = pd.Series(False, index=index)
            long_exits = pd.Series(False, index=index)
            # For short, we want LOW predictions (negative RR or low positive RR)
            bottom_quantile = params.get("bottom_quantile", 0.1)
            quantile_threshold_short = preds_series.quantile(bottom_quantile)
            short_entries_raw = preds_series <= quantile_threshold_short
            entry_mode = str(params.get("entry_mode", "level")).lower()
            if entry_mode == "cross":
                short_entries = short_entries_raw & (
                    ~short_entries_raw.shift(1).fillna(False)
                )
            else:
                short_entries = short_entries_raw
            short_exits = pd.Series(False, index=index)  # Exits handled by RR logic
        else:  # both
            # For both directions, use top quantile for long, bottom quantile for short
            top_quantile_long = params.get("top_quantile", 0.1)
            bottom_quantile_short = params.get("bottom_quantile", 0.1)
            quantile_threshold_long = preds_series.quantile(1 - top_quantile_long)
            quantile_threshold_short = preds_series.quantile(bottom_quantile_short)
            long_entries = preds_series >= quantile_threshold_long
            short_entries = preds_series <= quantile_threshold_short
            long_exits = pd.Series(False, index=index)
            short_exits = pd.Series(False, index=index)
    elif task_type == "multiclass" and preds.ndim == 2:
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
        # For binary probability outputs:
        # - If strategy is direction-fixed (long_only or short_only) we treat `preds` as
        #   "success probability for THAT direction" => enter when preds >= entry_threshold.
        # - If strategy_direction == both, keep legacy behavior with separate long/short thresholds.
        entry_threshold = params.get(
            "entry_threshold", params.get("long_entry_threshold", 0.6)
        )
        # Exit semantics:
        # - For RR-based labels/strategies, pred is "probability of achieving RR", NOT an exit signal.
        # - Therefore, probability-threshold exits are opt-in only via exit_mode="threshold".
        exit_mode = str(params.get("exit_mode", "none")).lower()  # none|threshold
        exit_threshold = params.get(
            "exit_threshold", params.get("long_exit_threshold", 0.4)
        )
        long_entry = params.get("long_entry_threshold", entry_threshold)
        long_exit = params.get("long_exit_threshold", exit_threshold)
        short_entry = params.get("short_entry_threshold", 0.4)
        short_exit = params.get("short_exit_threshold", 0.6)

        preds_series = pd.Series(preds, index=index)

        # 默认行为：仅根据预测得分构造多空信号（A 策略）
        if strategy_direction == "long_only":
            # Direction-fixed probability gating (success proba for long trades)
            long_entries_raw = preds_series >= entry_threshold
            # 上穿触发（edge-trigger）：只在从 <threshold 到 >=threshold 的那一根开仓
            entry_mode = str(params.get("entry_mode", "cross")).lower()
            if entry_mode == "cross":
                long_entries = long_entries_raw & (
                    ~long_entries_raw.shift(1).fillna(False)
                )
            else:
                long_entries = long_entries_raw
            long_exits = (
                (preds_series <= exit_threshold)
                if exit_mode == "threshold"
                else pd.Series(False, index=index)
            )
            short_entries = pd.Series(False, index=index)  # 不做空
            short_exits = pd.Series(False, index=index)
        elif strategy_direction == "short_only":
            long_entries = pd.Series(False, index=index)  # 不做多
            long_exits = pd.Series(False, index=index)
            short_entries_raw = preds_series >= entry_threshold
            entry_mode = str(params.get("entry_mode", "cross")).lower()
            if entry_mode == "cross":
                short_entries = short_entries_raw & (
                    ~short_entries_raw.shift(1).fillna(False)
                )
            else:
                short_entries = short_entries_raw
            short_exits = (
                (preds_series <= exit_threshold)
                if exit_mode == "threshold"
                else pd.Series(False, index=index)
            )
        else:  # both
            # 保留 legacy 双向阈值逻辑（不建议与 A 策略混用）
            long_entries = preds_series >= long_entry
            long_exits = preds_series <= long_exit
            short_entries = preds_series <= short_entry
            short_exits = preds_series >= short_exit

        # Critical: never allow exit on the same bar as entry (vectorbt may treat it as "no trade")
        try:
            long_exits = long_exits & (~long_entries)
            short_exits = short_exits & (~short_entries)
        except Exception:
            pass

        # Apply SR fuse mask (if enabled)
        if sr_fuse_enabled:
            long_entries = long_entries & sr_fuse_mask
            short_entries = short_entries & sr_fuse_mask

        if debug:
            debug_signals = pd.DataFrame(
                {
                    "price": price,
                    "pred": preds_series,
                    "long_entry": long_entries,
                    "short_entry": short_entries,
                }
            )

    # 如果启用 RR 驱动的平仓逻辑，则重写 exits/short_exits（与 compute_rr_label 保持一致）
    if use_rr_exit:
        # RR exits only require that we can infer direction for selected entries.
        # - If use_signal_direction=True, direction comes from signal (possibly gated by preds)
        # - If direction-fixed (long_only/short_only), direction comes from strategy_direction
        if (not use_signal_direction) and (
            strategy_direction not in {"long_only", "short_only"}
        ):
            raise ValueError(
                "use_rr_exit=True requires either use_signal_direction=True OR a direction-fixed strategy "
                "(strategy_direction=long_only/short_only)."
            )

        rr_params = params.get("rr", {})
        rr_max_holding_bars = int(rr_params.get("max_holding_bars", 24))
        rr_stop_loss_r = float(rr_params.get("stop_loss_r", 1.0))
        rr_take_profit_r = float(rr_params.get("take_profit_r", 2.0))
        rr_atr_window = int(rr_params.get("atr_window", 14))
        rr_entry_offset = int(rr_params.get("entry_offset", 1))
        rr_entry_price_col = rr_params.get("entry_price_col", None)
        # ✅ 支持 breakeven stop（从配置中读取，默认 False 以保持向后兼容）
        rr_use_breakeven_stop = bool(rr_params.get("use_breakeven_stop", False))

        # 构造仅包含"被模型选中的 SR 信号"的方向列：1=多，-1=空
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
            use_breakeven_stop=rr_use_breakeven_stop,  # ✅ 传递 breakeven 参数
        )

        # 用 RR 逻辑产生的 exits 覆盖概率退出
        long_exits = long_exits_rr.reindex(index).fillna(False)
        short_exits = short_exits_rr.reindex(index).fillna(False)

    # ------------------------------------------------------------------
    # Resolve entry/exit conflicts
    #
    # If entries are "level" signals (e.g. pred >= threshold) they can be True
    # on almost every bar. When exits are also True on many bars (RR exits),
    # vectorbt will see entry & exit on the same bar. Depending on conflict
    # handling, this can collapse into a single long-running trade.
    #
    # Opt-in via params to keep backward compatibility.
    # ------------------------------------------------------------------
    conflict_mode = str(params.get("entry_exit_conflict", "none")).lower()
    if conflict_mode in {"block_entry_on_exit", "prefer_exit"}:
        long_entries = (long_entries.astype(bool) & (~long_exits.astype(bool))).astype(
            bool
        )
        short_entries = (
            short_entries.astype(bool) & (~short_exits.astype(bool))
        ).astype(bool)

    # ------------------------------------------------------------------
    # A 策略：max_holding_bars 强制平仓 + 期末强平
    # - 避免持仓跨越数月导致 “Status=Open”
    # - 避免每根K“想开仓”造成 rr_signal 近似全1
    # ------------------------------------------------------------------
    max_holding_bars = params.get("max_holding_bars", None)
    force_close_on_end = bool(params.get("force_close_on_end", True))
    if max_holding_bars is not None:
        try:
            max_holding_bars = int(max_holding_bars)
        except Exception:
            max_holding_bars = None

    if max_holding_bars is not None and max_holding_bars > 0:
        # single-position state machine: open on entry; close on exit_threshold OR timeout
        long_entries = long_entries.fillna(False).astype(bool)
        short_entries = short_entries.fillna(False).astype(bool)
        long_exits = long_exits.fillna(False).astype(bool)
        short_exits = short_exits.fillna(False).astype(bool)

        in_long = False
        in_short = False
        entry_i_long = -1
        entry_i_short = -1

        for i in range(len(index)):
            # entries only when flat
            if not in_long and not in_short:
                if bool(long_entries.iloc[i]):
                    in_long = True
                    entry_i_long = i
                    # never exit on entry bar
                    long_exits.iloc[i] = False
                elif bool(short_entries.iloc[i]):
                    in_short = True
                    entry_i_short = i
                    short_exits.iloc[i] = False

            # exit rules (do not exit on the same bar as entry)
            if in_long:
                held = i - entry_i_long
                if held >= 1 and (bool(long_exits.iloc[i]) or held >= max_holding_bars):
                    long_exits.iloc[i] = True
                    in_long = False
                    entry_i_long = -1
            if in_short:
                held = i - entry_i_short
                if held >= 1 and (
                    bool(short_exits.iloc[i]) or held >= max_holding_bars
                ):
                    short_exits.iloc[i] = True
                    in_short = False
                    entry_i_short = -1

        if force_close_on_end and len(index) > 0:
            # force close any remaining open position on the final bar
            if in_long:
                long_exits.iloc[-1] = True
            if in_short:
                short_exits.iloc[-1] = True

        # Re-apply safety: no same-bar exit
        long_exits = long_exits & (~long_entries)
        short_exits = short_exits & (~short_entries)

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

    # If there are no trades, vectorbt stats can return inf/NaN (e.g., Sharpe = inf when std=0).
    # Return a consistent payload and keep Sharpe/drawdown/win_rate as NaN to indicate "N/A".
    try:
        trade_count = int(portfolio.trades.count())
    except Exception:
        trade_count = 0

    stats = portfolio.stats()

    debug_payload: Dict[str, Any] | None = None
    if debug:
        debug_payload = {}
        try:
            trades = portfolio.trades.records_readable
        except Exception:
            trades = None

        # Summary snapshot (may contain NaN/inf; downstream reports should sanitize)
        debug_payload["summary"] = {
            "total_return_pct": float(stats.get("Total Return [%]", 0.0)),
            "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
            "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)),
            "win_rate_pct": float(stats.get("Win Rate [%]", 0.0)),
        }

        if trades is not None and not trades.empty:
            n_trades = int(len(trades))
            n_win = int((trades["PnL"] > 0).sum())
            win_rate_manual = 100.0 * n_win / n_trades
            trades_sample = (
                trades.sort_values("Entry Timestamp").head(200).reset_index(drop=True)
            )
            debug_payload["trades"] = trades_sample.to_dict(orient="records")
            debug_payload["trades_meta"] = {
                "n_trades": n_trades,
                "n_win": n_win,
                "win_rate_manual": win_rate_manual,
            }

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

        try:
            returns = portfolio.returns()
            debug_payload["returns_stats"] = {
                "mean": float(returns.mean()),
                "std": float(returns.std()),
            }
        except Exception:
            pass

    if trade_count == 0:
        print(
            "   ⚠️  Backtest produced no trades; metrics like Sharpe/WinRate/Drawdown are N/A."
        )
        return {
            "total_return_pct": float(stats.get("Total Return [%]", 0.0)),
            "sharpe": float("nan"),
            "max_drawdown_pct": float("nan"),
            "win_rate": float("nan"),
            "total_trades": 0,
            **({"debug": debug_payload} if debug_payload is not None else {}),
        }

    result: Dict[str, Any] = {
        "total_return_pct": float(stats.get("Total Return [%]", 0.0)),
        "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
        "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)),
        "win_rate": float(stats.get("Win Rate [%]", 0.0)),
        "total_trades": int(stats.get("Total Trades", trade_count)),
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

    # Initialize DataHandler for unified data loading
    data_handler = DataHandler(data_path=args.data_path)

    symbol_list = [s.strip() for s in str(args.symbol).split(",") if s.strip()]
    is_multi_symbol = len(symbol_list) > 1

    # FeatureStore is always enabled (read-first + auto materialize on miss).
    fs_dir = str(getattr(args, "feature_store_dir", "feature_store"))
    raw_layer = str(getattr(args, "feature_store_layer", "AUTO"))
    fs_layer = (
        default_layer_from_config(config_dir)
        if raw_layer.upper() == "AUTO"
        else raw_layer
    )

    def _crop_df_by_env_dates(df_in: pd.DataFrame) -> pd.DataFrame:
        # Optional date cropping to align with available tick data or focus window
        start_override = os.getenv("TRAIN_START_DATE")
        end_override = os.getenv("TRAIN_END_DATE")
        if not (start_override or end_override) or df_in.empty:
            return df_in
        dt_idx = None
        for col in ("datetime", "timestamp", "date"):
            if col in df_in.columns:
                dt_idx = pd.to_datetime(df_in[col])
                break
        if dt_idx is None and isinstance(df_in.index, pd.DatetimeIndex):
            dt_idx = df_in.index
        if dt_idx is None:
            return df_in
        mask = pd.Series(True, index=df_in.index)
        if start_override:
            mask &= dt_idx >= pd.to_datetime(start_override)
        if end_override:
            mask &= dt_idx <= pd.to_datetime(end_override)
        df_out = df_in.loc[mask]
        print(
            f"   ℹ️  Cropped data to [{start_override or '-inf'}, {end_override or '+inf'}], rows={len(df_out)}"
        )
        return df_out

    if not is_multi_symbol:
        df_raw = data_handler.load_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        df_raw = _crop_df_by_env_dates(df_raw)
    else:
        # IMPORTANT: do NOT rely on DataHandler multi-symbol mode because it de-duplicates datetime
        # indices after concat, which would drop rows for other symbols. Load symbols one-by-one.
        raw_parts: list[pd.DataFrame] = []
        for sym in symbol_list:
            df_sym = data_handler.load_ohlcv(symbol=sym, timeframe=args.timeframe)
            df_sym = _crop_df_by_env_dates(df_sym)
            if df_sym is None or df_sym.empty:
                continue
            # Ensure explicit symbol columns for downstream grouping/ticks inference
            df_sym["_symbol"] = sym
            df_sym["symbol"] = sym
            raw_parts.append(df_sym)
        if not raw_parts:
            raise ValueError(f"No data found for symbol(s): {symbol_list}")
        df_raw = pd.concat(raw_parts, axis=0)
        # Keep duplicates (multiple symbols share timestamps); downstream we reset index after features.
        df_raw = df_raw.sort_index()

    # 监控源数据质量
    from src.features.utils.data_monitor import check_source_data_quality

    source_quality = check_source_data_quality(df_raw, args.data_path)

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
            if not is_multi_symbol:
                _ensure_ticks_configured(
                    feature_loader,
                    symbol=args.symbol,
                    data_path=args.data_path,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    requested_features=requested_features,
                )
            else:
                # Configure ticks per symbol (needed for tick-derived features)
                for sym in symbol_list:
                    df_sym = (
                        df_raw[df_raw.get("_symbol", sym) == sym]
                        if "_symbol" in df_raw.columns
                        else df_raw
                    )
                    if df_sym.empty:
                        continue
                    dtc = next(
                        (
                            c
                            for c in ("datetime", "timestamp", "date")
                            if c in df_sym.columns
                        ),
                        None,
                    )
                    if dtc:
                        dts = pd.to_datetime(df_sym[dtc])
                    elif isinstance(df_sym.index, pd.DatetimeIndex):
                        dts = df_sym.index
                    else:
                        continue
                    if len(dts) == 0:
                        continue
                    st = dts.min().strftime("%Y-%m-%d %H:%M:%S")
                    et = dts.max().strftime("%Y-%m-%d %H:%M:%S")
                    _ensure_ticks_configured(
                        feature_loader,
                        symbol=sym,
                        data_path=args.data_path,
                        start_ts=st,
                        end_ts=et,
                        requested_features=requested_features,
                    )
        else:
            raise ValueError(
                "No datetime/timestamp found in dataframe; cannot configure ticks"
            )
    else:
        raise ValueError("Empty dataframe; cannot configure ticks")

    if not is_multi_symbol:
        split_idx = int(len(df_raw) * (1 - args.test_size))
        df_train_raw = df_raw.iloc[:split_idx].copy()
        df_test_raw = df_raw.iloc[split_idx:].copy()
    else:
        # Split per symbol to keep chronology within each asset, then pool.
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        for sym in symbol_list:
            df_sym = (
                df_raw[df_raw["_symbol"] == sym].sort_index()
                if "_symbol" in df_raw.columns
                else df_raw
            )
            if df_sym.empty:
                continue
            split_idx = int(len(df_sym) * (1 - args.test_size))
            train_parts.append(df_sym.iloc[:split_idx].copy())
            test_parts.append(df_sym.iloc[split_idx:].copy())
        df_train_raw = pd.concat(train_parts, axis=0).sort_index()
        df_test_raw = pd.concat(test_parts, axis=0).sort_index()

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

    if not is_multi_symbol:
        df_train_features = run_feature_pipeline(
            df_train_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_config.features,
            fit=True,
            feature_store_dir=fs_dir,
            feature_store_layer=fs_layer,
            feature_store_symbol=str(args.symbol),
            feature_store_timeframe=str(args.timeframe),
        )
        feature_debug_stats_train = (
            getattr(df_train_features, "attrs", {}).get("feature_debug_stats") or {}
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
            feature_store_dir=fs_dir,
            feature_store_layer=fs_layer,
            feature_store_symbol=str(args.symbol),
            feature_store_timeframe=str(args.timeframe),
        )
        feature_debug_stats_test = (
            getattr(df_test_features, "attrs", {}).get("feature_debug_stats") or {}
        )
        print(
            f"   ✅ Feature pipeline (test) done: rows={len(df_test_features)}, cols={len(df_test_features.columns)}\n"
        )
    else:
        # Compute features per symbol (avoids duplicate datetime index issues) then pool.
        train_feat_parts: list[pd.DataFrame] = []
        test_feat_parts: list[pd.DataFrame] = []
        for sym in symbol_list:
            df_tr = df_train_raw[df_train_raw["_symbol"] == sym].sort_index()
            df_te = df_test_raw[df_test_raw["_symbol"] == sym].sort_index()
            if df_tr.empty or df_te.empty:
                continue
            feat_tr = run_feature_pipeline(
                df_tr,
                feature_loader=feature_loader,
                pipeline_cfg=strategy_config.features,
                fit=True,
                feature_store_dir=fs_dir,
                feature_store_layer=fs_layer,
                feature_store_symbol=str(sym),
                feature_store_timeframe=str(args.timeframe),
            )
            feat_te = run_feature_pipeline(
                df_te,
                feature_loader=feature_loader,
                pipeline_cfg=strategy_config.features,
                fit=False,
                feature_store_dir=fs_dir,
                feature_store_layer=fs_layer,
                feature_store_symbol=str(sym),
                feature_store_timeframe=str(args.timeframe),
            )
            # Ensure grouping columns are present post-feature-pipeline
            feat_tr["_symbol"] = sym
            feat_tr["symbol"] = sym
            feat_te["_symbol"] = sym
            feat_te["symbol"] = sym
            if isinstance(feat_tr.index, pd.DatetimeIndex):
                feat_tr["datetime"] = feat_tr.index
            if isinstance(feat_te.index, pd.DatetimeIndex):
                feat_te["datetime"] = feat_te.index
            train_feat_parts.append(feat_tr.reset_index(drop=True))
            test_feat_parts.append(feat_te.reset_index(drop=True))
        df_train_features = pd.concat(train_feat_parts, axis=0, ignore_index=True)
        df_test_features = pd.concat(test_feat_parts, axis=0, ignore_index=True)
        feature_debug_stats_train = {}
        feature_debug_stats_test = {}
        # Stable order for TSCV and backtests
        sort_cols = [
            c for c in ["datetime", "_symbol"] if c in df_train_features.columns
        ]
        if sort_cols:
            df_train_features = df_train_features.sort_values(sort_cols).reset_index(
                drop=True
            )
        sort_cols = [
            c for c in ["datetime", "_symbol"] if c in df_test_features.columns
        ]
        if sort_cols:
            df_test_features = df_test_features.sort_values(sort_cols).reset_index(
                drop=True
            )
        print(
            f"   ✅ Feature pipeline (train/test) pooled: train_rows={len(df_train_features)}, test_rows={len(df_test_features)}, cols={len(df_train_features.columns)}\n"
        )

    feature_cols = determine_feature_columns(
        df_train_features, strategy_config.features
    )
    # Multi-symbol pooled training: optionally include symbol as a categorical feature.
    # Prefer `_symbol` (already configured as categorical in config/feature_column_types.yaml).
    if is_multi_symbol:
        if "_symbol" in df_train_features.columns and "_symbol" not in feature_cols:
            feature_cols = list(feature_cols) + ["_symbol"]
        elif "symbol" in df_train_features.columns and "symbol" not in feature_cols:
            feature_cols = list(feature_cols) + ["symbol"]
    print(f"   ✅ Candidate features: {len(feature_cols)}")

    # Label generation
    label_func = import_callable(
        strategy_config.labels.generator.module,
        strategy_config.labels.generator.function,
    )

    # Label generation
    # NOTE: Some label generators (e.g., *_with_weights) attach `sample_weight` to the input df.
    # We call them on a temporary copy to avoid accidental feature mutation, but we propagate
    # `sample_weight` back if present so training can consume it.
    _train_tmp = df_train_features.copy()
    _test_tmp = df_test_features.copy()
    df_train_features[strategy_config.labels.target_column] = label_func(
        _train_tmp, **strategy_config.labels.generator.params
    )
    df_test_features[strategy_config.labels.target_column] = label_func(
        _test_tmp, **strategy_config.labels.generator.params
    )
    if "sample_weight" in _train_tmp.columns:
        df_train_features["sample_weight"] = _train_tmp["sample_weight"]
    if "sample_weight" in _test_tmp.columns:
        df_test_features["sample_weight"] = _test_tmp["sample_weight"]
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
        numeric_cols = (
            df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )
        if not numeric_cols:
            return
        # 正确区分 inf 和 NaN：只检查真正的 inf/-inf，不包括 NaN
        # 注意：np.isfinite() 对 NaN 也返回 False，所以不能用来检查 inf
        inf_mask = np.isinf(df[numeric_cols])
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

    # ------------------------------------------------------------------
    # Diagnostics snapshot (always persisted to results.json later)
    # - label distribution: catch "label too sparse" / mapping issues
    # - prediction distribution: catch collapsed models / overly strict entry gates
    # - entry/exit counts: provided by backtest payload (we also compute a quick summary)
    # ------------------------------------------------------------------
    def _value_counts_safe(s: pd.Series) -> dict:
        try:
            vc = s.value_counts(dropna=True).to_dict()
            return {str(k): int(v) for k, v in vc.items()}
        except Exception:
            return {}

    diagnostics_payload: dict = {
        "labels": {
            "target_col": None,  # filled after target_col resolved
            "task_type": None,  # filled after task_type resolved
            "train": {
                "n": int(len(df_train_filtered)),
                "value_counts": _value_counts_safe(
                    df_train_filtered[strategy_config.labels.target_column]
                ),
            },
            "test": {
                "n": int(len(df_test_filtered)),
                "value_counts": _value_counts_safe(
                    df_test_filtered[strategy_config.labels.target_column]
                ),
            },
        }
    }
    # Feature compute performance/cache diagnostics (best-effort)
    try:
        diagnostics_payload["features"] = {
            "train": feature_debug_stats_train,
            "test": feature_debug_stats_test,
        }
    except Exception:
        pass
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
    diagnostics_payload["labels"]["target_col"] = str(target_col)
    diagnostics_payload["labels"]["task_type"] = str(task_type)

    # Single-source-of-truth: propagate invert_features from features.yaml into trainer model_params.
    # This keeps training/inference consistent without needing a separate direction config file.
    try:
        inv = getattr(strategy_config.features, "invert_features", None)
        if isinstance(inv, list) and inv:
            mp = trainer_params.get("model_params") or {}
            if isinstance(mp, dict):
                mp = dict(mp)
                mp["invert_features"] = inv
                trainer_params["model_params"] = mp
    except Exception:
        pass

    # Seed plumbing: make `--seed` actually control model RNG (so multi-seed sweeps are meaningful,
    # and same-seed runs are stable). We intentionally override YAML seeds here.
    try:
        seed_int = int(getattr(args, "seed", 42))
        mp0 = trainer_params.get("model_params") or {}
        if isinstance(mp0, dict):
            mp = dict(mp0)
            mt = str(model_type).lower()
            if mt == "lightgbm":
                mp["seed"] = seed_int
                mp["feature_fraction_seed"] = seed_int
                mp["bagging_seed"] = seed_int
                mp["data_random_seed"] = seed_int
                mp["drop_seed"] = seed_int
            elif mt == "xgboost":
                mp["random_state"] = seed_int
                mp["seed"] = seed_int
            elif mt == "catboost":
                mp["random_seed"] = seed_int
            trainer_params["model_params"] = mp
    except Exception:
        # Never fail training due to seed plumbing.
        pass

    print(
        f"\n   🚀 Training model ({model_type}, task={task_type}) "
        f"on {len(df_train_filtered)} samples, {len(feature_cols)} features"
    )
    models, avg_metric, cv_results, used_features, preprocessor = trainer_func(
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

    X_test = preprocessor.transform(df_test_filtered, feature_cols=used_features)
    y_test = df_test_filtered[target_col].values

    print(f"   ▶️ Generating predictions on test set ({len(df_test_filtered)} samples)")
    preds = generate_predictions(
        models=models,
        model_type=model_type,
        task_type=task_type,
        X=X_test,
    )

    # Prediction diagnostics (saved in results.json)
    pred_diag: dict = {"task_type": str(task_type)}
    try:
        if (
            str(task_type).lower() == "multiclass"
            and isinstance(preds, np.ndarray)
            and preds.ndim == 2
        ):
            cls = np.argmax(preds, axis=1)
            pred_diag["shape"] = [int(x) for x in preds.shape]
            pred_diag["class_counts"] = {
                str(k): int(v)
                for k, v in pd.Series(cls).value_counts().to_dict().items()
            }
        else:
            arr = np.asarray(preds).astype(float)
            pred_diag["shape"] = list(arr.shape)
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size:
                s = pd.Series(flat)
                pred_diag["summary"] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "q25": float(s.quantile(0.25)),
                    "q50": float(s.quantile(0.50)),
                    "q75": float(s.quantile(0.75)),
                    "q90": float(s.quantile(0.90)),
                    "q95": float(s.quantile(0.95)),
                    "q99": float(s.quantile(0.99)),
                }
    except Exception:
        pred_diag["error"] = "pred_diag_failed"
    diagnostics_payload["predictions"] = pred_diag

    evaluation_results = evaluate_predictions(
        preds,
        y_test,
        strategy_config.evaluation,
    )

    for metric_name, score in evaluation_results.items():
        print(f"   ✅ {metric_name}: {score:.4f}")

    # Optionally persist minimal artifacts so we can replay backtests quickly
    # without retraining/recomputing features (useful for parameter sweeps like sr_fuse/breakeven).
    try:
        backtest_params = getattr(strategy_config, "backtest", None)
        bt_params = (
            getattr(backtest_params, "params", None) if backtest_params else None
        )
        bt_params = bt_params or {}
        save_artifacts = bool(bt_params.get("save_artifacts", False))
        if save_artifacts:

            price_col = str(bt_params.get("price_col", "close"))
            high_col = str(bt_params.get("high_col", "high"))
            low_col = str(bt_params.get("low_col", "low"))
            atr_col = str(bt_params.get("atr_col", "atr"))
            signal_col = str(bt_params.get("signal_col", "signal"))
            use_signal_direction = bool(bt_params.get("use_signal_direction", False))

            rr_cfg = bt_params.get("rr", {}) or {}
            rr_entry_price_col = rr_cfg.get("entry_price_col", None)

            sr_fuse_cfg = bt_params.get("sr_fuse", {}) or {}
            sr_dist_col = str(sr_fuse_cfg.get("dist_col", "dist_to_nearest_sr"))
            sr_atr_col = str(sr_fuse_cfg.get("atr_col", atr_col))

            needed_cols = {
                price_col,
                high_col,
                low_col,
                atr_col,
                sr_dist_col,
                sr_atr_col,
            }
            if rr_entry_price_col:
                needed_cols.add(str(rr_entry_price_col))
            if use_signal_direction:
                needed_cols.add(signal_col)

            cols_exist = [c for c in needed_cols if c in df_test_filtered.columns]
            df_bt = df_test_filtered[cols_exist].copy()

            # Preserve the datetime index in parquet for exact alignment
            bt_df_path = output_dir / "backtest_df_test.parquet"
            df_bt.to_parquet(bt_df_path)

            bt_pred_path = output_dir / "backtest_preds.npy"
            np.save(bt_pred_path, np.asarray(preds, dtype=float))

            bt_meta_path = output_dir / "backtest_artifacts_meta.json"
            meta = {
                "task_type": task_type,
                "model_type": model_type,
                "n_test_samples": int(len(df_test_filtered)),
                "saved_columns": cols_exist,
                "backtest_params": bt_params,
            }
            with open(bt_meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, default=str)

            print(
                f"   💾 Backtest artifacts saved: {bt_df_path.name}, {bt_pred_path.name}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Failed to save backtest artifacts: {exc}")

    results = {
        "strategy": strategy_config.name,
        "model_type": model_type,
        "task_type": task_type,
        "avg_cv_metric": float(avg_metric),
        "n_features": len(used_features),
        "n_train_samples": len(df_train_filtered),
        "n_test_samples": len(df_test_filtered),
        "evaluation": evaluation_results,
        "diagnostics": diagnostics_payload,
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
    if not is_multi_symbol or "_symbol" not in df_test_filtered.columns:
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
    else:
        # Run per-symbol backtests (pooling assets into one backtest is meaningless).
        bt_by_symbol: dict[str, Any] = {}
        for sym in symbol_list:
            mask = (df_test_filtered["_symbol"] == sym).to_numpy()
            if mask.sum() == 0:
                continue
            df_sym = df_test_filtered.loc[mask].copy()
            preds_sym = np.asarray(preds)[mask]
            bt = run_backtest_with_strategy(
                df_sym,
                preds_sym,
                strategy_config,
                task_type=task_type,
                vol_model=vol_model,
            )
            if bt:
                bt_by_symbol[sym] = bt
        if bt_by_symbol:
            results["backtest_by_symbol"] = bt_by_symbol
            # Also provide an overall summary (equal-weight mean across symbols)
            try:
                rets = [
                    v.get("total_return_pct")
                    for v in bt_by_symbol.values()
                    if v.get("total_return_pct") is not None
                ]
                sharps = [
                    v.get("sharpe")
                    for v in bt_by_symbol.values()
                    if v.get("sharpe") is not None
                ]
                dds = [
                    v.get("max_drawdown_pct")
                    for v in bt_by_symbol.values()
                    if v.get("max_drawdown_pct") is not None
                ]
                trades = [
                    v.get("total_trades")
                    for v in bt_by_symbol.values()
                    if v.get("total_trades") is not None
                ]
                results["backtest"] = {
                    "total_return_pct": float(np.mean(rets)) if rets else None,
                    "sharpe": float(np.mean(sharps)) if sharps else None,
                    "max_drawdown_pct": float(np.max(dds)) if dds else None,
                    "total_trades": int(np.sum(trades)) if trades else None,
                    "aggregate_mode": "multi_symbol_equal_weight_mean_return_sharpe_max_dd_sum_trades",
                }
            except Exception:
                pass
            print(f"   ✅ Backtest completed (per symbol): {list(bt_by_symbol.keys())}")

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

    # Reproducibility: fix RNG seeds as early as possible.
    try:
        import random

        np.random.seed(int(args.seed))
        random.seed(int(args.seed))
    except Exception:
        pass

    # Determinism knobs: best-effort (helps a lot for LightGBM + numpy reductions).
    # Setting these inside the process still affects libraries that read env at runtime.
    if bool(getattr(args, "deterministic", False)):
        # IMPORTANT: override (not setdefault) so repeated runs are consistent even if
        # the parent process exported thread env vars.
        os.environ["MLBOT_DETERMINISTIC"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Optional torch determinism for DL features (best effort; some CUDA ops can still be non-deterministic).
    try:
        import torch  # type: ignore

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
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
