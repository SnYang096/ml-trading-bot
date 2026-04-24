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
from dataclasses import replace
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
from datetime import datetime

from src.data_tools.data_handler import DataHandler
from src.data_tools.tick_loader import list_tick_files, serialize_tick_loader_params
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.feature_store.layer_naming import resolve_layer_name
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.features.cross_symbol.macro_tp_vwap_anchor import (
    ANCHOR_COLUMN,
    apply_macro_tp_vwap_anchor,
    apply_macro_tp_vwap_from_anchor_frame,
    ensure_datetime_column,
    parse_macro_tp_vwap_anchor_config,
)
from src.time_series_model.pipeline.training.label_utils import (
    simulate_rr_exits,
    future_volatility_label,
)
from src.time_series_model.pipeline.training.volatility_model_config import (
    load_volatility_model_config,
    prepare_volatility_model_data,
    get_volatility_model_params,
)

import yaml


# ============================================================
# Stub for deprecated VectorBTBacktest (old backtest class deleted)
# Training pipeline quick-eval now returns None; use backtest_execution_layer.py for proper backtest.
# ============================================================
class VectorBTBacktest:
    """Stub: VectorBTBacktest was removed. Use backtest_execution_layer.py instead."""

    def run(self, **kwargs):
        return None  # Skip quick backtest during training; real backtest done via CLI


# 原始/未归一化列：不传入模型，只用于标签或 backtest
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
    "cvd_roll20",
    "cvd_roll60",
    "cvd_roll288",
}

# 缓存 output_columns 集合（方案 C：基于元数据自动过滤）
_VALID_OUTPUT_COLUMNS: Optional[set] = None
# 缓存 price_unit 列集合（方案 D：基于 output_normalization_map 排除绝对价格特征）
_PRICE_UNIT_COLUMNS: Optional[set] = None

# 归一化后缀（方案 A + C 结合）
NORMALIZED_SUFFIXES = ("_pct", "_rank", "_zscore", "_normalized", "_f")
# 原始特征前缀（需要额外过滤）
RAW_FEATURE_PREFIXES = (
    "cvd_change_",  # cvd_change_1, cvd_change_5, cvd_change_20 (但保留 cvd_change_5_pct)
    "trade_cluster_",  # trade_cluster_* 原始列 (但保留 zscore 版本)
)
# 明确排除的单个列名
RAW_FEATURE_EXACT = {
    "_symbol",
    "macd",
    "macd_signal",
    "macd_histogram",
    "cvd",
    "cvd_normalized",
}


def _is_normalized_feature(col: str) -> bool:
    """判断是否为归一化特征（方案 A）。

    返回 True 表示应该保留，False 表示应该排除。
    """
    # 明确排除的列
    if col in RAW_FEATURE_EXACT:
        return False

    # 检查原始特征前缀
    for prefix in RAW_FEATURE_PREFIXES:
        if col.startswith(prefix):
            # 但如果有归一化后缀，则保留
            if any(col.endswith(suffix) for suffix in NORMALIZED_SUFFIXES):
                return True
            # 或者包含 zscore
            if "zscore" in col:
                return True
            return False

    return True


def _load_valid_output_columns(
    feature_deps_path: str = "config/feature_dependencies.yaml",
) -> set:
    """从 feature_dependencies.yaml 收集所有合法的 output_columns。

    只有在 output_columns 中声明的列才允许进入模型训练。
    原始数据列（如 cvd_change_1、macd 等）不在任何 output_columns 中，自动被排除。
    同时收集 output_normalization_map 中标记为 price_unit/raw/usd 的列，用于方案 D 过滤。
    """
    global _VALID_OUTPUT_COLUMNS, _PRICE_UNIT_COLUMNS
    if _VALID_OUTPUT_COLUMNS is not None:
        return _VALID_OUTPUT_COLUMNS

    try:
        p = Path(feature_deps_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        features = obj.get("features", {}) or {}

        valid_cols = set()
        price_unit_cols = set()
        for feat_name, feat_info in features.items():
            if isinstance(feat_info, dict):
                out_cols = feat_info.get("output_columns") or []
                for c in out_cols:
                    valid_cols.add(str(c))
                # 方案 D：收集 price_unit / raw / usd 特征
                norm_map = (feat_info.get("compute_params") or {}).get(
                    "output_normalization_map"
                ) or {}
                for col_name, norm_type in norm_map.items():
                    if str(norm_type) in {"price_unit", "raw", "usd"}:
                        price_unit_cols.add(str(col_name))

        _VALID_OUTPUT_COLUMNS = valid_cols
        _PRICE_UNIT_COLUMNS = price_unit_cols
        print(
            f"   ℹ️  Loaded {len(valid_cols)} valid output columns from feature_dependencies.yaml"
        )
        if price_unit_cols:
            print(
                f"   ℹ️  Found {len(price_unit_cols)} price_unit columns to exclude: {sorted(price_unit_cols)}"
            )
        return valid_cols
    except Exception as e:
        print(
            f"   ⚠️  Failed to load output_columns from feature_dependencies.yaml: {e}"
        )
        return set()


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
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional crop start date (YYYY-MM-DD). Overrides TRAIN_START_DATE env if provided.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional crop end date (YYYY-MM-DD). Overrides TRAIN_END_DATE env if provided.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help=(
            "Train a final model on ALL available (cropped) data. "
            "Skips the holdout test split/backtest, but still saves ModelArtifact."
        ),
    )
    parser.add_argument(
        "--holdout-start-date",
        type=str,
        default=None,
        help=(
            "Optional explicit holdout start date (YYYY-MM-DD). "
            "If set (with --holdout-end-date), the pipeline will train on data strictly before holdout_start "
            "and test on [holdout_start, holdout_end] instead of using --test-size."
        ),
    )
    parser.add_argument(
        "--holdout-end-date",
        type=str,
        default=None,
        help="Optional explicit holdout end date (YYYY-MM-DD). Requires --holdout-start-date.",
    )
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
        default=None,
        help="FeatureStore layer (dataset id). If not specified, auto-generated from config content.",
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
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Override labels config file path (e.g. config/strategies/bpc/labels_rr_extreme.yaml)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Override features config file path (e.g. config/strategies/bpc/features_gate.yaml)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Only run feature pipeline + label generation, save features_labeled.parquet "
            "(full period), then exit. Skips model training. "
            "Use for prefilter analysis and direction validation before training."
        ),
    )
    parser.add_argument(
        "--archetype-prefilter",
        type=str,
        default=None,
        help=(
            "Path to archetypes/prefilter.yaml. "
            "Applies environment prerequisite rules to filter training data BEFORE model training. "
            "Only rows satisfying ALL rules are kept for train/test. "
            "Example: config/strategies/me/archetypes/prefilter.yaml"
        ),
    )
    parser.add_argument(
        "--skip-gate-shap",
        action="store_true",
        help=(
            "Skip TreeSHAP / SHAP∩Gain / interaction in statistical risk_gate_draft export "
            "(gain-only; faster for turbo / threshold-only iterations)."
        ),
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
    """确定进入模型训练的特征列。

    方案 C：基于 feature_dependencies.yaml 的 output_columns 元数据自动过滤。
    只有在 output_columns 中声明的列才允许进入模型，原始数据列自动排除。
    """
    # YAML-driven input pruning: keep some columns for label/backtest, but never feed them into the model.
    exclude_cols = []
    try:
        exclude_cols = list(getattr(pipeline_cfg, "exclude_columns", []) or [])
    except Exception:
        exclude_cols = []
    exclude_cols = [str(c).strip() for c in exclude_cols if str(c).strip()]

    # 方案 C：加载合法的 output_columns 集合
    valid_output_cols = _load_valid_output_columns()

    if pipeline_cfg.selector:
        selector_func = import_callable(
            pipeline_cfg.selector.module, pipeline_cfg.selector.function
        )
        try:
            cols = selector_func(df, list(df.columns), **pipeline_cfg.selector.params)
        except TypeError:
            cols = selector_func(df, **pipeline_cfg.selector.params)

        if exclude_cols:
            cols = [c for c in (cols or []) if c not in set(exclude_cols)]
        return cols

    cols = [
        col
        for col in df.columns
        if col not in BASE_DATA_COLUMNS
        and not col.startswith(("signal", "binary_signal"))
    ]

    # 方案 C：只保留在 output_columns 中声明的列
    if valid_output_cols:
        before_count = len(cols)
        cols = [c for c in cols if c in valid_output_cols]
        filtered_count = before_count - len(cols)
        if filtered_count > 0:
            print(f"   ℹ️  Auto-filtered {filtered_count} columns not in output_columns")

    # 方案 A：进一步过滤非归一化的原始特征
    before_count = len(cols)
    cols = [c for c in cols if _is_normalized_feature(c)]
    filtered_count = before_count - len(cols)
    if filtered_count > 0:
        print(f"   ℹ️  Auto-filtered {filtered_count} raw features (not normalized)")

    # 方案 D：排除 output_normalization_map 中标记为 price_unit/raw/usd 的绝对价格特征
    if _PRICE_UNIT_COLUMNS:
        before_count = len(cols)
        cols = [c for c in cols if c not in _PRICE_UNIT_COLUMNS]
        filtered_count = before_count - len(cols)
        if filtered_count > 0:
            print(
                f"   ℹ️  Auto-filtered {filtered_count} price_unit features (not cross-asset comparable)"
            )

    if exclude_cols:
        cols = [c for c in cols if c not in set(exclude_cols)]
    return cols


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
            # DEPRECATED (no-op):
            # This filter used to require *all* feature columns to be non-null, which is usually
            # not what we want in modern ML pipelines:
            # - LightGBM / XGBoost can handle NaNs natively
            # - It can collapse the dataset when any feature has partial NaNs (false-negative for search)
            #
            # Keep as a no-op for backward compatibility with existing YAML configs.
            # If you need strict behavior for a specific model, implement it explicitly in the model
            # preprocessor (e.g., imputation) or add a targeted filter on required columns only.
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


def generate_training_html_report(
    results: Dict[str, Any],
    output_dir: Path,
    strategy_name: str,
    args: argparse.Namespace,
) -> Optional[Path]:
    """
    Generate an HTML report for training results.

    Args:
        results: Training results dictionary
        output_dir: Directory to save the report
        strategy_name: Name of the strategy
        args: Command line arguments

    Returns:
        Path to the generated HTML file, or None if generation fails
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_{timestamp}_report.html"
        report_path = output_dir / filename

        # Extract key metrics
        model_type = results.get("model_type", "unknown")
        task_type = results.get("task_type", "unknown")
        avg_cv_metric = results.get("avg_cv_metric")
        n_features = results.get("n_features", 0)
        n_train = results.get("n_train_samples", 0)
        n_test = results.get("n_test_samples", 0)

        # Backtest metrics
        backtest = results.get("backtest") or {}
        sharpe = backtest.get("sharpe")
        total_return = backtest.get("total_return_pct")
        max_dd = backtest.get("max_drawdown_pct")
        win_rate = backtest.get("win_rate")
        total_trades = backtest.get("total_trades")

        # Feature importance (top 20)
        feature_importance = results.get("feature_importance", {})
        top_features = list(feature_importance.items())[:20]

        # Per-symbol backtest
        backtest_by_symbol = results.get("backtest_by_symbol", {})

        # Build HTML content
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"  <title>{strategy_name} Training Report</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }",
            "    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }",
            "    h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }",
            "    h2 { color: #555; margin-top: 30px; }",
            "    .meta-info { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
            "    .meta-info span { margin-right: 20px; }",
            "    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }",
            "    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }",
            "    .metric-card.positive { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }",
            "    .metric-card.negative { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }",
            "    .metric-card.neutral { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }",
            "    .metric-value { font-size: 28px; font-weight: bold; }",
            "    .metric-label { font-size: 12px; opacity: 0.9; margin-top: 5px; }",
            "    table { width: 100%; border-collapse: collapse; margin: 15px 0; }",
            "    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
            "    th { background: #f8f9fa; font-weight: 600; }",
            "    tr:hover { background: #f5f5f5; }",
            "    .importance-bar { background: #4CAF50; height: 20px; border-radius: 3px; }",
            "    .warning { color: #f57c00; }",
            "    .error { color: #d32f2f; }",
            "    .success { color: #388e3c; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <div class='container'>",
            f"    <h1>📊 {strategy_name} Training Report</h1>",
            "    <div class='meta-info'>",
            f"      <span><strong>Model:</strong> {model_type}</span>",
            f"      <span><strong>Task:</strong> {task_type}</span>",
            f"      <span><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>",
            "    </div>",
        ]

        # Training parameters
        html_parts.extend(
            [
                "    <h2>🔧 Training Parameters</h2>",
                "    <table>",
                f"      <tr><td>Symbol(s)</td><td>{getattr(args, 'symbol', 'N/A')}</td></tr>",
                f"      <tr><td>Timeframe</td><td>{getattr(args, 'timeframe', 'N/A')}</td></tr>",
                f"      <tr><td>Start Date</td><td>{getattr(args, 'start_date', 'N/A')}</td></tr>",
                f"      <tr><td>End Date</td><td>{getattr(args, 'end_date', 'N/A')}</td></tr>",
                f"      <tr><td>Holdout Start</td><td>{getattr(args, 'holdout_start_date', 'N/A')}</td></tr>",
                f"      <tr><td>Holdout End</td><td>{getattr(args, 'holdout_end_date', 'N/A')}</td></tr>",
                f"      <tr><td>Seed</td><td>{getattr(args, 'seed', 'N/A')}</td></tr>",
                "    </table>",
            ]
        )

        # Sample statistics
        html_parts.extend(
            [
                "    <h2>📈 Sample Statistics</h2>",
                "    <div class='metrics-grid'>",
                f"      <div class='metric-card neutral'><div class='metric-value'>{n_train:,}</div><div class='metric-label'>Training Samples</div></div>",
                f"      <div class='metric-card neutral'><div class='metric-value'>{n_test:,}</div><div class='metric-label'>Test Samples</div></div>",
                f"      <div class='metric-card neutral'><div class='metric-value'>{n_features}</div><div class='metric-label'>Features Used</div></div>",
            ]
        )
        if avg_cv_metric is not None:
            html_parts.append(
                f"      <div class='metric-card neutral'><div class='metric-value'>{avg_cv_metric:.4f}</div><div class='metric-label'>Avg CV Metric</div></div>"
            )
        html_parts.append("    </div>")

        # Backtest results
        if backtest and sharpe is not None:
            sharpe_class = (
                "positive"
                if sharpe > 0.5
                else ("negative" if sharpe < 0 else "neutral")
            )
            return_class = "positive" if (total_return or 0) > 0 else "negative"
            dd_class = "negative" if (max_dd or 0) > 20 else "neutral"

            html_parts.extend(
                [
                    "    <h2>💰 Backtest Results</h2>",
                    "    <div class='metrics-grid'>",
                    f"      <div class='metric-card {sharpe_class}'><div class='metric-value'>{sharpe:.2f}</div><div class='metric-label'>Sharpe Ratio</div></div>",
                ]
            )
            if total_return is not None:
                html_parts.append(
                    f"      <div class='metric-card {return_class}'><div class='metric-value'>{total_return:.2f}%</div><div class='metric-label'>Total Return</div></div>"
                )
            if max_dd is not None:
                html_parts.append(
                    f"      <div class='metric-card {dd_class}'><div class='metric-value'>{max_dd:.2f}%</div><div class='metric-label'>Max Drawdown</div></div>"
                )
            if win_rate is not None:
                html_parts.append(
                    f"      <div class='metric-card neutral'><div class='metric-value'>{win_rate:.1f}%</div><div class='metric-label'>Win Rate</div></div>"
                )
            if total_trades is not None:
                html_parts.append(
                    f"      <div class='metric-card neutral'><div class='metric-value'>{total_trades}</div><div class='metric-label'>Total Trades</div></div>"
                )
            html_parts.append("    </div>")
        elif backtest is None:
            html_parts.extend(
                [
                    "    <h2>💰 Backtest Results</h2>",
                    "    <p class='warning'>⚠️ Backtest skipped (train-all mode or no holdout test set)</p>",
                ]
            )
        else:
            html_parts.extend(
                [
                    "    <h2>💰 Backtest Results</h2>",
                    f"    <p class='error'>❌ Backtest failed or skipped: {backtest.get('note', 'unknown reason')}</p>",
                ]
            )

        # Per-symbol backtest
        if backtest_by_symbol:
            html_parts.extend(
                [
                    "    <h2>📊 Per-Symbol Backtest</h2>",
                    "    <table>",
                    "      <tr><th>Symbol</th><th>Sharpe</th><th>Return %</th><th>Max DD %</th><th>Trades</th></tr>",
                ]
            )
            for sym, bt in backtest_by_symbol.items():
                s = bt.get("sharpe", "N/A")
                r = bt.get("total_return_pct", "N/A")
                d = bt.get("max_drawdown_pct", "N/A")
                t = bt.get("total_trades", "N/A")
                s_str = f"{s:.2f}" if isinstance(s, (int, float)) else str(s)
                r_str = f"{r:.2f}" if isinstance(r, (int, float)) else str(r)
                d_str = f"{d:.2f}" if isinstance(d, (int, float)) else str(d)
                html_parts.append(
                    f"      <tr><td>{sym}</td><td>{s_str}</td><td>{r_str}</td><td>{d_str}</td><td>{t}</td></tr>"
                )
            html_parts.append("    </table>")

        # Failure Analysis
        failure_analysis = results.get("failure_analysis", {})
        if failure_analysis:
            global_rr = failure_analysis.get("global_failure_rr_extreme", 0)
            global_no_opp = failure_analysis.get("global_failure_no_opportunity", 0)
            selected_rr = failure_analysis.get("selected_failure_rr_extreme", 0)
            selected_no_opp = failure_analysis.get("selected_failure_no_opportunity", 0)
            lift_rr = failure_analysis.get("lift_rr_extreme", 0)
            lift_no_opp = failure_analysis.get("lift_no_opportunity", 0)
            n_selected = failure_analysis.get("n_selected", 0)
            n_total = failure_analysis.get("n_total", 0)

            # 判断 lift 的正负性
            rr_class = "positive" if lift_rr < 1.0 else "negative"  # lift < 1 表示改善
            no_opp_class = "positive" if lift_no_opp < 1.0 else "negative"

            html_parts.extend(
                [
                    "    <h2>🔬 Failure Sub-label Analysis</h2>",
                    "    <p style='color: #666; margin-bottom: 15px;'>比较模型选中的交易 (top 30%) 与所有交易的失败率。Lift < 1.0 表示模型有效降低了失败率。</p>",
                    "    <div class='metrics-grid'>",
                    f"      <div class='metric-card neutral'><div class='metric-value'>{n_selected:,} / {n_total:,}</div><div class='metric-label'>Selected / Total Trades</div></div>",
                    f"      <div class='metric-card {rr_class}'><div class='metric-value'>{lift_rr:.2f}x</div><div class='metric-label'>RR Extreme Lift</div></div>",
                    f"      <div class='metric-card {no_opp_class}'><div class='metric-value'>{lift_no_opp:.2f}x</div><div class='metric-label'>No Opportunity Lift</div></div>",
                    "    </div>",
                    "    <table>",
                    "      <tr><th>Failure Type</th><th>Global Rate</th><th>Selected Rate</th><th>Lift</th></tr>",
                    f"      <tr><td>RR Extreme (踩大坑)</td><td>{global_rr:.1%}</td><td>{selected_rr:.1%}</td><td>{lift_rr:.2f}x</td></tr>",
                    f"      <tr><td>No Opportunity (入场即反)</td><td>{global_no_opp:.1%}</td><td>{selected_no_opp:.1%}</td><td>{lift_no_opp:.2f}x</td></tr>",
                    "    </table>",
                ]
            )

            # Lift vs Coverage Curve
            lift_curve = failure_analysis.get("lift_curve", [])
            if lift_curve:
                html_parts.extend(
                    [
                        "    <h3 style='margin-top: 25px; color: #666;'>📈 Lift vs Coverage Curve</h3>",
                        "    <p style='color: #666; margin-bottom: 15px;'>不同阈值（Top 20%-80%）下的 Lift 和覆盖率权衡。评估 Gate 的 veto 能力上限。</p>",
                        "    <table>",
                        "      <tr><th>Percentile</th><th>Coverage</th><th>n_selected</th><th>RR Extreme Lift</th><th>No Opportunity Lift</th></tr>",
                    ]
                )
                for curve_point in lift_curve:
                    percentile = curve_point.get("percentile", 0)
                    coverage = curve_point.get("coverage", 0)
                    n_sel = curve_point.get("n_selected", 0)
                    lift_rr = curve_point.get("lift_rr_extreme", 0)
                    lift_no_opp = curve_point.get("lift_no_opportunity", 0)

                    # 计算这行的颜色类（越低越好）
                    rr_class = (
                        "success"
                        if lift_rr < 0.9
                        else ("warning" if lift_rr < 1.0 else "error")
                    )
                    no_opp_class = (
                        "success"
                        if lift_no_opp < 0.7
                        else ("warning" if lift_no_opp < 1.0 else "error")
                    )

                    html_parts.append(
                        f"      <tr>"
                        f"<td>Top {100-percentile}% (p{percentile})</td>"
                        f"<td>{coverage:.1%}</td>"
                        f"<td>{n_sel:,}</td>"
                        f"<td class='{rr_class}'><strong>{lift_rr:.2f}x</strong></td>"
                        f"<td class='{no_opp_class}'><strong>{lift_no_opp:.2f}x</strong></td>"
                        f"</tr>"
                    )
                html_parts.append("    </table>")

                # 添加 SVG 折线图
                # 准备数据点
                chart_width = 800
                chart_height = 400
                padding = 60
                plot_width = chart_width - 2 * padding
                plot_height = chart_height - 2 * padding

                # 提取数据
                coverages = [p["coverage"] * 100 for p in lift_curve]  # 转为百分比
                lift_rr_values = [p["lift_rr_extreme"] for p in lift_curve]
                lift_no_opp_values = [p["lift_no_opportunity"] for p in lift_curve]

                # 计算坐标转换函数（coverage: 0-100%, lift: 0-max）
                max_lift = (
                    max(max(lift_rr_values), max(lift_no_opp_values)) * 1.1
                )  # 留10%余量

                def x_coord(coverage_pct):
                    return padding + (coverage_pct / 100.0) * plot_width

                def y_coord(lift_val):
                    return chart_height - padding - (lift_val / max_lift) * plot_height

                # 生成 RR Extreme 折线路径
                rr_path_parts = []
                for i, (cov, lift) in enumerate(zip(coverages, lift_rr_values)):
                    x = x_coord(cov)
                    y = y_coord(lift)
                    if i == 0:
                        rr_path_parts.append(f"M {x:.1f},{y:.1f}")
                    else:
                        rr_path_parts.append(f"L {x:.1f},{y:.1f}")
                rr_path = " ".join(rr_path_parts)

                # 生成 No Opportunity 折线路径
                no_opp_path_parts = []
                for i, (cov, lift) in enumerate(zip(coverages, lift_no_opp_values)):
                    x = x_coord(cov)
                    y = y_coord(lift)
                    if i == 0:
                        no_opp_path_parts.append(f"M {x:.1f},{y:.1f}")
                    else:
                        no_opp_path_parts.append(f"L {x:.1f},{y:.1f}")
                no_opp_path = " ".join(no_opp_path_parts)

                # 1.0 基准线
                baseline_y = y_coord(1.0)

                # 生成网格线（Y轴：lift 0.5, 0.75, 1.0, 1.25...）
                y_grid_lines = []
                y_grid_step = 0.25
                y_val = 0
                while y_val <= max_lift:
                    y_pos = y_coord(y_val)
                    y_grid_lines.append(
                        f"    <line x1='{padding}' y1='{y_pos:.1f}' x2='{chart_width - padding}' y2='{y_pos:.1f}' stroke='#e0e0e0' stroke-width='1' stroke-dasharray='2,2'/>"
                    )
                    y_grid_lines.append(
                        f"    <text x='{padding - 10}' y='{y_pos:.1f}' text-anchor='end' font-size='12' fill='#666'>{y_val:.2f}</text>"
                    )
                    y_val += y_grid_step

                # 生成网格线（X轴：coverage 0%, 20%, 40%, 60%, 80%, 100%）
                x_grid_lines = []
                for cov_pct in [0, 20, 40, 60, 80, 100]:
                    x_pos = x_coord(cov_pct)
                    x_grid_lines.append(
                        f"    <line x1='{x_pos:.1f}' y1='{padding}' x2='{x_pos:.1f}' y2='{chart_height - padding}' stroke='#e0e0e0' stroke-width='1' stroke-dasharray='2,2'/>"
                    )
                    x_grid_lines.append(
                        f"    <text x='{x_pos:.1f}' y='{chart_height - padding + 20}' text-anchor='middle' font-size='12' fill='#666'>{cov_pct}%</text>"
                    )

                html_parts.extend(
                    [
                        "    <div style='margin: 30px 0; text-align: center;'>",
                        f"      <svg width='{chart_width}' height='{chart_height}' style='border: 1px solid #ddd; border-radius: 5px; background: white;'>",
                        "        <!-- 网格线 -->",
                    ]
                    + y_grid_lines
                    + x_grid_lines
                    + [
                        "        <!-- 1.0 基准线 -->",
                        f"        <line x1='{padding}' y1='{baseline_y:.1f}' x2='{chart_width - padding}' y2='{baseline_y:.1f}' stroke='#ff9800' stroke-width='2' stroke-dasharray='5,5'/>",
                        f"        <text x='{chart_width - padding - 50}' y='{baseline_y - 5:.1f}' font-size='12' fill='#ff9800'>Lift = 1.0 (无效)</text>",
                        "        <!-- No Opportunity Lift 曲线 -->",
                        f"        <path d='{no_opp_path}' stroke='#2196F3' stroke-width='3' fill='none'/>",
                        "        <!-- RR Extreme Lift 曲线 -->",
                        f"        <path d='{rr_path}' stroke='#4CAF50' stroke-width='3' fill='none'/>",
                        "        <!-- 数据点 -->",
                    ]
                )

                # 添加数据点（圆圈）
                for cov, lift_rr, lift_no_opp in zip(
                    coverages, lift_rr_values, lift_no_opp_values
                ):
                    x = x_coord(cov)
                    y_rr = y_coord(lift_rr)
                    y_no_opp = y_coord(lift_no_opp)
                    html_parts.append(
                        f"        <circle cx='{x:.1f}' cy='{y_rr:.1f}' r='4' fill='#4CAF50' stroke='white' stroke-width='2'/>"
                    )
                    html_parts.append(
                        f"        <circle cx='{x:.1f}' cy='{y_no_opp:.1f}' r='4' fill='#2196F3' stroke='white' stroke-width='2'/>"
                    )

                html_parts.extend(
                    [
                        "        <!-- 坐标轴标签 -->",
                        f"        <text x='{chart_width / 2}' y='{chart_height - 10}' text-anchor='middle' font-size='14' font-weight='bold' fill='#333'>Coverage (%)</text>",
                        f"        <text x='15' y='{chart_height / 2}' text-anchor='middle' font-size='14' font-weight='bold' fill='#333' transform='rotate(-90 15 {chart_height / 2})'>Lift (改善倍数)</text>",
                        "        <!-- 图例 -->",
                        f"        <rect x='{chart_width - 200}' y='20' width='15' height='15' fill='#4CAF50'/>",
                        f"        <text x='{chart_width - 180}' y='32' font-size='12' fill='#333'>RR Extreme Lift</text>",
                        f"        <rect x='{chart_width - 200}' y='40' width='15' height='15' fill='#2196F3'/>",
                        f"        <text x='{chart_width - 180}' y='52' font-size='12' fill='#333'>No Opportunity Lift</text>",
                        "      </svg>",
                        "    </div>",
                    ]
                )

            # Per-symbol failure analysis
            by_symbol = failure_analysis.get("by_symbol", {})
            if by_symbol:
                html_parts.extend(
                    [
                        "    <h3 style='margin-top: 25px; color: #666;'>🔍 Per-Symbol Failure Analysis</h3>",
                        "    <table>",
                        "      <tr><th>Symbol</th><th>Global RR</th><th>Selected RR</th><th>RR Lift</th><th>Global NoOpp</th><th>Selected NoOpp</th><th>NoOpp Lift</th><th>n_selected</th></tr>",
                    ]
                )
                for sym, stats in by_symbol.items():
                    sym_global_rr = stats.get("global_failure_rr_extreme", 0)
                    sym_sel_rr = stats.get("selected_failure_rr_extreme", 0)
                    sym_lift_rr = stats.get("lift_rr_extreme", 0)
                    sym_global_no_opp = stats.get("global_failure_no_opportunity", 0)
                    sym_sel_no_opp = stats.get("selected_failure_no_opportunity", 0)
                    sym_lift_no_opp = stats.get("lift_no_opportunity", 0)
                    sym_n_sel = stats.get("n_selected", 0)

                    html_parts.append(
                        f"      <tr><td><strong>{sym}</strong></td>"
                        f"<td>{sym_global_rr:.1%}</td><td>{sym_sel_rr:.1%}</td><td>{sym_lift_rr:.2f}x</td>"
                        f"<td>{sym_global_no_opp:.1%}</td><td>{sym_sel_no_opp:.1%}</td><td>{sym_lift_no_opp:.2f}x</td>"
                        f"<td>{sym_n_sel}</td></tr>"
                    )
                html_parts.append("    </table>")

        # Feature importance
        if top_features:
            max_importance = top_features[0][1] if top_features else 1
            html_parts.extend(
                [
                    "    <h2>🎯 Top 20 Feature Importance</h2>",
                    "    <table>",
                    "      <tr><th>Rank</th><th>Feature</th><th>Importance</th><th></th></tr>",
                ]
            )
            for i, (feat, imp) in enumerate(top_features, 1):
                bar_width = (
                    int((imp / max_importance) * 100) if max_importance > 0 else 0
                )
                html_parts.append(
                    f"      <tr><td>{i}</td><td><code>{feat}</code></td><td>{imp:.2f}</td>"
                    f"<td><div class='importance-bar' style='width:{bar_width}%'></div></td></tr>"
                )
            html_parts.append("    </table>")

        # Return Tree KPI Section (if available)
        return_tree_kpi = results.get("return_tree_kpi", {})
        if return_tree_kpi and return_tree_kpi.get("spearman_corr") is not None:
            spearman = return_tree_kpi.get("spearman_corr", 0)
            monotonicity = return_tree_kpi.get("quantile_monotonicity")
            q5_q1_spread = return_tree_kpi.get("q5_q1_spread")
            top10_ratio = return_tree_kpi.get("top10_importance_ratio")

            # Status color classes
            spearman_class = (
                "positive"
                if spearman >= 0.15
                else ("neutral" if spearman >= 0 else "negative")
            )
            mono_class = (
                "positive"
                if monotonicity and monotonicity >= 0.8
                else ("neutral" if monotonicity and monotonicity >= 0.5 else "negative")
            )
            spread_class = (
                "positive"
                if q5_q1_spread and q5_q1_spread >= 0.3
                else ("neutral" if q5_q1_spread and q5_q1_spread >= 0.1 else "negative")
            )
            ratio_class = (
                "positive" if top10_ratio and 0.3 <= top10_ratio <= 0.6 else "neutral"
            )

            html_parts.extend(
                [
                    "    <h2>🎯 Return Tree KPI</h2>",
                    "    <p style='color: #666; margin-bottom: 15px;'>Return Tree 的核心目标不是 lift（只要不变差即可），而是排序能力和语义集中度。</p>",
                    "    <div class='metrics-grid'>",
                    f"      <div class='metric-card {spearman_class}'><div class='metric-value'>{spearman:.3f}</div><div class='metric-label'>Spearman Corr (≥0.15)</div></div>",
                ]
            )
            if monotonicity is not None:
                html_parts.append(
                    f"      <div class='metric-card {mono_class}'><div class='metric-value'>{monotonicity:.0%}</div><div class='metric-label'>分位单调性 (≥80%)</div></div>"
                )
            if q5_q1_spread is not None:
                html_parts.append(
                    f"      <div class='metric-card {spread_class}'><div class='metric-value'>{q5_q1_spread:.3f}R</div><div class='metric-label'>Q5-Q1 Spread (≥0.3R)</div></div>"
                )
            if top10_ratio is not None:
                html_parts.append(
                    f"      <div class='metric-card {ratio_class}'><div class='metric-value'>{top10_ratio:.0%}</div><div class='metric-label'>Top10 重要性 (30-60%)</div></div>"
                )
            html_parts.append("    </div>")

            # KPI 评估表格
            html_parts.extend(
                [
                    "    <h3 style='margin-top: 25px; color: #666;'>📊 KPI 达标评估</h3>",
                    "    <table>",
                    "      <tr><th>KPI</th><th>当前值</th><th>目标</th><th>状态</th></tr>",
                ]
            )
            # Spearman
            spearman_status = (
                "✅ 达标"
                if spearman >= 0.15
                else ("⚠️ 接近" if spearman >= 0 else "❌ 未达标")
            )
            spearman_status_class = (
                "success"
                if spearman >= 0.15
                else ("warning" if spearman >= 0 else "error")
            )
            html_parts.append(
                f"      <tr><td>Spearman 相关系数</td><td><strong>{spearman:.3f}</strong></td><td>≥ 0.15</td><td class='{spearman_status_class}'>{spearman_status}</td></tr>"
            )
            # Monotonicity
            if monotonicity is not None:
                mono_status = (
                    "✅ 达标"
                    if monotonicity >= 0.8
                    else ("⚠️ 接近" if monotonicity >= 0.5 else "❌ 未达标")
                )
                mono_status_class = (
                    "success"
                    if monotonicity >= 0.8
                    else ("warning" if monotonicity >= 0.5 else "error")
                )
                html_parts.append(
                    f"      <tr><td>分位单调性</td><td><strong>{monotonicity:.0%}</strong></td><td>≥ 80%</td><td class='{mono_status_class}'>{mono_status}</td></tr>"
                )
            # Q5-Q1 Spread
            if q5_q1_spread is not None:
                spread_status = (
                    "✅ 达标"
                    if q5_q1_spread >= 0.3
                    else ("⚠️ 接近" if q5_q1_spread >= 0.1 else "❌ 未达标")
                )
                spread_status_class = (
                    "success"
                    if q5_q1_spread >= 0.3
                    else ("warning" if q5_q1_spread >= 0.1 else "error")
                )
                html_parts.append(
                    f"      <tr><td>Q5-Q1 Spread</td><td><strong>{q5_q1_spread:.3f}R</strong></td><td>≥ 0.3R</td><td class='{spread_status_class}'>{spread_status}</td></tr>"
                )
            # Top10 ratio
            if top10_ratio is not None:
                ratio_status = "✅ 达标" if 0.3 <= top10_ratio <= 0.6 else "⚠️ 偏离"
                ratio_status_class = (
                    "success" if 0.3 <= top10_ratio <= 0.6 else "warning"
                )
                html_parts.append(
                    f"      <tr><td>Top10 重要性占比</td><td><strong>{top10_ratio:.0%}</strong></td><td>30%-60%</td><td class='{ratio_status_class}'>{ratio_status}</td></tr>"
                )
            html_parts.append("    </table>")

            # 分位组 RR 均值详情
            quantile_means = return_tree_kpi.get("quantile_means", {})
            if quantile_means:
                html_parts.extend(
                    [
                        "    <h3 style='margin-top: 25px; color: #666;'>📊 分位组 RR 均值</h3>",
                        "    <table>",
                        "      <tr><th>分位</th><th>RR 均值</th></tr>",
                    ]
                )
                for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                    q_val = quantile_means.get(q, 0)
                    html_parts.append(
                        f"      <tr><td>{q}</td><td>{q_val:+.3f}R</td></tr>"
                    )
                html_parts.append("    </table>")

            # Top5 特征
            top5_features = return_tree_kpi.get("top5_features", [])
            if top5_features:
                html_parts.extend(
                    [
                        "    <h3 style='margin-top: 25px; color: #666;'>🎯 Top5 特征重要性</h3>",
                        "    <table>",
                        "      <tr><th>Rank</th><th>特征</th><th>Importance</th></tr>",
                    ]
                )
                for i, (feat, imp) in enumerate(top5_features, 1):
                    html_parts.append(
                        f"      <tr><td>{i}</td><td><code>{feat}</code></td><td>{imp:.2f}</td></tr>"
                    )
                html_parts.append("    </table>")

            # 跨符号稳定性
            symbol_spearman = return_tree_kpi.get("symbol_spearman", {})
            symbol_consistency = return_tree_kpi.get("symbol_consistency", 0)
            if symbol_spearman:
                cons_class = "positive" if symbol_consistency >= 0.6 else "neutral"
                html_parts.extend(
                    [
                        "    <h3 style='margin-top: 25px; color: #666;'>🔗 跨符号稳定性</h3>",
                        f"    <p style='margin-bottom: 10px;'>符号一致性: <strong class='{cons_class}'>{symbol_consistency:.0%}</strong> (target: ≥60%)</p>",
                        "    <table>",
                        "      <tr><th>Symbol</th><th>Spearman</th><th>状态</th></tr>",
                    ]
                )
                for sym, corr in symbol_spearman.items():
                    status = "✅" if corr > 0 else "❌"
                    status_class = "success" if corr > 0 else "error"
                    html_parts.append(
                        f"      <tr><td>{sym}</td><td>{corr:.3f}</td><td class='{status_class}'>{status}</td></tr>"
                    )
                html_parts.append("    </table>")

        # Footer
        html_parts.extend(
            [
                "    <hr style='margin-top: 40px; border: none; border-top: 1px solid #eee;'>",
                f"    <p style='color: #999; font-size: 12px;'>Generated by mlbot train • {output_dir}</p>",
                "  </div>",
                "</body>",
                "</html>",
            ]
        )

        # Write HTML file
        html_content = "\n".join(html_parts)
        report_path.write_text(html_content, encoding="utf-8")
        return report_path

    except Exception as exc:
        print(f"   ⚠️  Failed to generate HTML report: {exc}")
        return None


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
    if df.empty:
        return df

    # First, handle duplicate columns in df
    if df.columns.duplicated().any():
        # Keep first occurrence of duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

    dedup_cols = list(dict.fromkeys(feature_cols))
    result = df.copy()
    # Only numeric columns can contain +/-inf in a meaningful way.
    # Some feature pipelines may include non-numeric columns (e.g. DTW match labels).
    # Filter to columns that actually exist in result
    existing_cols = [c for c in dedup_cols if c in result.columns]
    if not existing_cols:
        return result
    numeric_cols = (
        result[existing_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    if not numeric_cols:
        return result

    # First, detect rows that contain any inf/-inf (before replacing).
    # We intentionally do NOT drop NaN rows here.
    has_inf = np.isinf(result[numeric_cols]).any(axis=1)
    finite_mask = ~has_inf
    dropped = len(result) - finite_mask.sum()
    result = result[finite_mask].copy()  # explicit copy to avoid SettingWithCopyWarning

    # Safety: ensure no inf remains after filtering (convert to NaN).
    # Use numpy-based replacement to avoid pandas column name issues
    if len(result) > 0 and numeric_cols:
        for col in numeric_cols:
            if col in result.columns:
                arr = result[col].values
                arr = np.where(np.isinf(arr), np.nan, arr)
                result[col] = arr

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
        elif metric_type == "regression_mae" or metric_type == "mae":
            # Mean Absolute Error for regression tasks
            valid_mask = ~(np.isnan(preds) & ~np.isnan(y_true))
            if valid_mask.sum() > 0:
                score = float(np.mean(np.abs(preds[valid_mask] - y_true[valid_mask])))
            else:
                score = 0.0
        elif metric_type == "regression_mse" or metric_type == "mse":
            # Mean Squared Error for regression tasks
            valid_mask = ~(np.isnan(preds) & ~np.isnan(y_true))
            if valid_mask.sum() > 0:
                score = float(np.mean((preds[valid_mask] - y_true[valid_mask]) ** 2))
            else:
                score = 0.0
        elif metric_type == "regression_rmse" or metric_type == "rmse":
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
                    from src.time_series_model.pipeline.training.label_utils import (
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
    loader = StrategyConfigLoader(
        config_dir,
        labels_override=args.labels,
        features_override=getattr(args, "features", None),
    )
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
    raw_layer = getattr(args, "feature_store_layer", None)
    # Auto-generate layer name if not specified (unified handling for both CLI and direct script calls)
    fs_layer = resolve_layer_name(raw_layer, config_dir)

    def _crop_df_by_env_dates(df_in: pd.DataFrame) -> pd.DataFrame:
        # Optional date cropping to align with available tick data or focus window
        start_override = getattr(args, "start_date", None) or os.getenv(
            "TRAIN_START_DATE"
        )
        end_override = getattr(args, "end_date", None) or os.getenv("TRAIN_END_DATE")
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

    if bool(getattr(args, "train_all", False)) and (
        getattr(args, "holdout_start_date", None)
        or getattr(args, "holdout_end_date", None)
    ):
        raise ValueError(
            "--train-all cannot be used together with --holdout-start-date/--holdout-end-date"
        )

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

    holdout_start = getattr(args, "holdout_start_date", None)
    holdout_end = getattr(args, "holdout_end_date", None)
    if holdout_end and not holdout_start:
        raise ValueError("--holdout-end-date requires --holdout-start-date")

    def _dt_index(df_in: pd.DataFrame) -> pd.DatetimeIndex | None:
        for col in ("datetime", "timestamp", "date"):
            if col in df_in.columns:
                return pd.to_datetime(df_in[col])
        if isinstance(df_in.index, pd.DatetimeIndex):
            return df_in.index
        return None

    if holdout_start:
        hs = pd.to_datetime(holdout_start)
        he = pd.to_datetime(holdout_end) if holdout_end else hs
        if he < hs:
            raise ValueError("--holdout-end-date must be >= --holdout-start-date")

        if not is_multi_symbol:
            dts = _dt_index(df_raw)
            if dts is None:
                raise ValueError(
                    "Cannot apply holdout split: no datetime/timestamp found in dataframe"
                )
            mask_test = (dts >= hs) & (dts <= he)
            df_test_raw = df_raw.loc[mask_test].copy()
            df_train_raw = df_raw.loc[dts < hs].copy()
        else:
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
                dts = _dt_index(df_sym)
                if dts is None:
                    continue
                mask_test = (dts >= hs) & (dts <= he)
                test_parts.append(df_sym.loc[mask_test].copy())
                train_parts.append(df_sym.loc[dts < hs].copy())
            df_train_raw = (
                pd.concat(train_parts, axis=0).sort_index()
                if train_parts
                else df_raw.iloc[:0].copy()
            )
            df_test_raw = (
                pd.concat(test_parts, axis=0).sort_index()
                if test_parts
                else df_raw.iloc[:0].copy()
            )
        print(
            f"   🧪 Holdout split enabled: test=[{hs.date()}, {he.date()}], train=< {hs.date()}"
        )
    else:
        if not is_multi_symbol:
            if bool(getattr(args, "train_all", False)):
                df_train_raw = df_raw.copy()
                df_test_raw = df_raw.iloc[:0].copy()
            else:
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
                if bool(getattr(args, "train_all", False)):
                    train_parts.append(df_sym.copy())
                    test_parts.append(df_sym.iloc[:0].copy())
                else:
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

    # 💡 动态注入 freq 参数：从 strategy meta.yaml 的 timeframe 读取
    # 注意：strategy_config.meta 直接对应 meta.yaml 的 strategy 节点内容
    meta_timeframe = (strategy_config.meta or {}).get("timeframe")

    if meta_timeframe:
        print(f"   ℹ️  Detected strategy.timeframe from meta.yaml: {meta_timeframe}")
        # 注入到需要 freq 参数的特征配置中
        feature_deps = feature_loader.feature_deps.get("features", {})
        freq_required_features = [
            "vpin_base_aligned_features_f",
            "trade_cluster_base_aligned_features_f",
            "trade_cluster_semantic_scores_f",
        ]
        injected_count = 0
        for feat_name in freq_required_features:
            if feat_name in feature_deps:
                compute_params = feature_deps[feat_name].setdefault(
                    "compute_params", {}
                )
                if "freq" not in compute_params:
                    compute_params["freq"] = meta_timeframe
                    print(f"   ✅ Injected freq='{meta_timeframe}' to {feat_name}")
                    injected_count += 1
                else:
                    # 已有配置，不覆盖（保留用户显式配置的优先级）
                    print(
                        f"   ℹ️  {feat_name} already has freq='{compute_params['freq']}', skipping"
                    )
        if injected_count == 0 and freq_required_features:
            print(
                f"   ℹ️  No freq injection needed (features not in requested list or already configured)"
            )
    else:
        print(
            "   ⚠️  No strategy.timeframe found in meta.yaml, freq parameter will not be injected"
        )

    requested = list(strategy_config.features.requested_features)
    inv = getattr(strategy_config.features, "invert_features", None) or []
    effective_requested = requested + [c for c in inv if c not in requested]
    pipeline_cfg_effective = replace(
        strategy_config.features, requested_features=effective_requested
    )
    print(
        f"\n   ▶️ Feature pipeline (train) start: {len(effective_requested)} requested features (incl. invert)"
    )

    if not is_multi_symbol:
        df_train_features = run_feature_pipeline(
            df_train_raw,
            feature_loader=feature_loader,
            pipeline_cfg=pipeline_cfg_effective,
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
            pipeline_cfg=pipeline_cfg_effective,
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
            # Skip only if train set is empty; test can be empty in train-all mode
            if df_tr.empty:
                continue
            feat_tr = run_feature_pipeline(
                df_tr,
                feature_loader=feature_loader,
                pipeline_cfg=pipeline_cfg_effective,
                fit=True,
                feature_store_dir=fs_dir,
                feature_store_layer=fs_layer,
                feature_store_symbol=str(sym),
                feature_store_timeframe=str(args.timeframe),
            )
            feat_te = run_feature_pipeline(
                df_te,
                feature_loader=feature_loader,
                pipeline_cfg=pipeline_cfg_effective,
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

    # --- macro_tp_vwap_1200_position: optional cross-symbol anchor (default BTCUSDT) ---
    _meta_full: Dict[str, Any] = {}
    try:
        _mp = strategy_config.path / "meta.yaml"
        if _mp.exists():
            _meta_full = yaml.safe_load(_mp.read_text(encoding="utf-8")) or {}
    except Exception as _me:
        print(f"   ⚠️  macro_tp_vwap_anchor: could not read meta.yaml: {_me}")
    _anchor_en, _anchor_sym = parse_macro_tp_vwap_anchor_config(
        meta_strategy=strategy_config.meta,
        meta_yaml_full=_meta_full,
    )
    _sym_col = "symbol" if "symbol" in df_train_features.columns else "_symbol"
    if _anchor_en and ANCHOR_COLUMN in df_train_features.columns:
        if is_multi_symbol:
            if (
                _sym_col in df_train_features.columns
                and "datetime" in df_train_features.columns
            ):
                df_train_features = apply_macro_tp_vwap_anchor(
                    df_train_features,
                    anchor_symbol=_anchor_sym,
                    enabled=True,
                    symbol_col=_sym_col,
                    time_col="datetime",
                )
                df_test_features = apply_macro_tp_vwap_anchor(
                    df_test_features,
                    anchor_symbol=_anchor_sym,
                    enabled=True,
                    symbol_col=_sym_col,
                    time_col="datetime",
                )
                print(
                    f"   ✅ macro_tp_vwap_anchor: pooled overlay (anchor={_anchor_sym})"
                )
            else:
                print(
                    "   ⚠️  macro_tp_vwap_anchor: missing symbol/datetime columns, skip"
                )
        else:
            _main_sym = str(args.symbol).strip().upper()
            if _main_sym == str(_anchor_sym).strip().upper():
                print(
                    f"   ℹ️  macro_tp_vwap_anchor: symbol is anchor {_anchor_sym}, native VWAP"
                )
            else:
                df_train_features = ensure_datetime_column(df_train_features)
                df_test_features = ensure_datetime_column(df_test_features)
                _combo = pd.concat([df_train_raw, df_test_raw], axis=0)
                _dta_combo = _dt_index(_combo)
                df_ar_full = pd.DataFrame()
                try:
                    df_ar_full = data_handler.load_ohlcv(
                        symbol=_anchor_sym, timeframe=args.timeframe
                    )
                    df_ar_full = _crop_df_by_env_dates(df_ar_full)
                except Exception as _ae:
                    print(
                        f"   ⚠️  macro_tp_vwap_anchor: failed to load {_anchor_sym}: {_ae}"
                    )
                if df_ar_full is None or df_ar_full.empty or _dta_combo is None:
                    print(
                        "   ⚠️  macro_tp_vwap_anchor: no anchor OHLCV or no main dates, skip"
                    )
                else:
                    lo, hi = _dta_combo.min(), _dta_combo.max()
                    dta = _dt_index(df_ar_full)
                    if dta is None:
                        print(
                            "   ⚠️  macro_tp_vwap_anchor: anchor has no datetime, skip"
                        )
                    else:
                        _mask = (dta >= lo) & (dta <= hi)
                        atr_slice = df_ar_full.loc[_mask].copy()
                        if atr_slice.empty:
                            print(
                                f"   ⚠️  macro_tp_vwap_anchor: {_anchor_sym} empty in "
                                f"train+test window, skip"
                            )
                        else:
                            f_anchor = run_feature_pipeline(
                                atr_slice,
                                feature_loader=feature_loader,
                                pipeline_cfg=pipeline_cfg_effective,
                                fit=True,
                                feature_store_dir=fs_dir,
                                feature_store_layer=fs_layer,
                                feature_store_symbol=str(_anchor_sym),
                                feature_store_timeframe=str(args.timeframe),
                            )
                            f_anchor = ensure_datetime_column(f_anchor)
                            df_train_features = apply_macro_tp_vwap_from_anchor_frame(
                                df_train_features,
                                f_anchor,
                                time_col="datetime",
                            )
                            df_test_features = apply_macro_tp_vwap_from_anchor_frame(
                                df_test_features,
                                f_anchor,
                                time_col="datetime",
                            )
                            print(
                                f"   ✅ macro_tp_vwap_anchor: single-symbol overlay "
                                f"(anchor={_anchor_sym})"
                            )

    feature_cols = determine_feature_columns(
        df_train_features, strategy_config.features
    )

    # NOTE: Previously we auto-included `_symbol` for multi-symbol training,
    # but this causes data leakage (model learns symbol identity instead of features).
    # If needed, explicitly configure symbol as a feature in the strategy config.
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
    # Propagate forward_rr from label generation (needed for --prepare-only export)
    if "forward_rr" in _train_tmp.columns:
        df_train_features["forward_rr"] = _train_tmp["forward_rr"]
    if "forward_rr" in _test_tmp.columns:
        df_test_features["forward_rr"] = _test_tmp["forward_rr"]
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
        if df.empty:
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
        f"   \u2705 Valid samples after filtering - "
        f"Train: {len(df_train_filtered)}, Test: {len(df_test_filtered)}"
    )

    # ------------------------------------------------------------------
    # --archetype-prefilter: 读取 archetypes/prefilter.yaml 过滤训练数据
    # 语义: archetype 成立的前置条件，不满足的样本不应参与训练
    # ------------------------------------------------------------------
    prefilter_path = getattr(args, "archetype_prefilter", None)
    if prefilter_path:
        import yaml as _yaml
        import operator as _op

        pf_path = Path(prefilter_path)
        if not pf_path.exists():
            print(f"\u274c --archetype-prefilter 文件不存在: {pf_path}")
            return

        with open(pf_path, "r") as f:
            pf_cfg = _yaml.safe_load(f)

        pf_rules = pf_cfg.get("rules", [])
        if not pf_rules:
            print(f"\u26a0\ufe0f  --archetype-prefilter: {pf_path} 中没有 rules，跳过")
        else:
            _OPS = {
                ">=": _op.ge,
                ">": _op.gt,
                "<=": _op.le,
                "<": _op.lt,
                "==": _op.eq,
                "!=": _op.ne,
            }
            print(f"\n\U0001f6e1\ufe0f  Archetype Prefilter: {pf_path}")
            print(
                f"   \u89c4\u5219\u6570: {len(pf_rules)}, \u8bad\u7ec3\u524d Train={len(df_train_filtered)}, Test={len(df_test_filtered)}"
            )

            def _apply_single_rule(df, feat, op_str, val, _OPS):
                """Apply a single prefilter rule, return (mask, ok)."""
                op_func = _OPS.get(op_str)
                if op_func is None:
                    print(f"   ❌ 未知 operator: {op_str}，跳过规则 {feat}")
                    return None, False
                if feat not in df.columns:
                    print(f"   ❌ 特征列 '{feat}' 不存在，跳过规则")
                    return None, False
                return op_func(df[feat], val), True

            for rule in pf_rules:
                # ── any_of OR 组: 任一子规则的 PASS 条件成立即通过 ──
                # 语义: prefilter.yaml operator = PASS 方向 (正向选择数据)
                #        any_of = "至少一条 pass" = OR of pass_i
                if "any_of" in rule:
                    sub_rules = rule["any_of"]
                    rationale = rule.get("rationale", "")
                    n_before_train = len(df_train_filtered)
                    n_before_test = len(df_test_filtered)

                    or_pass_train = pd.Series(False, index=df_train_filtered.index)
                    or_pass_test = pd.Series(False, index=df_test_filtered.index)
                    sub_descs = []
                    for sub in sub_rules:
                        sf, sop, sv = sub["feature"], sub["operator"], sub["value"]
                        pass_tr, ok_tr = _apply_single_rule(
                            df_train_filtered, sf, sop, sv, _OPS
                        )
                        pass_te, ok_te = _apply_single_rule(
                            df_test_filtered, sf, sop, sv, _OPS
                        )
                        if ok_tr and pass_tr is not None:
                            or_pass_train |= pass_tr  # PASS 条件直接 OR
                        if ok_te and pass_te is not None:
                            or_pass_test |= pass_te
                        sub_descs.append(f"{sf}{sop}{sv}")

                    df_train_filtered = df_train_filtered[or_pass_train].copy()
                    df_test_filtered = df_test_filtered[or_pass_test].copy()
                    desc = " OR ".join(sub_descs)
                    print(
                        f"   ✅ any_of({desc}): "
                        f"Train {n_before_train}→{len(df_train_filtered)} "
                        f"(-{n_before_train - len(df_train_filtered)}), "
                        f"Test {n_before_test}→{len(df_test_filtered)} "
                        f"(-{n_before_test - len(df_test_filtered)})"
                        + (f"  [{rationale}]" if rationale else "")
                    )
                    continue

                # ── 普通 AND 规则: PASS 条件成立才保留 ──
                # 语义: prefilter.yaml operator = PASS 方向 (正向选择数据)
                #        keep = op_func(feat, val)
                feat = rule["feature"]
                op_str = rule["operator"]
                val = rule["value"]
                op_func = _OPS.get(op_str)
                if op_func is None:
                    print(f"   ❌ 未知 operator: {op_str}，跳过规则 {feat}")
                    continue
                if feat not in df_train_filtered.columns:
                    print(f"   ❌ 特征列 '{feat}' 不存在，跳过规则")
                    continue

                n_before_train = len(df_train_filtered)
                n_before_test = len(df_test_filtered)
                df_train_filtered = df_train_filtered[
                    op_func(df_train_filtered[feat], val)
                ].copy()
                df_test_filtered = df_test_filtered[
                    op_func(df_test_filtered[feat], val)
                ].copy()
                rationale = rule.get("rationale", "")
                print(
                    f"   ✅ {feat} {op_str} {val}: "
                    f"Train {n_before_train}→{len(df_train_filtered)} "
                    f"(-{n_before_train - len(df_train_filtered)}), "
                    f"Test {n_before_test}→{len(df_test_filtered)} "
                    f"(-{n_before_test - len(df_test_filtered)})"
                    + (f"  [{rationale}]" if rationale else "")
                )

            total_remain = len(df_train_filtered) + len(df_test_filtered)
            print(
                f"   \u21b3 Prefilter \u540e\u603b\u6837\u672c: {total_remain:,} (Train={len(df_train_filtered):,}, Test={len(df_test_filtered):,})"
            )

            # \u6570\u636e\u91cf < 1080 \u5fc5\u987b\u62a5\u9519\u7ec8\u6b62
            if len(df_train_filtered) < 1080:
                print(
                    f"\u274c Prefilter \u540e Train \u6837\u672c\u91cf {len(df_train_filtered)} < 1080\uff0c\u7edf\u8ba1\u4e0d\u53ef\u4fe1\uff0c\u7ec8\u6b62"
                )
                return
            if len(df_test_filtered) < 1080:
                print(
                    f"\u26a0\ufe0f  Prefilter \u540e Test \u6837\u672c\u91cf {len(df_test_filtered)} < 1080\uff0c\u7edf\u8ba1\u53ef\u4fe1\u5ea6\u4f4e"
                )

    # ------------------------------------------------------------------
    # --prepare-only: 导出 features_labeled.parquet 并提前退出
    # ------------------------------------------------------------------
    if getattr(args, "prepare_only", False):
        import pyarrow as pa
        import pyarrow.parquet as pq

        # 合并 train + test 为全周期数据
        df_all = pd.concat([df_train_filtered, df_test_filtered], axis=0)
        df_all = df_all.sort_index()

        # 保留: 特征列 + 标签列 + 元数据列
        target_col = strategy_config.labels.target_column
        meta_cols = [
            c
            for c in [
                "timestamp",
                "datetime",
                "date",
                "symbol",
                "_symbol",
                "forward_rr",
                target_col,
            ]
            if c in df_all.columns
        ]
        keep_cols = list(dict.fromkeys(meta_cols + feature_cols))  # 去重保序
        # 也保留 direction 相关列
        for c in df_all.columns:
            if "direction" in c.lower() and c not in keep_cols:
                keep_cols.append(c)
        keep_cols = [c for c in keep_cols if c in df_all.columns]

        df_save = df_all[keep_cols]
        out_file = output_dir / "features_labeled.parquet"
        table = pa.Table.from_pandas(df_save)
        pq.write_table(table, out_file)

        print(f"\n{'='*80}")
        print(f"\u2705 --prepare-only: 导出完成")
        print(f"   文件: {out_file}")
        print(
            f"   行数: {len(df_save):,} (train {len(df_train_filtered):,} + test {len(df_test_filtered):,})"
        )
        print(f"   列数: {len(keep_cols)} (特征 {len(feature_cols)} + 元数据)")
        print(f"   标签: {target_col}")
        print(f"\n用法示例:")
        print(f"   # Prefilter 分析")
        print(f"   python scripts/analyze_archetype_feature_stratification.py \\")
        print(f"     --logs {out_file} --strategy {strategy_config.name} \\")
        print(
            f"     --config config/strategies/{strategy_config.name}/prefilter.yaml --select-recent 6"
        )
        print(f"\n   # Direction 验证")
        print(f"   python scripts/direction_strict_validation.py \\")
        print(f"     --logs {out_file} --strategy {strategy_config.name}")
        print(f"{'='*80}")
        return

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
        # IMPORTANT:
        # feature-group-search expects each run to emit exactly one results.json.
        # When a candidate collapses the train set to empty (e.g. label too sparse after filters),
        # we treat it as an invalid candidate but still write a placeholder results.json so the
        # search loop can continue.
        try:
            # Infer basic metadata without training
            trainer_params = dict(strategy_config.model.trainer.params or {})
            # Labels config target_column takes priority (supports --labels override)
            target_col = strategy_config.labels.target_column
            model_type = str(trainer_params.get("model_type", "unknown"))
            task_type = str(trainer_params.get("task_type", "unknown"))
        except Exception:
            target_col = getattr(strategy_config.labels, "target_column", "target")
            model_type = "unknown"
            task_type = "unknown"

        results = {
            "strategy": strategy_config.name,
            "model_type": model_type,
            "task_type": task_type,
            "avg_cv_metric": None,
            "n_features": int(len(feature_cols)),
            "n_train_samples": int(len(df_train_filtered)),
            "n_test_samples": int(len(df_test_filtered)),
            "evaluation": {},
            "diagnostics": diagnostics_payload
            | {
                "skip": {
                    "skipped": True,
                    "reason": "insufficient_train_samples_after_filtering",
                    "min_required": 50,
                    "target_col": str(target_col),
                }
            },
            "backtest": {
                "total_return_pct": 0.0,
                "sharpe": -999.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "skipped": True,
                "reason": "insufficient_train_samples_after_filtering",
            },
        }

        try:
            output_cfg = strategy_config.model.output
            if output_cfg.get("save_results", True):
                filename = output_cfg.get("filename", "results.json")
                results_file = output_dir / filename
                with open(results_file, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, indent=2, default=str)
                print(f"   💾 Results saved to {results_file}")

                # Generate HTML training report (输出到 output_dir)
                html_report_path = generate_training_html_report(
                    results=results,
                    output_dir=output_dir,
                    strategy_name=strategy_config.name,
                    args=args,
                )
                if html_report_path:
                    print(f"   📄 HTML report saved to {html_report_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  Failed to save placeholder results.json: {exc}")
        return

    trainer_func = import_callable(
        strategy_config.model.trainer.module,
        strategy_config.model.trainer.function,
    )
    trainer_params = dict(strategy_config.model.trainer.params)
    # Labels config target_column takes priority (supports --labels override)
    trainer_params.pop("target_col", None)  # Remove model.yaml target_col if present
    target_col = strategy_config.labels.target_column

    # Apply model_hints from labels config (allows labels.yaml to override model.yaml)
    # This enables e.g. labels_return_tree.yaml to specify task_type=regression
    model_hints = getattr(strategy_config.labels, "model_hints", None)
    if model_hints and isinstance(model_hints, dict):
        # Separate top-level trainer params from model_params
        top_level_keys = {"task_type", "model_type", "n_splits", "tscv_gap", "use_gpu"}
        model_param_keys = {
            "objective",
            "metric",
            "max_depth",
            "min_data_in_leaf",
            "n_estimators",
            "num_leaves",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "seed",
        }

        hints_applied = []
        model_params_hints = {}
        for key, value in model_hints.items():
            if value is not None:
                if key in top_level_keys:
                    trainer_params[key] = value
                    hints_applied.append(key)
                elif key in model_param_keys:
                    model_params_hints[key] = value
                    hints_applied.append(f"model_params.{key}")

        # Merge model_params hints into existing model_params
        if model_params_hints:
            mp = trainer_params.get("model_params") or {}
            if isinstance(mp, dict):
                mp = dict(mp)
                mp.update(model_params_hints)
                trainer_params["model_params"] = mp

        if hints_applied:
            print(
                f"   \U0001f4cc Applied model_hints from labels config: {hints_applied}"
            )
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

    # If test set collapses to empty after filters/feature NaN trimming, we must not hard-fail.
    # This happens often when a candidate feature is all-NaN on the test window (e.g. missing ticks).
    # For feature-group-search we treat this as an invalid candidate (very low score), but keep producing
    # results.json so the search loop can continue.
    if len(df_test_filtered) == 0:
        print(
            "   ⚠️  Test set is empty after filtering; skipping prediction/eval/backtest."
        )
        preds = np.asarray([], dtype=float)
        diagnostics_payload["predictions"] = {
            "task_type": str(task_type),
            "skipped": True,
            "reason": "empty_test_after_filtering",
        }
        evaluation_results = {}
    else:
        X_test = preprocessor.transform(df_test_filtered, feature_cols=used_features)
        y_test = df_test_filtered[target_col].values

        print(
            f"   ▶️ Generating predictions on test set ({len(df_test_filtered)} samples)"
        )
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

    # ========================================
    # Extract column-level feature importance
    # ========================================
    try:
        model = models[0] if isinstance(models, list) else models
        feature_importance = {}
        if hasattr(model, "feature_importance"):
            # LightGBM
            importances = model.feature_importance(importance_type="gain")
            feature_importance = {
                feat: float(imp) for feat, imp in zip(used_features, importances)
            }
        elif hasattr(model, "feature_importances_"):
            # XGBoost, sklearn tree models
            importances = model.feature_importances_
            feature_importance = {
                feat: float(imp) for feat, imp in zip(used_features, importances)
            }
        if feature_importance:
            # Sort by importance (descending)
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            results["feature_importance"] = sorted_importance
            print(
                f"   📊 Feature importance extracted ({len(sorted_importance)} columns)"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Failed to extract feature importance: {exc}")

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
    if bool(getattr(args, "train_all", False)):
        # Final training mode: no holdout test; do not emit placeholder sharpe=-999.
        results["backtest"] = None
        results["backtest_note"] = "train_all_no_holdout_test"
        print("   ℹ️  Backtest skipped (train-all mode; no holdout test set).")
    elif len(df_test_filtered) == 0:
        # Hard guard: produce a deterministic "invalid candidate" backtest payload.
        # This keeps feature-group-search running without crashing.
        results["backtest"] = {
            "total_return_pct": 0.0,
            "sharpe": -999.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "note": "empty_test_after_filtering",
        }
        print("   ⚠️  Backtest skipped (empty test). Using placeholder sharpe=-999.")
    elif not is_multi_symbol or "_symbol" not in df_test_filtered.columns:
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

        # 保存 predictions.parquet （用于 analyze gate-residual 等分析工具）
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa

            # 构建 predictions DataFrame
            pred_df = df_test_filtered.copy()

            # 确保 timestamp 列存在（可能在 index 中）
            if "timestamp" not in pred_df.columns:
                if isinstance(pred_df.index, pd.DatetimeIndex):
                    pred_df["timestamp"] = pred_df.index
                elif pred_df.index.name == "timestamp":
                    pred_df = pred_df.reset_index()
                elif "datetime" in pred_df.columns:
                    pred_df["timestamp"] = pred_df["datetime"]

            pred_df["pred"] = preds
            pred_df["split"] = "holdout"  # 标记为 holdout/test 集

            # 保留必需列（用于 failure 计算和 gated backtest）
            required_cols = [
                "timestamp",
                "close",
                "high",
                "low",
                "open",
                "volume",
                "atr",
                "pred",
                "split",
            ]

            # 添加标签列（用于 gated backtest 计算 Sharpe）
            target_col = strategy_config.labels.target_column
            if target_col in pred_df.columns:
                required_cols.append(target_col)
            # 常见收益列
            for rr_col in ["forward_rr", "success_no_rr_extreme", "ret_mean"]:
                if rr_col in pred_df.columns and rr_col not in required_cols:
                    required_cols.append(rr_col)

            keep_cols = [c for c in required_cols if c in pred_df.columns]

            if "_symbol" in pred_df.columns:
                keep_cols.insert(1, "_symbol")  # 多币种时包含symbol

            # 添加所有特征列（用于后续分析）
            for col in used_features:
                if col not in keep_cols and col in pred_df.columns:
                    keep_cols.append(col)

            # 过滤存在的列
            keep_cols = [c for c in keep_cols if c in pred_df.columns]
            pred_df_save = pred_df[keep_cols]

            # 保存为 parquet
            pred_file = output_dir / "predictions.parquet"
            table = pa.Table.from_pandas(pred_df_save)
            pq.write_table(table, pred_file)
            print(
                f"   💾 Predictions saved to {pred_file} ({len(pred_df_save):,} rows, {len(keep_cols)} columns)"
            )

        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  Failed to save predictions.parquet: {exc}")

        # ========== Failure Sub-label Analysis ==========
        # Analyze failure distribution in model-selected vs unselected trades
        try:
            from src.time_series_model.strategies.labels.failure_first_label import (
                compute_failure_subtypes,
            )

            print(f"\n   📊 Failure Sub-label Analysis...")

            # Use test set for analysis
            analysis_df = df_test_filtered.copy()

            # Get model predictions
            X_test = analysis_df[used_features].values
            preds_list = []
            for model in models:
                if model is None:
                    continue
                try:
                    pred = model.predict(X_test)
                    preds_list.append(pred)
                except Exception:
                    pass

            if preds_list:
                preds = np.mean(preds_list, axis=0)

                # 📊 扫描多个阈值，评估 lift vs 覆盖率
                percentile_thresholds = [
                    80,
                    70,
                    60,
                    50,
                    40,
                    30,
                    20,
                ]  # top 20%, 30%, ..., 80%
                lift_curve_data = []

                print(f"\n      📈 Lift vs Coverage Curve:")
                print(
                    f"      {'Percentile':>12s} | {'Coverage':>10s} | {'n_selected':>12s} | {'RR Lift':>10s} | {'NoOpp Lift':>12s}"
                )
                print(f"      {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")

                # Compute failure subtypes
                direction = str(
                    getattr(strategy_config.labels.generator, "params", {}).get(
                        "direction", "long"
                    )
                )
                horizon = int(
                    getattr(strategy_config.labels.generator, "params", {}).get(
                        "horizon", 50
                    )
                )

                # 多币种支持：按 symbol 分别计算 failure
                if is_multi_symbol and "_symbol" in analysis_df.columns:
                    # 按币种分别计算
                    failure_by_symbol = {}
                    for sym in symbol_list:
                        sym_mask = analysis_df["_symbol"] == sym
                        if sym_mask.sum() == 0:
                            continue

                        sym_df = analysis_df[sym_mask].copy()
                        sym_failure_df = compute_failure_subtypes(
                            df=sym_df,
                            direction=direction,
                            horizon=horizon,
                        )
                        # 保持原始索引对齐
                        sym_failure_df.index = analysis_df[sym_mask].index
                        failure_by_symbol[sym] = sym_failure_df

                    # 合并所有币种的 failure 结果
                    if failure_by_symbol:
                        failure_df = pd.concat(
                            list(failure_by_symbol.values()), sort=False
                        ).sort_index()
                    else:
                        failure_df = None
                else:
                    # 单币种场景
                    failure_df = compute_failure_subtypes(
                        df=analysis_df,
                        direction=direction,
                        horizon=horizon,
                    )

                if failure_df is None:
                    raise ValueError("Failed to compute failure subtypes")

                # 添加 _symbol 列到 failure_df（用于后续按币种分组）
                if is_multi_symbol and "_symbol" in analysis_df.columns:
                    failure_df["_symbol"] = analysis_df.loc[
                        failure_df.index, "_symbol"
                    ].values

                # 计算 global baseline
                valid_mask = failure_df["failure_any"].notna()
                failure_valid = failure_df[valid_mask]

                if len(failure_valid) == 0:
                    print(f"      ⚠️  No valid failure data")
                else:
                    global_rr_extreme = (
                        failure_valid["failure_rr_extreme"] == 1
                    ).mean()
                    global_no_opp = (
                        failure_valid["failure_no_opportunity"] == 1
                    ).mean()

                    # 🔄 循环扫描多个阈值
                    for percentile in percentile_thresholds:
                        entry_threshold = np.percentile(preds, percentile)
                        selected_mask = preds >= entry_threshold

                        # 计算该阈值下的 lift
                        failure_df_copy = failure_df.copy()
                        failure_df_copy["selected"] = selected_mask
                        failure_valid_copy = failure_df_copy[valid_mask]

                        selected_df = failure_valid_copy[failure_valid_copy["selected"]]
                        coverage = (
                            len(selected_df) / len(failure_valid_copy)
                            if len(failure_valid_copy) > 0
                            else 0
                        )

                        if len(selected_df) > 0:
                            selected_rr = (
                                selected_df["failure_rr_extreme"] == 1
                            ).mean()
                            selected_no_opp = (
                                selected_df["failure_no_opportunity"] == 1
                            ).mean()
                            lift_rr = (
                                selected_rr / global_rr_extreme
                                if global_rr_extreme > 0
                                else 0
                            )
                            lift_no_opp = (
                                selected_no_opp / global_no_opp
                                if global_no_opp > 0
                                else 0
                            )

                            print(
                                f"      Top {100-percentile:2d}% (p{percentile:2d}) | {coverage:9.1%} | {len(selected_df):12,} | {lift_rr:9.2f}x | {lift_no_opp:11.2f}x"
                            )

                            lift_curve_data.append(
                                {
                                    "percentile": percentile,
                                    "coverage": float(coverage),
                                    "n_selected": int(len(selected_df)),
                                    "lift_rr_extreme": float(lift_rr),
                                    "lift_no_opportunity": float(lift_no_opp),
                                }
                            )

                    # 使用 top 30% (p70) 作为默认报告阈值
                    entry_threshold = np.percentile(preds, 70)
                    selected_mask = preds >= entry_threshold

                    # 重新计算 top 30% 的详细统计（用于主报告）
                    failure_df["selected"] = selected_mask
                    failure_valid = failure_df[valid_mask]

                    # Selected trades failure rates
                    selected_df = failure_valid[failure_valid["selected"]]
                    unselected_df = failure_valid[~failure_valid["selected"]]

                    if len(selected_df) > 0:
                        selected_rr_extreme = (
                            selected_df["failure_rr_extreme"] == 1
                        ).mean()
                        selected_no_opp = (
                            selected_df["failure_no_opportunity"] == 1
                        ).mean()

                        # Calculate lifts
                        lift_rr = (
                            selected_rr_extreme / global_rr_extreme
                            if global_rr_extreme > 0
                            else 0
                        )
                        lift_no_opp = (
                            selected_no_opp / global_no_opp if global_no_opp > 0 else 0
                        )

                        print(f"      ────────────────────────────────────────")
                        print(f"      🌍 Global Failure Rate (baseline):")
                        print(
                            f"         failure_rr_extreme:     {global_rr_extreme:.1%}  (踩大坑)"
                        )
                        print(
                            f"         failure_no_opportunity: {global_no_opp:.1%}  (入场即反)"
                        )
                        print(f"      ────────────────────────────────────────")
                        print(
                            f"      ✅ Selected Trades (top 30%, n={len(selected_df)}):"
                        )
                        print(
                            f"         failure_rr_extreme:     {selected_rr_extreme:.1%}  (lift={lift_rr:.2f}x)"
                        )
                        print(
                            f"         failure_no_opportunity: {selected_no_opp:.1%}  (lift={lift_no_opp:.2f}x)"
                        )

                        if len(unselected_df) > 0:
                            unselected_rr = (
                                unselected_df["failure_rr_extreme"] == 1
                            ).mean()
                            reduction = (
                                1 - selected_rr_extreme / unselected_rr
                                if unselected_rr > 0
                                else 0
                            )
                            print(f"      ────────────────────────────────────────")
                            print(f"      🎯 Reduction vs unselected: {reduction:+.1%}")
                            if reduction < 0:
                                print(f"      ⚠️  警告: 模型选中的 trades 失败率更高!")

                        # 🔍 多币种场景：按币种分别统计
                        failure_by_symbol_stats = {}
                        if is_multi_symbol and "_symbol" in failure_valid.columns:
                            print(f"\n      🔍 Per-Symbol Failure Analysis:")

                            for sym in symbol_list:
                                sym_mask = failure_valid["_symbol"] == sym
                                if sym_mask.sum() == 0:
                                    continue

                                sym_failure = failure_valid[sym_mask]
                                sym_selected = sym_failure[sym_failure["selected"]]

                                if len(sym_failure) > 0:
                                    sym_global_rr = (
                                        sym_failure["failure_rr_extreme"] == 1
                                    ).mean()
                                    sym_global_no_opp = (
                                        sym_failure["failure_no_opportunity"] == 1
                                    ).mean()

                                    if len(sym_selected) > 0:
                                        sym_sel_rr = (
                                            sym_selected["failure_rr_extreme"] == 1
                                        ).mean()
                                        sym_sel_no_opp = (
                                            sym_selected["failure_no_opportunity"] == 1
                                        ).mean()
                                        sym_lift_rr = (
                                            sym_sel_rr / sym_global_rr
                                            if sym_global_rr > 0
                                            else 0
                                        )
                                        sym_lift_no_opp = (
                                            sym_sel_no_opp / sym_global_no_opp
                                            if sym_global_no_opp > 0
                                            else 0
                                        )

                                        print(
                                            f"         {sym:10s}: RR={sym_sel_rr:.1%} (lift={sym_lift_rr:.2f}x), NoOpp={sym_sel_no_opp:.1%} (lift={sym_lift_no_opp:.2f}x), n={len(sym_selected)}"
                                        )

                                        failure_by_symbol_stats[sym] = {
                                            "global_failure_rr_extreme": float(
                                                sym_global_rr
                                            ),
                                            "global_failure_no_opportunity": float(
                                                sym_global_no_opp
                                            ),
                                            "selected_failure_rr_extreme": float(
                                                sym_sel_rr
                                            ),
                                            "selected_failure_no_opportunity": float(
                                                sym_sel_no_opp
                                            ),
                                            "lift_rr_extreme": float(sym_lift_rr),
                                            "lift_no_opportunity": float(
                                                sym_lift_no_opp
                                            ),
                                            "n_selected": int(len(sym_selected)),
                                            "n_total": int(len(sym_failure)),
                                        }

                        # Save to results
                        results["failure_analysis"] = {
                            "global_failure_rr_extreme": float(global_rr_extreme),
                            "global_failure_no_opportunity": float(global_no_opp),
                            "selected_failure_rr_extreme": float(selected_rr_extreme),
                            "selected_failure_no_opportunity": float(selected_no_opp),
                            "lift_rr_extreme": float(lift_rr),
                            "lift_no_opportunity": float(lift_no_opp),
                            "n_selected": int(len(selected_df)),
                            "n_total": int(len(failure_valid)),
                            "lift_curve": lift_curve_data,  # 添加 lift vs coverage 曲线
                        }

                        # 添加按币种的统计
                        if failure_by_symbol_stats:
                            results["failure_analysis"][
                                "by_symbol"
                            ] = failure_by_symbol_stats
        except Exception as exc:
            print(f"   ⚠️  Failure analysis skipped: {exc}")

        # ========== Return Tree KPI Analysis ==========
        # 如果有 kpi_definition，计算排序能力/语义集中度等指标
        kpi_definition = getattr(strategy_config.labels, "kpi_definition", {})
        if kpi_definition and task_type == "regression":
            try:
                from scipy.stats import spearmanr

                print(f"\n   🎯 Return Tree KPI Analysis...")

                # 获取特征重要性（从已保存的 results 中）
                feature_importance = results.get("feature_importance", {})
                top_features = (
                    list(feature_importance.items())[:20] if feature_importance else []
                )

                # 使用测试集
                analysis_df = df_test_filtered.copy()
                X_test = analysis_df[used_features].values
                y_actual = analysis_df[strategy_config.labels.target_column].values

                # 获取预测
                preds_for_kpi = []
                for model in models:
                    if model is not None:
                        try:
                            preds_for_kpi.append(model.predict(X_test))
                        except Exception:
                            pass

                if preds_for_kpi:
                    y_pred = np.mean(preds_for_kpi, axis=0)

                    # 初始化 KPI 变量
                    spearman_corr = 0.0
                    monotonicity = 0.0
                    q5_q1_spread = 0.0
                    top10_ratio = 0.0

                    # 1️⃣ 排序能力 KPI
                    print(f"\n      📊 排序能力 (Ranking Ability):")

                    # Spearman 相关系数
                    valid_mask = ~(np.isnan(y_actual) | np.isnan(y_pred))
                    if valid_mask.sum() > 10:
                        spearman_corr, _ = spearmanr(
                            y_actual[valid_mask], y_pred[valid_mask]
                        )
                        spearman_status = (
                            "✅"
                            if spearman_corr >= 0.15
                            else ("⚠️" if spearman_corr >= 0 else "❌")
                        )
                        print(
                            f"         Spearman Corr: {spearman_corr:.3f} {spearman_status} (target: >= 0.15)"
                        )
                    else:
                        spearman_corr = 0.0
                        print(f"         Spearman Corr: N/A (insufficient data)")

                    # 分位单调性: Q5_RR > Q4_RR > ... > Q1_RR
                    valid_df = analysis_df[valid_mask].copy()
                    valid_df["pred"] = y_pred[valid_mask]
                    valid_df["actual"] = y_actual[valid_mask]

                    try:
                        valid_df["quantile"] = pd.qcut(
                            valid_df["pred"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
                        )
                        q_means = valid_df.groupby("quantile")["actual"].mean()

                        # 检查单调性
                        monotonic_count = sum(
                            q_means.get(f"Q{i+1}", 0) > q_means.get(f"Q{i}", 0)
                            for i in range(1, 5)
                        )
                        monotonicity = monotonic_count / 4
                        mono_status = (
                            "✅"
                            if monotonicity >= 0.8
                            else ("⚠️" if monotonicity >= 0.5 else "❌")
                        )
                        print(
                            f"         分位单调性: {monotonicity:.0%} {mono_status} (target: >= 80%)"
                        )

                        # Q5-Q1 差距
                        q5_rr = q_means.get("Q5", 0)
                        q1_rr = q_means.get("Q1", 0)
                        q5_q1_spread = q5_rr - q1_rr
                        spread_status = (
                            "✅"
                            if q5_q1_spread >= 0.3
                            else ("⚠️" if q5_q1_spread >= 0.1 else "❌")
                        )
                        print(
                            f"         Q5-Q1 Spread: {q5_q1_spread:.3f}R {spread_status} (target: >= 0.3R)"
                        )

                        # 打印分位详情
                        print(f"\n         分位组 RR 均值:")
                        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                            q_val = q_means.get(q, 0)
                            n_q = (valid_df["quantile"] == q).sum()
                            print(f"           {q}: {q_val:+.3f}R (n={n_q})")

                    except Exception as qe:
                        print(f"         分位分析失败: {qe}")
                        monotonicity = 0
                        q5_q1_spread = 0

                    # 2️⃣ 语义集中度 KPI
                    print(f"\n      📊 语义集中度 (Semantic Concentration):")
                    if top_features:
                        total_imp = sum(imp for _, imp in top_features)
                        top10_imp = sum(imp for _, imp in top_features[:10])
                        top10_ratio = top10_imp / total_imp if total_imp > 0 else 0
                        ratio_status = "✅" if 0.3 <= top10_ratio <= 0.6 else "⚠️"
                        print(
                            f"         Top10 重要性占比: {top10_ratio:.1%} {ratio_status} (target: 30%-60%)"
                        )

                        # 显示 Top5 特征
                        print(f"         Top5 特征:")
                        for i, (feat, imp) in enumerate(top_features[:5], 1):
                            print(f"           {i}. {feat}: {imp:.3f}")
                    else:
                        top10_ratio = 0
                        print(f"         Top10 重要性: N/A")

                    # 3️⃣ 稳定性 KPI (跨符号)
                    if is_multi_symbol and "_symbol" in analysis_df.columns:
                        print(f"\n      📊 跨符号稳定性 (Cross-Symbol Stability):")
                        symbol_spearman = {}
                        for sym in symbol_list:
                            sym_mask = analysis_df["_symbol"] == sym
                            if sym_mask.sum() < 30:
                                continue
                            y_sym = y_actual[sym_mask]
                            p_sym = y_pred[sym_mask]
                            valid_sym = ~(np.isnan(y_sym) | np.isnan(p_sym))
                            if valid_sym.sum() > 10:
                                corr, _ = spearmanr(y_sym[valid_sym], p_sym[valid_sym])
                                symbol_spearman[sym] = corr
                                status = "✅" if corr > 0 else "❌"
                                print(f"           {sym}: {corr:.3f} {status}")

                        if symbol_spearman:
                            positive_count = sum(
                                1 for c in symbol_spearman.values() if c > 0
                            )
                            consistency = positive_count / len(symbol_spearman)
                            cons_status = "✅" if consistency >= 0.6 else "⚠️"
                            print(
                                f"         符号一致性: {consistency:.0%} ({positive_count}/{len(symbol_spearman)} positive) {cons_status}"
                            )
                    else:
                        symbol_spearman = {}
                        consistency = 0.0

                    # 保存 KPI 结果（包含详细数据）
                    results["return_tree_kpi"] = {
                        "spearman_corr": float(spearman_corr),
                        "quantile_monotonicity": float(monotonicity),
                        "q5_q1_spread": float(q5_q1_spread),
                        "top10_importance_ratio": float(top10_ratio),
                        # 详细数据
                        "quantile_means": (
                            {
                                q: float(q_means.get(q, 0))
                                for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]
                            }
                            if "q_means" in dir()
                            else {}
                        ),
                        "top5_features": (
                            [(f, float(imp)) for f, imp in top_features[:5]]
                            if top_features
                            else []
                        ),
                        "symbol_spearman": (
                            {k: float(v) for k, v in symbol_spearman.items()}
                            if symbol_spearman
                            else {}
                        ),
                        "symbol_consistency": (
                            float(consistency) if "consistency" in dir() else 0.0
                        ),
                    }

            except Exception as kpi_exc:
                print(f"   ⚠️  Return Tree KPI analysis failed: {kpi_exc}")

        # 💾 重新保存 results.json（包含 failure_analysis 数据）
        if output_cfg.get("save_results", True):
            filename = output_cfg.get("filename", "results.json")
            results_file = output_dir / filename
            with open(results_file, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, default=str)
            # print(f"   💾 Results updated with failure_analysis to {results_file}")

            # 📄 生成 HTML 报告（包含 failure_analysis + return_tree_kpi）
            html_report_path = generate_training_html_report(
                results=results,
                output_dir=output_dir,
                strategy_name=strategy_config.name,
                args=args,
            )
            if html_report_path:
                print(f"   📄 HTML report saved to {html_report_path}")

        # Save preprocessor (required for inference consistency)
        import joblib

        preprocessor_path = output_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        print(f"   💾 Preprocessor saved to {preprocessor_path}")

        # Optionally save as ModelArtifact (unified format)
        try:
            from src.time_series_model.strategies.models.model_artifact import (
                ModelArtifact,
            )

            artifact = ModelArtifact(
                model=models,
                preprocessor=preprocessor,
                used_features=used_features,
                feature_config=(
                    strategy_config.features.__dict__
                    if hasattr(strategy_config.features, "__dict__")
                    else None
                ),
                metadata={
                    "strategy": strategy_config.name,
                    "model_type": model_type,
                    "task_type": task_type,
                    "avg_cv_metric": float(avg_metric),
                    "n_train_samples": len(df_train_filtered),
                    "n_test_samples": len(df_test_filtered),
                },
            )
            artifact.save(output_dir)
            print(f"   ✅ ModelArtifact saved (unified format)")
        except Exception as exc:  # noqa: BLE001
            # Fallback: continue with individual saves if ModelArtifact fails
            print(f"   ⚠️  ModelArtifact save failed (using individual saves): {exc}")

        # Save volatility model if trained
        if vol_model:
            vol_model_file = output_dir / "volatility_model.pkl"
            joblib.dump(vol_model, vol_model_file)
            print(f"   💾 Volatility model saved to {vol_model_file}")

        # Auto-export tree rules and yaml config based on task type
        try:
            from scripts.export_lightgbm_rules_to_readme import (
                _collect_splits,
                _get_booster,
                _write_standalone_rules,
                _generate_risk_gate_yaml,
            )

            model_path = output_dir / "model.pkl"
            features_path = output_dir / "used_features.json"

            if model_path.exists():
                import json as json_module

                loaded_model = joblib.load(model_path)
                if isinstance(loaded_model, dict):
                    loaded_model = (
                        loaded_model.get("regression")
                        or loaded_model.get("model")
                        or list(loaded_model.values())[0]
                    )
                booster = _get_booster(loaded_model)

                feature_names = []
                if features_path.exists():
                    with open(features_path, encoding="utf-8") as f:
                        feature_names = json_module.load(f)
                if not feature_names and hasattr(booster, "feature_name"):
                    feature_names = booster.feature_name() or []

                rules = _collect_splits(booster, feature_names, max_splits=30)
                if not rules:
                    print(
                        f"   ⚠️  模型无树分裂规则可导出 "
                        f"(可能训练样本过少/min_data_in_leaf过大)"
                    )
                    if task_type == "regression":
                        print(
                            f"   → evidence_candidates.yaml 未生成, "
                            f"Evidence Optimize 将失败"
                        )
                if rules:
                    # Export tree rules
                    rules_path = output_dir / f"{strategy_config.name}_tree_rules.md"
                    _write_standalone_rules(
                        rules_path, rules, strategy_config.name, str(output_dir)
                    )
                    print(f"   \U0001f4dc Tree rules exported to {rules_path}")

                    # Choose export based on task_type:
                    # - regression (Return Tree): evidence_candidates.yaml
                    # - binary (Failure Tree): risk_gate_draft.yaml
                    if task_type == "regression":
                        # Evidence 候选发现不再依赖 LightGBM tree rules
                        # 改用 discover_evidence_candidates.py (Spearman + Quintile)
                        # 在 auto_research_pipeline Step 6 中自动调用
                        print(
                            f"   ℹ️  Evidence discovery deferred to "
                            f"discover_evidence_candidates.py"
                        )
                    else:
                        risk_gate_path = output_dir / "risk_gate_draft.yaml"
                        _predictions_path = output_dir / "predictions.parquet"
                        _generate_risk_gate_yaml(
                            risk_gate_path,
                            rules,
                            strategy_config.name,
                            str(output_dir),
                            predictions_path=(
                                _predictions_path
                                if _predictions_path.exists()
                                else None
                            ),
                            feature_names=feature_names if feature_names else None,
                            lgbm_model=loaded_model,  # A.7.2: teacher for distillation
                            skip_gate_shap_discovery=bool(
                                getattr(args, "skip_gate_shap", False)
                            ),
                            gate_draft_path=strategy_config.path / "gate_draft.yaml",
                        )
                        print(
                            f"   \U0001f4dc Risk gate draft exported to {risk_gate_path}"
                        )
        except Exception as exc:
            print(f"   \u26a0\ufe0f  Auto-export rules failed: {exc}")


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
