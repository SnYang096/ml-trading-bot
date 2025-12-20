#!/usr/bin/env python3
"""Single-factor time-series evaluation helper with IC decay and HTML reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, skew, kurtosis

# Ensure project root on sys.path so we can reuse existing modules
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]  # src/diagnostics -> src -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate individual factors via the strategy pipeline"
    )
    parser.add_argument("--strategy-config", required=True, help="Path to strategy dir")
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Factor columns to evaluate. If not specified, will use requested_features from strategy config's features.yaml",
    )
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="15T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--quantile", type=float, default=0.2, help="Top/Bottom quantile"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["strategy", "only", "append"],
        default="strategy",
        help="How to handle feature pipeline: use strategy defaults, only requested factors, or append requested factors.",
    )
    parser.add_argument("--output-dir", default="results/factor_ts_eval")
    parser.add_argument(
        "--ic-decay-lags",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated forward bars for IC decay analysis (e.g., '1,3,5,10,20')",
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        default=True,
        help="Generate HTML report (default: True)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=False,
        help="Automatically open HTML report in browser after generation",
    )
    return parser.parse_args()


def _compute_requested_features(
    df_raw: pd.DataFrame,
    feature_loader: StrategyFeatureLoader,
    requested: List[str],
    ensure_signal_cfg,
) -> pd.DataFrame:
    print(f"   🔧 Computing requested features: {requested}")
    print(f"   📊 Input DataFrame columns: {list(df_raw.columns)[:10]}...")

    df_features = feature_loader.load_features_from_requested(
        df_raw,
        requested_features=requested,
        fit=True,
    )

    # Check which columns were actually computed
    computed_cols = [c for c in df_features.columns if c not in df_raw.columns]
    missing_features = [f for f in requested if f not in df_features.columns]

    print(
        f"   ✅ Computed {len(computed_cols)} new columns: {computed_cols[:15]}{'...' if len(computed_cols) > 15 else ''}"
    )

    if missing_features:
        print(f"⚠️  Warning: Some requested factors were not found: {missing_features}")

        # Check each missing feature's output_columns configuration
        features_config = feature_loader.feature_deps.get("features", {})
        for feature_name in missing_features:
            if feature_name in features_config:
                feature_info = features_config[feature_name]
                expected_outputs = feature_info.get("output_columns", [feature_name])
                found_outputs = [
                    col for col in expected_outputs if col in df_features.columns
                ]

                if found_outputs:
                    print(
                        f"   ✅ Feature '{feature_name}' outputs found: {found_outputs} (expected: {expected_outputs})"
                    )
                else:
                    print(
                        f"   ❌ Feature '{feature_name}' not found. Expected outputs: {expected_outputs}"
                    )
                    print(
                        f"      Description: {feature_info.get('description', 'N/A')}"
                    )
                    if feature_info.get("required_columns"):
                        print(
                            f"      Required columns: {feature_info.get('required_columns')}"
                        )
                        missing_req = [
                            c
                            for c in feature_info.get("required_columns", [])
                            if c not in df_raw.columns
                        ]
                        if missing_req:
                            print(f"      ⚠️  Missing required columns: {missing_req}")
                        else:
                            print(f"      ✅ All required columns present")
                    # Check if compute function exists
                    compute_func_name = feature_info.get("compute_func", "")
                    print(f"      Compute function: {compute_func_name}")
            else:
                print(
                    f"   ❌ Feature '{feature_name}' not found in feature dependencies configuration"
                )
                print(
                    f"      Available features: {list(features_config.keys())[:30]}{'...' if len(features_config) > 30 else ''}"
                )

    return strategy_runner.ensure_signal_column(df_features, ensure_signal_cfg)


def prepare_dataset(args: argparse.Namespace, strategy_cfg) -> pd.DataFrame:
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )

    feature_loader = StrategyFeatureLoader()
    feature_mode = args.feature_mode
    extra_factors = args.factors or []

    # Get strategy's requested features if available
    strategy_requested = []
    if hasattr(strategy_cfg.features, "requested_features"):
        strategy_requested = strategy_cfg.features.requested_features or []
    elif isinstance(strategy_cfg.features, dict):
        strategy_requested = strategy_cfg.features.get("requested_features", [])

    # Check if requested factors are in strategy config
    # MACD outputs multiple columns: macd, macd_signal, macd_histogram
    # So if user requests "macd", we need to check if any MACD column exists
    missing_from_strategy = []
    if extra_factors and feature_mode == "strategy":
        strategy_requested_set = set(strategy_requested)

        for factor in extra_factors:
            factor_found = False

            # Direct match
            if factor in strategy_requested_set:
                factor_found = True
            else:
                # Check MACD variants (macd outputs: macd, macd_signal, macd_histogram)
                macd_variants = ["macd", "macd_signal", "macd_histogram", "macd_hist"]
                if factor in macd_variants:
                    # If user requests "macd", check if any MACD column is in strategy
                    if any(
                        variant in strategy_requested_set for variant in macd_variants
                    ):
                        factor_found = True
                    # Or if strategy has "macd" as a feature name (which generates all MACD columns)
                    if "macd" in strategy_requested_set:
                        factor_found = True

            if not factor_found:
                missing_from_strategy.append(factor)

        # Auto-switch to append mode if factors are missing
        if missing_from_strategy:
            print(
                f"⚠️  Warning: Factors {missing_from_strategy} not in strategy config '{strategy_cfg.name}'. Auto-switching to 'append' mode."
            )
            print(f"   Strategy has: {strategy_requested}")
            feature_mode = "append"

    if feature_mode == "strategy":
        df_features = strategy_runner.run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_cfg.features,
            fit=True,
        )
    elif feature_mode == "only":
        if not extra_factors:
            raise ValueError("--feature-mode=only requires --factors to be specified")
        df_features = _compute_requested_features(
            df_raw,
            feature_loader,
            extra_factors,
            strategy_cfg.features.ensure_signal,
        )
    elif feature_mode == "append":
        if not extra_factors:
            raise ValueError("--feature-mode=append requires --factors to be specified")
        base_features = strategy_runner.run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_cfg.features,
            fit=True,
        )
        requested_df = _compute_requested_features(
            df_raw,
            feature_loader,
            extra_factors,
            strategy_cfg.features.ensure_signal,
        )

        # Merge requested features into base features
        # Ensure all output columns from requested features are included
        features_config = feature_loader.feature_deps.get("features", {})
        for col in requested_df.columns:
            if col in base_features.columns:
                # Update existing column if it's a requested factor or its output
                if col in extra_factors:
                    base_features[col] = requested_df[col]
            else:
                base_features[col] = requested_df[col]

        # For each requested feature, ensure all its output_columns are included
        for feature_name in extra_factors:
            if feature_name in features_config:
                feature_info = features_config[feature_name]
                output_cols = feature_info.get("output_columns", [feature_name])
                for output_col in output_cols:
                    if (
                        output_col in requested_df.columns
                        and output_col not in base_features.columns
                    ):
                        base_features[output_col] = requested_df[output_col]

        df_features = base_features
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    label_func = strategy_runner.import_callable(
        strategy_cfg.labels.generator.module,
        strategy_cfg.labels.generator.function,
    )
    target_col = strategy_cfg.labels.target_column
    df_features[target_col] = label_func(
        df_features.copy(), **strategy_cfg.labels.generator.params
    )

    df_filtered = strategy_runner.apply_filters(
        df_features, strategy_cfg.labels.filters
    )
    df_filtered = strategy_runner.apply_post_label_filters(
        df_filtered,
        strategy_cfg.labels.post_label_filters,
        list(df_filtered.columns),
    )
    return df_filtered


def compute_ic_series(
    df: pd.DataFrame, factor: str, target_col: str, window: int = 60
) -> pd.Series:
    """Compute rolling IC series."""
    valid = df[[factor, target_col]].dropna()
    if len(valid) < window:
        return pd.Series(dtype=float)

    ic_series = []
    for i in range(window, len(valid) + 1):
        window_data = valid.iloc[i - window : i]
        factor_ranks = window_data[factor].rank(pct=True)
        target_ranks = window_data[target_col].rank(pct=True)
        ic = spearmanr(factor_ranks, target_ranks).correlation
        if not np.isnan(ic):
            ic_series.append(ic)
        else:
            ic_series.append(0.0)

    return pd.Series(ic_series, index=valid.index[window - 1 :])


def compute_ic_decay(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    decay_lags: List[int],
) -> Dict[str, float]:
    """Compute IC for different forward horizons (decay analysis)."""
    decay_metrics = {}
    base_target = df[target_col].copy()

    for lag in decay_lags:
        # Shift target forward by lag bars
        lag_target = base_target.shift(-lag)
        lag_target.name = f"{target_col}_lag{lag}"
        valid = df[[factor]].join(lag_target).dropna()

        if len(valid) < 50:
            decay_metrics[f"ic_lag_{lag}"] = np.nan
            decay_metrics[f"ic_lag_{lag}_pearson"] = np.nan
            continue

        factor_ranks = valid[factor].rank(pct=True)
        target_ranks = valid[lag_target.name].rank(pct=True)
        rank_ic = spearmanr(factor_ranks, target_ranks).correlation
        pearson_ic = np.corrcoef(valid[factor], valid[lag_target.name])[0, 1]

        # Keep NaN as NaN, don't convert to 0.0, so Best Lag calculation can skip invalid values
        decay_metrics[f"ic_lag_{lag}"] = (
            float(rank_ic) if not np.isnan(rank_ic) else np.nan
        )
        decay_metrics[f"ic_lag_{lag}_pearson"] = (
            float(pearson_ic) if not np.isnan(pearson_ic) else np.nan
        )

    return decay_metrics


def compute_factor_metrics(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    quantile: float,
    ic_decay_lags: List[int] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute comprehensive factor metrics including IC decay."""
    metrics: Dict[str, float] = {}
    ic_series_df = pd.DataFrame()

    # Check if factor exists
    if factor not in df.columns:
        metrics["error"] = "factor_missing"
        available = sorted(
            [
                c
                for c in df.columns
                if c
                not in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "timestamp",
                    "_symbol",
                    "signal",
                    "future_return",
                ]
            ]
        )[:30]
        metrics["available_columns"] = available
        print(f"   ❌ Factor '{factor}' not found in DataFrame columns")
        print(f"      Available feature columns: {available[:15]}...")
        return metrics, ic_series_df

    # ------------------------------------------------------------------
    # Strict numeric checks for factor & target
    # - 因子列：如果大部分都能转成数值，则使用转换后的结果；否则直接标记为 non_numeric_factor
    # - 目标列：必须是数值列，否则直接报错 non_numeric_target
    # ------------------------------------------------------------------
    factor_series = df[factor]
    target_series = df[target_col]

    # Factor: allow coercion but要求足够高的非 NaN 比例
    if not pd.api.types.is_numeric_dtype(factor_series):
        factor_numeric = pd.to_numeric(factor_series, errors="coerce")
        non_na_ratio = float(factor_numeric.notna().mean())
        if non_na_ratio < 0.8:
            metrics["error"] = "non_numeric_factor"
            metrics["non_na_ratio"] = non_na_ratio
            sample_values = factor_series.dropna().astype(str).unique().tolist()[:10]
            metrics["sample_values"] = sample_values
            print(
                f"   ❌ Factor '{factor}' is non-numeric (non-NaN ratio after coercion={non_na_ratio:.2f})."
            )
            print(f"      Example raw values: {sample_values}")
            return metrics, ic_series_df
    else:
        factor_numeric = pd.to_numeric(factor_series, errors="coerce")

    # Target: should already be numeric; if不是，直接报错
    if not pd.api.types.is_numeric_dtype(target_series):
        target_numeric = pd.to_numeric(target_series, errors="coerce")
        non_na_ratio_tgt = float(target_numeric.notna().mean())
        if non_na_ratio_tgt < 0.95:
            metrics["error"] = "non_numeric_target"
            metrics["non_na_ratio"] = non_na_ratio_tgt
            sample_values_tgt = (
                target_series.dropna().astype(str).unique().tolist()[:10]
            )
            metrics["sample_values_target"] = sample_values_tgt
            print(
                f"   ❌ Target '{target_col}' is non-numeric (non-NaN ratio after coercion={non_na_ratio_tgt:.2f})."
            )
            print(f"      Example raw target values: {sample_values_tgt}")
            return metrics, ic_series_df
    else:
        target_numeric = pd.to_numeric(target_series, errors="coerce")

    # Build a numeric-only DataFrame for downstream computations
    df_numeric = df.copy()
    df_numeric[factor] = factor_numeric
    df_numeric[target_col] = target_numeric

    valid = df_numeric[[factor, target_col]].dropna()
    if len(valid) < 50:
        metrics["error"] = "insufficient_samples"
        metrics["n_samples"] = int(len(valid))
        return metrics, ic_series_df

    # Basic IC metrics
    factor_ranks = valid[factor].rank(pct=True)
    target_ranks = valid[target_col].rank(pct=True)
    rank_ic = spearmanr(factor_ranks, target_ranks).correlation
    pearson = np.corrcoef(valid[factor], valid[target_col])[0, 1]

    # Rolling IC series for IR calculation
    ic_series = compute_ic_series(df_numeric, factor, target_col, window=60)
    if len(ic_series) > 0:
        metrics["ic_mean"] = float(ic_series.mean())
        metrics["ic_std"] = float(ic_series.std())
        metrics["ic_ir"] = (
            metrics["ic_mean"] / metrics["ic_std"] if metrics["ic_std"] > 0 else 0.0
        )
        metrics["ic_positive_ratio"] = float((ic_series > 0).mean())
        ic_series_df = pd.DataFrame({"ic": ic_series})
    else:
        metrics["ic_mean"] = float(rank_ic) if not np.isnan(rank_ic) else 0.0
        metrics["ic_std"] = 0.0
        metrics["ic_ir"] = 0.0
        metrics["ic_positive_ratio"] = 0.0

    # IC decay analysis
    if ic_decay_lags:
        decay_metrics = compute_ic_decay(df_numeric, factor, target_col, ic_decay_lags)
        metrics.update(decay_metrics)

    # Quantile analysis
    high_cut = valid[factor].quantile(1 - quantile)
    low_cut = valid[factor].quantile(quantile)

    long_mask = valid[factor] >= high_cut
    short_mask = valid[factor] <= low_cut

    long_returns = valid.loc[long_mask, target_col]
    short_returns = valid.loc[short_mask, target_col]

    win_rate_long = float((long_returns > 0).mean()) if len(long_returns) else 0.0
    win_rate_short = float((short_returns < 0).mean()) if len(short_returns) else 0.0

    # Simple long-short backtest
    position = pd.Series(0.0, index=valid.index)
    position[long_mask] = 1.0
    position[short_mask] = -1.0
    strategy_ret = position.shift().fillna(0.0) * valid[target_col]
    equity_curve = strategy_ret.cumsum()
    total_return = float(equity_curve.iloc[-1])
    max_dd = float((equity_curve.cummax() - equity_curve).max())

    # Calculate max drawdown duration
    dd_series = equity_curve.cummax() - equity_curve
    max_dd_duration = 0
    if max_dd > 0:
        in_dd = dd_series > 0
        if in_dd.any():
            dd_periods = (in_dd != in_dd.shift()).cumsum()
            max_dd_duration = (
                int(dd_periods.value_counts().max()) if len(dd_periods) > 0 else 0
            )

    # Risk-adjusted returns
    strategy_ret_clean = strategy_ret[strategy_ret != 0]
    if len(strategy_ret_clean) > 0:
        ret_std = float(strategy_ret_clean.std())
        ret_mean = float(strategy_ret_clean.mean())
        sharpe_ratio = (ret_mean / ret_std * np.sqrt(252)) if ret_std > 1e-8 else 0.0
        calmar_ratio = (total_return / abs(max_dd)) if max_dd < -1e-8 else 0.0
        # Sortino ratio (downside deviation)
        downside_returns = strategy_ret_clean[strategy_ret_clean < 0]
        downside_std = (
            float(downside_returns.std()) if len(downside_returns) > 0 else 0.0
        )
        sortino_ratio = (
            (ret_mean / downside_std * np.sqrt(252)) if downside_std > 1e-8 else 0.0
        )
    else:
        sharpe_ratio = 0.0
        calmar_ratio = 0.0
        sortino_ratio = 0.0

    # Quantile spread (long-short return difference)
    quantile_spread = (
        float(long_returns.mean() - short_returns.mean())
        if len(long_returns) > 0 and len(short_returns) > 0
        else 0.0
    )

    # Statistical significance of IC
    if len(ic_series) > 1:
        ic_t_stat, ic_p_value = ttest_1samp(ic_series, 0)
        metrics["ic_t_stat"] = float(ic_t_stat)
        metrics["ic_p_value"] = float(ic_p_value)
    else:
        metrics["ic_t_stat"] = 0.0
        metrics["ic_p_value"] = 1.0

    # Statistical significance of quantile returns
    if len(long_returns) > 1:
        long_t_stat, long_p_value = ttest_1samp(long_returns, 0)
        metrics["long_return_t_stat"] = float(long_t_stat)
        metrics["long_return_p_value"] = float(long_p_value)
    else:
        metrics["long_return_t_stat"] = 0.0
        metrics["long_return_p_value"] = 1.0

    if len(short_returns) > 1:
        short_t_stat, short_p_value = ttest_1samp(short_returns, 0)
        metrics["short_return_t_stat"] = float(short_t_stat)
        metrics["short_return_p_value"] = float(short_p_value)
    else:
        metrics["short_return_t_stat"] = 0.0
        metrics["short_return_p_value"] = 1.0

    # Quantile return statistics
    if len(long_returns) > 0:
        metrics["long_return_std"] = float(long_returns.std())
        metrics["long_return_skew"] = (
            float(skew(long_returns)) if len(long_returns) > 2 else 0.0
        )
        metrics["long_return_kurtosis"] = (
            float(kurtosis(long_returns)) if len(long_returns) > 3 else 0.0
        )
    else:
        metrics["long_return_std"] = 0.0
        metrics["long_return_skew"] = 0.0
        metrics["long_return_kurtosis"] = 0.0

    if len(short_returns) > 0:
        metrics["short_return_std"] = float(short_returns.std())
        metrics["short_return_skew"] = (
            float(skew(short_returns)) if len(short_returns) > 2 else 0.0
        )
        metrics["short_return_kurtosis"] = (
            float(kurtosis(short_returns)) if len(short_returns) > 3 else 0.0
        )
    else:
        metrics["short_return_std"] = 0.0
        metrics["short_return_skew"] = 0.0
        metrics["short_return_kurtosis"] = 0.0

    # Factor distribution stats
    factor_stats = valid[factor].describe()
    factor_skewness = float(skew(valid[factor])) if len(valid) > 2 else 0.0
    factor_kurt = float(kurtosis(valid[factor])) if len(valid) > 3 else 0.0

    # Factor autocorrelation (stability measure)
    factor_autocorr = float(valid[factor].autocorr(lag=1)) if len(valid) > 1 else 0.0

    # Turnover calculation (position change rate)
    position_changes = (
        (position.diff().abs().sum() / len(position)) if len(position) > 1 else 0.0
    )
    turnover = float(position_changes)

    # Factor-target correlation distribution stats
    target_stats = valid[target_col].describe()

    metrics.update(
        {
            "n_samples": int(len(valid)),
            "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            "pearson": float(pearson) if not np.isnan(pearson) else 0.0,
            "win_rate_long": win_rate_long,
            "win_rate_short": win_rate_short,
            "avg_return_long": float(long_returns.mean()) if len(long_returns) else 0.0,
            "avg_return_short": (
                float(short_returns.mean()) if len(short_returns) else 0.0
            ),
            "quantile_spread": quantile_spread,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio,
            "turnover": turnover,
            "factor_mean": float(factor_stats["mean"]),
            "factor_std": float(factor_stats["std"]),
            "factor_min": float(factor_stats["min"]),
            "factor_max": float(factor_stats["max"]),
            "factor_skewness": factor_skewness,
            "factor_kurtosis": factor_kurt,
            "factor_autocorr": factor_autocorr,
            "target_mean": float(target_stats["mean"]),
            "target_std": float(target_stats["std"]),
        }
    )
    return metrics, ic_series_df


def generate_html_report(
    results: Dict[str, Dict[str, float]],
    ic_series_data: Dict[str, pd.DataFrame],
    strategy_name: str,
    symbol: str,
    output_dir: Path,
    ic_decay_lags: List[int],
    diagnostics: Dict[str, Any] | None = None,
) -> Path:
    """Generate comprehensive HTML report."""
    html_path = output_dir / f"ts_eval_{strategy_name}_{symbol}.html"

    diagnostics = diagnostics or {}
    factor_resolution = diagnostics.get("factor_resolution", {})
    unknown_factors = factor_resolution.get("unknown_factors", [])
    missing_feature_outputs = factor_resolution.get("missing_feature_outputs", [])
    factor_mappings = factor_resolution.get("mappings", {})
    error_factors = diagnostics.get("error_factors", {})

    # Build summary table
    summary_rows = []
    for factor, metrics in results.items():
        if "error" in metrics:
            summary_rows.append(
                f"""
                <tr>
                    <td><strong>{factor}</strong></td>
                    <td colspan="12" style="color: red;">Error: {metrics['error']}</td>
                </tr>
                """
            )
            continue

        # Find best lag (highest IC value, not absolute value)
        # For positive IC: higher is better
        # For negative IC: less negative (closer to 0) is better, but we want the highest IC value
        best_lag = None
        best_ic = None
        for lag in ic_decay_lags:
            ic_val = metrics.get(f"ic_lag_{lag}", None)
            # Skip None, NaN
            if ic_val is None:
                continue
            # Check for NaN (can be float('nan') or np.nan)
            if isinstance(ic_val, float) and np.isnan(ic_val):
                continue
            # Convert to float and check if valid
            try:
                ic_val_float = float(ic_val)
                if np.isnan(ic_val_float):
                    continue
            except (ValueError, TypeError):
                continue

            # Update best if this IC value is higher (not absolute value)
            # This means: for positive IC, higher is better; for negative IC, less negative is better
            if best_ic is None or ic_val_float > best_ic:
                best_ic = ic_val_float
                best_lag = lag

        # Fallback: if no valid IC found in decay lags, use rank_ic as best_ic
        if best_lag is None:
            rank_ic = metrics.get("rank_ic", 0.0)
            if rank_ic is not None:
                try:
                    rank_ic_float = float(rank_ic)
                    if not np.isnan(rank_ic_float) and rank_ic_float != 0.0:
                        best_ic = rank_ic_float
                        best_lag = 0  # Use 0 to indicate current period
                except (ValueError, TypeError):
                    pass

        # Color coding for key metrics
        ic_ir = metrics.get("ic_ir", 0.0)
        ic_ir_class = "good" if ic_ir > 0.5 else ("warning" if ic_ir > 0 else "bad")

        sharpe = metrics.get("sharpe_ratio", 0.0)
        sharpe_class = "good" if sharpe > 1.0 else ("warning" if sharpe > 0 else "bad")

        quantile_spread = metrics.get("quantile_spread", 0.0)
        spread_class = "good" if quantile_spread > 0 else "bad"

        best_lag_display = f"{best_lag}" if best_lag is not None else "N/A"
        best_ic_display = f"{best_ic:.4f}" if best_ic is not None else "N/A"
        best_ic_class = (
            "good"
            if (best_ic is not None and best_ic > 0.05)
            else ("warning" if (best_ic is not None and best_ic > 0) else "bad")
        )

        summary_rows.append(
            f"""
            <tr>
                <td><strong>{factor}</strong></td>
                <td>{metrics.get('n_samples', 0)}</td>
                <td>{metrics.get('rank_ic', 0.0):.4f}</td>
                <td>{metrics.get('pearson', 0.0):.4f}</td>
                <td>{metrics.get('ic_mean', 0.0):.4f}</td>
                <td>{metrics.get('ic_std', 0.0):.4f}</td>
                <td class="{ic_ir_class}">{metrics.get('ic_ir', 0.0):.4f}</td>
                <td>{metrics.get('ic_positive_ratio', 0.0):.2%}</td>
                <td>{metrics.get('ic_t_stat', 0.0):.2f}</td>
                <td>{metrics.get('ic_p_value', 1.0):.4f}</td>
                <td class="{best_ic_class}"><strong>{best_lag_display}</strong></td>
                <td class="{best_ic_class}">{best_ic_display}</td>
                <td>{metrics.get('quantile_spread', 0.0):.4f}</td>
                <td>{metrics.get('win_rate_long', 0.0):.2%}</td>
                <td>{metrics.get('win_rate_short', 0.0):.2%}</td>
                <td>{metrics.get('avg_return_long', 0.0):.4f}</td>
                <td>{metrics.get('avg_return_short', 0.0):.4f}</td>
                <td>{metrics.get('long_return_t_stat', 0.0):.2f}</td>
                <td>{metrics.get('short_return_t_stat', 0.0):.2f}</td>
                <td class="{sharpe_class}">{metrics.get('sharpe_ratio', 0.0):.2f}</td>
                <td>{metrics.get('calmar_ratio', 0.0):.2f}</td>
                <td>{metrics.get('sortino_ratio', 0.0):.2f}</td>
                <td>{metrics.get('total_return', 0.0):.2f}</td>
                <td>{metrics.get('max_drawdown', 0.0):.2f}</td>
                <td>{metrics.get('max_drawdown_duration', 0)}</td>
                <td>{metrics.get('turnover', 0.0):.4f}</td>
                <td>{metrics.get('factor_autocorr', 0.0):.4f}</td>
            </tr>
            """
        )

    # Create detailed metrics section with IC decay charts for each factor
    detailed_sections = []
    chart_scripts = []

    for factor_idx, (factor, metrics) in enumerate(results.items()):
        if "error" in metrics:
            continue

        # Extract IC decay data
        ic_decay_values = []
        ic_decay_pearson = []
        best_lag = None
        best_ic = None

        for lag in ic_decay_lags:
            ic_val = metrics.get(f"ic_lag_{lag}", None)
            ic_pearson = metrics.get(f"ic_lag_{lag}_pearson", None)

            # For chart display, use 0.0 for NaN (Chart.js can handle it)
            ic_val_for_chart = 0.0
            ic_pearson_for_chart = 0.0
            if ic_val is not None:
                try:
                    ic_val_float = float(ic_val)
                    if not np.isnan(ic_val_float):
                        ic_val_for_chart = ic_val_float
                except (ValueError, TypeError):
                    pass
            if ic_pearson is not None:
                try:
                    ic_pearson_float = float(ic_pearson)
                    if not np.isnan(ic_pearson_float):
                        ic_pearson_for_chart = ic_pearson_float
                except (ValueError, TypeError):
                    pass

            ic_decay_values.append(ic_val_for_chart)
            ic_decay_pearson.append(ic_pearson_for_chart)

            # Find best lag (highest IC value, not absolute value) - skip NaN values
            if ic_val is not None:
                try:
                    ic_val_float = float(ic_val)
                    if not np.isnan(ic_val_float):
                        # Update best if this IC value is higher (not absolute value)
                        if best_ic is None or ic_val_float > best_ic:
                            best_ic = ic_val_float
                            best_lag = lag
                except (ValueError, TypeError):
                    pass

        # Fallback: if no valid IC found in decay lags, use rank_ic
        if best_lag is None:
            rank_ic = metrics.get("rank_ic", 0.0)
            if rank_ic is not None:
                try:
                    rank_ic_float = float(rank_ic)
                    if not np.isnan(rank_ic_float) and rank_ic_float != 0.0:
                        best_ic = rank_ic_float
                        best_lag = 0
                except (ValueError, TypeError):
                    pass

        best_lag_display = f"{best_lag}" if best_lag is not None else "N/A"
        best_ic_display = f"{best_ic:.4f}" if best_ic is not None else "N/A"

        # Chart ID for this factor
        chart_id = f"icDecayChart_{factor_idx}"

        # Create chart HTML
        chart_html = f"""
        <div class="chart-container">
            <canvas id="{chart_id}"></canvas>
        </div>
        """

        # Create chart script
        chart_script = f"""
        // Chart for {factor}
        const ctx_{factor_idx} = document.getElementById('{chart_id}');
        if (ctx_{factor_idx}) {{
            new Chart(ctx_{factor_idx}, {{
                type: 'line',
                data: {{
                    labels: {ic_decay_lags},
                    datasets: [{{
                        label: 'Rank IC (Spearman)',
                        data: {ic_decay_values},
                        borderColor: 'rgb(33, 150, 243)',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }}, {{
                        label: 'Pearson IC',
                        data: {ic_decay_pearson},
                        borderColor: 'rgb(76, 175, 80)',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        borderDash: [5, 5]
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'IC Decay Analysis - Best Lag: {best_lag_display} (IC={best_ic_display})',
                            font: {{
                                size: 16,
                                weight: 'bold'
                            }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Forward Bars (Lag)'
                            }},
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.05)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'IC Value'
                            }},
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.05)'
                            }},
                            zeroLine: {{
                                color: 'rgba(0, 0, 0, 0.2)',
                                width: 2
                            }}
                        }}
                    }},
                    interaction: {{
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }}
                }}
            }});
        }}
        """
        chart_scripts.append(chart_script)

        detailed_html = f"""
        <div class="factor-detail">
            <h3>📊 {factor} - Detailed Metrics</h3>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">IC Statistics</div>
                    <div class="metric-value">{metrics.get('rank_ic', 0.0):.4f}</div>
                    <div>Rank IC | IR: {metrics.get('ic_ir', 0.0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Best Lag</div>
                    <div class="metric-value" style="color: {'#4CAF50' if (best_ic is not None and best_ic > 0) else '#f44336'}">{best_lag if best_lag else 'N/A'}</div>
                    <div>IC: {best_ic_display} | Bars forward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Risk-Adjusted Returns</div>
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0.0):.2f}</div>
                    <div>Sharpe | Calmar: {metrics.get('calmar_ratio', 0.0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Quantile Spread</div>
                    <div class="metric-value">{metrics.get('quantile_spread', 0.0):.4f}</div>
                    <div>Long: {metrics.get('avg_return_long', 0.0):.4f} | Short: {metrics.get('avg_return_short', 0.0):.4f}</div>
                </div>
            </div>

            <h4>📉 IC Decay Analysis</h4>
            {chart_html}
        </div>
        """
        detailed_sections.append(detailed_html)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Time-Series Factor Evaluation: {symbol}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 95%;
                width: 100%;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-left: 4px solid #2196F3;
                padding-left: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 13px;
                table-layout: auto;
            }}
            th, td {{
                white-space: nowrap;
                padding: 8px 10px;
            }}
            th {{
                background-color: #2196F3;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .metric-card {{
                display: inline-block;
                background: #f8f9fa;
                border-left: 4px solid #4CAF50;
                padding: 15px;
                margin: 10px;
                border-radius: 4px;
                min-width: 200px;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .good {{ color: #4CAF50; }}
            .bad {{ color: #f44336; }}
            .warning {{ color: #ff9800; }}
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .factor-detail {{
                margin: 30px 0;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }}
            .chart-container {{
                position: relative;
                height: 400px;
                width: 100%;
                margin: 20px 0;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h4 {{
                color: #555;
                margin-top: 20px;
                margin-bottom: 15px;
                font-size: 18px;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Time-Series Factor Evaluation Report</h1>

            <div class="info-box">
                <strong>Strategy:</strong> {strategy_name}<br>
                <strong>Symbol:</strong> {symbol}<br>
                <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Factors Evaluated:</strong> {len(results)}
            </div>

            <h2>🧪 Diagnostics</h2>
            <div class="info-box">
                <p>Automatic checks from this run. Use them to identify bad factors, config issues, or data problems.</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Non-numeric / invalid factors</div>
                    <div class="metric-value {'bad' if error_factors else 'good'}">{len(error_factors)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Unknown factors in config</div>
                    <div class="metric-value {'bad' if unknown_factors else 'good'}">{len(unknown_factors)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Features with missing output columns</div>
                    <div class="metric-value {'warning' if missing_feature_outputs else 'good'}">{len(missing_feature_outputs)}</div>
                </div>
            </div>

            <h3>Factor Resolution (Config → DataFrame Columns)</h3>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>DataFrame Column</th>
                        <th>Requested As</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Render factor mappings (limit number of rows for readability)
    mapping_items = list(sorted(factor_mappings.items()))
    for col, sources in mapping_items[:200]:
        html_content += f"""
                    <tr>
                        <td><code>{col}</code></td>
                        <td>{", ".join(sorted(sources))}</td>
                    </tr>
        """

    if len(mapping_items) > 200:
        html_content += f"""
                    <tr>
                        <td colspan="2" style="color: #999;">... {len(mapping_items) - 200} more mappings not shown ...</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
            </div>

            <h3>Unknown Factors in Config</h3>
    """

    if unknown_factors:
        html_content += "<ul>"
        for f in unknown_factors:
            html_content += f"<li><code>{f}</code></li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No unknown factors. All requested factors were resolved to DataFrame columns or features.</p>"

    # Missing feature outputs
    html_content += """

            <h3>Features With Missing Output Columns</h3>
    """
    if missing_feature_outputs:
        html_content += "<ul>"
        for item in missing_feature_outputs:
            feature = item.get("feature")
            expected = item.get("expected", [])
            html_content += f"<li><code>{feature}</code> &mdash; expected columns not found in DataFrame: {', '.join(expected)}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>All requested features that were recognized had at least one output column present in the DataFrame.</p>"

    # Error factors (non-numeric, insufficient samples, etc.)
    html_content += """

            <h3>Problematic Factors (Errors During Evaluation)</h3>
    """

    if error_factors:
        html_content += """
            <table>
                <thead>
                    <tr>
                        <th>Factor</th>
                        <th>Error Type</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        """
        for factor, m in error_factors.items():
            error_type = m.get("error", "unknown")
            details_parts = []
            if error_type == "non_numeric_factor":
                nnr = m.get("non_na_ratio")
                if nnr is not None:
                    details_parts.append(f"non-NaN ratio after coercion = {nnr:.2f}")
                samples = m.get("sample_values") or []
                if samples:
                    details_parts.append(
                        f"sample raw values: {', '.join(map(str, samples[:10]))}"
                    )
            elif error_type == "non_numeric_target":
                nnr = m.get("non_na_ratio")
                if nnr is not None:
                    details_parts.append(
                        f"target non-NaN ratio after coercion = {nnr:.2f}"
                    )
                samples = m.get("sample_values_target") or []
                if samples:
                    details_parts.append(
                        f"sample raw target values: {', '.join(map(str, samples[:10]))}"
                    )
            elif error_type == "factor_missing":
                avail = m.get("available_columns") or []
                if avail:
                    details_parts.append(
                        f"first available feature columns: {', '.join(map(str, avail[:15]))}"
                    )
            elif error_type == "insufficient_samples":
                n = m.get("n_samples")
                if n is not None:
                    details_parts.append(f"valid samples after cleaning: {n}")

            details = "; ".join(details_parts) if details_parts else ""
            html_content += f"""
                    <tr>
                        <td><code>{factor}</code></td>
                        <td>{error_type}</td>
                        <td>{details}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        """
    else:
        html_content += "<p>No evaluation errors. All evaluated factors passed basic numeric and sample-size checks.</p>"

    # ------------------------------------------------------------------
    # Auto summary of top factors (based on IC IR + Sharpe)
    # ------------------------------------------------------------------

    # Build a list of factors with valid metrics (no error)
    ranked_factors = []
    for factor, m in results.items():
        if "error" in m:
            continue
        ic_ir_val = float(m.get("ic_ir", 0.0) or 0.0)
        sharpe_val = float(m.get("sharpe_ratio", 0.0) or 0.0)
        ic_mean_val = float(m.get("ic_mean", 0.0) or 0.0)
        ic_t_stat = float(m.get("ic_t_stat", 0.0) or 0.0)
        ic_p_value = float(m.get("ic_p_value", 1.0) or 1.0)
        ranked_factors.append(
            {
                "name": factor,
                "ic_ir": ic_ir_val,
                "sharpe": sharpe_val,
                "ic_mean": ic_mean_val,
                "ic_t_stat": ic_t_stat,
                "ic_p_value": ic_p_value,
            }
        )

    # Positive alpha candidates: IC Mean > 0, IC t-stat > 1.96 (statistically significant)
    # Optional: IC IR > 0 or Sharpe > 0 (loose constraint to keep more candidates)
    top_positive = sorted(
        [
            f
            for f in ranked_factors
            if f["ic_mean"] > 0
            and f["ic_t_stat"] > 1.96
            and (f["ic_ir"] > 0 or f["sharpe"] > 0)
        ],
        key=lambda x: (x["ic_ir"], x["sharpe"]),
        reverse=True,
    )
    # No truncation - show all qualified factors

    # Strong negative factors (potentially useful when reversed):
    # IC Mean < 0, IC t-stat < -1.96 (statistically significant)
    # Optional: IC IR < 0 or Sharpe < 0
    top_negative = sorted(
        [
            f
            for f in ranked_factors
            if f["ic_mean"] < 0
            and f["ic_t_stat"] < -1.96
            and (f["ic_ir"] < 0 or f["sharpe"] < 0)
        ],
        key=lambda x: (abs(x["ic_ir"]), abs(x["sharpe"])),
        reverse=True,
    )
    # No truncation - show all qualified factors

    html_content += """

            <h2>📊 Auto Summary: Qualified Factors</h2>
            <div class="info-box">
                <p><strong>Selection Criteria:</strong></p>
                <ul>
                    <li><strong>Positive Alpha Candidates:</strong> IC Mean > 0, IC t-stat > 1.96 (statistically significant), and (IC IR > 0 or Sharpe > 0)</li>
                    <li><strong>Strong Negative Factors:</strong> IC Mean < 0, IC t-stat < -1.96 (statistically significant), and (IC IR < 0 or Sharpe < 0)</li>
                </ul>
                <p><strong>Interpretation:</strong></p>
                <ul>
                    <li><strong>IC Mean:</strong> Average predictive power (direction and strength). Positive = higher factor value predicts higher future return.</li>
                    <li><strong>IC t-stat:</strong> Statistical significance. |t-stat| > 1.96 means p-value < 0.05 (95% confidence). Higher |t-stat| = more reliable signal.</li>
                    <li><strong>IC IR (Information Ratio):</strong> IC Mean / IC Std. Measures stability of predictive power. Higher = more consistent.</li>
                    <li><strong>Sharpe Ratio:</strong> Risk-adjusted return of simple long-short strategy. > 1.0 is good, > 0 is acceptable.</li>
                </ul>
                <p><strong>Usage:</strong></p>
                <ul>
                    <li><strong>Positive factors:</strong> Use directly as long signals. Higher factor value → higher expected return.</li>
                    <li><strong>Negative factors:</strong> Two options:
                        <ul>
                            <li><strong>Reverse:</strong> Multiply by -1 to create a positive signal (high negative factor → low expected return → reverse to get long signal)</li>
                            <li><strong>Risk filter:</strong> Use as-is to avoid trading when negative factors are extreme (e.g., reduce position size or skip trades)</li>
                        </ul>
                    </li>
                </ul>
                <p><strong>Note:</strong> All qualified factors are shown below (not limited to top 10). A features.yaml file with all qualified factors has been generated for iterative training.</p>
            </div>

            <div class="metrics-grid">
                <div>
                    <h3>Top Positive Alpha Candidates</h3>
    """

    if top_positive:
        html_content += f"""
                    <table>
                        <thead>
                            <tr>
                                <th>Factor</th>
                                <th>IC Mean</th>
                                <th>IC t-stat</th>
                                <th>IC p-value</th>
                                <th>IC IR</th>
                                <th>Sharpe</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        for f in top_positive:
            html_content += f"""
                            <tr>
                                <td><code>{f['name']}</code></td>
                                <td>{f['ic_mean']:.4f}</td>
                                <td>{f['ic_t_stat']:.2f}</td>
                                <td>{f['ic_p_value']:.4f}</td>
                                <td>{f['ic_ir']:.3f}</td>
                                <td>{f['sharpe']:.2f}</td>
                            </tr>
            """
        html_content += """
                        </tbody>
                    </table>
        """
    else:
        html_content += "<p>No positive-alpha candidates (IC Mean > 0, IC t-stat > 1.96, and (IC IR > 0 or Sharpe > 0)) were found in this run.</p>"

    html_content += """
                </div>
                <div>
                    <h3>Strong Negative Factors (Consider Reversing)</h3>
    """

    if top_negative:
        html_content += f"""
                    <table>
                        <thead>
                            <tr>
                                <th>Factor</th>
                                <th>IC Mean</th>
                                <th>IC t-stat</th>
                                <th>IC p-value</th>
                                <th>IC IR</th>
                                <th>Sharpe</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        for f in top_negative:
            html_content += f"""
                            <tr>
                                <td><code>{f['name']}</code></td>
                                <td>{f['ic_mean']:.4f}</td>
                                <td>{f['ic_t_stat']:.2f}</td>
                                <td>{f['ic_p_value']:.4f}</td>
                                <td>{f['ic_ir']:.3f}</td>
                                <td>{f['sharpe']:.2f}</td>
                            </tr>
            """
        html_content += """
                        </tbody>
                    </table>
        """
    else:
        html_content += "<p>No strong negative factors (IC Mean < 0, IC t-stat < -1.96, and (IC IR < 0 or Sharpe < 0)) were detected.</p>"

    html_content += """
                </div>
            </div>

            <!-- Main summary table -->

            <h2>📈 Summary Metrics</h2>
            <div class="info-box">
                <strong>Color Coding:</strong>
                <span class="good">Green</span> = Good (IC IR > 0.5, Sharpe > 1.0) |
                <span class="warning">Orange</span> = Warning (IC IR > 0, Sharpe > 0) |
                <span class="bad">Red</span> = Poor
            </div>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Factor</th>
                        <th>N</th>
                        <th>Rank IC</th>
                        <th>Pearson</th>
                        <th>IC Mean</th>
                        <th>IC Std</th>
                        <th>IC IR</th>
                        <th>IC+ %</th>
                        <th>IC t-stat</th>
                        <th>IC p-value</th>
                        <th>Best Lag</th>
                        <th>Best IC</th>
                        <th>Q Spread</th>
                        <th>WR Long</th>
                        <th>WR Short</th>
                        <th>Ret Long</th>
                        <th>Ret Short</th>
                        <th>Long t-stat</th>
                        <th>Short t-stat</th>
                        <th>Sharpe</th>
                        <th>Calmar</th>
                        <th>Sortino</th>
                        <th>Tot Ret</th>
                        <th>Max DD</th>
                        <th>DD Dur</th>
                        <th>Turnover</th>
                        <th>F Autocorr</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Inject summary rows
    html_content += "".join(summary_rows)

    html_content += """
                </tbody>
            </table>
            </div>

    """

    # Inject detailed sections for each factor
    html_content += "".join(detailed_sections)

    html_content += """
            <h2>📚 Metrics Explanation</h2>
            <div class="info-box">
                <h4>IC (Information Coefficient) Metrics:</h4>
                <ul>
                    <li><strong>Rank IC:</strong> Spearman rank correlation between factor and target</li>
                    <li><strong>Pearson IC:</strong> Pearson correlation between factor and target</li>
                    <li><strong>IC Mean/Std:</strong> Mean and std of rolling IC series (60-bar window)</li>
                    <li><strong>IC IR:</strong> Information Ratio = IC Mean / IC Std (higher is better, >0.5 is good)</li>
                    <li><strong>IC+ Ratio:</strong> Percentage of periods with positive IC</li>
                    <li><strong>IC t-stat/p-value:</strong> Statistical significance test of IC (p < 0.05 is significant)</li>
                    <li><strong>IC Lag N:</strong> IC at N bars forward (decay analysis - shows how IC decays over time)</li>
                </ul>

                <h4>Quantile Analysis:</h4>
                <ul>
                    <li><strong>Q Spread:</strong> Quantile spread = Long return - Short return (higher is better)</li>
                    <li><strong>Ret Long/Short:</strong> Average return of top/bottom quantile positions</li>
                    <li><strong>Long/Short t-stat:</strong> Statistical significance of quantile returns</li>
                    <li><strong>WR Long/Short:</strong> Win rate of long/short positions</li>
                </ul>

                <h4>Risk-Adjusted Returns:</h4>
                <ul>
                    <li><strong>Sharpe Ratio:</strong> (Mean return / Std) * sqrt(252) (higher is better, >1.0 is good)</li>
                    <li><strong>Calmar Ratio:</strong> Total return / Max drawdown (higher is better)</li>
                    <li><strong>Sortino Ratio:</strong> Downside risk-adjusted Sharpe (uses only negative returns)</li>
                </ul>

                <h4>Factor Characteristics:</h4>
                <ul>
                    <li><strong>Turnover:</strong> Position change rate (lower = more stable signals)</li>
                    <li><strong>F Autocorr:</strong> Factor autocorrelation (higher = more persistent/stable)</li>
                    <li><strong>Max DD Duration:</strong> Maximum drawdown duration in periods</li>
                </ul>

                <h4>📝 Using the Generated features.yaml:</h4>
                <p>This evaluation has generated a <code>features_suggested.yaml</code> file containing all qualified factors (positive + negative).</p>
                <ol>
                    <li><strong>Review the qualified factors</strong> in the "Auto Summary" section above</li>
                    <li><strong>Locate the generated file:</strong> <code>results/factor_ts_eval/{strategy_name}_{symbol}_features_suggested.yaml</code></li>
                    <li><strong>Replace your strategy's features.yaml:</strong> Copy the generated file to your strategy config directory:
                        <pre>cp results/factor_ts_eval/{strategy_name}_{symbol}_features_suggested.yaml config/strategies/{strategy_name}/features.yaml</pre>
                    </li>
                    <li><strong>Iterative refinement:</strong>
                        <ul>
                            <li>Train your model with the new features.yaml</li>
                            <li>Evaluate performance using <code>ts-strategy-feature-compare</code></li>
                            <li>Re-run <code>ts-factor-eval</code> to refine factor selection</li>
                            <li>Repeat until optimal factor combination is found</li>
                        </ul>
                    </li>
                </ol>
                <p><strong>Tip:</strong> Start with all qualified factors, then use ablation studies (<code>ts-strategy-feature-compare</code>) to identify the optimal subset.</p>
            </div>

            <div class="footer">
                <p>Generated by ts-factor-eval | Time-Series Factor Evaluation Tool</p>
            </div>
        </div>

        <script>
            // Initialize all IC decay charts
        </script>
    </body>
    </html>
    """

    # Inject chart scripts after f-string is defined (since chart_scripts is generated in the loop above)
    html_content = html_content.replace(
        "// Initialize all IC decay charts",
        f"// Initialize all IC decay charts\n            {''.join(chart_scripts)}",
    )

    html_path.write_text(html_content, encoding="utf-8")

    # Return qualified factors for YAML export
    qualified_factors = {
        "positive": [f["name"] for f in top_positive],
        "negative": [f["name"] for f in top_negative],
    }
    return html_path, qualified_factors


def export_features_yaml(
    qualified_factors: Dict[str, List[str]],
    strategy_name: str,
    symbol: str,
    output_dir: Path,
    strategy_config_path: Path | None = None,
) -> Path:
    """Export qualified factors to a features.yaml file that can be used for training."""
    import yaml
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

    # Combine positive and negative factors (negative factors can be used as-is or reversed)
    all_factors = qualified_factors["positive"] + qualified_factors["negative"]

    if not all_factors:
        print("   ⚠️  No qualified factors to export.")
        return None

    # Map output columns to feature compute function names (with _f suffix)
    # Some factors are output columns (e.g., bb_lower, bb_middle) that need to be mapped to feature compute functions (e.g., bb_width_f)
    feature_loader = StrategyFeatureLoader()
    feature_definitions = feature_loader.feature_deps.get("features", {})

    # Build mapping: output_column -> feature_compute_function_name (with _f suffix)
    output_to_feature = {}
    for feat_name, feat_info in feature_definitions.items():
        output_cols = feat_info.get("output_columns", [])
        for col in output_cols:
            if col not in output_to_feature:
                output_to_feature[col] = feat_name

    # Map factors to feature compute function names (only keep feature compute function names, not output columns)
    # This ensures the generated YAML only contains feature compute function names that can be loaded
    mapped_features = set()
    unmapped_factors = []
    mapped_outputs = []  # Track which output columns were mapped

    for factor in all_factors:
        # Check if it's already a feature compute function name (with _f suffix)
        if factor in feature_definitions:
            mapped_features.add(factor)
        # Check if it's an output column - map to source feature compute function
        elif factor in output_to_feature:
            source_feature = output_to_feature[factor]
            mapped_features.add(source_feature)
            mapped_outputs.append((factor, source_feature))
        else:
            # Not found - check if it's an old name (without _f suffix)
            potential_old_name = factor
            potential_new_name = f"{factor}_f"
            if potential_new_name in feature_definitions:
                # This is an old name without _f suffix - raise error
                raise ValueError(
                    f"Feature compute function name must end with '_f' suffix. "
                    f"Got: '{factor}'. Did you mean '{potential_new_name}'?"
                )
            # Not found - keep it anyway (might be a raw column or deprecated feature)
            unmapped_factors.append(factor)
            mapped_features.add(factor)

    if mapped_outputs:
        print(
            f"   📋 Mapped {len(mapped_outputs)} feature output columns to feature compute functions:"
        )
        for output_col, source_feat in sorted(mapped_outputs)[:5]:
            print(f"      {output_col} → {source_feat}")
        if len(mapped_outputs) > 5:
            print(f"      ... and {len(mapped_outputs) - 5} more")

    if unmapped_factors:
        print(f"   ⚠️  {len(unmapped_factors)} factors not found in feature registry:")
        for factor in sorted(unmapped_factors)[:10]:
            print(f"      - {factor}")
        if len(unmapped_factors) > 10:
            print(f"      ... and {len(unmapped_factors) - 10} more")

    # Load base strategy config if available to preserve other settings
    base_config = {}
    if strategy_config_path and strategy_config_path.exists():
        try:
            with open(strategy_config_path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"   ⚠️  Could not load base strategy config: {e}")

    # Build features.yaml structure
    features_config = {
        "name": base_config.get("name", strategy_name),
        "description": base_config.get(
            "description",
            f"Feature pipeline with qualified factors from factor evaluation ({symbol})",
        ),
        "feature_pipeline": {
            "requested_features": sorted(mapped_features),
            "ensure_signal_column": base_config.get("feature_pipeline", {}).get(
                "ensure_signal_column",
                {"name": "signal", "default_value": 0},
            ),
        },
        "notes": f"""
Auto-generated from factor evaluation results.
- Positive factors ({len(qualified_factors['positive'])}): Direct alpha signals
- Negative factors ({len(qualified_factors['negative'])}): Consider reversing (multiply by -1) or use as risk filters

Selection criteria:
- Positive: IC Mean > 0, IC t-stat > 1.96, (IC IR > 0 or Sharpe > 0)
- Negative: IC Mean < 0, IC t-stat < -1.96, (IC IR < 0 or Sharpe < 0)

Note: Feature output columns (e.g., bb_lower, bb_middle) have been mapped to their source feature compute functions (e.g., bb_width_f).
Only feature compute function names (with _f suffix) are included in requested_features for cleaner configuration.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip(),
    }

    # Write YAML file
    yaml_path = output_dir / f"{strategy_name}_{symbol}_features_suggested.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            features_config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    return yaml_path


def main() -> None:
    args = parse_args()
    strategy_config_path = Path(args.strategy_config)

    # Support both directory and file paths
    # If it's a file (e.g., features_all.yaml), use its parent directory as strategy config
    # and temporarily use that file as features.yaml
    features_file_override = None
    if strategy_config_path.is_file():
        # It's a features.yaml file, extract the directory
        features_file_override = strategy_config_path
        strategy_config_dir = strategy_config_path.parent
        print(f"📁 Detected features file: {features_file_override}")
        print(f"   Using strategy config directory: {strategy_config_dir}")
    else:
        # It's a directory, use default features.yaml
        strategy_config_dir = strategy_config_path

    loader = StrategyConfigLoader(strategy_config_dir)
    strategy_cfg = loader.load()

    # If a features file was specified, override the features config
    if features_file_override:
        import yaml

        with open(features_file_override, "r", encoding="utf-8") as f:
            features_override = yaml.safe_load(f)
        # Override the features configuration
        if "feature_pipeline" in features_override:
            strategy_cfg.features.requested_features = features_override[
                "feature_pipeline"
            ].get("requested_features", [])
            print(
                f"   ✅ Loaded {len(strategy_cfg.features.requested_features)} features from {features_file_override.name}"
            )

    # If factors not specified, use requested_features from strategy config
    if args.factors is None:
        strategy_requested = strategy_cfg.features.requested_features or []
        if not strategy_requested:
            raise ValueError(
                f"No factors specified and strategy config '{args.strategy_config}' "
                f"has no requested_features in features.yaml. "
                f"Please either specify --factors or add requested_features to the config."
            )
        args.factors = strategy_requested
        print(f"📋 Using factors from strategy config: {len(args.factors)} factors")
        print(
            f"   Factors: {', '.join(args.factors[:10])}{'...' if len(args.factors) > 10 else ''}"
        )

    df = prepare_dataset(args, strategy_cfg)
    target_col = strategy_cfg.labels.target_column

    # Parse IC decay lags
    ic_decay_lags = [int(x.strip()) for x in args.ic_decay_lags.split(",") if x.strip()]

    results: Dict[str, Dict[str, float]] = {}
    ic_series_data: Dict[str, pd.DataFrame] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve factors:
    # - 如果因子名是 DataFrame 中的列名：直接使用该列
    # - 如果因子名是特征名（存在于 feature_dependencies.yaml）：
    #     使用其 output_columns 中「实际存在于 DataFrame」的列
    # - 同一个列名只计算一次（即使既写了特征名又写了列名）
    # ------------------------------------------------------------------
    feature_loader = StrategyFeatureLoader()
    features_config = feature_loader.feature_deps.get("features", {})

    # Diagnostics containers
    diagnostics: Dict[str, Any] = {}
    factor_resolution: Dict[str, Any] = {
        "mappings": {},  # col_name -> set(sources)
        "unknown_factors": [],
        "missing_feature_outputs": [],
    }

    # col_name -> set(sources)  例如: "hilbert_price_env" <- {"hilbert_phase", "hilbert_price_env"}
    cols_to_evaluate: Dict[str, set] = {}

    # Build reverse mapping: output_column -> feature_compute_function_name
    output_to_feature = {}
    for feat_name, feat_info in features_config.items():
        output_cols = feat_info.get("output_columns", [])
        for col in output_cols:
            if col not in output_to_feature:
                output_to_feature[col] = feat_name

    for factor in args.factors:
        # 1. Check if it's a DataFrame column
        if factor in df.columns:
            cols_to_evaluate.setdefault(factor, set()).add(factor)
            continue

        # 2. Check if it's a feature compute function name (with _f suffix)
        if factor in features_config:
            feature_info = features_config[factor]
            output_cols = feature_info.get("output_columns", [factor])
            existing_cols = [c for c in output_cols if c in df.columns]

            if not existing_cols:
                msg = {
                    "feature": factor,
                    "expected": output_cols,
                }
                factor_resolution["missing_feature_outputs"].append(msg)
                print(
                    f"   ❌ Feature compute function '{factor}' found in config, but none of its output columns "
                    f"exist in DataFrame. Expected: {output_cols}"
                )
                continue

            for col in existing_cols:
                cols_to_evaluate.setdefault(col, set()).add(factor)
        # 3. Check if it's a feature output column name (reverse lookup)
        elif factor in output_to_feature:
            source_feature = output_to_feature[factor]
            # Check if this output column exists in DataFrame
            if factor in df.columns:
                cols_to_evaluate.setdefault(factor, set()).add(source_feature)
            else:
                # Output column doesn't exist in DataFrame
                feature_info = features_config[source_feature]
                output_cols = feature_info.get("output_columns", [])
                msg = {
                    "feature": source_feature,
                    "expected": output_cols,
                }
                factor_resolution["missing_feature_outputs"].append(msg)
                print(
                    f"   ❌ Feature output column '{factor}' (from feature compute function '{source_feature}') "
                    f"does not exist in DataFrame. Expected output columns: {output_cols}"
                )
        else:
            # 4. Not found - check if it's an old feature name (without _f suffix)
            potential_old_name = factor
            potential_new_name = f"{factor}_f"
            if potential_new_name in features_config:
                factor_resolution["unknown_factors"].append(factor)
                print(
                    f"   ❌ Factor '{factor}' is an old feature compute function name (without '_f' suffix). "
                    f"Did you mean '{potential_new_name}'? "
                    f"Or if you want to use the output column, please check if '{factor}' is in the output columns of '{potential_new_name}'."
                )
            else:
                factor_resolution["unknown_factors"].append(factor)
                print(
                    f"   ❌ Factor '{factor}' is neither a DataFrame column nor a known feature compute function "
                    f"(with '_f' suffix) or feature output column in feature_dependencies.yaml"
                )

    if not cols_to_evaluate:
        print("   ❌ No valid factors/columns to evaluate. Exiting.")
        return

    # 打印映射关系，方便调试
    print("\n   🔍 Factor resolution (config entry -> actual DataFrame columns):")
    for col, sources in sorted(cols_to_evaluate.items()):
        print(f"      - {col}: requested as {sorted(sources)}")

    # 保存映射到 diagnostics (convert sets to lists for JSON serialization)
    factor_resolution["mappings"] = {
        col: sorted(list(sources)) for col, sources in cols_to_evaluate.items()
    }
    diagnostics["factor_resolution"] = factor_resolution

    # 逐列计算指标（每个实际列只计算一次）
    for col in cols_to_evaluate.keys():
        if col not in df.columns:
            # 理论上不会发生，这里仅做保护
            print(f"   ❌ Column '{col}' not found in DataFrame (skipped).")
            continue

        metrics, ic_series_df = compute_factor_metrics(
            df, col, target_col, args.quantile, ic_decay_lags
        )
        results[col] = metrics
        if not ic_series_df.empty:
            ic_series_data[col] = ic_series_df
            # Save IC series to CSV
            ic_csv_path = output_dir / f"ic_series_{col}_{args.symbol}.csv"
            ic_series_df.to_csv(ic_csv_path)
            print(f"   💾 Saved IC series to {ic_csv_path}")

    # Collect error factors for diagnostics
    error_factors = {f: m for f, m in results.items() if "error" in m}
    diagnostics["error_factors"] = error_factors

    # Save JSON summary
    summary_path = output_dir / f"ts_eval_{strategy_cfg.name}_{args.symbol}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "strategy": strategy_cfg.name,
                "symbol": args.symbol,
                "ic_decay_lags": ic_decay_lags,
                "results": results,
                "diagnostics": diagnostics,
            },
            fh,
            indent=2,
        )

    print(f"✅ Saved time-series factor evaluation to {summary_path}")

    # Generate HTML report
    qualified_factors = None
    if args.generate_html:
        html_path, qualified_factors = generate_html_report(
            results,
            ic_series_data,
            strategy_cfg.name,
            args.symbol,
            output_dir,
            ic_decay_lags,
            diagnostics=diagnostics,
        )
        print(f"✅ Generated HTML report: {html_path}")

        # Export features.yaml with qualified factors
        if qualified_factors:
            strategy_config_path = Path(args.strategy_config) / "features.yaml"
            yaml_path = export_features_yaml(
                qualified_factors,
                strategy_cfg.name,
                args.symbol,
                output_dir,
                strategy_config_path=strategy_config_path,
            )
            if yaml_path:
                print(f"✅ Exported qualified factors to: {yaml_path}")
                print(f"   - Positive factors: {len(qualified_factors['positive'])}")
                print(f"   - Negative factors: {len(qualified_factors['negative'])}")
                print(
                    f"   - Total: {len(qualified_factors['positive']) + len(qualified_factors['negative'])}"
                )
                print(
                    f"   💡 You can use this file to replace your strategy's features.yaml for iterative training."
                )

        # Open in browser if requested (only in non-Docker/local environment)
        if args.open_browser:
            # Check if running in Docker (common indicators)
            in_docker = (
                os.path.exists("/.dockerenv")
                or os.environ.get("DOCKER_CONTAINER") == "1"
                or os.environ.get("DEV_CONTAINER") == "1"
            )

            if in_docker:
                print(f"⚠️  Running in Docker - cannot open browser automatically")
                print(f"   Please open manually: {html_path}")
                # Try to print a local path if mounted
                local_path = str(html_path).replace("/workspace/", "")
                if local_path != str(html_path):
                    print(f"   Local path (if mounted): {local_path}")
            else:
                try:
                    # Convert to absolute path and use file:// URL
                    abs_path = html_path.resolve()
                    file_url = f"file://{abs_path}"
                    webbrowser.open(file_url)
                    print(f"🌐 Opened report in browser: {file_url}")
                except Exception as e:
                    print(f"⚠️  Could not open browser automatically: {e}")
                    print(f"   Please open manually: {html_path}")


if __name__ == "__main__":
    main()
