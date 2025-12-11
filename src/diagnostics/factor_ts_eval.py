#!/usr/bin/env python3
"""Single-factor time-series evaluation helper with IC decay and HTML reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
    StrategyFeatureLoader, )  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate individual factors via the strategy pipeline")
    parser.add_argument("--strategy-config",
                        required=True,
                        help="Path to strategy dir")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--factors",
                        nargs="+",
                        required=True,
                        help="Factor columns")
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="15T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--quantile",
                        type=float,
                        default=0.2,
                        help="Top/Bottom quantile")
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
        print(
            f"⚠️  Warning: Some requested factors were not found: {missing_features}"
        )

        # Check each missing feature's output_columns configuration
        features_config = feature_loader.feature_deps.get("features", {})
        for feature_name in missing_features:
            if feature_name in features_config:
                feature_info = features_config[feature_name]
                expected_outputs = feature_info.get("output_columns",
                                                    [feature_name])
                found_outputs = [
                    col for col in expected_outputs
                    if col in df_features.columns
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
                            print(
                                f"      ⚠️  Missing required columns: {missing_req}"
                            )
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
        strategy_requested = strategy_cfg.features.get("requested_features",
                                                       [])

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
                macd_variants = [
                    "macd", "macd_signal", "macd_histogram", "macd_hist"
                ]
                if factor in macd_variants:
                    # If user requests "macd", check if any MACD column is in strategy
                    if any(variant in strategy_requested_set
                           for variant in macd_variants):
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
            raise ValueError(
                "--feature-mode=only requires --factors to be specified")
        df_features = _compute_requested_features(
            df_raw,
            feature_loader,
            extra_factors,
            strategy_cfg.features.ensure_signal,
        )
    elif feature_mode == "append":
        if not extra_factors:
            raise ValueError(
                "--feature-mode=append requires --factors to be specified")
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
                output_cols = feature_info.get("output_columns",
                                               [feature_name])
                for output_col in output_cols:
                    if (output_col in requested_df.columns
                            and output_col not in base_features.columns):
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
        df_features.copy(), **strategy_cfg.labels.generator.params)

    df_filtered = strategy_runner.apply_filters(df_features,
                                                strategy_cfg.labels.filters)
    df_filtered = strategy_runner.apply_post_label_filters(
        df_filtered,
        strategy_cfg.labels.post_label_filters,
        list(df_filtered.columns),
    )
    return df_filtered


def compute_ic_series(df: pd.DataFrame,
                      factor: str,
                      target_col: str,
                      window: int = 60) -> pd.Series:
    """Compute rolling IC series."""
    valid = df[[factor, target_col]].dropna()
    if len(valid) < window:
        return pd.Series(dtype=float)

    ic_series = []
    for i in range(window, len(valid) + 1):
        window_data = valid.iloc[i - window:i]
        factor_ranks = window_data[factor].rank(pct=True)
        target_ranks = window_data[target_col].rank(pct=True)
        ic = spearmanr(factor_ranks, target_ranks).correlation
        if not np.isnan(ic):
            ic_series.append(ic)
        else:
            ic_series.append(0.0)

    return pd.Series(ic_series, index=valid.index[window - 1:])


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
        decay_metrics[f"ic_lag_{lag}"] = (float(rank_ic)
                                          if not np.isnan(rank_ic) else np.nan)
        decay_metrics[f"ic_lag_{lag}_pearson"] = (
            float(pearson_ic) if not np.isnan(pearson_ic) else np.nan)

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
        available = sorted([
            c for c in df.columns if c not in [
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
        ])[:30]
        metrics["available_columns"] = available
        print(f"   ❌ Factor '{factor}' not found in DataFrame columns")
        print(f"      Available feature columns: {available[:15]}...")
        return metrics, ic_series_df

    valid = df[[factor, target_col]].dropna()
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
    ic_series = compute_ic_series(df, factor, target_col, window=60)
    if len(ic_series) > 0:
        metrics["ic_mean"] = float(ic_series.mean())
        metrics["ic_std"] = float(ic_series.std())
        metrics["ic_ir"] = (metrics["ic_mean"] / metrics["ic_std"]
                            if metrics["ic_std"] > 0 else 0.0)
        metrics["ic_positive_ratio"] = float((ic_series > 0).mean())
        ic_series_df = pd.DataFrame({"ic": ic_series})
    else:
        metrics["ic_mean"] = float(rank_ic) if not np.isnan(rank_ic) else 0.0
        metrics["ic_std"] = 0.0
        metrics["ic_ir"] = 0.0
        metrics["ic_positive_ratio"] = 0.0

    # IC decay analysis
    if ic_decay_lags:
        decay_metrics = compute_ic_decay(df, factor, target_col, ic_decay_lags)
        metrics.update(decay_metrics)

    # Quantile analysis
    high_cut = valid[factor].quantile(1 - quantile)
    low_cut = valid[factor].quantile(quantile)

    long_mask = valid[factor] >= high_cut
    short_mask = valid[factor] <= low_cut

    long_returns = valid.loc[long_mask, target_col]
    short_returns = valid.loc[short_mask, target_col]

    win_rate_long = float(
        (long_returns > 0).mean()) if len(long_returns) else 0.0
    win_rate_short = float(
        (short_returns < 0).mean()) if len(short_returns) else 0.0

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
            max_dd_duration = (int(dd_periods.value_counts().max())
                               if len(dd_periods) > 0 else 0)

    # Risk-adjusted returns
    strategy_ret_clean = strategy_ret[strategy_ret != 0]
    if len(strategy_ret_clean) > 0:
        ret_std = float(strategy_ret_clean.std())
        ret_mean = float(strategy_ret_clean.mean())
        sharpe_ratio = (ret_mean / ret_std *
                        np.sqrt(252)) if ret_std > 1e-8 else 0.0
        calmar_ratio = (total_return / abs(max_dd)) if max_dd < -1e-8 else 0.0
        # Sortino ratio (downside deviation)
        downside_returns = strategy_ret_clean[strategy_ret_clean < 0]
        downside_std = (float(downside_returns.std())
                        if len(downside_returns) > 0 else 0.0)
        sortino_ratio = ((ret_mean / downside_std *
                          np.sqrt(252)) if downside_std > 1e-8 else 0.0)
    else:
        sharpe_ratio = 0.0
        calmar_ratio = 0.0
        sortino_ratio = 0.0

    # Quantile spread (long-short return difference)
    quantile_spread = (float(long_returns.mean() -
                             short_returns.mean()) if len(long_returns) > 0
                       and len(short_returns) > 0 else 0.0)

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
        metrics["long_return_skew"] = (float(skew(long_returns))
                                       if len(long_returns) > 2 else 0.0)
        metrics["long_return_kurtosis"] = (float(kurtosis(long_returns))
                                           if len(long_returns) > 3 else 0.0)
    else:
        metrics["long_return_std"] = 0.0
        metrics["long_return_skew"] = 0.0
        metrics["long_return_kurtosis"] = 0.0

    if len(short_returns) > 0:
        metrics["short_return_std"] = float(short_returns.std())
        metrics["short_return_skew"] = (float(skew(short_returns))
                                        if len(short_returns) > 2 else 0.0)
        metrics["short_return_kurtosis"] = (float(kurtosis(short_returns))
                                            if len(short_returns) > 3 else 0.0)
    else:
        metrics["short_return_std"] = 0.0
        metrics["short_return_skew"] = 0.0
        metrics["short_return_kurtosis"] = 0.0

    # Factor distribution stats
    factor_stats = valid[factor].describe()
    factor_skewness = float(skew(valid[factor])) if len(valid) > 2 else 0.0
    factor_kurt = float(kurtosis(valid[factor])) if len(valid) > 3 else 0.0

    # Factor autocorrelation (stability measure)
    factor_autocorr = float(valid[factor].autocorr(
        lag=1)) if len(valid) > 1 else 0.0

    # Turnover calculation (position change rate)
    position_changes = ((position.diff().abs().sum() /
                         len(position)) if len(position) > 1 else 0.0)
    turnover = float(position_changes)

    # Factor-target correlation distribution stats
    target_stats = valid[target_col].describe()

    metrics.update({
        "n_samples":
        int(len(valid)),
        "rank_ic":
        float(rank_ic) if not np.isnan(rank_ic) else 0.0,
        "pearson":
        float(pearson) if not np.isnan(pearson) else 0.0,
        "win_rate_long":
        win_rate_long,
        "win_rate_short":
        win_rate_short,
        "avg_return_long":
        float(long_returns.mean()) if len(long_returns) else 0.0,
        "avg_return_short":
        (float(short_returns.mean()) if len(short_returns) else 0.0),
        "quantile_spread":
        quantile_spread,
        "total_return":
        total_return,
        "max_drawdown":
        max_dd,
        "max_drawdown_duration":
        max_dd_duration,
        "sharpe_ratio":
        sharpe_ratio,
        "calmar_ratio":
        calmar_ratio,
        "sortino_ratio":
        sortino_ratio,
        "turnover":
        turnover,
        "factor_mean":
        float(factor_stats["mean"]),
        "factor_std":
        float(factor_stats["std"]),
        "factor_min":
        float(factor_stats["min"]),
        "factor_max":
        float(factor_stats["max"]),
        "factor_skewness":
        factor_skewness,
        "factor_kurtosis":
        factor_kurt,
        "factor_autocorr":
        factor_autocorr,
        "target_mean":
        float(target_stats["mean"]),
        "target_std":
        float(target_stats["std"]),
    })
    return metrics, ic_series_df


def generate_html_report(
    results: Dict[str, Dict[str, float]],
    ic_series_data: Dict[str, pd.DataFrame],
    strategy_name: str,
    symbol: str,
    output_dir: Path,
    ic_decay_lags: List[int],
) -> Path:
    """Generate comprehensive HTML report."""
    html_path = output_dir / f"ts_eval_{strategy_name}_{symbol}.html"

    # Build summary table
    summary_rows = []
    for factor, metrics in results.items():
        if "error" in metrics:
            summary_rows.append(f"""
                <tr>
                    <td><strong>{factor}</strong></td>
                    <td colspan="12" style="color: red;">Error: {metrics['error']}</td>
                </tr>
                """)
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
        ic_ir_class = "good" if ic_ir > 0.5 else (
            "warning" if ic_ir > 0 else "bad")

        sharpe = metrics.get("sharpe_ratio", 0.0)
        sharpe_class = "good" if sharpe > 1.0 else (
            "warning" if sharpe > 0 else "bad")

        quantile_spread = metrics.get("quantile_spread", 0.0)
        spread_class = "good" if quantile_spread > 0 else "bad"

        best_lag_display = f"{best_lag}" if best_lag is not None else "N/A"
        best_ic_display = f"{best_ic:.4f}" if best_ic is not None else "N/A"
        best_ic_class = ("good" if
                         (best_ic is not None and best_ic > 0.05) else
                         ("warning" if
                          (best_ic is not None and best_ic > 0) else "bad"))

        summary_rows.append(f"""
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
            """)

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
                    <div class="metric-value" style="color: {'#4CAF50' if best_ic > 0 else '#f44336'}">{best_lag if best_lag else 'N/A'}</div>
                    <div>IC: {best_ic:.4f} | Bars forward</div>
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
                    {''.join(summary_rows)}
                </tbody>
            </table>
            </div>
            
            {''.join(detailed_sections)}

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
            </div>

            <div class="footer">
                <p>Generated by ts-factor-eval | Time-Series Factor Evaluation Tool</p>
            </div>
        </div>
        
        <script>
            // Initialize all IC decay charts
            {''.join(chart_scripts)}
        </script>
    </body>
    </html>
    """

    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def main() -> None:
    args = parse_args()
    loader = StrategyConfigLoader(Path(args.strategy_config))
    strategy_cfg = loader.load()

    df = prepare_dataset(args, strategy_cfg)
    target_col = strategy_cfg.labels.target_column

    # Parse IC decay lags
    ic_decay_lags = [
        int(x.strip()) for x in args.ic_decay_lags.split(",") if x.strip()
    ]

    results = {}
    ic_series_data = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for factor in args.factors:
        metrics, ic_series_df = compute_factor_metrics(df, factor, target_col,
                                                       args.quantile,
                                                       ic_decay_lags)
        results[factor] = metrics
        if not ic_series_df.empty:
            ic_series_data[factor] = ic_series_df
            # Save IC series to CSV
            ic_csv_path = output_dir / f"ic_series_{factor}_{args.symbol}.csv"
            ic_series_df.to_csv(ic_csv_path)
            print(f"   💾 Saved IC series to {ic_csv_path}")

    # Save JSON summary
    summary_path = output_dir / f"ts_eval_{strategy_cfg.name}_{args.symbol}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "strategy": strategy_cfg.name,
                "symbol": args.symbol,
                "ic_decay_lags": ic_decay_lags,
                "results": results,
            },
            fh,
            indent=2,
        )

    print(f"✅ Saved time-series factor evaluation to {summary_path}")

    # Generate HTML report
    if args.generate_html:
        html_path = generate_html_report(
            results,
            ic_series_data,
            strategy_cfg.name,
            args.symbol,
            output_dir,
            ic_decay_lags,
        )
        print(f"✅ Generated HTML report: {html_path}")


if __name__ == "__main__":
    main()
