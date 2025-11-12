#!/usr/bin/env python3
"""Correlation-based timeframe and forward horizon selection for time-series models.

This script evaluates, for each symbol, how predictive a collection of fast-to-compute
technical/statistical features are for future returns across multiple timeframes and
forward horizons. The analysis relies on Pearson and Spearman correlations between the
features and forward returns, producing both detailed metrics and a markdown report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from datetime import datetime
from data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)

AGG_MAP: Dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "trade_count": "sum",
    "buy_qty": "sum",
    "sell_qty": "sum",
    "taker_buy_ratio": "mean",
    "cvd": "last",
    "cvd_short": "last",
    "cvd_medium": "last",
    "cvd_long": "last",
    "cvd_change_1": "mean",
    "cvd_change_5": "mean",
    "cvd_change_20": "mean",
    "cvd_normalized": "mean",
    "symbol": "last",
}

SUPPORTED_EXTRA_FEATURES = {
    "sma_ratio_5_20",
    "ema_ratio_12_26",
    "rsi_14",
    "macd_diff",
    "macd_signal",
    "bb_zscore_20",
    "atr_14",
    "stoch_k_14_3",
    "stoch_d_14_3",
}


@dataclass
class AnalysisConfig:
    data_dir: Path
    symbols: Sequence[str]
    timeframes: Sequence[str]
    forward_bars: Sequence[int]
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    max_lag: int
    min_samples: int
    pearson_threshold: float
    output_dir: Path
    top_k: int
    markdown_report: Path | None
    html_report: Path
    run_tag: str
    feature_type: str
    extra_features: Sequence[str]


def _normalise_timeframe(timeframe: str) -> str:
    tf = timeframe.strip()
    if tf.endswith("T"):
        return tf[:-1] + "min"
    return tf


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(
        description=
        "Select optimal timeframe and forward bars via correlation analysis.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/parquet_data"),
        help="Directory containing per-symbol parquet files.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to analyse (space-separated).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5T", "15T", "30T", "60T", "120T", "240T"],
        help="Candidate pandas timeframes (e.g. 5T 15T 60T).",
    )
    parser.add_argument(
        "--forward-bars",
        nargs="+",
        type=int,
        default=[1, 3, 6, 12, 24],
        help="Forward horizons (number of bars) to evaluate.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (inclusive, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (inclusive, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Number of lagged return features to include.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum overlapping observations required for correlation.",
    )
    parser.add_argument(
        "--pearson-threshold",
        type=float,
        default=0.25,
        help="Absolute correlation threshold for highlighting strong signals.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="baseline",
        help=(
            "Feature bundle to use from ComprehensiveFeatureEngineer "
            "(e.g. baseline, default, enhanced, order_flow, comprehensive, or combinations)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/timeframe_forward"),
        help="Directory to store detailed metrics and report.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help=
        "Optional path for markdown report. Defaults to <output-dir>/timeframe_forward_report.md if not provided.",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help=
        "Path for the HTML report (defaults to <output-dir>/timeframe_forward_report.html).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top combinations per symbol to highlight in the markdown report.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional label for this run; defaults to current timestamp.",
    )
    parser.add_argument(
        "--extra-features",
        nargs="+",
        default=[],
        help=("Optional additional feature names to include. "
              f"Supported: {', '.join(sorted(SUPPORTED_EXTRA_FEATURES))}."),
    )

    args = parser.parse_args()

    start_ts = pd.Timestamp(args.start) if args.start else None
    end_ts = pd.Timestamp(args.end) if args.end else None

    if start_ts and end_ts and start_ts > end_ts:
        raise ValueError("Start date must be earlier than end date.")

    return AnalysisConfig(
        data_dir=args.data_dir,
        symbols=args.symbols,
        timeframes=args.timeframes,
        forward_bars=args.forward_bars,
        start=start_ts,
        end=end_ts,
        max_lag=args.max_lag,
        min_samples=args.min_samples,
        pearson_threshold=args.pearson_threshold,
        output_dir=args.output_dir,
        top_k=args.top_k,
        markdown_report=args.markdown_report,
        html_report=args.html_report,
        run_tag=args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S"),
        feature_type=args.feature_type.strip(),
        extra_features=_validate_extra_features(args.extra_features),
    )


def _validate_extra_features(extra_features: Sequence[str]) -> List[str]:
    validated: List[str] = []
    for feat in extra_features:
        if feat not in SUPPORTED_EXTRA_FEATURES:
            print(
                f"[WARN] Unsupported extra feature '{feat}' requested; skipping."
            )
            continue
        validated.append(feat)
    return validated


def discover_symbol_files(data_dir: Path, symbol: str) -> List[Path]:
    pattern = f"{symbol}_*.parquet"
    files = sorted(data_dir.glob(pattern))
    return files


def load_symbol_dataframe(
    data_dir: Path,
    symbol: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    files = discover_symbol_files(data_dir, symbol)
    if not files:
        raise FileNotFoundError(
            f"No parquet files found for symbol {symbol} in {data_dir}")

    frames = []
    for file_path in files:
        df = pd.read_parquet(file_path)
        if "timestamp" not in df.columns:
            raise ValueError(f"File {file_path} missing 'timestamp' column.")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("timestamp")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=False)
    data = data.set_index("timestamp")

    if start:
        data = data[data.index >= start]
    if end:
        data = data[data.index <= end]

    if data.empty:
        raise ValueError(
            f"Filtered data for {symbol} is empty after applying date range.")

    return data


def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    pandas_timeframe = _normalise_timeframe(timeframe)
    # Ensure only columns present in AGG_MAP are passed to resample
    available_columns = [col for col in df.columns if col in AGG_MAP]
    missing_cols = set(AGG_MAP.keys()) - set(available_columns)

    if missing_cols:
        # It's acceptable if some optional columns are missing; log by returning only the available ones.
        pass

    agg_map = {col: AGG_MAP[col] for col in available_columns}
    resampled = df[available_columns].resample(pandas_timeframe).agg(agg_map)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled


def _compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=length).mean()
    loss = (-delta.clip(upper=0)).rolling(window=length).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return atr


def _apply_additional_features(
        df: pd.DataFrame,
        feature_names: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    feature_columns: List[str] = []

    for name in feature_names:
        if name == "sma_ratio_5_20":
            sma5 = df["close"].rolling(window=5).mean()
            sma20 = df["close"].rolling(window=20).mean()
            ratio5 = df["close"] / sma5.replace(0, np.nan)
            ratio20 = df["close"] / sma20.replace(0, np.nan)
            df[name] = ratio5 - ratio20
        elif name == "ema_ratio_12_26":
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            df[name] = ema12 / ema26.replace(0, np.nan) - 1
        elif name == "rsi_14":
            df[name] = _compute_rsi(df["close"], length=14)
        elif name in {"macd_diff", "macd_signal"}:
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            if name == "macd_diff":
                df[name] = macd - signal
            else:
                df[name] = signal
        elif name == "bb_zscore_20":
            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std(ddof=0)
            df[name] = (df["close"] - sma20) / std20.replace(0, np.nan)
        elif name == "atr_14":
            required_cols = {"high", "low", "close"}
            if not required_cols.issubset(df.columns):
                print(
                    f"[WARN] Skipping {name}: missing columns {required_cols - set(df.columns)}"
                )
                continue
            df[name] = _compute_atr(df, length=14)
        elif name in {"stoch_k_14_3", "stoch_d_14_3"}:
            required_cols = {"high", "low", "close"}
            if not required_cols.issubset(df.columns):
                print(
                    f"[WARN] Skipping {name}: missing columns {required_cols - set(df.columns)}"
                )
                continue
            lowest_low = df["low"].rolling(window=14).min()
            highest_high = df["high"].rolling(window=14).max()
            denom = (highest_high - lowest_low).replace(0, np.nan)
            stoch_k = 100 * (df["close"] - lowest_low) / denom
            stoch_k_smoothed = stoch_k.rolling(window=3).mean()
            if name == "stoch_k_14_3":
                df[name] = stoch_k_smoothed
            else:
                df[name] = stoch_k_smoothed.rolling(window=3).mean()
        else:
            print(
                f"[WARN] Unsupported extra feature '{name}' requested; skipping."
            )
            continue

        feature_columns.append(name)

    return df, feature_columns


def enrich_features(
    df: pd.DataFrame,
    max_lag: int,
    extra_features: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    work["log_price"] = np.log(work["close"])
    work["return_1"] = work["log_price"].diff()

    feature_columns: List[str] = []

    for lag in range(1, max_lag + 1):
        col = f"return_lag_{lag}"
        work[col] = work["return_1"].shift(lag)
        feature_columns.append(col)

    work["return_vol_3"] = work["return_1"].rolling(window=3).std()
    work["return_vol_12"] = work["return_1"].rolling(window=12).std()
    work["volume_pct_change"] = work["volume"].pct_change()

    with np.errstate(divide="ignore", invalid="ignore"):
        vol_mean = work["volume"].rolling(window=20).mean()
        vol_std = work["volume"].rolling(window=20).std(ddof=0)
        work["volume_zscore_20"] = (work["volume"] - vol_mean) / vol_std

    work["taker_buy_ratio_level"] = work.get("taker_buy_ratio")

    for col in ["cvd", "cvd_short", "cvd_medium", "cvd_long"]:
        if col in work.columns:
            diff_col = f"{col}_diff"
            work[diff_col] = work[col].diff()
            feature_columns.append(diff_col)

    cvd_change_cols = [
        c for c in
        ["cvd_change_1", "cvd_change_5", "cvd_change_20", "cvd_normalized"]
        if c in work.columns
    ]
    feature_columns.extend(cvd_change_cols)

    feature_columns.extend([
        "return_vol_3",
        "return_vol_12",
        "volume_pct_change",
        "volume_zscore_20",
        "taker_buy_ratio_level",
    ])

    if extra_features:
        work, extra_cols = _apply_additional_features(work, extra_features)
        feature_columns.extend(extra_cols)

    # Remove potential duplicates and columns that may not exist
    feature_columns = [
        col for col in dict.fromkeys(feature_columns) if col in work.columns
    ]

    return work, feature_columns


def compute_correlations(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    forward_bars: Sequence[int],
    min_samples: int,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    if not feature_cols:
        return results

    for horizon in forward_bars:
        target_col = f"future_return_{horizon}"
        df[target_col] = df["log_price"].shift(-horizon) - df["log_price"]

        # Ensure target column exists and has valid values
        target_data = df[[target_col]].dropna()
        if len(target_data) < min_samples:
            continue

        target_values_all = df[target_col].values
        if np.isclose(np.std(target_values_all[~np.isnan(target_values_all)]), 0):
            continue

        # Compute correlation for each feature individually
        # This allows features with different NaN patterns to use all available data
        for feature in feature_cols:
            if feature not in df.columns:
                continue
            
            # Get valid pairs for this specific feature-target combination
            feature_values = df[feature].values
            valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values_all))
            
            if valid_mask.sum() < min_samples:
                continue
            
            feature_valid = feature_values[valid_mask]
            target_valid = target_values_all[valid_mask]
            
            if np.isclose(np.std(feature_valid), 0) or np.isclose(np.std(target_valid), 0):
                continue

            pearson_corr, pearson_p = pearsonr(feature_valid, target_valid)
            spearman_corr, spearman_p = spearmanr(feature_valid, target_valid)

            results.append({
                "forward_bars": horizon,
                "feature": feature,
                "pearson_corr": pearson_corr,
                "pearson_p": pearson_p,
                "spearman_corr": spearman_corr,
                "spearman_p": spearman_p,
                "samples": valid_mask.sum(),
            })

    return results


def summarise_results(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()

    details = details.copy()
    details["abs_pearson"] = details["pearson_corr"].abs()
    details["abs_spearman"] = details["spearman_corr"].abs()

    grouped = details.groupby(["symbol", "timeframe", "forward_bars"],
                              as_index=False)
    summary = grouped.agg(
        mean_abs_pearson=("abs_pearson", "mean"),
        median_abs_pearson=("abs_pearson", "median"),
        mean_abs_spearman=("abs_spearman", "mean"),
        samples=("samples", "max"),
        feature_count=("feature", "nunique"),
    )

    idx = details.groupby(["symbol", "timeframe",
                           "forward_bars"])["abs_pearson"].idxmax()
    best_rows = details.loc[idx, [
        "symbol", "timeframe", "forward_bars", "feature", "pearson_corr",
        "pearson_p", "spearman_corr", "spearman_p"
    ]]
    best_rows = best_rows.rename(
        columns={
            "feature": "best_feature",
            "pearson_corr": "best_pearson_corr",
            "pearson_p": "best_pearson_p",
            "spearman_corr": "best_spearman_corr",
            "spearman_p": "best_spearman_p",
        })

    summary = summary.merge(best_rows,
                            on=["symbol", "timeframe", "forward_bars"],
                            how="left")
    summary["max_abs_pearson"] = summary["best_pearson_corr"].abs()
    summary = summary.sort_values(["symbol", "max_abs_pearson"],
                                  ascending=[True, False])
    return summary


def dataframe_to_markdown(df: pd.DataFrame,
                          index: bool = False,
                          floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)
    except ImportError:

        def _format_value(val):
            if isinstance(val, (int, float, np.floating)):
                return f"{val:{floatfmt}}"
            return str(val)

        formatted = df.applymap(_format_value)
        content = formatted.to_string(index=index)
        return f"```\n{content}\n```"


def write_markdown_report(
    output_path: Path,
    config: AnalysisConfig,
    summary: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as report:
        report.write("# Timeframe vs Forward Horizon Correlation Analysis\n\n")
        report.write("## Parameters\n\n")
        report.write(f"- Data directory: `{config.data_dir}`\n")
        report.write(f"- Symbols: {', '.join(config.symbols)}\n")
        report.write(
            f"- Timeframes evaluated: {', '.join(config.timeframes)}\n")
        report.write(
            f"- Forward horizons (bars): {', '.join(map(str, config.forward_bars))}\n"
        )
        report.write(f"- Feature type: {config.feature_type}\n")
        report.write(f"- Max lag features: {config.max_lag}\n")
        report.write(f"- Min samples per test: {config.min_samples}\n")
        if config.extra_features:
            report.write(
                f"- Extra features: {', '.join(config.extra_features)}\n")
        else:
            report.write("- Extra features: none (baseline set)\n")
        if config.start or config.end:
            report.write(
                f"- Date range: {config.start.date() if config.start else 'Start'} → "
                f"{config.end.date() if config.end else 'Latest'}\n")
        report.write("\n")
        report.write(
            f"**Threshold note:** default absolute correlation threshold is {config.pearson_threshold} "
            "for both Pearson and Spearman. Override via `TF_CONFIG_PEARSON=<value>` when invoking "
            "`make timeframe-forward-report` to tighten or relax the selection; the HTML and grouped configs "
            "will refresh automatically.\n\n"
        )

        if summary.empty:
            report.write(
                "No valid correlations were computed (insufficient data or constant series).\n"
            )
            return

        for symbol in summary["symbol"].unique():
            report.write(f"## {symbol}\n\n")
            symbol_summary = summary[summary["symbol"] == symbol].copy()
            top_rows = symbol_summary.head(config.top_k)
            display_cols = [
                "timeframe",
                "forward_bars",
                "best_feature",
                "best_pearson_corr",
                "best_spearman_corr",
                "mean_abs_pearson",
                "samples",
            ]
            top_rows = top_rows[display_cols]
            report.write(
                dataframe_to_markdown(top_rows, index=False, floatfmt=".4f"))
            report.write("\n\n")

            report.write("Top detailed correlations:\n\n")
            symbol_details = details[details["symbol"] == symbol].copy()
            symbol_details = symbol_details.sort_values(
                "pearson_corr", key=lambda s: s.abs(),
                ascending=False).head(10)
            detail_cols = [
                "timeframe",
                "forward_bars",
                "feature",
                "pearson_corr",
                "spearman_corr",
                "samples",
            ]
            report.write(
                dataframe_to_markdown(symbol_details[detail_cols],
                                      index=False,
                                      floatfmt=".4f"))
            report.write("\n\n")

        report.write("## Notes\n\n")
        report.write(
            "- `best_feature` corresponds to the feature with the highest absolute Pearson correlation for the timeframe/horizon.\n"
        )
        report.write(
            "- Consider both mean absolute correlation and sample size when selecting production configurations.\n"
        )
        report.write(
            "- Statistical significance can be inferred from the included p-values in the detailed CSV.\n"
        )


def dataframe_to_markdown(df: pd.DataFrame,
                          index: bool = False,
                          floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)
    except ImportError:
        float_formatter = lambda x: f"{x:{floatfmt}}" if isinstance(
            x, (int, float, np.floating)) else str(x)
        formatted = df.to_string(
            index=index,
            formatters={col: float_formatter
                        for col in df.columns})
        return f"```\n{formatted}\n```"


def _order_timeframes(timeframes: Sequence[str]) -> List[str]:

    def _tf_key(tf: str) -> float:
        normalized = _normalise_timeframe(tf)
        try:
            return pd.to_timedelta(normalized).total_seconds()
        except Exception:
            return float("inf")

    return sorted(set(timeframes), key=_tf_key)


def write_html_report(
    output_path: Path,
    config: AnalysisConfig,
    summary: pd.DataFrame,
    details: pd.DataFrame,
    top_feature_rows: int = 10,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if summary.empty:
        html = "<html><body><h1>No correlation results available</h1></body></html>"
        output_path.write_text(html, encoding="utf-8")
        return

    abs_threshold = config.pearson_threshold

    if config.extra_features:
        extra_list = ", ".join(config.extra_features)
    else:
        extra_list = "None (baseline feature set)"

    timeframe_order = _order_timeframes(config.timeframes)
    forward_order = sorted(set(config.forward_bars))

    html_parts = [
        "<html>",
        "<head>",
        "<meta charset='utf-8' />",
        "<title>Timeframe vs Forward Horizon Correlation Analysis</title>",
        "<style>",
        "body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#fafafa; color:#222; margin:40px; }",
        "h1 { margin-bottom:0.5rem; }",
        ".meta { margin-bottom:1.5rem; }",
        ".symbol-block { margin-bottom:2.5rem; padding:1.5rem; background:#fff; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }",
        ".combo-card { border-left:4px solid #ccc; padding:1rem; margin:1rem 0; background:#f8f9fc; border-radius:8px; }",
        ".combo-card.ic-good { border-left-color:#2ecc71; background:#eefbf3; }",
        ".combo-card .summary { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem; }",
        "table { width:100%; border-collapse:collapse; margin-bottom:1rem; font-size:0.92rem; }",
        "th, td { border:1px solid #e1e5ee; padding:6px 8px; text-align:left; }",
        "th { background:#f1f4fb; font-weight:600; }",
        ".badge { display:inline-block; padding:2px 6px; margin-left:8px; border-radius:4px; background:#dde4f5; font-size:0.8rem; }",
        ".ic-good-label { color:#1f8c45; font-weight:600; }",
        ".feature-pill { display:inline-block; padding:4px 8px; margin:0 6px 6px 0; background:#eef2ff; border-radius:6px; font-size:0.85rem; }",
        ".corr-alert { color:#c0392b; font-weight:600; }",
        "</style>",
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>",
        "</head>",
        "<body>",
        "<h1>Timeframe vs Forward Horizon Correlation</h1>",
        "<div class='meta'>",
        f"<div><strong>Feature type:</strong> {config.feature_type}</div>",
        f"<div><strong>Extra features:</strong> {extra_list}</div>",
        f"<div><strong>Good IC threshold:</strong> |ρ| ≥ {abs_threshold}</div>",
        "</div>",
        "<div class='meta' style='background:#fff;padding:1rem;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:2rem;'>",
        "<strong>How to read:</strong>",
        "<ul style='margin:0.5rem 0 0 1.2rem;'>",
        "<li>热力图显示各 (timeframe, forward) 组合的最大绝对 Pearson，越亮说明线性预测力越强。</li>",
        "<li>下方卡片列出该品种最优的两个组合及 Top10 特征，绿色表示 |Pearson| 达到阈值。</li>",
        "<li>表格中的 p-value 越小，说明相关性越不可能由噪声造成（通常 &lt;1e-3 即较可靠）。</li>",
        "<li>末尾的 Cross-symbol 段落会列出在多个品种中重复出现的核心特征，适合做多资产共振信号。</li>",
        f"<li>当前默认阈值 |Pearson| = |Spearman| = {abs_threshold}。可在 make 参数 <code>TF_CONFIG_PEARSON</code> 中自定义阈值，报告会自动刷新标记。</li>",
        "<li>绿色组合卡片表示该 timeframe-forward 的最佳特征 |Pearson| ≥ 阈值；表格中红色数字说明该特征当前相关性低于阈值，需谨慎使用。</li>",
        "<li><strong>Pearson</strong> 衡量线性相关性；<strong>p-value</strong> 是线性相关显著性的统计检验；<strong>Spearman</strong> 衡量排序/单调相关性，可识别非线性但单调的关系。</li>",
        "</ul>",
        "</div>",
    ]

    shared_feature_map: Dict[str, set] = defaultdict(set)

    for symbol in sorted(summary["symbol"].unique()):
        symbol_summary = summary[summary["symbol"] == symbol].copy()
        symbol_details = details[details["symbol"] == symbol].copy()
        if symbol_summary.empty:
            continue

        symbol_details = symbol_details.copy()
        symbol_details["abs_pearson"] = symbol_details["pearson_corr"].abs()

        top_combos = (
            symbol_summary.sort_values("max_abs_pearson", ascending=False)
            .head(2)
            .copy()
        )
        if top_combos.empty:
            continue

        html_parts.append(f"<div class='symbol-block'><h2>{symbol}</h2>")

        symbol_pivot = (
            symbol_summary.pivot_table(
                index="timeframe",
                columns="forward_bars",
                values="max_abs_pearson",
                aggfunc="max",
            )
            .reindex(index=timeframe_order)
            .reindex(columns=forward_order)
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=symbol_pivot.values,
                x=[str(col) for col in symbol_pivot.columns],
                y=symbol_pivot.index.tolist(),
                colorscale="Viridis",
                colorbar=dict(title="|Pearson|"),
                hovertemplate=(
                    "Timeframe: %{y}<br>"
                    "Forward bars: %{x}<br>"
                    "|Pearson|: %{z:.4f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=f"{symbol} | max |Pearson| per (timeframe, forward)",
            xaxis_title="Forward Bars",
            yaxis_title="Timeframe",
            height=460,
        )
        html_parts.append('<div class="chart">')
        html_parts.append(fig.to_html(include_plotlyjs=False, full_html=False))
        html_parts.append("</div>")

        for _, combo in top_combos.iterrows():
            timeframe = combo["timeframe"]
            forward = int(combo["forward_bars"])
            ic_score = combo["max_abs_pearson"]
            best_feature = combo["best_feature"]
            metrics = (
                f"{combo['mean_abs_pearson']:.4f} mean |ρ| · "
                f"{ic_score:.4f} max |ρ| · {int(combo['samples'])} samples"
            )
            ic_class = "ic-good" if ic_score >= abs_threshold else ""
            ic_label = (
                "<span class='ic-good-label'>✓ meets threshold</span>"
                if ic_score >= abs_threshold
                else "<span style='color:#d35400;font-weight:600;'>⚠ below threshold</span>"
            )

            combo_details = symbol_details[
                (symbol_details["timeframe"] == timeframe)
                & (symbol_details["forward_bars"] == forward)
            ].sort_values("abs_pearson", ascending=False)

            top_features = combo_details.head(top_feature_rows)

            html_parts.append(f"<div class='combo-card {ic_class}'>")
            html_parts.append(
                "<div class='summary'>"
                f"<div><strong>{timeframe}</strong> · forward {forward} bars "
                f"<span class='badge'>{best_feature}</span></div>"
                f"<div>{ic_label}</div>"
                "</div>"
            )
            html_parts.append(f"<div style='margin-bottom:0.6rem;color:#555;'>{metrics}</div>")

            if not top_features.empty:
                html_parts.append(
                    "<table><thead><tr>"
                    "<th>#</th><th>Feature</th><th>Pearson</th><th>p-value</th><th>Spearman</th><th>p-value</th><th>Samples</th>"
                    "</tr></thead><tbody>"
                )
                for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
                    feature_name = row["feature"]
                    pearson = row["pearson_corr"]
                    pearson_p = row["pearson_p"]
                    spearman = row["spearman_corr"]
                    spearman_p = row["spearman_p"]
                    samples = int(row["samples"])
                    html_parts.append(
                        "<tr>"
                        f"<td>{rank}</td>"
                        f"<td>{feature_name}</td>"
                        f"<td class='{ 'corr-alert' if abs(pearson) < abs_threshold else '' }'>{pearson:.4f}</td>"
                        f"<td class='{ 'corr-alert' if abs(pearson) < abs_threshold else '' }'>{pearson_p:.1e}</td>"
                        f"<td class='{ 'corr-alert' if abs(spearman) < abs_threshold else '' }'>{spearman:.4f}</td>"
                        f"<td class='{ 'corr-alert' if abs(spearman) < abs_threshold else '' }'>{spearman_p:.1e}</td>"
                        f"<td>{samples}</td>"
                        "</tr>"
                    )
                    shared_feature_map[feature_name].add(symbol)
                html_parts.append("</tbody></table>")
            else:
                html_parts.append("<p>No sufficient feature correlations for this combination.</p>")

            html_parts.append("</div>")  # combo-card

        html_parts.append("</div>")  # symbol-block

    duplicate_features = [
        (feature, sorted(symbols))
        for feature, symbols in shared_feature_map.items()
        if len(symbols) > 1
    ]
    duplicate_features.sort(key=lambda item: (-len(item[1]), item[0]))

    html_parts.append("<div class='symbol-block'>")
    html_parts.append("<h2>Cross-symbol Feature Overlap</h2>")
    if duplicate_features:
        html_parts.append(
            "<table><thead><tr><th>Feature</th><th>Symbols</th></tr></thead><tbody>"
        )
        for feature, symbols in duplicate_features:
            symbol_list = ", ".join(symbols)
            html_parts.append(
                f"<tr><td>{feature}</td><td>{symbol_list}</td></tr>"
            )
        html_parts.append("</tbody></table>")
    else:
        html_parts.append(
            "<p>No overlapping top features detected across symbols for the selected combinations.</p>"
        )
    html_parts.append("</div>")

    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.output_dir / config.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    all_details: List[pd.DataFrame] = []

    for symbol in config.symbols:
        raw_df = load_symbol_dataframe(config.data_dir, symbol, config.start,
                                       config.end)

        for timeframe in tqdm(config.timeframes, desc=f"{symbol} timeframes"):
            try:
                tf_df = resample_timeframe(raw_df, timeframe)
            except Exception as exc:
                print(
                    f"[WARN] Failed to resample {symbol} at {timeframe}: {exc}"
                )
                continue

            if tf_df.empty:
                continue

            try:
                engineer = ComprehensiveFeatureEngineer(
                    feature_types=config.feature_type)
                tf_reset = tf_df.reset_index()
                timestamps = pd.to_datetime(tf_reset["timestamp"], utc=False)
                if "symbol" not in tf_reset.columns:
                    tf_reset["symbol"] = symbol
                engineered_df = engineer.engineer_features(tf_reset, fit=True)
            except Exception as exc:
                print(
                    f"[WARN] Feature engineering failed for {symbol} at {timeframe}: {exc}"
                )
                continue

            engineered_df = engineered_df.copy()
            engineered_df["timestamp"] = timestamps.values
            engineered_df["timestamp"] = pd.to_datetime(
                engineered_df["timestamp"], utc=False)
            engineered_df = engineered_df.drop_duplicates(
                subset=["timestamp"]).set_index("timestamp").sort_index()

            base_feature_cols = get_feature_columns_by_type(
                engineered_df, config.feature_type)
            base_feature_cols = [
                col for col in base_feature_cols if col not in tf_df.columns
            ]
            if not base_feature_cols:
                print(
                    f"[WARN] No engineered features available for {symbol} at {timeframe} (feature_type={config.feature_type})"
                )
                continue

            aligned_df = tf_df.join(
                engineered_df[base_feature_cols],
                how="inner",
            )
            if aligned_df.empty:
                continue

            enriched_df, manual_features = enrich_features(
                aligned_df, config.max_lag, config.extra_features)

            feature_cols = [
                col for col in base_feature_cols if col in enriched_df.columns
            ]
            feature_cols.extend(manual_features)
            feature_cols = [
                col for col in dict.fromkeys(feature_cols)
                if col in enriched_df.columns
            ]

            if not feature_cols:
                continue

            corr_results = compute_correlations(enriched_df, feature_cols,
                                                config.forward_bars,
                                                config.min_samples)
            if not corr_results:
                continue

            detail_df = pd.DataFrame(corr_results)
            detail_df.insert(0, "timeframe", timeframe)
            detail_df.insert(0, "symbol", symbol)
            all_details.append(detail_df)

    if not all_details:
        print(
            "No correlation results generated. Check input parameters and data availability."
        )
        return

    details_df = pd.concat(all_details, ignore_index=True)
    details_path = run_dir / "timeframe_forward_details.csv"
    details_df.to_csv(details_path, index=False)

    summary_df = summarise_results(details_df)
    summary_path = run_dir / "timeframe_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    markdown_path = (config.markdown_report if config.markdown_report
                     is not None else run_dir / "timeframe_forward_report.md")
    write_markdown_report(markdown_path, config, summary_df, details_df)

    html_path = config.html_report if config.html_report else run_dir / "timeframe_forward_report.html"
    write_html_report(html_path, config, summary_df, details_df)

    print(f"Saved detailed metrics to {details_path}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Wrote markdown report to {markdown_path}")
    print(f"Wrote HTML report to {html_path}")


if __name__ == "__main__":
    main()
