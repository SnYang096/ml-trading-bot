"""Generate feature IC/IR diagnostics and export HTML summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from ml_trading.models.train_model import prepare_ohlcv_dataframe
from ml_trading.data_tools.feature_engineering import FeatureEngineer
from ml_trading.data_tools.feature_engineering_enhanced import (
    EnhancedFeatureEngineer, )
from ml_trading.data_tools.dl_sequence_features import add_dl_sequence_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Generate feature IC/IR diagnostics and export HTML summary.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to OHLCV parquet/CSV file (single asset time-series).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output HTML report.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (inclusive, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (inclusive, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--future-horizon",
        type=int,
        default=1,
        help="Forward return horizon in bars (default: 1).",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=20,
        help="Minimum samples per day to compute a daily IC (default: 20).",
    )
    parser.add_argument(
        "--include-enhanced",
        action="store_true",
        help="Include enhanced (wavelet/hilbert/order-flow) features.",
    )
    parser.add_argument(
        "--no-enhanced",
        dest="include_enhanced",
        action="store_false",
        help="Skip enhanced features (basic + TA-Lib only).",
    )
    parser.add_argument(
        "--include-dl",
        action="store_true",
        help="Include deep-learning sequence features.",
    )
    parser.add_argument(
        "--no-dl",
        dest="include_dl",
        action="store_false",
        help="Skip deep-learning sequence features.",
    )
    parser.set_defaults(include_enhanced=True, include_dl=True)
    return parser.parse_args()


def load_price_data(path: Path, start: str | None,
                    end: str | None) -> pd.DataFrame:
    df = prepare_ohlcv_dataframe(
        pd.read_parquet(path) if path.suffix.lower() ==
        ".parquet" else pd.read_csv(path))
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df


def engineer_features(
    price_df: pd.DataFrame,
    include_enhanced: bool,
    include_dl: bool,
) -> pd.DataFrame:
    basic_engineer = FeatureEngineer()
    features = basic_engineer.add_technical_indicators(price_df)

    if include_enhanced:
        enhanced = EnhancedFeatureEngineer(scaler_type="standard")
        features = enhanced.add_hurst_features(features)
        features = enhanced.add_wavelet_packet_features(features)
        features = enhanced.add_hilbert_features(features)
        features = enhanced.add_spectral_features(features)
        features = enhanced.add_advanced_derived_features(features)
        features = enhanced.add_order_flow_features(features)

    if include_dl:
        try:
            features = add_dl_sequence_features(features)
        except Exception as err:
            print(f"⚠️  Deep-learning feature extraction failed: {err}")

    return features


def compute_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-horizon)
    returns = future / close - 1.0
    return returns


def compute_ic_stats(
    feature_series: pd.Series,
    future_returns: pd.Series,
    min_group_size: int,
) -> Tuple[float, float, float, int, float]:
    joined = pd.concat([feature_series, future_returns], axis=1,
                       join="inner").dropna()
    if joined.empty or joined.iloc[:, 0].nunique() < 5:
        return (np.nan, np.nan, np.nan, 0, np.nan)

    # Daily rank ICs
    grouped = joined.groupby(joined.index.normalize())
    ic_values: list[float] = []
    for _, group in grouped:
        if len(group) < min_group_size:
            continue
        corr = group.iloc[:, 0].corr(group.iloc[:, 1], method="spearman")
        if pd.notna(corr):
            ic_values.append(corr)

    if not ic_values:
        return (np.nan, np.nan, np.nan, 0, np.nan)

    ic_array = np.asarray(ic_values)
    ic_mean = float(ic_array.mean())
    ic_std = float(ic_array.std(ddof=1)) if ic_array.size > 1 else np.nan
    ir = ic_mean / ic_std if ic_std and not np.isclose(ic_std, 0.0) else np.nan
    global_ic = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")

    return ic_mean, ic_std, ir, len(ic_values), float(global_ic)


def build_report(
    feature_df: pd.DataFrame,
    future_returns: pd.Series,
    min_group_size: int,
) -> pd.DataFrame:
    exclude = {"open", "high", "low", "close", "volume", "timestamp"}
    feature_columns = [col for col in feature_df.columns if col not in exclude]

    rows: list[Dict[str, object]] = []
    for col in feature_columns:
        ic_mean, ic_std, ir, ic_observations, global_ic = compute_ic_stats(
            feature_df[col], future_returns, min_group_size)
        if np.isnan(ic_mean):
            continue
        rows.append({
            "feature": col,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "information_ratio": ir,
            "ic_observations": ic_observations,
            "global_ic": global_ic,
            "non_na": int(feature_df[col].count()),
        })

    report_df = pd.DataFrame(rows)
    if report_df.empty:
        return report_df

    report_df.sort_values(
        by="information_ratio",
        ascending=False,
        inplace=True,
    )
    report_df.reset_index(drop=True, inplace=True)
    return report_df


def render_html(
    report_df: pd.DataFrame,
    input_path: Path,
    start: str | None,
    end: str | None,
    horizon: int,
    include_enhanced: bool,
    include_dl: bool,
) -> str:
    date_range = f"{start or 'min'} → {end or 'max'}"
    summary_rows = [
        ("Data file", input_path),
        ("Date range", date_range),
        ("Forward horizon", f"{horizon} bars"),
        ("Enhanced features", "Yes" if include_enhanced else "No"),
        ("Deep-learning features", "Yes" if include_dl else "No"),
        ("Evaluated features", len(report_df)),
    ]

    summary_table = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    html = [
        "<html><head><meta charset='utf-8'><title>Feature IC Report</title>",
        "<style>body{font-family:Arial, sans-serif;margin:20px;}table{border-collapse:collapse;width:100%;margin-bottom:20px;}th,td{border:1px solid #ddd;padding:8px;}th{background:#f2f2f2;text-align:left;}tr:nth-child(even){background:#fafafa;}</style>",
        "</head><body>",
        "<h1>Feature IC/IR Report</h1>",
        summary_table.to_html(index=False, escape=False),
    ]

    if not report_df.empty:
        display_df = report_df.copy()
        for col in ["ic_mean", "ic_std", "information_ratio", "global_ic"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}"
                                                  if pd.notna(x) else "")
        html.append("<h2>Feature Statistics</h2>")
        html.append(display_df.to_html(index=False, escape=False))
    else:
        html.append(
            "<p><strong>No features produced valid IC statistics.</strong></p>"
        )

    html.append("</body></html>")
    return "\n".join(html)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    price_data = load_price_data(input_path, args.start_date, args.end_date)
    if price_data.empty:
        raise ValueError(
            "No price data available after applying date filters.")

    feature_df = engineer_features(price_data,
                                   include_enhanced=args.include_enhanced,
                                   include_dl=args.include_dl)

    future_returns = compute_forward_returns(price_data["close"],
                                             args.future_horizon)

    report_df = build_report(feature_df, future_returns, args.min_group_size)

    html = render_html(report_df, input_path, args.start_date, args.end_date,
                       args.future_horizon, args.include_enhanced,
                       args.include_dl)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Feature IC report saved to {output_path}")


if __name__ == "__main__":
    main()
