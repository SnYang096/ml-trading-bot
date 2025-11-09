#!/usr/bin/env python3
"""
Train cross-sectional models (boosting or Fama-MacBeth) on multi-asset factor panels.

Example:
    python scripts/cross_sectional/train_cross_sectional_model.py \
        --input "results/training/*/features/*.parquet" \
        --symbols "BTCUSDT,ETHUSDT,SOLUSDT" \
        --horizon 12 \
        --model boosting \
        --output-dir results/cross_sectional/models/demo
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import dump

from ml_trading.cross_sectional import (
    FactorPanelBuilder,
    PanelConfig,
    CrossSectionalBoostingModel,
    CrossSectionalRegressor,
    ReportContext,
    add_crypto_cross_sectional_factors,
    cross_sectional_zscore,
    winsorize_by_sigma,
    generate_markdown_report,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cross-sectional models on multi-asset factor panels."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Parquet/CSV files or glob patterns containing engineered features (must include timestamp & symbol).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_sectional/models",
        help="Directory to save model artefacts and diagnostics.",
    )
    parser.add_argument(
        "--model",
        choices=["boosting", "fama_macbeth"],
        default="boosting",
        help="Cross-sectional model type (default: boosting).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to keep. Default: use all symbols present.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forward return horizon in bars used as the target (future_return_{horizon}).",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Optional comma-separated feature list. If omitted, numeric columns excluding OHLCV/labels are used.",
    )
    parser.add_argument(
        "--winsor",
        type=float,
        default=3.0,
        help="Sigma threshold for cross-sectional winsorisation (<=0 disables).",
    )
    parser.add_argument(
        "--zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply cross-sectional z-scoring per timestamp.",
    )
    parser.add_argument(
        "--crypto-factors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment panel with crypto-specific cross-sectional factors.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualisation factor for report metrics (e.g., 17520 for 5-min bars).",
    )
    parser.add_argument(
        "--save-markdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a Markdown diagnostics report alongside metrics.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cs_boosting.joblib",
        help="File name for saved model (if boosting).",
    )
    parser.add_argument(
        "--predictions-name",
        type=str,
        default="predictions.parquet",
        help="File name for saved predictions (MultiIndex parquet).",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="metrics.json",
        help="File name for saved evaluation metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = collect_inputs(args.input)
    raw_df = load_frames(input_paths)
    filtered_df = filter_symbols(raw_df, args.symbols)

    if filtered_df.empty:
        raise ValueError("No data available after applying symbol filters.")

    filtered_df, target_col = ensure_future_return_column(filtered_df, args.horizon)

    feature_cols = (
        [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        if args.feature_cols
        else None
    )

    panel, base_features = build_panel(
        filtered_df,
        target_col=target_col,
        feature_cols=feature_cols,
        min_assets=3,
        horizon=args.horizon,
    )
    factor_cols = list(feature_cols) if feature_cols else list(base_features)

    if args.crypto_factors:
        panel = add_crypto_cross_sectional_factors(panel)
        crypto_cols = [
            col
            for col in panel.columns
            if col.startswith("cs_crypto_") and col not in factor_cols and col != target_col
        ]
        factor_cols.extend(crypto_cols)

    processed_panel = preprocess_panel(panel, factor_cols, args.winsor, args.zscore)

    if args.model == "boosting":
        model = CrossSectionalBoostingModel()
        model.fit(processed_panel, factor_cols=factor_cols, target_col=target_col)
        predictions = model.predict(processed_panel)
        eval_result = model.evaluate(processed_panel, predictions=predictions, target_col=target_col)

        save_predictions(predictions, output_dir / args.predictions_name)
        save_metrics(
            eval_result,
            output_dir / args.metrics_name,
        )
        dump(model, output_dir / args.model_name)

        if args.save_markdown:
            report = build_report_from_eval(
                processed_panel,
                factor_cols,
                eval_result,
                args,
            )
            write_report(output_dir / "boosting_report.md", report)

        print(f"✅ Boosting model trained. Artefacts saved under {output_dir}")

    elif args.model == "fama_macbeth":
        reg = CrossSectionalRegressor(add_intercept=True, min_assets=3)
        result = reg.fit(processed_panel, factor_cols=factor_cols, target_col=target_col)
        metrics = {
            "factor_summary": result.factor_summary(args.periods_per_year).to_dict(),
            "ic_summary": result.ic_summary(args.periods_per_year).to_dict(),
            "newey_west": result.newey_west_summary(
                max_lag=5, periods_per_year=args.periods_per_year
            ).to_dict(),
        }
        (output_dir / args.metrics_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if args.save_markdown:
            context = FactorPanelBuilder.describe_panel(processed_panel)
            report = generate_markdown_report(
                result,
                ReportContext(
                    title="Cross-Sectional Fama-MacBeth Training Report",
                    max_lag=5,
                    periods_per_year=args.periods_per_year,
                    preprocessing=_describe_preprocessing(args.winsor, args.zscore),
                    symbols=args.symbols or ", ".join(sorted(filtered_df["symbol"].unique())),
                    horizon=args.horizon,
                    observations=int(context.get("num_observations", 0)),
                    timestamps=int(context.get("num_timestamps", 0)),
                    assets_per_timestamp=float(context.get("mean_assets_per_timestamp", 0.0)),
                ),
            )
            write_report(output_dir / "fama_macbeth_report.md", report)

        print(f"✅ Fama-MacBeth regression complete. Metrics saved under {output_dir}")
    else:
        raise ValueError(f"Unsupported model: {args.model}")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def collect_inputs(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded and Path(pattern).exists():
            expanded = [pattern]
        files.extend(expanded)
    unique = sorted({os.path.abspath(p) for p in files})
    if not unique:
        raise FileNotFoundError(f"No input files match: {patterns}")
    return unique


def load_frames(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")
        if df.empty:
            continue
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' column missing in {path}")
        if "symbol" not in df.columns:
            df["symbol"] = _infer_symbol_from_path(path)
        frames.append(df)
    if not frames:
        raise ValueError("All input frames are empty.")
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    combined = combined.sort_values(["timestamp", "symbol"])
    return combined


def filter_symbols(df: pd.DataFrame, symbols: Optional[str]) -> pd.DataFrame:
    if not symbols:
        return df
    symbol_list = [s.strip().upper() for s in symbols.replace(" ", ",").split(",") if s.strip()]
    return df[df["symbol"].str.upper().isin(symbol_list)].copy()


def ensure_future_return_column(
    df: pd.DataFrame,
    horizon: int,
    price_col: str = "close",
) -> tuple[pd.DataFrame, str]:
    col_name = f"future_return_{horizon}"
    if col_name in df.columns:
        return df, col_name
    if price_col not in df.columns:
        raise ValueError(f"{price_col} column missing; cannot compute forward return.")
    df_sorted = df.sort_values(["symbol", "timestamp"]).copy()
    df_sorted[col_name] = df_sorted.groupby("symbol")[price_col].apply(
        lambda x: x.shift(-horizon) / x - 1.0
    )
    return df_sorted, col_name


def build_panel(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]],
    min_assets: int,
    horizon: int,
) -> tuple[pd.DataFrame, List[str]]:
    config = PanelConfig(
        feature_cols=feature_cols,
        target_col=target_col,
        forward_return_horizon=horizon,
        min_assets_per_ts=min_assets,
        fill_method="ffill",
        dropna_after_fill=False,
        align_intersection_only=False,
    )
    builder = FactorPanelBuilder(config)
    panel = builder.from_concat_frame(df)
    if feature_cols:
        return panel, list(feature_cols)

    exclude_cols = {target_col, "open", "high", "low", "close", "volume", "timestamp", "symbol"}
    numeric_cols = [
        col
        for col in panel.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(panel[col])
    ]
    return panel, numeric_cols


def preprocess_panel(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    winsor_sigma: float,
    apply_zscore: bool,
) -> pd.DataFrame:
    processed = panel.copy()
    if winsor_sigma and winsor_sigma > 0:
        processed = winsorize_by_sigma(processed, factor_cols, sigma=winsor_sigma)
    if apply_zscore:
        processed = cross_sectional_zscore(processed, factor_cols)
    return processed


def save_predictions(predictions: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = predictions.to_frame(name="predicted_return")
    df.to_parquet(path)


def save_metrics(eval_result, path: Path) -> None:
    metrics = {
        "ic_mean": float(eval_result.information_coefficients.mean()),
        "ic_std": float(eval_result.information_coefficients.std(ddof=0)),
        "rank_ic_mean": float(eval_result.rank_ic.mean()),
        "rank_ic_std": float(eval_result.rank_ic.std(ddof=0)),
    }
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def build_report_from_eval(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    eval_result,
    args: argparse.Namespace,
) -> str:
    ic_summary = eval_result.information_coefficients.describe().to_dict()
    rank_ic_summary = eval_result.rank_ic.describe().to_dict()
    mse_summary = eval_result.mse_by_timestamp.describe().to_dict()

    diagnostics = FactorPanelBuilder.describe_panel(panel)

    lines = [
        "# Cross-Sectional Boosting Training Report",
        "",
        f"- Symbols: {args.symbols or 'all'}",
        f"- Horizon: {args.horizon} bars",
        f"- Periods/year: {args.periods_per_year}",
        f"- Preprocessing: {_describe_preprocessing(args.winsor, args.zscore)}",
        f"- Features used: {len(factor_cols)}",
        "",
        "## Panel diagnostics",
        "",
        json.dumps(diagnostics, indent=2),
        "",
        "## Information coefficient summary",
        "",
        json.dumps(ic_summary, indent=2),
        "",
        "## Rank IC summary",
        "",
        json.dumps(rank_ic_summary, indent=2),
        "",
        "## MSE by timestamp summary",
        "",
        json.dumps(mse_summary, indent=2),
    ]
    return "\n".join(lines) + "\n"


def _describe_preprocessing(winsor_sigma: float, apply_zscore: bool) -> str:
    steps = []
    if winsor_sigma and winsor_sigma > 0:
        steps.append(f"winsor |σ|<{winsor_sigma}")
    if apply_zscore:
        steps.append("z-score")
    if not steps:
        return "none"
    return " + ".join(steps)


def _infer_symbol_from_path(path: str) -> str:
    stem = Path(path).stem.upper()
    for sep in ("_", "-"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


if __name__ == "__main__":
    main()

