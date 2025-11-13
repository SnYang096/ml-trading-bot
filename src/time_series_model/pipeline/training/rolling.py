from __future__ import annotations
"""
Rolling regression training (returns + uncertainty + volatility) with unified feature selection.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from data_tools.rolling_data import load_and_process_file
from data_tools.baseline_feature_engineering import (
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)
from time_series_model.models.lightgbm_model import LightGBMModel
from time_series_model.models.quant_trading_model import QuantTradingModel
from time_series_model.pipeline.training.classification_model_trainer import ClassificationModelTrainer
from time_series_model.pipeline.training.preprocessing import RobustWinsorizer
from .label_utils import (
    log_return_magnitude,
    invert_log_return_magnitude,
    rolling_quantile_classification_labels,
    rolling_rms_volatility,
)
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from time_series_model.pipeline.training.train import _compute_direction_threshold
from time_series_model.utils.training import (
    print_backtest_results,
    evaluate_signal_performance,
)


def load_top_factors(path: str) -> Optional[List[str]]:
    """Load top factors from various formats.
    
    Supports:
    1. JSON file: list or dict with 'features'/'selected_features' key
    2. TXT file: one feature per line
    3. Directory: auto-finds best_combination/selected_features.txt or best_combination_summary.json
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of feature names, or None if loading failed
    """
    if not path:
        return None

    path_obj = Path(path)

    # If it's a directory, try to find dim-compare output files
    if path_obj.is_dir():
        # Try best_combination/selected_features.txt
        txt_path = path_obj / "best_combination" / "selected_features.txt"
        if txt_path.exists():
            path_obj = txt_path
        else:
            # Try best_combination/best_combination_summary.json
            json_path = path_obj / "best_combination" / "best_combination_summary.json"
            if json_path.exists():
                path_obj = json_path
            else:
                # Try selected_features.txt in root
                txt_path = path_obj / "selected_features.txt"
                if txt_path.exists():
                    path_obj = txt_path
                else:
                    # Try best_combination_summary.json in root
                    json_path = path_obj / "best_combination_summary.json"
                    if json_path.exists():
                        path_obj = json_path
                    else:
                        print(
                            f"   ⚠️  Could not find selected_features.txt or best_combination_summary.json in {path}"
                        )
                        return None

    # Load based on file extension
    try:
        if path_obj.suffix.lower() == '.txt':
            # TXT file: one feature per line
            with open(path_obj, 'r', encoding='utf-8') as f:
                features = [line.strip() for line in f if line.strip()]
            print(f"   ✅ Loaded {len(features)} features from {path_obj}")
            return features
        elif path_obj.suffix.lower() == '.json':
            # JSON file
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Try different keys
            if isinstance(data, list):
                features = data
            elif isinstance(data, dict):
                # Try 'selected_features' first (dim-compare format)
                if 'selected_features' in data:
                    features = data['selected_features']
                # Then try 'features'
                elif 'features' in data:
                    features = data['features']
                else:
                    print(
                        f"   ⚠️  JSON file does not contain 'features' or 'selected_features' key"
                    )
                    return None
            else:
                print(f"   ⚠️  JSON file format not recognized")
                return None

            if isinstance(features, list):
                print(f"   ✅ Loaded {len(features)} features from {path_obj}")
                return features
            else:
                print(f"   ⚠️  Features data is not a list")
                return None
        else:
            # Try as JSON first, then TXT
            try:
                with open(path_obj, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    features = data
                elif isinstance(data, dict):
                    if 'selected_features' in data:
                        features = data['selected_features']
                    elif 'features' in data:
                        features = data['features']
                    else:
                        raise ValueError("No features key found")
                else:
                    raise ValueError("Invalid format")
                print(
                    f"   ✅ Loaded {len(features)} features from {path_obj} (as JSON)"
                )
                return features
            except (json.JSONDecodeError, ValueError):
                # Try as TXT
                with open(path_obj, 'r', encoding='utf-8') as f:
                    features = [line.strip() for line in f if line.strip()]
                print(
                    f"   ✅ Loaded {len(features)} features from {path_obj} (as TXT)"
                )
                return features
    except Exception as exc:
        print(f"   ⚠️  Failed to load top factors from {path_obj}: {exc}")
        return None


def find_all_available_files(data_dir: str, symbols: str) -> List[Dict]:
    """Find all available files for one or multiple symbols.
    
    Args:
        symbols: Single symbol or comma-separated symbols (e.g., "BTCUSDT" or "BTCUSDT,ETHUSDT,SOLUSDT")
    """
    files: List[Dict] = []
    from pathlib import Path
    import re

    data_path = Path(data_dir)
    if not data_path.exists():
        return files

    # Support multiple symbols (comma-separated)
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

    all_files = []
    for symbol in symbol_list:
        normalized = symbol.upper().replace("-", "").replace("/", "")
        if not normalized.endswith("USDT"):
            normalized = f"{normalized}USDT"
        legacy_symbol = normalized.replace("USDT", "-USD")
        patterns = [
            f"{normalized}-aggTrades-*.parquet",
            f"{normalized}-aggTrades-*.zip",
            f"{normalized}_*.parquet",
            f"{normalized}_*.zip",
            f"{legacy_symbol}-aggTrades-*.parquet",
            f"{legacy_symbol}-aggTrades-*.zip",
            f"{legacy_symbol}_*.parquet",
            f"{legacy_symbol}_*.zip",
        ]

        date_patterns = [
            rf"{normalized}-aggTrades-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            rf"{normalized}_(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            rf"{legacy_symbol}-aggTrades-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            rf"{legacy_symbol}_(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            rf"(?P<year>\d{{4}})-(?P<month>\d{{2}})",
        ]

        for pattern in patterns:
            for file_path in data_path.glob(pattern):
                stem = file_path.stem
                match = None
                for dp in date_patterns:
                    match = re.search(dp, stem)
                    if match:
                        break
                if match:
                    try:
                        year = int(match.group("year"))
                        month = int(match.group("month"))
                        file_info = {
                            "path": str(file_path),
                            "symbol": normalized,  # Track normalized symbol
                            "year": year,
                            "month": month,
                            "month_str": f"{year}-{month:02d}",
                            "timestamp": pd.Timestamp(year, month, 1),
                        }
                        # Avoid duplicates
                        if not any(f["path"] == file_info["path"]
                                   for f in all_files):
                            all_files.append(file_info)
                    except Exception:
                        continue

    all_files.sort(key=lambda x: x["timestamp"])
    return all_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling regression training")
    parser.add_argument("--data-dir",
                        type=str,
                        default=os.environ.get("DATA_DIR",
                                               "data/parquet_data"))
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help=
        "Symbol(s) for rolling training. Can be comma-separated (e.g., BTCUSDT,ETHUSDT,SOLUSDT) for multi-asset training"
    )
    parser.add_argument("--initial-train-months", type=int, default=6)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--forward-bars", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--freq",
                        type=str,
                        action="append",
                        default=["5T"],
                        help="--freq 5T or comma-separated")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="TimeSeries CV folds on each training window (0=disable)")
    parser.add_argument("--cv-on-rolling",
                        action="store_true",
                        default=False,
                        help="Enable CV evaluation per rolling window")
    parser.add_argument("--start",
                        type=str,
                        default=None,
                        help="Start YYYY-MM (inclusive)")
    parser.add_argument("--end",
                        type=str,
                        default=None,
                        help="End YYYY-MM (inclusive)")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="comprehensive",
        help="baseline/default/enhanced/dl_sequence/comprehensive")
    parser.add_argument(
        "--direction-threshold",
        type=str,
        default="f1_optimize",
        help=
        "Threshold method for directional prediction (zero|median|f1_optimize)",
    )
    parser.add_argument(
        "--use-top-factors",
        type=str,
        default=None,
        help="Path to selected features file. Supports: "
        "1) JSON file (list or dict with 'features'/'selected_features' key), "
        "2) TXT file (one feature per line), "
        "3) dim-compare output directory (auto-finds best_combination/selected_features.txt or best_combination_summary.json)"
    )
    parser.add_argument("--topk",
                        type=int,
                        default=0,
                        help="Optional: keep only top-K features (0=disabled)")
    parser.add_argument(
        "--topk-source",
        type=str,
        default=None,
        help=
        "Ranking CSV(feature,score) or JSON list; fallback to Spearman |IC|")
    parser.add_argument(
        "--use-autoencoder",
        type=str,
        default=None,
        help=
        "Path to a trained autoencoder .pth (UnifiedAutoencoder). If provided, engineered features will be transformed to compressed embeddings before training."
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=None,
        help=
        "Encoding dimension of the provided autoencoder (required with --use-autoencoder)"
    )
    parser.add_argument("--gpu", action="store_true", default=True)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔄 Rolling Training")
    print("=" * 80)

    def _parse_list(values):
        out = []
        for v in (values if isinstance(values, list) else [values]):
            if isinstance(v, str) and "," in v:
                out.extend([x.strip() for x in v.split(",") if x.strip()])
            else:
                out.append(v)
        return out

    freqs = _parse_list(args.freq)

    # Parse symbols for multi-asset training
    symbol_list = [s.strip() for s in args.symbol.split(",") if s.strip()]
    symbols_str = ",".join(symbol_list) if len(
        symbol_list) > 1 else symbol_list[0] if symbol_list else "UNKNOWN"
    print(f"📊 Rolling training with symbol(s): {symbols_str}")
    if len(symbol_list) > 1:
        print(f"   Multi-asset training: {len(symbol_list)} assets")

    # Validate autoencoder arguments
    if args.use_autoencoder and not args.encoding_dim:
        print(
            "❌ --encoding-dim is required when --use-autoencoder is provided")
        return

    if args.use_autoencoder:
        print(
            f"   Autoencoder: {args.use_autoencoder} (dim={args.encoding_dim})"
        )

    files = find_all_available_files(args.data_dir, args.symbol)

    if args.start or args.end:

        def _in_range(ym: str) -> bool:
            if args.start and ym < args.start:
                return False
            if args.end and ym > args.end:
                return False
            return True

        files = [f for f in files if _in_range(f["month_str"])]

    if not files or len(files) < args.min_train_months + 1:
        print("❌ Not enough monthly files to run rolling training.")
        return

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Handle multiple symbols in output name
        symbol_name = symbols_str.replace(
            ",",
            "_").lower() if len(symbol_list) > 1 else symbol_list[0].lower(
            ) if symbol_list else "unknown"
        args.output = f"rolling_{symbol_name}_{ts}"
    results_dir = os.path.join("results", args.output)
    os.makedirs(results_dir, exist_ok=True)

    for freq in freqs:
        start_idx = args.initial_train_months
        all_results = []
        baseline_engineer = None
        comp_engineer = None
        combo_dir = results_dir
        os.makedirs(combo_dir, exist_ok=True)

        importance_accumulators = {
            "classification": {},
            "return": {},
            "volatility": {},
        }
        classification_trainer = ClassificationModelTrainer(use_gpu=args.gpu)
        artifacts_entries: List[Dict[str, str]] = []

        def _format_pct(value: Optional[float],
                        good_threshold: float = 0.5,
                        excellent_threshold: float = 0.55) -> str:
            if value is None:
                return "<td>N/A</td>"
            try:
                val = float(value)
                color = ("green" if val >= excellent_threshold else
                         "#90EE90" if val >= good_threshold else "red")
                return (f'<td style="color: {color}; font-weight: bold;">'
                        f"{val * 100:.2f}%</td>")
            except (TypeError, ValueError):
                return f"<td>{value}</td>"

        def _format_corr(value: Optional[float],
                         good_threshold: float = 0.05,
                         excellent_threshold: float = 0.1) -> str:
            if value is None:
                return "<td>N/A</td>"
            try:
                val = float(value)
                abs_val = abs(val)
                color = ("green" if abs_val >= excellent_threshold else
                         "#90EE90" if abs_val >= good_threshold else "red")
                return (f'<td style="color: {color}; font-weight: bold;">'
                        f"{val:.4f}</td>")
            except (TypeError, ValueError):
                return f"<td>{value}</td>"

        def _format_r2(value: Optional[float],
                       good_threshold: float = 0.0,
                       excellent_threshold: float = 0.05) -> str:
            if value is None:
                return "<td>N/A</td>"
            try:
                val = float(value)
                color = ("green" if val >= excellent_threshold else
                         "#90EE90" if val >= good_threshold else "red")
                return (f'<td style="color: {color}; font-weight: bold;">'
                        f"{val:.4f}</td>")
            except (TypeError, ValueError):
                return f"<td>{value}</td>"

        def _format_float(value: Optional[float], digits: int = 6) -> str:
            if value is None:
                return "<td>N/A</td>"
            try:
                return f"<td>{float(value):.{digits}f}</td>"
            except (TypeError, ValueError):
                return f"<td>{value}</td>"

        def _build_feature_table(df: Optional[pd.DataFrame],
                                 title: str) -> str:
            if df is None or df.empty:
                return ""
            rows = []
            for rank, row in enumerate(df.head(20).itertuples(), start=1):
                feat = getattr(row, "feature", "N/A")
                imp = getattr(row, "importance", None)
                try:
                    imp_str = f"{float(imp):.6f}"
                except (TypeError, ValueError):
                    imp_str = str(imp)
                rows.append(
                    f"<tr><td>{rank}</td><td>{feat}</td><td>{imp_str}</td></tr>"
                )
            return (f"<h3>{title}</h3>"
                    "<table><tr><th>#</th><th>Feature</th>"
                    "<th>Importance (Gain)</th></tr>"
                    f"{''.join(rows)}</table>")

        def _write_window_report(res_row: Dict,
                                 cls_imp: Optional[pd.DataFrame],
                                 ret_imp: Optional[pd.DataFrame],
                                 vol_imp: Optional[pd.DataFrame],
                                 train_months: List[Dict], test_month: Dict,
                                 output_dir: str) -> None:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir,
                                       f"{test_month['month_str']}.html")

            train_start = train_months[0]["month_str"]
            train_end = train_months[-1]["month_str"]
            info_rows = [
                ("Symbol(s)", symbols_str),
                ("Timeframe", freq),
                ("Forward Bars", fb),
                ("Feature Type", args.feature_type),
                ("Train Months", len(train_months)),
                ("Train Range", f"{train_start} → {train_end}"),
                ("Test Month", test_month["month_str"]),
                ("Train Samples", res_row.get("train_samples")),
                ("Test Samples", res_row.get("test_samples")),
            ]
            info_table = "".join([
                f"<tr><th>{label}</th><td>{value}</td></tr>"
                for label, value in info_rows
            ])

            cls_rows = [
                f"<tr><td>Accuracy</td>{_format_pct(res_row.get('cls_accuracy'))}</tr>",
                f"<tr><td>Precision</td>{_format_pct(res_row.get('cls_precision'))}</tr>",
                f"<tr><td>Recall</td>{_format_pct(res_row.get('cls_recall'))}</tr>",
                f"<tr><td>F1</td>{_format_pct(res_row.get('cls_f1'))}</tr>",
            ]
            if res_row.get("cls_auc") is not None:
                cls_rows.append(
                    f"<tr><td>AUC</td>{_format_pct(res_row.get('cls_auc'))}</tr>"
                )
            if res_row.get("cls_pr_auc") is not None:
                cls_rows.append(
                    f"<tr><td>PR-AUC</td>{_format_pct(res_row.get('cls_pr_auc'))}</tr>"
                )
            if res_row.get("cls_threshold") is not None:
                cls_rows.append(
                    "<tr><td>Best Threshold (F1)</td>"
                    f"{_format_float(res_row.get('cls_threshold'), digits=3)}</tr>"
                )
            cls_rows.append(
                f"<tr><td>IC (Spearman)</td>{_format_corr(res_row.get('cls_ic_spearman'))}</tr>"
            )
            cls_rows.append(
                f"<tr><td>IC (Pearson)</td>{_format_corr(res_row.get('cls_ic_pearson'))}</tr>"
            )

            return_rows = [
                f"<tr><td>RMSE</td>{_format_float(res_row.get('test_rmse_return'))}</tr>",
                f"<tr><td>MAE</td>{_format_float(res_row.get('test_mae_return'))}</tr>",
                f"<tr><td>R²</td>{_format_r2(res_row.get('test_r2_return'))}</tr>",
            ]
            vol_rows = [
                f"<tr><td>RMSE</td>{_format_float(res_row.get('test_rmse_vol'))}</tr>",
                f"<tr><td>MAE</td>{_format_float(res_row.get('test_mae_vol'))}</tr>",
                f"<tr><td>R²</td>{_format_r2(res_row.get('test_r2_vol'))}</tr>",
            ]

            feature_sections = "".join([
                _build_feature_table(
                    cls_imp, "🔍 Directional Feature Importance (Top 20)"),
                _build_feature_table(
                    ret_imp,
                    "📈 Return Regression Feature Importance (Top 20)"),
                _build_feature_table(
                    vol_imp, "📉 Volatility Feature Importance (Top 20)"),
            ])

            html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Rolling Window Report - {test_month['month_str']}</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#222;background:#f5f5f5}}
.container{{max-width:1100px;margin:0 auto;background:white;padding:24px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
h2{{color:#34495e;margin-top:28px;margin-bottom:14px;padding-left:10px;border-left:4px solid #3498db}}
table{{border-collapse:collapse;width:100%;margin:15px 0;background:white}}
th{{background:#3498db;color:#fff;padding:10px;text-align:left;font-weight:600}}
td{{border:1px solid #ddd;padding:10px}}
tr:nth-child(even){{background:#f9f9f9}}
tr:hover{{background:#f0f8ff}}
.info-section{{background:#ecf0f1;padding:15px;border-radius:6px;margin:15px 0}}
.note{{font-size:0.9em;color:#7f8c8d;margin:6px 0}}
</style>
</head><body>
<div class="container">
<h1>📊 Rolling Window Report ({test_month['month_str']})</h1>
<div class="info-section">
<table>{info_table}</table>
<p class="note">红色标注表示指标低于默认阈值（准确率/精确率/召回率/F1/AUC &lt; 50%，PR-AUC &lt; 50%，IC &lt; 0.05，R² &lt; 0）。</p>
</div>
<h2>🎯 Directional Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{''.join(cls_rows)}</table>
<h2>📈 Return Regression Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{''.join(return_rows)}</table>
<h2>📉 Volatility Regression Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{''.join(vol_rows)}</table>
{feature_sections if feature_sections else ''}
</div></body></html>"""

            try:
                with open(report_path, "w", encoding="utf-8") as handle:
                    handle.write(html)
                print(f"   - window report: {report_path}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"   ⚠️  Failed to write window report for {test_month['month_str']}: {exc}"
                )

        def _accumulate_importance(df: Optional[pd.DataFrame],
                                   bucket: str) -> None:
            if df is None or df.empty:
                return
            store = importance_accumulators.setdefault(bucket, {})
            for _, row in df.iterrows():
                feat = row["feature"]
                val = float(row["importance"])
                store[feat] = store.get(feat, 0.0) + val

        def _extract_importance(
                model: LightGBMModel,
                feature_names: List[str]) -> Optional[pd.DataFrame]:
            if not model or not hasattr(model, "model") or model.model is None:
                return None
            try:
                booster = model.model
                gains = booster.feature_importance(importance_type='gain')
                names = booster.feature_name()
                if gains is None or names is None:
                    return None
                if len(gains) != len(names):
                    # fallback to provided feature names
                    names = feature_names
                    if len(gains) != len(feature_names):
                        return None
                df_imp = pd.DataFrame({
                    "feature": names,
                    "importance": gains,
                })
                df_imp = df_imp.groupby("feature",
                                        as_index=False)["importance"].sum()
                df_imp = df_imp.sort_values("importance",
                                            ascending=False).head(100)
                return df_imp
            except Exception:
                return None

        for i in range(start_idx, len(files)):
            train_files = files[:i]
            test_file = files[i]

            print("\n" + "-" * 80)
            print(
                f"Train: {train_files[0]['month_str']} → {train_files[-1]['month_str']} (assets: {len(symbol_list)})"
            )
            print(f"Test:  {test_file['month_str']}")

            # Load train (for multi-asset training, all assets' data are merged)
            train_parts = []
            for fi in train_files:
                df = load_and_process_file(fi["path"], freq=freq)
                if df is not None and len(df) > 0:
                    train_parts.append(df)
            if not train_parts:
                print("   ⚠️  No training data, skip")
                continue
            # Merge all training data (multi-asset training: all assets combined)
            # All features are normalized (asset-agnostic), so the model learns
            # common patterns across different assets
            train_df = pd.concat(train_parts, axis=0).sort_index()
            if len(symbol_list) > 1:
                print(
                    f"   Multi-asset training: {len(train_parts)} files merged, {len(train_df)} samples"
                )

            # Load test
            test_df = load_and_process_file(test_file["path"], freq=freq)
            if test_df is None or len(test_df) == 0:
                print("   ⚠️  No test data, skip")
                continue

            # Features
            print("   🧪 Engineering features (fit on train, apply to test)...")
            if args.feature_type == "baseline":
                train_df, baseline_engineer = engineer_baseline_features(
                    train_df, baseline_engineer, fit=True)
                test_df, _ = engineer_baseline_features(test_df,
                                                        baseline_engineer,
                                                        fit=False)
            else:
                comp_engineer = comp_engineer or ComprehensiveFeatureEngineer(
                    feature_types=args.feature_type)
                train_df = comp_engineer.engineer_all_features(train_df,
                                                               fit=True)
                test_df = comp_engineer.engineer_all_features(test_df,
                                                              fit=False)

            # Targets
            fb = args.forward_bars

            def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out["future_return"] = out["close"].shift(
                    -fb) / out["close"] - 1
                # Volatility proxy: trailing rolling RMS on future returns (no lookahead beyond target horizon)
                vol_window = max(5, fb)
                out["future_volatility"] = rolling_rms_volatility(
                    out["future_return"],
                    window=vol_window,
                    min_periods=min(3, vol_window),
                )
                return out.dropna()

            train_labeled = _add_targets(train_df)
            test_labeled = _add_targets(test_df)

            # Feature columns
            if args.feature_type == "baseline":
                feat_cols = get_baseline_feature_columns(train_labeled)
            else:
                feat_cols = get_feature_columns_by_type(
                    train_labeled, args.feature_type)
            feat_cols = [
                c for c in feat_cols
                if pd.api.types.is_numeric_dtype(train_labeled[c])
            ]

            # Optional top-factors
            if args.use_top_factors:
                keep = load_top_factors(args.use_top_factors)
                if keep:
                    s = set(keep)
                    original_count = len(feat_cols)
                    feat_cols = [c for c in feat_cols if c in s]
                    print(
                        f"   📋 Filtered features: {original_count} → {len(feat_cols)} (using top factors from {args.use_top_factors})"
                    )
                    if len(feat_cols) == 0:
                        print(
                            f"   ⚠️  Warning: No features matched after filtering! Check feature names in top factors file."
                        )
                else:
                    print(
                        f"   ⚠️  Failed to load top factors from {args.use_top_factors}, using all features"
                    )

            # Optional Top-K
            if args.topk and args.topk > 0 and len(feat_cols) > args.topk:
                ranked = None
                if args.topk_source:
                    try:
                        if args.topk_source.lower().endswith('.csv'):
                            _df = pd.read_csv(args.topk_source)
                            if {'feature', 'score'}.issubset(set(_df.columns)):
                                _df = _df.sort_values('score', ascending=False)
                                ranked = [
                                    f for f in _df['feature'].tolist()
                                    if f in feat_cols
                                ]
                        else:
                            with open(args.topk_source, 'r',
                                      encoding='utf-8') as _f:
                                lst = json.load(_f)
                            if isinstance(lst, dict) and 'features' in lst:
                                lst = lst['features']
                            if isinstance(lst, list):
                                ranked = [f for f in lst if f in feat_cols]
                    except Exception:
                        ranked = None
                if ranked is None:
                    try:
                        from scipy.stats import spearmanr
                        ic = []
                        for c in feat_cols:
                            try:
                                r, _ = spearmanr(
                                    train_labeled[c].values,
                                    train_labeled['future_return'].values,
                                    nan_policy='omit')
                                ic.append((c, abs(r) if pd.notna(r) else 0.0))
                            except Exception:
                                ic.append((c, 0.0))
                        ic.sort(key=lambda x: x[1], reverse=True)
                        ranked = [c for c, _ in ic]
                    except Exception:
                        ranked = feat_cols
                feat_cols = ranked[:args.topk]

            # Optional Autoencoder compression
            if args.use_autoencoder:
                try:
                    import torch
                    from time_series_model.models.autoencoder import UnifiedAutoencoder

                    print(
                        f"   🔄 Applying autoencoder compression ({len(feat_cols)} → {args.encoding_dim})..."
                    )

                    # Prepare features for autoencoder
                    X_train_raw = train_labeled[feat_cols].values
                    X_test_raw = test_labeled[feat_cols].values

                    # Standardize features (required for autoencoder)
                    scaler_ae = StandardScaler()
                    X_train_scaled = scaler_ae.fit_transform(X_train_raw)
                    X_test_scaled = scaler_ae.transform(X_test_raw)

                    # Load autoencoder
                    input_dim = len(feat_cols)
                    encoding_dim = int(args.encoding_dim)
                    autoencoder = UnifiedAutoencoder(
                        input_dim,
                        encoding_dim,
                        architecture="production",
                    )
                    state = torch.load(args.use_autoencoder,
                                       map_location="cpu")
                    autoencoder.load_state_dict(state)
                    autoencoder.eval()

                    # Transform features
                    with torch.no_grad():
                        X_train_tensor = torch.as_tensor(X_train_scaled,
                                                         dtype=torch.float32)
                        X_test_tensor = torch.as_tensor(X_test_scaled,
                                                        dtype=torch.float32)
                        _, Z_train = autoencoder(X_train_tensor)
                        _, Z_test = autoencoder(X_test_tensor)
                        X_train = Z_train.numpy()
                        X_test = Z_test.numpy()

                    # Update feature columns for compressed features
                    feat_cols = [
                        f"compressed_feature_{i}" for i in range(encoding_dim)
                    ]

                    print(
                        f"   ✓ Applied autoencoder compression: {len(feat_cols)} compressed features"
                    )
                except Exception as exc:
                    print(
                        f"   ⚠️ Failed to apply autoencoder compression: {exc}"
                    )
                    print("   Falling back to original features")
                    # Fall back to original features
                    X_train = train_labeled[feat_cols].values
                    X_test = test_labeled[feat_cols].values
            else:
                # Use original features
                X_train = train_labeled[feat_cols].values
                X_test = test_labeled[feat_cols].values

            X_train_df = pd.DataFrame(X_train,
                                      index=train_labeled.index,
                                      columns=feat_cols)
            X_test_df = pd.DataFrame(X_test,
                                     index=test_labeled.index,
                                     columns=feat_cols)

            y_return_train = train_labeled["future_return"]
            y_vol_train = train_labeled["future_volatility"]
            y_return_test = test_labeled["future_return"]
            y_vol_test = test_labeled["future_volatility"]

            groups = None
            groups_series = None
            if "symbol" in train_labeled.columns:
                try:
                    groups_series = train_labeled["symbol"].astype(
                        "category").cat.codes
                    groups = groups_series.to_numpy()
                except Exception:
                    groups = None
                    groups_series = None

            quantile_kwargs = dict(
                window=classification_trainer.quantile_window,
                lower_quantile=classification_trainer.quantile_lower,
                upper_quantile=classification_trainer.quantile_upper,
                min_periods=classification_trainer.quantile_min_periods,
            )
            train_cls_series, _, _, _ = rolling_quantile_classification_labels(
                y_return_train, **quantile_kwargs)
            if train_cls_series.nunique() < 2:
                train_cls_series = pd.Series((y_return_train
                                              > 0).astype(int).values,
                                             index=y_return_train.index)

            train_valid_indices = train_cls_series.index
            X_train_cls = X_train_df.loc[train_valid_indices]
            y_cls_train_filtered = train_cls_series.loc[
                train_valid_indices].astype(int)
            groups_filtered = (
                groups_series.loc[train_valid_indices].to_numpy()
                if groups_series is not None else None)

            combined_returns = pd.concat([y_return_train, y_return_test])
            combined_cls_series, _, _, _ = rolling_quantile_classification_labels(
                combined_returns, **quantile_kwargs)
            if combined_cls_series.nunique() < 2:
                y_cls_test_quant = pd.Series((y_return_test
                                              > 0).astype(int).values,
                                             index=y_return_test.index)
            else:
                y_cls_test_quant = combined_cls_series.reindex(
                    y_return_test.index)

            test_valid_mask_array = (~y_cls_test_quant.isna()).to_numpy()
            y_cls_test_directional = (y_return_test > 0).astype(int)

            classification_preprocess_params = None
            return_preprocess_params = None
            vol_preprocess_params = None
            trainer_splits = max(
                2,
                args.cv_folds if (args.cv_on_rolling and args.cv_folds
                                  and args.cv_folds > 0) else 2,
            )
            try:
                models_dict, metrics_dict, preprocess_params_dict = classification_trainer.train_models(
                    X_df=X_train_df,
                    y_return=y_return_train,
                    y_vol=y_vol_train,
                    train_df=train_labeled,
                    n_splits=trainer_splits,
                    groups=groups,
                    preprocess_fn=None,
                    preprocess_kwargs={},
                    feature_winsorize_k=4.0,
                )
                model_cls = models_dict.get("classification")
                model_return = models_dict.get("return")
                model_vol = models_dict.get("vol")
                classification_preprocess_params = preprocess_params_dict.get(
                    "classification")
                return_preprocess_params = preprocess_params_dict.get("return")
                vol_preprocess_params = preprocess_params_dict.get("vol")
                classification_metrics_cv = metrics_dict.get(
                    "classification", {})
                if classification_metrics_cv:
                    cv_acc = classification_metrics_cv.get("accuracy")
                    cv_f1 = classification_metrics_cv.get("f1")
                    cv_auc = classification_metrics_cv.get("auc")
                    if all(val is not None for val in [cv_acc, cv_f1, cv_auc]):
                        print(
                            f"   CV metrics → Accuracy: {cv_acc:.3f} | F1: {cv_f1:.3f} | AUC: {cv_auc:.3f}"
                        )
            except Exception as exc:
                print(
                    f"   ⚠️ Classification trainer fallback: {exc}. Reverting to direct LightGBM training."
                )
                fallback_cv = (args.cv_folds if
                               (args.cv_on_rolling and args.cv_folds
                                and args.cv_folds > 1) else 0)
                model_cls = LightGBMModel(model_type="classification",
                                          use_gpu=args.gpu)
                _ = model_cls.train(X_train_cls,
                                    y_cls_train_filtered,
                                    n_splits=fallback_cv,
                                    use_time_series_cv=bool(fallback_cv),
                                    groups=groups_filtered)

                model_return = LightGBMModel(model_type="regression",
                                             use_gpu=args.gpu)
                _ = model_return.train(X_train_df,
                                       log_return_magnitude(y_return_train),
                                       n_splits=fallback_cv,
                                       use_time_series_cv=bool(fallback_cv))
                return_preprocess_params = {"target_transform": "log1p_abs"}

                model_vol = LightGBMModel(model_type="regression",
                                          use_gpu=args.gpu)
                _ = model_vol.train(X_train_df,
                                    y_vol_train,
                                    n_splits=fallback_cv,
                                    use_time_series_cv=bool(fallback_cv))

            # Ensure models are trained even if trainer returned placeholders
            fallback_cv = (args.cv_folds if
                           (args.cv_on_rolling and args.cv_folds
                            and args.cv_folds > 1) else 0)
            if model_cls is None or not getattr(model_cls, "is_trained",
                                                False):
                print(
                    "   ⚠️ Classification model unavailable after trainer run; using fallback LightGBM training."
                )
                model_cls = LightGBMModel(model_type="classification",
                                          use_gpu=args.gpu)
                _ = model_cls.train(X_train_cls,
                                    y_cls_train_filtered,
                                    n_splits=fallback_cv,
                                    use_time_series_cv=bool(fallback_cv),
                                    groups=groups_filtered)
                classification_preprocess_params = None

            if model_return is None or not getattr(model_return, "is_trained",
                                                   False):
                print(
                    "   ⚠️ Return model unavailable after trainer run; using fallback LightGBM training."
                )
                model_return = LightGBMModel(model_type="regression",
                                             use_gpu=args.gpu)
                _ = model_return.train(X_train_df,
                                       log_return_magnitude(y_return_train),
                                       n_splits=fallback_cv,
                                       use_time_series_cv=bool(fallback_cv))
                return_preprocess_params = {"target_transform": "log1p_abs"}

            if model_vol is None or not getattr(model_vol, "is_trained",
                                                False):
                print(
                    "   ⚠️ Volatility model unavailable after trainer run; using fallback LightGBM training."
                )
                model_vol = LightGBMModel(model_type="regression",
                                          use_gpu=args.gpu)
                _ = model_vol.train(X_train_df,
                                    y_vol_train,
                                    n_splits=fallback_cv,
                                    use_time_series_cv=bool(fallback_cv))
                vol_preprocess_params = None

            # Guard against edge cases where LightGBMModel preserves booster but flag remains unset
            for mdl in (model_cls, model_return, model_vol):
                if mdl is not None and getattr(mdl, "model", None) is not None:
                    mdl.is_trained = True

            # Evaluate
            print(
                f"   Model status → cls_trained={getattr(model_cls, 'is_trained', False)}, "
                f"return_trained={getattr(model_return, 'is_trained', False)}, "
                f"vol_trained={getattr(model_vol, 'is_trained', False)}")
            y_prob = model_cls.predict(X_test_df)
            valid_eval_mask = test_valid_mask_array
            if valid_eval_mask.sum() >= 2:
                prob_eval = y_prob[valid_eval_mask]
                y_cls_eval = y_cls_test_quant[valid_eval_mask].astype(
                    int).to_numpy()
                try:
                    threshold = _compute_direction_threshold(
                        prob_eval, y_cls_eval, method=args.direction_threshold)
                except Exception:
                    threshold = 0.5
                y_pred_dir_eval = (prob_eval > threshold).astype(int)
                cls_accuracy = float(
                    accuracy_score(y_cls_eval, y_pred_dir_eval))
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_cls_eval,
                    y_pred_dir_eval,
                    average='binary',
                    zero_division=0)
                try:
                    cls_auc = float(roc_auc_score(y_cls_eval, prob_eval))
                except Exception:
                    cls_auc = None
                try:
                    cls_pr_auc = float(
                        average_precision_score(y_cls_eval, prob_eval))
                except Exception:
                    cls_pr_auc = None
            else:
                prob_eval = y_prob
                y_cls_eval = y_cls_test_directional.to_numpy()
                try:
                    threshold = _compute_direction_threshold(
                        prob_eval, y_cls_eval, method=args.direction_threshold)
                except Exception:
                    threshold = 0.5
                y_pred_dir_eval = (prob_eval > threshold).astype(int)
                cls_accuracy = float(
                    accuracy_score(y_cls_eval, y_pred_dir_eval))
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_cls_eval,
                    y_pred_dir_eval,
                    average='binary',
                    zero_division=0)
                try:
                    cls_auc = float(roc_auc_score(y_cls_eval, prob_eval))
                except Exception:
                    cls_auc = None
                try:
                    cls_pr_auc = float(
                        average_precision_score(y_cls_eval, prob_eval))
                except Exception:
                    cls_pr_auc = None

            y_pred_return_log = model_return.predict(X_test_df)
            y_pred_return = invert_log_return_magnitude(y_pred_return_log)
            y_true_return_mag = np.abs(y_return_test.to_numpy())
            y_true_return_log = np.log1p(y_true_return_mag)
            y_pred_vol = model_vol.predict(X_test_df)
            ret_rmse = float(
                np.sqrt(mean_squared_error(y_true_return_mag, y_pred_return)))
            ret_mae = float(
                mean_absolute_error(y_true_return_mag, y_pred_return))
            ret_r2 = float(r2_score(y_true_return_mag, y_pred_return))
            ret_rmse_log = float(
                np.sqrt(
                    mean_squared_error(y_true_return_log, y_pred_return_log)))
            ret_mae_log = float(
                mean_absolute_error(y_true_return_log, y_pred_return_log))
            ret_r2_log = float(r2_score(y_true_return_log, y_pred_return_log))
            vol_rmse = float(
                np.sqrt(mean_squared_error(y_vol_test, y_pred_vol)))
            vol_mae = float(mean_absolute_error(y_vol_test, y_pred_vol))
            vol_r2 = float(r2_score(y_vol_test, y_pred_vol))

            train_samples = int(X_train_df.shape[0])
            test_samples = int(X_test_df.shape[0])
            eval_index = (y_return_test.index[valid_eval_mask] if
                          valid_eval_mask.sum() >= 2 else y_return_test.index)
            prob_series = pd.Series(prob_eval, index=eval_index)
            cls_series = pd.Series(y_cls_eval, index=eval_index)
            try:
                cls_ic_spearman = float(
                    cls_series.corr(prob_series, method="spearman"))
            except Exception:
                cls_ic_spearman = None
            try:
                cls_ic_pearson = float(
                    cls_series.corr(prob_series, method="pearson"))
            except Exception:
                cls_ic_pearson = None

            risk_adjusted_signal = (
                (2 * y_prob - 1) *
                (y_pred_return / np.maximum(y_pred_vol, 1e-6)))
            signals_df = pd.DataFrame(
                {
                    "signal_strength": risk_adjusted_signal,
                    "class_proba": y_prob,
                    "return_pred": y_pred_return,
                    "vol_pred": y_pred_vol,
                },
                index=X_test_df.index,
            )

            res = {
                "symbol": symbols_str,
                "timeframe": freq,
                "forward_bars": fb,
                "test_month": test_file["month_str"],
                "train_months": len(train_files),
                "num_features": len(feat_cols),
                "train_samples": train_samples,
                "test_samples": test_samples,
                "feature_type": args.feature_type,
                "cls_accuracy": cls_accuracy,
                "cls_precision": float(precision),
                "cls_recall": float(recall),
                "cls_f1": float(f1),
                "cls_auc": cls_auc,
                "cls_pr_auc": cls_pr_auc,
                "cls_threshold": float(threshold),
                "cls_ic_spearman": cls_ic_spearman,
                "cls_ic_pearson": cls_ic_pearson,
                "cls_eval_samples": int(len(eval_index)),
                "avg_signal_strength":
                float(np.mean(np.abs(risk_adjusted_signal))),
                "test_rmse_return": ret_rmse,
                "test_mae_return": ret_mae,
                "test_r2_return": ret_r2,
                "test_rmse_return_log": ret_rmse_log,
                "test_mae_return_log": ret_mae_log,
                "test_r2_return_log": ret_r2_log,
                "test_rmse_vol": vol_rmse,
                "test_mae_vol": vol_mae,
                "test_r2_vol": vol_r2,
            }

            imp_cls = _extract_importance(model_cls, feat_cols)
            imp_ret = _extract_importance(model_return, feat_cols)
            imp_vol = _extract_importance(model_vol, feat_cols)

            _accumulate_importance(imp_cls, "classification")
            _accumulate_importance(imp_ret, "return")
            _accumulate_importance(imp_vol, "volatility")

            res["feature_importance"] = {
                "classification":
                imp_cls.to_dict("records") if imp_cls is not None else None,
                "return":
                imp_ret.to_dict("records") if imp_ret is not None else None,
                "volatility":
                imp_vol.to_dict("records") if imp_vol is not None else None,
            }

            # Backtest the classification probabilities
            bt_results = evaluate_signal_performance(
                signals_df,
                y_return_test,
            )
            print_backtest_results(bt_results,
                                   f"{test_file['month_str']} Backtest")
            res.update({
                "total_trades":
                int(bt_results.get("total_trades", 0)),
                "total_return":
                float(bt_results.get("total_return", 0.0)),
                "win_rate":
                float(bt_results.get("win_rate", 0.0)),
                "avg_win":
                float(bt_results.get("avg_win", 0.0)),
                "avg_loss":
                float(bt_results.get("avg_loss", 0.0)),
                "profit_factor":
                float(bt_results.get("profit_factor", 0.0)),
                "max_drawdown":
                float(bt_results.get("max_drawdown", 0.0)),
                "final_equity":
                float(bt_results.get("final_equity", 0.0)),
            })

            all_results.append(res)
            print(
                f"      Test return RMSE(|r|): {ret_rmse:.6f}, MAE(|r|): {ret_mae:.6f}, log-RMSE: {ret_rmse_log:.6f}; cls F1: {f1:.3f}"
            )

            _write_window_report(
                res_row=res,
                cls_imp=imp_cls,
                ret_imp=imp_ret,
                vol_imp=imp_vol,
                train_months=train_files,
                test_month=test_file,
                output_dir=os.path.join(combo_dir, "window_reports"),
            )

            # Persist model artifacts
            base = f"fb{fb}_tf{freq}_{test_file['month_str']}"
            artifact_paths: Dict[str, str] = {}

            cls_pipeline = QuantTradingModel(
                model_type="classification",
                forward_bars=fb,
                feature_cols=feat_cols,
                preprocess_params=classification_preprocess_params,
                use_gpu=args.gpu,
            )
            cls_pipeline.model = model_cls.model
            if classification_preprocess_params:
                cls_pipeline.preprocessor = RobustWinsorizer.from_params(
                    classification_preprocess_params, forward_bars=fb)
            cls_pipeline_path = os.path.join(
                combo_dir, f"classification_pipeline_{base}.pkl")
            cls_pipeline.save(cls_pipeline_path)
            artifact_paths["classification_pipeline"] = cls_pipeline_path

            return_pipeline = QuantTradingModel(
                model_type="regression",
                forward_bars=fb,
                feature_cols=feat_cols,
                preprocess_params=return_preprocess_params,
                use_gpu=args.gpu,
                target_transform="log1p_abs",
            )
            return_pipeline.model = model_return.model
            if return_preprocess_params:
                return_pipeline.preprocessor = RobustWinsorizer.from_params(
                    return_preprocess_params, forward_bars=fb)
            return_pipeline_path = os.path.join(combo_dir,
                                                f"return_pipeline_{base}.pkl")
            return_pipeline.save(return_pipeline_path)
            artifact_paths["return_pipeline"] = return_pipeline_path

            vol_pipeline = QuantTradingModel(
                model_type="regression",
                forward_bars=fb,
                feature_cols=feat_cols,
                preprocess_params=None,
                use_gpu=args.gpu,
            )
            vol_pipeline.model = model_vol.model
            vol_pipeline_path = os.path.join(combo_dir,
                                             f"vol_pipeline_{base}.pkl")
            vol_pipeline.save(vol_pipeline_path)
            artifact_paths["vol_pipeline"] = vol_pipeline_path

            feature_engineer_obj = (baseline_engineer if args.feature_type
                                    == "baseline" else comp_engineer)
            scaler_path = os.path.join(combo_dir, f"scalers_{base}.pkl")
            if feature_engineer_obj is not None and hasattr(
                    feature_engineer_obj, "save_scalers"):
                try:
                    feature_engineer_obj.save_scalers(scaler_path)
                    artifact_paths["scalers"] = scaler_path
                except Exception as exc:
                    print(f"   ⚠️ Failed to save scalers: {exc}")

            features_path = os.path.join(combo_dir, f"features_{base}.txt")
            try:
                with open(features_path, "w") as f:
                    f.write("\n".join(feat_cols))
                artifact_paths["features"] = features_path
            except Exception as exc:
                print(f"   ⚠️ Failed to write features file: {exc}")

            trades = bt_results.get("trades", [])
            if trades:
                trades_path = os.path.join(combo_dir, f"trades_{base}.json")
                try:
                    serialized_trades = []
                    for trade in trades:
                        serialized_trades.append({
                            key: (str(value) if isinstance(
                                value, pd.Timestamp) else value)
                            for key, value in trade.items()
                        })
                    with open(trades_path, "w") as f:
                        json.dump(serialized_trades, f, indent=2)
                    artifact_paths["trades"] = trades_path
                except Exception as exc:
                    print(f"   ⚠️ Failed to persist trade log: {exc}")

            res["artifacts"] = artifact_paths
            artifacts_entries.append({
                "test_month": test_file["month_str"],
                **artifact_paths
            })

            # Save raw LightGBM models for reference
            model_cls.model.save_model(
                os.path.join(combo_dir, f"model_direction_{base}.txt"))
            model_return.model.save_model(
                os.path.join(combo_dir, f"model_return_{base}.txt"))
            model_vol.model.save_model(
                os.path.join(combo_dir, f"model_volatility_{base}.txt"))

        # Save summary/report
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(combo_dir, "monthly_results.csv"),
                          index=False)
        train_start_date = files[0]["month_str"] if files else None
        test_end_date = files[-1]["month_str"] if files else None
        if not results_df.empty and "test_month" in results_df.columns:
            test_end_date = results_df["test_month"].max(
            ) if not results_df.empty else test_end_date
        avg_test_rmse = float(results_df["test_rmse_return"].mean(
        )) if "test_rmse_return" in results_df.columns and len(
            results_df) > 0 else None
        avg_test_mae = float(results_df["test_mae_return"].mean(
        )) if "test_mae_return" in results_df.columns and len(
            results_df) > 0 else None
        avg_cls_f1 = float(results_df["cls_f1"].mean(
        )) if "cls_f1" in results_df.columns and len(results_df) > 0 else None
        avg_cls_auc = float(results_df["cls_auc"].mean(
        )) if "cls_auc" in results_df.columns and len(results_df) > 0 else None
        avg_return_r2 = float(results_df["test_r2_return"].mean(
        )) if "test_r2_return" in results_df.columns and len(
            results_df) > 0 else None
        avg_vol_r2 = float(results_df["test_r2_vol"].mean()
                           ) if "test_r2_vol" in results_df.columns and len(
                               results_df) > 0 else None
        total_trades_sum = int(results_df["total_trades"].sum()
                               ) if "total_trades" in results_df.columns else 0
        avg_total_return = float(results_df["total_return"].mean(
        )) if "total_return" in results_df.columns and len(
            results_df) > 0 else None
        avg_win_rate = float(
            results_df["win_rate"].mean()
        ) if "win_rate" in results_df.columns and len(results_df) > 0 else None
        avg_profit_factor = float(results_df["profit_factor"].mean(
        )) if "profit_factor" in results_df.columns and len(
            results_df) > 0 else None
        avg_max_drawdown = float(results_df["max_drawdown"].mean(
        )) if "max_drawdown" in results_df.columns and len(
            results_df) > 0 else None
        avg_final_equity = float(results_df["final_equity"].mean(
        )) if "final_equity" in results_df.columns and len(
            results_df) > 0 else None
        summary = {
            "symbol":
            symbols_str,
            "feature_type":
            args.feature_type,
            "total_months_tested":
            len(results_df),
            "train_start_date":
            train_start_date,
            "test_end_date":
            test_end_date,
            "avg_test_rmse":
            avg_test_rmse,
            "avg_test_mae":
            avg_test_mae,
            "avg_cls_f1":
            avg_cls_f1,
            "avg_cls_auc":
            avg_cls_auc,
            "avg_return_r2":
            avg_return_r2,
            "avg_vol_r2":
            avg_vol_r2,
            "avg_total_return":
            avg_total_return,
            "avg_win_rate":
            avg_win_rate,
            "avg_profit_factor":
            avg_profit_factor,
            "avg_max_drawdown":
            avg_max_drawdown,
            "avg_final_equity":
            avg_final_equity,
            "total_trades":
            total_trades_sum,
            "created_at":
            datetime.now().isoformat(),
            "feature_engineering":
            "ComprehensiveFeatureEngineer"
            if args.feature_type != "baseline" else "BaselineFeatureEngineer",
            "configuration": {
                "symbol": symbols_str,
                "data_dir": args.data_dir,
                "initial_train_months": args.initial_train_months,
                "min_train_months": args.min_train_months,
                "forward_bars": args.forward_bars,
                "gpu": args.gpu,
                "timeframe": freq,
                "start": getattr(args, "start", None),
                "end": getattr(args, "end", None),
                "feature_type": args.feature_type,
            },
            "artifacts":
            artifacts_entries,
            "monthly_results":
            all_results,
        }

        def _finalize_importance(
                acc: Dict[str, Dict[str, float]]) -> Dict[str, List[Dict]]:
            finalized: Dict[str, List[Dict]] = {}
            for bucket, data in acc.items():
                if not data:
                    continue
                sorted_items = sorted(data.items(),
                                      key=lambda x: x[1],
                                      reverse=True)[:100]
                finalized[bucket] = [{
                    "feature": feat,
                    "importance": float(val)
                } for feat, val in sorted_items]
            return finalized

        summary["feature_importance"] = _finalize_importance(
            importance_accumulators)
        with open(os.path.join(combo_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        try:
            from time_series_model.pipeline.dimensionality.report_generator import write_rolling_report
            from time_series_model.pipeline.training.generate_summary_report import generate_summary_report
            report_path = write_rolling_report(
                combo_dir,
                summary_path=os.path.join(combo_dir, "summary.json"),
                results_csv_path=os.path.join(combo_dir,
                                              "monthly_results.csv"),
                report_type="monthly",
            )
            print(f"   - combo report: {report_path}")
            # Generate summary report (HTML) for rolling results
            rolling_summary_dir = os.path.join(combo_dir, "summary")
            os.makedirs(rolling_summary_dir, exist_ok=True)
            summary_html = generate_summary_report(results_dir=combo_dir,
                                                   output_path=os.path.join(
                                                       rolling_summary_dir,
                                                       "summary_report.html"))
            if summary_html:
                print(f"   - summary report (train-style): {summary_html}")
        except Exception as exc:
            print(f"   ⚠️  Failed to generate HTML report: {exc}")

    print("\n✅ Rolling completed. Results saved to:")
    print(f"   {results_dir}")


if __name__ == "__main__":
    main()
