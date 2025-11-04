from __future__ import annotations
"""
Single-run regression training:
- Predict future_return (mean), future_return quantiles (q10/q90) for uncertainty,
- Predict future_volatility (realized volatility) for risk-aware sizing.

Feature selection options: --feature-type, --use-top-factors, --topk, --topk-source
"""

# Copied from baseline.train_baseline with naming/docs adjusted
import os
import argparse
import json
from typing import List
import numpy as np
import pandas as pd

from ml_trading.data_tools.rolling_data import load_parquet_file
from ml_trading.data_tools.baseline_feature_engineering import (
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)
from ml_trading.models.lightgbm_model import LightGBMModel


def _load_many(files: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for f in files:
        df = load_parquet_file(f) if f.endswith(".parquet") else None
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid data files loaded")
    return pd.concat(frames, axis=0).sort_index()


def _collect_files(data: List[str],
                   data_dir: str | None,
                   start: str | None,
                   end: str | None,
                   symbol: str | None = None) -> List[str]:
    files: List[str] = []
    files.extend(data)
    if data_dir and os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".parquet"):
                files.append(os.path.join(data_dir, name))
    files = [os.path.abspath(p) for p in files if os.path.exists(p)]

    if symbol:
        mapping = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
            "BNBUSDT": "BNB-USD",
            "ADAUSDT": "ADA-USD",
            "SOLUSDT": "SOL-USD"
        }
        file_symbol = mapping.get(symbol, symbol.replace("USDT", "-USD"))
        filtered = []
        for p in files:
            fn = os.path.basename(p).upper()
            if (fn.startswith(symbol.upper())
                    or fn.startswith(file_symbol.upper())
                    or fn.startswith(file_symbol.replace("-", "_").upper())):
                filtered.append(p)
        files = filtered

    if start or end:
        import re

        def _ym(n: str) -> str | None:
            m = re.search(r"(20\d{2})[-_](\d{2})", os.path.basename(n))
            return f"{m.group(1)}-{m.group(2)}" if m else None

        filtered = []
        for p in files:
            ym = _ym(p)
            if ym is None:
                continue
            if start and ym < start:
                continue
            if end and ym > end:
                continue
            filtered.append(p)
        files = filtered
    if not files:
        raise FileNotFoundError("No parquet files found from inputs")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regression training (returns + uncertainty + volatility)")
    parser.add_argument("--data",
                        type=str,
                        action="append",
                        default=[],
                        help="Parquet file(s) to use")
    parser.add_argument("--data-dir",
                        type=str,
                        default=None,
                        help="Directory containing parquet files")
    parser.add_argument("--symbol",
                        type=str,
                        default="BTCUSDT",
                        help="Symbol metadata for report")
    parser.add_argument("--freq",
                        type=str,
                        default="5T",
                        help="Bar timeframe(s), comma-separated: 5T,15T")
    parser.add_argument("--start",
                        type=str,
                        default=None,
                        help="Start YYYY-MM (inclusive)")
    parser.add_argument("--end",
                        type=str,
                        default=None,
                        help="End YYYY-MM (inclusive)")
    parser.add_argument("--forward-bars",
                        type=str,
                        default="3",
                        help="Bars ahead (e.g., 1,5,10)")
    parser.add_argument("--cv-folds",
                        type=int,
                        default=0,
                        help="TimeSeries CV folds (0=disable)")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="baseline",
        help="baseline/default/enhanced/dl_sequence/comprehensive or combos")
    parser.add_argument("--oos-months",
                        type=int,
                        default=3,
                        help="OOS months after train end (0=disable)")
    parser.add_argument("--oos-start",
                        type=str,
                        default=None,
                        help="OOS start (YYYY-MM-DD)")
    parser.add_argument("--oos-end",
                        type=str,
                        default=None,
                        help="OOS end (YYYY-MM-DD)")
    parser.add_argument("--use-top-factors",
                        type=str,
                        default=None,
                        help="JSON of selected features to keep")
    parser.add_argument("--topk",
                        type=int,
                        default=0,
                        help="Keep only Top-K features (0=disabled)")
    parser.add_argument(
        "--topk-source",
        type=str,
        default=None,
        help="Ranking CSV(feature,score) or JSON list; fallback |IC|")
    parser.add_argument("--gpu",
                        action="store_true",
                        default=True,
                        help="Use GPU for LightGBM")
    args = parser.parse_args()

    freqs = [f.strip() for f in args.freq.split(",") if f.strip()]
    fbs = [int(x.strip()) for x in args.forward_bars.split(",") if x.strip()]

    files = _collect_files(args.data,
                           args.data_dir,
                           args.start,
                           args.end,
                           symbol=args.symbol)
    raw = _load_many(files)

    for freq in freqs:
        for fb in fbs:
            feat_df = raw.copy()
            if args.feature_type == "baseline":
                feat_df, base_eng = engineer_baseline_features(feat_df,
                                                               None,
                                                               fit=True)
                feature_engineer = base_eng
            else:
                feature_engineer = ComprehensiveFeatureEngineer(
                    feature_types=args.feature_type)
                feat_df = feature_engineer.engineer_all_features(feat_df,
                                                                 fit=True)

            feat_df["future_return"] = feat_df["close"].shift(
                -fb) / feat_df["close"] - 1
            one = feat_df["close"].pct_change()
            # Use a safe rolling window: window>=2 and ddof=0 so fb=1 works
            from math import prod as _prod  # dummy import to avoid unused import lints elsewhere
            safe_window = max(2, fb)
            feat_df["future_volatility"] = (one.shift(-1).rolling(
                window=safe_window, min_periods=1).std(ddof=0))
            # Only drop rows where targets are NaN; allow feature NaNs (handled later)
            feat_df = feat_df.dropna(
                subset=["future_return", "future_volatility"]).copy()

            from dateutil.relativedelta import relativedelta
            train_end = feat_df.index.max() if not feat_df.empty else None
            oos_start_dt = None
            oos_end_dt = None
            oos_df = pd.DataFrame()
            if args.oos_months > 0 or args.oos_start is not None:
                if args.oos_start:
                    try:
                        oos_start_dt = pd.to_datetime(args.oos_start)
                    except Exception:
                        oos_start_dt = None
                if oos_start_dt is None and train_end is not None:
                    oos_start_dt = train_end + relativedelta(
                        months=args.oos_months)
                if args.oos_end:
                    try:
                        oos_end_dt = pd.to_datetime(args.oos_end)
                    except Exception:
                        oos_end_dt = None
                if oos_end_dt is None and oos_start_dt is not None:
                    oos_end_dt = oos_start_dt + relativedelta(months=3)
                if oos_start_dt is not None and oos_end_dt is not None:
                    oos_mask = (feat_df.index >= oos_start_dt) & (
                        feat_df.index <= oos_end_dt)
                    oos_df = feat_df[oos_mask].copy()

            train_df = feat_df if len(
                oos_df) == 0 or oos_start_dt is None else feat_df[
                    feat_df.index < oos_start_dt]

            if args.feature_type == "baseline":
                feature_cols = get_baseline_feature_columns(train_df)
            else:
                feature_cols = get_feature_columns_by_type(
                    train_df, args.feature_type)
            # optional top-factors
            if args.use_top_factors:
                try:
                    with open(args.use_top_factors, "r",
                              encoding="utf-8") as f:
                        top = json.load(f)
                    if isinstance(top, dict) and "features" in top:
                        top = top["features"]
                    if isinstance(top, list):
                        s = set(top)
                        feature_cols = [c for c in feature_cols if c in s]
                except Exception:
                    pass
            # numeric only
            feature_cols = [
                c for c in feature_cols
                if pd.api.types.is_numeric_dtype(train_df[c])
            ]
            # optional top-k
            if args.topk and args.topk > 0 and len(feature_cols) > args.topk:
                ranked = None
                if args.topk_source:
                    try:
                        if args.topk_source.lower().endswith(".csv"):
                            _df = pd.read_csv(args.topk_source)
                            if {"feature", "score"}.issubset(set(_df.columns)):
                                _df = _df.sort_values("score", ascending=False)
                                ranked = [
                                    f for f in _df["feature"].tolist()
                                    if f in feature_cols
                                ]
                        else:
                            lst = json.load(
                                open(args.topk_source, "r", encoding="utf-8"))
                            if isinstance(lst, dict) and "features" in lst:
                                lst = lst["features"]
                            if isinstance(lst, list):
                                ranked = [f for f in lst if f in feature_cols]
                    except Exception:
                        ranked = None
                if ranked is None:
                    try:
                        from scipy.stats import spearmanr
                        ic = []
                        for c in feature_cols:
                            try:
                                r, _ = spearmanr(
                                    train_df[c].values,
                                    train_df["future_return"].values,
                                    nan_policy="omit")
                                ic.append((c, abs(r) if pd.notna(r) else 0.0))
                            except Exception:
                                ic.append((c, 0.0))
                        ic.sort(key=lambda x: x[1], reverse=True)
                        ranked = [c for c, _ in ic]
                    except Exception:
                        ranked = feature_cols
                feature_cols = ranked[:args.topk]

            X_df = pd.DataFrame(train_df[feature_cols].values,
                                columns=feature_cols,
                                index=train_df.index)
            y_return = pd.Series(train_df["future_return"].values,
                                 index=train_df.index)
            y_vol = pd.Series(train_df["future_volatility"].values,
                              index=train_df.index)

            use_cv = args.cv_folds > 0
            n_splits = args.cv_folds if use_cv else 0

            # q50: median as primary point estimate (using new quantile API)
            model_q50 = LightGBMModel(model_type="quantile",
                                      quantile_alpha=0.5,
                                      use_gpu=args.gpu)
            # Use TimeSeries CV by default to avoid random split failures on edge cases
            q50_metrics = model_q50.train(X_df,
                                          y_return,
                                          n_splits=max(2, args.cv_folds or 2),
                                          use_time_series_cv=True)

            # q10: 10% quantile for uncertainty estimation
            model_q10 = LightGBMModel(model_type="quantile",
                                      quantile_alpha=0.1,
                                      use_gpu=args.gpu)
            _ = model_q10.train(X_df,
                                y_return,
                                n_splits=max(2, args.cv_folds or 2),
                                use_time_series_cv=True)

            # q90: 90% quantile for uncertainty estimation
            model_q90 = LightGBMModel(model_type="quantile",
                                      quantile_alpha=0.9,
                                      use_gpu=args.gpu)
            _ = model_q90.train(X_df,
                                y_return,
                                n_splits=max(2, args.cv_folds or 2),
                                use_time_series_cv=True)

            # volatility: regression model for volatility prediction
            model_vol = LightGBMModel(model_type="regression",
                                      use_gpu=args.gpu)
            vol_metrics = model_vol.train(X_df,
                                          y_vol,
                                          n_splits=n_splits,
                                          use_time_series_cv=use_cv)

            # Classification/regression metrics containers
            oos_metrics = {}
            cls_metrics_train = {}
            if len(oos_df) > 0:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
                X_oos = oos_df[feature_cols].values
                y_ret_oos = oos_df["future_return"].values
                y_vol_oos = oos_df["future_volatility"].values
                y_pred_q50 = model_q50.model.predict(X_oos)
                oos_rmse = float(
                    np.sqrt(mean_squared_error(y_ret_oos, y_pred_q50)))
                oos_mae = float(mean_absolute_error(y_ret_oos, y_pred_q50))
                y_pred_q10 = model_q10.model.predict(X_oos)
                y_pred_q90 = model_q90.model.predict(X_oos)
                coverage = float(
                    np.mean((y_ret_oos >= y_pred_q10)
                            & (y_ret_oos <= y_pred_q90)))
                width = float(np.mean(np.maximum(0.0,
                                                 y_pred_q90 - y_pred_q10)))
                conf = float(
                    np.mean(
                        np.abs(y_pred_q50) /
                        (np.maximum(1e-8, y_pred_q90 - y_pred_q10))))
                y_pred_vol = model_vol.model.predict(X_oos)
                oos_vol_rmse = float(
                    np.sqrt(mean_squared_error(y_vol_oos, y_pred_vol)))
                oos_vol_mae = float(mean_absolute_error(y_vol_oos, y_pred_vol))
                # Derive classification labels from returns (directional)
                y_true_cls = (y_ret_oos > 0).astype(int)
                y_score = y_pred_q50
                y_pred_cls = (y_score > 0).astype(int)
                acc = float(accuracy_score(y_true_cls, y_pred_cls))
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_cls, y_pred_cls, average="binary", zero_division=0)
                try:
                    auc = float(roc_auc_score(y_true_cls, y_score))
                except Exception:
                    auc = float("nan")
                try:
                    pr_auc = float(average_precision_score(
                        y_true_cls, y_score))
                except Exception:
                    pr_auc = float("nan")
                oos_metrics = {
                    "stage1": {
                        "accuracy": acc,
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1": float(f1),
                        "auc": auc,
                        "pr_auc": pr_auc,
                        "samples": int(len(y_true_cls)),
                        "best_threshold": 0.0,
                        "quality_check": {
                            "passed":
                            bool(f1 >= 0.3
                                 or (not np.isnan(auc) and auc >= 0.6)),
                            "issues": []
                        },
                    },
                    "regression_return": {
                        "rmse": oos_rmse,
                        "mae": oos_mae,
                        "samples": len(oos_df)
                    },
                    "uncertainty": {
                        "coverage_10_90": coverage,
                        "avg_interval_width": width,
                        "avg_confidence": conf
                    },
                    "regression_volatility": {
                        "rmse": oos_vol_rmse,
                        "mae": oos_vol_mae,
                        "samples": len(oos_df)
                    },
                }
            else:
                # In-sample directional metrics for visibility when no OOS period
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
                X_all = train_df[feature_cols].values
                y_ret_all = train_df["future_return"].values
                y_score_all = model_q50.model.predict(X_all)
                y_true_cls_all = (y_ret_all > 0).astype(int)
                y_pred_cls_all = (y_score_all > 0).astype(int)
                acc = float(accuracy_score(y_true_cls_all, y_pred_cls_all))
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_cls_all,
                    y_pred_cls_all,
                    average="binary",
                    zero_division=0)
                try:
                    auc = float(roc_auc_score(y_true_cls_all, y_score_all))
                except Exception:
                    auc = float("nan")
                try:
                    pr_auc = float(
                        average_precision_score(y_true_cls_all, y_score_all))
                except Exception:
                    pr_auc = float("nan")
                cls_metrics_train = {
                    "accuracy": acc,
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "auc": auc,
                    "pr_auc": pr_auc,
                    "samples": int(len(y_true_cls_all)),
                }

            # Stage1 classification-style metrics (direction from q50 regression)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
            y_score = model_q50.model.predict(X_df.values)
            y_true_bin = (y_return.values > 0).astype(int)
            y_pred_bin = (y_score > 0).astype(int)
            try:
                auc = float(roc_auc_score(y_true_bin, y_score))
            except Exception:
                auc = None
            try:
                pr_auc = float(average_precision_score(y_true_bin, y_score))
            except Exception:
                pr_auc = None
            cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()
            stage1_metrics = {
                "accuracy":
                float(accuracy_score(y_true_bin, y_pred_bin)),
                "precision":
                float(precision_score(y_true_bin, y_pred_bin,
                                      zero_division=0)),
                "recall":
                float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "f1":
                float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
                "auc":
                auc,
                "pr_auc":
                pr_auc,
                "best_threshold":
                0.0,
                "samples":
                int(len(y_true_bin)),
                "confusion_matrix":
                cm,
            }

            # Save artifacts and report (neutral naming, no 'baseline')
            combo_dir = "results/training"
            if len(freqs) > 1 or len(fbs) > 1:
                combo_dir = os.path.join("results/training",
                                         f"fb{fb}_tf{freq}")
            os.makedirs(combo_dir, exist_ok=True)
            model_q50.model.save_model(
                os.path.join(combo_dir, "return_q50_model.txt"))
            model_q10.model.save_model(
                os.path.join(combo_dir, "return_q10_model.txt"))
            model_q90.model.save_model(
                os.path.join(combo_dir, "return_q90_model.txt"))
            model_vol.model.save_model(
                os.path.join(combo_dir, "volatility_model.txt"))

            scaler_path = os.path.join(combo_dir, "scalers.pkl")
            if args.feature_type == "baseline":
                if feature_engineer is not None:
                    feature_engineer.save_scalers(scaler_path)
            else:
                if feature_engineer is not None and hasattr(
                        feature_engineer, "save_scalers"):
                    feature_engineer.save_scalers(scaler_path)

            with open(os.path.join(combo_dir, "features.txt"), "w") as f:
                f.write("\n".join(feature_cols))

            from datetime import datetime as _dt
            info_path = os.path.join(combo_dir, "training_info.json")
            model_info = {
                "model_path":
                os.path.join(combo_dir, "return_q50_model.txt"),
                "scaler_path":
                scaler_path,
                "training_date":
                _dt.now().isoformat(),
                "symbol":
                args.symbol,
                "actual_start":
                feat_df.index.min().isoformat() if not feat_df.empty else None,
                "actual_end":
                feat_df.index.max().isoformat() if not feat_df.empty else None,
                "train_start":
                train_df.index.min().isoformat()
                if not train_df.empty else None,
                "train_end":
                train_df.index.max().isoformat()
                if not train_df.empty else None,
                "total_bars":
                len(feat_df),
                "train_bars":
                len(train_df),
                "oos_months":
                args.oos_months if len(oos_df) > 0 else 0,
                "timeframes": {
                    freq: len(feat_df)
                },
                "price_range": [
                    float(feat_df["close"].min()) if not feat_df.empty else 0,
                    float(feat_df["close"].max()) if not feat_df.empty else 0
                ],
                "metrics": {
                    "stage2": {
                        freq: q50_metrics
                    },
                    "volatility": {
                        freq: vol_metrics
                    },
                    "classification_train": {
                        freq: cls_metrics_train
                    } if cls_metrics_train else {}
                },
                "feature_engineering":
                "BaselineFeatureEngineer" if args.feature_type == "baseline"
                else f"ComprehensiveFeatureEngineer({args.feature_type})",
                "feature_type":
                args.feature_type,
                "forward_bars":
                fb,
                "timeframe":
                freq,
                "data_files":
                files,
            }
            if oos_metrics:
                model_info["oos_metrics"] = oos_metrics
            with open(info_path, "w") as f:
                json.dump(model_info, f, indent=2, default=str)

            # Write a compact training HTML report (self-contained)
            report_path = os.path.join(combo_dir, "training_report.html")
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info_json = json.load(f)
                tf_metrics = info_json.get("metrics", {}).get("stage2", {})
                cls_train_metrics = info_json.get("metrics", {}).get(
                    "classification_train", {})
                oos_section = ""
                if info_json.get("oos_metrics"):
                    s1 = info_json["oos_metrics"].get("stage1", {})
                    oos_section = (
                        f"<h2>Classification (OOS)</h2><table><tr><th>Metric</th><th>Value</th></tr>"
                        f"<tr><td>Accuracy</td><td>{s1.get('accuracy','N/A')}</td></tr>"
                        f"<tr><td>Precision</td><td>{s1.get('precision','N/A')}</td></tr>"
                        f"<tr><td>Recall</td><td>{s1.get('recall','N/A')}</td></tr>"
                        f"<tr><td>F1</td><td>{s1.get('f1','N/A')}</td></tr>"
                        f"<tr><td>AUC</td><td>{s1.get('auc','N/A')}</td></tr>"
                        f"<tr><td>PR-AUC</td><td>{s1.get('pr_auc','N/A')}</td></tr>"
                        f"<tr><td>Samples</td><td>{s1.get('samples',0)}</td></tr></table>"
                    )
                rows = []
                for tf, m in tf_metrics.items():
                    rows.append(
                        f"<tr><td>{tf}</td><td>{m.get('cv_rmse','N/A')}</td><td>{m.get('cv_mse','N/A')}</td></tr>"
                    )
                cls_section = ""
                if cls_train_metrics:
                    cls_rows = "".join([
                        f"<tr><td>{tf}</td><td>{cls_train_metrics.get(tf,{}).get('accuracy','N/A')}</td><td>{cls_train_metrics.get(tf,{}).get('precision','N/A')}</td><td>{cls_train_metrics.get(tf,{}).get('recall','N/A')}</td><td>{cls_train_metrics.get(tf,{}).get('f1','N/A')}</td><td>{cls_train_metrics.get(tf,{}).get('auc','N/A')}</td><td>{cls_train_metrics.get(tf,{}).get('pr_auc','N/A')}</td><td>{cls_train_metrics.get(tf,{}).get('samples','N/A')}</td></tr>"
                        for tf in cls_train_metrics
                    ])
                    cls_section = (
                        "<h2>Classification (Train, directional)</h2>"
                        "<table><tr><th>Timeframe</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>PR-AUC</th><th>Samples</th></tr>"
                        + cls_rows + "</table>")
                html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Training Report</title>
                <style>body{{font-family:Arial;margin:24px;color:#222}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:8px 10px}}</style>
                </head><body>
                <h1>Training Report</h1>
                <p><strong>Symbol:</strong> {info_json.get('symbol')} &nbsp; <strong>Period:</strong> {info_json.get('actual_start','N/A')} → {info_json.get('actual_end','N/A')}</p>
                <p><strong>Total Bars:</strong> {info_json.get('total_bars',0)} &nbsp; <strong>Train Bars:</strong> {info_json.get('train_bars','N/A')}</p>
                {cls_section}
                <h2>Stage2 (Regression) CV Metrics</h2>
                <table><tr><th>Timeframe</th><th>CV RMSE</th><th>CV MSE</th></tr>{''.join(rows)}</table>
                {oos_section}
                <h2>Artifacts</h2>
                <ul>
                  <li>Model: {info_json.get('model_path')}</li>
                  <li>Scalers: {info_json.get('scaler_path')}</li>
                </ul>
                </body></html>"""
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"📝 HTML report written to: {report_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Note: Could not write compact training report: {exc}")

            # Additionally update training summary report across configs
            try:
                from ml_trading.pipeline.training.generate_summary_report import generate_summary_report
                generate_summary_report("results/training", None)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Note: Could not generate training summary report: {exc}")


if __name__ == "__main__":
    main()
