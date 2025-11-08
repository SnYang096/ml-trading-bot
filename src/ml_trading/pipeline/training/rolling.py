from __future__ import annotations
"""
Rolling regression training (returns + uncertainty + volatility) with unified feature selection.
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict

import pandas as pd
import numpy as np

from ml_trading.data_tools.rolling_data import load_and_process_file
from ml_trading.data_tools.baseline_feature_engineering import (
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)
from ml_trading.models.lightgbm_model import LightGBMModel
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score)
from sklearn.preprocessing import StandardScaler
from ml_trading.pipeline.training.train import _compute_direction_threshold


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
    symbol_mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "BNBUSDT": "BNB-USD",
        "ADAUSDT": "ADA-USD",
        "SOLUSDT": "SOL-USD"
    }
    
    all_files = []
    for symbol in symbol_list:
        file_symbol = symbol_mapping.get(symbol, symbol)
        patterns = [
            f"{symbol}-aggTrades-*.parquet", f"{file_symbol}_*.parquet",
            f"{file_symbol}-*.parquet"
        ]
        
        date_patterns = [
            rf"{symbol}-aggTrades-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            rf"{file_symbol}_(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            rf"{file_symbol}-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
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
                            "symbol": symbol,  # Track which symbol this file belongs to
                            "year": year,
                            "month": month,
                            "month_str": f"{year}-{month:02d}",
                            "timestamp": pd.Timestamp(year, month, 1),
                        }
                        # Avoid duplicates
                        if not any(f["path"] == file_info["path"] for f in all_files):
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
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Symbol(s) for rolling training. Can be comma-separated (e.g., BTCUSDT,ETHUSDT,SOLUSDT) for multi-asset training")
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
        help="Threshold method for directional prediction (zero|median|f1_optimize)",
    )
    parser.add_argument("--use-top-factors",
                        type=str,
                        default=None,
                        help="Optional JSON of selected features to keep")
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
    symbols_str = ",".join(symbol_list) if len(symbol_list) > 1 else symbol_list[0] if symbol_list else "UNKNOWN"
    print(f"📊 Rolling training with symbol(s): {symbols_str}")
    if len(symbol_list) > 1:
        print(f"   Multi-asset training: {len(symbol_list)} assets")
    
    # Validate autoencoder arguments
    if args.use_autoencoder and not args.encoding_dim:
        print("❌ --encoding-dim is required when --use-autoencoder is provided")
        return
    
    if args.use_autoencoder:
        print(f"   Autoencoder: {args.use_autoencoder} (dim={args.encoding_dim})")
    
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
        symbol_name = symbols_str.replace(",", "_").lower() if len(symbol_list) > 1 else symbol_list[0].lower() if symbol_list else "unknown"
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

        def _accumulate_importance(df: Optional[pd.DataFrame],
                                   bucket: str) -> None:
            if df is None or df.empty:
                return
            store = importance_accumulators.setdefault(bucket, {})
            for _, row in df.iterrows():
                feat = row["feature"]
                val = float(row["importance"])
                store[feat] = store.get(feat, 0.0) + val

        def _extract_importance(model: LightGBMModel,
                                feature_names: List[str]
                                ) -> Optional[pd.DataFrame]:
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
                f"Train: {train_files[0]['month_str']} → {train_files[-1]['month_str']} ({len(train_files)} months)"
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
                print(f"   Multi-asset training: {len(train_parts)} files merged, {len(train_df)} samples")

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
                # 🔒 CRITICAL FIX: Cannot use rolling std for future_volatility as it introduces future information
                # Previous bug: one.shift(-1).rolling(window=fb).std() was using future data (data leakage!)
                # future_volatility[t] = std(returns[t+1:t+1+fb]) requires future returns, introducing future information
                # ✅ Correct approach: Use abs(future_return) or future_return^2 as volatility proxy
                # This is consistent with train.py and safe_multi_asset_preprocessing.py
                out["future_volatility"] = out["future_return"].abs()
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
                try:
                    with open(args.use_top_factors, 'r',
                              encoding='utf-8') as _f:
                        keep = json.load(_f)
                    if isinstance(keep, dict) and 'features' in keep:
                        keep = keep['features']
                    if isinstance(keep, list):
                        s = set(keep)
                        feat_cols = [c for c in feat_cols if c in s]
                except Exception:
                    pass

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
                    from ml_trading.models.autoencoder import UnifiedAutoencoder
                    
                    print(f"   🔄 Applying autoencoder compression ({len(feat_cols)} → {args.encoding_dim})...")
                    
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
                    state = torch.load(args.use_autoencoder, map_location="cpu")
                    autoencoder.load_state_dict(state)
                    autoencoder.eval()
                    
                    # Transform features
                    with torch.no_grad():
                        X_train_tensor = torch.as_tensor(X_train_scaled, dtype=torch.float32)
                        X_test_tensor = torch.as_tensor(X_test_scaled, dtype=torch.float32)
                        _, Z_train = autoencoder(X_train_tensor)
                        _, Z_test = autoencoder(X_test_tensor)
                        X_train = Z_train.numpy()
                        X_test = Z_test.numpy()
                    
                    # Update feature columns for compressed features
                    feat_cols = [f"compressed_feature_{i}" for i in range(encoding_dim)]
                    
                    print(f"   ✓ Applied autoencoder compression: {len(feat_cols)} compressed features")
                except Exception as exc:
                    print(f"   ⚠️ Failed to apply autoencoder compression: {exc}")
                    print("   Falling back to original features")
                    # Fall back to original features
                    X_train = train_labeled[feat_cols].values
                    X_test = test_labeled[feat_cols].values
            else:
                # Use original features
                X_train = train_labeled[feat_cols].values
                X_test = test_labeled[feat_cols].values
            
            y_ret_tr = train_labeled['future_return'].values
            y_vol_tr = train_labeled['future_volatility'].values
            y_ret_te = test_labeled['future_return'].values
            y_vol_te = test_labeled['future_volatility'].values

            n_splits = args.cv_folds if (args.cv_on_rolling
                                         and args.cv_folds > 0) else 0
            y_cls_tr = (y_ret_tr > 0).astype(int)
            y_cls_te = (y_ret_te > 0).astype(int)

            model_cls = LightGBMModel(model_type="classification",
                                      use_gpu=args.gpu)
            _ = model_cls.train(pd.DataFrame(X_train, columns=feat_cols),
                                pd.Series(y_cls_tr),
                                n_splits=n_splits,
                                use_time_series_cv=bool(n_splits))

            model_return = LightGBMModel(model_type="regression",
                                         use_gpu=args.gpu)
            _ = model_return.train(pd.DataFrame(X_train, columns=feat_cols),
                                   pd.Series(y_ret_tr),
                                   n_splits=n_splits,
                                   use_time_series_cv=bool(n_splits))

            model_vol = LightGBMModel(model_type="regression",
                                      use_gpu=args.gpu)
            _ = model_vol.train(pd.DataFrame(X_train, columns=feat_cols),
                                pd.Series(y_vol_tr),
                                n_splits=n_splits,
                                use_time_series_cv=bool(n_splits))

            # Evaluate
            y_prob = model_cls.model.predict(X_test, raw_score=False)
            try:
                threshold = _compute_direction_threshold(
                    y_prob, y_cls_te, method=args.direction_threshold)
            except Exception:
                threshold = 0.5
            y_pred_dir = (y_prob > threshold).astype(int)
            cls_accuracy = float(accuracy_score(y_cls_te, y_pred_dir))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_cls_te, y_pred_dir, average='binary', zero_division=0)
            try:
                cls_auc = float(roc_auc_score(y_cls_te, y_prob))
            except Exception:
                cls_auc = None
            try:
                cls_pr_auc = float(average_precision_score(y_cls_te, y_prob))
            except Exception:
                cls_pr_auc = None

            y_pred_return = model_return.model.predict(X_test)
            y_pred_vol = model_vol.model.predict(X_test)
            ret_rmse = float(np.sqrt(mean_squared_error(y_ret_te,
                                                        y_pred_return)))
            ret_mae = float(mean_absolute_error(y_ret_te, y_pred_return))
            ret_r2 = float(r2_score(y_ret_te, y_pred_return))
            vol_rmse = float(np.sqrt(mean_squared_error(y_vol_te,
                                                        y_pred_vol)))
            vol_mae = float(mean_absolute_error(y_vol_te, y_pred_vol))
            vol_r2 = float(r2_score(y_vol_te, y_pred_vol))

            res = {
                "symbol": symbols_str,
                "timeframe": freq,
                "forward_bars": fb,
                "test_month": test_file["month_str"],
                "train_months": len(train_files),
                "num_features": len(feat_cols),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "cls_accuracy": cls_accuracy,
                "cls_precision": float(precision),
                "cls_recall": float(recall),
                "cls_f1": float(f1),
                "cls_auc": cls_auc,
                "cls_pr_auc": cls_pr_auc,
                "cls_threshold": float(threshold),
                "test_rmse_return": ret_rmse,
                "test_mae_return": ret_mae,
                "test_r2_return": ret_r2,
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

            all_results.append(res)
            print(
                f"      Test return RMSE: {ret_rmse:.6f}, MAE: {ret_mae:.6f}; cls F1: {f1:.3f}"
            )

            # Save models
            base = f"fb{fb}_tf{freq}_{test_file['month_str']}"
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
        avg_cls_f1 = float(results_df["cls_f1"].mean()
                           ) if "cls_f1" in results_df.columns and len(
                               results_df) > 0 else None
        avg_cls_auc = float(results_df["cls_auc"].mean()
                            ) if "cls_auc" in results_df.columns and len(
                                results_df) > 0 else None
        avg_return_r2 = float(results_df["test_r2_return"].mean()
                              ) if "test_r2_return" in results_df.columns and len(
                                  results_df) > 0 else None
        avg_vol_r2 = float(results_df["test_r2_vol"].mean()
                           ) if "test_r2_vol" in results_df.columns and len(
                               results_df) > 0 else None
        summary = {
            "symbol":
            symbols_str,
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
            from ml_trading.pipeline.dimensionality.report_generator import write_rolling_report
            from ml_trading.pipeline.training.generate_summary_report import generate_summary_report
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
            summary_html = generate_summary_report(
                results_dir=combo_dir, output_path=os.path.join(
                    rolling_summary_dir, "summary_report.html"))
            if summary_html:
                print(
                    f"   - summary report (train-style): {summary_html}")
        except Exception as exc:
            print(f"   ⚠️  Failed to generate HTML report: {exc}")

    print("\n✅ Rolling completed. Results saved to:")
    print(f"   {results_dir}")


if __name__ == "__main__":
    main()
