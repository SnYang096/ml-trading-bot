from __future__ import annotations
"""
Rolling training using only baseline SR + compression features.

Usage example:
  python -m ml_trading.pipeline.baseline.rolling_baseline \
    --data-dir /home/yin/trading/ml_trading_bot/data/parquet_data \
    --symbol BTCUSDT \
    --initial-train-months 6 \
    --forward-bars 3 \
    --gpu
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

from ml_trading.data_tools.rolling_data import (
    load_and_process_file, )
from ml_trading.data_tools.baseline_feature_engineering import (
    engineer_baseline_features,
    get_baseline_feature_columns,
    create_binary_labels_baseline,
)
from ml_trading.models.lightgbm_model import LightGBMModel
from sklearn.metrics import accuracy_score


def find_all_available_files(data_dir: str, symbol: str) -> List[Dict]:
    files: List[Dict] = []
    from pathlib import Path
    import re

    data_path = Path(data_dir)
    if not data_path.exists():
        return files

    symbol_mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "BNBUSDT": "BNB-USD",
    }
    file_symbol = symbol_mapping.get(symbol, symbol)

    patterns = [
        f"{symbol}-aggTrades-*.parquet",
        f"{file_symbol}_*.parquet",
        f"{file_symbol}-*.parquet",
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
                    files.append({
                        "path": str(file_path),
                        "year": year,
                        "month": month,
                        "month_str": f"{year}-{month:02d}",
                        "timestamp": pd.Timestamp(year, month, 1),
                    })
                except Exception:
                    continue

    files.sort(key=lambda x: x["timestamp"])
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline rolling training (SR+compression only)")
    parser.add_argument("--data-dir",
                        type=str,
                        default=os.environ.get("DATA_DIR",
                                               "data/parquet_data"))
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--initial-train-months", type=int, default=6)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--forward-bars", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--freq",
        type=str,
        action="append",
        default=["5T"],
        help=
        "Bar timeframe(s), repeat or comma-separate: --freq 5T --freq 15T or --freq 5T,15T"
    )
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
    parser.add_argument("--gpu", action="store_true", default=True)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔄 Baseline Rolling Training")
    print("=" * 80)

    # Normalize lists: freqs and forward bars

    def _parse_list(values):
        out = []
        for v in (values if isinstance(values, list) else [values]):
            if isinstance(v, str) and "," in v:
                out.extend([x.strip() for x in v.split(",") if x.strip()])
            else:
                out.append(v)
        return out

    freqs = _parse_list(args.freq)
    # forward_bars could be single int; allow comma string via env/cli
    fb_raw = os.environ.get("FB_LIST")
    if fb_raw:
        forward_bars_list = [int(x) for x in _parse_list([fb_raw])]
    else:
        forward_bars_list = [args.forward_bars] if not isinstance(
            args.forward_bars, list) else [int(x) for x in args.forward_bars]

    print(
        f"Data dir: {args.data_dir}, Symbol: {args.symbol}, Timeframes: {freqs}, Horizons: {forward_bars_list}"
    )

    files = find_all_available_files(args.data_dir, args.symbol)
    # Optional date range filtering
    if args.start or args.end:

        def _in_range(ym: str) -> bool:
            if args.start and ym < args.start:
                return False
            if args.end and ym > args.end:
                return False
            return True

        files = [f for f in files if _in_range(f["month_str"])]
    if not files or len(files) < args.min_train_months + 1:
        print("❌ Not enough monthly files to run rolling baseline.")
        return

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"baseline_rolling_{args.symbol.lower()}_{ts}"
    results_dir = os.path.join("results", args.output)
    os.makedirs(results_dir, exist_ok=True)

    # Iterate over combinations of freq and forward bars
    for freq in freqs:
        start_idx = args.initial_train_months
        all_results = []
        baseline_engineer = None
        combo_dir = os.path.join(
            results_dir,
            f"fb{forward_bars_list[0] if len(forward_bars_list) == 1 else 'multi'}_tf{freq}"
        ) if (len(freqs) > 1 or len(forward_bars_list) > 1) else results_dir
        os.makedirs(combo_dir, exist_ok=True)

        for i in range(start_idx, len(files)):
            train_files = files[:i]
            test_file = files[i]

            print("\n" + "-" * 80)
            print(
                f"Train: {train_files[0]['month_str']} → {train_files[-1]['month_str']} ({len(train_files)} months)"
            )
            print(f"Test:  {test_file['month_str']}")

            # Load train
            train_parts = []
            for fi in train_files:
                df = load_and_process_file(fi["path"], freq=freq)
                if df is not None and len(df) > 0:
                    train_parts.append(df)
            if not train_parts:
                print("   ⚠️  No training data, skip")
                continue
            train_df = pd.concat(train_parts, axis=0).sort_index()

            # Load test
            test_df = load_and_process_file(test_file["path"], freq=freq)
            if test_df is None or len(test_df) == 0:
                print("   ⚠️  No test data, skip")
                continue

            # Engineer baseline features
            print(
                "   🧪 Engineering baseline features (fit on train, apply to test)..."
            )
            train_df, baseline_engineer = engineer_baseline_features(
                train_df, baseline_engineer, fit=True)
            test_df, _ = engineer_baseline_features(test_df,
                                                    baseline_engineer,
                                                    fit=False)

            # Labels for each horizon; train one model per horizon (binary classification)
            for fb in forward_bars_list:
                train_labeled = create_binary_labels_baseline(
                    train_df, forward_bars=fb).dropna()
                test_labeled = create_binary_labels_baseline(
                    test_df, forward_bars=fb).dropna()

                # Train (binary classification: 1=Long, 0=not Long)
                feat_cols = get_baseline_feature_columns(train_labeled)
                X_train = train_labeled[feat_cols].values
                y_train = train_labeled[
                    "binary_signal"].values  # Use binary_signal for binary classification
                X_test = test_labeled[feat_cols].values

                print(
                    f"   🎯 Training LightGBM (binary, fb={fb}, tf={freq}) (N={len(X_train):,}, F={len(feat_cols)})"
                )

                # Use LightGBMModel.train() which performs TimeSeriesSplit CV (same as train command)
                model = LightGBMModel(model_type="classification",
                                      use_gpu=args.gpu)
                # Override to use binary classification instead of multiclass
                model.params["objective"] = "binary"
                model.params["metric"] = "binary_logloss"
                if "num_class" in model.params:
                    del model.params["num_class"]

                # Use TimeSeriesSplit CV (same as train command)
                n_splits = args.cv_folds if (args.cv_on_rolling
                                             and args.cv_folds > 0) else 5
                model_metrics = model.train(pd.DataFrame(X_train,
                                                         columns=feat_cols),
                                            pd.Series(y_train),
                                            n_splits=n_splits,
                                            use_time_series_cv=True)

                cv_accuracy_mean = model_metrics.get("cv_accuracy", None)
                cv_accuracy_std = model_metrics.get("cv_accuracy_std", None)

                if cv_accuracy_mean is not None:
                    print(
                        f"      CV (folds={n_splits}) accuracy: mean={cv_accuracy_mean:.4f}, std={cv_accuracy_std:.4f}"
                    )

                # Evaluate on test set (same approach as train command - use accuracy)
                y_test = test_labeled["binary_signal"].values
                X_test_df = pd.DataFrame(X_test, columns=feat_cols)
                test_preds_proba = model.model.predict(X_test_df.values)
                test_preds = (test_preds_proba > 0.5).astype(int)
                test_accuracy = accuracy_score(y_test, test_preds)

                res = {
                    "symbol":
                    args.symbol,
                    "timeframe":
                    freq,
                    "forward_bars":
                    fb,
                    "test_month":
                    test_file["month_str"],
                    "train_months":
                    len(train_files),
                    "num_features":
                    len(feat_cols),
                    "train_samples":
                    len(X_train),
                    "test_samples":
                    len(X_test),
                    "cv_folds":
                    n_splits if
                    (args.cv_on_rolling and args.cv_folds > 0) else 0,
                    "cv_accuracy_mean":
                    cv_accuracy_mean,
                    "cv_accuracy_std":
                    cv_accuracy_std,
                    "test_accuracy":
                    test_accuracy,
                }
                all_results.append(res)
                print(f"      Test accuracy: {test_accuracy:.4f}")

                # Save model per month per combo
                model.model.save_model(
                    os.path.join(
                        combo_dir,
                        f"model_fb{fb}_tf{freq}_{test_file['month_str']}.txt")
                )  # model.model is the lgb.Booster object

        # Save summary/report for this combo
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(combo_dir, "monthly_results.csv"),
                          index=False)

        # Extract training time range from files and results
        train_start_date = files[0]["month_str"] if files else None
        test_end_date = files[-1]["month_str"] if files else None
        if not results_df.empty and "test_month" in results_df.columns:
            test_end_date = results_df["test_month"].max(
            ) if not results_df.empty else test_end_date

        # Calculate summary metrics (same as train command - use accuracy)
        avg_cv_accuracy = float(results_df["cv_accuracy_mean"].mean(
        )) if "cv_accuracy_mean" in results_df.columns and len(
            results_df) > 0 else None
        avg_test_accuracy = float(results_df["test_accuracy"].mean(
        )) if "test_accuracy" in results_df.columns and len(
            results_df) > 0 else None

        summary = {
            "symbol": args.symbol,
            "total_months_tested": len(results_df),
            "train_start_date": train_start_date,
            "test_end_date": test_end_date,
            "avg_cv_accuracy": avg_cv_accuracy,
            "avg_test_accuracy": avg_test_accuracy,
            "created_at": datetime.now().isoformat(),
            "feature_engineering": "BaselineFeatureEngineer",
            "configuration": {
                "symbol": args.symbol,
                "data_dir": args.data_dir,
                "initial_train_months": args.initial_train_months,
                "min_train_months": args.min_train_months,
                "forward_bars": forward_bars_list,
                "gpu": args.gpu,
                "timeframe": freq,
                "start": getattr(args, "start", None),
                "end": getattr(args, "end", None),
            },
        }
        with open(os.path.join(combo_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Generate HTML report using existing generator
        try:
            from ml_trading.pipeline.dimensionality.report_generator import write_rolling_report
            report_path = write_rolling_report(
                combo_dir,
                summary_path=os.path.join(combo_dir, "summary.json"),
                results_csv_path=os.path.join(combo_dir,
                                              "monthly_results.csv"),
                report_type="monthly",
            )
            print(f"   - combo report: {report_path}")
        except Exception as exc:
            print(f"   ⚠️  Failed to generate HTML report: {exc}")

    print("\n✅ Baseline rolling completed. Results saved to:")
    print(f"   {results_dir}")


if __name__ == "__main__":
    main()
