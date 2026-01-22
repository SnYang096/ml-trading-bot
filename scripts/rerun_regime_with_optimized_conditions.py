#!/usr/bin/env python3
"""
重新运行regime分类，使用优化后的MEAN_REGIME条件。

这个脚本会：
1. 从原始logs读取数据
2. 使用优化后的PhysicsRegimeConfig重新运行regime分类
3. 输出包含物理特征和优化后regime的parquet文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.time_series_model.rule.regime import (
    PhysicsRegimeConfig,
    classify_regime,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-run regime classification with optimized MEAN_REGIME conditions."
    )
    p.add_argument(
        "--logs",
        required=True,
        help="Input logs file (must contain symbol, timestamp, and required features)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output parquet file with optimized regime classification and physical features",
    )
    p.add_argument(
        "--feature-store-root",
        default=None,
        help="Optional FeatureStore root (if logs lack required features)",
    )
    p.add_argument(
        "--layer",
        default="tier0",
        help="FeatureStore layer (default: tier0)",
    )
    p.add_argument(
        "--timeframe",
        default="240T",
        help="Timeframe (default: 240T)",
    )
    args = p.parse_args()

    # Load logs
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"Error: Logs file not found: {logs_path}", file=sys.stderr)
        return 1

    print(f"Loading logs from: {logs_path}", file=sys.stderr)
    if logs_path.suffix.lower() == ".parquet":
        logs_df = pd.read_parquet(logs_path)
    else:
        logs_df = pd.read_csv(logs_path)

    # Ensure timestamp column
    if logs_df.index.name == "timestamp":
        logs_df = logs_df.reset_index()
    if "timestamp" not in logs_df.columns:
        print("Error: timestamp column not found", file=sys.stderr)
        return 1

    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")

    # Convert head_dir_score to pred_dir_prob if needed
    if "pred_dir_prob" not in logs_df.columns and "head_dir_score" in logs_df.columns:
        import numpy as np

        # head_dir_score is in logit space, convert to probability using sigmoid
        logs_df["pred_dir_prob"] = 1.0 / (1.0 + np.exp(-logs_df["head_dir_score"]))
        print(f"Converted head_dir_score to pred_dir_prob", file=sys.stderr)

    # Load features from FeatureStore if needed
    required_features = [
        "atr",
        "atr_percentile",
        "high",
        "low",
        "close",
        "pred_dir_prob",
    ]
    missing_features = [f for f in required_features if f not in logs_df.columns]

    if missing_features and args.feature_store_root:
        print(
            f"Loading missing features from FeatureStore: {missing_features}",
            file=sys.stderr,
        )
        try:
            from src.feature_store import FeatureStore, FeatureStoreSpec

            store = FeatureStore(str(args.feature_store_root))

            # Group by symbol and load features
            all_features = []
            for symbol in logs_df["symbol"].unique():
                spec = FeatureStoreSpec(
                    layer=args.layer,
                    symbol=str(symbol),
                    timeframe=args.timeframe,
                )

                symbol_logs = logs_df[logs_df["symbol"] == symbol]
                start_date = pd.Timestamp(symbol_logs["timestamp"].min())
                end_date = pd.Timestamp(symbol_logs["timestamp"].max())

                try:
                    feat_df = store.read_range(spec, start=start_date, end=end_date)
                    if not feat_df.empty:
                        # Ensure timestamp is datetime
                        if "timestamp" in feat_df.columns:
                            feat_df["timestamp"] = pd.to_datetime(
                                feat_df["timestamp"], errors="coerce"
                            )
                        elif feat_df.index.name == "timestamp":
                            feat_df = feat_df.reset_index()
                            feat_df["timestamp"] = pd.to_datetime(
                                feat_df["timestamp"], errors="coerce"
                            )
                        else:
                            print(
                                f"Warning: No timestamp column in FeatureStore data for {symbol}",
                                file=sys.stderr,
                            )
                            continue

                        feat_df["symbol"] = str(symbol)
                        all_features.append(feat_df)
                except Exception as e:
                    print(
                        f"Warning: Failed to load features for {symbol}: {e}",
                        file=sys.stderr,
                    )
                    import traceback

                    traceback.print_exc()

            if all_features:
                features_df = pd.concat(all_features, ignore_index=True)
                features_df["timestamp"] = pd.to_datetime(
                    features_df["timestamp"], errors="coerce"
                )

                # Merge with logs
                logs_df = logs_df.merge(
                    features_df[
                        ["symbol", "timestamp"]
                        + [f for f in missing_features if f in features_df.columns]
                    ],
                    on=["symbol", "timestamp"],
                    how="left",
                    suffixes=("", "_fs"),
                )

                # Prefer FeatureStore values
                for feat in missing_features:
                    fs_col = f"{feat}_fs"
                    if fs_col in logs_df.columns:
                        logs_df[feat] = logs_df[fs_col].fillna(logs_df.get(feat))
                        logs_df = logs_df.drop(columns=[fs_col])
        except Exception as e:
            print(f"Warning: Failed to load from FeatureStore: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    # Check if we have all required features
    still_missing = [f for f in required_features if f not in logs_df.columns]
    if still_missing:
        print(f"Error: Missing required features: {still_missing}", file=sys.stderr)
        print(f"Available columns: {list(logs_df.columns)}", file=sys.stderr)
        return 1

    # Use optimized PhysicsRegimeConfig
    # The config is already optimized in regime.py with:
    # - mean_deviation_z_abs_min_pct: 0.6 (relaxed from 0.85)
    # - mean_path_length_min_pct: 0.5 (relaxed from 0.7)
    # - mean_atr_percentile_min: 0.5 (relaxed from 0.8)
    # - mean_path_efficiency_max_pct: 0.4 (new)
    # - mean_price_dir_consistency_max_pct: 0.5 (new)
    # - mean_jump_risk_max_pct: 0.3 (new)
    cfg = PhysicsRegimeConfig()

    print("Running regime classification with optimized conditions...", file=sys.stderr)

    # Group by symbol and classify
    all_results = []
    for symbol in logs_df["symbol"].unique():
        symbol_logs = logs_df[logs_df["symbol"] == symbol].copy()
        symbol_logs = symbol_logs.sort_values("timestamp")

        # Classify regime
        regime_df = classify_regime(symbol_logs, cfg=cfg)

        # Merge back
        result_df = symbol_logs[["symbol", "timestamp"]].copy()
        result_df = result_df.join(regime_df, how="left")

        # Also keep original columns
        for col in symbol_logs.columns:
            if col not in result_df.columns:
                result_df[col] = symbol_logs[col].values

        all_results.append(result_df)

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df.sort_values(["symbol", "timestamp"])

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_path, index=False)

    # Print statistics
    print(
        f"\n✅ Saved optimized regime classification to: {output_path}", file=sys.stderr
    )
    print(f"\nRegime distribution:", file=sys.stderr)
    print(combined_df["regime"].value_counts().to_string(), file=sys.stderr)

    # Check physical features
    physical_features = [
        "path_efficiency_pct",
        "price_dir_consistency_pct",
        "deviation_z_abs_pct",
    ]
    print(f"\nPhysical features coverage:", file=sys.stderr)
    for feat in physical_features:
        if feat in combined_df.columns:
            non_null = combined_df[feat].notna().sum()
            print(
                f"  {feat}: {non_null}/{len(combined_df)} ({non_null/len(combined_df)*100:.1f}%)",
                file=sys.stderr,
            )
        else:
            print(f"  {feat}: NOT FOUND", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
