"""Config-driven factor inspection tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test specific factors via config.")
    parser.add_argument("--factors", nargs="+", required=True, help="Factor names")
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="240T")
    parser.add_argument(
        "--features-config",
        type=str,
        default="config/tests/factor_test/features.yaml",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_requested_features(config_path: Path) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    pipeline = data.get("feature_pipeline", {})
    return pipeline.get("requested_features", []) or []


def main() -> None:
    args = parse_args()
    config_path = Path(args.features_config)

    # Load config if it exists, but don't require factors to be in it
    # The config is mainly used for feature pipeline settings
    config_features = []
    if config_path.exists():
        config_features = load_requested_features(config_path)
        if config_features:
            print(
                f"ℹ️  Config file defines {len(config_features)} features: {config_features[:5]}{'...' if len(config_features) > 5 else ''}"
            )
    else:
        print(
            f"⚠️  Config file not found: {config_path}, proceeding without config constraints"
        )

    # Use command-line factors directly - the feature loader will compute them
    # regardless of whether they're in the config
    requested_factors = args.factors
    print(f"🧪 Testing {len(requested_factors)} factors: {requested_factors}")

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )

    feature_loader = StrategyFeatureLoader()
    try:
        df_features = feature_loader.load_features_from_requested(
            df_raw,
            requested_features=requested_factors,
            fit=True,
        )
    except Exception as e:
        print(f"❌ Error computing features: {e}")
        print(
            f"   Make sure the factor names are correct and the feature modules are available."
        )
        raise

    # Check which factors were actually computed
    computed_factors = [f for f in requested_factors if f in df_features.columns]
    missing_factors = [f for f in requested_factors if f not in df_features.columns]

    if missing_factors:
        print(
            f"⚠️  Warning: {len(missing_factors)} factors were not computed: {missing_factors}"
        )
        print(
            f"   Available columns: {sorted(df_features.columns.tolist())[:20]}{'...' if len(df_features.columns) > 20 else ''}"
        )

    if not computed_factors:
        raise ValueError(
            "No factors were successfully computed. Check factor names and feature configuration."
        )

    summary = df_features[computed_factors].describe().transpose()
    print("\n📊 Factor summary statistics:")
    print(summary)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"factor_test_{args.symbol}.csv"
        df_features[computed_factors].to_csv(output_file)
        print(f"   💾 Saved factor values to {output_file}")


if __name__ == "__main__":
    main()
