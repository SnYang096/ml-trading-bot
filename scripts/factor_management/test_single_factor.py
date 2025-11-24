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
    if not config_path.exists():
        raise FileNotFoundError(f"Features config not found: {config_path}")

    available_features = load_requested_features(config_path)
    missing = [f for f in args.factors if f not in available_features]
    if missing:
        raise ValueError(
            f"Requested factors not defined in config: {missing}. "
            f"Available: {available_features}"
        )

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )

    feature_loader = StrategyFeatureLoader()
    df_features = feature_loader.load_features_from_requested(
        df_raw,
        requested_features=args.factors,
        fit=True,
    )

    summary = df_features[args.factors].describe().transpose()
    print("\n📊 Factor summary statistics:")
    print(summary)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"factor_test_{args.symbol}.csv"
        df_features[args.factors].to_csv(output_file)
        print(f"   💾 Saved factor values to {output_file}")


if __name__ == "__main__":
    main()
