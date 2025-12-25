#!/usr/bin/env python3
"""
Run inference with a trained NN path-primitives multi-head MLP (model.pt) on real data.

Outputs a dataframe with prediction columns:
  pred_dir_prob, pred_mfe_atr, pred_mae_atr, pred_t_to_mfe (+ optional pred_persistence)

Example:
  python scripts/predict_path_primitives_mlp.py \
    --config config/nnmultihead/path_primitives_4h_80h_min \
    --symbol BTCUSDT \
    --data-path data/parquet_data \
    --timeframe 240T \
    --model results/nnmultihead/.../model.pt \
    --output results/nnmultihead/.../preds.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402
from src.time_series_model.models.nn.path_primitives_model import (
    MultiHeadPathPrimitivesMLP,
)  # noqa: E402
from src.time_series_model.models.nn.path_primitives_reporting import (
    predict_path_primitives,
)  # noqa: E402
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    determine_feature_columns,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict NN path-primitives multi-head MLP."
    )
    p.add_argument(
        "--config", required=True, help="Config directory containing features.yaml."
    )
    p.add_argument(
        "--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT"
    )
    p.add_argument("--data-path", default="data/parquet_data")
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--model", required=True, help="Path to model.pt produced by nnmultihead train"
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path (.parquet/.csv). If multi-symbol, treat as output directory.",
    )
    p.add_argument("--device", default=None, help="cpu|cuda (default auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config).resolve()
    loader = StrategyConfigLoader(cfg_dir)
    cfg = loader.load()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")

    payload = torch.load(args.model, map_location="cpu")
    if "model" not in payload:
        raise ValueError("Invalid model payload: missing 'model' key")
    model = MultiHeadPathPrimitivesMLP.from_export(payload["model"])

    out_root = Path(args.output)
    multi = len(symbols) > 1
    if multi:
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root.parent.mkdir(parents=True, exist_ok=True)

    feature_loader = StrategyFeatureLoader()
    for sym in symbols:
        df_raw = load_raw_data(
            data_path=args.data_path,
            symbol=sym,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
        )
        if df_raw.empty:
            raise ValueError(f"No raw data loaded for symbol={sym}")
        if "symbol" not in df_raw.columns:
            df_raw["symbol"] = sym

        df_features = run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=cfg.features,
            fit=False,
        )
        if "symbol" not in df_features.columns:
            df_features["symbol"] = sym
        feature_cols = determine_feature_columns(df_features, cfg.features)

        preds = predict_path_primitives(
            model=model,
            df=df_features,
            feature_cols=feature_cols,
            device=args.device,
            fill_nan_value=0.0,
        )
        out = df_features.join(preds)

        if multi:
            # For multi-symbol, always save per-symbol parquet for stability.
            out_path = out_root / f"preds_{sym}.parquet"
            out.to_parquet(out_path, index=True)
        else:
            out_path = out_root
            if out_path.suffix.lower() == ".parquet":
                out.to_parquet(out_path, index=True)
            else:
                out.to_csv(out_path, index=True)
        print("✅ Saved preds to:", out_path)


if __name__ == "__main__":
    main()
