#!/usr/bin/env python3
"""
Train the NN path-primitives multi-head MLP on top of an existing strategy feature pipeline.

This is intentionally minimal and production-safe:
- It reuses StrategyConfigLoader + StrategyFeatureLoader + run_feature_pipeline
- It does NOT change existing tree training pipeline
- It saves model (.pt) + meta/metrics artifacts for evaluation

Example:
  python scripts/train_path_primitives_mlp.py \
    --config config/strategies/sr_reversal_long \
    --symbol BTCUSDT \
    --data-path data/parquet_data \
    --timeframe 240T \
    --horizon-hours 80 \
    --output-dir results/nn_path_primitives
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402
from src.time_series_model.models.nn.path_primitives_labels import (  # noqa: E402
    PathPrimitivesLabelConfig,
)
from src.time_series_model.models.nn.path_primitives_trainer import (  # noqa: E402
    TrainConfig,
    train_path_primitives_mlp,
)
from src.time_series_model.models.nn.path_primitives_reporting import (  # noqa: E402
    evaluate_model_on_df,
    save_train_artifacts,
)
from scripts.train_strategy_pipeline import (  # noqa: E402
    run_feature_pipeline,
    determine_feature_columns,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NN path-primitives multi-head MLP.")
    p.add_argument(
        "--config",
        required=True,
        help="Config directory containing features.yaml (nnmultihead or strategy config).",
    )
    p.add_argument(
        "--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT"
    )
    p.add_argument("--data-path", default="data/parquet_data")
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)

    # Horizon
    p.add_argument(
        "--horizon-hours",
        type=float,
        default=80.0,
        help="Future horizon in hours (e.g. 80H)",
    )
    p.add_argument(
        "--bar-hours",
        type=float,
        default=4.0,
        help="Bar duration in hours (4H bars => 4)",
    )

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default=None, help="cpu|cuda (default auto)")

    # Output
    p.add_argument("--output-dir", default="results/nn_path_primitives")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config).resolve()
    loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = loader.load()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")

    feature_loader = StrategyFeatureLoader()
    feats_all = []
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
        # Ensure symbol column exists for grouping
        if "symbol" not in df_raw.columns:
            df_raw["symbol"] = sym
        df_features_sym = run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_cfg.features,
            fit=True,
        )
        # Keep symbol column after pipeline
        if "symbol" not in df_features_sym.columns:
            df_features_sym["symbol"] = sym
        feats_all.append(df_features_sym)

    df_features = pd.concat(feats_all, axis=0, ignore_index=False)
    feature_cols = determine_feature_columns(df_features, strategy_cfg.features)

    # Horizon conversion
    horizon_bars = int(round(float(args.horizon_hours) / float(args.bar_hours)))
    if horizon_bars <= 0:
        raise ValueError(f"Invalid horizon_bars computed: {horizon_bars}")

    # Train
    label_cfg = PathPrimitivesLabelConfig(
        horizon_bars=horizon_bars,
        entry_offset=1,
        entry_price_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        atr_col="atr",
    )
    train_cfg = TrainConfig(
        label_cfg=label_cfg,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden=int(args.hidden),
        depth=int(args.depth),
        dropout=float(args.dropout),
        device=args.device,
    )

    sym_tag = "multi" if len(symbols) > 1 else symbols[0]
    out_dir = Path(args.output_dir) / f"{strategy_cfg.name}_{sym_tag}_{args.timeframe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(out_dir / "model.pt")

    model, meta = train_path_primitives_mlp(
        df_features,
        feature_cols=feature_cols,
        cfg=train_cfg,
        save_path=model_path,
        group_col="symbol" if len(symbols) > 1 else None,
    )

    # Evaluate on the same df (phase-1 sanity). For true OOS, use rolling_train integration later.
    metrics, df_eval = evaluate_model_on_df(
        model=model,
        df_features=df_features,
        feature_cols=feature_cols,
        label_cfg=label_cfg,
        group_col="symbol" if len(symbols) > 1 else None,
    )

    # Save artifacts
    save_train_artifacts(
        out_dir=str(out_dir),
        model_path=model_path,
        meta=meta,
        metrics=metrics,
        df_pred_sample=df_eval.tail(200)[
            [
                "pred_dir_prob",
                "pred_mfe_atr",
                "pred_mae_atr",
                "pred_t_to_mfe",
                "dir_y",
                "mfe_atr",
                "mae_atr",
                "t_to_mfe",
                "mfe_valid",
            ]
        ],
    )

    print("✅ Training complete")
    print(f"   output_dir: {out_dir}")
    print(f"   model_path: {model_path}")
    print("   metrics:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
