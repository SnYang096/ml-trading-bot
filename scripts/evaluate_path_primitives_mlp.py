#!/usr/bin/env python3
"""
Evaluate a trained NN path-primitives multi-head MLP (model.pt) on real data.

It computes labels on the fly (path primitives), runs predictions, and saves:
  - meta.json / metrics.json / pred_sample.csv / model_path.txt / report.html

Example:
  python scripts/evaluate_path_primitives_mlp.py \
    --config config/nnmultihead/path_primitives_4h_80h_min \
    --symbol BTCUSDT \
    --data-path data/parquet_data \
    --timeframe 240T \
    --model results/nnmultihead/.../model.pt \
    --horizon-hours 80 --bar-hours 4 \
    --output-dir results/nnmultihead_eval
"""

from __future__ import annotations

import argparse
import json
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
from src.time_series_model.models.nn.path_primitives_labels import (
    PathPrimitivesLabelConfig,
)  # noqa: E402
from src.time_series_model.models.nn.path_primitives_model import (
    MultiHeadPathPrimitivesMLP,
)  # noqa: E402
from src.time_series_model.models.nn.path_primitives_reporting import (
    evaluate_model_on_df,
    save_train_artifacts,
)  # noqa: E402
from src.time_series_model.models.nn.path_primitives_artifact import (  # noqa: E402
    PathPrimitivesModelArtifact,
)
from src.time_series_model.models.nn.feature_contract import (  # noqa: E402
    load_feature_contract,
    validate_minimal_required_cols,
)
from src.time_series_model.tasks import make_primitives_task_id  # noqa: E402
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    determine_feature_columns,
)  # noqa: E402
from src.time_series_model.models.nn.path_primitives_dataset import (  # noqa: E402
    resolve_block_cols_by_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate NN path-primitives multi-head MLP."
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

    # Horizon (needed to build labels)
    p.add_argument("--horizon-hours", type=float, default=80.0)
    p.add_argument("--bar-hours", type=float, default=4.0)

    p.add_argument("--device", default=None, help="cpu|cuda (default auto)")
    p.add_argument(
        "--output-dir", required=True, help="Output directory for eval artifacts"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config).resolve()
    # TaskSpec-only enforcement (no legacy config mode).
    if not (cfg_dir / "derived_from_task_spec.json").exists():
        raise SystemExit(
            "ERROR: nnmultihead is TaskSpec-only.\n"
            f"Config dir is not TaskSpec-derived: {cfg_dir}\n"
            "Please run via `mlbot nnmultihead eval --task-spec ...` (recommended),\n"
            "or materialize first via `mlbot nnmultihead materialize-config-from-task-spec --task-spec ...`."
        )
    cfg_loader = StrategyConfigLoader(cfg_dir)
    cfg = cfg_loader.load()

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
        if "symbol" not in df_raw.columns:
            df_raw["symbol"] = sym
        df_features_sym = run_feature_pipeline(
            df_raw, feature_loader=feature_loader, pipeline_cfg=cfg.features, fit=False
        )
        if "symbol" not in df_features_sym.columns:
            df_features_sym["symbol"] = sym
        feats_all.append(df_features_sym)

    df_features = pd.concat(feats_all, axis=0, ignore_index=False)
    feature_cols = determine_feature_columns(df_features, cfg.features)

    # nnmultihead-only: feature contract (does not affect tree pipeline)
    contract = load_feature_contract(cfg_dir)
    if contract is not None:
        validate_minimal_required_cols(
            available_columns=df_features.columns.tolist(), contract=contract
        )

    artifact = PathPrimitivesModelArtifact.load(
        model_path=args.model, config_dir=cfg_dir
    )
    model = artifact.model
    feature_scaler = artifact.feature_scaler
    model_feature_cols = artifact.feature_cols

    horizon_bars = int(round(float(args.horizon_hours) / float(args.bar_hours)))
    if horizon_bars <= 0:
        raise ValueError(f"Invalid horizon_bars computed: {horizon_bars}")

    label_cfg = PathPrimitivesLabelConfig(
        horizon_bars=horizon_bars,
        entry_offset=1,
        entry_price_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        atr_col="atr",
    )

    # Use the same feature columns + optional block mask settings as training/prediction.
    if model_feature_cols is not None:
        feature_cols_eval = list(model_feature_cols)
    else:
        feature_cols_eval = feature_cols

    block_cols_by_name = artifact.block_cols_by_name
    append_block_mask = artifact.append_block_mask

    metrics, df_eval, extra = evaluate_model_on_df(
        model=model,
        df_features=df_features,
        feature_cols=feature_cols_eval,
        label_cfg=label_cfg,
        group_col="symbol" if len(symbols) > 1 else None,
        feature_scaler=feature_scaler,
        block_cols_by_name=block_cols_by_name,
        append_block_mask=append_block_mask,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": getattr(cfg, "name", str(cfg_dir.name)),
        "symbols": symbols,
        "timeframe": args.timeframe,
        "config_dir": str(cfg_dir),
        "feature_cols": feature_cols,
        "label_cfg": {
            "horizon_bars": horizon_bars,
            "horizon_hours": float(args.horizon_hours),
            "bar_hours": float(args.bar_hours),
        },
    }
    meta["task_id"] = make_primitives_task_id(
        config_dir=cfg_dir,
        timeframe=str(args.timeframe),
        horizon_hours=float(args.horizon_hours),
        bar_hours=float(args.bar_hours),
        version="v1",
    )
    meta["task_spec_hint"] = (
        "docs/architecture/task_specs/primitives_path_primitives_4h_80h_v1.yaml"
    )
    if isinstance(extra, dict) and extra.get("rolling_ic") is not None:
        meta["rolling_ic"] = extra.get("rolling_ic")
    if contract is not None:
        meta["feature_contract"] = contract.to_dict()

    # Prefer a sample where labels exist (avoid tail horizon NaNs).
    sample_df = df_eval
    if "mfe_valid" in sample_df.columns:
        sample_df = sample_df[
            pd.to_numeric(sample_df["mfe_valid"], errors="coerce").fillna(0.0) > 0.5
        ]
    if "dir_y" in sample_df.columns:
        sample_df = sample_df[
            pd.to_numeric(sample_df["dir_y"], errors="coerce").notna()
        ]
    sample_df = sample_df.tail(200) if len(sample_df) else df_eval.tail(200)

    save_train_artifacts(
        out_dir=str(out_dir),
        model_path=str(args.model),
        meta=meta,
        metrics=metrics,
        df_pred_sample=sample_df[
            (
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
                if all(
                    c in df_eval.columns
                    for c in ["dir_y", "mfe_atr", "mae_atr", "t_to_mfe", "mfe_valid"]
                )
                else df_eval.columns[: min(30, len(df_eval.columns))].tolist()
            )
        ],
    )

    print("✅ Eval complete")
    print(f"   output_dir: {out_dir}")
    print("   metrics:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
