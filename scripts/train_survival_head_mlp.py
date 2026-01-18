#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.diagnostics.survival_head_mlp import (  # noqa: E402
    SurvivalHeadTrainConfig,
    save_survival_head_artifacts,
    train_survival_head,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Survival Head (tiny MLP) from extinction labels."
    )
    p.add_argument(
        "--logs", required=True, help="logs_3action.parquet (from build-logs/run-e2e)"
    )
    p.add_argument(
        "--labels",
        required=True,
        help="labels.parquet (from extinction-replay-3action)",
    )
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--config",
        default="config/ood/survival_head_mlp.yaml",
        help="Config YAML for survival head training",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_obj = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    data = cfg_obj.get("data") or {}
    split = cfg_obj.get("split") or {}
    tr = cfg_obj.get("train") or {}
    loss = cfg_obj.get("loss") or {}
    rep = cfg_obj.get("report") or {}

    cfg = SurvivalHeadTrainConfig(
        symbol_col=str(data.get("symbol_col", "symbol")),
        timestamp_col=str(data.get("timestamp_col", "timestamp")),
        label_col=str(data.get("label_col", "y_surv")),
        feature_cols=tuple([str(x) for x in (data.get("feature_cols") or [])]),
        include_mode_onehot=bool(data.get("include_mode_onehot", True)),
        train_ratio=float(split.get("train_ratio", 0.7)),
        val_ratio_within_train=float(split.get("val_ratio_within_train", 0.15)),
        seed=int(tr.get("seed", 0)),
        device=str(tr.get("device", "cpu")),
        epochs=int(tr.get("epochs", 5)),
        batch_size=int(tr.get("batch_size", 512)),
        lr=float(tr.get("lr", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        hidden=int(tr.get("hidden", 128)),
        depth=int(tr.get("depth", 2)),
        dropout=float(tr.get("dropout", 0.1)),
        pos_weight=float(loss.get("pos_weight", 1.0)),
        n_calibration_bins=int(rep.get("n_calibration_bins", 12)),
    )

    df_logs = pd.read_parquet(args.logs)
    df_labels = pd.read_parquet(args.labels)

    metrics, preds_df, curves, roc_png, pr_png, cal_png = train_survival_head(
        df_logs, df_labels, cfg=cfg
    )
    save_survival_head_artifacts(
        out_dir=out_dir,
        metrics=metrics,
        preds_df=preds_df,
        curves=curves,
        roc_png=roc_png,
        pr_png=pr_png,
        cal_png=cal_png,
    )
    print(
        f"[ok] wrote: {out_dir}/model.pt, survival_preds.parquet, metrics.json, curves.json, report.html"
    )


if __name__ == "__main__":
    main()
