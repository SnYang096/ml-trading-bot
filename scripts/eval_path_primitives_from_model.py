#!/usr/bin/env python3
"""
Evaluate an existing nnmultihead (path-primitives) model.pt and write standard artifacts.

Why this exists:
- Sometimes a training run is interrupted after saving model.pt but before writing
  meta/metrics/report artifacts (e.g., killed during evaluation/reporting).
- This script regenerates artifacts from model.pt + FeatureStore, in a controlled,
  memory-safe way (supports per-symbol sampling).

Outputs (via save_train_artifacts):
  - meta.json
  - metrics.json
  - metrics_summary.md
  - pred_sample.csv
  - report.html
  - model_path.txt
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

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec  # noqa: E402
from src.time_series_model.models.nn.path_primitives_artifact import (  # noqa: E402
    PathPrimitivesModelArtifact,
)
from src.time_series_model.models.nn.path_primitives_dataset import (  # noqa: E402
    DatasetConfig,
)
from src.time_series_model.models.nn.path_primitives_labels import (  # noqa: E402
    PathPrimitivesLabelConfig,
)
from src.time_series_model.models.nn.path_primitives_reporting import (  # noqa: E402
    evaluate_model_on_df,
    save_train_artifacts,
)
from src.time_series_model.rule.router_3action import Rule3ActionConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.pt")
    ap.add_argument(
        "--config",
        required=False,
        default=None,
        help="Optional config dir (for FeatureContract).",
    )
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--features-store-root", default="feature_store")
    ap.add_argument("--features-store-layer", required=True)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for artifacts (default: model parent dir).",
    )
    ap.add_argument(
        "--max-rows-per-symbol",
        type=int,
        default=6000,
        help="Memory-safety: evaluate on last N rows per symbol (0 disables sampling).",
    )
    # Optional: Router thresholds for report-aligned evaluation
    ap.add_argument("--router-mfe-min", type=float, default=None)
    ap.add_argument("--router-eff-min", type=float, default=None)
    ap.add_argument("--router-dir-conf-trend-min", type=float, default=None)
    return ap.parse_args()


def _read_features(
    *,
    store_root: str,
    layer: str,
    symbols: list[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    store = FeatureStore(str(store_root))
    parts = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(layer), symbol=str(sym), timeframe=str(timeframe)
        )
        df = store.read_range(spec, start=start, end=end)
        if df.empty:
            raise ValueError(f"Empty FeatureStore read for symbol={sym}, layer={layer}")
        if "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = sym
        parts.append(df)
    return pd.concat(parts, axis=0, ignore_index=False)


def main() -> int:
    args = _parse_args()

    model_path = Path(args.model).resolve()
    cfg_dir = Path(args.config).resolve() if args.config else None
    out_dir = (
        Path(args.out_dir).resolve() if args.out_dir else model_path.parent.resolve()
    )

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    artifact = PathPrimitivesModelArtifact.load(
        model_path=model_path,
        config_dir=cfg_dir,
    )
    if not artifact.feature_cols:
        raise ValueError("model.pt meta missing feature_cols; cannot evaluate safely.")

    label_cfg = PathPrimitivesLabelConfig(**(artifact.meta.get("label_cfg") or {}))
    dataset_cfg = DatasetConfig(**(artifact.meta.get("dataset_cfg") or {}))

    start = (
        pd.Timestamp(args.start_date) if args.start_date else pd.Timestamp("1970-01-01")
    )
    end = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp("2100-01-01")

    df = _read_features(
        store_root=str(args.features_store_root),
        layer=str(args.features_store_layer),
        symbols=symbols,
        timeframe=str(args.timeframe),
        start=start,
        end=end,
    )

    # Optional: Router thresholds for Router-aligned eval inside evaluate_model_on_df.
    # (We pass via a private DataFrame attribute to avoid changing many call sites.)
    router_eval_cfg: Rule3ActionConfig | None = None
    if (
        args.router_mfe_min is not None
        or args.router_eff_min is not None
        or args.router_dir_conf_trend_min is not None
    ):
        rcfg = Rule3ActionConfig()
        if args.router_mfe_min is not None:
            rcfg = Rule3ActionConfig(
                **{**rcfg.__dict__, "mfe_min": float(args.router_mfe_min)}
            )
        if args.router_eff_min is not None:
            rcfg = Rule3ActionConfig(
                **{**rcfg.__dict__, "eff_min": float(args.router_eff_min)}
            )
        if args.router_dir_conf_trend_min is not None:
            rcfg = Rule3ActionConfig(
                **{
                    **rcfg.__dict__,
                    "dir_conf_trend_min": float(args.router_dir_conf_trend_min),
                }
            )
        router_eval_cfg = rcfg
        try:
            setattr(df, "_router_eval_cfg", router_eval_cfg)
        except Exception:
            pass

    # Memory safety: use the last N rows per symbol (time-ordered)
    max_n = int(args.max_rows_per_symbol)
    if max_n and max_n > 0 and "symbol" in df.columns:
        df = (
            df.reset_index(drop=False)
            .sort_values(
                ["symbol", "timestamp"] if "timestamp" in df.columns else ["symbol"]
            )
            .groupby("symbol", sort=False)
            .tail(max_n)
            .reset_index(drop=True)
        )
        # reset_index/sort_values creates a new DataFrame; re-attach router cfg if needed
        if router_eval_cfg is not None:
            try:
                setattr(df, "_router_eval_cfg", router_eval_cfg)
            except Exception:
                pass

    metrics, df_eval, extra = evaluate_model_on_df(
        model=artifact.model,
        df_features=df,
        feature_cols=list(artifact.feature_cols),
        label_cfg=label_cfg,
        dataset_cfg=dataset_cfg,
        group_col="symbol" if len(symbols) > 1 else None,
        block_cols_by_name=artifact.block_cols_by_name,
        append_block_mask=artifact.append_block_mask,
        feature_scaler=artifact.feature_scaler,
    )

    meta = dict(artifact.meta or {})
    meta["eval_note"] = {
        "script": "scripts/eval_path_primitives_from_model.py",
        "symbols": symbols,
        "timeframe": str(args.timeframe),
        "start": str(start),
        "end": str(end),
        "max_rows_per_symbol": int(args.max_rows_per_symbol),
    }
    # Optional: attach router thresholds (so report can run Router-aligned threshold metrics)
    if (
        args.router_mfe_min is not None
        or args.router_eff_min is not None
        or args.router_dir_conf_trend_min is not None
    ):
        meta["router_thresholds_for_report"] = {
            "mfe_min": args.router_mfe_min,
            "eff_min": args.router_eff_min,
            "dir_conf_trend_min": args.router_dir_conf_trend_min,
        }
    if isinstance(extra, dict) and extra.get("rolling_ic") is not None:
        meta["rolling_ic"] = extra.get("rolling_ic")
    # Optional: attach threshold eval tables for HTML report
    if isinstance(extra, dict) and extra.get("threshold_eval") is not None:
        meta["threshold_eval"] = extra.get("threshold_eval")
    if isinstance(extra, dict) and extra.get("threshold_eval_by_symbol") is not None:
        meta["threshold_eval_by_symbol"] = extra.get("threshold_eval_by_symbol")

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
        model_path=str(model_path),
        meta=meta,
        metrics=metrics,
        df_pred_sample=sample_df[
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

    (out_dir / "eval_from_model.done.json").write_text(
        json.dumps({"ok": True, "model": str(model_path)}, indent=2),
        encoding="utf-8",
    )
    print("✅ Wrote artifacts to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
