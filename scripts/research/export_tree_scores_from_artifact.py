#!/usr/bin/env python3
"""Generate event score injection parquet from a frozen tree artifact over a date range."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.export_tree_scores_for_event_backtest import (
    export_scores,
)  # noqa: E402
from scripts.research.predict_tree_from_artifact import (  # noqa: E402
    predict_from_artifact,
    validate_score_distribution,
)

DEFAULT_GATE_COLS = [
    "atr",
    "trend_confidence",
    "bpc_semantic_chop_ts_q",
    "me_accel_5k",
    "vol_accel",
    "macro_tp_vwap_1200_position",
]


def export_from_artifact(
    *,
    artifact_dir: Path,
    symbols: list[str],
    output: Path,
    start_date: str,
    end_date: str,
    data_path: str = "data/parquet_data",
    timeframe: str = "120T",
    feature_store_layer: str = "features",
    config_dir: Path | None = None,
    extra_cols: list[str] | None = None,
    validate: bool = True,
    save_predictions: Path | None = None,
    include_gate_features: bool = True,
    gate_feature_names: list[str] | None = None,
    validate_short_entry: float | None = None,
    validate_long_entry: float | None = None,
) -> Path:
    df = predict_from_artifact(
        artifact_dir=artifact_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_path=data_path,
        timeframe=timeframe,
        feature_store_layer=feature_store_layer,
        config_dir=config_dir,
        include_gate_features=include_gate_features,
        gate_feature_names=gate_feature_names,
    )
    # Always compute the distribution so a degenerate artifact cannot be hidden
    # behind --no-validate; write a sidecar + a DEGENERATE marker when applicable.
    short_thr = (
        float(validate_short_entry) if validate_short_entry is not None else -0.0074
    )
    scores = pd.to_numeric(df["pred"], errors="coerce")
    dist = {
        "n": int(scores.notna().sum()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "frac_le_short_entry": float((scores <= short_thr).mean()),
        "short_entry": short_thr,
    }
    degenerate = dist["frac_le_short_entry"] < 0.001 or dist["min"] == dist["max"]
    dist["degenerate"] = bool(degenerate)
    print(json.dumps({"score_distribution": dist}, indent=2))
    sidecar = output.with_suffix(".score_distribution.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(dist, indent=2), encoding="utf-8")
    if degenerate:
        print(
            "⚠️  DEGENERATE score distribution — artifact likely unusable as ranker "
            f"(min={dist['min']:.4f}, max={dist['max']:.4f}, "
            f"frac<= {short_thr}={dist['frac_le_short_entry']:.4f}). "
            "Marker written; do NOT promote."
        )
        output.with_suffix(".DEGENERATE").write_text(
            json.dumps(dist, indent=2), encoding="utf-8"
        )
    if validate and degenerate:
        raise ValueError(
            "degenerate score distribution; pass --no-validate only for diagnostics"
        )

    tmp = output.with_suffix(".predictions.tmp.parquet")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, index=False)
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_predictions, index=False)
        print(f"Wrote full predictions {save_predictions} rows={len(df)}")
    avail_extra = [c for c in (extra_cols or DEFAULT_GATE_COLS) if c in df.columns]
    missing_extra = [
        c for c in (extra_cols or DEFAULT_GATE_COLS) if c not in df.columns
    ]
    if missing_extra:
        print(f"warning: gate cols not in pipeline output: {missing_extra}")
    if include_gate_features and missing_extra:
        raise ValueError(
            "gate feature parity failed; missing columns in predictions: "
            f"{missing_extra}"
        )
    return export_scores(
        tmp,
        output,
        symbols=symbols,
        split=None,
        score_col="pred",
        start_date=start_date,
        end_date=end_date,
        extra_cols=avail_extra or None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--config", default="config/strategies/tree_strategies/fast_scalp")
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--timeframe", default="120T")
    ap.add_argument("--extra-cols", default=None)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument(
        "--no-gate-features",
        action="store_true",
        help="Skip supplementing pipeline with adverse-gate input columns",
    )
    ap.add_argument(
        "--validate-short-entry",
        default=None,
        help="Override short_entry for score distribution sanity check",
    )
    ap.add_argument(
        "--feature-store-layer",
        default="features_tree_core_120T_c005db49f7",
    )
    ap.add_argument(
        "--save-predictions",
        default=None,
        help="Also write full feature+pred parquet for gate training",
    )
    args = ap.parse_args()

    artifact = Path(args.artifact_dir)
    if not artifact.is_absolute():
        artifact = (PROJECT_ROOT / artifact).resolve()
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = (PROJECT_ROOT / cfg).resolve()
    out = Path(args.output)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    extra = (
        [c.strip() for c in args.extra_cols.split(",") if c.strip()]
        if args.extra_cols
        else None
    )
    save_preds = Path(args.save_predictions) if args.save_predictions else None
    if save_preds and not save_preds.is_absolute():
        save_preds = (PROJECT_ROOT / save_preds).resolve()
    export_from_artifact(
        artifact_dir=artifact,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        output=out,
        start_date=args.start_date,
        end_date=args.end_date,
        data_path=args.data_path,
        timeframe=args.timeframe,
        feature_store_layer=str(args.feature_store_layer),
        config_dir=cfg,
        extra_cols=extra,
        validate=not args.no_validate,
        save_predictions=save_preds,
        include_gate_features=not args.no_gate_features,
        validate_short_entry=(
            float(args.validate_short_entry)
            if args.validate_short_entry is not None
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
