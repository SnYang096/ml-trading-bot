#!/usr/bin/env python3
"""
Fix historical nnmultihead train/eval artifacts by recomputing metrics with the *current* evaluation logic.

Motivation:
- Older runs can have incorrect aggregate metrics when the evaluation dataframe used a duplicated
  DatetimeIndex across symbols and labels/preds were misaligned via join().
- We now fixed evaluate_model_on_df() to normalize index ordering; this script regenerates:
    - metrics_fixed.json
    - metrics_fixed_summary.md
    - report_fixed.html

It does NOT overwrite the original metrics.json/report.html by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec  # noqa: E402
from src.time_series_model.models.nn.path_primitives_artifact import (  # noqa: E402
    PathPrimitivesModelArtifact,
)
from src.time_series_model.models.nn.path_primitives_labels import (  # noqa: E402
    PathPrimitivesLabelConfig,
)
from src.time_series_model.models.nn.path_primitives_reporting import (  # noqa: E402
    evaluate_model_on_df,
    render_html_dashboard,
)


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _metrics_summary_md(metrics: Dict[str, Any]) -> str:
    keys = sorted([k for k, v in metrics.items() if isinstance(v, (int, float))])
    lines = ["# metrics_fixed summary\n\n", "| key | value |\n", "|---|---:|\n"]
    for k in keys:
        v = metrics.get(k)
        try:
            vf = float(v)
            lines.append(f"| `{k}` | {vf:.6g} |\n")
        except Exception:
            lines.append(f"| `{k}` | {v} |\n")
    return "".join(lines)


def _find_train_artifact_dir(run_dir: Path) -> Path:
    # Prefer explicit subdir containing model.pt + meta.json + report.html
    cands: List[Path] = []
    for p in run_dir.glob("**/model.pt"):
        if p.parent.name.startswith("nnmh_config_train"):
            cands.append(p.parent)
    if not cands:
        # fallback: any dir with model.pt + meta.json
        for p in run_dir.glob("**/model.pt"):
            if (p.parent / "meta.json").exists():
                cands.append(p.parent)
    if not cands:
        raise FileNotFoundError(f"No train artifact dir found under {run_dir}")
    return max(cands, key=lambda x: x.stat().st_mtime)


def _read_features_for_eval(
    meta: Dict[str, Any], model_art: PathPrimitivesModelArtifact
) -> pd.DataFrame:
    note = (meta.get("eval_note") or {}) if isinstance(meta, dict) else {}
    syms = note.get("symbols") or []
    timeframe = note.get("timeframe") or "240T"
    start = note.get("start") or None
    end = note.get("end") or None
    max_rows_per_symbol = int(note.get("max_rows_per_symbol") or 0)

    symbols = [str(s) for s in list(syms) if str(s).strip()]
    if not symbols:
        raise ValueError(
            "meta.json missing eval_note.symbols; cannot reproduce eval window."
        )
    if start is None or end is None:
        raise ValueError(
            "meta.json missing eval_note.start/end; cannot reproduce eval window."
        )

    # FeatureStore params: try meta->config_dir_arg is TaskSpec-derived, but eval_note didn't store store-layer.
    # For historical runs we assume the default feature_store root and that the run was produced from the unified
    # tree union layer used by pipeline. If you need a custom layer, pass via --feature-store-layer override.
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="results/runs/<RUN_ID>")
    ap.add_argument(
        "--train-dir",
        default=None,
        help="Override train artifact dir (contains model.pt/meta.json)",
    )
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument(
        "--feature-store-layer",
        required=True,
        help="FeatureStore layer for eval reproduction",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing *fixed outputs"
    )
    ap.add_argument(
        "--max-rows-per-symbol",
        type=int,
        default=None,
        help="Override meta eval_note.max_rows_per_symbol",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()

    train_dir = (
        Path(args.train_dir).resolve()
        if args.train_dir
        else _find_train_artifact_dir(run_dir)
    )
    if not train_dir.is_absolute():
        train_dir = (PROJECT_ROOT / train_dir).resolve()

    meta_p = train_dir / "meta.json"
    model_p = train_dir / "model.pt"
    if not meta_p.exists() or not model_p.exists():
        raise FileNotFoundError(f"Missing meta.json/model.pt in {train_dir}")

    meta = _read_json(meta_p)
    note = meta.get("eval_note") or {}
    symbols = [str(s) for s in (note.get("symbols") or [])]
    timeframe = str(note.get("timeframe") or "240T")
    start = pd.Timestamp(note.get("start"))
    end = pd.Timestamp(note.get("end"))
    max_rows = (
        int(args.max_rows_per_symbol)
        if args.max_rows_per_symbol is not None
        else int(note.get("max_rows_per_symbol") or 0)
    )

    art = PathPrimitivesModelArtifact.load(model_path=model_p, config_dir=None)
    if not art.feature_cols:
        raise ValueError("model.pt meta missing feature_cols")

    label_cfg = PathPrimitivesLabelConfig(**(art.meta.get("label_cfg") or {}))

    need_cols = list(
        dict.fromkeys(
            list(art.feature_cols)
            + [
                label_cfg.entry_price_col,
                label_cfg.high_col,
                label_cfg.low_col,
                label_cfg.close_col,
                label_cfg.atr_col,
            ]
        )
    )

    store = FeatureStore(str(args.feature_store_root))
    parts: List[pd.DataFrame] = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(args.feature_store_layer),
            symbol=str(sym),
            timeframe=str(timeframe),
        )
        df = store.read_range(spec, start=start, end=end, columns=need_cols)
        if df.empty:
            continue
        df = df.copy()
        df["symbol"] = sym
        df = df.reset_index(drop=False).rename(columns={"index": "timestamp"})
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        ).dt.tz_localize(None)
        df = (
            df.dropna(subset=["timestamp"])
            .sort_values(["symbol", "timestamp"])
            .reset_index(drop=True)
        )
        if max_rows and max_rows > 0:
            df = df.groupby("symbol", sort=False).tail(max_rows).reset_index(drop=True)
        parts.append(df)
    if not parts:
        raise ValueError("FeatureStore read produced no data")
    df_features = pd.concat(parts, axis=0, ignore_index=True)

    metrics, df_eval, extra = evaluate_model_on_df(
        model=art.model,
        df_features=df_features,
        feature_cols=list(art.feature_cols),
        label_cfg=label_cfg,
        group_col="symbol" if len(symbols) > 1 else None,
        block_cols_by_name=art.block_cols_by_name,
        append_block_mask=art.append_block_mask,
        feature_scaler=art.feature_scaler,
    )

    # Attach extra previews into meta for the fixed report
    meta2 = dict(meta)
    meta2["eval_note_fixed"] = {
        "script": "scripts/fix_nnmh_train_metrics.py",
        "feature_store_root": str(Path(args.feature_store_root).resolve()),
        "feature_store_layer": str(args.feature_store_layer),
        "symbols": symbols,
        "timeframe": timeframe,
        "start": str(start),
        "end": str(end),
        "max_rows_per_symbol": int(max_rows),
    }
    if isinstance(extra, dict) and extra.get("rolling_ic") is not None:
        meta2["rolling_ic"] = extra.get("rolling_ic")

    # Sample like training (last 200 labeled rows)
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
    cols = [
        c
        for c in [
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
        if c in sample_df.columns
    ]
    df_pred_sample = sample_df[cols].copy() if cols else None

    out_metrics = train_dir / "metrics_fixed.json"
    out_md = train_dir / "metrics_fixed_summary.md"
    out_html = train_dir / "report_fixed.html"

    if not args.overwrite:
        for p in [out_metrics, out_md, out_html]:
            if p.exists():
                raise SystemExit(
                    f"Refusing to overwrite existing: {p} (use --overwrite)"
                )

    out_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    out_md.write_text(_metrics_summary_md(metrics), encoding="utf-8")
    html = render_html_dashboard(
        meta=meta2, metrics=metrics, df_pred_sample=df_pred_sample
    )
    out_html.write_text(html, encoding="utf-8")

    print("✅ Wrote:", out_metrics)
    print("✅ Wrote:", out_md)
    print("✅ Wrote:", out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
