#!/usr/bin/env python3
"""
Per-symbol KPI report for nnmultihead path-primitives models.

Why:
- When training across multiple symbols, aggregate AUC can hide that 1-2 symbols are broken
  (bad data, label mismatch, distribution shift, etc.).
- This script computes per-symbol dir_auc on:
  1) all samples
  2) Router-defined trade subset (MEAN/TREND), under a *fixed* router thresholds JSON

Inputs:
- model.pt (nnmultihead path-primitives)
- FeatureStore (root+layer) for features
- timeframe + date range
- symbols list
- optional router thresholds json to define the trade subset

Outputs:
- per_symbol_kpi.json
- per_symbol_kpi.csv
- per_symbol_kpi.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec  # noqa: E402
from src.time_series_model.models.nn.path_primitives_artifact import (  # noqa: E402
    PathPrimitivesModelArtifact,
)
from src.time_series_model.models.nn.path_primitives_labels import (  # noqa: E402
    PathPrimitivesLabelConfig,
    compute_path_primitives_labels,
)
from src.time_series_model.models.nn.path_primitives_reporting import (  # noqa: E402
    predict_path_primitives,
)
from src.time_series_model.models.nn.path_primitives_eval import (  # noqa: E402
    evaluate_path_primitives,
)
from src.time_series_model.rule.router_3action import (  # noqa: E402
    Rule3ActionConfig,
    compute_mode_3action,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.pt")
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--features-store-root", default="feature_store")
    ap.add_argument("--features-store-layer", required=True)
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--router-thresholds-json",
        default=None,
        help="Optional router thresholds JSON to define trade subset (MEAN/TREND).",
    )
    ap.add_argument(
        "--max-rows-per-symbol",
        type=int,
        default=0,
        help="Memory-safety: keep last N rows per symbol (0 disables).",
    )
    return ap.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_router_cfg(router_json: Optional[str]) -> Rule3ActionConfig:
    cfg = Rule3ActionConfig()
    if not router_json:
        return cfg
    p = Path(router_json)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    obj = _read_json(p) if p.exists() else {}
    if not isinstance(obj, dict):
        return cfg
    d = {**cfg.__dict__}
    for k in [
        "mfe_min",
        "eff_min",
        "dir_conf_trend_min",
        "mfe_trend_min",
        "ttm_trend_min",
        "eff_mean_min",
        "ttm_mean_max",
    ]:
        if k in obj and obj.get(k) is not None:
            try:
                d[k] = float(obj.get(k))
            except Exception:
                pass
    return Rule3ActionConfig(**d)


def _read_features(
    *,
    store_root: str,
    layer: str,
    symbols: List[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    store = FeatureStore(str(store_root))
    parts: List[pd.DataFrame] = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=str(layer), symbol=str(sym), timeframe=str(timeframe)
        )
        df = store.read_range(spec, start=start, end=end, columns=columns)
        if df is None or df.empty:
            continue
        if "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = sym
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df_all = pd.concat(parts, axis=0, ignore_index=False)
    # Ensure timestamp exists (FeatureStore uses index)
    if "timestamp" not in df_all.columns:
        df_all = df_all.reset_index(drop=False).rename(columns={"index": "timestamp"})
    df_all["timestamp"] = pd.to_datetime(
        df_all["timestamp"], errors="coerce"
    ).dt.tz_localize(None)
    df_all = (
        df_all.dropna(subset=["timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    return df_all


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)

    art = PathPrimitivesModelArtifact.load(model_path=Path(args.model), config_dir=None)
    if not art.feature_cols:
        raise ValueError(
            "model.pt meta missing feature_cols; cannot evaluate per-symbol safely."
        )

    label_cfg = PathPrimitivesLabelConfig(**(art.meta.get("label_cfg") or {}))

    # Read only needed columns to reduce memory footprint
    need_cols = list(art.feature_cols) + [
        label_cfg.entry_price_col,
        label_cfg.high_col,
        label_cfg.low_col,
        label_cfg.close_col,
        label_cfg.atr_col,
    ]
    # de-dup
    need_cols = list(dict.fromkeys([c for c in need_cols if c]))

    df = _read_features(
        store_root=str(args.features_store_root),
        layer=str(args.features_store_layer),
        symbols=symbols,
        timeframe=str(args.timeframe),
        start=start,
        end=end,
        columns=need_cols,
    )
    if df.empty:
        raise ValueError("Empty FeatureStore read (no data in range).")

    # Optional memory safety: keep last N rows per symbol
    max_n = int(args.max_rows_per_symbol)
    if max_n and max_n > 0:
        df = df.groupby("symbol", sort=False).tail(max_n).reset_index(drop=True)

    # Compute labels per symbol (group-safe)
    df_labels = compute_path_primitives_labels(df, cfg=label_cfg, group_col="symbol")
    work = df.join(df_labels)

    # Predict (adds pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe)
    preds = predict_path_primitives(
        model=art.model,
        df=work,
        feature_cols=list(art.feature_cols),
        fill_nan_value=float(
            (art.meta.get("dataset_cfg") or {}).get("fill_nan_value", 0.0)
        ),
        block_cols_by_name=art.block_cols_by_name,
        append_block_mask=art.append_block_mask,
        feature_scaler=art.feature_scaler,
        device=None,
    )
    work = work.join(preds)

    # Router trade subset (optional)
    router_cfg = _load_router_cfg(args.router_thresholds_json)
    preds_in_log1p = art.preds_in_log1p(True)
    router_out = compute_mode_3action(
        work, cfg=router_cfg, preds_in_log1p=preds_in_log1p, out_col="router_mode"
    )
    work = work.join(router_out[["router_mode"]])

    pred_cols = {
        "dir": "pred_dir_prob",
        "mfe_atr": "pred_mfe_atr",
        "mae_atr": "pred_mae_atr",
        "t_to_mfe": "pred_t_to_mfe",
    }
    true_cols = {
        "dir_y": "dir_y",
        "mfe_atr": "mfe_atr",
        "mae_atr": "mae_atr",
        "t_to_mfe": "t_to_mfe",
    }

    rows = []
    for sym, g in work.groupby("symbol", sort=False):
        g = g.dropna(subset=["dir_y", "pred_dir_prob"])
        if g.empty:
            continue
        m_all = evaluate_path_primitives(
            df=g, pred_cols=pred_cols, true_cols=true_cols, mask_col="mfe_valid"
        )
        trade_mask = (
            g["router_mode"].astype(str).isin(["MEAN", "TREND"])
            if "router_mode" in g.columns
            else pd.Series(False, index=g.index)
        )
        g_trade = g.loc[trade_mask] if bool(trade_mask.any()) else g.iloc[:0]
        m_trade = (
            evaluate_path_primitives(
                df=g_trade,
                pred_cols=pred_cols,
                true_cols=true_cols,
                mask_col="mfe_valid",
            )
            if len(g_trade)
            else {}
        )
        rows.append(
            {
                "symbol": str(sym),
                "n_all": int(len(g)),
                "dir_auc_all": float(m_all.get("dir_auc", 0.0)),
                "dir_acc_all": float(m_all.get("dir_acc", 0.0)),
                "trade_rate": float(trade_mask.mean()) if len(g) else 0.0,
                "n_trade": int(len(g_trade)),
                "dir_auc_trade": (
                    float(m_trade.get("dir_auc", np.nan)) if m_trade else float("nan")
                ),
                "dir_acc_trade": (
                    float(m_trade.get("dir_acc", np.nan)) if m_trade else float("nan")
                ),
            }
        )

    df_out = pd.DataFrame(rows).sort_values("dir_auc_all", ascending=True)
    meta = {
        "model": str(Path(args.model).resolve()),
        "symbols": symbols,
        "timeframe": str(args.timeframe),
        "start": str(start),
        "end": str(end),
        "features_store_root": str(args.features_store_root),
        "features_store_layer": str(args.features_store_layer),
        "router_thresholds_json": args.router_thresholds_json,
        "preds_in_log1p": bool(preds_in_log1p),
        "n_symbols": int(len(df_out)),
    }

    (out_dir / "per_symbol_kpi.json").write_text(
        json.dumps(
            {"meta": meta, "rows": df_out.to_dict(orient="records")},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    df_out.to_csv(out_dir / "per_symbol_kpi.csv", index=False)

    md = []
    md.append("# nnmh per-symbol primitives KPI\n\n")
    md.append(f"- model: `{meta['model']}`\n")
    md.append(
        f"- timeframe: `{meta['timeframe']}` range=[{meta['start']}..{meta['end']}]\n"
    )
    md.append(f"- layer: `{meta['features_store_layer']}`\n")
    md.append(f"- router_thresholds_json: `{meta['router_thresholds_json']}`\n\n")
    md.append(
        "| symbol | n_all | dir_auc_all | dir_acc_all | trade_rate | n_trade | dir_auc_trade |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in df_out.to_dict(orient="records"):
        md.append(
            f"| `{r['symbol']}` | {r['n_all']} | {r['dir_auc_all']:.6g} | {r['dir_acc_all']:.6g} | {r['trade_rate']:.6g} | {r['n_trade']} | {r['dir_auc_trade']:.6g} |\n"
        )
    (out_dir / "per_symbol_kpi.md").write_text("".join(md), encoding="utf-8")

    print("✅ Wrote:", (out_dir / "per_symbol_kpi.json"))
    print("✅ Wrote:", (out_dir / "per_symbol_kpi.csv"))
    print("✅ Wrote:", (out_dir / "per_symbol_kpi.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
