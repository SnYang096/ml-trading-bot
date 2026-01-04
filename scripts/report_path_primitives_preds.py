#!/usr/bin/env python3
"""
Generate lightweight diagnostics for nnmultihead path-primitives predictions:
- per-symbol head summary stats / quantiles
- mode distribution (if `mode` column exists)

Inputs:
  --preds: a parquet/csv file OR a directory containing preds_*.parquet
Outputs:
  Writes CSV + JSON into --out-dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# We support both:
# - raw prediction heads (pred_*) from nnmultihead predict outputs
# - transformed heads (head_*) from RL logs (already inverse-log1p where applicable)
PRED_HEAD_COLS = ["pred_dir_prob", "pred_mfe_atr", "pred_mae_atr", "pred_t_to_mfe"]
LOG_HEAD_COLS = ["head_dir_score", "head_mfe_atr", "head_mae_atr", "head_t_to_mfe"]


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_pred_files(preds_path: Path) -> List[Path]:
    if preds_path.is_dir():
        files = sorted(preds_path.glob("preds_*.parquet"))
        if not files:
            files = sorted(preds_path.glob("*.parquet"))
        if not files:
            files = sorted(preds_path.glob("*.csv"))
        return files
    return [preds_path]


def _dir_conf(p: np.ndarray) -> np.ndarray:
    # p in [0,1] -> confidence in [0,1]
    return np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)


def _summary_for_series(x: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    out: Dict[str, float] = {
        "n": float(v.shape[0]),
        "n_nan": float(v.isna().sum()),
        "mean": float(v.mean(skipna=True)) if v.notna().any() else 0.0,
        "std": float(v.std(skipna=True, ddof=1)) if v.notna().sum() > 1 else 0.0,
        "min": float(v.min(skipna=True)) if v.notna().any() else 0.0,
        "p01": float(v.quantile(0.01)) if v.notna().any() else 0.0,
        "p05": float(v.quantile(0.05)) if v.notna().any() else 0.0,
        "p50": float(v.quantile(0.50)) if v.notna().any() else 0.0,
        "p95": float(v.quantile(0.95)) if v.notna().any() else 0.0,
        "p99": float(v.quantile(0.99)) if v.notna().any() else 0.0,
        "max": float(v.max(skipna=True)) if v.notna().any() else 0.0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preds", required=True, help="Preds file or directory of preds_*.parquet"
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output directory for report artifacts"
    )
    args = ap.parse_args()

    preds_path = Path(args.preds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mode_rows = []
    global_mode_counts: Dict[str, int] = {}

    for f in _collect_pred_files(preds_path):
        df = _read_any(f)
        if "symbol" not in df.columns:
            sym = f.stem.replace("preds_", "")
            df = df.copy()
            df["symbol"] = sym

        # If a single file contains multiple symbols (e.g., logs_3action.parquet),
        # compute stats per symbol group.
        groups = [("", df)]
        if "symbol" in df.columns:
            uniq = df["symbol"].astype(str).unique().tolist()
            if len(uniq) > 1:
                groups = [
                    (s, df[df["symbol"].astype(str) == str(s)].copy()) for s in uniq
                ]

        for sym, g in groups:
            if len(g) == 0:
                continue

            # derived: dir_conf
            if "pred_dir_prob" in g.columns:
                p = (
                    pd.to_numeric(g["pred_dir_prob"], errors="coerce")
                    .astype(float)
                    .clip(0.0, 1.0)
                )
                g["pred_dir_conf"] = _dir_conf(p.to_numpy(dtype=float))

            cols = []
            # prefer interpreted head_* if present (logs)
            cols.extend([c for c in LOG_HEAD_COLS if c in g.columns])
            # then pred_* for raw preds
            cols.extend([c for c in PRED_HEAD_COLS if c in g.columns])
            if "pred_dir_conf" in g.columns:
                cols = ["pred_dir_conf"] + cols
            # de-dup while preserving order
            seen = set()
            cols = [c for c in cols if not (c in seen or seen.add(c))]

            for c in cols:
                s = _summary_for_series(g[c])
                rows.append({"symbol": str(sym), "column": c, **s})

            if "mode" in g.columns:
                vc = g["mode"].astype(str).value_counts()
                for k, v in vc.to_dict().items():
                    global_mode_counts[k] = int(global_mode_counts.get(k, 0) + int(v))
                    mode_rows.append(
                        {"symbol": str(sym), "mode": str(k), "count": int(v)}
                    )

    out_csv = out_dir / "preds_head_summary_per_symbol.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if mode_rows:
        out_mode_csv = out_dir / "preds_mode_counts_per_symbol.csv"
        pd.DataFrame(mode_rows).to_csv(out_mode_csv, index=False)

    (out_dir / "preds_mode_counts_global.json").write_text(
        json.dumps(global_mode_counts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ Wrote:", out_csv.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
