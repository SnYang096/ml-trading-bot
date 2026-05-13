#!/usr/bin/env python3
"""Search simple add-on gating rules on bar-level panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


LABEL_MAP = {
    "long": ("add_good_long", "future_quality_long"),
    "short": ("add_good_short", "future_quality_short"),
}


def _expand_side(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, (ycol, qcol) in LABEL_MAP.items():
        if ycol not in panel.columns or qcol not in panel.columns:
            continue
        df = panel.copy()
        df["side"] = side
        df["y"] = panel[ycol].astype(float)
        df["quality"] = pd.to_numeric(panel[qcol], errors="coerce")
        df["side_sign"] = 1.0 if side == "long" else -1.0
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, axis=0, ignore_index=True)
    # Direction-aligned variants for sign-sensitive features.
    for col in (
        "roc_5",
        "roc_10",
        "roc_20",
        "macd_atr",
        "ema_1200_position",
        "macro_tp_vwap_1200_position",
    ):
        if col in out.columns:
            out[f"{col}_aligned"] = (
                pd.to_numeric(out[col], errors="coerce") * out["side_sign"]
            )
    out["year"] = pd.to_datetime(out["timestamp"], utc=True).dt.year.astype(int)
    return out


def _safe_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _eval_subset(sub: pd.DataFrame) -> Dict[str, float]:
    if sub.empty:
        return {
            "n": 0.0,
            "keep_rate": 0.0,
            "good_rate": 0.0,
            "quality_mean": 0.0,
        }
    y = sub["y"].to_numpy(dtype=float)
    q = pd.to_numeric(sub["quality"], errors="coerce").to_numpy(dtype=float)
    q = q[np.isfinite(q)]
    return {
        "n": float(len(sub)),
        "keep_rate": 0.0,  # filled outside
        "good_rate": float(np.mean(y > 0.5)) if len(y) else 0.0,
        "quality_mean": float(np.mean(q)) if len(q) else 0.0,
    }


def _candidate_features(df: pd.DataFrame, min_non_nan: int) -> List[str]:
    skip = {
        "timestamp",
        "symbol",
        "year",
        "side",
        "y",
        "quality",
        "add_good_long",
        "add_good_short",
        "future_quality_long",
        "future_quality_short",
        "future_mfe_atr_long",
        "future_mfe_atr_short",
        "future_mae_atr_long",
        "future_mae_atr_short",
    }
    cands: List[str] = []
    for col in df.columns:
        if col in skip:
            continue
        s = _safe_num_series(df, col)
        if int(s.notna().sum()) >= int(min_non_nan):
            cands.append(col)
    return sorted(set(cands))


def _scan_rules(
    df: pd.DataFrame,
    *,
    features: List[str],
    min_keep_rate: float,
    max_keep_rate: float,
    min_test_rows: int,
) -> pd.DataFrame:
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < 2:
        raise RuntimeError("Need at least two years for walk-forward rule evaluation")
    folds = years[1:]
    rows: List[Dict[str, Any]] = []
    for feat in features:
        vals_all = _safe_num_series(df, feat)
        if vals_all.notna().sum() < 50:
            continue
        quantiles = sorted(
            set(float(np.nanquantile(vals_all, q)) for q in np.linspace(0.1, 0.9, 17))
        )
        for direction in ("lte", "gte"):
            for thr in quantiles:
                test_parts: List[pd.DataFrame] = []
                train_parts: List[pd.DataFrame] = []
                for y in folds:
                    train = df[df["year"] < y]
                    test = df[df["year"] == y]
                    if len(train) < min_test_rows or len(test) < min_test_rows:
                        continue
                    xtr = _safe_num_series(train, feat)
                    xte = _safe_num_series(test, feat)
                    keep_tr = xtr <= thr if direction == "lte" else xtr >= thr
                    keep_te = xte <= thr if direction == "lte" else xte >= thr
                    tr_kept = train.loc[keep_tr.fillna(False)]
                    te_kept = test.loc[keep_te.fillna(False)]
                    tr_keep_rate = len(tr_kept) / max(len(train), 1)
                    te_keep_rate = len(te_kept) / max(len(test), 1)
                    if not (min_keep_rate <= tr_keep_rate <= max_keep_rate):
                        continue
                    if not (min_keep_rate <= te_keep_rate <= max_keep_rate):
                        continue
                    test_parts.append(te_kept)
                    train_parts.append(tr_kept)
                if not test_parts:
                    continue
                all_te = pd.concat(test_parts, axis=0, ignore_index=True)
                all_base = df[df["year"].isin(folds)]
                m_kept = _eval_subset(all_te)
                m_base = _eval_subset(all_base)
                m_kept["keep_rate"] = len(all_te) / max(len(all_base), 1)
                delta_quality = m_kept["quality_mean"] - m_base["quality_mean"]
                delta_good = m_kept["good_rate"] - m_base["good_rate"]
                # Reward better quality/good-rate and adequate retained opportunity.
                score = delta_quality + 0.5 * delta_good + 0.1 * m_kept["keep_rate"]
                rows.append(
                    {
                        "feature": feat,
                        "direction": direction,
                        "threshold": float(thr),
                        "score": float(score),
                        "delta_quality": float(delta_quality),
                        "delta_good_rate": float(delta_good),
                        "keep_rate": float(m_kept["keep_rate"]),
                        "kept_rows": int(len(all_te)),
                        "base_rows": int(len(all_base)),
                        "kept_quality_mean": float(m_kept["quality_mean"]),
                        "base_quality_mean": float(m_base["quality_mean"]),
                        "kept_good_rate": float(m_kept["good_rate"]),
                        "base_good_rate": float(m_base["good_rate"]),
                        "n_folds": int(len(test_parts)),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-non-nan", type=int, default=300)
    ap.add_argument("--min-keep-rate", type=float, default=0.15)
    ap.add_argument("--max-keep-rate", type=float, default=0.85)
    ap.add_argument("--min-test-rows", type=int, default=100)
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    df = _expand_side(panel)
    if df.empty:
        raise RuntimeError(
            "Panel missing side labels (add_good_long/short, future_quality_long/short)"
        )

    feats = _candidate_features(df, min_non_nan=int(args.min_non_nan))
    ranked = _scan_rules(
        df,
        features=feats,
        min_keep_rate=float(args.min_keep_rate),
        max_keep_rate=float(args.max_keep_rate),
        min_test_rows=int(args.min_test_rows),
    )
    ranked_path = out_dir / "rule_candidates.csv"
    ranked.to_csv(ranked_path, index=False)

    if ranked.empty:
        summary = {
            "panel": str(args.panel),
            "rows": int(len(df)),
            "features_scanned": int(len(feats)),
            "message": "no valid rule candidates",
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    best = ranked.iloc[0].to_dict()
    best_rule = {
        "feature": str(best["feature"]),
        "direction": str(best["direction"]),
        "threshold": float(best["threshold"]),
        "score": float(best["score"]),
        "keep_rate": float(best["keep_rate"]),
        "delta_quality": float(best["delta_quality"]),
        "delta_good_rate": float(best["delta_good_rate"]),
    }
    (out_dir / "best_rule.json").write_text(
        json.dumps(best_rule, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "panel": str(args.panel),
        "rows": int(len(df)),
        "features_scanned": int(len(feats)),
        "top_candidates": ranked.head(int(args.top_n)).to_dict("records"),
        "best_rule": best_rule,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"best_rule": best_rule, "ranked_csv": str(ranked_path)},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
