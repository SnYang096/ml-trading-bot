#!/usr/bin/env python3
"""Train walk-forward ML add-on scores on bar-level panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _expand_side(panel: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[pd.DataFrame] = []
    for side, ycol, qcol in (
        ("long", "add_good_long", "future_quality_long"),
        ("short", "add_good_short", "future_quality_short"),
    ):
        if ycol not in panel.columns or qcol not in panel.columns:
            continue
        df = panel.copy()
        df["side"] = side
        df["side_sign"] = 1.0 if side == "long" else -1.0
        df["y"] = panel[ycol].astype(int)
        df["quality"] = pd.to_numeric(panel[qcol], errors="coerce")
        out_rows.append(df)
    if not out_rows:
        return pd.DataFrame()
    out = pd.concat(out_rows, axis=0, ignore_index=True)
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


def _feature_columns(df: pd.DataFrame, min_non_nan: int) -> List[str]:
    skip = {
        "timestamp",
        "symbol",
        "side",
        "year",
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
    feats: List[str] = []
    for col in df.columns:
        if col in skip:
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= int(min_non_nan):
            feats.append(col)
    return sorted(set(feats))


def _prepare_x(df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    x = (
        df[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return x.to_numpy(dtype=float, copy=False)


def _precision_at_top(
    y_true: np.ndarray, score: np.ndarray, frac: float = 0.1
) -> float:
    if y_true.size == 0:
        return 0.0
    k = max(1, int(round(len(y_true) * float(frac))))
    idx = np.argpartition(score, -k)[-k:]
    return float(np.mean(y_true[idx] > 0.5))


def _fit_predict_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if model_name == "logreg":
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x_train)
        xte = scaler.transform(x_test)
        clf = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(xtr, y_train)
        return clf.predict_proba(xte)[:, 1]
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=40,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(x_train, y_train)
        return clf.predict_proba(x_test)[:, 1]
    raise ValueError(f"unknown model: {model_name}")


def _walk_forward(
    df: pd.DataFrame, feat_cols: List[str], model_name: str
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    years = sorted(int(y) for y in df["year"].unique())
    preds = np.full(len(df), np.nan, dtype=float)
    fold_metrics: List[Dict[str, Any]] = []
    for y in years:
        tr = df[df["year"] < y]
        te = df[df["year"] == y]
        if tr.empty or te.empty:
            continue
        y_train = tr["y"].to_numpy(dtype=int, copy=False)
        if np.unique(y_train).size < 2:
            continue
        x_train = _prepare_x(tr, feat_cols)
        x_test = _prepare_x(te, feat_cols)
        p = _fit_predict_model(model_name, x_train, y_train, x_test)
        idx = te.index.to_numpy(dtype=int, copy=False)
        preds[idx] = p
        y_test = te["y"].to_numpy(dtype=int, copy=False)
        metric = {
            "year": int(y),
            "n_test": int(len(te)),
            "auc": (
                float(roc_auc_score(y_test, p)) if np.unique(y_test).size > 1 else 0.5
            ),
            "precision_top10": _precision_at_top(y_test, p, frac=0.1),
            "mean_score": float(np.mean(p)),
            "good_rate": float(np.mean(y_test > 0.5)),
        }
        fold_metrics.append(metric)
    preds = np.where(np.isnan(preds), 0.5, preds)
    return preds, fold_metrics


def _choose_threshold(
    oos_df: pd.DataFrame, *, min_keep: float, max_keep: float
) -> Dict[str, Any]:
    vals = oos_df["score_side"].to_numpy(dtype=float, copy=False)
    y = oos_df["y"].to_numpy(dtype=float, copy=False)
    q = oos_df["quality"].to_numpy(dtype=float, copy=False)
    base_q = float(np.nanmean(q)) if q.size else 0.0
    base_good = float(np.mean(y > 0.5)) if y.size else 0.0
    best = None
    for thr in sorted(
        set(float(np.quantile(vals, qv)) for qv in np.linspace(0.6, 0.97, 20))
    ):
        keep = vals >= thr
        keep_rate = float(np.mean(keep))
        if not (min_keep <= keep_rate <= max_keep):
            continue
        if int(np.sum(keep)) < 50:
            continue
        yk = y[keep]
        qk = q[keep]
        good = float(np.mean(yk > 0.5))
        qmean = float(np.nanmean(qk))
        score = (qmean - base_q) + 0.5 * (good - base_good) + 0.1 * keep_rate
        row = {
            "threshold": float(thr),
            "keep_rate": keep_rate,
            "good_rate": good,
            "quality_mean": qmean,
            "delta_good_rate": good - base_good,
            "delta_quality": qmean - base_q,
            "score": float(score),
        }
        if best is None or row["score"] > best["score"]:
            best = row
    if best is None:
        best = {
            "threshold": 0.55,
            "keep_rate": 0.0,
            "good_rate": 0.0,
            "quality_mean": 0.0,
            "delta_good_rate": 0.0,
            "delta_quality": 0.0,
            "score": -1e9,
        }
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-non-nan", type=int, default=300)
    ap.add_argument("--min-keep-rate", type=float, default=0.15)
    ap.add_argument("--max-keep-rate", type=float, default=0.85)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    df = _expand_side(panel)
    if df.empty:
        raise RuntimeError("Panel missing labels for long/short sides")

    feat_cols = _feature_columns(df, min_non_nan=int(args.min_non_nan))
    model_summaries: List[Dict[str, Any]] = []
    oos_scores: Dict[str, np.ndarray] = {}
    for model_name in ("logreg", "rf"):
        preds, folds = _walk_forward(df, feat_cols, model_name=model_name)
        oos_scores[model_name] = preds
        aucs = [f["auc"] for f in folds]
        p10s = [f["precision_top10"] for f in folds]
        model_summaries.append(
            {
                "model": model_name,
                "n_folds": int(len(folds)),
                "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
                "mean_precision_top10": float(np.mean(p10s)) if p10s else 0.0,
                "folds": folds,
            }
        )

    model_summaries = sorted(
        model_summaries, key=lambda x: x["mean_precision_top10"], reverse=True
    )
    best_model = model_summaries[0]["model"]
    df = df.copy()
    df["score_side"] = oos_scores[best_model]
    best_thr = _choose_threshold(
        df,
        min_keep=float(args.min_keep_rate),
        max_keep=float(args.max_keep_rate),
    )

    # Pivot long/short side scores back to bar rows for event-backtest injection.
    wide = (
        df[["symbol", "timestamp", "side", "score_side"]]
        .pivot_table(
            index=["symbol", "timestamp"],
            columns="side",
            values="score_side",
            aggfunc="last",
        )
        .reset_index()
    )
    wide.columns.name = None
    if "long" not in wide.columns:
        wide["long"] = 0.5
    if "short" not in wide.columns:
        wide["short"] = 0.5
    wide = wide.rename(
        columns={"long": "add_ml_score_long", "short": "add_ml_score_short"}
    )
    wide["add_ml_score"] = wide[["add_ml_score_long", "add_ml_score_short"]].max(axis=1)
    score_path = out_dir / "add_ml_scores.parquet"
    wide.to_parquet(score_path, index=False)

    best_gate = {
        "model": best_model,
        "feature_by_side": {
            "long": "add_ml_score_long",
            "short": "add_ml_score_short",
        },
        "direction": "gte",
        **best_thr,
    }
    (out_dir / "best_ml_gate.json").write_text(
        json.dumps(best_gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "panel": str(args.panel),
        "rows_side_expanded": int(len(df)),
        "feature_count": int(len(feat_cols)),
        "best_model": best_model,
        "models": model_summaries,
        "best_ml_gate": best_gate,
        "score_parquet": str(score_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
