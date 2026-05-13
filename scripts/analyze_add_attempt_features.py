#!/usr/bin/env python3
"""Rank add-on attempt features for simple gating rules.

This is intentionally a statistics helper, not a strategy optimizer. It answers:

- Among actually executed add-on legs, which feature values separate positive
  add legs from negative add legs?
- Which one-feature threshold would have retained a reasonable number of add
  legs while improving total R / worst loss / add-leg drawdown?

The script should be run on a no-kill-switch event backtest so the sample is not
truncated by early account halts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _max_drawdown(xs: np.ndarray) -> float:
    if xs.size == 0:
        return 0.0
    eq = np.cumsum(xs.astype(float))
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    dd = peak - eq
    return float(np.max(dd)) if dd.size else 0.0


def _read_add_trades(event_json: Path) -> pd.DataFrame:
    obj = json.loads(event_json.read_text(encoding="utf-8"))
    rows = []
    for t in obj.get("trades") or []:
        if not bool(t.get("is_add_position")):
            continue
        rows.append(
            {
                "symbol": str(t.get("symbol", "")).upper(),
                "timestamp": pd.to_datetime(t.get("entry_time"), utc=True),
                "pnl_r": float(t.get("pnl_r", 0.0) or 0.0),
                "exit_reason": t.get("exit_reason"),
            }
        )
    return pd.DataFrame(rows)


def _load_joined(add_attempts_csv: Path, event_json: Path) -> pd.DataFrame:
    attempts = pd.read_csv(add_attempts_csv)
    if attempts.empty:
        return attempts
    attempts["timestamp"] = pd.to_datetime(attempts["timestamp"], utc=True)
    attempts["symbol"] = attempts["symbol"].astype(str).str.upper()
    attempts["added"] = attempts["added"].astype(str).str.lower().isin({"true", "1"})
    adds = _read_add_trades(event_json)
    if adds.empty:
        attempts["pnl_r"] = np.nan
        return attempts
    return attempts.merge(adds, on=["symbol", "timestamp"], how="left")


def _metrics(xs: pd.DataFrame) -> Dict[str, Any]:
    pnl = xs["pnl_r"].dropna().to_numpy(dtype=float)
    if pnl.size == 0:
        return {
            "n": 0,
            "total_r": 0.0,
            "mean_r": 0.0,
            "win_rate": 0.0,
            "worst_r": 0.0,
            "add_max_drawdown_r": 0.0,
        }
    return {
        "n": int(pnl.size),
        "total_r": float(np.sum(pnl)),
        "mean_r": float(np.mean(pnl)),
        "win_rate": float(np.mean(pnl > 0.0)),
        "worst_r": float(np.min(pnl)),
        "add_max_drawdown_r": _max_drawdown(pnl),
    }


def _threshold_scan(
    df_added: pd.DataFrame,
    feature: str,
    *,
    min_keep: int,
) -> List[Dict[str, Any]]:
    vals = pd.to_numeric(df_added[feature], errors="coerce")
    mask = vals.notna() & df_added["pnl_r"].notna()
    sub = df_added.loc[mask].copy()
    vals = vals.loc[mask]
    if len(sub) < max(min_keep, 8):
        return []
    qs = np.linspace(0.1, 0.9, 17)
    thresholds = sorted(set(float(np.nanquantile(vals, q)) for q in qs))
    rows: List[Dict[str, Any]] = []
    for direction in ("lte", "gte"):
        for thr in thresholds:
            keep = vals <= thr if direction == "lte" else vals >= thr
            kept = sub.loc[keep]
            if len(kept) < min_keep:
                continue
            m = _metrics(kept)
            dropped = len(sub) - len(kept)
            rows.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "threshold": float(thr),
                    "dropped": int(dropped),
                    **m,
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-attempts", required=True, type=Path)
    ap.add_argument("--event-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-keep", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_joined(args.add_attempts, args.event_json)
    added = df[df["added"] & df["pnl_r"].notna()].copy()
    baseline = _metrics(added)

    skip = {
        "timestamp",
        "symbol",
        "archetype",
        "side",
        "path_type",
        "outcome",
        "added",
        "pnl_r",
        "exit_reason",
    }
    numeric_features = [
        c
        for c in added.columns
        if c not in skip
        and pd.to_numeric(added[c], errors="coerce").notna().sum() >= args.min_keep
    ]

    feature_rows: List[Dict[str, Any]] = []
    y = (added["pnl_r"] > 0).astype(int)
    for feat in numeric_features:
        x = pd.to_numeric(added[feat], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < args.min_keep or y.loc[mask].nunique() < 2:
            continue
        auc = float(roc_auc_score(y.loc[mask], x.loc[mask]))
        # Use max(auc, 1-auc): feature can be useful in either direction.
        feature_rows.append(
            {
                "feature": feat,
                "n": int(mask.sum()),
                "auc": auc,
                "auc_strength": max(auc, 1.0 - auc),
                "good_mean": float(x.loc[mask & (y == 1)].mean()),
                "bad_mean": float(x.loc[mask & (y == 0)].mean()),
            }
        )

    threshold_rows: List[Dict[str, Any]] = []
    for feat in numeric_features:
        threshold_rows.extend(_threshold_scan(added, feat, min_keep=args.min_keep))

    thresh = pd.DataFrame(threshold_rows)
    if not thresh.empty:
        # Prefer rules that reduce add-leg DD and worst loss without killing total R.
        thresh["dd_reduction"] = (
            baseline["add_max_drawdown_r"] - thresh["add_max_drawdown_r"]
        )
        thresh["worst_improvement"] = thresh["worst_r"] - baseline["worst_r"]
        thresh["score"] = (
            thresh["dd_reduction"]
            + 0.5 * thresh["worst_improvement"]
            + 0.05 * thresh["total_r"]
        )
        thresh = thresh.sort_values("score", ascending=False)

    feat_df = pd.DataFrame(feature_rows).sort_values("auc_strength", ascending=False)
    feat_df.to_csv(args.out_dir / "feature_auc.csv", index=False)
    thresh.to_csv(args.out_dir / "threshold_candidates.csv", index=False)

    summary = {
        "event_json": str(args.event_json),
        "add_attempts": str(args.add_attempts),
        "attempt_rows": int(len(df)),
        "added_rows": int(len(added)),
        "baseline_add_metrics": baseline,
        "top_features": feat_df.head(args.top_n).to_dict("records"),
        "top_thresholds": (
            thresh.head(args.top_n).to_dict("records") if not thresh.empty else []
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
