#!/usr/bin/env python3
"""Train adverse excursion tree gate (P(bad|entry)) for fast_scalp experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.profile_tree_entry_excursion import (  # noqa: E402
    _cross_entries,
    _excursion_for_entry,
    _load_1min,
    _load_direction_thresholds,
    _to_naive_ts,
    _to_utc_ts,
)
from src.time_series_model.gating.tree_gate import (  # noqa: E402
    TreeGateTrainConfig,
    export_tree_gate_artifact,
    gate_predict,
    save_tree_gate_artifact,
    train_tree_gate,
)


# Columns that must never enter the gate as features (forward-looking / label-side
# leakage, identifiers, the entry score itself).
_LEAKAGE_COLS = frozenset(
    {
        "mae_atr",
        "mfe_atr",
        "forward_rr",
        "realized_r_long",
        "realized_r_short",
        "label",
        "pred",
        "score",
        "split",
        "symbol",
        "_symbol",
        "timestamp",
    }
)


def _load_candidate_pool(
    gate_df: pd.DataFrame, features_gate_yaml: Path | None
) -> list[str]:
    """Candidate gate features = numeric columns of the wide gate parquet, minus
    leakage/meta. If a features_gate.yaml is supplied, intersect with its declared
    requested_features (its exclude_columns are honored at prepare time)."""
    numeric = [
        c
        for c in gate_df.columns
        if c not in _LEAKAGE_COLS and pd.api.types.is_numeric_dtype(gate_df[c])
    ]
    if features_gate_yaml is not None and features_gate_yaml.exists():
        raw = yaml.safe_load(features_gate_yaml.read_text(encoding="utf-8")) or {}
        fp = raw.get("feature_pipeline", {}) or {}
        declared = set(fp.get("requested_features", []) or [])
        # requested_features may name nodes (e.g. atr_f) producing column atr; keep
        # any column whose name or name+"_f" was declared.
        keep = [
            c
            for c in numeric
            if c in declared or f"{c}_f" in declared or c.replace("_f", "") in declared
        ]
        if keep:
            return keep
    return numeric


def select_gate_features_ic(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    min_abs_ic: float = 0.03,
    min_lift: float = 0.05,
    top_k: int = 8,
) -> tuple[list[str], list[dict[str, Any]]]:
    """IC-prune + lift filter for gate candidate features against the adverse label.

    For each candidate column we compute:
      - point-biserial |IC| with the adverse label (y_bad = 1 - allow),
      - tail lift: adverse-rate in the worst-half vs overall (a feature is kept only
        if its extreme split actually concentrates adverse events).
    Survivors with |IC| >= min_abs_ic AND |lift| >= min_lift are ranked by |IC| and
    capped at top_k. Returns (selected, diagnostics)."""
    y_bad = 1 - np.asarray(y, dtype=int)  # 1 = adverse
    base_rate = float(np.mean(y_bad)) if y_bad.size else 0.0
    diags: list[dict[str, Any]] = []
    for col in X.columns:
        v = pd.to_numeric(X[col], errors="coerce").to_numpy()
        m = np.isfinite(v)
        if m.sum() < 100 or np.unique(v[m]).size < 5:
            continue
        vv, bb = v[m], y_bad[m]
        if np.std(vv) == 0:
            continue
        ic = float(np.corrcoef(vv, bb)[0, 1])
        # tail lift: split at the side of the IC sign, top 30% by adverse direction
        thr = np.quantile(vv, 0.70 if ic >= 0 else 0.30)
        tail = vv >= thr if ic >= 0 else vv <= thr
        tail_rate = float(np.mean(bb[tail])) if tail.any() else base_rate
        lift = (tail_rate - base_rate) / base_rate if base_rate > 0 else 0.0
        diags.append(
            {
                "feature": col,
                "ic": round(ic, 4),
                "abs_ic": round(abs(ic), 4),
                "tail_adverse_rate": round(tail_rate, 4),
                "base_rate": round(base_rate, 4),
                "lift": round(lift, 4),
            }
        )
    diags.sort(key=lambda d: d["abs_ic"], reverse=True)
    selected = [
        d["feature"]
        for d in diags
        if d["abs_ic"] >= min_abs_ic and abs(d["lift"]) >= min_lift
    ][:top_k]
    return selected, diags


def _prepare_entry_scores(
    predictions: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load the score parquet WITHOUT the holdout-only filter used by τ-scan.

    The adverse gate is fit on the in-sample (pre-holdout) entries, so we must
    keep the training rows here; ``train_end_date`` later cuts at the holdout
    boundary. Timestamp is the parquet index and is restored to a column."""
    df = pd.read_parquet(predictions)
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
    return df


def _build_training_rows(
    *,
    config_dir: Path,
    predictions: Path,
    gate_features: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_path: str,
    max_holding_bars: int,
    bar_minutes: int,
    mae_bad_r: float,
    candidate_features: list[str],
    train_end_date: str | None = None,
    long_entry: float | None = None,
    short_entry: float | None = None,
    entry_mode: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (X_candidates, y_allow) over entries.

    Entries + side come from the score parquet (`predictions`); candidate gate
    feature *values* are joined from the WIDE gate parquet (`gate_features`) at the
    entry (symbol, timestamp). The adverse label is the realized 1-minute MAE
    (>= mae_bad_r ATR ⇒ adverse ⇒ allow=0)."""
    direction_cfg = _load_direction_thresholds(config_dir)
    # The gate uses the same entry thresholds as the ranker it protects. For an
    # execution-aligned ranker the score range differs from the H=3 sign config,
    # so callers pass the τ-scan long/short entries explicitly.
    thr = dict(direction_cfg.get("thresholds") or {})
    patched = False
    if long_entry is not None:
        thr["long_entry"] = float(long_entry)
        patched = True
    if short_entry is not None:
        thr["short_entry"] = float(short_entry)
        patched = True
    if entry_mode is not None:
        thr["entry_mode"] = str(entry_mode).lower()
        patched = True
    if patched:
        direction_cfg = {**direction_cfg, "thresholds": thr}
    df = _prepare_entry_scores(predictions, start_date=start_date, end_date=end_date)
    if "_symbol" not in df.columns and "symbol" in df.columns:
        df["_symbol"] = df["symbol"]
    sym_set = {s.upper() for s in symbols}
    df = df[df["_symbol"].astype(str).str.upper().isin(sym_set)].copy()
    # One open position per symbol: block re-entry within the holding window so
    # the gate trains on realistic trade entries, not every qualifying bar (a
    # level-mode ranker fires on nearly every bar and over-samples the gate).
    entries = _cross_entries(
        df,
        direction_cfg=direction_cfg,
        score_col="pred",
        min_gap_bars=max_holding_bars,
    )

    # wide gate features, indexed per symbol by UTC timestamp
    gdf = pd.read_parquet(gate_features)
    if "symbol" not in gdf.columns and "_symbol" in gdf.columns:
        gdf["symbol"] = gdf["_symbol"]
    if not isinstance(gdf.index, pd.DatetimeIndex):
        ts_col = "timestamp" if "timestamp" in gdf.columns else gdf.columns[0]
        gdf = gdf.set_index(ts_col)
    gdf.index = pd.to_datetime(gdf.index, utc=True)
    cand = [c for c in candidate_features if c in gdf.columns]
    if not cand:
        raise ValueError(
            f"none of the candidate gate features present in {gate_features}"
        )
    gate_by_sym = {
        sym: gdf[gdf["symbol"].astype(str).str.upper() == sym].sort_index()
        for sym in sym_set
    }

    # ATR for MAE-normalization comes from the score parquet (the wide gate
    # parquet does not carry a reliable atr column); index per (symbol, ts).
    atr_by_sym: dict[str, pd.Series] = {}
    if "atr" in df.columns:
        for sym in sym_set:
            sub = df[df["_symbol"].astype(str).str.upper() == sym]
            atr_by_sym[sym] = pd.to_numeric(sub["atr"], errors="coerce")

    max_1m = int(max_holding_bars * bar_minutes)
    bars_cache = {
        sym: _load_1min(sym, data_path=data_path, start=start_date, end=end_date)
        for sym in sorted(sym_set)
    }
    train_cut = _to_utc_ts(train_end_date) if train_end_date else None

    rows: list[dict[str, float]] = []
    y_rows: list[int] = []
    for _, row in entries.iterrows():
        sym = str(row["symbol"]).upper()
        ts_utc = _to_utc_ts(row["timestamp"])
        if train_cut is not None and ts_utc >= train_cut:
            continue
        side = str(row["side"])
        gsym = gate_by_sym.get(sym)
        if gsym is None or ts_utc not in gsym.index:
            continue
        feat_row = gsym.loc[ts_utc]
        if isinstance(feat_row, pd.DataFrame):
            feat_row = feat_row.iloc[0]
        atr_ser = atr_by_sym.get(sym)
        atr = np.nan
        if atr_ser is not None and ts_utc in atr_ser.index:
            atr_at = atr_ser.loc[ts_utc]
            atr = float(atr_at.iloc[0] if hasattr(atr_at, "iloc") else atr_at)
        if not np.isfinite(atr):
            atr = float(pd.to_numeric(feat_row.get("atr"), errors="coerce"))
        exc = _excursion_for_entry(
            bars_cache.get(sym, pd.DataFrame()),
            entry_ts=ts_utc,
            side=side,
            atr=atr,
            max_bars=max_1m,
        )
        if exc is None:
            continue
        vec = {c: pd.to_numeric(feat_row.get(c), errors="coerce") for c in cand}
        bad = int(exc["mae_atr"] >= mae_bad_r)
        rows.append(vec)
        y_rows.append(0 if bad else 1)
    if not rows:
        raise ValueError("no training rows produced for adverse gate")
    X = pd.DataFrame(rows)
    return X, np.asarray(y_rows, dtype=int)


def _real_gate_metrics(allow_pred: np.ndarray, y_allow: np.ndarray) -> dict[str, Any]:
    """Lift on the REAL adverse label (no pseudo returns)."""
    a = np.asarray(allow_pred, dtype=int)
    bad = 1 - np.asarray(y_allow, dtype=int)
    n = a.size
    allowed = a == 1
    vetoed = a == 0
    rate = lambda mask: float(np.mean(bad[mask])) if mask.any() else float("nan")
    base = float(np.mean(bad))
    allow_adv = rate(allowed)
    veto_adv = rate(vetoed)
    return {
        "n": int(n),
        "base_adverse_rate": round(base, 4),
        "allow_rate": round(float(np.mean(allowed)), 4),
        "adverse_rate_allowed": round(allow_adv, 4),
        "adverse_rate_vetoed": round(veto_adv, 4),
        # >0 means vetoed trades are genuinely riskier than allowed ones
        "adverse_avoided": round(veto_adv - allow_adv, 4),
        # of vetoed trades, fraction that were actually fine (false rejects)
        "false_reject_rate": round(
            float(np.mean(1 - bad[vetoed])) if vetoed.any() else float("nan"), 4
        ),
        "adverse_lift_allowed": round(
            allow_adv / base if base > 0 else float("nan"), 4
        ),
    }


def train_adverse_gate(
    *,
    config_dir: Path,
    predictions: Path,
    gate_features: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
    data_path: str = "data/parquet_data",
    max_holding_bars: int = 6,
    bar_minutes: int = 120,
    mae_bad_r: float = 1.5,
    features_gate_yaml: Path | None = None,
    min_abs_ic: float = 0.03,
    min_lift: float = 0.05,
    top_k: int = 8,
    reject_threshold: float = 0.55,
    train_end_date: str | None = "2025-10-01",
    long_entry: float | None = None,
    short_entry: float | None = None,
    entry_mode: str | None = None,
) -> dict[str, Any]:
    gdf_probe = pd.read_parquet(gate_features)
    candidate_pool = _load_candidate_pool(gdf_probe, features_gate_yaml)
    X_all, y = _build_training_rows(
        config_dir=config_dir,
        predictions=predictions,
        gate_features=gate_features,
        symbols=symbols,
        long_entry=long_entry,
        short_entry=short_entry,
        entry_mode=entry_mode,
        start_date=start_date,
        end_date=end_date,
        data_path=data_path,
        max_holding_bars=max_holding_bars,
        bar_minutes=bar_minutes,
        mae_bad_r=mae_bad_r,
        candidate_features=candidate_pool,
        train_end_date=train_end_date,
    )
    selected, ic_diag = select_gate_features_ic(
        X_all, y, min_abs_ic=min_abs_ic, min_lift=min_lift, top_k=top_k
    )
    if not selected:
        raise ValueError(
            "IC-prune selected 0 gate features — no candidate separates the adverse "
            f"label (min_abs_ic={min_abs_ic}, min_lift={min_lift}). Diagnostics: "
            f"{ic_diag[:8]}"
        )
    # train only on selected features, dropping rows with NaN among them
    Xs = X_all[selected].apply(pd.to_numeric, errors="coerce")
    mask = Xs.notna().all(axis=1).to_numpy()
    Xs, ys = Xs[mask].to_numpy(dtype=float), y[mask]
    clf = train_tree_gate(
        Xs,
        ys,
        gate_name="adverse_excursion_gate",
        feature_names=selected,
        cfg=TreeGateTrainConfig(max_depth=4, min_samples_leaf=50),
    )
    allow_pred = gate_predict(clf, Xs)
    metrics = _real_gate_metrics(allow_pred, ys)
    artifact = export_tree_gate_artifact(
        clf=clf,
        gate_name="adverse_excursion_gate",
        feature_names=selected,
        metrics=metrics,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # JSON rules/metrics/meta + the pickled sklearn classifier the live overlay
    # loads via `gate_model` (matches the proven track_b gate layout).
    save_tree_gate_artifact(artifact, out_dir=out)
    model_path = out / "model.joblib"
    joblib.dump(clf, model_path)
    overlay = {
        "enabled": True,
        "gate_model": str(model_path.relative_to(PROJECT_ROOT)),
        "gate_feature_names": selected,
        "reject_if_prob_bad_gt": reject_threshold,
        "train_end_date": train_end_date,
    }
    (out / "gate_overlay.yaml").write_text(
        yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "n_samples": int(len(ys)),
        "candidate_pool_size": len(candidate_pool),
        "selected_features": selected,
        "ic_prune": ic_diag,
        "metrics": metrics,
        "model_path": str(model_path),
        "overlay": overlay,
    }
    (out / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "ic_prune"}, indent=2))
    print("IC-prune (top candidates):")
    for d in ic_diag[:12]:
        print(f"  {d['feature']:<32} ic={d['ic']:+.4f} lift={d['lift']:+.4f}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True, help="score parquet (entries+side)")
    ap.add_argument(
        "--gate-features",
        required=True,
        help="WIDE prepared features parquet (candidate gate feature pool)",
    )
    ap.add_argument(
        "--features-gate-yaml",
        default=None,
        help="optional features_gate.yaml declaring the allowed candidate pool",
    )
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--mae-bad-r", type=float, default=1.5)
    ap.add_argument("--min-abs-ic", type=float, default=0.03)
    ap.add_argument("--min-lift", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--reject-threshold", type=float, default=0.55)
    ap.add_argument(
        "--train-end-date",
        default="2025-10-01",
        help="Use entries strictly before this date for training (OOS gate)",
    )
    ap.add_argument(
        "--long-entry",
        type=float,
        default=None,
        help="Override long entry threshold (match the ranker's τ-scan value)",
    )
    ap.add_argument(
        "--short-entry",
        type=float,
        default=None,
        help="Override short entry threshold (match the ranker's τ-scan value)",
    )
    ap.add_argument(
        "--entry-mode",
        choices=("level", "cross"),
        default=None,
        help="Entry timing mode; must match event backtest (G3/H=3 uses level)",
    )
    args = ap.parse_args()

    def _abs(p: str | None) -> Path | None:
        if not p:
            return None
        pp = Path(p)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()

    train_adverse_gate(
        config_dir=_abs(args.config),
        predictions=_abs(args.predictions),
        gate_features=_abs(args.gate_features),
        features_gate_yaml=_abs(args.features_gate_yaml),
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=_abs(args.output_dir),
        data_path=args.data_path,
        mae_bad_r=args.mae_bad_r,
        min_abs_ic=args.min_abs_ic,
        min_lift=args.min_lift,
        top_k=args.top_k,
        reject_threshold=args.reject_threshold,
        train_end_date=args.train_end_date or None,
        long_entry=args.long_entry,
        short_entry=args.short_entry,
        entry_mode=args.entry_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
