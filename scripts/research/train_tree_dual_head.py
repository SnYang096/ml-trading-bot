#!/usr/bin/env python3
"""Train dual binary heads (P(long_win|x), P(short_win|x)) for fast_scalp experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.tree_holdout_tau_rr_scan import _prepare_df  # noqa: E402
from src.time_series_model.strategies.labels.long_short_win_label import (  # noqa: E402
    compute_long_short_win_labels,
)

META_COLS = {
    "_symbol",
    "symbol",
    "close",
    "high",
    "low",
    "open",
    "volume",
    "atr",
    "pred",
    "split",
    "label",
    "timestamp",
    "forward_rr",
    "long_win",
    "short_win",
    "score_long",
    "score_short",
}


def _load_model_features(config_dir: Path) -> list[str]:
    path = config_dir / "archetypes" / "model_features.yaml"
    if not path.exists():
        path = config_dir.parent / "fast_scalp" / "archetypes" / "model_features.yaml"
    if not path.exists():
        raise FileNotFoundError(f"model_features.yaml not found under {config_dir}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cols = raw.get("columns") or []
    return [str(c["feature"]) for c in cols if isinstance(c, dict) and c.get("feature")]


def _attach_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    rr_floor: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    sym_col = "_symbol" if "_symbol" in df.columns else "symbol"
    for sym, grp in df.groupby(sym_col):
        labels = compute_long_short_win_labels(
            grp,
            horizon=horizon,
            rr_floor=rr_floor,
            price_col="close",
            atr_col="atr",
        )
        out = grp.copy()
        out["long_win"] = labels["long_win"]
        out["short_win"] = labels["short_win"]
        parts.append(out)
    return pd.concat(parts).sort_index()


_LGBM_PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    "reg": {
        "n_estimators": 200,
        "learning_rate": 0.02,
        "num_leaves": 8,
        "max_depth": 3,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    },
}


def _fit_binary_head(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    profile: str = "default",
) -> Any:
    import lightgbm as lgb

    hp = dict(_LGBM_PROFILES.get(profile) or _LGBM_PROFILES["default"])
    pos = float(np.mean(y == 1))
    scale = float((1.0 - pos) / max(pos, 1e-6))
    clf = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale,
        random_state=seed,
        verbose=-1,
        **hp,
    )
    clf.fit(X, y)
    return clf


def _merge_ema_column(
    df: pd.DataFrame, ema_parquet: Path, *, ema_col: str = "ema_1200_position"
) -> pd.DataFrame:
    """Left-join slow macro column for regime-conditioned head training."""
    extra = pd.read_parquet(ema_parquet)
    if "timestamp" not in extra.columns:
        if "datetime" in extra.columns:
            extra = extra.rename(columns={"datetime": "timestamp"})
        else:
            extra = extra.reset_index().rename(
                columns={extra.index.name or "index": "timestamp"}
            )
    extra["timestamp"] = pd.to_datetime(extra["timestamp"], utc=True)
    sym_col = "_symbol" if "_symbol" in extra.columns else "symbol"
    if ema_col not in extra.columns:
        raise ValueError(f"{ema_col} not in {ema_parquet}")
    left = df.reset_index()
    if "timestamp" not in left.columns:
        left = left.rename(columns={"index": "timestamp"})
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    sym_left = "_symbol" if "_symbol" in left.columns else "symbol"
    if sym_col not in extra.columns and "_symbol" in extra.columns:
        sym_col = "_symbol"
    if sym_left != sym_col:
        extra = extra.rename(columns={sym_col: sym_left})
    keys = ["timestamp", sym_left]
    merged = left.merge(
        extra[keys + [ema_col]].drop_duplicates(keys),
        on=keys,
        how="left",
    )
    merged = merged.set_index("timestamp").sort_index()
    return merged


def train_dual_head(
    *,
    config_dir: Path,
    predictions: Path,
    symbols: list[str],
    output_dir: Path,
    horizon: int = 3,
    rr_floor: float = 0.30,
    train_end_date: str = "2026-01-01",
    score_start_date: str | None = None,
    profile: str = "default",
    ema_parquet: Path | None = None,
    ema_col: str = "ema_1200_position",
    long_ema_min: float | None = None,
    short_ema_max: float | None = None,
) -> dict[str, Any]:
    feature_names = _load_model_features(config_dir)
    df = pd.read_parquet(predictions)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    elif df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
    else:
        raise ValueError("predictions missing timestamp")
    if "_symbol" not in df.columns and "symbol" in df.columns:
        df["_symbol"] = df["symbol"]
    sym_set = {s.upper() for s in symbols}
    df = df[df["_symbol"].astype(str).str.upper().isin(sym_set)].copy()
    if ema_parquet is not None:
        df = _merge_ema_column(df, ema_parquet, ema_col=ema_col)
    df = _attach_labels(df, horizon=horizon, rr_floor=rr_floor)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"missing model features in predictions: {missing[:5]}")

    train_cut = pd.Timestamp(train_end_date, tz="UTC")
    train_df = df[df.index < train_cut].copy()
    score_cut = (
        pd.Timestamp(score_start_date, tz="UTC") if score_start_date else train_cut
    )
    holdout_df = df[df.index >= score_cut].copy()
    if train_df.empty or holdout_df.empty:
        raise ValueError(
            f"empty train ({len(train_df)}) or score ({len(holdout_df)}) "
            f"after temporal split at {train_end_date}"
        )

    def _xy(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
        sub = frame.dropna(subset=[target])
        if target == "long_win" and long_ema_min is not None and ema_col in sub.columns:
            sub = sub[
                pd.to_numeric(sub[ema_col], errors="coerce") >= float(long_ema_min)
            ]
        if (
            target == "short_win"
            and short_ema_max is not None
            and ema_col in sub.columns
        ):
            sub = sub[
                pd.to_numeric(sub[ema_col], errors="coerce") <= float(short_ema_max)
            ]
        X = sub[feature_names].astype(float).to_numpy()
        y = sub[target].astype(int).to_numpy()
        mask = np.isfinite(X).all(axis=1)
        return X[mask], y[mask]

    X_train_l, y_train_l = _xy(train_df, "long_win")
    X_train_s, y_train_s = _xy(train_df, "short_win")
    clf_long = _fit_binary_head(X_train_l, y_train_l, profile=profile)
    clf_short = _fit_binary_head(X_train_s, y_train_s, profile=profile)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import joblib

    long_path = out / "long_head.joblib"
    short_path = out / "short_head.joblib"
    joblib.dump(clf_long, long_path)
    joblib.dump(clf_short, short_path)

    score_df = df[df.index >= score_cut].copy()
    X_score = score_df[feature_names].astype(float).to_numpy()
    score_df["score_long"] = clf_long.predict_proba(X_score)[:, 1]
    score_df["score_short"] = clf_short.predict_proba(X_score)[:, 1]

    scored_path = out / "dual_head_scored.parquet"
    score_df.reset_index().rename(columns={"index": "timestamp"}).to_parquet(
        scored_path, index=False
    )

    export_cols = ["timestamp", "_symbol", "score_long", "score_short"]
    export_df = score_df.reset_index().rename(columns={"index": "timestamp"})
    export_df = export_df[export_cols].rename(columns={"_symbol": "symbol"})
    export_df["timestamp"] = pd.to_datetime(export_df["timestamp"], utc=True)
    export_df["symbol"] = export_df["symbol"].astype(str).str.upper()
    export_path = out / "dual_head_event_scores.parquet"
    export_df.to_parquet(export_path, index=False)

    summary = {
        "n_train_long": int(len(y_train_l)),
        "n_train_short": int(len(y_train_s)),
        "n_score_rows": int(len(score_df)),
        "profile": profile,
        "regime_train_filter": {
            "ema_col": ema_col,
            "long_ema_min": long_ema_min,
            "short_ema_max": short_ema_max,
        },
        "long_win_rate_train": float(np.mean(y_train_l)),
        "short_win_rate_train": float(np.mean(y_train_s)),
        "feature_names": feature_names,
        "horizon": horizon,
        "rr_floor": rr_floor,
        "train_end_date": train_end_date,
        "score_start_date": str(score_cut.date()),
        "models": {
            "long_head": str(long_path),
            "short_head": str(short_path),
        },
        "holdout_scores": str(export_path),
    }
    (out / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--rr-floor", type=float, default=0.30)
    ap.add_argument("--train-end-date", default="2026-01-01")
    ap.add_argument("--score-start-date", default=None)
    ap.add_argument(
        "--profile",
        choices=sorted(_LGBM_PROFILES),
        default="default",
        help="LGBM capacity profile (reg = stronger regularization)",
    )
    ap.add_argument(
        "--ema-parquet",
        default=None,
        help="Optional parquet with ema_1200_position for regime-conditioned training",
    )
    ap.add_argument("--ema-col", default="ema_1200_position")
    ap.add_argument(
        "--long-ema-min",
        type=float,
        default=None,
        help="Train long head only when ema >= this (e.g. 0.10)",
    )
    ap.add_argument(
        "--short-ema-max",
        type=float,
        default=None,
        help="Train short head only when ema <= this (e.g. -0.10)",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = (PROJECT_ROOT / cfg).resolve()
    preds = Path(args.predictions)
    if not preds.is_absolute():
        preds = (PROJECT_ROOT / preds).resolve()

    ema_p = Path(args.ema_parquet) if args.ema_parquet else None
    if ema_p and not ema_p.is_absolute():
        ema_p = (PROJECT_ROOT / ema_p).resolve()
    train_dual_head(
        config_dir=cfg,
        predictions=preds,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        output_dir=out,
        horizon=args.horizon,
        rr_floor=args.rr_floor,
        train_end_date=args.train_end_date,
        score_start_date=args.score_start_date,
        profile=str(args.profile),
        ema_parquet=ema_p,
        ema_col=str(args.ema_col),
        long_ema_min=args.long_ema_min,
        short_ema_max=args.short_ema_max,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
