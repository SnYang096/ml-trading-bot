#!/usr/bin/env python3
"""
Pure rule Router (3-action): NO_TRADE/MEAN/TREND based only on nnmultihead heads.

Input:
  - A preds parquet/csv file OR a directory containing multiple per-symbol preds_*.parquet
    produced by `mlbot nnmultihead predict` (multi-symbol mode).

Output:
  - Single parquet/csv with columns: symbol,timestamp,mode,mode_action + diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    QualityScoreConfig,
    compute_mode_3action_regime_aware,
)  # noqa: E402


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_pred_files(preds_path: Path) -> List[Path]:
    if preds_path.is_dir():
        files = sorted(preds_path.glob("preds_*.parquet"))
        if not files:
            files = sorted(preds_path.glob("*.parquet"))
        return files
    return [preds_path]


def _ensure_timestamp_col(df: pd.DataFrame, *, col: str = "timestamp") -> pd.DataFrame:
    if df.index.name == col:
        df = df.reset_index()
    if col in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out[col] = out.index
        return out
    return df


def _load_feature_store(
    *,
    feature_store_root: Path,
    layer: str,
    symbol: str,
    timeframe: str,
    columns: List[str],
) -> pd.DataFrame:
    sym_dir = feature_store_root / layer / symbol / timeframe
    if not sym_dir.exists():
        raise FileNotFoundError(f"FeatureStore path not found: {sym_dir}")
    frames = []
    for p in sorted(sym_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        if "timestamp" not in df.columns:
            if df.index.name == "timestamp":
                df = df.reset_index()
            else:
                raise ValueError(f"timestamp column missing in {p}")
        keep = ["timestamp"] + [c for c in columns if c in df.columns]
        frames.append(df[keep])
    if not frames:
        raise ValueError(f"No featurestore parquet found under {sym_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rule mode 3-action router based on nnmultihead heads."
    )
    p.add_argument(
        "--preds",
        required=True,
        help="Preds file (.parquet/.csv) or directory of per-symbol parquet files",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Optional model.pt to infer whether preds are log1p targets",
    )
    p.add_argument(
        "--preds-in-log1p",
        default=None,
        choices=["yes", "no"],
        help="Override preds space (yes=log1p)",
    )
    p.add_argument(
        "--calibration-json",
        default=None,
        help="Optional router calibration JSON (dir_prob + linear bias for heads).",
    )
    p.add_argument("--output", required=True, help="Output path (.parquet or .csv)")
    p.add_argument(
        "--router-mode",
        default="regime_quality",
        choices=["regime_quality"],
        help="regime_quality=Physics regime + NN quality",
    )
    p.add_argument("--feature-store-root", default=None)
    p.add_argument("--feature-store-layer", default=None)
    p.add_argument("--timeframe", default="240T")

    # Threshold overrides (ATR units after inverse-transform)
    p.add_argument("--mfe-min", type=float, default=None)
    p.add_argument("--eff-min", type=float, default=None)
    p.add_argument("--dir-conf-trend-min", type=float, default=None)
    p.add_argument("--mfe-trend-min", type=float, default=None)
    p.add_argument("--ttm-trend-min", type=float, default=None)
    p.add_argument("--eff-mean-min", type=float, default=None)
    p.add_argument("--ttm-mean-max", type=float, default=None)
    p.add_argument(
        "--trend-confirm-mode",
        default=None,
        choices=["and", "or"],
        help="Trend confirm logic: and (legacy) or (dir_conf AND (mfe OR ttm)).",
    )
    # Quality thresholds (B/C)
    p.add_argument("--quality-trend-min", type=float, default=None)
    p.add_argument("--quality-mean-min", type=float, default=None)
    p.add_argument("--quality-te-min", type=float, default=None)
    p.add_argument(
        "--quality-use-dir-conf",
        choices=["yes", "no"],
        default=None,
        help="Use eff*dir_conf (yes) or eff only (no).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preds_path = Path(args.preds)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds_in_log1p = True
    if args.preds_in_log1p is not None:
        preds_in_log1p = bool(args.preds_in_log1p == "yes")
    elif args.model:
        payload = torch.load(args.model, map_location="cpu")
        meta = payload.get("meta") or {}
        ds_cfg = meta.get("dataset_cfg") or {}
        preds_in_log1p = bool(ds_cfg.get("log1p_targets", True))
    calibration = None
    if args.calibration_json:
        calibration = json.loads(
            Path(args.calibration_json).read_text(encoding="utf-8")
        )

    cfg0 = Rule3ActionConfig()
    overrides = {
        "mfe_min": args.mfe_min,
        "eff_min": args.eff_min,
        "dir_conf_trend_min": args.dir_conf_trend_min,
        "mfe_trend_min": args.mfe_trend_min,
        "ttm_trend_min": args.ttm_trend_min,
        "eff_mean_min": args.eff_mean_min,
        "ttm_mean_max": args.ttm_mean_max,
        "trend_confirm_mode": args.trend_confirm_mode,
    }
    merged_cfg = {
        **cfg0.__dict__,
        **{k: v for k, v in overrides.items() if v is not None},
    }
    cfg = Rule3ActionConfig(**merged_cfg)

    score_cfg0 = QualityScoreConfig()
    score_overrides = {
        "quality_trend_min": args.quality_trend_min,
        "quality_mean_min": args.quality_mean_min,
        "quality_te_min": args.quality_te_min,
        "use_dir_conf": (
            None
            if args.quality_use_dir_conf is None
            else bool(args.quality_use_dir_conf == "yes")
        ),
    }
    merged_score_cfg = {
        **score_cfg0.__dict__,
        **{k: v for k, v in score_overrides.items() if v is not None},
    }
    score_cfg = QualityScoreConfig(**merged_score_cfg)

    def _process_symbol(df_sym: pd.DataFrame, sym: str) -> pd.DataFrame:
        df_sym = _ensure_timestamp_col(df_sym, col="timestamp")
        df_sym["symbol"] = sym
        if not args.feature_store_root or not args.feature_store_layer:
            raise ValueError(
                "--feature-store-root and --feature-store-layer are required for regime_quality mode"
            )
        from src.time_series_model.rule.regime import PhysicsRegimeConfig

        regime_cfg = PhysicsRegimeConfig()
        feature_cols = [
            regime_cfg.atr_col,
            regime_cfg.atr_percentile_col,
            regime_cfg.high_col,
            regime_cfg.low_col,
            regime_cfg.close_col,
        ]
        feature_cols = [c for c in feature_cols if c]
        missing_cols = [c for c in feature_cols if c not in df_sym.columns]
        if missing_cols:
            feats = _load_feature_store(
                feature_store_root=Path(args.feature_store_root),
                layer=str(args.feature_store_layer),
                symbol=sym,
                timeframe=str(args.timeframe),
                columns=missing_cols,
            )
            df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"])
            if df_sym.index.name == "timestamp":
                df_sym = df_sym.reset_index(drop=True)
            if feats.index.name == "timestamp":
                feats = feats.reset_index(drop=True)
            df_sym = pd.merge(df_sym, feats, on="timestamp", how="left")
        mode_df = compute_mode_3action_regime_aware(
            df_sym,
            rule_cfg=cfg,
            score_cfg=score_cfg,
            preds_in_log1p=preds_in_log1p,
            calibration=calibration,
            use_physics_regime=True,
        )
        merged = df_sym[["symbol"]].copy()
        if "timestamp" in df_sym.columns:
            merged["timestamp"] = df_sym["timestamp"]
        merged = merged.join(mode_df)
        return merged

    parts = []
    for f in _collect_pred_files(preds_path):
        df = _ensure_timestamp_col(_read_any(f), col="timestamp")
        if "symbol" not in df.columns:
            sym = f.stem.replace("preds_", "")
            df["symbol"] = sym
        if df["symbol"].nunique() > 1:
            for sym, g in df.groupby("symbol", sort=False):
                parts.append(_process_symbol(g.copy(), str(sym)))
        else:
            sym = str(df["symbol"].iloc[0])
            parts.append(_process_symbol(df.copy(), sym))

    out = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()
    if out_path.suffix.lower() == ".parquet":
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)

    print("✅ Saved:", out_path)
    print("   preds_in_log1p:", preds_in_log1p)
    print(
        "   mode counts:",
        json.dumps(
            out["mode"].value_counts().to_dict() if "mode" in out.columns else {},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
