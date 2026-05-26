#!/usr/bin/env python3
"""Create minimal labeled feature parquets for ABC validation smoke (no live changes)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_tpc_parquet(out: Path, n: int = 8000) -> Path:
    rng = np.random.default_rng(42)
    dt = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    ema = rng.uniform(-0.25, 0.25, size=n)
    chop = rng.uniform(0, 1, size=n)
    pullback = rng.uniform(0, 1, size=n)
    success = ((ema > 0.08) & (chop < 0.4) & (pullback < 0.7)).astype(int)
    df = pd.DataFrame(
        {
            "datetime": dt,
            "symbol": "BTCUSDT",
            "ema_1200_position": ema,
            "tpc_semantic_chop": chop,
            "tpc_pullback_depth": pullback,
            "vol_persistence": rng.uniform(0, 1, size=n),
            "vol_leverage_asymmetry": rng.normal(0, 0.1, size=n),
            "macd_atr": rng.normal(0, 1, size=n),
            "bb_width_normalized_pct": rng.uniform(0, 1, size=n),
            "cvd_short_normalized": rng.normal(0, 1, size=n),
            "vpin_short": rng.uniform(0, 1, size=n),
            "hurst_short": rng.uniform(0, 1, size=n),
            "atr_percentile": rng.uniform(0, 1, size=n),
            "trend_confidence": rng.uniform(0, 1, size=n),
            "ema_1200_slope_10": rng.normal(0, 0.002, size=n),
            "hurst_long": rng.uniform(0, 1, size=n),
            "success_no_rr_extreme": success,
            "forward_rr": rng.normal(0, 0.5, size=n),
            "forward_rr_3": rng.normal(0, 0.3, size=n),
            "forward_rr_20": rng.normal(0, 0.8, size=n),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def make_chop_features_from_segments(segments_csv: Path, out: Path) -> Path:
    seg = pd.read_csv(segments_csv)
    seg["start"] = pd.to_datetime(seg["start"], utc=True)
    rows = []
    for _, r in seg.head(200).iterrows():
        rows.append(
            {
                "datetime": r["start"],
                "symbol": r["symbol"],
                "bpc_semantic_chop": float(r.get("entry_chop", 0.5)),
                "tpc_semantic_chop": float(r.get("median_chop", 0.5)) * 0.9,
                "chop_not_box": 0.0 if r.get("entry_box_prefilter") else 1.0,
            }
        )
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--segments-csv",
        default="results/chop_grid/check_current_20240101_20260511/grid_segments.csv",
    )
    args = p.parse_args()

    tpc_out = PROJECT_ROOT / "results/validation_smoke/tpc/features_labeled.parquet"
    make_tpc_parquet(tpc_out)
    print(f"wrote {tpc_out}")

    seg_path = Path(args.segments_csv)
    if not seg_path.is_absolute():
        seg_path = (PROJECT_ROOT / seg_path).resolve()
    if seg_path.exists():
        chop_out = (
            PROJECT_ROOT / "results/validation_smoke/chop_grid/features_labeled.parquet"
        )
        make_chop_features_from_segments(seg_path, chop_out)
        print(f"wrote {chop_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
