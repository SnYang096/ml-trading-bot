#!/usr/bin/env python3
"""Compare trend_scalp MTM vs regime_only per-symbol win rate and returns.

Reads ``segment_by_symbol.csv`` from two experiment output directories and
produces a side-by-side comparison table.

Usage::

    python scripts/compare_mtm_vs_regime.py \\
      --regime-only results/trend_scalp/experiments/mtm_vs_regime_20260613/regime_only \\
      --mtm results/trend_scalp/experiments/mtm_vs_regime_20260613/mtm \\
      --out results/trend_scalp/experiments/mtm_vs_regime_20260613/comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_symbol_csv(root: Path) -> pd.DataFrame:
    path = root / "segment_by_symbol.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_csv(path)
    df["source"] = root.name
    return df


def _load_summary_csv(root: Path) -> pd.DataFrame:
    path = root / "segment_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime-only", required=True, help="regime_only output dir")
    ap.add_argument("--mtm", required=True, help="mtm output dir")
    ap.add_argument("--out", default="comparison.csv", help="output CSV path")
    args = ap.parse_args()

    ro_root = Path(args.regime_only)
    mtm_root = Path(args.mtm_root) if hasattr(args, "mtm_root") else Path(args.mtm)

    if not ro_root.is_absolute():
        ro_root = PROJECT_ROOT / ro_root
    if not mtm_root.is_absolute():
        mtm_root = PROJECT_ROOT / mtm_root
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    # ── Per-symbol breakdown ──
    ro_sym = _load_symbol_csv(ro_root)
    mtm_sym = _load_symbol_csv(mtm_root)

    merged = ro_sym.merge(
        mtm_sym,
        on=["segment", "symbol"],
        how="outer",
        suffixes=("_regime_only", "_mtm"),
    )
    merged["return_pct_diff"] = merged["return_pct_mtm"].fillna(0) - merged[
        "return_pct_regime_only"
    ].fillna(0)
    merged["win_rate_diff"] = merged["trade_win_rate_mtm"].fillna(0) - merged[
        "trade_win_rate_regime_only"
    ].fillna(0)

    # ── Aggregate by symbol ──
    agg_cols: List[Dict[str, Any]] = []
    for sym, grp in merged.groupby("symbol"):
        ro_pnl = float(grp["return_pct_regime_only"].sum())
        mtm_pnl = float(grp["return_pct_mtm"].sum())
        ro_trades = int(grp["trades_regime_only"].sum())
        mtm_trades = int(grp["trades_mtm"].sum())
        ro_wins = int(
            (grp["trades_regime_only"] * grp["trade_win_rate_regime_only"]).sum()
        )
        mtm_wins = int((grp["trades_mtm"] * grp["trade_win_rate_mtm"]).sum())
        agg_cols.append(
            {
                "symbol": sym,
                "return_pct_regime_only": round(ro_pnl, 2),
                "return_pct_mtm": round(mtm_pnl, 2),
                "return_pct_diff": round(mtm_pnl - ro_pnl, 2),
                "trades_regime_only": ro_trades,
                "trades_mtm": mtm_trades,
                "win_rate_regime_only": round(ro_wins / max(ro_trades, 1), 3),
                "win_rate_mtm": round(mtm_wins / max(mtm_trades, 1), 3),
                "win_rate_diff": round(
                    mtm_wins / max(mtm_trades, 1) - ro_wins / max(ro_trades, 1), 3
                ),
            }
        )
    agg_df = pd.DataFrame(agg_cols)

    # ── Segment-level summary ──
    ro_seg = _load_summary_csv(ro_root)
    mtm_seg = _load_summary_csv(mtm_root)
    seg_merged = ro_seg.merge(
        mtm_seg,
        on="segment",
        how="outer",
        suffixes=("_regime_only", "_mtm"),
    )

    print("=" * 70)
    print("Per-symbol aggregate (across all 4 segments)")
    print("=" * 70)
    print(
        agg_df.sort_values("symbol")[
            [
                "symbol",
                "return_pct_regime_only",
                "return_pct_mtm",
                "return_pct_diff",
                "win_rate_regime_only",
                "win_rate_mtm",
                "win_rate_diff",
                "trades_regime_only",
                "trades_mtm",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("Segment-level summary")
    print("=" * 70)
    seg_cols = [
        "segment",
        "return_pct_timeline_regime_only",
        "return_pct_timeline_mtm",
        "daily_sharpe_regime_only",
        "daily_sharpe_mtm",
        "segment_win_rate_regime_only",
        "segment_win_rate_mtm",
        "risk_stop_rate_regime_only",
        "risk_stop_rate_mtm",
        "trades_regime_only",
        "trades_mtm",
    ]
    show = [c for c in seg_cols if c in seg_merged.columns]
    print(seg_merged[show].to_string(index=False))

    # ── Per-symbol × segment detail ──
    detail_cols = [
        "segment",
        "symbol",
        "trades_regime_only",
        "trades_mtm",
        "return_pct_regime_only",
        "return_pct_mtm",
        "return_pct_diff",
        "trade_win_rate_regime_only",
        "trade_win_rate_mtm",
        "win_rate_diff",
    ]
    detail = merged[detail_cols].sort_values(["symbol", "segment"])
    detail_path = out_path.parent / "comparison_detail.csv"
    detail.to_csv(detail_path, index=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()
