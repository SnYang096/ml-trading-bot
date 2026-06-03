"""Run chop_grid backtest across config/market_segment.yaml windows.

Example::

    python scripts/experiment_chop_grid_market_segment.py \\
      --out-root results/chop_grid/experiments/segment_validate_20260602 \\
      -- \\
      --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \\
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \\
      --timeframe 2h --execution-timeframe 1min --no-maps
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKTEST = PROJECT_ROOT / "scripts" / "chop_grid_backtest.py"
DEFAULT_SEGMENT_PATH = PROJECT_ROOT / "config" / "market_segment.yaml"


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return list(argv[:i]), list(argv[i + 1 :])
    return list(argv), []


def _load_segments(path: Path, segment_ids: List[str] | None) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("segments") or []
    by_id = {str(r["id"]): r for r in rows if isinstance(r, dict) and r.get("id")}
    if segment_ids:
        missing = [s for s in segment_ids if s not in by_id]
        if missing:
            raise KeyError(f"unknown segment ids: {missing}; known: {sorted(by_id)}")
        return [by_id[s] for s in segment_ids]
    return list(by_id.values())


def _portfolio_dd_from_segments(segments: pd.DataFrame) -> float:
    if segments.empty or "end" not in segments.columns:
        return 0.0
    df = segments[["end", "pnl_per_capital"]].copy()
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.dropna(subset=["end"]).sort_values("end")
    if df.empty:
        return 0.0
    cum = df["pnl_per_capital"].cumsum().to_numpy(dtype=float)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def _per_symbol_rows(trades: pd.DataFrame, segment_id: str) -> List[Dict[str, Any]]:
    if trades.empty or "symbol" not in trades.columns:
        return []
    rows: List[Dict[str, Any]] = []
    for sym, grp in trades.groupby("symbol"):
        pnl = float(grp["pnl_per_capital"].sum())
        rows.append(
            {
                "segment": segment_id,
                "symbol": str(sym),
                "trades": len(grp),
                "return_pct": pnl * 100.0,
                "trade_win_rate": float((grp["pnl_pct"] > 0).mean()),
            }
        )
    return rows


def _run_segment(
    *,
    seg: Dict[str, Any],
    out_dir: Path,
    forward: List[str],
) -> Dict[str, Any]:
    seg_id = str(seg["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BACKTEST),
        *forward,
        "--start",
        str(seg["start_date"]),
        "--end",
        str(seg["end_date"]),
        "--out-dir",
        str(out_dir),
    ]
    print(f"\n=== {seg_id} ({seg['start_date']} → {seg['end_date']}) ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    summary = pd.read_csv(out_dir / "summary.csv")
    segments = pd.read_csv(out_dir / "grid_segments.csv")
    row = summary.iloc[0].to_dict()
    row["segment"] = seg_id
    row["segment_label"] = seg.get("label")
    row["start_date"] = seg["start_date"]
    row["end_date"] = seg["end_date"]
    row["portfolio_cum_dd"] = _portfolio_dd_from_segments(segments)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        default="results/chop_grid/experiments/segment_validate",
    )
    ap.add_argument(
        "--market-segment-path",
        default=str(DEFAULT_SEGMENT_PATH),
    )
    ap.add_argument("--segments", default="")
    args, forward = _split_argv(sys.argv[1:])
    ns = ap.parse_args(args)

    out_root = Path(ns.out_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    seg_path = Path(ns.market_segment_path)
    if not seg_path.is_absolute():
        seg_path = PROJECT_ROOT / seg_path
    seg_ids = [s.strip() for s in ns.segments.split(",") if s.strip()] or None
    segments = _load_segments(seg_path, seg_ids)

    rows: List[Dict[str, Any]] = []
    symbol_rows: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = str(seg["id"])
        seg_dir = out_root / seg_id
        row = _run_segment(seg=seg, out_dir=seg_dir, forward=forward)
        rows.append(row)
        trades = pd.read_csv(seg_dir / "grid_trades.csv")
        symbol_rows.extend(_per_symbol_rows(trades, seg_id))

    summary_df = pd.DataFrame(rows)
    symbol_df = pd.DataFrame(symbol_rows)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "segment_summary.csv"
    symbol_path = out_root / "segment_by_symbol.csv"
    summary_df.to_csv(summary_path, index=False)
    if not symbol_df.empty:
        symbol_df.to_csv(symbol_path, index=False)
    (out_root / "segment_summary.json").write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}")
    cols = [
        "segment",
        "return_pct",
        "return_pct_timeline",
        "return_pct_eq_mean",
        "return_pct_pooled",
        "max_drawdown_portfolio",
        "daily_sharpe",
        "n_symbols",
        "segment_win_rate",
        "worst_segment",
        "portfolio_cum_dd",
        "segments",
        "trades",
    ]
    show = [c for c in cols if c in summary_df.columns]
    print("\n=== Segment summary ===")
    print(summary_df[show].to_string(index=False))


if __name__ == "__main__":
    main()
