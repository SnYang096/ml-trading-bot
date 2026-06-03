#!/usr/bin/env python3
"""Run chop_grid fee × exec × universe reconciliation grid.

Example::

    python scripts/experiment_chop_grid_reconcile_grid.py \\
      --manifest config/experiments/20260603_chop_grid_reconcile_grid/grid_oos.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.multileg_portfolio_metrics import portfolio_metrics_from_trades

BACKTEST = PROJECT_ROOT / "scripts" / "chop_grid_backtest.py"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "config/experiments/20260603_chop_grid_reconcile_grid/grid_oos.yaml"
)


def _load_manifest(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sym_map = {
        "symbols_5": str(data.get("symbols_5", "")),
        "symbols_6": str(data.get("symbols_6", "")),
    }
    cells = data.get("cells") or []
    out_cells: List[Dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        sym_key = str(cell.get("symbols", "symbols_5"))
        symbols = sym_map.get(sym_key, sym_key)
        out_cells.append({**cell, "symbols_resolved": symbols})
    data["cells"] = out_cells
    return data


def _fee_args(fee_bps: float) -> List[str]:
    fb = str(float(fee_bps))
    # Match calibrate_roll.default.yaml: stress tier applies funding at same bps.
    funding = fb if float(fee_bps) >= 20.0 else "0"
    return [
        "--fee-bps",
        fb,
        "--maker-fee-bps",
        fb,
        "--taker-fee-bps",
        fb,
        "--forced-exit-slippage-bps",
        fb,
        "--funding-cost-bps-per-8h",
        funding,
    ]


def _run_cell(
    *,
    cell: Dict[str, Any],
    manifest: Dict[str, Any],
    out_root: Path,
) -> Dict[str, Any]:
    cell_id = str(cell["id"])
    out_dir = out_root / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BACKTEST),
        "--config",
        str(manifest.get("config")),
        "--symbols",
        str(cell["symbols_resolved"]),
        "--timeframe",
        "2h",
        "--start",
        str(manifest["start_date"]),
        "--end",
        str(manifest["end_date"]),
        "--out-dir",
        str(out_dir),
        "--initial-capital",
        "10000",
        "--no-maps",
        *_fee_args(float(cell.get("fee_bps", 4))),
    ]
    exec_tf = cell.get("execution_timeframe")
    if exec_tf:
        cmd.extend(["--execution-timeframe", str(exec_tf)])

    print(f"\n=== {cell_id}: {cell.get('label', '')} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    trades_path = out_dir / "grid_trades.csv"
    segments_path = out_dir / "grid_segments.csv"
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    segments = pd.read_csv(segments_path) if segments_path.exists() else pd.DataFrame()
    metrics = portfolio_metrics_from_trades(trades)
    replenish = 0
    if not segments.empty and "replenish_trades" in segments.columns:
        replenish = int(
            pd.to_numeric(segments["replenish_trades"], errors="coerce").fillna(0).sum()
        )
    row: Dict[str, Any] = {
        "cell_id": cell_id,
        "label": cell.get("label"),
        "execution_timeframe": exec_tf or "2h",
        "fee_bps": float(cell.get("fee_bps", 4)),
        "symbols": str(cell["symbols_resolved"]),
        "n_symbols": metrics.get("n_symbols", 0),
        "return_pct": metrics.get("return_pct", 0.0),
        "return_pct_timeline": metrics.get("return_pct_timeline", 0.0),
        "return_pct_eq_mean": metrics.get("return_pct_eq_mean", 0.0),
        "return_pct_pooled": metrics.get("return_pct_pooled", 0.0),
        "max_drawdown_portfolio": metrics.get("max_drawdown_portfolio", 0.0),
        "daily_sharpe": metrics.get("daily_sharpe", 0.0),
        "trades": len(trades),
        "segments": len(segments),
        "replenish_trades": replenish,
        "segment_win_rate": (
            float((segments["pnl_per_capital"] > 0).mean())
            if not segments.empty and "pnl_per_capital" in segments.columns
            else float("nan")
        ),
        "worst_segment": (
            float(segments["pnl_per_capital"].min())
            if not segments.empty and "pnl_per_capital" in segments.columns
            else float("nan")
        ),
        "out_dir": str(out_dir),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Grid manifest YAML",
    )
    ap.add_argument(
        "--cells",
        default="",
        help="Comma-separated cell ids (default: all in manifest)",
    )
    ap.add_argument(
        "--out-root",
        default="",
        help="Override manifest output_root",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = _load_manifest(manifest_path)

    out_root = Path(
        args.out_root or manifest.get("output_root", "results/chop_grid/reconcile")
    )
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    cell_filter = {s.strip() for s in args.cells.split(",") if s.strip()} or None
    rows: List[Dict[str, Any]] = []
    for cell in manifest["cells"]:
        cell_id = str(cell.get("id", ""))
        if cell_filter and cell_id not in cell_filter:
            continue
        rows.append(_run_cell(cell=cell, manifest=manifest, out_root=out_root))

    summary = pd.DataFrame(rows)
    summary_path = out_root / "reconcile_summary.csv"
    summary.to_csv(summary_path, index=False)
    (out_root / "reconcile_summary.json").write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}")
    if not summary.empty:
        cols = [
            "cell_id",
            "execution_timeframe",
            "fee_bps",
            "n_symbols",
            "return_pct",
            "return_pct_pooled",
            "trades",
            "segment_win_rate",
        ]
        show = [c for c in cols if c in summary.columns]
        print("\n=== Reconcile summary ===")
        print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()
