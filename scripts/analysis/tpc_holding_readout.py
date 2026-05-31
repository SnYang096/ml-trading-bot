#!/usr/bin/env python3
"""Summarize TPC event_backtest runs: holding, exit reasons, direction/EMA conflict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _load_capital_report(run_dir: Path) -> Dict[str, Any]:
    cap_path = run_dir / "capital_report.json"
    if not cap_path.is_file():
        return {}
    return json.loads(cap_path.read_text(encoding="utf-8"))


def _find_trades_csv(run_dir: Path) -> Optional[Path]:
    for name in ("event_trades_tpc.csv", "event_trades.csv"):
        p = run_dir / name
        if p.is_file():
            return p
    matches = list(run_dir.glob("event_trades_*.csv"))
    return matches[0] if matches else None


def summarize_run(
    run_dir: Path, *, initial_capital: float = 10_000.0
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    trades_path = _find_trades_csv(run_dir)
    cap = _load_capital_report(run_dir)

    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "trades_csv": str(trades_path) if trades_path else None,
        "trades": 0,
        "cagr_pct": None,
        "total_r": cap.get("total_r"),
        "final_capital": cap.get("final_capital"),
        "profit_usd": cap.get("estimated_profit_usd"),
        "exit_reason_counts": {},
        "structural_ema1200_n": 0,
        "structural_ema1200_pct": 0.0,
        "structural_median_hold_min": None,
        "structural_le2min_pct": None,
        "direction_ema_conflict_pct": None,
        "median_notional_over_equity": None,
        "median_bars_held": None,
    }

    if trades_path is None:
        return out

    df = pd.read_csv(trades_path)
    if df.empty:
        return out

    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values("exit_time")

    df["hold_minutes"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
    equity_before = initial_capital + df["pnl_usd_realized"].fillna(0).cumsum().shift(
        1
    ).fillna(0)
    df["equity_at_entry"] = equity_before.replace(0, initial_capital)
    df["notional_over_equity"] = df["notional_usdt"] / df["equity_at_entry"]

    out["trades"] = len(df)
    if cap.get("cagr") is not None:
        out["cagr_pct"] = round(float(cap["cagr"]) * 100, 2)

    exit_counts = df["exit_reason"].value_counts().to_dict()
    out["exit_reason_counts"] = {str(k): int(v) for k, v in exit_counts.items()}

    structural = df[df["exit_reason"] == "structural_exit_ema1200"]
    out["structural_ema1200_n"] = len(structural)
    out["structural_ema1200_pct"] = round(100.0 * len(structural) / len(df), 1)
    if len(structural):
        out["structural_median_hold_min"] = round(
            float(structural["hold_minutes"].median()), 1
        )
        out["structural_le2min_pct"] = round(
            100.0 * (structural["hold_minutes"] <= 2).sum() / len(structural), 1
        )

    if "feat_ema_1200_position" in df.columns:
        conflict = ((df["side"] == "LONG") & (df["feat_ema_1200_position"] < 0)) | (
            (df["side"] == "SHORT") & (df["feat_ema_1200_position"] > 0)
        )
        out["direction_ema_conflict_pct"] = round(100.0 * conflict.sum() / len(df), 1)

    out["median_notional_over_equity"] = round(
        float(df["notional_over_equity"].median()), 4
    )
    if "bars_held" in df.columns:
        out["median_bars_held"] = int(df["bars_held"].median())

    return out


def _format_row(label: str, s: Dict[str, Any]) -> str:
    struct_pct = s.get("structural_ema1200_pct", 0)
    struct_hold = s.get("structural_median_hold_min")
    struct_le2 = s.get("structural_le2min_pct")
    conflict = s.get("direction_ema_conflict_pct")
    cagr = s.get("cagr_pct")
    trades = s.get("trades", 0)
    total_r = s.get("total_r")
    nom = s.get("median_notional_over_equity")

    parts = [
        f"| {label} | {trades} | {cagr if cagr is not None else '—'}% | "
        f"{total_r if total_r is not None else '—'} | {struct_pct}% | "
        f"{struct_hold if struct_hold is not None else '—'} | "
        f"{struct_le2 if struct_le2 is not None else '—'}% | "
        f"{conflict if conflict is not None else '—'}% | {nom if nom is not None else '—'} |"
    ]
    return parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPC holding / exit readout for backtest runs"
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Directories containing event_trades_*.csv and capital_report.json",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels (same count as run_dirs); default = dir name",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Append summary table rows to markdown file (creates header if missing)",
    )
    args = parser.parse_args()

    labels: List[str] = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise SystemExit("--labels count must match run_dirs count")

    rows: List[str] = []
    for label, run_dir in zip(labels, args.run_dirs):
        summary = summarize_run(Path(run_dir))
        row = _format_row(label, summary)
        rows.append(row)
        print(f"\n=== {label} ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "| variant | trades | CAGR | total_r | structural% | "
            "struct_med_min | struct_≤2min% | dir/EMA conflict% | med_notional/eq |\n"
            "|---------|--------|------|---------|-------------|"
            "----------------|----------------|-------------------|-----------------|"
        )
        section = "\n".join([header] + rows) + "\n"
        if md_path.is_file():
            existing = md_path.read_text(encoding="utf-8")
            md_path.write_text(existing.rstrip() + "\n\n" + section, encoding="utf-8")
        else:
            md_path.write_text(
                "# TPC direction-align retest\n\n" + section,
                encoding="utf-8",
            )
        print(f"\nWrote markdown table to {md_path}")


if __name__ == "__main__":
    main()
