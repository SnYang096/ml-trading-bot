#!/usr/bin/env python3
"""Export vectorbt tree trades aligned with deploy frozen τ (holdout predictions)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.tree_holdout_tau_rr_scan import (  # noqa: E402
    _prepare_df,
    _run_bt,
)
from src.time_series_model.strategy_config.loader import (
    StrategyConfigLoader,
)  # noqa: E402


def _atr_at_index(df: pd.DataFrame, ts: pd.Timestamp) -> float:
    if ts not in df.index:
        sub = df.loc[df.index <= ts]
        if sub.empty:
            return float("nan")
        row = sub.iloc[-1]
    else:
        row = df.loc[ts]
    atr = row.get("atr") if hasattr(row, "get") else row["atr"]
    try:
        v = float(atr)
    except (TypeError, ValueError):
        return float("nan")
    return v if v == v and v > 0 else float("nan")


def _vector_trade_rows(
    trades: list[dict[str, Any]],
    *,
    symbol: str,
    df: pd.DataFrame,
    archetype: str,
    initial_r: float = 4.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in trades:
        entry_ts = pd.Timestamp(t.get("Entry Timestamp"))
        exit_ts = pd.Timestamp(t.get("Exit Timestamp"))
        direction = str(t.get("Direction", "")).lower()
        side = "LONG" if "long" in direction else "SHORT"
        entry_px = float(t.get("Avg Entry Price", 0) or 0)
        exit_px = float(t.get("Avg Exit Price", 0) or 0)
        atr = _atr_at_index(df, entry_ts)
        if side == "LONG":
            move = exit_px - entry_px
        else:
            move = entry_px - exit_px
        risk_unit = atr * initial_r if np.isfinite(atr) and atr > 0 else np.nan
        pnl_r = (
            float(move / risk_unit)
            if np.isfinite(risk_unit) and risk_unit > 0
            else float(t.get("Return", 0) or 0) / 100.0
        )
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_time": str(entry_ts),
                "exit_time": str(exit_ts),
                "entry_price": round(entry_px, 6),
                "exit_price": round(exit_px, 6),
                "pnl_r": round(pnl_r, 4),
                "exit_reason": "vectorbt_exit",
                "archetype": archetype,
            }
        )
    return rows


def export_vector_trades(
    *,
    config: str | Path,
    predictions: str | Path,
    output_csv: str | Path,
    summary_json: str | Path | None = None,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fixed_quantile: float | None = None,
    long_entry_threshold: float | None = None,
    short_entry_threshold: float | None = None,
    entry_mode: str = "cross",
    per_symbol: bool = True,
    filter_split: str | None = "holdout",
) -> Path:
    cfg_dir = Path(config)
    if not cfg_dir.is_absolute():
        cfg_dir = (PROJECT_ROOT / cfg_dir).resolve()
    strategy_config = StrategyConfigLoader(cfg_dir).load()
    archetype = strategy_config.name

    pred_path = Path(predictions)
    if not pred_path.is_absolute():
        pred_path = (PROJECT_ROOT / pred_path).resolve()
    df = _prepare_df(pred_path, start_date=start_date, end_date=end_date)
    if filter_split and "split" in df.columns:
        df = df.reset_index()
        df = df[df["split"].astype(str).str.lower() == filter_split.lower()].copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()

    sym_col = "_symbol" if "_symbol" in df.columns else "symbol"
    if symbols:
        syms = {s.strip().upper() for s in symbols}
        df = df[df[sym_col].astype(str).str.upper().isin(syms)].copy()

    use_abs = long_entry_threshold is not None or short_entry_threshold is not None
    all_rows: list[dict[str, Any]] = []
    stats_by_symbol: dict[str, Any] = {}

    if sym_col in df.columns:
        sym_list = sorted(df[sym_col].dropna().unique())
    else:
        sym_list = ["ALL"]
        df = df.copy()
        df[sym_col] = "ALL"

    for sym in sym_list:
        part = df[df[sym_col] == sym] if sym != "ALL" else df
        if part.empty:
            continue
        if use_abs:
            bt = _run_bt(
                part,
                strategy_config,
                long_entry_threshold=long_entry_threshold,
                short_entry_threshold=short_entry_threshold,
                entry_mode=entry_mode,
                debug=True,
                debug_trades_limit=None,
            )
        elif fixed_quantile is not None:
            bt = _run_bt(
                part,
                strategy_config,
                top_quantile=fixed_quantile,
                bottom_quantile=fixed_quantile,
                entry_mode=entry_mode,
                debug=True,
                debug_trades_limit=None,
            )
        else:
            raise ValueError("Provide fixed_quantile or long/short_entry_threshold")
        if not bt:
            continue
        stats_by_symbol[str(sym)] = {
            k: bt.get(k)
            for k in ("sharpe", "total_return_pct", "total_trades", "win_rate")
        }
        dbg = bt.get("debug") or {}
        trades = dbg.get("trades") or []
        all_rows.extend(
            _vector_trade_rows(
                trades,
                symbol=str(sym),
                df=part,
                archetype=archetype,
            )
        )

    out_csv = Path(output_csv)
    if not out_csv.is_absolute():
        out_csv = (PROJECT_ROOT / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "predictions": str(pred_path),
        "config": str(cfg_dir),
        "n_trades": len(out_df),
        "symbols": stats_by_symbol,
        "tau": {
            "fixed_quantile": fixed_quantile,
            "long_entry_threshold": long_entry_threshold,
            "short_entry_threshold": short_entry_threshold,
            "entry_mode": entry_mode,
            "per_symbol_quantile": bool(not use_abs and per_symbol),
        },
    }
    if summary_json:
        sj = Path(summary_json)
        if not sj.is_absolute():
            sj = (PROJECT_ROOT / sj).resolve()
        sj.parent.mkdir(parents=True, exist_ok=True)
        sj.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {out_csv} rows={len(out_df)}")
    return out_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--fixed-quantile", type=float, default=None)
    ap.add_argument("--long-entry-threshold", type=float, default=None)
    ap.add_argument("--short-entry-threshold", type=float, default=None)
    ap.add_argument("--entry-mode", default="cross")
    ap.add_argument("--filter-split", default="holdout")
    args = ap.parse_args()
    syms = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )
    export_vector_trades(
        config=args.config,
        predictions=args.predictions,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        symbols=syms,
        start_date=args.start_date,
        end_date=args.end_date,
        fixed_quantile=args.fixed_quantile,
        long_entry_threshold=args.long_entry_threshold,
        short_entry_threshold=args.short_entry_threshold,
        entry_mode=args.entry_mode,
        filter_split=args.filter_split or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
