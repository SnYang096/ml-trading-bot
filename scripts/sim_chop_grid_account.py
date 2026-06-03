#!/usr/bin/env python3
"""Real-account simulation for chop_grid dense-3L backtests.

Why this exists
---------------
The per-trade ``pnl_per_capital`` column normalises PnL to a *per-symbol capital
bucket* (each symbol assumed to hold ``2 * max_levels_per_side`` equal slices of
its slice of the account).  Pooled metrics then divide again by ``n_symbols``.
That answers "what is the return per unit of deployed capital", but it bakes in
the unrealistic assumption that every symbol and every grid level is funded
simultaneously.

A live account does NOT work that way: it holds ONE equity pool, posts a *fixed*
``unit_notional`` (USDT) per filled level, and at any instant only a handful of
levels across a few symbols are actually open.  This script reconstructs that
view by replaying entry/exit events on a single shared account:

  - +unit_notional gross at each fill (entry_time)
  - -unit_notional gross + realised PnL (pnl_pct * unit_notional) at exit_time

and reports the true account return %, the peak *concurrent* gross exposure
(both portfolio-wide and per-symbol), and the equity-curve max drawdown.

Usage
-----
    python scripts/sim_chop_grid_account.py \
        --root results/chop_grid/experiments/stack_ablation_20260604/E7_four_segment/dense_3l_live \
        --equity 10000 \
        --segments recent_6m_oos bear_2022 bull_2023_2024 recent_range_to_bear
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_segment(root: Path, seg: str) -> Optional[pd.DataFrame]:
    path = root / seg / "grid_trades.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def simulate(
    df: pd.DataFrame, *, equity: float, unit_notional: float
) -> Dict[str, float]:
    """Replay fills on a single shared account; fixed unit_notional per level."""
    events: List[tuple] = []
    for _, r in df.iterrows():
        events.append((r["entry_time"], "open", r["symbol"], 0.0))
        events.append(
            (r["exit_time"], "close", r["symbol"], r["pnl_pct"] * unit_notional)
        )
    # close before open at the same timestamp so freed capital can be reused
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1))

    gross = 0.0
    peak_gross = 0.0
    sym_gross: Dict[str, float] = {}
    peak_sym = 0.0
    cum_pnl = 0.0
    peak_eq = equity
    max_dd = 0.0
    for _, kind, sym, pnl in events:
        if kind == "open":
            gross += unit_notional
            sym_gross[sym] = sym_gross.get(sym, 0.0) + unit_notional
        else:
            gross -= unit_notional
            sym_gross[sym] = sym_gross.get(sym, 0.0) - unit_notional
            cum_pnl += pnl
        peak_gross = max(peak_gross, gross)
        if sym_gross:
            peak_sym = max(peak_sym, max(sym_gross.values()))
        eq = equity + cum_pnl
        peak_eq = max(peak_eq, eq)
        max_dd = min(max_dd, eq - peak_eq)
    return {
        "ret_pct": cum_pnl / equity * 100.0,
        "pnl_usd": cum_pnl,
        "peak_gross_pct": peak_gross / equity * 100.0,
        "peak_sym_pct": peak_sym / equity * 100.0,
        "max_dd_pct": max_dd / equity * 100.0,
        "n_trades": float(len(df)),
    }


def unit_for_dd_target(
    df: pd.DataFrame, *, equity: float, dd_target_pct: float, probe_unit: float = 400.0
) -> float:
    """DD scales linearly with unit_notional → solve for the target."""
    m = simulate(df, equity=equity, unit_notional=probe_unit)
    if m["max_dd_pct"] == 0.0:
        return float("inf")
    return probe_unit * (dd_target_pct / abs(m["max_dd_pct"]))


def _print_table(
    title: str, df: pd.DataFrame, equity: float, units: List[float]
) -> None:
    print(f"=== {title} (equity={equity:.0f}, single shared account) ===")
    print(
        f'{"unit":>6} {"ret%":>8} {"pnl_usd":>9} {"peakGross%":>11} '
        f'{"peakSym%":>9} {"maxDD%":>8} {"trades":>7}'
    )
    for u in units:
        m = simulate(df, equity=equity, unit_notional=u)
        print(
            f'{u:>6.0f} {m["ret_pct"]:>8.2f} {m["pnl_usd"]:>9.0f} '
            f'{m["peak_gross_pct"]:>11.1f} {m["peak_sym_pct"]:>9.1f} '
            f'{m["max_dd_pct"]:>8.3f} {m["n_trades"]:>7.0f}'
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument(
        "--segments",
        nargs="+",
        default=[
            "recent_6m_oos",
            "bear_2022",
            "bull_2023_2024",
            "recent_range_to_bear",
        ],
    )
    ap.add_argument(
        "--units", nargs="+", type=float, default=[200, 400, 800, 1200, 1600, 2400]
    )
    ap.add_argument(
        "--dd-target",
        type=float,
        default=1.0,
        help="drawdown budget %% for the aggressive-sizing solve",
    )
    args = ap.parse_args()

    loaded = {s: load_segment(args.root, s) for s in args.segments}
    loaded = {s: d for s, d in loaded.items() if d is not None}
    if not loaded:
        raise SystemExit(f"no grid_trades.csv found under {args.root}")

    for seg, df in loaded.items():
        _print_table(seg, df, args.equity, args.units)

    if len(loaded) > 1:
        combined = pd.concat(loaded.values(), ignore_index=True)
        _print_table("ALL SEGMENTS combined", combined, args.equity, args.units)

    print(f"=== unit_notional for {args.dd_target:.1f}% maxDD target ===")
    for seg, df in loaded.items():
        u = unit_for_dd_target(df, equity=args.equity, dd_target_pct=args.dd_target)
        m = simulate(df, equity=args.equity, unit_notional=u)
        print(
            f'  {seg:<22} unit~{u:>6.0f}  ret~{m["ret_pct"]:>6.2f}%  '
            f'peakGross~{m["peak_gross_pct"]:>5.0f}%  peakSym~{m["peak_sym_pct"]:>4.0f}%'
        )
    if len(loaded) > 1:
        combined = pd.concat(loaded.values(), ignore_index=True)
        u = unit_for_dd_target(
            combined, equity=args.equity, dd_target_pct=args.dd_target
        )
        m = simulate(combined, equity=args.equity, unit_notional=u)
        print(
            f'  {"ALL (worst-case)":<22} unit~{u:>6.0f}  ret~{m["ret_pct"]:>6.2f}%  '
            f'peakGross~{m["peak_gross_pct"]:>5.0f}%  peakSym~{m["peak_sym_pct"]:>4.0f}%'
        )


if __name__ == "__main__":
    main()
