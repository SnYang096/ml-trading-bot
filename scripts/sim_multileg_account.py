#!/usr/bin/env python3
"""Real-account simulation for the shared multi-leg account (chop_grid + trend_scalp).

Backtests run each symbol independently; this script replays **trade fills** and
**segment boundaries** on one shared account timeline with live-like gates.

Per-strategy sizing (constitution ``multi_leg.sizing``)::

    chop:  unit = chop.segment_dd_target × equity / (max_loss_per_grid × 2 × levels)
    trend: unit = trend.segment_dd_target × equity / (max_loss_per_segment × max_gross_units)

Usage
-----
    python scripts/sim_multileg_account.py \\
        --constitution-yaml live/highcap/config/constitution/constitution.yaml \\
        --chop-root results/chop_grid/experiments/stack_ablation_20260604/E7_four_segment/dense_3l_live \\
        --trend-root results/trend_scalp/experiments/segment_validate_20260603_timeline \\
        --segments bear_2022 bull_2023_2024 recent_range_to_bear recent_6m_oos \\
        --max-concurrent-multi-leg-symbols 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.multileg_sizing import (
    grid_capital_units,
    resolve_multi_leg_unit_notionals,
    trend_capital_units,
)
from src.live_data_stream.constitution_config import (
    load_constitution_dict,
    multi_leg_section,
    resolve_multileg_sim_limits,
)
from src.sim.multileg_account_sim import (
    apply_multileg_segment_gates,
    filter_trades_by_segment_blocks,
    load_chop_segments,
    load_chop_trades,
    load_trend_segments,
    load_trend_trades,
    simulate_account_trades,
    simulate_account_with_constitution,
)


def _print_metrics(title: str, m: Dict[str, float], *, extra: str = "") -> None:
    suffix = f"  {extra}" if extra else ""
    peak_dd = m.get("max_dd_peak_pct")
    dd_str = (
        f"maxDD={m['max_dd_pct']:.3f}%"
        if peak_dd is None
        else f"maxDD={m['max_dd_pct']:.1f}%(init) / {peak_dd:.1f}%(peak)"
    )
    print(
        f"{title}: ret={m['ret_pct']:.2f}% pnl={m['pnl_usd']:.0f} "
        f"peakGross={m['peak_gross_pct']:.1f}% peakSym={m['peak_sym_pct']:.1f}% "
        f"{dd_str} trades={int(m['n_trades'])}{suffix}"
    )


def _resolve_units(args: argparse.Namespace) -> Dict[str, float]:
    if args.unit_notional is not None:
        unit = float(args.unit_notional)
        return {"chop_grid": unit, "trend_scalp": unit}

    if args.constitution_yaml:
        path = Path(args.constitution_yaml)
        ml = multi_leg_section(load_constitution_dict(path))
        sr = str(args.strategies_root or "live/highcap/config/strategies")
        chop_exe = Path(sr) / "chop_grid" / "archetypes" / "execution.yaml"
        trend_exe = Path(sr) / "trend_scalp" / "archetypes" / "execution.yaml"
        units = resolve_multi_leg_unit_notionals(
            ml,
            equity_usdt=float(args.equity),
            chop_grid_execution_path=chop_exe if chop_exe.is_file() else None,
            trend_scalp_execution_path=trend_exe if trend_exe.is_file() else None,
        )
        return units

    from src.config.multileg_sizing import (
        unit_notional_from_segment_dd,
        unit_notional_from_trend_segment_dd,
    )

    chop_dd = args.chop_segment_dd_target
    if chop_dd is None:
        chop_dd = args.segment_dd_target
    trend_dd = args.trend_segment_dd_target
    if trend_dd is None:
        trend_dd = args.segment_dd_target

    units: Dict[str, float] = {}
    if chop_dd is not None:
        units["chop_grid"] = unit_notional_from_segment_dd(
            equity_usdt=args.equity,
            segment_dd_target=float(chop_dd),
            max_loss_per_grid=float(args.max_loss_per_grid),
            max_levels_per_side=int(args.max_levels_per_side),
        )
    if trend_dd is not None:
        units["trend_scalp"] = unit_notional_from_trend_segment_dd(
            equity_usdt=args.equity,
            segment_dd_target=float(trend_dd),
            max_loss_per_segment=float(args.max_loss_per_segment),
            max_gross_exposure_units=int(args.max_gross_exposure_units),
        )
    if not units:
        units = {"chop_grid": 556.0, "trend_scalp": 556.0}
    return units


def _print_sizing(units: Dict[str, float], args: argparse.Namespace) -> None:
    print("=== Per-strategy segment-risk sizing ===")
    print(f"  equity = {args.equity:.0f} USDT")
    if "chop_grid" in units:
        cap = grid_capital_units(int(args.max_levels_per_side))
        u = units["chop_grid"]
        print(
            f"  chop_grid   unit={u:.1f}  "
            f"worst-case segment loss={args.max_loss_per_grid * cap * u / args.equity * 100:.2f}%"
        )
    if "trend_scalp" in units:
        cap = trend_capital_units(int(args.max_gross_exposure_units))
        u = units["trend_scalp"]
        print(
            f"  trend_scalp unit={u:.1f}  "
            f"worst-case segment loss={args.max_loss_per_segment * cap * u / args.equity * 100:.2f}%"
        )
    print()


def _segment_dd_table(
    chop_root: Optional[Path],
    market_segments: List[str],
    *,
    equity: float,
    unit: float,
    max_levels_per_side: int,
) -> None:
    if chop_root is None:
        return
    cap = grid_capital_units(max_levels_per_side)
    print(
        f"=== chop_grid observed segment DD @ unit={unit:.0f} "
        f"(|max_drawdown| × {cap} × unit / equity) ==="
    )
    print(f'{"window":<22} {"median%":>8} {"p95%":>8} {"worst%":>8} {"#>1%":>6}')
    for seg in market_segments:
        sdf = load_chop_segments(chop_root, seg)
        if sdf.empty or "max_drawdown" not in sdf.columns:
            continue
        if "status" in sdf.columns:
            sdf = sdf[sdf["status"] == "ok"]
        acct = sdf["max_drawdown"].abs() * cap * unit / equity * 100.0
        print(
            f"{seg:<22} {acct.median():>8.2f} {acct.quantile(0.95):>8.2f} "
            f"{acct.max():>8.2f} {int((acct > 1.0).sum()):>6}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chop-root", "--root", dest="chop_root", type=Path)
    ap.add_argument("--trend-root", type=Path, default=None)
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--segments", nargs="+", default=["recent_6m_oos"])
    ap.add_argument(
        "--constitution-yaml",
        type=Path,
        default=None,
        help="load per-strategy sizing from multi_leg.sizing",
    )
    ap.add_argument(
        "--strategies-root",
        default="live/highcap/config/strategies",
        help="execution.yaml lookup when using --constitution-yaml",
    )
    ap.add_argument(
        "--segment-dd-target",
        type=float,
        default=None,
        help="legacy: same target for both strategies",
    )
    ap.add_argument("--chop-segment-dd-target", type=float, default=None)
    ap.add_argument("--trend-segment-dd-target", type=float, default=None)
    ap.add_argument("--max-loss-per-grid", type=float, default=0.03)
    ap.add_argument("--max-levels-per-side", type=int, default=3)
    ap.add_argument("--max-loss-per-segment", type=float, default=0.02)
    ap.add_argument("--max-gross-exposure-units", type=int, default=4)
    ap.add_argument("--max-concurrent-multi-leg-symbols", type=int, default=0)
    ap.add_argument("--unit-notional", type=float, default=None)
    ap.add_argument(
        "--with-constitution",
        action="store_true",
        default=False,
        help="Apply constitution risk limits (max_gross, max_dd, daily_loss_limit)",
    )
    ap.add_argument(
        "--fuse-mode",
        choices=("hard", "tier_derate", "tier_daily_scaled"),
        default="hard",
        help="Graded fuse mode when --with-constitution (default: hard halt)",
    )
    ap.add_argument("--fuse-soft-dd-ratio", type=float, default=0.5)
    ap.add_argument("--fuse-derate-factor", type=float, default=0.5)
    args = ap.parse_args()

    if args.chop_root is None and args.trend_root is None:
        raise SystemExit("provide --chop-root and/or --trend-root")

    units = _resolve_units(args)
    _print_sizing(units, args)

    chop_trades_parts: List[pd.DataFrame] = []
    trend_trades_parts: List[pd.DataFrame] = []
    chop_seg_parts: List[pd.DataFrame] = []
    trend_seg_parts: List[pd.DataFrame] = []

    for seg in args.segments:
        if args.chop_root is not None:
            ct = load_chop_trades(args.chop_root, seg)
            if not ct.empty:
                chop_trades_parts.append(ct)
            cs = load_chop_segments(args.chop_root, seg)
            if not cs.empty:
                chop_seg_parts.append(cs)
        if args.trend_root is not None:
            tt = load_trend_trades(args.trend_root, seg)
            if not tt.empty:
                trend_trades_parts.append(tt)
            ts = load_trend_segments(args.trend_root, seg)
            if not ts.empty:
                trend_seg_parts.append(ts)

    chop_tr = (
        pd.concat(chop_trades_parts, ignore_index=True)
        if chop_trades_parts
        else pd.DataFrame()
    )
    trend_tr = (
        pd.concat(trend_trades_parts, ignore_index=True)
        if trend_trades_parts
        else pd.DataFrame()
    )
    chop_seg = (
        pd.concat(chop_seg_parts, ignore_index=True)
        if chop_seg_parts
        else pd.DataFrame()
    )
    trend_seg = (
        pd.concat(trend_seg_parts, ignore_index=True)
        if trend_seg_parts
        else pd.DataFrame()
    )

    if chop_tr.empty and trend_tr.empty:
        raise SystemExit("no trades found under provided roots/segments")

    u_chop = units.get("chop_grid", units.get("trend_scalp", 556.0))
    u_trend = units.get("trend_scalp", units.get("chop_grid", 556.0))
    unit_by = {"chop_grid": u_chop, "trend_scalp": u_trend}

    # ── Resolve constitution risk limits ──
    const_limits: Dict[str, object] = {}
    if args.with_constitution and args.constitution_yaml:
        const = load_constitution_dict(str(args.constitution_yaml))
        const_limits = resolve_multileg_sim_limits(const)
        const_limits["fuse_mode"] = args.fuse_mode
        const_limits["fuse_soft_dd_ratio"] = float(args.fuse_soft_dd_ratio)
        const_limits["fuse_derate_factor"] = float(args.fuse_derate_factor)
        print("=== Constitution risk limits ===")
        for k, v in const_limits.items():
            if v is not None:
                print(f"  {k}: {v}")
        print()

    def _sim(trades_df, eq, unit, unit_by_strat=None):
        if const_limits:
            sim_kw = {
                k: const_limits[k]
                for k in (
                    "max_drawdown_pct",
                    "daily_loss_limit_pct",
                    "max_gross_notional_pct",
                    "max_net_notional_pct",
                    "max_symbol_gross_notional_pct",
                    "max_symbol_net_notional_pct",
                    "max_gross_leverage",
                    "fuse_mode",
                    "fuse_soft_dd_ratio",
                    "fuse_derate_factor",
                )
                if const_limits.get(k) is not None
            }
            m = simulate_account_with_constitution(
                trades_df,
                equity=eq,
                unit_notional=unit,
                unit_by_strategy=unit_by_strat,
                **sim_kw,
            )
            extra = f" halted={m.get('halted')} rejected={int(m.get('n_rejected',0))}"
            if m.get("halted_reason"):
                extra += f" [{m['halted_reason']}]"
        else:
            m = simulate_account_trades(
                trades_df, equity=eq, unit_notional=unit, unit_by_strategy=unit_by_strat
            )
            extra = ""
        return m, extra

    if not chop_tr.empty:
        m, _ = _sim(chop_tr, args.equity, u_chop)
        _print_metrics("chop_grid alone", m)
    if not trend_tr.empty:
        m, _ = _sim(trend_tr, args.equity, u_trend)
        _print_metrics("trend_scalp alone", m)
    print()

    gate_stats = apply_multileg_segment_gates(
        chop_seg,
        trend_seg,
        max_concurrent_multi_leg_symbols=int(
            args.max_concurrent_multi_leg_symbols or 0
        ),
    )
    chop_allowed = filter_trades_by_segment_blocks(
        chop_tr, gate_stats.blocked_chop_segment_ids
    )
    trend_allowed = filter_trades_by_segment_blocks(
        trend_tr, gate_stats.blocked_trend_segment_ids
    )

    print("=== Live-like gates (shared account) ===")
    print(
        f"  chop blocked segments: {gate_stats.blocked_chop_segments} "
        f"(peak chop symbols={gate_stats.peak_chop_symbols})"
    )
    print(
        f"  trend blocked segments: {gate_stats.blocked_trend_segments} "
        f"(symbol conflicts={gate_stats.peak_symbol_conflicts})"
    )
    if args.max_concurrent_multi_leg_symbols > 0:
        print(
            f"  max_concurrent_multi_leg_symbols={args.max_concurrent_multi_leg_symbols}"
        )

    if not chop_allowed.empty:
        m, _ = _sim(chop_allowed, args.equity, u_chop)
        _print_metrics("  chop_grid", m)
    if not trend_allowed.empty:
        m, _ = _sim(trend_allowed, args.equity, u_trend)
        _print_metrics("  trend_scalp", m)

    combined_allowed = pd.concat(
        [x for x in (chop_allowed, trend_allowed) if not x.empty],
        ignore_index=True,
    )
    if not combined_allowed.empty:
        m, extra = _sim(combined_allowed, args.equity, 0.0, unit_by_strat=unit_by)
        _print_metrics("  COMBINED account", m, extra=extra)
        if "strategy" in combined_allowed.columns:
            for strat, grp in combined_allowed.groupby("strategy"):
                u = unit_by.get(str(strat), u_chop)
                sm, _ = _sim(grp, args.equity, u)
                _print_metrics(f"    └─ {strat} contribution", sm)
    print()

    if args.chop_root is not None and "chop_grid" in units:
        _segment_dd_table(
            args.chop_root,
            args.segments,
            equity=args.equity,
            unit=units["chop_grid"],
            max_levels_per_side=int(args.max_levels_per_side),
        )


if __name__ == "__main__":
    main()
