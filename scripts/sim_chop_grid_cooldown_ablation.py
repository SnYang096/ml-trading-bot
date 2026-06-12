#!/usr/bin/env python3
"""Ablation: cooldown bars after chop↔trend strategy switch.

Delays trend_scalp segment starts after a chop_grid segment ends on the same
symbol (and vice versa). Shortened/blocked segments filtered out before account
simulation. Compares PnL, DD, switch count vs baseline.

Usage:
  python3 scripts/sim_chop_grid_cooldown_ablation.py \
    --chop-root results/chop_grid/experiments/stack_ablation_20260604/E7_four_segment/dense_3l_live \
    --segments bear_2022 bull_2023_2024 recent_range_to_bear recent_6m_oos \
    --cooldown-bars 3
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.multileg_account_sim import (
    apply_multileg_segment_gates,
    filter_trades_by_segment_blocks,
    load_chop_segments,
    load_chop_trades,
    load_trend_segments,
    load_trend_trades,
    simulate_account_trades,
)
from src.config.multileg_sizing import resolve_multi_leg_unit_notionals
from src.live_data_stream.constitution_config import (
    load_constitution_dict,
    multi_leg_section,
)

BAR_MINUTES = 120  # 2h bars

CooldownMode = str  # "none" | "after_end" | "from_last_start"


def _add_timedelta(timestamp, bars: int):
    ts = pd.Timestamp(timestamp)
    return ts + timedelta(minutes=int(bars) * BAR_MINUTES)


def apply_cooldown_to_segments(
    chop: pd.DataFrame,
    trend: pd.DataFrame,
    *,
    cooldown_bars: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Delay segment starts that switch strategy too soon after previous end."""
    if cooldown_bars <= 0:
        return chop, trend, {"switches": 0, "delayed": 0, "blocked": 0}
    if trend.empty:
        return chop, trend, {"switches": 0, "delayed": 0, "blocked": 0}

    dup_chop = chop.copy()
    dup_trend = trend.copy()
    dup_chop["start"] = pd.to_datetime(dup_chop["start"])
    dup_chop["end"] = pd.to_datetime(dup_chop["end"])
    dup_trend["start"] = pd.to_datetime(dup_trend["start"])
    dup_trend["end"] = pd.to_datetime(dup_trend["end"])

    stats: Dict[str, int] = {"switches": 0, "delayed": 0, "blocked": 0}

    for sym in sorted(
        set(dup_chop["symbol"].unique()) & set(dup_trend["symbol"].unique())
    ):
        chop_sym = dup_chop[dup_chop["symbol"] == sym].sort_values("start")
        trend_sym = dup_trend[dup_trend["symbol"] == sym].sort_values("start")

        # Build timeline: interleave chop/trend segments.
        timeline: list = []
        for _, row in chop_sym.iterrows():
            timeline.append((row["start"], row["end"], "chop_grid", row["segment_id"]))
        for _, row in trend_sym.iterrows():
            timeline.append(
                (row["start"], row["end"], "trend_scalp", row["segment_id"])
            )
        timeline.sort(key=lambda x: x[0])

        prev_end = None
        prev_strategy = None

        for i, (start, end, strat, seg_id) in enumerate(timeline):
            if (
                prev_end is not None
                and prev_strategy is not None
                and prev_strategy != strat
            ):
                stats["switches"] += 1
                earliest = _add_timedelta(prev_end, cooldown_bars)
                if start < earliest:
                    stats["delayed"] += 1
                    new_start = earliest
                    if new_start >= end:
                        # Segment entirely blocked by cooldown.
                        stats["blocked"] += 1
                        if strat == "chop_grid":
                            dup_chop.loc[dup_chop["segment_id"] == seg_id, "start"] = (
                                end  # collapse to zero-length → filtered later
                            )
                        else:
                            dup_trend.loc[
                                dup_trend["segment_id"] == seg_id, "start"
                            ] = end
                    else:
                        # Shift start, keep end.
                        if strat == "chop_grid":
                            dup_chop.loc[dup_chop["segment_id"] == seg_id, "start"] = (
                                new_start
                            )
                        else:
                            dup_trend.loc[
                                dup_trend["segment_id"] == seg_id, "start"
                            ] = new_start
            prev_end = end
            prev_strategy = strat

    # Drop zero-length segments.
    dup_chop = dup_chop[dup_chop["start"] < dup_chop["end"]].copy()
    dup_trend = dup_trend[dup_trend["start"] < dup_trend["end"]].copy()
    return dup_chop, dup_trend, stats


def _print_metrics(title: str, m: Dict[str, float], *, extra: str = "") -> None:
    suffix = f"  {extra}" if extra else ""
    print(
        f"{title}: ret={m['ret_pct']:.2f}% pnl={m['pnl_usd']:.0f} "
        f"peakGross={m['peak_gross_pct']:.1f}% peakSym={m['peak_sym_pct']:.1f}% "
        f"maxDD={m['max_dd_pct']:.3f}% trades={int(m['n_trades'])}{suffix}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chop-root", type=Path, required=True)
    ap.add_argument("--trend-root", type=Path, default=None)
    ap.add_argument(
        "--segments",
        nargs="+",
        default=[
            "bear_2022",
            "bull_2023_2024",
            "recent_range_to_bear",
            "recent_6m_oos",
        ],
    )
    ap.add_argument("--cooldown-bars", type=int, default=3)
    ap.add_argument("--equity", type=float, default=12100)
    ap.add_argument(
        "--constitution-yaml",
        type=Path,
        default=Path("live/highcap/config/constitution/constitution.yaml"),
    )
    ap.add_argument("--max-concurrent-multi-leg-symbols", type=int, default=6)
    ap.add_argument("--symbols", type=int, default=6, help="Active symbol count")
    args = ap.parse_args()

    chop_root = args.chop_root
    trend_root = args.trend_root or chop_root

    # Load all segments into single DataFrames.
    chop_seg = pd.concat(
        [load_chop_segments(chop_root, s) for s in args.segments], ignore_index=True
    )
    trend_seg = pd.concat(
        [load_trend_segments(trend_root, s) for s in args.segments], ignore_index=True
    )
    chop_raw_trades = pd.concat(
        [load_chop_trades(chop_root, s) for s in args.segments], ignore_index=True
    )
    trend_raw_trades = pd.concat(
        [load_trend_trades(trend_root, s) for s in args.segments], ignore_index=True
    )

    # Baseline: no cooldown
    gate_stats = apply_multileg_segment_gates(
        chop_seg,
        trend_seg,
        max_concurrent_multi_leg_symbols=args.max_concurrent_multi_leg_symbols,
    )
    chop_trades = filter_trades_by_segment_blocks(
        chop_raw_trades, gate_stats.blocked_chop_segment_ids
    )
    trend_trades = filter_trades_by_segment_blocks(
        trend_raw_trades, gate_stats.blocked_trend_segment_ids
    )
    all_trades = pd.concat([chop_trades, trend_trades], ignore_index=True)
    baseline = simulate_account_trades(
        all_trades, equity=args.equity, unit_notional=100.0
    )
    print("=== Baseline (no cooldown) ===")
    _print_metrics("COMBINED", baseline)
    print(
        f"  chop blocked={gate_stats.blocked_chop_segments} trend blocked={gate_stats.blocked_trend_segments} "
        f"peak_symbols={gate_stats.peak_chop_symbols} conflicts={gate_stats.peak_symbol_conflicts}"
    )
    print()

    # Cooldown variants
    for cb in [3, 5, 8]:
        c_chop, c_trend, cd_stats = apply_cooldown_to_segments(
            chop_seg.copy(), trend_seg.copy(), cooldown_bars=cb
        )
        c_gate = apply_multileg_segment_gates(
            c_chop,
            c_trend,
            max_concurrent_multi_leg_symbols=args.max_concurrent_multi_leg_symbols,
        )
        c_chop_t = filter_trades_by_segment_blocks(
            chop_raw_trades, c_gate.blocked_chop_segment_ids
        )
        c_trend_t = filter_trades_by_segment_blocks(
            trend_raw_trades, c_gate.blocked_trend_segment_ids
        )
        c_all = pd.concat([c_chop_t, c_trend_t], ignore_index=True)
        c_result = simulate_account_trades(
            c_all, equity=args.equity, unit_notional=100.0
        )
        print(f"=== Cooldown {cb} bars ===")
        print(
            f"  switches={cd_stats['switches']} delayed={cd_stats['delayed']} blocked={cd_stats['blocked']}"
        )
        _print_metrics("COMBINED", c_result)
        print(
            f"  chop blocked={c_gate.blocked_chop_segments} trend blocked={c_gate.blocked_trend_segments} "
            f"peak_symbols={c_gate.peak_chop_symbols} conflicts={c_gate.peak_symbol_conflicts}"
        )
        print()


if __name__ == "__main__":
    main()
