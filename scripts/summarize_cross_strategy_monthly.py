#!/usr/bin/env python3
"""Cross-strategy monthly R summary with proper attribution.

Fixes the month-attribution distortion in fast_month's event_backtest_*.json,
which credits a trade's full pnl_r to its exit month. For multi-month trend
positions (e.g. BTC opened 2024-10-11, closed 2025-02-03 with +211R), this
concentrates 4 months of accumulated trend pnl onto the exit month.

This tool:
  1. Dedupes trades across fast_month rolling-sim runs (key = symbol + side +
     entry_time + exit_time + is_add_position). A long-holding position shows
     up in every fast_month_* run between its entry and exit but should only be
     counted once.
  2. Attributes per-month R using one of three modes:
       - entry_month (default): full pnl_r → the month the signal fired. Best
         for "which signal decision produced this pnl" view.
       - exit_month: legacy view (matches fast_month json), credits to exit.
       - linear_days: pro-rata pnl_r by days overlapping each calendar month.
         Approximate MTM; exact bar-level MTM is out of scope for this tool.
  3. Computes per-strategy health: n, win_rate, median, mean, total_r, top5,
     bot5, and total_r_excluding_top5 (robustness to fat-tail concentration).

Usage:
  python scripts/summarize_cross_strategy_monthly.py \
    --strategies bpc,tpc,me,srb \
    --start 2023-09 --end 2026-03 \
    --attribution entry_month \
    --output reports/cross_strategy_monthly_summary.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from calendar import monthrange
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]


def _parse_ts(ts: str) -> datetime:
    # Accept '2024-03-04T08:00:00' or '2024-03-04T08:00:00+00:00'
    if not ts:
        return datetime(1970, 1, 1)
    s = ts[:19]  # drop tz suffix
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return datetime(1970, 1, 1)


def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _iter_months(start: str, end: str) -> List[str]:
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _load_deduped_trades(strat: str, results_root: Path) -> List[Dict]:
    root = results_root / strat / "slow-rolling-sim" / "_rolling_sim"
    if not root.exists():
        return []
    paths = sorted(
        glob.glob(
            str(root / "*" / "fast_month_*" / strat / f"event_trades_{strat}.csv")
        ),
        key=os.path.getmtime,
    )
    # for each fast_month target, keep the latest run
    by_target: Dict[str, str] = {}
    for p in paths:
        tgt = p.split("fast_month_")[1].split("/")[0]
        by_target[tgt] = p

    seen: set = set()
    out: List[Dict] = []
    for tgt, p in by_target.items():
        try:
            with open(p) as fh:
                for row in csv.DictReader(fh):
                    key = (
                        row["symbol"],
                        row["side"],
                        row["entry_time"],
                        row["exit_time"],
                        row.get("is_add_position", ""),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        row["_pnl_r"] = float(row["pnl_r"])
                    except Exception:
                        continue
                    row["_entry_dt"] = _parse_ts(row["entry_time"])
                    row["_exit_dt"] = _parse_ts(row["exit_time"])
                    row["_src_target"] = tgt
                    out.append(row)
        except Exception:
            pass
    return out


def _attribute(trade: Dict, mode: str) -> Dict[str, float]:
    """Return {month: pnl_portion} for this trade per the attribution mode."""
    r = trade["_pnl_r"]
    em = _month_key(trade["_entry_dt"])
    xm = _month_key(trade["_exit_dt"])
    if mode == "entry_month":
        return {em: r}
    if mode == "exit_month":
        return {xm: r}
    if mode == "linear_days":
        # pro-rata by calendar days of overlap per month
        e = trade["_entry_dt"]
        x = trade["_exit_dt"]
        if x <= e:
            return {em: r}
        total_seconds = (x - e).total_seconds()
        if total_seconds <= 0:
            return {em: r}
        parts: Dict[str, float] = defaultdict(float)
        y, m = e.year, e.month
        cursor = e
        while (y, m) <= (x.year, x.month):
            _, last_day = monthrange(y, m)
            month_end = datetime(y, m, last_day, 23, 59, 59)
            seg_end = min(month_end, x)
            seg_seconds = (seg_end - cursor).total_seconds()
            if seg_seconds > 0:
                parts[f"{y:04d}-{m:02d}"] += r * (seg_seconds / total_seconds)
            m += 1
            if m > 12:
                m = 1
                y += 1
            cursor = datetime(y, m, 1) if (y, m) <= (x.year, x.month) else x
        return dict(parts)
    raise ValueError(f"unknown attribution mode: {mode}")


def _max_drawdown_r(rs_sorted_by_time: List[float]) -> float:
    cum, peak, dd = 0.0, 0.0, 0.0
    for r in rs_sorted_by_time:
        cum += r
        if cum > peak:
            peak = cum
        dd = max(dd, peak - cum)
    return dd


def build_summary(
    strategies: List[str],
    results_root: Path,
    start: str,
    end: str,
    attribution: str,
) -> Tuple[Dict, Dict]:
    months = _iter_months(start, end)
    by_strat_month_r: Dict[str, Dict[str, float]] = {
        s: defaultdict(float) for s in strategies
    }
    by_strat_month_n: Dict[str, Dict[str, int]] = {
        s: defaultdict(int) for s in strategies
    }
    all_trades: Dict[str, List[Dict]] = {}

    for s in strategies:
        trades = _load_deduped_trades(s, results_root)
        all_trades[s] = trades
        for t in trades:
            # For n-count, use entry_month irrespective of attribution (a trade
            # is "one trade" regardless of how its pnl is split across months).
            em = _month_key(t["_entry_dt"])
            if not (start <= em <= end):
                continue
            by_strat_month_n[s][em] += 1
            for month, portion in _attribute(t, attribution).items():
                if start <= month <= end:
                    by_strat_month_r[s][month] += portion
    return by_strat_month_r, by_strat_month_n, all_trades, months


def compute_health(trades: List[Dict], start: str, end: str) -> Dict:
    scoped = [t for t in trades if start <= _month_key(t["_entry_dt"]) <= end]
    if not scoped:
        return {"n": 0}
    rs = sorted([t["_pnl_r"] for t in scoped], reverse=True)
    n = len(rs)
    wins = sum(1 for r in rs if r > 0)
    # max DD on time-ordered sequence
    sorted_by_exit = sorted(scoped, key=lambda t: t["_exit_dt"])
    dd_r = _max_drawdown_r([t["_pnl_r"] for t in sorted_by_exit])
    top5_sum = sum(rs[:5])
    bot5_sum = sum(rs[-5:])
    return {
        "n": n,
        "win_rate": wins / n if n else 0.0,
        "total_r": sum(rs),
        "mean": sum(rs) / n if n else 0.0,
        "median": rs[n // 2] if n else 0.0,
        "top5": rs[:5],
        "bot5": rs[-5:],
        "top5_sum": top5_sum,
        "total_r_excl_top5": sum(rs) - top5_sum,
        "max_dd_r": dd_r,
    }


def render_markdown(
    strategies: List[str],
    months: List[str],
    by_strat_month_r: Dict,
    by_strat_month_n: Dict,
    health: Dict[str, Dict],
    attribution: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# Cross-strategy monthly R summary")
    lines.append("")
    lines.append(f"- attribution mode: **{attribution}**")
    lines.append(f"- months: {months[0]} → {months[-1]}")
    lines.append(f"- strategies: {', '.join(strategies)}")
    lines.append("")
    lines.append(
        "Note: trade counts are assigned by entry month (one trade = one "
        "count, regardless of attribution mode). R values follow the "
        "attribution mode."
    )
    lines.append("")
    lines.append("## Per-strategy health (全期 deduped)")
    lines.append("")
    lines.append(
        "| strategy | n | win% | total_r | median | mean | max_dd_r | top5 sum | total_r excl top5 |"
    )
    lines.append(
        "|----------|---|------|---------|--------|------|----------|----------|-------------------|"
    )
    for s in strategies:
        h = health[s]
        if h["n"] == 0:
            lines.append(f"| {s} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {s} | {h['n']} | {h['win_rate']:.1%} | "
            f"{h['total_r']:+.1f} | {h['median']:+.2f} | {h['mean']:+.2f} | "
            f"{h['max_dd_r']:+.1f} | {h['top5_sum']:+.1f} | "
            f"{h['total_r_excl_top5']:+.1f} |"
        )
    lines.append("")
    lines.append("## Top / Bottom 5 trades per strategy")
    lines.append("")
    for s in strategies:
        h = health[s]
        if h["n"] == 0:
            continue
        lines.append(f"### {s}")
        lines.append(f"- top5 pnl_r: {[round(v, 2) for v in h['top5']]}")
        lines.append(f"- bot5 pnl_r: {[round(v, 2) for v in h['bot5']]}")
    lines.append("")
    lines.append("## Monthly n/R matrix")
    lines.append("")
    header = "| month | " + " | ".join(strategies) + " |"
    sep = "|-------|" + "|".join(["---"] * len(strategies)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in months:
        cells = []
        for s in strategies:
            n = by_strat_month_n[s].get(m, 0)
            r = by_strat_month_r[s].get(m, 0.0)
            if n == 0 and abs(r) < 1e-6:
                cells.append("—")
            else:
                cells.append(f"{n}/{r:+.1f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategies", default="bpc,tpc,me,srb")
    ap.add_argument("--start", default="2023-09")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument(
        "--attribution",
        default="entry_month",
        choices=["entry_month", "exit_month", "linear_days"],
    )
    ap.add_argument(
        "--results-root",
        default=str(REPO / "results"),
        help="dir containing <strategy>/slow-rolling-sim/_rolling_sim",
    )
    ap.add_argument("--output", default=None, help="markdown output path")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    results_root = Path(args.results_root)

    by_r, by_n, all_trades, months = build_summary(
        strategies, results_root, args.start, args.end, args.attribution
    )
    health = {
        s: compute_health(all_trades[s], args.start, args.end) for s in strategies
    }
    md = render_markdown(strategies, months, by_r, by_n, health, args.attribution)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"✅ wrote {out}")
    print(md)


if __name__ == "__main__":
    main()
