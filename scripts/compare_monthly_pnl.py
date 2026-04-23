#!/usr/bin/env python3
"""Compare monthly cumulative R between two sets of rolling-sim runs.

Purpose (Wave 3 label refactor验证 Step 0):
  For each Wave 3 step (label改造), we need a fast, reproducible way to
  compare the new rule set's month-by-month performance against a baseline
  (current Wave 1+2 state) on key attribution months.

Inputs:
  --baseline-runs TIMESTAMPS  comma-separated rolling_sim run timestamps that
                              constitute the baseline (e.g. 20260423_104715 or
                              a glob pattern resolved outside).
  --new-runs TIMESTAMPS       same format, for the candidate run set.
  --strategy STRAT            strategy name (bpc/tpc/me/srb/fbf). Determines
                              the rolling_sim subtree.
  --results-root PATH         defaults to ./results
  --attribution MODE          entry_month | exit_month | linear_days (default
                              linear_days per Wave 3 plan).
  --output PATH               optional markdown output path.

Key months (hard-coded highlights):
  2024-04, 2024-05, 2024-06 — trend-favorable months where BPC historically
                              produced +1086R but later Wave 1+2 reduced to
                              -157R.
  2025-11, 2025-12          — death months (BPC 0 trades).
  2024-01, 2024-03          — small-sample diagnostic months.

Output:
  Monthly matrix with columns: month, baseline_n/R, new_n/R, delta_R.
  Plus a summary block for each "bucket" of months (trend / death / small).
  Also emits a pass/fail verdict per Wave 3 step's gating criteria.

Dedup logic (identical to summarize_cross_strategy_monthly.py):
  key = (symbol, side, entry_time, exit_time, is_add_position)
  A trade shows up across multiple fast_month dirs; we de-dupe across all.

Usage example:
  python scripts/compare_monthly_pnl.py \\
    --strategy bpc \\
    --baseline-runs 20260423_104715 \\
    --new-runs 20260424_120000 \\
    --output results/wave3/wave3a_bpc_monthly_diff.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from calendar import monthrange
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]

# Buckets used in Wave 3 pass/fail verdict. Each entry: (bucket_name, months, verdict_fn).
KEY_MONTH_BUCKETS: Dict[str, List[str]] = {
    "trend_favorable": ["2024-04", "2024-05", "2024-06"],
    "death_months": ["2025-11", "2025-12"],
    "small_sample": ["2024-01", "2024-03"],
}

# Wave 3 pass/fail thresholds (plan §前提与验证口径).
PASS_THRESHOLDS = {
    "trend_favorable_delta_floor": -20.0,  # new - baseline >= -20R
    "trend_favorable_abs_floor": -100.0,  # new cumulative R >= -100R
    "death_months_min_trades": 1,  # at least one death month has trades
}


def _parse_ts(ts: str) -> datetime:
    if not ts:
        return datetime(1970, 1, 1)
    s = ts[:19]
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
    out: List[str] = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _load_fast_month_trades(
    strat: str, run_timestamps: Sequence[str], results_root: Path
) -> List[Dict]:
    """Load & dedupe event_trades from the fast_month subtrees of the given runs.

    run_timestamps: list of rolling_sim timestamps (dir names under _rolling_sim/).
    """
    root = results_root / strat / "slow-rolling-sim" / "_rolling_sim"
    if not root.exists():
        return []
    seen: set = set()
    out: List[Dict] = []
    # Collect all event_trades CSVs across the specified run timestamps.
    paths: List[Tuple[str, str]] = []  # (month_tag, csv_path)
    for ts in run_timestamps:
        ts = ts.strip()
        if not ts:
            continue
        run_dir = root / ts
        for p in sorted(
            glob.glob(
                str(run_dir / "fast_month_*" / strat / f"event_trades_{strat}.csv")
            ),
            key=os.path.getmtime,
        ):
            tag = p.split("fast_month_")[1].split("/")[0]
            paths.append((tag, p))
    # Later-modified paths win on duplicate month tag.
    by_target: Dict[str, str] = {}
    for tag, p in paths:
        by_target[tag] = p
    for tag, p in by_target.items():
        try:
            with open(p) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    key = (
                        row.get("symbol", ""),
                        row.get("side", ""),
                        row.get("entry_time", ""),
                        row.get("exit_time", ""),
                        row.get("is_add_position", ""),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        row["_pnl_r"] = float(row.get("pnl_r", "") or 0.0)
                    except Exception:
                        continue
                    row["_entry_dt"] = _parse_ts(row.get("entry_time", ""))
                    row["_exit_dt"] = _parse_ts(row.get("exit_time", ""))
                    row["_src_target"] = tag
                    row["_src_path"] = p
                    out.append(row)
        except FileNotFoundError:
            continue
    return out


def _attribute(trade: Dict, mode: str) -> Dict[str, float]:
    r = trade["_pnl_r"]
    em = _month_key(trade["_entry_dt"])
    xm = _month_key(trade["_exit_dt"])
    if mode == "entry_month":
        return {em: r}
    if mode == "exit_month":
        return {xm: r}
    if mode == "linear_days":
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


def aggregate_monthly(
    trades: List[Dict], attribution: str
) -> Tuple[Dict[str, float], Dict[str, int]]:
    by_r: Dict[str, float] = defaultdict(float)
    by_n: Dict[str, int] = defaultdict(int)
    for t in trades:
        # N counted by entry month (one trade = one count regardless of split).
        em = _month_key(t["_entry_dt"])
        by_n[em] += 1
        for m, portion in _attribute(t, attribution).items():
            by_r[m] += portion
    return dict(by_r), dict(by_n)


def render_markdown(
    strategy: str,
    baseline_label: str,
    new_label: str,
    attribution: str,
    months: List[str],
    base_r: Dict[str, float],
    base_n: Dict[str, int],
    new_r: Dict[str, float],
    new_n: Dict[str, int],
    baseline_runs: Sequence[str],
    new_runs: Sequence[str],
) -> Tuple[str, Dict]:
    lines: List[str] = []
    lines.append(
        f"# Monthly R diff: {strategy.upper()} — {new_label} vs {baseline_label}"
    )
    lines.append("")
    lines.append(f"- attribution mode: **{attribution}**")
    lines.append(f"- baseline runs: {', '.join(baseline_runs) or '—'}")
    lines.append(f"- new runs: {', '.join(new_runs) or '—'}")
    lines.append(f"- window: {months[0]} → {months[-1]}")
    lines.append("")

    # Per-bucket summaries & verdict.
    verdict: Dict[str, Dict] = {}
    lines.append("## 关键月桶 (buckets)")
    lines.append("")
    lines.append("| bucket | months | baseline n/R | new n/R | delta R | verdict |")
    lines.append("|--------|--------|--------------|---------|---------|---------|")
    for bucket, ms in KEY_MONTH_BUCKETS.items():
        b_n = sum(base_n.get(m, 0) for m in ms)
        b_r = sum(base_r.get(m, 0.0) for m in ms)
        n_n = sum(new_n.get(m, 0) for m in ms)
        n_r = sum(new_r.get(m, 0.0) for m in ms)
        delta = n_r - b_r
        v = _bucket_verdict(bucket, b_r, n_r, b_n, n_n)
        verdict[bucket] = {
            "baseline_n": b_n,
            "baseline_r": b_r,
            "new_n": n_n,
            "new_r": n_r,
            "delta_r": delta,
            "verdict": v,
        }
        lines.append(
            f"| {bucket} | {', '.join(ms)} | {b_n}/{b_r:+.1f} | "
            f"{n_n}/{n_r:+.1f} | {delta:+.1f} | {v} |"
        )
    lines.append("")

    # Full-window totals.
    tb_n = sum(base_n.get(m, 0) for m in months)
    tb_r = sum(base_r.get(m, 0.0) for m in months)
    tn_n = sum(new_n.get(m, 0) for m in months)
    tn_r = sum(new_r.get(m, 0.0) for m in months)
    lines.append("## 全窗口合计")
    lines.append("")
    lines.append(f"- baseline: n={tb_n}, R={tb_r:+.1f}")
    lines.append(f"- new:      n={tn_n}, R={tn_r:+.1f}")
    lines.append(f"- delta:    n={tn_n - tb_n:+d}, R={tn_r - tb_r:+.1f}")
    lines.append("")
    verdict["total"] = {
        "baseline_n": tb_n,
        "baseline_r": tb_r,
        "new_n": tn_n,
        "new_r": tn_r,
        "delta_r": tn_r - tb_r,
    }

    # Per-month matrix.
    lines.append("## 月度矩阵")
    lines.append("")
    lines.append("| month | baseline n/R | new n/R | delta R | note |")
    lines.append("|-------|--------------|---------|---------|------|")
    key_months_flat = {m for ms in KEY_MONTH_BUCKETS.values() for m in ms}
    for m in months:
        b_n = base_n.get(m, 0)
        b_r = base_r.get(m, 0.0)
        n_n = new_n.get(m, 0)
        n_r = new_r.get(m, 0.0)
        delta = n_r - b_r
        if b_n == 0 and n_n == 0 and abs(delta) < 1e-6:
            b_cell = "—"
            n_cell = "—"
            d_cell = "—"
        else:
            b_cell = f"{b_n}/{b_r:+.1f}" if (b_n or abs(b_r) > 0.01) else "—"
            n_cell = f"{n_n}/{n_r:+.1f}" if (n_n or abs(n_r) > 0.01) else "—"
            d_cell = f"{delta:+.1f}"
        note = "KEY" if m in key_months_flat else ""
        lines.append(f"| {m} | {b_cell} | {n_cell} | {d_cell} | {note} |")
    lines.append("")

    overall = _overall_verdict(verdict)
    lines.append(f"## 整体结论: **{overall}**")
    lines.append("")
    lines.append("判断口径 (Wave 3 plan §前提与验证口径):")
    lines.append(
        f"- trend_favorable (2024-04~06): new R 不比 baseline 低于 "
        f"{PASS_THRESHOLDS['trend_favorable_delta_floor']:+.0f}R, 且绝对值 ≥ "
        f"{PASS_THRESHOLDS['trend_favorable_abs_floor']:+.0f}R"
    )
    lines.append(
        f"- death_months (2025-11~12): 至少 "
        f"{PASS_THRESHOLDS['death_months_min_trades']} 月 trade 数 > 0"
    )
    lines.append("- small_sample: 仅诊断, 不阻塞")
    lines.append("")
    verdict["_overall"] = overall
    return "\n".join(lines) + "\n", verdict


def _bucket_verdict(
    bucket: str, base_r: float, new_r: float, base_n: int, new_n: int
) -> str:
    delta = new_r - base_r
    if bucket == "trend_favorable":
        if (
            delta >= PASS_THRESHOLDS["trend_favorable_delta_floor"]
            and new_r >= PASS_THRESHOLDS["trend_favorable_abs_floor"]
        ):
            return "PASS"
        return "FAIL"
    if bucket == "death_months":
        if new_n >= PASS_THRESHOLDS["death_months_min_trades"]:
            return "PASS"
        return "FAIL"
    return "INFO"


def _overall_verdict(per_bucket: Dict[str, Dict]) -> str:
    fails = [
        b
        for b, info in per_bucket.items()
        if b != "total" and info.get("verdict") == "FAIL"
    ]
    if not fails:
        return "PASS"
    return f"FAIL ({', '.join(fails)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True)
    ap.add_argument(
        "--baseline-runs",
        required=True,
        help="comma-separated rolling_sim timestamps for baseline",
    )
    ap.add_argument(
        "--new-runs",
        required=True,
        help="comma-separated rolling_sim timestamps for candidate",
    )
    ap.add_argument(
        "--results-root",
        default=str(REPO / "results"),
    )
    ap.add_argument(
        "--attribution",
        default="linear_days",
        choices=["entry_month", "exit_month", "linear_days"],
    )
    ap.add_argument("--start", default="2023-09")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--new-label", default="new")
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--json-output",
        default=None,
        help="optional JSON dump of verdict dict for CI",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    baseline_runs = [s.strip() for s in args.baseline_runs.split(",") if s.strip()]
    new_runs = [s.strip() for s in args.new_runs.split(",") if s.strip()]

    base_trades = _load_fast_month_trades(args.strategy, baseline_runs, results_root)
    new_trades = _load_fast_month_trades(args.strategy, new_runs, results_root)

    if not base_trades and not new_trades:
        print("⚠️  Both baseline and new returned 0 trades. Check run timestamps.")
        print(f"   baseline runs: {baseline_runs}")
        print(f"   new runs:      {new_runs}")
        return 1

    base_r, base_n = aggregate_monthly(base_trades, args.attribution)
    new_r, new_n = aggregate_monthly(new_trades, args.attribution)
    months = _iter_months(args.start, args.end)

    md, verdict = render_markdown(
        args.strategy,
        args.baseline_label,
        args.new_label,
        args.attribution,
        months,
        base_r,
        base_n,
        new_r,
        new_n,
        baseline_runs,
        new_runs,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"wrote {out}")

    if args.json_output:
        jp = Path(args.json_output)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
        print(f"wrote {jp}")

    print(md)
    # Exit code: 0 if overall PASS or INFO only, 1 if any FAIL.
    return 0 if verdict.get("_overall", "").startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
