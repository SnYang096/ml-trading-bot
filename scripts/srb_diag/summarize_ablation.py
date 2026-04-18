"""
Summarize the SRB 4-way ablation against baseline.

Reads baseline CSVs (event_trades_srb.csv) and each exp's per-month
event_backtest_srb.json to produce a side-by-side comparison:
  - n_trades, total_R, win_rate
  - trailing_sl count + mean_R
  - sl mean_R
  - add_position count + mean_R
  - reverse count + mean_R
  - post-exit MFE of trailing_sl (washout check) — optional, requires the
    enriched csv from scripts/srb_diag/wide_sr_and_trailing_diag.py

Usage:
  python scripts/srb_diag/summarize_ablation.py \
    --baseline-root results/srb/slow-rolling-sim/_rolling_sim/20260417_163432 \
    --ablation-root results/srb/diag/ablation_fast_20260418 \
    --out results/srb/diag/ablation_fast_20260418/COMPARE.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_baseline_trades(root: Path) -> List[Dict]:
    rows = []
    for f in sorted(root.glob("fast_month_*/srb/event_trades_srb.csv")):
        with open(f) as fp:
            rows.extend(list(csv.DictReader(fp)))
    return rows


def _load_exp_trades(exp_root: Path) -> List[Dict]:
    rows = []
    for jf in sorted(exp_root.glob("month_*/event_backtest_srb.json")):
        try:
            j = json.loads(jf.read_text())
        except Exception:
            continue
        for t in j.get("trades", []):
            # coerce to the csv schema used above
            rows.append(
                {
                    "symbol": t.get("symbol", ""),
                    "side": t.get("side", ""),
                    "exit_reason": t.get("exit_reason", ""),
                    "pnl_r": str(t.get("pnl_r", 0.0)),
                    "is_add_position": str(t.get("is_add_position", False)),
                    "is_reverse": str(t.get("is_reverse", False)),
                    "entry_time": str(t.get("entry_time", "")),
                    "exit_time": str(t.get("exit_time", "")),
                }
            )
    return rows


def _stat(rows: List[Dict]) -> Dict:
    n = len(rows)
    if n == 0:
        return {}
    total_r = sum(float(r["pnl_r"]) for r in rows)
    wins = sum(1 for r in rows if float(r["pnl_r"]) > 0)
    by_reason: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_reason[r["exit_reason"]].append(float(r["pnl_r"]))

    def _r_stat(reason: str):
        lst = by_reason.get(reason, [])
        if not lst:
            return (0, 0.0, 0.0)
        return (len(lst), round(sum(lst) / len(lst), 3), round(sum(lst), 3))

    add = [
        float(r["pnl_r"]) for r in rows if str(r["is_add_position"]).lower() == "true"
    ]
    rev = [float(r["pnl_r"]) for r in rows if str(r["is_reverse"]).lower() == "true"]
    return {
        "n": n,
        "total_r": round(total_r, 2),
        "mean_r": round(total_r / n, 3),
        "win_rate": round(wins / n, 3),
        "sl": _r_stat("sl"),
        "trailing_sl": _r_stat("trailing_sl"),
        "eob": _r_stat("end_of_backtest"),
        "add": (
            len(add),
            round(sum(add) / len(add), 3) if add else 0.0,
            round(sum(add), 3),
        ),
        "reverse": (
            len(rev),
            round(sum(rev) / len(rev), 3) if rev else 0.0,
            round(sum(rev), 3),
        ),
    }


def _row(label: str, s: Dict) -> str:
    if not s:
        return f"| {label} | n/a |"
    tr = s["trailing_sl"]
    sl = s["sl"]
    ad = s["add"]
    rv = s["reverse"]
    return (
        f"| {label} | {s['n']} | {s['total_r']:+.2f} | {s['mean_r']:+.3f} | "
        f"{s['win_rate']*100:.1f}% | {sl[0]} / {sl[1]:+.2f} | "
        f"{tr[0]} / {tr[1]:+.2f} | {ad[0]} / {ad[1]:+.2f} | {rv[0]} / {rv[1]:+.2f} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--ablation-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline = _load_baseline_trades(Path(args.baseline_root))
    ab = Path(args.ablation_root)
    exps = {"baseline": baseline}
    for tag in sorted([p.name for p in ab.iterdir() if p.is_dir()]):
        exps[tag] = _load_exp_trades(ab / tag)

    lines = ["# SRB Ablation — fast-replay summary\n"]
    lines.append(f"Baseline: `{args.baseline_root}`\n")
    lines.append(f"Ablation root: `{args.ablation_root}`\n\n")
    lines.append(
        "| tag | n | total_R | mean_R | win_rate | sl (n / mean_R) | "
        "trailing_sl (n / mean_R) | add (n / mean_R) | reverse (n / mean_R) |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    stats = {k: _stat(v) for k, v in exps.items()}
    base = stats.get("baseline", {})
    lines.append(_row("baseline", base) + "\n")
    for tag, s in stats.items():
        if tag == "baseline":
            continue
        lines.append(_row(tag, s) + "\n")

    # deltas
    if base:
        lines.append("\n## Δ vs baseline\n\n")
        lines.append(
            "| tag | Δn | ΔR | Δwin_rate | Δtrailing_sl mean_R | Δtrailing_sl count | Δadd mean_R | Δadd count |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for tag, s in stats.items():
            if tag == "baseline" or not s:
                continue
            dn = s["n"] - base["n"]
            dr = s["total_r"] - base["total_r"]
            dw = (s["win_rate"] - base["win_rate"]) * 100
            dtr_m = s["trailing_sl"][1] - base["trailing_sl"][1]
            dtr_n = s["trailing_sl"][0] - base["trailing_sl"][0]
            dad_m = s["add"][1] - base["add"][1]
            dad_n = s["add"][0] - base["add"][0]
            lines.append(
                f"| {tag} | {dn:+d} | {dr:+.2f} | {dw:+.1f}pp | {dtr_m:+.3f} | "
                f"{dtr_n:+d} | {dad_m:+.3f} | {dad_n:+d} |\n"
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(lines)
    print("".join(lines))
    print(f"\n[summary] written -> {args.out}")


if __name__ == "__main__":
    main()
