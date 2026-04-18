"""
Fast event-backtest replay for SRB execution-only ablations.

Why: full rolling_sim takes ~4.5h per knob because it re-calibrates prefilter /
gate / thresholds each month. Our ablations only toggle execution-layer knobs
(trailing, add-position, fake-break-reverse, passive feature injection), which
by construction leave signal calibration untouched. So we reuse the baseline's
per-month calibrated `threshold_calibration/strategies/srb/` folders and only
patch each month's execution.yaml, then rerun event_backtest.py for the month.

Input:
  --baseline-root  results/srb/slow-rolling-sim/_rolling_sim/20260417_163432
  --patch          exp1 | exp2 | exp3 | exp4 | exp23 | exp234
  --out-root       results/srb/diag/ablation_fast_20260418/<tag>/

For each fast_month_YYYY-MM under baseline:
  1. copy baseline strategies/srb/ to <out-root>/<tag>/month_YYYY-MM/strategies/srb/
  2. patch execution.yaml
  3. run event_backtest.py --strategy srb --strategies-root <copied> \
        --start-date YYYY-MM-01 --end-date YYYY-MM-END \
        --export <csv> --output <json>
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

ROOT = Path("/home/yin/trading/ml_trading_bot")


def _patch_execution_yaml(path: Path, patch: str) -> None:
    src = path.read_text()

    if "wide" in patch:
        if "sr_feature_injection:" not in src:
            inj = (
                "\n# 宽窗 SR (wide swing) — ablation: passive injection only\n"
                "sr_feature_injection:\n"
                "  swing_lookback_wide_bars: 96\n\n"
            )
            src = src.replace("add_position:", inj + "add_position:", 1)

    if "adaptive_atr" in patch:
        # calibrated yaml has sorted keys under trailing: (activation_r, enabled, trail_r)
        # so we target any key inside that block. Insert after trailing: header.
        if "expand_with_primary_atr" not in src:
            src = re.sub(
                r"(\n  trailing:\n)",
                r"\1    expand_with_primary_atr: true\n",
                src,
                count=1,
            )

    if "wider_defaults" in patch:
        src = re.sub(r"    activation_r: \d+(?:\.\d+)?", "    activation_r: 7.0", src)
        src = re.sub(r"    trail_r: \d+(?:\.\d+)?", "    trail_r: 6.0", src)

    if "low_adx_high_er" in patch:
        # NOTE: the calibrated yaml already contains "low_adx_high_er:" under
        # regime_execution.buckets (that's just a bucket-params key; unrelated
        # to whether we allow adds in that regime). So guard on the SECTION we
        # want to modify (allow_regime_buckets list) rather than a generic
        # substring check.
        m = re.search(
            r"(allow_regime_buckets:\n(?:  - [^\n]+\n)+)",
            src,
        )
        if m and "- low_adx_high_er" not in m.group(0):
            src = src.replace(m.group(0), m.group(0) + "  - low_adx_high_er\n", 1)

    if "wide_sr_fallback_2atr" in patch:
        # inject fake_break_reverse.true_sr_wide_fallback_atr: 2.0
        if "true_sr_wide_fallback_atr" not in src:
            src = re.sub(
                r"(fake_break_reverse:\n)",
                r"\1  true_sr_wide_fallback_atr: 2.0\n",
                src,
                count=1,
            )

    if "wide_sr_entry_guard_2atr" in patch:
        # add top-level sr_wide_entry_guard block; must co-exist with
        # sr_feature_injection (this patch implies wide SR injection).
        if "sr_wide_entry_guard" not in src:
            block = (
                "\nsr_wide_entry_guard:\n"
                "  enabled: true\n"
                "  min_distance_atr: 2.0\n"
            )
            src += block
        # also ensure wide SR is injected (if not already)
        if "sr_feature_injection:" not in src:
            src = "sr_feature_injection:\n  swing_lookback_wide_bars: 96\n\n" + src

    path.write_text(src)


PATCH_MAP = {
    "exp1": ["wide"],
    "exp2": ["adaptive_atr"],
    "exp3": ["wider_defaults"],
    "exp4": ["low_adx_high_er"],
    "exp2_wide": ["wide", "adaptive_atr"],
    "exp23": ["adaptive_atr", "wider_defaults"],
    "exp234": ["adaptive_atr", "wider_defaults", "low_adx_high_er"],
    "exp1234": ["wide", "adaptive_atr", "wider_defaults", "low_adx_high_er"],
    "baseline_replay": [],  # sanity check: should match baseline exactly
    # Round 2: on top of exp2 (adaptive ATR, now the default)
    # exp4a: adaptive trailing + low_adx_high_er add bucket
    "exp4a": ["adaptive_atr", "low_adx_high_er"],
    # exp4b: adaptive trailing + wide SR fallback for true_sr_level (2 ATR threshold)
    "exp4b": ["adaptive_atr", "wide", "wide_sr_fallback_2atr"],
    # exp4c: adaptive trailing + wide SR prefilter (block entry within 2 ATR of opposing wide SR)
    "exp4c": ["adaptive_atr", "wide", "wide_sr_entry_guard_2atr"],
    # sanity / new baseline = adaptive_atr alone with wide SR injected (matches committed execution.yaml)
    "new_baseline": ["adaptive_atr", "wide"],
}


def _month_bounds(ym: str) -> tuple[str, str]:
    y, m = ym.split("-")
    y_i, m_i = int(y), int(m)
    last = calendar.monthrange(y_i, m_i)[1]
    return f"{ym}-01", f"{ym}-{last:02d}"


def _run_one_month(
    month_parent: Path,
    out_month_dir: Path,
    patches: List[str],
    ym: str,
    resume_state_path: str = "",
    keep_open: bool = True,
) -> dict:
    """
    month_parent  = .../fast_month_YYYY-MM  (contains strategies_calibrated/)
    """
    src_strat = month_parent / "strategies_calibrated"
    strategies_root = out_month_dir / "strategies_calibrated"
    if strategies_root.exists():
        shutil.rmtree(strategies_root)
    shutil.copytree(src_strat, strategies_root)

    exec_yaml = strategies_root / "srb" / "archetypes" / "execution.yaml"
    for p in patches:
        _patch_execution_yaml(exec_yaml, p)

    start, end = _month_bounds(ym)
    csv_path = out_month_dir / f"event_trades_srb.csv"
    json_path = out_month_dir / f"event_backtest_srb.json"
    log_path = out_month_dir / f"event_backtest.log"
    end_state_path = out_month_dir / f"end_state.json"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "event_backtest.py"),
        "--strategy",
        "srb",
        "--start-date",
        start,
        "--end-date",
        end,
        "--strategies-root",
        str(strategies_root),
        "--data-path",
        "data/parquet_data",
        "--export",
        str(csv_path),
        "--output",
        str(json_path),
        "--fast",
        "--dump-end-state",
        str(end_state_path),
    ]
    if keep_open:
        cmd.append("--keep-open-positions")
    if resume_state_path:
        cmd += ["--resume-state", resume_state_path]
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return {
        "month": ym,
        "exit_code": proc.returncode,
        "csv": str(csv_path),
        "json": str(json_path),
        "end_state": str(end_state_path),
    }


def run(baseline_root: Path, patch_tag: str, out_root: Path, workers: int) -> None:
    """
    Important: rolling_sim carries state month→month via --resume-state.
    Replay must match that to be comparable, which forces SEQUENTIAL execution.
    (workers param kept for debugging but normally use workers=1.)
    """
    patches = PATCH_MAP[patch_tag]
    tag_dir = out_root / patch_tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    month_dirs = sorted(
        [p for p in baseline_root.iterdir() if p.name.startswith("fast_month_")]
    )
    if _ONLY_MONTHS:
        month_dirs = [
            m for m in month_dirs if m.name.replace("fast_month_", "") in _ONLY_MONTHS
        ]
    print(
        f"[replay] {patch_tag}: {len(month_dirs)} months, patches={patches}, workers={workers}"
    )

    if workers <= 1:
        # sequential: carry end_state month→month
        prev_end_state = ""
        # NOTE: to match baseline rolling_sim semantics, the "last month" in the
        # original 16-month rolling was 2024-12 (where force_close was applied).
        # For partial reruns (--only-months), we should NOT force-close — keep
        # open positions so comparisons against baseline CSV (which excludes
        # open positions) stay on equal footing.
        _full_range = len(month_dirs) == 16 and not _ONLY_MONTHS
        for idx, md in enumerate(month_dirs):
            ym = md.name.replace("fast_month_", "")
            out_month = tag_dir / f"month_{ym}"
            out_month.mkdir(parents=True, exist_ok=True)
            is_last = _full_range and idx == len(month_dirs) - 1
            r = _run_one_month(
                md,
                out_month,
                patches,
                ym,
                resume_state_path=prev_end_state,
                keep_open=not is_last,
            )
            print(
                f"  [{r['month']}] exit={r['exit_code']}"
                + (" OK" if r["exit_code"] == 0 else " FAIL")
            )
            prev_end_state = r["end_state"] if Path(r["end_state"]).exists() else ""
    else:
        # parallel: each month starts fresh (no resume) — ONLY for debugging
        jobs = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for md in month_dirs:
                ym = md.name.replace("fast_month_", "")
                out_month = tag_dir / f"month_{ym}"
                out_month.mkdir(parents=True, exist_ok=True)
                jobs.append(
                    ex.submit(_run_one_month, md, out_month, patches, ym, "", True)
                )
            for fut in as_completed(jobs):
                r = fut.result()
                print(
                    f"  [{r['month']}] exit={r['exit_code']}"
                    + (" OK" if r["exit_code"] == 0 else " FAIL")
                )

    agg = tag_dir / "summary.json"
    totals = {
        "n_trades": 0,
        "total_r": 0.0,
        "n_sl": 0,
        "n_trailing_sl": 0,
        "n_eob": 0,
        "n_reverse": 0,
        "n_add": 0,
    }
    per_month = []
    for md in month_dirs:
        ym = md.name.replace("fast_month_", "")
        j = tag_dir / f"month_{ym}" / "event_backtest_srb.json"
        if not j.exists():
            per_month.append({"month": ym, "missing": True})
            continue
        data = json.loads(j.read_text())
        trades = data.get("trades", [])
        totals["n_trades"] += len(trades)
        totals["total_r"] += sum(float(t.get("pnl_r", 0.0)) for t in trades)
        for t in trades:
            r = t.get("exit_reason", "")
            if r == "sl":
                totals["n_sl"] += 1
            elif r == "trailing_sl":
                totals["n_trailing_sl"] += 1
            elif r == "end_of_backtest":
                totals["n_eob"] += 1
            if t.get("is_reverse"):
                totals["n_reverse"] += 1
            if t.get("is_add_position"):
                totals["n_add"] += 1
        per_month.append(
            {
                "month": ym,
                "n_trades": len(trades),
                "total_r": round(sum(float(t.get("pnl_r", 0.0)) for t in trades), 3),
            }
        )
    totals["total_r"] = round(totals["total_r"], 3)
    agg.write_text(
        json.dumps(
            {"patch_tag": patch_tag, "totals": totals, "per_month": per_month}, indent=2
        )
    )
    print("=== totals ===")
    print(json.dumps(totals, indent=2))


_ONLY_MONTHS: set = set()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--patch", required=True, choices=list(PATCH_MAP.keys()))
    ap.add_argument("--out-root", default="results/srb/diag/ablation_fast_20260418")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--only-months",
        default="",
        help="comma-separated YYYY-MM list; run only these months",
    )
    args = ap.parse_args()
    if args.only_months:
        _ONLY_MONTHS = {m.strip() for m in args.only_months.split(",") if m.strip()}
    run(Path(args.baseline_root), args.patch, Path(args.out_root), args.workers)
