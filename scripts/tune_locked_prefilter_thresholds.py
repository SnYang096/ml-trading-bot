#!/usr/bin/env python3
"""
Tune locked prefilter thresholds with rolling windows.

This script keeps locked semantic features fixed, but searches threshold values
on validation/test windows by repeatedly running auto_research_pipeline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.locked_prefilter_utils import (
    build_override_prefilter as build_override_prefilter_base,
)


@dataclass
class CaseParams:
    fer_lower: float
    fer_upper: float
    sr_min: float
    dist_max: float

    def key(self) -> str:
        return (
            f"fer[{self.fer_lower:.4g},{self.fer_upper:.4g}]_"
            f"sr>={self.sr_min:.4g}_dist<=±{self.dist_max:.4g}"
        )


def parse_float_list(text: str) -> List[float]:
    vals = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    return vals


def parse_text_list(text: str) -> List[str]:
    out = []
    for part in text.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def list_strategy_runs(history_dir: Path, strategy: str) -> List[str]:
    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        return []
    return sorted([p.name for p in strat_dir.iterdir() if p.is_dir()])


def build_override_prefilter(
    prod_prefilter_path: Path, params: CaseParams, output_path: Path
) -> Path:
    return build_override_prefilter_base(
        prod_prefilter_path,
        output_path,
        {
            "fer_lower": params.fer_lower,
            "fer_upper": params.fer_upper,
            "sr_min": params.sr_min,
            "dist_max": params.dist_max,
        },
    )


def run_one_window(
    strategy: str,
    config_path: Path,
    end_date: str,
    override_prefilter: Path,
    history_dir: Path,
    skip_shap: bool = False,
) -> Dict[str, Any]:
    before = set(list_strategy_runs(history_dir, strategy))
    cmd = [
        sys.executable,
        "scripts/auto_research_pipeline.py",
        "--strategy",
        strategy,
        "--config",
        str(config_path),
        "--no-adopt",
        "--locked-prefilter-override",
        str(override_prefilter),
    ]
    if end_date:
        cmd.extend(["--end-date", end_date])
    if skip_shap:
        cmd.append("--skip-shap")
    cmd.append("--disable-auto-locked-tuning")

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    after = set(list_strategy_runs(history_dir, strategy))
    new_runs = sorted(after - before)
    run_id = new_runs[-1] if new_runs else None
    report = None
    metrics: Dict[str, Any] = {}
    decision = "ERROR"

    if run_id:
        rp = history_dir / strategy / run_id / "report.json"
        if rp.exists():
            report = rp
            data = json.loads(rp.read_text(encoding="utf-8"))
            metrics = data.get("backtest_metrics", {}) or {}
            decision = (data.get("comparison", {}) or {}).get("decision", "UNKNOWN")

    return {
        "return_code": proc.returncode,
        "run_id": run_id,
        "report_path": str(report) if report else "",
        "decision": decision,
        "metrics": metrics,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def aggregate_case(
    case_results: List[Dict[str, Any]],
    min_trades_target: int,
    trade_penalty: float,
) -> Dict[str, Any]:
    sharpes: List[float] = []
    trades: List[int] = []
    ok_windows = 0

    for r in case_results:
        m = r.get("metrics") or {}
        if not m:
            continue
        s = m.get("sharpe_per_trade")
        t = m.get("total_trades")
        if isinstance(s, (float, int)) and isinstance(t, (float, int)):
            sharpes.append(float(s))
            trades.append(int(t))
            ok_windows += 1

    if not sharpes:
        return {
            "ok_windows": 0,
            "median_sharpe": float("-inf"),
            "positive_ratio": 0.0,
            "median_trades": 0.0,
            "score": float("-inf"),
        }

    median_sharpe = statistics.median(sharpes)
    positive_ratio = sum(1 for s in sharpes if s > 0) / len(sharpes)
    median_trades = float(statistics.median(trades))
    trade_gap = max(0.0, float(min_trades_target) - median_trades)
    score = median_sharpe - trade_penalty * trade_gap

    return {
        "ok_windows": ok_windows,
        "median_sharpe": median_sharpe,
        "positive_ratio": positive_ratio,
        "median_trades": median_trades,
        "score": score,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Tune locked prefilter thresholds")
    p.add_argument("--strategy", default="fer-short")
    p.add_argument("--config", default="config/research_pipeline.yaml")
    p.add_argument(
        "--end-dates",
        default="",
        help="comma-separated end dates; empty means single latest-date run",
    )
    p.add_argument("--fer-lower-values", default="0.0,0.05")
    p.add_argument("--fer-upper-values", default="0.25,0.35,0.45")
    p.add_argument("--sr-min-values", default="0.45,0.55,0.65")
    p.add_argument("--dist-max-values", default="0.8,1.2,1.6")
    p.add_argument("--max-cases", type=int, default=0, help="0 means all")
    p.add_argument("--min-trades-target", type=int, default=60)
    p.add_argument("--trade-penalty", type=float, default=0.002)
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--output-dir",
        default="results/locked_tuning",
        help="where tuning summary files are saved",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    scfg = (cfg.get("strategies", {}) or {}).get(args.strategy, {})
    if not scfg:
        raise SystemExit(f"unknown strategy: {args.strategy}")

    prod_cfg_dir = PROJECT_ROOT / scfg["config"]
    prod_prefilter = prod_cfg_dir / "archetypes" / "prefilter.yaml"
    if not prod_prefilter.exists():
        raise SystemExit(f"prefilter not found: {prod_prefilter}")

    out_root = PROJECT_ROOT / args.output_dir / args.strategy
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    history_dir = PROJECT_ROOT / (cfg.get("output", {}) or {}).get(
        "history_dir", "results/research_history"
    )

    end_dates = parse_text_list(args.end_dates)
    if not end_dates:
        # Empty means let pipeline auto-detect latest end date once.
        end_dates = [""]

    fer_lows = parse_float_list(args.fer_lower_values)
    fer_ups = parse_float_list(args.fer_upper_values)
    sr_mins = parse_float_list(args.sr_min_values)
    dist_maxs = parse_float_list(args.dist_max_values)

    cases: List[CaseParams] = []
    for lo, hi, sr, dist in itertools.product(fer_lows, fer_ups, sr_mins, dist_maxs):
        if lo > hi:
            continue
        cases.append(CaseParams(fer_lower=lo, fer_upper=hi, sr_min=sr, dist_max=dist))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    print("=" * 90)
    print(f"🔬 Locked Prefilter Threshold Tuning: {args.strategy}")
    print(f"Cases={len(cases)}, Windows={len(end_dates)}, dry_run={args.dry_run}")
    print("=" * 90)

    summary_rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        print(f"\n[{idx}/{len(cases)}] {case.key()}")
        case_dir = out_dir / f"case_{idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        override_path = build_override_prefilter(
            prod_prefilter, case, case_dir / "prefilter_locked_override.yaml"
        )

        per_window: List[Dict[str, Any]] = []
        for win_idx, end_date in enumerate(end_dates, 1):
            if args.dry_run:
                r = {
                    "window": win_idx,
                    "end_date": end_date or "<auto>",
                    "return_code": 0,
                    "run_id": "",
                    "decision": "DRY_RUN",
                    "metrics": {},
                }
            else:
                print(
                    f"  - window {win_idx}/{len(end_dates)} end_date={end_date or '<auto>'}"
                )
                r = run_one_window(
                    strategy=args.strategy,
                    config_path=cfg_path,
                    end_date=end_date if end_date else "",
                    override_prefilter=override_path,
                    history_dir=history_dir,
                    skip_shap=args.skip_shap,
                )
            per_window.append(r)

        agg = aggregate_case(
            per_window,
            min_trades_target=args.min_trades_target,
            trade_penalty=args.trade_penalty,
        )
        row = {
            "case_id": idx,
            "fer_lower": case.fer_lower,
            "fer_upper": case.fer_upper,
            "sr_min": case.sr_min,
            "dist_max": case.dist_max,
            **agg,
            "windows": per_window,
        }
        summary_rows.append(row)
        print(
            f"  => score={row['score']:+.4f}, median_sharpe={row['median_sharpe']:+.4f}, "
            f"positive_ratio={row['positive_ratio']:.1%}, median_trades={row['median_trades']:.1f}"
        )

    summary_rows.sort(key=lambda x: x["score"], reverse=True)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "strategy": args.strategy,
                "config": str(cfg_path),
                "end_dates": end_dates,
                "min_trades_target": args.min_trades_target,
                "trade_penalty": args.trade_penalty,
                "rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "fer_lower",
                "fer_upper",
                "sr_min",
                "dist_max",
                "score",
                "median_sharpe",
                "positive_ratio",
                "median_trades",
                "ok_windows",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["case_id"],
                    r["fer_lower"],
                    r["fer_upper"],
                    r["sr_min"],
                    r["dist_max"],
                    r["score"],
                    r["median_sharpe"],
                    r["positive_ratio"],
                    r["median_trades"],
                    r["ok_windows"],
                ]
            )

    print("\nTop 5 cases:")
    for i, r in enumerate(summary_rows[:5], 1):
        print(
            f"  {i}. case={r['case_id']:03d} score={r['score']:+.4f} "
            f"sharpe={r['median_sharpe']:+.4f} trades={r['median_trades']:.1f} "
            f"fer=[{r['fer_lower']:.3g},{r['fer_upper']:.3g}] sr>={r['sr_min']:.3g} dist<=±{r['dist_max']:.3g}"
        )

    print(f"\n✅ Saved: {summary_json}")
    print(f"✅ Saved: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
