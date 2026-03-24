#!/usr/bin/env python3
"""Run production pipeline repeatedly and emit GO/NO-GO report."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class StrategyEval:
    strategy: str
    n: int
    sharpe_mean: float
    sharpe_std: float
    trades_mean: float
    trades_std: float
    trades_cv: float
    adopt_rate: float
    passed: bool
    reasons: List[str]
    reports: List[Dict[str, Any]]


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_pipeline(config: Path, end_date: str, runs: int) -> None:
    for i in range(1, runs + 1):
        print(f"\n===== RUN {i}/{runs} =====")
        cmd = [
            sys.executable,
            "scripts/auto_research_pipeline.py",
            "--config",
            str(config),
            "--all",
            "--end-date",
            end_date,
        ]
        subprocess.run(cmd, check=True)


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_strategy_reports(
    history_root: Path, strategy: str, end_date: str, runs: int
) -> List[Dict[str, Any]]:
    sdir = history_root / strategy
    if not sdir.exists():
        return []
    candidates: List[Dict[str, Any]] = []
    for report_path in sdir.glob("*/report.json"):
        try:
            rpt = _load_report(report_path)
        except Exception:
            continue
        if rpt.get("data_range", {}).get("end_date") == end_date:
            candidates.append(rpt)
    candidates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    # Use oldest->newest ordering for readability.
    return list(reversed(candidates[:runs]))


def evaluate_one_strategy(
    strategy: str,
    reports: List[Dict[str, Any]],
    *,
    runs: int,
    min_sharpe_mean: float,
    max_sharpe_std: float,
    min_trades_mean: float,
    max_trades_cv: float,
    min_adopt_rate: float,
) -> StrategyEval:
    reasons: List[str] = []
    if len(reports) < runs:
        reasons.append(f"insufficient_runs({len(reports)}/{runs})")

    sharps: List[float] = []
    trades: List[float] = []
    adopts = 0
    for rpt in reports:
        bm = rpt.get("backtest_metrics", {})
        cmp = rpt.get("comparison", {})
        sharps.append(float(bm.get("sharpe_per_trade", 0.0) or 0.0))
        trades.append(float(bm.get("total_trades", 0.0) or 0.0))
        if str(cmp.get("decision", "")).upper() == "ADOPT":
            adopts += 1

    sharpe_mean = _mean(sharps)
    sharpe_std = _std(sharps)
    trades_mean = _mean(trades)
    trades_std = _std(trades)
    trades_cv = (trades_std / trades_mean) if trades_mean > 0 else 0.0
    adopt_rate = (adopts / len(reports)) if reports else 0.0

    if sharpe_mean < min_sharpe_mean:
        reasons.append(f"sharpe_mean<{min_sharpe_mean:.3f}")
    if sharpe_std > max_sharpe_std:
        reasons.append(f"sharpe_std>{max_sharpe_std:.3f}")
    if trades_mean < min_trades_mean:
        reasons.append(f"trades_mean<{min_trades_mean:.1f}")
    if trades_cv > max_trades_cv:
        reasons.append(f"trades_cv>{max_trades_cv:.3f}")
    if adopt_rate < min_adopt_rate:
        reasons.append(f"adopt_rate<{min_adopt_rate:.2f}")

    passed = len(reasons) == 0
    return StrategyEval(
        strategy=strategy,
        n=len(reports),
        sharpe_mean=sharpe_mean,
        sharpe_std=sharpe_std,
        trades_mean=trades_mean,
        trades_std=trades_std,
        trades_cv=trades_cv,
        adopt_rate=adopt_rate,
        passed=passed,
        reasons=reasons,
        reports=reports,
    )


def render_markdown(
    *,
    end_date: str,
    config: Path,
    runs: int,
    thresholds: Dict[str, float],
    evaluations: List[StrategyEval],
    overall_go: bool,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Production GO/NO-GO Report",
        "",
        f"- Generated: `{now}`",
        f"- Config: `{config}`",
        f"- End date: `{end_date}`",
        f"- Runs: `{runs}`",
        "",
        "## Thresholds",
        "",
        f"- min_sharpe_mean: `{thresholds['min_sharpe_mean']}`",
        f"- max_sharpe_std: `{thresholds['max_sharpe_std']}`",
        f"- min_trades_mean: `{thresholds['min_trades_mean']}`",
        f"- max_trades_cv: `{thresholds['max_trades_cv']}`",
        f"- min_adopt_rate: `{thresholds['min_adopt_rate']}`",
        "",
        "## Strategy Summary",
        "",
        "| strategy | n | sharpe_mean | sharpe_std | trades_mean | trades_std | trades_cv | adopt_rate | pass | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]

    for e in evaluations:
        reasons = ",".join(e.reasons) if e.reasons else "-"
        lines.append(
            "| {s} | {n} | {sm:.4f} | {ss:.4f} | {tm:.1f} | {ts:.1f} | {cv:.3f} | {ar:.1%} | {p} | {r} |".format(
                s=e.strategy,
                n=e.n,
                sm=e.sharpe_mean,
                ss=e.sharpe_std,
                tm=e.trades_mean,
                ts=e.trades_std,
                cv=e.trades_cv,
                ar=e.adopt_rate,
                p="GO" if e.passed else "NO",
                r=reasons,
            )
        )

    lines.extend(["", "## Final Verdict", ""])
    lines.append("`GO`" if overall_go else "`NO-GO`")
    lines.append("")
    lines.append("## Latest Runs")
    lines.append("")
    lines.append("| strategy | run_idx | timestamp | sharpe | trades | decision |")
    lines.append("|---|---:|---|---:|---:|---|")
    for e in evaluations:
        for idx, rpt in enumerate(e.reports, start=1):
            bm = rpt.get("backtest_metrics", {})
            cmp = rpt.get("comparison", {})
            lines.append(
                "| {s} | {i} | {ts} | {sh:.4f} | {tr} | {dc} |".format(
                    s=e.strategy,
                    i=idx,
                    ts=rpt.get("timestamp", ""),
                    sh=float(bm.get("sharpe_per_trade", 0.0) or 0.0),
                    tr=int(float(bm.get("total_trades", 0.0) or 0.0)),
                    dc=str(cmp.get("decision", "")),
                )
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run prod pipeline N times, then emit GO/NO-GO report."
    )
    p.add_argument(
        "--config",
        default="config/prod_train_pipeline_2h.yaml",
        help="Pipeline config path.",
    )
    p.add_argument("--end-date", required=True, help="Pipeline end date (YYYY-MM-DD).")
    p.add_argument("--runs", type=int, default=3, help="Number of repeated runs.")
    p.add_argument(
        "--min-sharpe-mean", type=float, default=0.05, help="Minimum mean sharpe."
    )
    p.add_argument(
        "--max-sharpe-std", type=float, default=0.08, help="Maximum sharpe std."
    )
    p.add_argument(
        "--min-trades-mean", type=float, default=50.0, help="Minimum mean trades."
    )
    p.add_argument(
        "--max-trades-cv", type=float, default=0.35, help="Maximum trades CV."
    )
    p.add_argument(
        "--min-adopt-rate", type=float, default=0.67, help="Minimum ADOPT rate."
    )
    p.add_argument(
        "--output-file",
        default="",
        help="Optional output markdown report path.",
    )
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = load_config(config_path)
    history_root = Path(cfg["output"]["history_dir"])
    strategy_names = list(cfg["strategies"].keys())

    run_pipeline(config=config_path, end_date=args.end_date, runs=args.runs)

    evaluations: List[StrategyEval] = []
    for s in strategy_names:
        reports = collect_strategy_reports(
            history_root=history_root,
            strategy=s,
            end_date=args.end_date,
            runs=args.runs,
        )
        evaluations.append(
            evaluate_one_strategy(
                strategy=s,
                reports=reports,
                runs=args.runs,
                min_sharpe_mean=args.min_sharpe_mean,
                max_sharpe_std=args.max_sharpe_std,
                min_trades_mean=args.min_trades_mean,
                max_trades_cv=args.max_trades_cv,
                min_adopt_rate=args.min_adopt_rate,
            )
        )

    overall_go = all(e.passed for e in evaluations)
    thresholds = {
        "min_sharpe_mean": args.min_sharpe_mean,
        "max_sharpe_std": args.max_sharpe_std,
        "min_trades_mean": args.min_trades_mean,
        "max_trades_cv": args.max_trades_cv,
        "min_adopt_rate": args.min_adopt_rate,
    }
    report_md = render_markdown(
        end_date=args.end_date,
        config=config_path,
        runs=args.runs,
        thresholds=thresholds,
        evaluations=evaluations,
        overall_go=overall_go,
    )

    if args.output_file:
        out = Path(args.output_file)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = history_root / f"go_nogo_{args.end_date}_{ts}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_md, encoding="utf-8")

    verdict = "GO" if overall_go else "NO-GO"
    print(f"\nFinal verdict: {verdict}")
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
