#!/usr/bin/env python3
"""Run multi-leg no-lookahead walk-forward replay via rolling_sim."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from scripts.pipeline.config import load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expand_months(spec: str) -> List[str]:
    raw = str(spec or "").strip()
    if not raw:
        return []
    if ":" not in raw:
        return [x.strip() for x in raw.replace(" ", ",").split(",") if x.strip()]
    left, right = [x.strip() for x in raw.split(":", 1)]
    sd = datetime.strptime(left + "-01", "%Y-%m-%d")
    ed = datetime.strptime(right + "-01", "%Y-%m-%d")
    if sd > ed:
        sd, ed = ed, sd
    out: List[str] = []
    cur = sd
    while cur <= ed:
        out.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return out


def _latest_rolling_root(history_dir: Path) -> Path:
    rs = history_dir / "_rolling_sim"
    if not rs.exists():
        raise FileNotFoundError(f"rolling root missing: {rs}")
    runs = [d for d in rs.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no rolling runs under: {rs}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _render_html(path: Path, report: Dict[str, object]) -> None:
    lines = [
        "<!doctype html>",
        "<html><meta charset='utf-8'><title>multileg replay</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:20px;max-width:1000px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px}"
        "th{background:#f4f4f4;text-align:left}</style>",
        "<h1>Multi-leg Replay Summary</h1>",
        f"<p>run: {report.get('run_root')}</p>",
        f"<p>ledger: {report.get('ledger_path')}</p>",
        f"<p>stitched: {report.get('stitched_summary_path')}</p>",
        "</html>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-leg walk-forward replay runner.")
    p.add_argument(
        "--config",
        default="config/pipelines/multileg_orchestrate_2h.yaml",
        help="multi-leg pipeline YAML",
    )
    p.add_argument(
        "--months",
        default="",
        help="YYYY-MM list or range (YYYY-MM:YYYY-MM)",
    )
    p.add_argument("--strategy", default="", help="single strategy (optional)")
    p.add_argument("--all", action="store_true", help="run all strategies in config")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--use-1min", action="store_true")
    p.add_argument("--live-root", default="live/highcap")
    p.add_argument("--end-date", default="")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_pipeline_config(cfg_path)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/auto_research_pipeline.py"),
        "--config",
        str(cfg_path),
        "--stage",
        "rolling_sim",
    ]
    if args.all:
        cmd.append("--all")
    elif str(args.strategy).strip():
        cmd.extend(["--strategy", str(args.strategy).strip()])
    else:
        cmd.append("--all")
    if str(args.end_date).strip():
        cmd.extend(["--end-date", str(args.end_date).strip()])
    months = _expand_months(args.months)
    if months:
        cmd.extend(["--month", ",".join(months)])
    if args.use_1min:
        cmd.append("--use-1min")
    if str(args.live_root).strip() and str(args.live_root).strip() != "live/highcap":
        cmd.extend(["--live-root", str(args.live_root).strip()])
    if args.dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        return int(proc.returncode)

    history_dir = PROJECT_ROOT / str(
        (cfg.get("output") or {}).get("history_dir", "") or ""
    )
    run_root = _latest_rolling_root(history_dir)
    ledger_path = run_root / "monthly_ledger.jsonl"
    stitched_path = run_root / "stitched_summary.json"
    report = {
        "run_root": str(run_root),
        "config": str(cfg_path),
        "months": months,
        "ledger_path": str(ledger_path),
        "stitched_summary_path": str(stitched_path),
    }
    out_json = run_root / "multileg_replay_summary.json"
    out_html = run_root / "multileg_replay_summary.html"
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _render_html(out_html, report)
    print(
        json.dumps(
            {"ok": True, "summary_json": str(out_json), "summary_html": str(out_html)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
