#!/usr/bin/env python3
"""Generate docs/decisions/<topic>_<date>.md skeleton from EXPERIMENT_INDEX + capital reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_capital(dir_path: Path) -> Dict[str, Any]:
    cap = dir_path / "capital_report.json"
    if not cap.exists():
        return {}
    return json.loads(cap.read_text(encoding="utf-8"))


def _period_label(period: str) -> str:
    if "/" in period:
        a, b = period.split("/", 1)
        return f"{a} → {b}"
    return period


def build_markdown(
    *,
    topic: str,
    experiment_id: str,
    runs: List[Dict[str, Any]],
    promoted_variant: Optional[str],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# {topic.replace('_', ' ').title()}",
        "",
        f"- **日期**: {ts}",
        f"- **experiment_id**: `{experiment_id}`",
    ]
    if promoted_variant:
        lines.append(f"- **决策**: Promote **{promoted_variant}** (fill after review)")
    lines.extend(
        ["", "## 1. 变体定义", "", "| ID | strategies_root | 说明 |", "|---|---|---|"]
    )
    seen: set[str] = set()
    for run in runs:
        vid = str(run.get("variant", ""))
        if not vid or vid in seen:
            continue
        seen.add(vid)
        root = run.get("strategies_root") or "config/strategies"
        lines.append(f"| **{vid}** | `{root}` | _(fill)_ |")

    lines.extend(["", "## 2. Event backtest 结果", ""])
    by_period: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        period = str(run.get("period", "unknown"))
        by_period.setdefault(period, []).append(run)

    sec = 1
    for period, group in sorted(by_period.items()):
        lines.append(f"### 2.{sec} {_period_label(period)}")
        sec += 1
        lines.append("")
        lines.append("| 变体 | trades | totR | ret% | maxDD% | dir |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for run in group:
            vid = run.get("variant", "?")
            d = Path(str(run.get("dir", "")))
            if not d.is_absolute():
                d = (PROJECT_ROOT / d).resolve()
            cap = _load_capital(d)
            trades = cap.get("trades", "—")
            tot_r = cap.get("total_r")
            ret = cap.get("total_return")
            mdd = cap.get("max_drawdown_pct")
            tot_s = f"{tot_r:+.2f}" if tot_r is not None else "—"
            ret_s = f"{100 * float(ret):.2f}%" if ret is not None else "—"
            mdd_s = f"{100 * abs(float(mdd)):.2f}%" if mdd is not None else "—"
            lines.append(f"| {vid} | {trades} | {tot_s} | {ret_s} | {mdd_s} | `{d}` |")
        lines.append("")

    lines.extend(
        [
            "## 2.3 按 side 分解（placeholder）",
            "",
            "| 变体 | LONG totR | SHORT totR |",
            "|---|---:|---:|",
            "| _(fill from event_trades CSV)_ | | |",
            "",
            "## 3. 离线 label / IC（placeholder）",
            "",
            "- quick_layer_scan condition-set / ic-decay",
            "",
            "## 4. 决策",
            "",
            "- [ ] Promote variant: ___",
            "- [ ] Reject reason: ___",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Decision doc skeleton generator")
    p.add_argument("--experiment-index", required=True)
    p.add_argument(
        "--topic", required=True, help="Slug for docs/decisions/<topic>_<date>.md"
    )
    p.add_argument("--out", default=None, help="Override output path")
    p.add_argument("--date", default=None, help="YYYYMMDD suffix (default today UTC)")
    args = p.parse_args()

    idx_path = Path(args.experiment_index)
    if not idx_path.is_absolute():
        idx_path = (PROJECT_ROOT / idx_path).resolve()
    if not idx_path.exists():
        print(f"ERROR: index not found: {idx_path}", file=sys.stderr)
        return 3

    blob = json.loads(idx_path.read_text(encoding="utf-8"))
    runs = blob.get("runs") or []
    if not isinstance(runs, list):
        runs = []

    date_suffix = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
    out = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "docs" / "decisions" / f"{args.topic}_{date_suffix}.md"
    )
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    md = build_markdown(
        topic=args.topic,
        experiment_id=str(blob.get("experiment_id", args.topic)),
        runs=[r for r in runs if isinstance(r, dict)],
        promoted_variant=blob.get("promoted_variant"),
    )
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
