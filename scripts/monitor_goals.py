#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StrategyStatus:
    name: str
    suggested_yaml: Path
    results_dir: Path


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_mtime(path: Path) -> str:
    if not path.exists():
        return "missing"
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _summarize_feature_group_search(result_json: Path) -> str:
    obj = _read_json(result_json)
    if not obj:
        return "result_json: unreadable"
    best = obj.get("best") or {}
    summary = best.get("summary") or {}
    return (
        f"best_step={best.get('step')} "
        f"groups={len(best.get('requested_groups') or [])} "
        f"requested_features={len(best.get('requested_features') or [])} "
        f"Sharpe_mean={summary.get('Sharpe_mean')} "
        f"trades_mean={summary.get('trades_mean')}"
    )


def main() -> int:
    now = datetime.now().isoformat(timespec="seconds")
    out = ROOT / "docs" / "architecture" / "reports" / "progress_monitor.md"

    # For sr_reversal_rr_reg_long we may run both:
    # - expanded semantic singleton search (large search space)
    # - a faster group-level search (smaller search space)
    strategies = [
        StrategyStatus(
            name="sr_reversal_rr_reg_long_quick",
            suggested_yaml=ROOT
            / "config/strategies/sr_reversal_rr_reg_long/features_suggested_quick.yaml",
            results_dir=ROOT
            / "results/feature_group_search/sr_reversal_rr_reg_long_quick",
        ),
        StrategyStatus(
            name="sr_breakout_quick",
            suggested_yaml=ROOT
            / "config/strategies/sr_breakout/features_suggested_quick.yaml",
            results_dir=ROOT / "results/feature_group_search/sr_breakout_quick",
        ),
        StrategyStatus(
            name="compression_breakout_quick",
            suggested_yaml=ROOT
            / "config/strategies/compression_breakout/features_suggested_quick.yaml",
            results_dir=ROOT
            / "results/feature_group_search/compression_breakout_quick",
        ),
        StrategyStatus(
            name="trend_following_quick",
            suggested_yaml=ROOT
            / "config/strategies/trend_following/features_suggested_quick.yaml",
            results_dir=ROOT / "results/feature_group_search/trend_following_quick",
        ),
    ]

    lines = []
    lines.append(f"## Progress Monitor ({now})")
    lines.append("")
    # Goal 1: global normalization
    global_report = ROOT / "docs/architecture/树模型策略report/norm_contract_global.md"
    lines.append("### Goal 1 — Global normalization contract")
    if global_report.exists():
        head = global_report.read_text(encoding="utf-8").splitlines()[:10]
        raw_line = next((x for x in head if "raw_columns" in x), None)
        lines.append(
            f"- **norm_contract_global.md**: mtime={_file_mtime(global_report)}; {raw_line}"
        )
    else:
        lines.append("- **norm_contract_global.md**: missing")
    lines.append("")

    lines.append("### Goal 2/3 — Feature group search (4 strategies)")
    for s in strategies:
        result_json = s.results_dir / "feature_group_search_result.json"
        status = []
        status.append(f"suggested_yaml_mtime={_file_mtime(s.suggested_yaml)}")
        status.append(f"result_json_mtime={_file_mtime(result_json)}")
        if result_json.exists():
            status.append(_summarize_feature_group_search(result_json))
        lines.append(f"- **{s.name}**: " + "; ".join(status))
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
