#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StrategyRun:
    name: str
    results_dir: Path
    suggested_yaml: Path
    base_dir: Path


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text_if_exists(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _md_link(label: str, path: Path) -> str:
    # use repo-relative paths for clickable links in Cursor
    rel = path.relative_to(ROOT).as_posix()
    return f"[{label}](`{rel}`)"


def main() -> int:
    now = datetime.now().isoformat(timespec="seconds")
    out = ROOT / "docs/architecture/树模型策略report/feature_group_search_summary.md"

    runs = [
        StrategyRun(
            name="sr_reversal_rr_reg_long",
            base_dir=ROOT / "config/strategies/sr_reversal_rr_reg_long",
            results_dir=ROOT / "results/feature_group_search/sr_reversal_rr_reg_long",
            suggested_yaml=ROOT
            / "config/strategies/sr_reversal_rr_reg_long/features_suggested.yaml",
        ),
        StrategyRun(
            name="sr_breakout",
            base_dir=ROOT / "config/strategies/sr_breakout",
            results_dir=ROOT / "results/feature_group_search/sr_breakout",
            suggested_yaml=ROOT
            / "config/strategies/sr_breakout/features_suggested.yaml",
        ),
        StrategyRun(
            name="compression_breakout",
            base_dir=ROOT / "config/strategies/compression_breakout",
            results_dir=ROOT / "results/feature_group_search/compression_breakout",
            suggested_yaml=ROOT
            / "config/strategies/compression_breakout/features_suggested.yaml",
        ),
        StrategyRun(
            name="trend_following",
            base_dir=ROOT / "config/strategies/trend_following",
            results_dir=ROOT / "results/feature_group_search/trend_following",
            suggested_yaml=ROOT
            / "config/strategies/trend_following/features_suggested.yaml",
        ),
    ]

    lines = []
    lines.append(f"## Feature Group Search Summary ({now})")
    lines.append("")
    lines.append(
        "This report summarizes the latest greedy feature-group-search runs and points to artifacts."
    )
    lines.append("")

    for r in runs:
        result_json = r.results_dir / "feature_group_search_result.json"
        history_csv = r.results_dir / "feature_group_search_history.csv"
        candidates_csv = r.results_dir / "feature_group_search_candidates.csv"
        report_html = r.results_dir / "feature_group_search_report.html"
        why_html = r.results_dir / "feature_group_search_why.html"

        lines.append(f"### {r.name}")
        lines.append(
            f"- **base_strategy**: `{r.base_dir.relative_to(ROOT).as_posix()}`"
        )
        lines.append(
            f"- **suggested_yaml**: `{r.suggested_yaml.relative_to(ROOT).as_posix()}`"
        )
        lines.append(
            f"- **results_dir**: `{r.results_dir.relative_to(ROOT).as_posix()}`"
        )
        links = []
        if result_json.exists():
            links.append(_md_link("result.json", result_json))
        if history_csv.exists():
            links.append(_md_link("history.csv", history_csv))
        if candidates_csv.exists():
            links.append(_md_link("candidates.csv", candidates_csv))
        if report_html.exists():
            links.append(_md_link("report.html", report_html))
        if why_html.exists():
            links.append(_md_link("why.html", why_html))
        if links:
            lines.append(f"- **artifacts**: " + " | ".join(links))
        else:
            lines.append("- **artifacts**: (not finished / missing)")

        obj = _read_json(result_json)
        if obj and (obj.get("best") or {}).get("summary"):
            best = obj["best"]
            summ = best["summary"]
            lines.append(
                f"- **best**: step={best.get('step')} groups={len(best.get('requested_groups') or [])} "
                f"requested_features={len(best.get('requested_features') or [])} "
                f"Sharpe_mean={summ.get('Sharpe_mean')} trades_mean={summ.get('trades_mean')}"
            )
            added = best.get("requested_groups") or []
            if added:
                lines.append(f"- **selected_groups**: {', '.join(added)}")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
