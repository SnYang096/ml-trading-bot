#!/usr/bin/env python3
"""
Summarize historical feature-group-search runs across strategies.

Outputs a single Markdown report that includes:
- selected_groups (group-level selection)
- final_features (model feature columns)
- invert_features (column-level inversions)
- objective score (when available)
- paths to result JSON and writeback YAML

This script is intentionally "repo-local" (no external deps beyond PyYAML).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import re


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "feature_group_search"
STRATEGIES_ROOT = ROOT / "config" / "strategies"
DOCS_ROOT = ROOT / "docs" / "strategies"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _safe_list(v) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _fmt_list(items: List[str], max_items: int = 30) -> str:
    items = [str(x) for x in items if str(x).strip()]
    if not items:
        return "-"
    if len(items) <= max_items:
        return ", ".join(items)
    head = ", ".join(items[:max_items])
    return f"{head}, … (+{len(items) - max_items})"


def _infer_strategy_from_dir(dir_name: str, known: List[str]) -> Optional[str]:
    # pick the longest matching prefix so sr_reversal_rr_reg_long wins over sr_reversal
    matches = [s for s in known if dir_name.startswith(s)]
    if not matches:
        return None
    return sorted(matches, key=len, reverse=True)[0]


def _extract_run_date_yyyymmdd(run_name: str) -> Optional[int]:
    """
    Extract YYYYMMDD token from run name, if present.
    We use the *last* occurrence to align with tags like:
      sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide_retry1
    """
    m = re.findall(r"(\d{8})", run_name or "")
    if not m:
        return None
    try:
        return int(m[-1])
    except Exception:
        return None


def _score_from_result_json(d: Dict[str, Any]) -> Optional[float]:
    # Prefer prune.score (post-beam prune result), else beam.score, else baseline.score.
    prune = d.get("prune") or {}
    if isinstance(prune, dict) and isinstance(prune.get("score"), (int, float)):
        return float(prune["score"])
    beam = d.get("beam") or {}
    if isinstance(beam, dict) and isinstance(beam.get("score"), (int, float)):
        return float(beam["score"])
    baseline = d.get("baseline") or {}
    if isinstance(baseline, dict) and isinstance(baseline.get("score"), (int, float)):
        return float(baseline["score"])
    return None


def _objective_from_result_json(d: Dict[str, Any]) -> Optional[str]:
    # objective is embedded in summaries; also present in YAML writeback.
    # Use algo_params if present? Keep simple: None here, YAML will fill.
    return None


@dataclass
class RunEntry:
    strategy: str
    run_dir_name: str
    status: str  # completed|pending|unknown
    result_json: Optional[Path] = None
    writeback_yaml: Optional[Path] = None

    # extracted
    search_algo: Optional[str] = None
    objective: Optional[str] = None
    score: Optional[float] = None
    selected_groups: List[str] = None
    final_features: List[str] = None
    invert_features: List[str] = None
    base_features: List[str] = None
    stop_reason: Optional[str] = None
    pool_b_yaml: Optional[str] = None
    groups_source: Optional[str] = None
    is_latest_poolb_semantic: bool = False

    def __post_init__(self):
        self.selected_groups = self.selected_groups or []
        self.final_features = self.final_features or []
        self.invert_features = self.invert_features or []
        self.base_features = self.base_features or []


def _extract_from_writeback_yaml(entry: RunEntry) -> None:
    if not entry.writeback_yaml or not entry.writeback_yaml.exists():
        return
    y = _load_yaml(entry.writeback_yaml)
    fp = y.get("feature_pipeline") or {}
    fgs = y.get("feature_group_search") or {}

    entry.invert_features = _safe_list(fp.get("invert_features"))
    entry.final_features = _safe_list(fgs.get("final_features")) or _safe_list(
        fp.get("requested_features")
    )
    entry.base_features = _safe_list(fgs.get("base_features"))
    entry.selected_groups = _safe_list(fgs.get("selected_groups"))
    entry.objective = fgs.get("objective") or entry.objective
    entry.stop_reason = fgs.get("stop_reason") or entry.stop_reason
    entry.pool_b_yaml = fgs.get("pool_b_yaml") or entry.pool_b_yaml
    entry.groups_source = fgs.get("groups_source") or entry.groups_source


def _extract_from_result_json(entry: RunEntry) -> None:
    if not entry.result_json or not entry.result_json.exists():
        return
    d = _load_json(entry.result_json)
    entry.search_algo = d.get("search_algo") or entry.search_algo
    entry.score = _score_from_result_json(d)
    entry.selected_groups = (
        _safe_list(d.get("selected_groups")) or entry.selected_groups
    )
    entry.final_features = _safe_list(d.get("final_features")) or entry.final_features
    entry.stop_reason = d.get("stop_reason") or entry.stop_reason


def collect_runs() -> List[RunEntry]:
    known_strategies = sorted([p.name for p in STRATEGIES_ROOT.iterdir() if p.is_dir()])
    runs: List[RunEntry] = []

    if RESULTS_ROOT.exists():
        for d in sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()]):
            strategy = _infer_strategy_from_dir(d.name, known_strategies) or "unknown"
            result_json = d / "feature_group_search_result.json"
            status = "completed" if result_json.exists() else "pending"

            # Map to writeback YAML convention: config/strategies/<strategy>/features_suggested_<suffix>.yaml
            writeback_yaml = None
            if strategy != "unknown":
                suffix = (
                    d.name[len(strategy) + 1 :]
                    if d.name.startswith(strategy + "_")
                    else None
                )
                if suffix:
                    cand = (
                        STRATEGIES_ROOT / strategy / f"features_suggested_{suffix}.yaml"
                    )
                    if cand.exists():
                        writeback_yaml = cand

            runs.append(
                RunEntry(
                    strategy=strategy,
                    run_dir_name=d.name,
                    status=status,
                    result_json=result_json if result_json.exists() else None,
                    writeback_yaml=writeback_yaml,
                )
            )

    # Also include writeback YAMLs that have feature_group_search metadata even if results dir missing.
    for yml in STRATEGIES_ROOT.rglob("features_suggested*.yaml"):
        try:
            y = _load_yaml(yml)
        except Exception:
            continue
        if "feature_group_search" not in (y or {}):
            continue
        strategy = yml.parent.name
        stem = yml.stem  # features_suggested_...
        suffix = stem.replace("features_suggested_", "", 1)
        run_dir_name = f"{strategy}_{suffix}" if suffix else strategy
        # Avoid duplicates
        if any(r.run_dir_name == run_dir_name for r in runs):
            continue
        runs.append(
            RunEntry(
                strategy=strategy,
                run_dir_name=run_dir_name,
                status="unknown",
                result_json=None,
                writeback_yaml=yml,
            )
        )

    # Enrich
    for r in runs:
        _extract_from_writeback_yaml(r)
        _extract_from_result_json(r)
        # Known superseded runs (manual policy):
        # - tf_solo2 was interrupted/superseded by the tz-fix rerun (tf_solo3_tzfix)
        if (
            r.status != "completed"
            and r.strategy == "trend_following"
            and "tf_solo2" in r.run_dir_name
            and r.result_json is None
        ):
            r.status = "cancelled"
        if r.status == "unknown":
            r.status = (
                "completed"
                if r.result_json
                else "completed" if r.writeback_yaml else "unknown"
            )

    # Mark latest poolb+semantic run per strategy (based on YYYYMMDD token when available)
    by_strategy: Dict[str, List[RunEntry]] = {}
    for r in runs:
        by_strategy.setdefault(r.strategy, []).append(r)

    for strat, items in by_strategy.items():
        cands = [r for r in items if "poolb_semantic" in (r.run_dir_name or "")]
        if not cands:
            continue

        def _key(r: RunEntry):
            d = _extract_run_date_yyyymmdd(r.run_dir_name) or -1
            # tie-breakers: prefer completed over pending/cancelled; then lexicographic
            status_rank = {
                "completed": 2,
                "pending": 1,
                "cancelled": 0,
                "unknown": 0,
            }.get(r.status, 0)
            return (d, status_rank, r.run_dir_name)

        latest = sorted(cands, key=_key, reverse=True)[0]
        latest.is_latest_poolb_semantic = True

    # Sort: by strategy then run_dir_name
    runs.sort(key=lambda x: (x.strategy, x.run_dir_name))
    return runs


def render_markdown(runs: List[RunEntry]) -> str:
    lines: List[str] = []
    lines.append("## Feature Group Search History (Auto Summary)")
    lines.append("")
    lines.append(
        "This document is **auto-generated** from `results/feature_group_search/**/feature_group_search_result.json` "
        "and writeback YAMLs under `config/strategies/*/features_suggested*.yaml`."
    )
    lines.append("")
    lines.append("Regenerate:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/summarize_feature_group_search_history.py")
    lines.append("```")
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- **selected_groups** are *feature-group nodes* (semantic groups / Pool-B candidates)."
    )
    lines.append(
        "- **final_features** are *model columns* (after expanding node outputs)."
    )
    lines.append(
        "- **invert_features** are *column-level inversions* applied by `feature_pipeline.invert_features`."
    )
    lines.append(
        "- **status**: `completed` means result JSON exists; `pending` means directory exists but no final result yet; `cancelled` means the run was intentionally stopped/superseded."
    )
    lines.append(
        "- **latest_poolb_semantic** marks the latest run (per strategy) whose name contains `poolb_semantic`."
    )
    lines.append("")
    lines.append("## Latest Pool-B + Semantic run per strategy")
    lines.append("")
    lines.append(
        "| strategy | run_dir | status | score | selected_groups | final_features | invert_features |"
    )
    lines.append("|---|---|---|---:|---|---|---|")
    for r in [x for x in runs if x.is_latest_poolb_semantic]:
        score = f"{r.score:.6f}" if isinstance(r.score, (int, float)) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r.strategy}`",
                    f"`{r.run_dir_name}`",
                    r.status,
                    score,
                    _fmt_list(r.selected_groups),
                    _fmt_list(r.final_features),
                    _fmt_list(r.invert_features),
                ]
            )
            + " |"
        )
    lines.append("")

    by_strategy: Dict[str, List[RunEntry]] = {}
    for r in runs:
        by_strategy.setdefault(r.strategy, []).append(r)

    for strategy, items in by_strategy.items():
        lines.append(f"## Strategy: `{strategy}`")
        lines.append("")
        lines.append(
            "| run_dir | status | latest_poolb_semantic | search_algo | objective | score | selected_groups | final_features | invert_features |"
        )
        lines.append("|---|---|---|---|---|---:|---|---|---|")
        for r in items:
            score = f"{r.score:.6f}" if isinstance(r.score, (int, float)) else ""
            latest_flag = "✅" if r.is_latest_poolb_semantic else ""
            if r.is_latest_poolb_semantic and r.status == "pending":
                latest_flag = "⏳"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{r.run_dir_name}`",
                        r.status,
                        latest_flag,
                        r.search_algo or "",
                        r.objective or "",
                        score,
                        _fmt_list(r.selected_groups),
                        _fmt_list(r.final_features),
                        _fmt_list(r.invert_features),
                    ]
                )
                + " |"
            )
        lines.append("")

        # Per-run path details
        lines.append("### Artifacts")
        lines.append("")
        for r in items:
            lines.append(f"- **`{r.run_dir_name}`**")
            if r.result_json:
                lines.append(f"  - **result_json**: `{r.result_json}`")
            else:
                # expected path
                exp = RESULTS_ROOT / r.run_dir_name / "feature_group_search_result.json"
                if (RESULTS_ROOT / r.run_dir_name).exists():
                    lines.append(f"  - **result_json**: *(pending)* (expected `{exp}`)")
            if r.writeback_yaml:
                lines.append(f"  - **writeback_yaml**: `{r.writeback_yaml}`")
            else:
                # expected path if strategy known
                if r.strategy != "unknown" and r.run_dir_name.startswith(
                    r.strategy + "_"
                ):
                    suffix = r.run_dir_name[len(r.strategy) + 1 :]
                    expy = (
                        STRATEGIES_ROOT
                        / r.strategy
                        / f"features_suggested_{suffix}.yaml"
                    )
                    lines.append(
                        f"  - **writeback_yaml**: *(pending)* (expected `{expy}`)"
                    )
            if r.pool_b_yaml:
                lines.append(f"  - **pool_b_yaml**: `{r.pool_b_yaml}`")
            if r.groups_source:
                lines.append(f"  - **groups_source**: `{r.groups_source}`")
            if r.stop_reason:
                lines.append(f"  - **stop_reason**: `{r.stop_reason}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=str,
        default=str(DOCS_ROOT / "FEATURE_GROUP_SEARCH_HISTORY_SUMMARY.md"),
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = collect_runs()
    md = render_markdown(runs)
    out_path.write_text(md, encoding="utf-8")
    print(f"✅ Wrote {out_path} (runs={len(runs)})")


if __name__ == "__main__":
    main()
