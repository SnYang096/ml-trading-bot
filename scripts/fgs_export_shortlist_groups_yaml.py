#!/usr/bin/env python3
"""
Export a *shortlisted* groups YAML from a previous feature-group-search run.

Motivation:
To make A/B/C presets a real workflow:
- Run preset A (fast) to get a ranked/filtered set of group names.
- Export those group names into a smaller groups.yaml.
- Run preset B/C using --groups-yaml <shortlist.yaml> to restrict the candidate space.

This makes later stages much faster and more stable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import yaml
import sys


# Ensure repo root is on sys.path when running as a script (sys.path[0] is scripts/).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _merge_poolb_singletons(
    groups: Dict[str, List[str]], pool_b_yaml: Path
) -> Dict[str, List[str]]:
    pool_obj = _load_yaml(pool_b_yaml)
    pool_fp = pool_obj.get("feature_pipeline") if isinstance(pool_obj, dict) else None
    pool_req = pool_fp.get("requested_features") if isinstance(pool_fp, dict) else None
    pool_req = pool_req if isinstance(pool_req, list) else []

    used_nodes = set()
    for feats in (groups or {}).values():
        for f in feats or []:
            used_nodes.add(str(f))

    for f in pool_req:
        f = str(f).strip()
        if not f or f in used_nodes:
            continue
        key = f"poolb__{f}"
        if key in groups:
            i = 2
            while f"{key}__{i}" in groups:
                i += 1
            key = f"{key}__{i}"
        groups[key] = [f]
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-strategy-config", required=True, help="config/strategies/<strategy_dir>"
    )
    ap.add_argument(
        "--result-json", required=True, help="feature_group_search_result.json path"
    )
    ap.add_argument("--output-yaml", required=True, help="Output groups YAML path")
    ap.add_argument(
        "--mode",
        default="prefilter_survivors",
        choices=["selected_groups", "prefilter_survivors", "beam_selected"],
        help="Which group-name list to export from the result JSON.",
    )
    ap.add_argument(
        "--pool-b-yaml",
        default="",
        help="Optional features_pool_b.yaml used in the run",
    )
    ap.add_argument(
        "--expand-semantic-singletons",
        action="store_true",
        default=False,
        help="Apply the same singleton expansion before filtering (must match your run).",
    )
    ap.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="If >0, keep only the first N names from the chosen list (useful for very fast B/C).",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_strategy_config)
    result_json = Path(args.result_json)
    out_yaml = Path(args.output_yaml)
    pool_b = Path(args.pool_b_yaml) if str(args.pool_b_yaml).strip() else None

    # Import helper(s) from feature_group_search to keep group resolution consistent
    from src.time_series_model.diagnostics import feature_group_search as fgs

    groups, src, auto = fgs._load_groups_with_source(
        strategy_dir_name=base_dir.name, groups_json=None, groups_yaml=None
    )
    if args.expand_semantic_singletons:
        groups = fgs._expand_semantic_groups_to_singletons(groups)
    if pool_b is not None and pool_b.exists():
        groups = _merge_poolb_singletons(groups, pool_b)

    d = _load_json(result_json)
    names: List[str] = []
    if args.mode == "selected_groups":
        names = list(d.get("selected_groups") or [])
    elif args.mode == "beam_selected":
        beam = d.get("beam") or {}
        names = (
            list((beam.get("selected_groups") or [])) if isinstance(beam, dict) else []
        )
    else:
        pre = d.get("prefilter") or {}
        names = list((pre.get("survivors") or [])) if isinstance(pre, dict) else []

    names = [str(x) for x in names if str(x).strip()]
    if args.max_groups and args.max_groups > 0:
        names = names[: int(args.max_groups)]

    missing = [n for n in names if n not in groups]
    kept = {k: groups[k] for k in names if k in groups}

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(
        yaml.safe_dump(kept, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"✅ Wrote shortlist groups YAML: {out_yaml}")
    print(f"   - base groups source: {src} (auto={auto})")
    print(f"   - requested names: {len(names)}")
    print(f"   - kept: {len(kept)}")
    if missing:
        print(
            f"   - missing (not found in resolved groups): {len(missing)} (first 10): {missing[:10]}"
        )


if __name__ == "__main__":
    main()
