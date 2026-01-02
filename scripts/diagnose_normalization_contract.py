#!/usr/bin/env python3
"""
Diagnose normalization contract from config/feature_dependencies.yaml.

Use cases:
- Global: list all raw columns (not cross-asset comparable) that still need normalization work.
- Per-strategy: restrict to features referenced in config/strategies/<...>/features.yaml,
  optionally including their dependency closure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

import sys

# Ensure repo root is importable so `import src.*` works when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.normalization.feature_contract import (
    collect_feature_normalization_meta,
    validate_feature_dependencies_normalization,
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_requested_features_from_strategy_features_yaml(
    cfg: Dict[str, Any],
) -> List[str]:
    fp = cfg.get("feature_pipeline") or {}
    req = fp.get("requested_features") or []
    return [str(x) for x in req]


def _resolve_output_cols_to_features(
    features_cfg: Dict[str, Any], names: Iterable[str]
) -> List[str]:
    # Map output columns back to feature function name (same logic as loader, simplified)
    out_to_feat: Dict[str, str] = {}
    for feat_name, info in features_cfg.items():
        for col in info.get("output_columns", [feat_name]) or [feat_name]:
            out_to_feat[str(col)] = feat_name

    resolved: List[str] = []
    seen: Set[str] = set()
    for n in names:
        if n in features_cfg:
            if n not in seen:
                resolved.append(n)
                seen.add(n)
        elif n in out_to_feat:
            feat = out_to_feat[n]
            if feat not in seen:
                resolved.append(feat)
                seen.add(feat)
        else:
            # keep unknown as-is (will be filtered out later)
            if n not in seen:
                resolved.append(n)
                seen.add(n)
    return resolved


def _dependency_closure(
    features_cfg: Dict[str, Any], requested: Iterable[str]
) -> List[str]:
    needed: Set[str] = set()
    queue: List[str] = []
    for r in requested:
        if r in features_cfg:
            needed.add(r)
            queue.append(r)

    while queue:
        cur = queue.pop(0)
        deps = features_cfg.get(cur, {}).get("dependencies", []) or []
        for d in deps:
            if d in features_cfg and d not in needed:
                needed.add(d)
                queue.append(d)
    return sorted(list(needed))


def _to_markdown(rows: List[Dict[str, Any]], *, title: str) -> str:
    total = len(rows)
    missing = [r for r in rows if r["method"] == "MISSING"]
    raw = [r for r in rows if r["method"] == "raw"]

    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"- **total_output_columns**: {total}")
    lines.append(f"- **missing_method**: {len(missing)}")
    lines.append(f"- **raw_columns**: {len(raw)}")
    lines.append("")

    # Group raw by feature
    by_feat: Dict[str, List[Dict[str, Any]]] = {}
    for r in raw:
        by_feat.setdefault(r["feature"], []).append(r)

    if raw:
        lines.append("### Raw columns (not cross-asset comparable yet)")
        for feat in sorted(by_feat.keys()):
            group = by_feat[feat]
            cat = group[0].get("category")
            lines.append(f"- **{feat}** (category={cat}):")
            for r in group:
                lines.append(f"  - `{r['column']}`")
        lines.append("")

    # Show a small sample of non-raw methods (so user sees conventions)
    sample = [r for r in rows if r["method"] not in ("raw", "MISSING")][:20]
    if sample:
        lines.append("### Sample of normalized/unitless columns (first 20)")
        for r in sample:
            rng = r.get("expected_range")
            rng_s = f"{rng}" if rng is not None else ""
            lines.append(
                f"- `{r['column']}`: **{r['method']}** {rng_s} (feature={r['feature']})"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-deps", default="config/feature_dependencies.yaml")
    ap.add_argument(
        "--strategy-features-yaml",
        default=None,
        help="Optional strategy features.yaml to filter (e.g. config/strategies/sr_reversal_rr_reg_long/features.yaml)",
    )
    ap.add_argument(
        "--include-deps",
        action="store_true",
        help="If set, include dependency closure of requested features.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output markdown path. If omitted, prints to stdout.",
    )
    args = ap.parse_args()

    deps = _load_yaml(args.feature_deps)
    # enforce: no MISSING methods
    validate_feature_dependencies_normalization(deps, mode="error")

    features_cfg = (deps or {}).get("features", {}) or {}
    only_features: Optional[List[str]] = None
    title = "Normalization Contract Report (global)"

    if args.strategy_features_yaml:
        cfg = _load_yaml(args.strategy_features_yaml)
        requested = _get_requested_features_from_strategy_features_yaml(cfg)
        requested = _resolve_output_cols_to_features(features_cfg, requested)
        if args.include_deps:
            only_features = _dependency_closure(features_cfg, requested)
        else:
            only_features = [r for r in requested if r in features_cfg]
        title = f"Normalization Contract Report ({Path(args.strategy_features_yaml).as_posix()})"

    rows = collect_feature_normalization_meta(deps, only_features=only_features)
    md = _to_markdown(rows, title=title)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
