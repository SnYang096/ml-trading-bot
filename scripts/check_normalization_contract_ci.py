#!/usr/bin/env python3
"""
CI-oriented normalization contract check.

What it does:
- Loads config/feature_dependencies.yaml
- Ensures "no missing method" (hard error)
- For a list of strategy features.yaml files, computes:
  - total_output_columns
  - raw_columns count
  - and writes a markdown report per strategy
- Fails CI if any report contains raw_columns > 0 (or missing_method > 0)

Why this exists:
- `scripts/diagnose_normalization_contract.py` is great for humans, but CI needs a strict pass/fail gate.
- We avoid installing heavy deps (ta-lib) by only importing the contract module + pyyaml.
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

from src.features.normalization.feature_contract import (  # noqa: E402
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

    return "\n".join(lines)


def _write_report(out_dir: Path, name: str, content: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"norm_contract_{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-deps", default="config/feature_dependencies.yaml")
    ap.add_argument(
        "--strategy-features-yaml",
        action="append",
        default=[],
        help="Repeatable. e.g. config/strategies/sr_breakout/features.yaml",
    )
    ap.add_argument(
        "--include-deps",
        action="store_true",
        help="Include dependency closure of requested features.",
    )
    ap.add_argument("--out-dir", default="ci_artifacts/normalization_contract")
    args = ap.parse_args()

    deps = _load_yaml(args.feature_deps)
    # enforce: no MISSING methods
    validate_feature_dependencies_normalization(deps, mode="error")

    features_cfg = (deps or {}).get("features", {}) or {}
    out_dir = Path(args.out_dir)

    # Always generate a global report
    global_rows = collect_feature_normalization_meta(deps, only_features=None)
    global_md = _to_markdown(
        global_rows, title="Normalization Contract Report (global)"
    )
    _write_report(out_dir, "global", global_md)

    failures: List[str] = []
    global_raw = [r for r in global_rows if r["method"] == "raw"]
    if global_raw:
        failures.append(f"global: raw_columns={len(global_raw)}")

    for strategy_yaml in args.strategy_features_yaml:
        cfg = _load_yaml(strategy_yaml)
        requested = _get_requested_features_from_strategy_features_yaml(cfg)
        requested = _resolve_output_cols_to_features(features_cfg, requested)
        if args.include_deps:
            only_features: Optional[List[str]] = _dependency_closure(
                features_cfg, requested
            )
        else:
            only_features = [r for r in requested if r in features_cfg]

        rows = collect_feature_normalization_meta(deps, only_features=only_features)
        title = f"Normalization Contract Report ({Path(strategy_yaml).as_posix()})"
        md = _to_markdown(rows, title=title)
        name = (
            Path(strategy_yaml).parts[-2]
            if len(Path(strategy_yaml).parts) >= 2
            else Path(strategy_yaml).stem
        )
        _write_report(out_dir, name, md)

        raw = [r for r in rows if r["method"] == "raw"]
        if raw:
            failures.append(f"{strategy_yaml}: raw_columns={len(raw)}")

    if failures:
        print("❌ Normalization contract CI gate failed (raw columns remain):")
        for f in failures:
            print(f"  - {f}")
        print(f"Reports written under: {out_dir.as_posix()}")
        return 2

    print("✅ Normalization contract CI gate passed (no raw columns).")
    print(f"Reports written under: {out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
