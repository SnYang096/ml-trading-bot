#!/usr/bin/env python3
"""One-time migration: config_experiments/ → config/experiments/<exp>/variants/.

For each experiment directory under config/experiments/, scan yaml/md/sh for
config_experiments/ references, copy only the referenced variant trees (pruned to
the experiment's strategy), and rewrite paths in place.

Usage:
  PYTHONPATH=src:scripts python scripts/migrate_config_experiments_to_experiments.py --dry-run
  PYTHONPATH=src:scripts python scripts/migrate_config_experiments_to_experiments.py
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO / "config" / "experiments"
CE_ROOT = REPO / "config_experiments"
REF_RE = re.compile(r"config_experiments/[a-zA-Z0-9_][a-zA-Z0-9_./-]*")

STRATEGY_DIRS = {
    "bpc",
    "tpc",
    "me",
    "srb",
    "chop_grid",
    "spot_accum_simple",
    "spot_ft",
    "trend_scalp",
    "tree_strategies",
    "rolling_trend",
    "fast_scalp",
    "fast_scalp_alts",
    "fast_scalp_majors",
}
FAST_SCALP_FAMILY = {"fast_scalp", "fast_scalp_alts", "fast_scalp_majors"}
KEEP_TOP_DIRS = {"_shared", "constitution", "bad-candidates"}
SHARED_ASSETS = (
    "feature_dependencies.yaml",
    "_shared",
    "constitution",
    "bad-candidates",
)

SCAN_SUFFIXES = {".yaml", ".yml", ".md", ".sh"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _experiment_dirs() -> List[Path]:
    return sorted(
        p for p in EXP_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def _resolve_existing(rel: str) -> Optional[Tuple[str, Path]]:
    """Return (resolved_rel_under_config_experiments, absolute_path) or None."""
    rel = rel.removeprefix("config_experiments/").rstrip("/")
    if not rel:
        return None
    parts = rel.split("/")
    for end in range(len(parts), 0, -1):
        candidate = CE_ROOT.joinpath(*parts[:end])
        if candidate.exists():
            return "/".join(parts[:end]), candidate
    return None


def _infer_strategies_from_exp(exp_dir: Path) -> Set[str]:
    strategies: Set[str] = set()
    name_parts = set(exp_dir.name.lower().split("_"))
    for token in ("bpc", "tpc", "me", "srb", "chop_grid", "fast_scalp", "trend_scalp"):
        if token in name_parts:
            if token == "fast_scalp":
                strategies |= FAST_SCALP_FAMILY
            else:
                strategies.add(token)

    for path in exp_dir.rglob("*"):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        try:
            data = yaml.safe_load(_read_text(path)) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        strat = data.get("strategy")
        if isinstance(strat, str) and strat:
            if strat == "fast_scalp":
                strategies |= FAST_SCALP_FAMILY
            elif strat == "short_term_swing":
                strategies.add("tree_strategies")
            else:
                strategies.add(strat)
        runs = data.get("runs") or data.get("segment_matrix", {}).get("variants") or []
        if isinstance(runs, list):
            for run in runs:
                if isinstance(run, dict):
                    rs = run.get("strategy")
                    if isinstance(rs, str) and rs:
                        if rs == "fast_scalp":
                            strategies |= FAST_SCALP_FAMILY
                        else:
                            strategies.add(rs)
    return strategies


def _infer_from_source_rel(rel: str) -> Set[str]:
    base = rel.split("/")[0].lower()
    out: Set[str] = set()
    for prefix, strat in (
        ("bpc_", "bpc"),
        ("tpc_", "tpc"),
        ("me_", "me"),
        ("srb_", "srb"),
        ("chop_grid_", "chop_grid"),
        ("fast_scalp_", "fast_scalp"),
    ):
        if base.startswith(prefix):
            if strat == "fast_scalp":
                out |= FAST_SCALP_FAMILY
            else:
                out.add(strat)
    if base in {
        "be_combo_strategies",
        "b_gate_only_chop_strategies",
        "h_bull_conditional_vol_strategies",
    }:
        out.add("tpc")
    if base == "fp_ema_plus_slope_strategies":
        out.add("tpc")
    return out


def _keep_strategies(exp_dir: Path, rel: str, src: Path) -> Set[str]:
    keep = _infer_strategies_from_exp(exp_dir) | _infer_from_source_rel(rel)
    scan_root = src if src.is_dir() else src.parent
    present = {p.name for p in scan_root.iterdir() if p.is_dir()} & STRATEGY_DIRS
    if len(present) == 1:
        keep |= present
    if not keep:
        keep = present or {"tpc"}
    return keep


def _minimal_copy_set(rels: Iterable[str]) -> Set[str]:
    """Drop strict subpaths when a parent directory is already copied."""
    ordered = sorted(set(rels), key=len)
    minimal: Set[str] = set()
    for rel in ordered:
        if any(rel.startswith(parent + "/") for parent in minimal):
            continue
        minimal = {m for m in minimal if not m.startswith(rel + "/")}
        minimal.add(rel)
    return minimal


def _collect_shared_parent_assets(rel: str) -> List[str]:
    """Parent dirs may host constitution / feature_dependencies for nested variants."""
    extra: List[str] = []
    parent = Path(rel).parent
    while parent.parts:
        for asset in SHARED_ASSETS:
            candidate = parent / asset
            if (CE_ROOT / candidate).exists():
                extra.append(str(candidate))
        parent = parent.parent
    return extra


def _prune_tree(dest: Path, keep: Set[str]) -> List[str]:
    removed: List[str] = []
    if not dest.is_dir():
        return removed
    for child in dest.iterdir():
        if not child.is_dir():
            continue
        if child.name in KEEP_TOP_DIRS:
            continue
        if child.name in STRATEGY_DIRS and child.name not in keep:
            shutil.rmtree(child)
            removed.append(child.name)
    return removed


def _copy_tree(src: Path, dest: Path) -> None:
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, dirs_exist_ok=False)


def _scan_refs(exp_dir: Path) -> Set[str]:
    refs: Set[str] = set()
    for path in exp_dir.rglob("*"):
        if path.suffix not in SCAN_SUFFIXES:
            continue
        for match in REF_RE.findall(_read_text(path)):
            refs.add(match.rstrip("/"))
    return refs


def _target_prefix(exp_dir: Path) -> str:
    return f"config/experiments/{exp_dir.name}/variants"


def _rewrite_file(path: Path, mapping: Dict[str, str], dry_run: bool) -> int:
    text = _read_text(path)
    original = text
    for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(old, new)
    if text == original:
        return 0
    if not dry_run:
        path.write_text(text, encoding="utf-8")
    return 1


def migrate(*, dry_run: bool) -> int:
    jobs: Dict[str, Set[str]] = defaultdict(set)
    missing: List[Tuple[str, str]] = []
    path_map_by_exp: Dict[str, Dict[str, str]] = defaultdict(dict)

    for exp_dir in _experiment_dirs():
        exp_name = exp_dir.name
        for ref in _scan_refs(exp_dir):
            resolved = _resolve_existing(ref.removeprefix("config_experiments/"))
            if resolved is None:
                missing.append((exp_name, ref))
                continue
            rel, _ = resolved
            jobs[exp_name].add(rel)
            for extra in _collect_shared_parent_assets(rel):
                jobs[exp_name].add(extra)
            old = f"config_experiments/{rel}"
            new = f"{_target_prefix(exp_dir)}/{rel}"
            path_map_by_exp[exp_name][old] = new
            if rel.endswith(".yaml"):
                path_map_by_exp[exp_name][old + "/"] = new

    copied = 0
    pruned_report: List[str] = []

    for exp_name, rels in sorted(jobs.items()):
        exp_dir = EXP_ROOT / exp_name
        for rel in sorted(_minimal_copy_set(rels)):
            src = CE_ROOT / rel
            if not src.exists():
                continue
            dest = exp_dir / "variants" / rel
            keep = _keep_strategies(exp_dir, rel, src if src.is_dir() else src.parent)

            if dry_run:
                print(
                    f"[dry-run] copy {src.relative_to(REPO)} -> "
                    f"{dest.relative_to(REPO)} keep={sorted(keep)}"
                )
            else:
                _copy_tree(src, dest)
                removed = _prune_tree(dest if dest.is_dir() else dest.parent, keep)
                if removed:
                    pruned_report.append(f"{dest.relative_to(REPO)}: dropped {removed}")
            copied += 1

    rewritten = 0
    for exp_dir in _experiment_dirs():
        mapping = path_map_by_exp.get(exp_dir.name, {})
        if not mapping:
            continue
        for path in exp_dir.rglob("*"):
            if path.suffix not in SCAN_SUFFIXES:
                continue
            rewritten += _rewrite_file(path, mapping, dry_run)

    report_path = EXP_ROOT / "CONFIG_EXPERIMENTS_MIGRATION.md"
    lines = [
        "# config_experiments → experiments/variants migration",
        "",
        f"Mode: {'dry-run' if dry_run else 'applied'}",
        "",
        "## Copied variant trees",
        "",
    ]
    for exp_name, rels in sorted(jobs.items()):
        for rel in sorted(rels):
            lines.append(f"- `{exp_name}/variants/{rel}`")
    lines.extend(["", "## Pruned sibling strategies", ""])
    if pruned_report:
        lines.extend(f"- {row}" for row in pruned_report)
    else:
        lines.append("- (see dry-run output or none)")
    lines.extend(["", "## Missing references (not on disk)", ""])
    for exp_name, ref in sorted(set(missing)):
        lines.append(f"- `{exp_name}`: `{ref}`")
    lines.extend(
        [
            "",
            "## Orphan dirs still under config_experiments/",
            "",
            "These were not referenced by any experiment card and were left in place.",
            "",
        ]
    )
    referenced_roots = {rel.split("/")[0] for rels in jobs.values() for rel in rels}
    if CE_ROOT.is_dir():
        for child in sorted(CE_ROOT.iterdir()):
            if child.name == "README.md":
                continue
            if child.is_dir() and child.name not in referenced_roots:
                lines.append(f"- `{child.name}/`")

    if not dry_run:
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Copied {copied} variant paths across {len(jobs)} jobs")
    print(f"Rewrote {rewritten} files under config/experiments/")
    print(f"Missing refs: {len(set(missing))}")
    if not dry_run:
        print(f"Report: {report_path.relative_to(REPO)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")
    args = parser.parse_args()
    return migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
