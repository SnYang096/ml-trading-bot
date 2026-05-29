"""mlbot research promote — explicit human step with locked-merge + backup + diff."""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from scripts.research._common import PROJECT_ROOT


def _is_protected_rule(rule: Dict[str, Any]) -> bool:
    return bool(
        rule.get("locked") or rule.get("frozen") or rule.get("promote_never_disable")
    )


def _merge_gate_sections(
    production: Dict[str, Any],
    draft: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge draft into production; preserve protected rules from production."""
    out = deepcopy(production)
    notes: List[str] = []
    for section in ("hard_gates", "system_safety", "guardrails"):
        prod_rules = {
            r.get("id"): r
            for r in (out.get(section) or [])
            if isinstance(r, dict) and r.get("id")
        }
        draft_rules = {
            r.get("id"): r
            for r in (draft.get(section) or [])
            if isinstance(r, dict) and r.get("id")
        }
        if not draft_rules:
            continue
        merged: List[Dict[str, Any]] = []
        seen = set()
        for rule in out.get(section) or []:
            if not isinstance(rule, dict):
                merged.append(rule)
                continue
            rid = rule.get("id")
            if not rid:
                merged.append(rule)
                continue
            seen.add(rid)
            if _is_protected_rule(rule):
                if rid in draft_rules:
                    notes.append(f"preserved locked rule: {rid}")
                merged.append(rule)
            elif rid in draft_rules:
                merged.append(deepcopy(draft_rules[rid]))
            else:
                merged.append(rule)
        for rid, drule in draft_rules.items():
            if rid not in seen and not _is_protected_rule(drule):
                merged.append(deepcopy(drule))
                notes.append(f"added draft rule: {rid}")
        out[section] = merged
    if draft.get("schema") and not out.get("schema"):
        out["schema"] = draft["schema"]
    return out, notes


def _yaml_text(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def promote_yaml(
    src: Path,
    dst: Path,
    *,
    layer: str = "gate",
) -> Tuple[str, List[str]]:
    """Return merged yaml text and merge notes."""
    draft_raw = src.read_text(encoding="utf-8")
    if draft_raw.lstrip().startswith("#"):
        draft_raw = "\n".join(
            line
            for line in draft_raw.splitlines()
            if not line.startswith("# DRAFT — human review")
        )
    draft = yaml.safe_load(draft_raw) or {}

    if not dst.exists():
        return _yaml_text(draft), ["target did not exist — writing draft as new file"]

    production = yaml.safe_load(dst.read_text(encoding="utf-8")) or {}
    if layer == "gate":
        merged, notes = _merge_gate_sections(production, draft)
        return _yaml_text(merged), notes
    return _yaml_text(draft), ["non-gate promote: draft replaces target (no merge)"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research promote (draft → archetypes, manual gate, locked-merge)"
    )
    p.add_argument("--from", dest="from_path", required=True)
    p.add_argument("--to", dest="to_path", required=True)
    p.add_argument(
        "--layer",
        default="gate",
        choices=["gate", "prefilter", "entry", "regime", "direction"],
    )
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument(
        "--dry-run", action="store_true", help="Show diff only, do not write"
    )
    args = p.parse_args(argv)

    src = Path(args.from_path)
    dst = Path(args.to_path)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not dst.is_absolute():
        dst = PROJECT_ROOT / dst
    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 3
    if not args.yes and not args.dry_run:
        print(f"Promote {src} → {dst}")
        print("Refusing without --yes (human review required). Use --dry-run for diff.")
        return 2

    merged_text, notes = promote_yaml(src, dst, layer=args.layer)
    old_text = dst.read_text(encoding="utf-8") if dst.exists() else ""
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        merged_text.splitlines(keepends=True),
        fromfile=str(dst),
        tofile=f"{dst} (after promote)",
    )
    diff_text = "".join(diff)
    print(diff_text if diff_text else "(no textual diff)")
    for n in notes:
        print(f"  note: {n}")

    if args.dry_run:
        print("dry-run: no files written")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = dst.with_suffix(dst.suffix + f".bak.{ts}")
        shutil.copy2(dst, backup)
        print(f"backup: {backup}")
    dst.write_text(merged_text, encoding="utf-8")
    print(f"promoted to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
