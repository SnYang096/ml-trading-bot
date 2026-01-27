#!/usr/bin/env python3
"""
Extract quantile_* evidence keys from execution_archetypes.yaml (when_then_rules).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Set

import yaml


def _collect_from_when(when: Any, out: Set[str]) -> None:
    if isinstance(when, list):
        for item in when:
            _collect_from_when(item, out)
        return
    if not isinstance(when, dict):
        return
    if "all_of" in when:
        for item in when.get("all_of") or []:
            _collect_from_when(item, out)
        return
    if "any_of" in when:
        for item in when.get("any_of") or []:
            _collect_from_when(item, out)
        return
    if "not" in when:
        _collect_from_when(when.get("not"), out)
        return
    if "key" in when and "op" in when:
        op = str(when.get("op") or "")
        key = str(when.get("key") or "").strip()
        if op.startswith("quantile_") and key:
            out.add(key)
        return
    if "any_key_contains" in when:
        return
    if len(when) == 1:
        k = next(iter(when.keys()))
        cond = when.get(k) or {}
        if isinstance(cond, dict) and len(cond) == 1:
            op = next(iter(cond.keys()))
            if str(op).startswith("quantile_") and str(k).strip():
                out.add(str(k).strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract quantile_* evidence keys from execution_archetypes.yaml"
    )
    parser.add_argument(
        "--config",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml path",
    )
    args = parser.parse_args()

    cfg = Path(args.config)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    keys: Set[str] = set()

    def _collect_from_archetypes(arches: Any) -> None:
        if not isinstance(arches, dict):
            return
        for arch in arches.values():
            rules = arch.get("when_then_rules") or []
            for rule in rules:
                _collect_from_when(rule.get("when"), keys)

    _collect_from_archetypes(data.get("archetypes") or {})
    _collect_from_archetypes(data.get("overlays") or {})

    print(",".join(sorted(keys)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
