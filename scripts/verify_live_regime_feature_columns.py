#!/usr/bin/env python3
"""Verify live archetype regime.yaml pulls box_structure_f into Feature Bus plan."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from time_series_model.live.live_feature_plan import extract_features_from_archetypes

LIVE_ROOT = PROJECT_ROOT / "live" / "highcap" / "config" / "strategies"
DEPS = PROJECT_ROOT / "config" / "feature_dependencies.yaml"


def main() -> int:
    failed = 0
    for slug in ("tpc",):
        arch = LIVE_ROOT / slug / "archetypes"
        regime = arch / "regime.yaml"
        if not regime.is_file():
            print(f"FAIL {slug}: missing {regime}")
            failed += 1
            continue
        cols, nodes = extract_features_from_archetypes(arch, feature_deps_path=DEPS)
        need = {"box_pos_120", "tpc_semantic_chop", "box_breakout_up"}
        missing = need - cols
        if missing:
            print(f"FAIL {slug}: missing columns {missing}")
            failed += 1
            continue
        if "box_structure_f" not in nodes:
            print(f"FAIL {slug}: box_structure_f node not in plan")
            failed += 1
            continue
        print(
            f"OK {slug}: cols={len(cols)} nodes={len(nodes)} includes box_structure_f"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
