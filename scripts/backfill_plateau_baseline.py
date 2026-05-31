"""Backfill last_calibration.plateaus from research plateau.json (dry-run by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.writeback.plateau_baseline import (
    load_yaml,
    merge_plateau_baseline_from_json,
    write_yaml,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Backfill last_calibration.plateaus from research plateau.json"
    )
    p.add_argument("--from-plateau", required=True, help="plateau.json from research")
    p.add_argument("--target-yaml", required=True, help="archetype yaml to update")
    p.add_argument("--feature", default=None)
    p.add_argument("--operator", default=None)
    p.add_argument(
        "--write",
        action="store_true",
        help="Write target yaml (default: dry-run print only)",
    )
    args = p.parse_args(argv)

    src = Path(args.from_plateau)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    target = Path(args.target_yaml)
    if not target.is_absolute():
        target = PROJECT_ROOT / target

    payload = json.loads(src.read_text(encoding="utf-8"))
    raw = load_yaml(target)
    updated = merge_plateau_baseline_from_json(
        raw,
        payload,
        feature=args.feature,
        operator=args.operator,
    )
    entry = updated["last_calibration"]["plateaus"][-1]
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    if args.write:
        write_yaml(target, updated)
        print(f"wrote {target}")
    else:
        print("dry-run: pass --write to update target yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
