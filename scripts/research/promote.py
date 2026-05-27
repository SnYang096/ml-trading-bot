"""mlbot research promote — explicit human step to copy draft into archetypes."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from scripts.research._common import PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research promote (draft → archetypes, manual gate)"
    )
    p.add_argument("--from", dest="from_path", required=True)
    p.add_argument("--to", dest="to_path", required=True)
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
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
    if not args.yes:
        print(f"Promote {src} → {dst}")
        print("Refusing without --yes (human review required).")
        return 2
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"promoted to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
