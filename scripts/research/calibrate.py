"""mlbot research calibrate — write draft yaml from plateau json (no live promote)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.research._common import PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research calibrate (draft yaml from plateau)"
    )
    p.add_argument("--from-plateau", required=True)
    p.add_argument("--output", required=True, help="Draft yaml path")
    args = p.parse_args(argv)

    src = Path(args.from_plateau)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    data = json.loads(src.read_text(encoding="utf-8"))
    rec = (
        data.get("recommended")
        or data.get("mid")
        or data.get("recommended_threshold")
        or data.get("plateau_mid")
    )
    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# DRAFT — human review required before promote\n"
        f"# source: {src}\n"
        f"recommended_threshold: {rec}\n",
        encoding="utf-8",
    )
    print(f"wrote draft {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
