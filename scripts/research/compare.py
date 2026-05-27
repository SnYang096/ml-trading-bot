"""mlbot research compare — compare scan/ic json artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.research._common import PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research compare (json artifacts side-by-side)"
    )
    p.add_argument("paths", nargs="+", help="JSON result files to compare")
    args = p.parse_args(argv)

    rows = []
    for raw in args.paths:
        pth = Path(raw)
        if not pth.is_absolute():
            pth = PROJECT_ROOT / pth
        blob = json.loads(pth.read_text(encoding="utf-8"))
        if isinstance(blob, list):
            rows.append({"path": str(pth), "n_rows": len(blob), "sample": blob[:3]})
        else:
            rows.append({"path": str(pth), "keys": list(blob.keys())[:10]})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
