#!/usr/bin/env python3
"""Batch gate plateau — DEPRECATED; use optimize_gate_unified.py per strategy."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "DEPRECATED: optimize_all_archetypes_plateau.py removed dependency on "
        "missing optimize_gate_plateau.py.\n"
        "Use: python scripts/optimize_gate_unified.py --strategy bpc --logs ... --output ...",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
