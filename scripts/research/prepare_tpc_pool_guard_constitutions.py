#!/usr/bin/env python3
"""Write trend_pool_guard constitution variants for TPC pool-guard sweep."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "config/experiments/20260612_tpc_trend_pool_guard_sweep"
_BASE = _EXP / "constitution" / "base_tpc_prod.yaml"
_OUT = _EXP / "constitution"

# (enabled, max_unprotected_symbols, max_symbols_after_unlock)
VARIANTS: dict[str, tuple[bool, int, int]] = {
    "G0_prod_1_2": (True, 1, 2),
    "G1_be1_3": (True, 1, 3),
    "G2_be3_3": (True, 3, 3),
    "G3_be3_6": (True, 3, 6),
    "G4_guard_off": (False, 0, 0),
}


def main() -> None:
    if not _BASE.is_file():
        raise SystemExit(f"missing base constitution: {_BASE}")
    base = yaml.safe_load(_BASE.read_text(encoding="utf-8"))
    _OUT.mkdir(parents=True, exist_ok=True)
    for name, (enabled, max_unprot, max_after) in VARIANTS.items():
        doc = deepcopy(base)
        guard = (
            doc.setdefault("resource_allocation", {})
            .setdefault("slot_policy", {})
            .setdefault("trend_pool_guard", {})
        )
        guard["enabled"] = bool(enabled)
        guard["max_unprotected_symbols"] = int(max_unprot)
        guard["max_symbols_after_unlock"] = int(max_after)
        guard.setdefault("unlock_on", "breakeven_locked")
        out = _OUT / f"{name}.yaml"
        out.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(f"wrote {out.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
