"""Merge direction_rules for promote: keep locked from archetypes, replace unlocked from workspace."""

from __future__ import annotations

import copy
from typing import Any, List


def merge_direction_rules_for_promote(
    archetypes_rules: Any,
    workspace_rules: Any,
) -> List[dict]:
    """Locked rules from archetypes (order preserved); non-locked tail from workspace only.

    Workspace rows with locked: true are ignored (SoT is archetypes for locked).
    """
    arch = [r for r in (archetypes_rules or []) if isinstance(r, dict)]
    locked = [copy.deepcopy(r) for r in arch if r.get("locked")]
    ws = [r for r in (workspace_rules or []) if isinstance(r, dict)]
    unlocked_ws = [copy.deepcopy(r) for r in ws if not r.get("locked")]
    return locked + unlocked_ws
