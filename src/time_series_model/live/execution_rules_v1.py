from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ExecutionRulesV1:
    """
    Minimal v1 execution rules:
    - per-archetype list of required feature keys (presence check)
    - optional veto flags (fail-closed)
    """

    required_keys_by_archetype: Dict[str, List[str]]


def load_execution_rules_v1(path: str | Path) -> ExecutionRulesV1:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    req = obj.get("required_keys_by_archetype") or {}
    out: Dict[str, List[str]] = {}
    if isinstance(req, dict):
        for k, v in req.items():
            if isinstance(v, list):
                out[str(k)] = [str(x) for x in v]
    return ExecutionRulesV1(required_keys_by_archetype=out)


def apply_execution_rules_v1(
    *,
    rules: ExecutionRulesV1,
    archetype_name: str,
    features: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, reasons). ok=False means veto (NO_TRADE).
    """
    feats = features or {}
    req = rules.required_keys_by_archetype.get(str(archetype_name)) or []
    missing = [k for k in req if k not in feats]
    if missing:
        return False, [f"exec_rules_missing_keys={missing}"]
    return True, []
