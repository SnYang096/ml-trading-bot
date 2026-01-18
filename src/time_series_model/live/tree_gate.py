from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)


@dataclass(frozen=True)
class GateRules:
    rules: List[Dict[str, Any]]
    deny_if: List[str]
    allow_if: List[str]
    allow_mode: str
    default_action: str


def _parse_gate_rules(raw: Dict[str, Any]) -> GateRules:
    rules = list(raw.get("rules") or [])
    deny_if = [str(x) for x in (raw.get("deny_if") or [])]
    allow_if = [str(x) for x in (raw.get("allow_if") or [])]
    allow_mode = str(raw.get("allow_mode") or "any").lower()
    default_action = str(raw.get("default_action") or "allow").lower()
    return GateRules(
        rules=rules,
        deny_if=deny_if,
        allow_if=allow_if,
        allow_mode=allow_mode,
        default_action=default_action,
    )


def apply_gate_rules(
    *,
    gate_rules: Dict[str, Any],
    features: Dict[str, Any],
    quantiles: Dict[str, Any] | None = None,
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, reasons). ok=False means veto (NO_TRADE).
    """
    rules = _parse_gate_rules(gate_rules or {})
    flags = compute_execution_evidence(
        features=features, rules=list(rules.rules or []), quantiles=quantiles
    )
    deny_hits = [name for name in rules.deny_if if flags.get(name)]
    if deny_hits:
        return False, [f"gate_deny={deny_hits}"]
    allow_if = list(rules.allow_if or [])
    if allow_if:
        if rules.allow_mode == "all":
            ok = all(flags.get(name, False) for name in allow_if)
        elif rules.allow_mode.startswith("min"):
            raw = rules.allow_mode.replace("min", "").replace(":", "").strip()
            try:
                min_hits = int(raw)
            except Exception:
                min_hits = 1
            hit_count = sum(1 for name in allow_if if flags.get(name, False))
            ok = hit_count >= max(1, min_hits)
        elif rules.allow_mode.startswith("at_least_"):
            raw = rules.allow_mode.replace("at_least_", "").strip()
            try:
                min_hits = int(raw)
            except Exception:
                min_hits = 1
            hit_count = sum(1 for name in allow_if if flags.get(name, False))
            ok = hit_count >= max(1, min_hits)
        else:
            ok = any(flags.get(name, False) for name in allow_if)
        if not ok:
            return False, [f"gate_allow_not_met={allow_if}"]
        return True, []
    return rules.default_action != "deny", []
