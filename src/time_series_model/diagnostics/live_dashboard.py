"""
Live dashboard: minimal monitoring payload for state snapshots.
OOD/survival removed; safety is handled by constitution only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Hardcoded dashboard keys (no OOD dependency)
DASHBOARD_KEYS = [
    "active_archetype",
    "size_cap",
    "kill_switch_state",
    "drawdown",
    "daily_loss",
]


@dataclass(frozen=True)
class LiveDashboardMetrics:
    """
    A small, stable surface for 'don't do stupid things' monitoring.

    We keep it as a dict-like payload so it can be used in both research reports
    and live snapshots, without forcing heavy dependencies.
    """

    payload: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.payload or {})


def build_live_dashboard_payload(
    *,
    active_archetype: Optional[str] = None,
    size_cap: Optional[float] = None,
    kill_switch_state: Optional[str] = None,
    drawdown: Optional[float] = None,
    daily_loss: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> LiveDashboardMetrics:
    """
    Build a minimal dashboard payload (no OOD dependency).
    Missing values are allowed (None), but keys are always present.
    """
    base = {
        "active_archetype": active_archetype,
        "size_cap": size_cap,
        "kill_switch_state": kill_switch_state,
        "drawdown": drawdown,
        "daily_loss": daily_loss,
    }
    out: Dict[str, Any] = {}
    for k in DASHBOARD_KEYS:
        out[str(k)] = base.get(str(k))
    if extra:
        out.update({str(k): v for k, v in extra.items()})
    return LiveDashboardMetrics(payload=out)


def validate_live_dashboard_payload(
    *, payload: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Ensure required keys exist (for auditability).
    """
    missing = [k for k in DASHBOARD_KEYS if k not in (payload or {})]
    if missing:
        return False, [f"missing_dashboard_keys={missing}"]
    return True, []
