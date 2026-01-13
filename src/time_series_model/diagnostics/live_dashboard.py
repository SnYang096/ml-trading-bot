from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.time_series_model.diagnostics.ood_config import OODConfigV1


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
    ood_cfg: OODConfigV1,
    ood_score: Optional[float],
    top_archetype_survival_prob: Optional[float],
    active_archetype: Optional[str],
    size_cap: Optional[float],
    kill_switch_state: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> LiveDashboardMetrics:
    """
    Build a payload that conforms to ood_cfg.dashboard_keys.
    Missing values are allowed (None), but keys are always present.
    """
    base = {
        "ood_score": ood_score,
        "top_archetype_survival_prob": top_archetype_survival_prob,
        "active_archetype": active_archetype,
        "size_cap": size_cap,
        "kill_switch_state": kill_switch_state,
    }
    out: Dict[str, Any] = {}
    for k in ood_cfg.dashboard_keys:
        out[str(k)] = base.get(str(k))
    if extra:
        out.update({str(k): v for k, v in extra.items()})
    return LiveDashboardMetrics(payload=out)


def validate_live_dashboard_payload(
    *, ood_cfg: OODConfigV1, payload: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Ensure required keys exist (for auditability).
    """
    missing = [k for k in (ood_cfg.dashboard_keys or []) if k not in (payload or {})]
    if missing:
        return False, [f"missing_dashboard_keys={missing}"]
    return True, []
