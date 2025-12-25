from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .router_types import RouterAction, RouterDecision


@dataclass(frozen=True)
class AppliedDecision:
    """
    RouterDecision after applying RL action constraints.

    We keep the original RouterDecision class unchanged (for compatibility), but
    provide a wrapper that also captures why the decision was overridden.
    """

    decision: RouterDecision
    reason: Optional[str] = None


def apply_action_to_decision(
    *,
    base: RouterDecision,
    action: RouterAction,
    min_mult: float = 0.0,
    max_mult: float = 2.0,
) -> AppliedDecision:
    """
    Apply RouterAction to a single RouterDecision.

    Rules (production-safe defaults):
    - global_pause => force gated=False and position_size=0
    - router_enabled[router_name]=False => force gated=False and position_size=0
    - capital_multiplier[router_name] scales position_size only (score unchanged)
      (score is often a ranking signal; scaling it can create cross-router coupling)
    """
    a = action.clipped(min_mult=min_mult, max_mult=max_mult)

    # 1) Global pause: veto new risk
    if a.global_pause:
        d = RouterDecision(
            router_name=base.router_name,
            gated=False,
            score=float(base.score),
            position_size=0.0,
        )
        return AppliedDecision(decision=d, reason="global_pause")

    # 2) Router disabled: veto this router
    enabled = a.router_enabled.get(base.router_name, True)
    if not enabled:
        d = RouterDecision(
            router_name=base.router_name,
            gated=False,
            score=float(base.score),
            position_size=0.0,
        )
        return AppliedDecision(decision=d, reason="router_disabled")

    # 3) Apply capital multiplier (position sizing only)
    mult = float(a.capital_multiplier.get(base.router_name, 1.0))
    pos = float(base.position_size) * mult

    d = RouterDecision(
        router_name=base.router_name,
        gated=bool(base.gated) and pos > 0.0,
        score=float(base.score),
        position_size=float(max(0.0, pos)),
    )
    reason = None if mult == 1.0 else f"capital_multiplier={mult:.4g}"
    return AppliedDecision(decision=d, reason=reason)


def apply_action_to_decisions(
    *,
    bases: Iterable[RouterDecision],
    action: RouterAction,
    min_mult: float = 0.0,
    max_mult: float = 2.0,
) -> List[AppliedDecision]:
    return [
        apply_action_to_decision(
            base=b, action=action, min_mult=min_mult, max_mult=max_mult
        )
        for b in bases
    ]
