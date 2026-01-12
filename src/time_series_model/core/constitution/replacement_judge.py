from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ReplacementDecision(Enum):
    REJECT = "REJECT"
    REPLACE = "REPLACE"


@dataclass(frozen=True)
class ReplacementInputs:
    """
    Minimal evidence vector for replacement decision.

    This is intentionally small + auditable:
    - We do NOT allow multi-dimensional scoring / weighted sums.
    - Replacement must be attributable ("guilty"), not just "new looks better".
    """

    # Slot context
    has_free_slot: bool

    # Old position evidence
    old_position_id: str
    old_remaining_rr: float  # "还能赚多少" (not floating PnL)
    old_failure_reasons: Sequence[str]  # e.g. ["Path Failure", "Time Decay"]

    # New candidate evidence
    new_signal_id: str
    new_expected_rr: float

    # Policy knobs (low freedom)
    beta: float = 1.25  # require new_expected_rr > old_remaining_rr * beta
    allowed_failure_reasons: Sequence[str] = (
        "Path Failure",
        "MAE Breach",
        "Time Decay",
        "R Collapse",
    )


@dataclass(frozen=True)
class ReplacementResult:
    decision: ReplacementDecision
    reason: str
    context: Dict[str, Any]


def decide_replacement_v1(inp: ReplacementInputs) -> ReplacementResult:
    """
    Three-stage Replacement Judge (low DOF):
    1) If there's a free slot -> REPLACE (no need to evict)
    2) Otherwise, old position must be "guilty" (at least one allowed failure reason)
    3) Single-dimension conservative RR improvement: new_expected_rr > old_remaining_rr * beta
    """
    ctx: Dict[str, Any] = {
        "has_free_slot": bool(inp.has_free_slot),
        "old_position_id": str(inp.old_position_id),
        "new_signal_id": str(inp.new_signal_id),
        "old_remaining_rr": float(inp.old_remaining_rr),
        "new_expected_rr": float(inp.new_expected_rr),
        "beta": float(inp.beta),
        "old_failure_reasons": list(inp.old_failure_reasons or []),
        "allowed_failure_reasons": list(inp.allowed_failure_reasons or []),
    }

    if bool(inp.has_free_slot):
        return ReplacementResult(
            decision=ReplacementDecision.REPLACE,
            reason="free_slot",
            context=ctx,
        )

    # Stage 2: must be guilty (at least one allowed failure reason)
    allowed = {
        str(x).strip() for x in (inp.allowed_failure_reasons or []) if str(x).strip()
    }
    guilty = [
        str(x).strip()
        for x in (inp.old_failure_reasons or [])
        if str(x).strip() in allowed
    ]
    if not guilty:
        ctx["guilty_reasons"] = []
        return ReplacementResult(
            decision=ReplacementDecision.REJECT,
            reason="old_not_guilty",
            context=ctx,
        )
    ctx["guilty_reasons"] = guilty

    # Stage 3: single-dimension conservative advantage
    thr = float(inp.old_remaining_rr) * float(inp.beta)
    ctx["threshold_new_rr_gt"] = thr
    if float(inp.new_expected_rr) > thr:
        return ReplacementResult(
            decision=ReplacementDecision.REPLACE,
            reason="rr_improvement",
            context=ctx,
        )
    return ReplacementResult(
        decision=ReplacementDecision.REJECT,
        reason="rr_not_better",
        context=ctx,
    )
