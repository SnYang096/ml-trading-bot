"""Multi-leg live funnel: regime/prefilter + engine audit (not TPC direction/no_dir)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def _action_types(actions: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for a in actions or ():
        if isinstance(a, Mapping):
            out.add(str(a.get("action", "") or "").lower())
        else:
            out.add(str(getattr(a, "action", "") or "").lower())
    return out


def chop_grid_bar_outcome(
    *,
    active_at_open: bool,
    wanted_enter: bool,
    is_box: bool,
    prefilter_ok: bool = True,
    chop: float,
    entry_chop_min: float,
    actions: Iterable[Any],
) -> str:
    """Same labels as chop_grid_live_engine audit / Prometheus bar outcomes."""
    act_types = _action_types(actions)
    if "market_exit" in act_types:
        return "exit_close"
    if not active_at_open and wanted_enter and "place" in act_types:
        return "open_grid_placed"
    if active_at_open:
        return "active_holding"
    if not active_at_open:
        if is_box:
            return "flat_blocked_box"
        if not prefilter_ok:
            return "flat_blocked_prefilter"
        if chop < entry_chop_min:
            return "flat_blocked_chop_low"
        return "flat_other"
    return "other"


def trend_scalp_bar_outcome(
    *,
    active_at_open: bool,
    wanted_enter: bool,
    trend_conf: float,
    chop: float,
    entry_trend_min: float,
    max_entry_chop: float,
    exclude_box: bool,
    is_box: bool,
    actions: Iterable[Any],
    explicit: Optional[str] = None,
) -> str:
    """Outcome for one trend_scalp bar (explicit overrides inference)."""
    if explicit:
        return str(explicit)
    act_types = _action_types(actions)
    if "market_exit" in act_types:
        return "exit_close"
    if not active_at_open and wanted_enter and act_types & {"place"}:
        return "segment_open_placed"
    if active_at_open:
        if "place" in act_types:
            return "active_seed_inventory"
        return "active_manage"
    if not active_at_open:
        if exclude_box and is_box:
            return "flat_blocked_box"
        if trend_conf < entry_trend_min:
            return "flat_blocked_trend_low"
        if chop > max_entry_chop:
            return "flat_blocked_chop_high"
        return "flat_other"
    return "other"


def _multileg_reject_reasons(rejected: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for rej in rejected or ():
        reason = ""
        if isinstance(rej, Mapping):
            reason = str(rej.get("reason") or "")
        else:
            reason = str(getattr(rej, "reason", "") or "")
        if reason:
            out.append(reason[:60])
    return out


def _risk_layer(
    actions: Iterable[Any],
    approved_actions: Iterable[Any],
    rejected: Iterable[Any],
) -> tuple[bool, bool, List[str]]:
    actions_list = list(actions or ())
    approved_list = list(approved_actions or ())
    has_action = bool(actions_list)
    has_approved = bool(approved_list)
    return has_action, has_approved, _multileg_reject_reasons(rejected)


def funnel_for_chop_grid_bar(
    *,
    audit: Mapping[str, Any],
    actions: Iterable[Any],
    approved_actions: Iterable[Any],
    rejected: Iterable[Any],
) -> Dict[str, Any]:
    """record_strategy_eval-shaped dict aligned with chop_grid audit fields."""
    is_box = bool(audit.get("is_box"))
    prefilter_ok = bool(audit.get("prefilter_ok", not is_box))
    wanted_enter = bool(audit.get("wanted_enter"))
    active_at_open = bool(audit.get("active_at_open"))
    outcome = str(audit.get("outcome") or "")
    _, risk_ok, risk_reasons = _risk_layer(actions, approved_actions, rejected)
    act_types = _action_types(actions)
    open_grid = outcome == "open_grid_placed" or (
        not active_at_open and wanted_enter and "place" in act_types
    )
    exit_grid = outcome == "exit_close" or "market_exit" in act_types
    return {
        "multileg": True,
        "engine": "chop_grid",
        "regime": True,
        "prefilter": prefilter_ok and not is_box,
        "wanted_enter": wanted_enter,
        "active_segment": active_at_open,
        "outcome": outcome,
        "open_grid": open_grid,
        "exit_grid": exit_grid,
        "risk_gate": risk_ok,
        "gate_reasons": risk_reasons,
        "orders_intent": bool(act_types & {"place", "market_exit"}),
    }


def funnel_for_trend_scalp_bar(
    *,
    audit: Mapping[str, Any],
    actions: Iterable[Any],
    approved_actions: Iterable[Any],
    rejected: Iterable[Any],
) -> Dict[str, Any]:
    """record_strategy_eval-shaped dict for dual_add trend_scalp."""
    trend_side = str(audit.get("trend_side") or "").upper()
    dv = 1 if trend_side == "LONG" else (-1 if trend_side == "SHORT" else 0)
    wanted_enter = bool(audit.get("wanted_enter"))
    active_at_open = bool(audit.get("active_at_open"))
    trend_conf = float(audit.get("trend_conf") or 0.0)
    entry_trend_min = float(audit.get("entry_trend_min") or 0.0)
    is_box = bool(audit.get("is_box"))
    exclude_box = bool(audit.get("exclude_box_prefilter"))
    outcome = str(audit.get("outcome") or "")
    _, risk_ok, risk_reasons = _risk_layer(actions, approved_actions, rejected)
    act_types = _action_types(actions)
    regime_ok = active_at_open or trend_conf >= entry_trend_min
    prefilter_ok = not (exclude_box and is_box)
    exit_close = outcome == "exit_close" or "market_exit" in act_types
    return {
        "multileg": True,
        "engine": "trend_scalp",
        "regime": regime_ok,
        "prefilter": prefilter_ok,
        "wanted_enter": wanted_enter,
        "active_segment": active_at_open,
        "outcome": outcome,
        "direction": dv != 0,
        "direction_value": dv,
        "exit_close": exit_close,
        "risk_gate": risk_ok,
        "gate_reasons": risk_reasons,
        "gate": risk_ok,
        "entry_filter": risk_ok and bool(act_types),
        "evidence": risk_ok and bool(act_types),
        "orders_intent": bool(act_types & {"place", "market_exit"}),
    }


def funnel_for_multileg_bar(
    *,
    strategy: str,
    engine_audit: Optional[Mapping[str, Any]],
    actions: Iterable[Any],
    approved_actions: Iterable[Any],
    rejected: Iterable[Any],
) -> Dict[str, Any]:
    """Dispatch to engine-specific funnel; legacy generic shape if audit missing."""
    audit = engine_audit if isinstance(engine_audit, Mapping) else None
    engine = str((audit or {}).get("engine") or "").lower()
    if not engine:
        sn = str(strategy or "").lower()
        if "chop" in sn:
            engine = "chop_grid"
        elif "trend" in sn:
            engine = "trend_scalp"

    if engine == "chop_grid" and audit:
        return funnel_for_chop_grid_bar(
            audit=audit,
            actions=actions,
            approved_actions=approved_actions,
            rejected=rejected,
        )
    if engine == "trend_scalp" and audit:
        return funnel_for_trend_scalp_bar(
            audit=audit,
            actions=actions,
            approved_actions=approved_actions,
            rejected=rejected,
        )

    # Fallback (no engine audit on bar): risk layer only — avoid fake no_dir semantics.
    _, risk_ok, risk_reasons = _risk_layer(actions, approved_actions, rejected)
    return {
        "multileg": True,
        "engine": engine or "unknown",
        "regime": True,
        "prefilter": True,
        "risk_gate": risk_ok,
        "gate_reasons": risk_reasons,
        "gate": risk_ok,
        "orders_intent": bool(_action_types(actions)),
    }
