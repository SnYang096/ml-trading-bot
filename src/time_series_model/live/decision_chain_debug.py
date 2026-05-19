"""Chain debug logging for live decision paths (trend PCM, multi-leg engines)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def chain_debug_enabled(scope: str) -> bool:
    """True when MLBOT_CHAIN_DEBUG or MLBOT_{SCOPE}_CHAIN_DEBUG is truthy."""
    if _env_truthy("MLBOT_CHAIN_DEBUG"):
        return True
    key = f"MLBOT_{str(scope or '').strip().upper()}_CHAIN_DEBUG"
    return _env_truthy(key)


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _compact_funnel(funnel: Any) -> Dict[str, Any]:
    if not isinstance(funnel, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in funnel.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[str(k)] = v
        elif isinstance(v, list) and len(v) <= 6:
            out[str(k)] = v
    return out


def log_trend_no_intent(
    symbol: str,
    decision_handler: Any,
    features: Dict[str, Any],
) -> None:
    """Log why trend/PCM returned no TradeIntent on this decision cycle."""
    from src.time_series_model.portfolio.live_pcm import LivePCM

    ts = features.get("timestamp")
    if isinstance(decision_handler, LivePCM):
        trace = dict(getattr(decision_handler, "_last_decide_trace", None) or {})
        strat_funnels: Dict[str, Dict[str, Any]] = {}
        for arch, strat in (
            getattr(decision_handler, "_strategies", None) or {}
        ).items():
            strat_funnels[str(arch)] = _compact_funnel(
                getattr(strat, "_last_funnel", None)
            )
        logger.info(
            "[%s] signal-check no intent ts=%s pcm_trace=%s strategy_funnels=%s",
            symbol,
            ts,
            trace,
            strat_funnels,
        )
        return

    funnel = _compact_funnel(getattr(decision_handler, "_last_funnel", None))
    logger.info(
        "[%s] signal-check no intent ts=%s handler=%s funnel=%s",
        symbol,
        ts,
        type(decision_handler).__name__,
        funnel,
    )


def _multileg_engine_snapshot(engine: Any, features: Dict[str, Any]) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    state = getattr(engine, "state", None)
    if state is not None:
        snap["active"] = getattr(state, "active", None)
        inv = getattr(state, "inventory", None)
        pending = getattr(state, "pending_orders", None)
        snap["inventory"] = len(inv) if inv is not None else 0
        snap["pending_orders"] = len(pending) if pending is not None else 0
        snap["trend_side"] = getattr(state, "trend_side", None)
        snap["grid_id"] = getattr(state, "grid_id", None)
    snap["semantic_chop"] = features.get(
        "semantic_chop", features.get("bpc_semantic_chop")
    )
    snap["box_prefilter"] = features.get("box_prefilter")
    snap["trend_confidence"] = features.get("trend_confidence")
    snap["trend_direction"] = features.get("trend_direction")
    return snap


def log_multileg_bar_no_actions(
    *,
    strategy: str,
    symbol: str,
    timestamp: str,
    engine: Any,
    features: Dict[str, Any],
) -> None:
    """Log once per processed bar when the multi-leg engine emitted no actions."""
    logger.info(
        "[%s] %s bar-check no actions ts=%s state=%s",
        symbol,
        strategy,
        timestamp,
        _multileg_engine_snapshot(engine, features),
    )
