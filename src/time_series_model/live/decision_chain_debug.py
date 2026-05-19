"""Chain debug logging for live decision paths (trend PCM, multi-leg engines)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# One log per (scope, symbol, 2h bar bucket) by default.
_BAR_DEDUPE_KEYS: Set[str] = set()
_BAR_DEDUPE_MAX = 4000

TREND_FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "ema_200",
    "atr",
    "semantic_chop",
    "bpc_semantic_chop",
    "weekly_ema_200_position",
    "trend_confidence",
    "trend_direction",
    "box_prefilter",
)

SPOT_FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "weekly_ema_200_position",
    "atr_percentile",
)

MULTILEG_FEATURE_KEYS: tuple[str, ...] = (
    "semantic_chop",
    "bpc_semantic_chop",
    "box_prefilter",
    "trend_confidence",
    "trend_direction",
)


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


def _dedupe_bucket() -> str:
    """``MLBOT_CHAIN_DEBUG_BUCKET``: ``2h`` (default) or ``15min``."""
    raw = (os.getenv("MLBOT_CHAIN_DEBUG_BUCKET") or "2h").strip().lower()
    if raw in {"15m", "15min", "15"}:
        return "15min"
    return "2h"


def _bar_dedupe_key(ts: Any) -> str:
    """Floor timestamp to dedupe bucket (2h for trend/spot, 15min optional)."""
    if ts is None:
        return "unknown"
    try:
        import pandas as pd

        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        freq = "15min" if _dedupe_bucket() == "15min" else "2h"
        return t.floor(freq).isoformat()
    except Exception:
        return str(ts)


def throttle_allows(
    scope: str,
    symbol: str,
    bar_ts: Any,
    *,
    strategy: str = "",
) -> bool:
    """At most one chain-debug line per scope/symbol/(strategy)/2h bar."""
    if not chain_debug_enabled(scope):
        return False
    sym = str(symbol or "").upper().strip()
    strat = str(strategy or "").strip().lower()
    parts = [str(scope).lower(), sym, _bar_dedupe_key(bar_ts)]
    if strat:
        parts.append(strat)
    key = ":".join(parts)
    if key in _BAR_DEDUPE_KEYS:
        return False
    if len(_BAR_DEDUPE_KEYS) >= _BAR_DEDUPE_MAX:
        _BAR_DEDUPE_KEYS.clear()
    _BAR_DEDUPE_KEYS.add(key)
    return True


def _compact_funnel(funnel: Any) -> Dict[str, Any]:
    if not isinstance(funnel, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in funnel.items():
        key = str(k)
        if isinstance(v, (bool, int, float)) or v is None:
            out[key] = v
        elif isinstance(v, str):
            out[key] = v if len(v) <= 200 else v[:200] + "…"
        elif isinstance(v, list):
            text = ", ".join(str(x) for x in v[:8])
            if len(v) > 8:
                text += f", …(+{len(v) - 8})"
            out[key] = text
        elif isinstance(v, dict):
            out[key] = {str(sk): v[sk] for sk in list(v.keys())[:6]}
    return out


def infer_block_stage(funnel: Dict[str, Any]) -> str:
    """Best-effort stage where decide() stopped or PCM pre-filtered."""
    f = funnel or {}
    if not f:
        return "not_evaluated"
    if f.get("pcm_direction_filter") is False:
        return "pcm_ema_filter"
    if f.get("simple_deep_bear") is False:
        return "simple_not_deep_bear"
    if str(f.get("accumulation_policy") or "").startswith("bull"):
        return "accumulation_policy"
    if f.get("prefilter") is False:
        return "prefilter_deny"
    if f.get("direction") is False or f.get("direction_value") == 0:
        return "no_direction"
    if f.get("gate") is False:
        return "gate_deny"
    if f.get("entry_filter") is False:
        return "entry_filter_deny"
    if f.get("reject_srb_wide_sr_too_close"):
        return "srb_wide_guard"
    if f.get("prefilter") is True and f.get("direction") is True:
        if f.get("gate") is True and f.get("entry_filter") is not False:
            return "strategy_layers_passed"
    return "unknown"


def summarize_layers(funnel: Dict[str, Any]) -> Dict[str, str]:
    """Fixed layer slots: pass / fail / n/a for quick scanning."""
    f = funnel or {}

    def _mark(present: bool, passed: Any) -> str:
        if not present:
            return "n/a"
        if passed is True:
            return "pass"
        if passed is False:
            return "fail"
        return str(passed)

    return {
        "simple_deep_bear": _mark("simple_deep_bear" in f, f.get("simple_deep_bear")),
        "prefilter": _mark("prefilter" in f, f.get("prefilter")),
        "direction": _mark("direction" in f, f.get("direction")),
        "gate": _mark("gate" in f, f.get("gate")),
        "entry_filter": _mark("entry_filter" in f, f.get("entry_filter")),
    }


def pick_features(features: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in features:
            continue
        val = features.get(key)
        if val is None:
            out[key] = None
        elif isinstance(val, float):
            out[key] = round(val, 6)
        else:
            out[key] = val
    return out


def log_trend_no_intent(
    symbol: str,
    decision_handler: Any,
    features: Dict[str, Any],
) -> None:
    """Log why trend/PCM returned no TradeIntent on this decision cycle."""
    ts = features.get("timestamp")
    if not throttle_allows("trend", symbol, ts):
        return

    from src.time_series_model.portfolio.live_pcm import LivePCM

    feat = pick_features(features, TREND_FEATURE_KEYS)
    if isinstance(decision_handler, LivePCM):
        trace = dict(getattr(decision_handler, "_last_decide_trace", None) or {})
        per_strategy: Dict[str, Any] = {}
        for arch, strat in (
            getattr(decision_handler, "_strategies", None) or {}
        ).items():
            funnel = _compact_funnel(getattr(strat, "_last_funnel", None))
            per_strategy[str(arch)] = {
                "block": infer_block_stage(funnel),
                "layers": summarize_layers(funnel),
                "funnel": funnel,
            }
        logger.info(
            "[%s] signal-check no intent ts=%s features=%s pcm_trace=%s strategies=%s",
            symbol,
            ts,
            feat,
            trace,
            per_strategy,
        )
        return

    funnel = _compact_funnel(getattr(decision_handler, "_last_funnel", None))
    logger.info(
        "[%s] signal-check no intent ts=%s handler=%s block=%s layers=%s "
        "features=%s funnel=%s",
        symbol,
        ts,
        type(decision_handler).__name__,
        infer_block_stage(funnel),
        summarize_layers(funnel),
        feat,
        funnel,
    )


def log_spot_no_intent(
    symbol: str,
    strategy: Any,
    features: Dict[str, Any],
) -> None:
    ts = features.get("timestamp")
    if not throttle_allows("spot", symbol, ts):
        return
    funnel = _compact_funnel(getattr(strategy, "_last_funnel", None) or {})
    logger.info(
        "[%s] signal-check no intent ts=%s block=%s layers=%s features=%s funnel=%s",
        symbol,
        ts,
        infer_block_stage(funnel),
        summarize_layers(funnel),
        pick_features(features, SPOT_FEATURE_KEYS),
        funnel,
    )


def _multileg_regime_snapshot(engine: Any, features: Dict[str, Any]) -> Dict[str, Any]:
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
        snap["spacing"] = getattr(state, "spacing", None)
    chop = features.get("semantic_chop", features.get("bpc_semantic_chop"))
    snap["features"] = pick_features(features, MULTILEG_FEATURE_KEYS)
    cfg = getattr(engine, "cfg", None)
    if cfg is not None:
        if hasattr(cfg, "entry_chop_min"):
            snap["regime"] = {
                "chop": chop,
                "entry_chop_min": getattr(cfg, "entry_chop_min", None),
                "exit_chop_below": getattr(cfg, "exit_chop_below", None),
                "wanted_enter": (
                    float(chop or 0) >= float(getattr(cfg, "entry_chop_min", 0))
                    if chop is not None
                    else None
                ),
                "should_exit_chop": (
                    float(chop or 1) < float(getattr(cfg, "exit_chop_below", 0))
                    if chop is not None
                    else None
                ),
            }
        elif hasattr(cfg, "entry_trend_min"):
            snap["regime"] = {
                "chop": chop,
                "trend_conf": features.get("trend_confidence"),
                "entry_trend_min": getattr(cfg, "entry_trend_min", None),
                "exit_trend_below": getattr(cfg, "exit_trend_below", None),
                "max_entry_chop": getattr(cfg, "max_entry_chop", None),
                "max_hold_chop": getattr(cfg, "max_hold_chop", None),
                "box_blocked": (
                    bool(getattr(cfg, "exclude_box_prefilter", False))
                    and bool(features.get("box_prefilter"))
                ),
            }
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
    if not throttle_allows("multi_leg", symbol, timestamp, strategy=strategy):
        return
    snap = _multileg_regime_snapshot(engine, features)
    block = "flat_inactive"
    if snap.get("active"):
        block = "active_no_action"
    regime = snap.get("regime") or {}
    if isinstance(regime, dict):
        if regime.get("should_exit_chop") is True:
            block = "would_exit_chop"
        elif regime.get("wanted_enter") is False:
            block = "chop_below_entry"
        elif regime.get("box_blocked"):
            block = "box_prefilter"
        elif regime.get("trend_conf") is not None:
            try:
                tc = float(regime.get("trend_conf"))
                if tc < float(regime.get("entry_trend_min", 0)):
                    block = "trend_conf_low"
                elif float(regime.get("chop") or 1) > float(
                    regime.get("max_entry_chop", 1)
                ):
                    block = "chop_above_entry_cap"
            except (TypeError, ValueError):
                pass
    logger.info(
        "[%s] %s bar-check no actions ts=%s block=%s snapshot=%s",
        symbol,
        strategy,
        timestamp,
        block,
        snap,
    )
