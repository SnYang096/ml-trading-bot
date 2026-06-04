"""Funnel helpers for non-trend (spot / multi_leg) live runners.

The trend stack writes 15min funnel rows via ``StatsCollector`` from
``run_live.py``; this module gives the spot accumulation and multi-leg
runners the same hook so the console "策略漏斗" panel can show A/C-layer
counts. The runners construct a ``funnel`` dict per evaluation and call
``StatsCollector.record_strategy_eval``; this file only owns the small
shared bits (a wall-clock 15min flusher and per-domain funnel mappers).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ── Path resolution ────────────────────────────────────────────


def default_live_monitor_db_path() -> Path:
    """Same path the trend runner uses, so all layers share one stats_15min table."""
    base = os.getenv("MLBOT_LIVE_BASE", "live/highcap")
    override = os.getenv("MLBOT_STATS_DB_PATH")
    if override:
        return Path(override)
    return Path(base) / "data" / "db" / "live_monitor.db"


# ── Funnel dict builders ───────────────────────────────────────

# Spot has no regime/prefilter chain; multi-leg engines apply their own gating
# inside ``engine.on_bar`` and the portfolio risk governor. Both report into a
# single shape so the console can render them next to trend.


def funnel_for_spot_decision(
    *,
    has_intent: bool,
    can_submit: bool,
    blocker: Optional[str] = None,
) -> Dict[str, Any]:
    """``record_strategy_eval``-shaped dict for one spot bar evaluation."""
    gate_reasons: List[str] = []
    if has_intent and not can_submit and blocker:
        gate_reasons.append(str(blocker)[:60])
    return {
        "regime": True,
        "prefilter": True,
        "direction": bool(has_intent),
        "direction_value": 1 if has_intent else 0,
        "gate": bool(has_intent and can_submit),
        "gate_reasons": gate_reasons,
        "entry_filter": bool(has_intent and can_submit),
        "evidence": bool(has_intent and can_submit),
    }


def funnel_for_multileg_bar(
    *,
    strategy: str = "",
    engine_audit: Optional[Mapping[str, Any]] = None,
    actions: Iterable[Any],
    approved_actions: Iterable[Any],
    rejected: Iterable[Any],
) -> Dict[str, Any]:
    """``record_strategy_eval``-shaped dict for one multi-leg bar evaluation."""
    from src.time_series_model.live.multileg_funnel import (
        funnel_for_multileg_bar as _build,
    )

    return _build(
        strategy=strategy,
        engine_audit=engine_audit,
        actions=actions,
        approved_actions=approved_actions,
        rejected=rejected,
    )


# ── 15min wall-clock flusher ───────────────────────────────────


class FifteenMinFlusher:
    """Calls ``StatsCollector.flush`` once per ``interval_s`` of wall time.

    Trend uses the bar listener's per-bar flush hook; spot/multi-leg loops
    don't have an equivalent symbol-aligned cadence, so a simple monotonic
    timer is used. ``maybe_flush`` is cheap and idempotent — call it once per
    poll iteration.
    """

    def __init__(
        self,
        stats_collector: Any,
        *,
        interval_s: float = 900.0,
        regime: str = "NORMAL",
    ) -> None:
        self.stats_collector = stats_collector
        self.interval_s = max(1.0, float(interval_s))
        self.regime = str(regime or "NORMAL")
        self._last_flush_mono = time.monotonic()

    def maybe_flush(
        self,
        *,
        symbol: str = "ALL",
        positions: Optional[Dict[str, Any]] = None,
        system_health: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.stats_collector is None:
            return False
        now = time.monotonic()
        if now - self._last_flush_mono < self.interval_s:
            return False
        try:
            self.stats_collector.flush(
                regime=self.regime,
                positions=positions or {},
                system_health=system_health or {},
                symbol=symbol,
            )
        finally:
            self._last_flush_mono = now
        return True

    def force_flush(
        self,
        *,
        symbol: str = "ALL",
        positions: Optional[Dict[str, Any]] = None,
        system_health: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.stats_collector is None:
            return False
        try:
            self.stats_collector.flush(
                regime=self.regime,
                positions=positions or {},
                system_health=system_health or {},
                symbol=symbol,
            )
        finally:
            self._last_flush_mono = time.monotonic()
        return True
