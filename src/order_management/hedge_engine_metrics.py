"""Low-cardinality Prometheus hooks for multi-leg **live engines** (chop / dual-add).

Callers pass ``metrics_strategy`` only when wired from ``run_multi_leg_live``; unit
tests leave it empty so counters stay inert.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def record_multi_leg_engine_bar_outcome(
    *,
    metrics_strategy: str,
    symbol: str,
    engine: str,
    outcome: str,
) -> None:
    """One classification per processed bar (why flat / open / holding / exit)."""
    if not metrics_strategy:
        return
    try:
        from src.time_series_model.live.metrics_exporter import METRICS

        METRICS.multi_leg_engine_bar_outcome_total.labels(
            strategy=str(metrics_strategy),
            symbol=str(symbol or "").strip().upper(),
            engine=str(engine),
            outcome=str(outcome),
        ).inc(1)
    except Exception:
        logger.debug("hedge engine bar outcome metric skipped", exc_info=True)
