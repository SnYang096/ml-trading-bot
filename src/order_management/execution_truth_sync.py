"""Process-local reconciliation metrics helper (ExecutionTruthSync contract).

Not a standalone daemon — imported by existing live processes
(``quant-hedge-multileg``, ``quant-trend-fattail``, ``quant-spot-accum``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

RECONCILIATION_ISSUE_BUCKETS: tuple[str, ...] = (
    "missing_exchange_order",
    "orphan_exchange_order",
    "stale_local_order",
    "position_mismatch",
    "api_error",
    "open_reconcile_updated",
)

# Self-healing actions; do not alone flip reconciliation_ok to 0.
SELF_HEALING_RECONCILIATION_ISSUES: frozenset[str] = frozenset(
    {"open_reconcile_updated"}
)

UNRESOLVED_RECONCILIATION_ISSUES: frozenset[str] = frozenset(
    issue
    for issue in RECONCILIATION_ISSUE_BUCKETS
    if issue not in SELF_HEALING_RECONCILIATION_ISSUES
)


def _issue_count(issue_counts: Mapping[str, Any], issue: str) -> float:
    raw = issue_counts.get(issue, 0)
    try:
        return max(0.0, float(raw or 0))
    except (TypeError, ValueError):
        return 0.0


def reconciliation_ok_from_issues(issue_counts: Optional[Mapping[str, Any]]) -> bool:
    """True when no unresolved reconciliation issue buckets are > 0."""
    buckets = issue_counts or {}
    return all(_issue_count(buckets, issue) <= 0.0 for issue in UNRESOLVED_RECONCILIATION_ISSUES)


def publish_reconciliation_metrics(
    *,
    scope: str,
    strategy: str,
    symbol: str,
    issue_counts: Optional[Mapping[str, Any]] = None,
    ok: Optional[bool] = None,
    source: str = "unspecified",
    ts_seconds: Optional[float] = None,
) -> None:
    """Best-effort reconciliation gauge publish with unified issue buckets."""
    from src.time_series_model.live.metrics_exporter import METRICS

    counts: MutableMapping[str, Any] = dict(issue_counts or {})
    resolved_ok = reconciliation_ok_from_issues(counts) if ok is None else bool(ok)
    try:
        METRICS.update_reconciliation_metrics(
            scope=scope,
            strategy=strategy,
            symbol=symbol,
            ok=resolved_ok,
            issue_counts=counts,
            ts_seconds=ts_seconds,
        )
    except Exception:
        logger.debug(
            "reconciliation metrics publish skipped scope=%s source=%s",
            scope,
            source,
            exc_info=True,
        )
        return
    logger.debug(
        "reconciliation metrics published scope=%s strategy=%s symbol=%s ok=%s source=%s ts=%s",
        scope,
        strategy,
        symbol,
        resolved_ok,
        source,
        float(ts_seconds or time.time()),
    )
