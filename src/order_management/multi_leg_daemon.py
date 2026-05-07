"""Daemon loop for multi-leg live/shadow strategies."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator
from src.time_series_model.live.metrics_exporter import METRICS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiLegBarEvent:
    symbol: str
    timestamp: str
    high: float
    low: float
    close: float
    atr: float
    features: Dict[str, Any] = field(default_factory=dict)


class MultiLegBarProvider(Protocol):
    """Source of completed bars plus features for the multi-leg daemon."""

    def latest_closed_bars(self, symbols: Iterable[str]) -> List[MultiLegBarEvent]:
        """Return at most one latest completed bar per symbol."""


@dataclass(frozen=True)
class StrategyRuntime:
    name: str
    symbol: str
    engine: Any
    orchestrator: MultiLegLiveOrchestrator


@dataclass(frozen=True)
class MultiLegDaemonReport:
    bars_seen: int
    action_count: int
    rejected_count: int
    execution_count: int
    reconciliation_issue_count: int


class MultiLegLiveDaemon:
    """Poll completed bars and route engine actions through orchestrators."""

    def __init__(
        self,
        *,
        bar_provider: MultiLegBarProvider,
        runtimes: Iterable[StrategyRuntime],
        poll_seconds: float = 30.0,
    ) -> None:
        self.bar_provider = bar_provider
        self.runtimes = list(runtimes)
        self.poll_seconds = float(poll_seconds)
        self._last_processed: set[tuple[str, str, str]] = set()
        self._running = False

    def run_once(self) -> MultiLegDaemonReport:
        symbols = sorted({rt.symbol for rt in self.runtimes})
        bars = self.bar_provider.latest_closed_bars(symbols)
        bars_by_symbol = {bar.symbol: bar for bar in bars}
        # Contract: at most one multi-leg strategy may hold/open on one symbol.
        symbol_owner: Dict[str, str] = {}
        for rt in self.runtimes:
            try:
                has_pos = bool(list(rt.engine.local_position_snapshots()))
            except Exception:
                has_pos = False
            if has_pos:
                sym = str(rt.symbol or "").upper().strip()
                if sym and sym not in symbol_owner:
                    symbol_owner[sym] = str(rt.name or "")
        action_count = 0
        rejected_count = 0
        execution_count = 0
        reconciliation_issue_count = 0
        bars_seen = 0

        for rt in self.runtimes:
            bar = bars_by_symbol.get(rt.symbol)
            if bar is None:
                continue
            key = (rt.name, rt.symbol, bar.timestamp)
            if key in self._last_processed:
                continue
            self._last_processed.add(key)
            bars_seen += 1
            actions = rt.engine.on_bar(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                atr=bar.atr,
                features=bar.features,
            )
            sym = str(rt.symbol or "").upper().strip()
            owner = symbol_owner.get(sym, "")
            if owner and owner != str(rt.name or ""):
                dropped = [
                    a
                    for a in actions
                    if str((a or {}).get("action", "") or "").lower() == "place"
                ]
                if dropped:
                    actions = [
                        a
                        for a in actions
                        if str((a or {}).get("action", "") or "").lower() != "place"
                    ]
                    rejected_count += len(dropped)
                    logger.info(
                        "multi-leg symbol conflict: reject %d opening actions for %s (%s owned by %s)",
                        len(dropped),
                        sym,
                        rt.name,
                        owner,
                    )
            action_count += len(actions)
            report = rt.orchestrator.run_actions(actions)
            rejected_count += len(report.risk.rejected)
            if any(
                str((a or {}).get("action", "") or "").lower() == "place"
                for a in (report.risk.approved_actions or [])
            ):
                symbol_owner[sym] = str(rt.name or "")
            execution_count += len(report.execution_results) + len(
                report.reconciliation_results
            )
            if report.reconciliation is not None and not report.reconciliation.ok:
                reconciliation_issue_count += 1
            try:
                METRICS.multi_leg_bars_processed.labels(
                    strategy=rt.name, symbol=rt.symbol
                ).inc(1)
                METRICS.multi_leg_actions_total.labels(
                    strategy=rt.name, symbol=rt.symbol
                ).inc(len(actions))
                METRICS.multi_leg_risk_rejected_total.labels(
                    strategy=rt.name, symbol=rt.symbol
                ).inc(len(report.risk.rejected))
                METRICS.multi_leg_execution_results_total.labels(
                    strategy=rt.name, symbol=rt.symbol
                ).inc(
                    len(report.execution_results) + len(report.reconciliation_results)
                )
                if report.reconciliation is not None and not report.reconciliation.ok:
                    METRICS.multi_leg_reconciliation_issues_total.labels(
                        strategy=rt.name
                    ).inc(1)
            except Exception:
                logger.debug("multi-leg metrics update skipped", exc_info=True)

        return MultiLegDaemonReport(
            bars_seen=bars_seen,
            action_count=action_count,
            rejected_count=rejected_count,
            execution_count=execution_count,
            reconciliation_issue_count=reconciliation_issue_count,
        )

    async def run_forever(self, *, max_iterations: Optional[int] = None) -> None:
        self._running = True
        iterations = 0
        while self._running:
            report = self.run_once()
            try:
                METRICS.multi_leg_daemon_polls_total.inc(1)
            except Exception:
                logger.debug("multi-leg poll metric skipped", exc_info=True)
            logger.info(
                "multi-leg daemon tick: bars=%s actions=%s rejected=%s executed=%s reconcile_issues=%s",
                report.bars_seen,
                report.action_count,
                report.rejected_count,
                report.execution_count,
                report.reconciliation_issue_count,
            )
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            await asyncio.sleep(self.poll_seconds)

    def stop(self) -> None:
        self._running = False
