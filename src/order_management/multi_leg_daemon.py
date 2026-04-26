"""Daemon loop for multi-leg live/shadow strategies."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator

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
            action_count += len(actions)
            report = rt.orchestrator.run_actions(actions)
            rejected_count += len(report.risk.rejected)
            execution_count += len(report.execution_results) + len(
                report.reconciliation_results
            )
            if report.reconciliation is not None and not report.reconciliation.ok:
                reconciliation_issue_count += 1

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
