"""Daemon loop for multi-leg live/shadow strategies."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator
from src.order_management.multi_leg_risk_governor import RiskRejection
from src.time_series_model.live.decision_chain_debug import (
    chain_debug_enabled,
    log_multileg_bar_no_actions,
)
from src.time_series_model.live.metrics_exporter import METRICS
from src.time_series_model.live.non_trend_funnel import funnel_for_multileg_bar

logger = logging.getLogger(__name__)

_MAX_RISK_LOG_REASON = 420


def _env_record_hedge_bar_tick_metrics() -> bool:
    """Heartbeat counters on mlbot_strategy_event_total; off by default (noisy)."""
    v = os.environ.get("MLBOT_HEDGE_BAR_TICK_METRICS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _risk_reject_metric_code(reason: str) -> str:
    """Map portfolio governor prose to low-cardinality Prometheus label."""
    r = (reason or "").lower()
    if "max_drawdown" in r:
        return "max_drawdown"
    if "max_resting_orders" in r:
        return "max_resting_orders"
    if "unsupported place side" in r or "unsupported side" in r:
        return "unsupported_side"
    if "max_symbol_gross_notional" in r:
        return "symbol_gross_limit"
    if "exchange_min_qty" in r or "exchange_min_notional" in r:
        return "exchange_min_order"
    if "max_symbol_net_notional" in r:
        return "symbol_net_limit"
    if "max_gross_notional" in r:
        return "gross_limit"
    if "max_net_notional" in r:
        return "net_limit"
    return "other"


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
        """Return newly completed bars in processing order."""


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
        reconcile_interval_seconds: float = 60.0,
        stats_collector: Optional[Any] = None,
        funnel_flusher: Optional[Any] = None,
    ) -> None:
        self.bar_provider = bar_provider
        self.runtimes = list(runtimes)
        self.poll_seconds = float(poll_seconds)
        self.reconcile_interval_seconds = max(0.0, float(reconcile_interval_seconds))
        self._last_processed: set[tuple[str, str, str]] = set()
        self._last_exchange_synced_at: Dict[str, float] = {}
        self._last_reconciled_at: Dict[str, float] = {}
        self._running = False
        # Optional 15min funnel hooks (writes A/C-layer rows to live_monitor.db
        # so the console "策略漏斗" panel matches B-layer trend stats).
        self.stats_collector = stats_collector
        self.funnel_flusher = funnel_flusher

    def run_once(self) -> MultiLegDaemonReport:
        symbols = sorted({rt.symbol for rt in self.runtimes})
        bars = self.bar_provider.latest_closed_bars(symbols)
        # Contract: at most one multi-leg strategy may hold/open on one symbol.
        symbol_owner: Dict[str, str] = {}
        for rt in self.runtimes:
            if not self._runtime_holds_symbol(rt):
                continue
            sym = str(rt.symbol or "").upper().strip()
            if sym and sym not in symbol_owner:
                symbol_owner[sym] = str(rt.name or "")
        action_count = 0
        rejected_count = 0
        execution_count = 0
        reconciliation_issue_count = 0
        bars_seen = 0
        reconcile_due_symbols = {
            str(sym).upper(): self._reconcile_due(sym) for sym in symbols
        }
        exchange_snapshots: Dict[
            str, tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        ] = {}

        for bar in bars:
            for rt in self.runtimes:
                if str(rt.symbol).upper() != str(bar.symbol).upper():
                    continue
                key = (rt.name, rt.symbol, bar.timestamp)
                if key in self._last_processed:
                    continue
                self._last_processed.add(key)
                bars_seen += 1
                sym = str(rt.symbol or "").upper().strip()
                exchange_orders = None
                exchange_positions = None
                live_sync = getattr(rt.engine, "sync_live_exchange_state", None)
                if callable(live_sync):
                    exchange_orders, exchange_positions = self._exchange_snapshot(
                        sym, rt, exchange_snapshots
                    )
                    self._last_exchange_synced_at[sym] = time.monotonic()
                    live_sync(
                        exchange_orders=exchange_orders,
                        exchange_positions=exchange_positions,
                    )
                actions = rt.engine.on_bar(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    atr=bar.atr,
                    features=bar.features,
                )
                # Refresh from engine slot state so chop regime_exit on this bar
                # releases the symbol before trend on_bar (timeline parity).
                self._refresh_symbol_owner(symbol_owner, sym)
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
                        try:
                            for da in dropped:
                                side_conflict = str(
                                    (da or {}).get("side", "na") or "na"
                                ).lower()
                                METRICS.multi_leg_risk_reject_codes_total.labels(
                                    strategy=rt.name,
                                    symbol=rt.symbol,
                                    code="symbol_conflict",
                                ).inc(1)
                                METRICS.record_strategy_event(
                                    scope="hedge",
                                    strategy=rt.name,
                                    symbol=rt.symbol,
                                    event="symbol_conflict",
                                    side=side_conflict,
                                )
                        except Exception:
                            logger.debug(
                                "multi-leg symbol-conflict metrics skipped",
                                exc_info=True,
                            )
                action_count += len(actions)
                if not actions and chain_debug_enabled("multi_leg"):
                    log_multileg_bar_no_actions(
                        strategy=str(rt.name),
                        symbol=str(bar.symbol),
                        timestamp=str(bar.timestamp),
                        engine=rt.engine,
                        features=dict(bar.features or {}),
                    )
                should_reconcile = bool(actions) or bool(reconcile_due_symbols.get(sym))
                if should_reconcile:
                    if exchange_orders is None or exchange_positions is None:
                        exchange_orders, exchange_positions = self._exchange_snapshot(
                            sym, rt, exchange_snapshots
                        )
                    # Force sync inventory with exchange truth during reconcile
                    live_sync = getattr(rt.engine, "sync_live_exchange_state", None)
                    if callable(live_sync):
                        self._last_exchange_synced_at[sym] = time.monotonic()
                        live_sync(
                            exchange_orders=exchange_orders,
                            exchange_positions=exchange_positions,
                        )
                    self._last_reconciled_at[sym] = time.monotonic()
                report = rt.orchestrator.run_actions(
                    actions,
                    exchange_orders=exchange_orders,
                    exchange_positions=exchange_positions,
                    reconcile=should_reconcile,
                )
                if self.stats_collector is not None:
                    self._record_multileg_funnel(rt=rt, actions=actions, report=report)
                rejected_count += len(report.risk.rejected)
                self._refresh_symbol_owner(symbol_owner, sym)
                execution_count += len(report.execution_results) + len(
                    report.reconciliation_results
                )
                if report.reconciliation is not None and not report.reconciliation.ok:
                    reconciliation_issue_count += 1
                try:
                    timeframe = str(
                        (bar.features or {}).get("_signal_timeframe") or "primary"
                    )
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
                        len(report.execution_results)
                        + len(report.reconciliation_results)
                    )
                    if (
                        report.reconciliation is not None
                        and not report.reconciliation.ok
                    ):
                        METRICS.multi_leg_reconciliation_issues_total.labels(
                            strategy=rt.name
                        ).inc(1)
                    METRICS.update_strategy_symbol_ohlc(
                        strategy=rt.name,
                        symbol=rt.symbol,
                        timeframe=timeframe,
                        values=bar.features or {},
                    )
                    METRICS.update_strategy_feature_values(
                        strategy=rt.name,
                        symbol=rt.symbol,
                        timeframe=timeframe,
                        values=bar.features or {},
                        layer="hedge",
                    )
                    for action in report.risk.approved_actions or []:
                        if not isinstance(action, dict):
                            continue
                        evt = str(action.get("action", "action") or "action").lower()
                        side = str(action.get("side", "na") or "na").lower()
                        price = action.get("price")
                        METRICS.record_strategy_event(
                            scope="hedge",
                            strategy=rt.name,
                            symbol=rt.symbol,
                            event=evt,
                            side=side,
                            price=(
                                price if isinstance(price, (int, float, str)) else None
                            ),
                        )
                    for rej in report.risk.rejected or []:
                        rejected_action: Dict[str, Any]
                        reason_txt = ""
                        if isinstance(rej, RiskRejection):
                            rejected_action = rej.action or {}
                            reason_txt = rej.reason or ""
                        elif isinstance(rej, dict):
                            rejected_action = rej
                            reason_txt = str(rej.get("reason", "") or "")
                        else:
                            continue
                        code = _risk_reject_metric_code(reason_txt)
                        side_r = str(
                            (rejected_action or {}).get("side", "na") or "na"
                        ).lower()
                        act_kind = str(
                            (rejected_action or {}).get("action", "") or ""
                        ).lower()
                        METRICS.multi_leg_risk_reject_codes_total.labels(
                            strategy=rt.name, symbol=rt.symbol, code=code
                        ).inc(1)
                        METRICS.record_strategy_event(
                            scope="hedge",
                            strategy=rt.name,
                            symbol=rt.symbol,
                            event="risk_reject",
                            side=side_r,
                        )
                        logger.info(
                            "multi-leg risk veto: strategy=%s symbol=%s "
                            "code=%s action=%s side=%s reason=%s",
                            rt.name,
                            rt.symbol,
                            code,
                            act_kind or "na",
                            side_r,
                            (reason_txt or "")[:_MAX_RISK_LOG_REASON],
                        )
                    for result in report.execution_results or []:
                        evt = str(getattr(result, "action", "execution") or "execution")
                        side = "na"
                        price = None
                        raw = getattr(result, "raw", None)
                        if isinstance(raw, dict):
                            side = str(raw.get("side", side) or side).lower()
                            price = raw.get("price")
                        METRICS.record_strategy_event(
                            scope="hedge",
                            strategy=rt.name,
                            symbol=rt.symbol,
                            event=evt.lower(),
                            side=side,
                            price=(
                                price if isinstance(price, (int, float, str)) else None
                            ),
                        )
                    if exchange_positions is not None:
                        METRICS.update_position_metrics(
                            scope="hedge",
                            strategy=rt.name,
                            positions=exchange_positions,
                        )
                    if _env_record_hedge_bar_tick_metrics():
                        METRICS.record_strategy_event(
                            scope="hedge",
                            strategy=rt.name,
                            symbol=rt.symbol,
                            event="bar_tick",
                            side="na",
                        )
                except Exception:
                    logger.debug("multi-leg metrics update skipped", exc_info=True)

        return MultiLegDaemonReport(
            bars_seen=bars_seen,
            action_count=action_count,
            rejected_count=rejected_count,
            execution_count=execution_count,
            reconciliation_issue_count=reconciliation_issue_count,
        )

    def _record_multileg_funnel(
        self,
        *,
        rt: StrategyRuntime,
        actions: List[Dict[str, Any]],
        report: Any,
    ) -> None:
        """Push one bar of multi-leg funnel + order counts into stats_collector."""
        sc = self.stats_collector
        if sc is None:
            return
        try:
            sc.record_bar_processed(1)
            engine_audit = getattr(rt.engine, "_last_bar_audit", None)
            funnel = funnel_for_multileg_bar(
                strategy=str(rt.name or ""),
                engine_audit=engine_audit,
                actions=actions,
                approved_actions=report.risk.approved_actions or [],
                rejected=report.risk.rejected or [],
            )
            sc.record_strategy_eval(rt.symbol, rt.name, funnel)
            placed = sum(
                1
                for a in (report.risk.approved_actions or [])
                if isinstance(a, dict)
                and str(a.get("action", "") or "").lower() == "place"
            )
            for _ in range(placed):
                sc.record_order_placed(rt.symbol, rt.name)
        except Exception:
            logger.debug("multi-leg funnel record skipped", exc_info=True)

    def _refresh_symbol_owner(self, symbol_owner: Dict[str, str], sym: str) -> None:
        """Rebuild owner from current engine slots (chop runtimes precede trend)."""
        sym_u = str(sym or "").upper().strip()
        owner = ""
        for rt in self.runtimes:
            if str(rt.symbol or "").upper().strip() != sym_u:
                continue
            if self._runtime_holds_symbol(rt):
                owner = str(rt.name or "")
                break
        if owner:
            symbol_owner[sym_u] = owner
        else:
            symbol_owner.pop(sym_u, None)

    @staticmethod
    def _runtime_holds_symbol(rt: StrategyRuntime) -> bool:
        """True when this engine occupies the symbol (not just filled inventory)."""
        holds = getattr(rt.engine, "holds_real_grid_slot", None)
        if callable(holds):
            try:
                if bool(holds()):
                    return True
            except Exception:
                logger.debug(
                    "multi-leg daemon: holds_real_grid_slot raised for %s",
                    rt.name,
                    exc_info=True,
                )
        try:
            if bool(list(rt.engine.local_position_snapshots())):
                return True
        except Exception:
            logger.debug(
                "multi-leg daemon: local_position_snapshots raised for %s",
                rt.name,
                exc_info=True,
            )
        return False

    def _reconcile_due(self, symbol: str) -> bool:
        if self.reconcile_interval_seconds <= 0:
            return False
        last = self._last_reconciled_at.get(symbol)
        if last is None:
            return True
        return time.monotonic() - last >= self.reconcile_interval_seconds

    def _exchange_snapshot(
        self,
        symbol: str,
        runtime: StrategyRuntime,
        cache: Dict[str, tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sym = str(symbol or "").upper().strip()
        if sym not in cache:
            adapter = runtime.orchestrator.adapter
            cache[sym] = (
                list(adapter.sync_open_orders(sym)),
                list(adapter.sync_positions(sym)),
            )
        return cache[sym]

    async def run_forever(self, *, max_iterations: Optional[int] = None) -> None:
        self._running = True
        iterations = 0
        while self._running:
            report = self.run_once()
            try:
                METRICS.multi_leg_daemon_polls_total.inc(1)
            except Exception:
                logger.debug("multi-leg poll metric skipped", exc_info=True)
            if self.funnel_flusher is not None:
                try:
                    self.funnel_flusher.maybe_flush(symbol="ALL")
                except Exception:
                    logger.debug("multi-leg funnel flush skipped", exc_info=True)
            fmt = (
                "multi-leg daemon tick: bars=%s actions=%s rejected=%s "
                "executed=%s reconcile_issues=%s"
            )
            args = (
                report.bars_seen,
                report.action_count,
                report.rejected_count,
                report.execution_count,
                report.reconciliation_issue_count,
            )
            # Poll loop line is extremely noisy at INFO; detail lives in orchestrator,
            # adapter, and risk veto lines above.
            logger.debug(fmt, *args)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            await asyncio.sleep(self.poll_seconds)

    def stop(self) -> None:
        self._running = False
