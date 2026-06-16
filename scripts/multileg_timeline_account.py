"""Account tracking for multileg timeline backtest.

Equity is sourced entirely from the mock wallet (``mock.wallet_usdt`` +
unrealized P&L), which is updated per-fill by ``MockBinanceAPI._apply_open`` /
``_apply_reduce`` using real fill qty/price. There is no separate ledger
double-entry: a side ledger keyed by client_order_id could not reliably pair
TP/SL exits back to their entry lots, so it was removed to avoid a misleading
``ledger_realized_pnl`` that diverged from the authoritative wallet equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.order_management.mock_binance_api import MockBinanceAPI
from src.order_management.multi_leg_orchestrator import OrchestrationReport


@dataclass
class MultilegTimelineAccount:
    initial_equity: float
    mock: MockBinanceAPI
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    current_day: str = ""
    halted: bool = False
    halt_reason: str = ""
    trades_ok: int = 0
    trades_rej: int = 0
    max_dd_peak: float = 0.0
    _engine_realized_baseline: Dict[int, float] = field(default_factory=dict)
    _day_start_equity: float = 0.0

    def __post_init__(self) -> None:
        self.mock.set_wallet(float(self.initial_equity))
        self.peak_equity = float(self.initial_equity)
        self._day_start_equity = float(self.initial_equity)

    @property
    def current(self) -> float:
        return float(self.mock.wallet_usdt + self.mock.unrealized_pnl_usdt())

    def drawdown_pct(self) -> float:
        peak = max(self.peak_equity, 1e-12)
        return max(0.0, (self.peak_equity - self.current) / peak)

    def sync_engine_realized_bridge(self, engines: Dict[str, Dict[str, Any]]) -> None:
        """Credit chop internal realized_pnl deltas (simulation-only engines)."""
        for engs in engines.values():
            for eng in engs.values():
                if not bool(getattr(eng, "bar_simulation", False)):
                    continue
                state = getattr(eng, "state", None)
                if state is None:
                    continue
                rp = float(getattr(state, "realized_pnl", 0.0) or 0.0)
                key = id(eng)
                prev = float(self._engine_realized_baseline.get(key, 0.0))
                delta = rp - prev
                if abs(delta) > 1e-12:
                    self.mock.wallet_usdt += delta
                    self._engine_realized_baseline[key] = rp

    def record_orchestration(
        self,
        report: OrchestrationReport,
        *,
        symbol_conflict_drops: int = 0,
    ) -> None:
        self.trades_ok += len(report.execution_results)
        self.trades_rej += len(report.risk.rejected) + int(symbol_conflict_drops)

    def on_bar_close(
        self,
        *,
        day_key: str,
        max_dd: float,
        daily_loss_limit: float,
        ts_label: str,
    ) -> None:
        if self.current_day and day_key != self.current_day:
            # New day — snapshot equity as day start
            self._day_start_equity = self.current
            self.daily_pnl = 0.0
        self.current_day = day_key
        self.daily_pnl = self.current - self._day_start_equity

        if not self.halted and self.current <= 0:
            self.halted = True
            self.halt_reason = f"equity<=0 at {ts_label}"

        if not self.halted and self.current <= self.peak_equity * (1.0 - max_dd):
            self.halted = True
            self.halt_reason = f"dd>{max_dd*100:.0f}% at {ts_label}"

        self.peak_equity = max(self.peak_equity, self.current)
        dd = (self.current - self.peak_equity) / max(self.peak_equity, 1.0)
        self.max_dd_peak = min(self.max_dd_peak, dd)

    def daily_loss_blocks_new_entries(self, daily_loss_limit: float) -> bool:
        return daily_loss_limit > 0 and self.daily_pnl <= -abs(daily_loss_limit)

    def to_summary(self) -> Dict[str, Any]:
        ret_pct = (
            (self.current - self.initial_equity) / max(self.initial_equity, 1.0) * 100.0
        )
        return {
            "equity_start": self.initial_equity,
            "equity_end": self.current,
            "return_pct": ret_pct,
            "peak_equity": self.peak_equity,
            "max_drawdown_pct": self.max_dd_peak * 100.0,
            "trades_ok": self.trades_ok,
            "trades_rej": self.trades_rej,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "realized_pnl": self.current - self.initial_equity,
            "total_fees_usdt": self.mock.total_fees_usdt,
        }
