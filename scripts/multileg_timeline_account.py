"""Account tracking for multileg timeline backtest (ledger + mock wallet)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional

from scripts.account_ledger import AccountLedger
from src.order_management.grid_execution_adapter import MultiLegExecutionResult
from src.order_management.mock_binance_api import MockBinanceAPI
from src.order_management.multi_leg_orchestrator import OrchestrationReport


def _fill_price(raw: Mapping[str, Any], action: Mapping[str, Any]) -> float:
    for key in ("average_price", "price", "avgPrice", "exit_price", "mark_price"):
        val = raw.get(key) if key in raw else action.get(key)
        if val is None:
            continue
        try:
            px = float(val)
        except (TypeError, ValueError):
            continue
        if px > 0:
            return px
    return 0.0


def _filled_qty(raw: Mapping[str, Any], action: Mapping[str, Any]) -> float:
    for key in ("filled_quantity", "filled", "executedQty", "quantity"):
        val = raw.get(key) if key in raw else action.get(key)
        if val is None:
            continue
        try:
            qty = float(val)
        except (TypeError, ValueError):
            continue
        if qty > 0:
            return qty
    return 0.0


def _lot_id(action: Mapping[str, Any], result: MultiLegExecutionResult) -> str:
    raw = dict(result.raw or {})
    for key in ("local_order_id", "order_id", "client_order_id"):
        val = raw.get(key) or action.get(key) or getattr(result, key, None)
        if val:
            return str(val)
    return f"{result.symbol}:{result.action}:{id(result)}"


@dataclass
class MultilegTimelineAccount:
    initial_equity: float
    mock: MockBinanceAPI
    ledger: AccountLedger = field(init=False)
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
        self.ledger = AccountLedger(
            account="multileg_timeline",
            initial_cash_usdt=float(self.initial_equity),
        )
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
                    self.ledger.realized_pnl_usdt += delta
                    self._engine_realized_baseline[key] = rp

    def record_execution_results(
        self,
        results: Iterable[MultiLegExecutionResult],
        *,
        strategy: str,
        fee_bps: float,
    ) -> None:
        fee_rate = max(0.0, float(fee_bps or 0.0)) / 10000.0
        for result in results:
            status = str(result.status or "").lower()
            if status not in {"filled", "submitted"} and result.action != "market_exit":
                continue
            action = dict(result.raw or {})
            if result.action == "place" and status == "filled":
                px = _fill_price(action, action)
                qty = _filled_qty(action, action)
                if px <= 0 or qty <= 0:
                    continue
                notional = px * qty
                side = str(action.get("side", "") or "").upper()
                lot_side = "LONG" if side == "BUY" else "SHORT"
                lid = _lot_id(action, result)
                if self.ledger.get_lot(lid) is None:
                    self.ledger.open_lot(
                        lot_id=lid,
                        strategy=strategy,
                        symbol=str(result.symbol or action.get("symbol", "")),
                        side=lot_side,
                        notional_usdt=notional,
                        entry_price=px,
                        fee_rate=fee_rate,
                        cash_mode="fee_only",
                        opened_at=datetime.utcnow(),
                    )
                else:
                    self.ledger.merge_lot(
                        lot_id=lid,
                        add_notional_usdt=notional,
                        add_price=px,
                        fee_rate=fee_rate,
                    )
            elif result.action == "market_exit" and status in {
                "filled",
                "submitted",
                "skipped_no_position",
            }:
                px = _fill_price(action, action)
                if px <= 0:
                    px = _fill_price(action, action) or float(
                        action.get("exit_price") or 0.0
                    )
                if px <= 0:
                    continue
                lid = _lot_id(action, result)
                if self.ledger.get_lot(lid) is not None:
                    self.ledger.close_lot(lot_id=lid, exit_price=px, fee_rate=fee_rate)

    def record_pending_fills(self, fills: list) -> None:
        """Record fills from MockBinanceAPI.match_pending_orders in the ledger.

        Entry LIMIT fills → open lot; reduce-only (TP/SL) fills → close lot.
        Wallet state is already correct (updated by mock._apply_*); this
        keeps the ledger approximately in sync for PnL/fee reporting.
        """
        fee_rate = max(0.0, float(self.mock.default_fee_bps)) / 10000.0
        for fill in fills:
            qty = float(fill.get("quantity", 0) or 0)
            px = float(fill.get("average_price", 0) or fill.get("price", 0) or 0)
            if qty <= 0 or px <= 0:
                continue
            symbol = str(fill.get("symbol", ""))
            side = str(fill.get("side", "")).upper()
            reduce_only = bool(fill.get("reduce_only", False))
            cid = str(fill.get("client_order_id", ""))
            if reduce_only:
                # TP/SL fill: close lot.  Primary key is client_order_id.
                if self.ledger.get_lot(cid) is not None:
                    self.ledger.close_lot(lot_id=cid, exit_price=px, fee_rate=fee_rate)
                else:
                    # Fallback: close first open lot by symbol + inferred
                    # lot side (SELL fill closes LONG, BUY fills close SHORT).
                    lot_side = "LONG" if side == "SELL" else "SHORT"
                    self.ledger.close_lot_by_symbol(
                        symbol=symbol,
                        side=lot_side,
                        exit_price=px,
                        fee_rate=fee_rate,
                    )
            else:
                # Entry LIMIT fill: open lot.
                if self.ledger.get_lot(cid) is None:
                    lot_side = "LONG" if side == "BUY" else "SHORT"
                    self.ledger.open_lot(
                        lot_id=cid,
                        strategy="pending_fill",
                        symbol=symbol,
                        side=lot_side,
                        notional_usdt=qty * px,
                        entry_price=px,
                        fee_rate=fee_rate,
                        cash_mode="fee_only",
                        opened_at=datetime.utcnow(),
                    )

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
            "ledger_realized_pnl": self.ledger.realized_pnl_usdt,
            "total_fees_usdt": self.ledger.total_fees_usdt,
        }
