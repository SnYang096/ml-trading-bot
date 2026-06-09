"""Live/shadow engine for trend_scalp (C-layer; formerly dual_add_trend).

The engine owns multi-leg inventory state and emits exchange-facing action
dicts. Execution, portfolio risk, and reconciliation are handled by
``MultiLegLiveOrchestrator`` outside this module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config.multileg_config import load_multileg_effective_config
from src.features.semantic_chop import as_finite_float, resolve_semantic_chop
from src.time_series_model.live.regime_box_prefilter import (
    stable_box_blocks_trend_entry,
)
from src.config.strategy_layout import resolve_strategy_config_input
from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_reconciliation import (
    LocalOrderSnapshot,
    LocalPositionSnapshot,
    ReconciliationReport,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualAddEngineConfig:
    entry_trend_min: float = 0.80
    exit_trend_below: float = 0.50
    max_entry_chop: float = 0.25
    max_hold_chop: float = 0.40
    exclude_box_prefilter: bool = True
    step_atr_mult: float = 0.50
    tp_atr_mult: float = 0.25
    tp_pct: float = 0.0005
    tp_abs: float = 0.0
    take_profit_mode: str = "basket"
    fee_bps: float = 4.0
    max_adds_per_side: int = 3
    max_gross_units: int = 4
    max_net_units: int = 2
    max_loss_per_segment: float = 0.01
    protection_stop_mode: str = "catastrophic"
    catastrophic_stop_atr_mult: float = 8.0
    catastrophic_stop_tp_mult: float = 8.0
    flip_action: str = "close_offside_all"
    reseed_on_flip: bool = True
    initial_hedge: bool = True
    entry_order_type: str = "marketable_limit"
    add_order_type: str = "marketable_limit"
    max_slippage_bps: float = 5.0
    pending_timeout_bars: int = 1


@dataclass
class DualAddOrder:
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    reason: str
    seq: int = 0
    status: str = "pending"
    created_at: str = ""
    exchange_order_id: str = ""
    client_order_id: str = ""
    filled_quantity: float = 0.0
    created_bar: int = 0
    reference_price: float = 0.0
    max_slippage_bps: float = 0.0


@dataclass
class DualAddPosition:
    leg_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    seq: int
    entry_time: str
    protection_order_ids: List[str] = field(default_factory=list)


@dataclass
class DualAddTrendState:
    segment_id: str = ""
    symbol: str = ""
    active: bool = False
    center: float = 0.0
    atr: float = 0.0
    last_add_long: float = 0.0
    last_add_short: float = 0.0
    add_long_count: int = 0
    add_short_count: int = 0
    trend_side: str = ""
    realized_pnl: float = 0.0
    pending_orders: List[DualAddOrder] = field(default_factory=list)
    inventory: List[DualAddPosition] = field(default_factory=list)
    last_timestamp: str = ""
    bar_index: int = 0
    last_reconciliation_ok: bool = True
    last_reconciliation_issues: List[str] = field(default_factory=list)
    block_reseed_after_flip: bool = False
    # ── Retroactive-fill guard ──
    # Order lookups keyed by exchange_order_id / client_order_id / order_id.
    # Populated when orders are pruned from pending_orders; never persisted
    # to JSON state (non-serialisable values).  Enables on_execution_report
    # to handle fills detected hours later by the REST backfill.
    _order_history: dict = field(default_factory=dict, repr=False, compare=False)


def _load_dual_add_config(path: str | Path) -> DualAddEngineConfig:
    cfg_path = Path(path)
    config_dir, profile_path, engine_path = resolve_strategy_config_input(cfg_path)
    obj = load_multileg_effective_config(
        config_dir=config_dir,
        strategy_type="trend_scalp",
        profile_path=profile_path,
        engine_path=engine_path,
    )
    regime = obj.get("regime", {}) or {}
    inv = obj.get("inventory", {}) or {}
    spacing = obj.get("add_spacing", {}) or {}
    tp = obj.get("take_profit", {}) or {}
    risk = obj.get("risk", {}) or {}
    order_model = obj.get("order_model", {}) or {}
    return DualAddEngineConfig(
        entry_trend_min=float(regime.get("entry_min", 0.80)),
        exit_trend_below=float(regime.get("exit_below", 0.50)),
        max_entry_chop=float(
            regime.get("cap_entry", regime.get("max_semantic_chop_entry", 0.25))
        ),
        max_hold_chop=float(
            regime.get("cap_hold", regime.get("max_semantic_chop_hold", 0.40))
        ),
        exclude_box_prefilter=bool(regime.get("exclude_box_prefilter", True)),
        step_atr_mult=float(spacing.get("atr_mult", 0.50)),
        tp_atr_mult=float(tp.get("atr_mult", 0.25)),
        tp_pct=float(tp.get("min_pct", 0.0005)),
        tp_abs=float(tp.get("min_abs", 0.0)),
        take_profit_mode=str(tp.get("mode", "basket")),
        fee_bps=float(risk.get("diagnostic_fee_bps", risk.get("fee_bps", 4.0))),
        max_adds_per_side=int(inv.get("max_adds_per_side", 3)),
        max_gross_units=int(inv.get("max_gross_exposure_units", 4)),
        max_net_units=int(inv.get("max_net_exposure_units", 2)),
        max_loss_per_segment=float(risk.get("max_loss_per_segment", 0.01)),
        protection_stop_mode=str(risk.get("protection_stop_mode", "catastrophic")),
        catastrophic_stop_atr_mult=float(risk.get("catastrophic_stop_atr_mult", 8.0)),
        catastrophic_stop_tp_mult=float(risk.get("catastrophic_stop_tp_mult", 8.0)),
        flip_action=str(inv.get("flip_action", "close_offside_all")),
        reseed_on_flip=bool(inv.get("reseed_on_flip", True)),
        initial_hedge=set(inv.get("initial_legs", ["LONG", "SHORT"]))
        == {"LONG", "SHORT"},
        entry_order_type=str(order_model.get("entry_order_type", "marketable_limit")),
        add_order_type=str(order_model.get("add_order_type", "marketable_limit")),
        max_slippage_bps=float(order_model.get("max_slippage_bps", 5.0)),
        pending_timeout_bars=int(order_model.get("pending_timeout_bars", 1)),
    )


class DualAddTrendLiveEngine:
    """Bar-driven dual-add engine that implements multi-leg orchestration hooks."""

    def __init__(
        self,
        *,
        config_path: (
            str | Path
        ) = "config/strategies/trend_scalp/research/calibrate_roll.default.yaml",
        state_path: str | Path = "results/trend_scalp/live_state.json",
        unit_notional: float = 1.0,
        metrics_strategy: str = "",
    ) -> None:
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        cfg_path = Path(config_path)
        config_dir, profile_path, engine_path = resolve_strategy_config_input(cfg_path)
        multileg = load_multileg_effective_config(
            config_dir=config_dir,
            strategy_type="trend_scalp",
            profile_path=profile_path,
            engine_path=engine_path,
        )
        self.regime = multileg.get("regime", {}) or {}
        self._prefilter_rules = multileg.get("rules", []) or []
        self.cfg = _load_dual_add_config(self.config_path)
        self.unit_notional = float(unit_notional)
        self.state = self.load_state()
        self._pending_actions: List[Dict[str, Any]] = []
        self.metrics_strategy = str(metrics_strategy or "")

    def _emit_dual_bar_outcome(self, symbol: str, outcome: str) -> None:
        from src.order_management.hedge_engine_metrics import (
            record_multi_leg_engine_bar_outcome,
        )

        if not self.metrics_strategy:
            return
        record_multi_leg_engine_bar_outcome(
            metrics_strategy=self.metrics_strategy,
            symbol=symbol,
            engine="trend_scalp",
            outcome=outcome,
        )

    def _record_trend_bar_audit(
        self,
        symbol: str,
        *,
        actions: List[Dict[str, Any]],
        active_at_open: bool,
        should_enter: bool,
        trend_conf: float,
        chop: float,
        is_box: bool,
        trend_side: str,
        explicit_outcome: Optional[str] = None,
    ) -> None:
        from src.time_series_model.live.multileg_funnel import trend_scalp_bar_outcome

        outcome = trend_scalp_bar_outcome(
            active_at_open=active_at_open,
            wanted_enter=should_enter,
            trend_conf=trend_conf,
            chop=chop,
            entry_trend_min=self.cfg.entry_trend_min,
            max_entry_chop=self.cfg.max_entry_chop,
            exclude_box=self.cfg.exclude_box_prefilter,
            is_box=is_box,
            actions=actions,
            explicit=explicit_outcome,
        )
        self._last_bar_audit = {
            "engine": "trend_scalp",
            "trend_conf": trend_conf,
            "chop": chop,
            "is_box": is_box,
            "exclude_box_prefilter": self.cfg.exclude_box_prefilter,
            "entry_trend_min": self.cfg.entry_trend_min,
            "wanted_enter": should_enter,
            "active_at_open": active_at_open,
            "trend_side": trend_side,
            "outcome": outcome,
        }
        self._emit_dual_bar_outcome(symbol, outcome)

    def load_state(self) -> DualAddTrendState:
        if not self.state_path.exists():
            return DualAddTrendState()
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return DualAddTrendState(
            segment_id=str(raw.get("segment_id", "")),
            symbol=str(raw.get("symbol", "")),
            active=bool(raw.get("active", False)),
            center=_as_float(raw.get("center")),
            atr=_as_float(raw.get("atr")),
            last_add_long=_as_float(raw.get("last_add_long")),
            last_add_short=_as_float(raw.get("last_add_short")),
            add_long_count=int(raw.get("add_long_count", 0) or 0),
            add_short_count=int(raw.get("add_short_count", 0) or 0),
            trend_side=str(raw.get("trend_side", "")),
            realized_pnl=_as_float(raw.get("realized_pnl")),
            pending_orders=[DualAddOrder(**o) for o in raw.get("pending_orders", [])],
            inventory=[DualAddPosition(**p) for p in raw.get("inventory", [])],
            last_timestamp=str(raw.get("last_timestamp", "")),
            bar_index=int(raw.get("bar_index", 0) or 0),
            last_reconciliation_ok=bool(raw.get("last_reconciliation_ok", True)),
            last_reconciliation_issues=[
                str(x) for x in raw.get("last_reconciliation_issues", [])
            ],
        )

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        raw = asdict(self.state)
        raw.pop("_order_history", None)
        self.state_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def on_bar(
        self,
        *,
        symbol: str,
        timestamp: str,
        high: float,
        low: float,
        close: float,
        atr: float,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Process one completed bar and return desired order actions."""
        actions: List[Dict[str, Any]] = []
        trend_conf = as_finite_float(features.get("trend_confidence"))
        if trend_conf is None:
            trend_conf = 0.0
        chop = resolve_semantic_chop(features, default=1.0)
        if chop is None:
            chop = 1.0
        is_box = stable_box_blocks_trend_entry(
            features,
            self.regime,
            rules=self._prefilter_rules,
        )
        trend_side = (
            "LONG" if str(features.get("trend_direction", "UP")) == "UP" else "SHORT"
        )
        self.state.last_timestamp = timestamp
        if self.state.active and self.state.symbol == symbol:
            self.state.bar_index += 1

        active_at_open = self.state.active and self.state.symbol == symbol

        should_enter = (
            not self.state.active
            and trend_conf >= self.cfg.entry_trend_min
            and chop <= self.cfg.max_entry_chop
            and not (self.cfg.exclude_box_prefilter and is_box)
        )
        if should_enter:
            actions.extend(
                self._start_segment(symbol, timestamp, close, atr, trend_side)
            )
            self.save_state()
            self._record_trend_bar_audit(
                symbol,
                actions=actions,
                active_at_open=active_at_open,
                should_enter=should_enter,
                trend_conf=trend_conf,
                chop=chop,
                is_box=is_box,
                trend_side=trend_side,
                explicit_outcome="segment_open_placed",
            )
            return actions

        if self.state.active and self.state.symbol == symbol:
            actions.extend(self._cancel_stale_pending_orders(timestamp))
            if trend_conf < self.cfg.exit_trend_below or chop > self.cfg.max_hold_chop:
                actions.extend(self._exit_all(close, timestamp, reason="regime_exit"))
            else:
                if trend_side != self.state.trend_side:
                    actions.extend(
                        self._handle_trend_flip(close, timestamp, trend_side)
                    )
                    if not self.cfg.reseed_on_flip:
                        self.state.block_reseed_after_flip = True
                if (
                    not self.state.inventory
                    and not self.state.pending_orders
                    and not self.state.block_reseed_after_flip
                ):
                    actions.extend(
                        self._seed_inventory_orders(close, timestamp, trend_side)
                    )
                    self.save_state()
                    self._record_trend_bar_audit(
                        symbol,
                        actions=actions,
                        active_at_open=True,
                        should_enter=False,
                        trend_conf=trend_conf,
                        chop=chop,
                        is_box=is_box,
                        trend_side=trend_side,
                        explicit_outcome="active_seed_inventory",
                    )
                    return actions
                actions.extend(self._target_exits(high, low, close, timestamp))
                if not self.state.active:
                    self.save_state()
                    has_mx = any(
                        str(a.get("action", "") or "").lower() == "market_exit"
                        for a in actions
                    )
                    self._record_trend_bar_audit(
                        symbol,
                        actions=actions,
                        active_at_open=active_at_open,
                        should_enter=should_enter,
                        trend_conf=trend_conf,
                        chop=chop,
                        is_box=is_box,
                        trend_side=trend_side,
                        explicit_outcome=(
                            "exit_close" if has_mx else "segment_inactive_other"
                        ),
                    )
                    return actions
                actions.extend(self._trend_adds(high, low, timestamp, trend_side))

        self._record_trend_bar_audit(
            symbol,
            actions=actions,
            active_at_open=active_at_open,
            should_enter=should_enter,
            trend_conf=trend_conf,
            chop=chop,
            is_box=is_box,
            trend_side=trend_side,
        )
        self.save_state()
        return actions

    def local_order_snapshots(self) -> List[LocalOrderSnapshot]:
        return [
            LocalOrderSnapshot(
                order_id=o.order_id,
                symbol=o.symbol,
                side=o.side,
                quantity=max(0.0, o.quantity - o.filled_quantity),
                price=o.price,
                exchange_order_id=o.exchange_order_id,
                client_order_id=o.client_order_id,
            )
            for o in self.state.pending_orders
            if o.status not in {"filled", "canceled"}
        ]

    def local_position_snapshots(self) -> List[LocalPositionSnapshot]:
        return [
            LocalPositionSnapshot(p.symbol, p.side, p.quantity)
            for p in self.state.inventory
        ]

    def on_execution_results(self, results: Iterable[GridExecutionResult]) -> None:
        for result in results:
            raw = result.raw or {}
            local_id = str(raw.get("local_order_id") or raw.get("order_id") or "")
            if result.action == "place":
                order = self._find_order(
                    local_id=local_id, client_id=result.client_order_id
                )
                if order is not None:
                    order.exchange_order_id = result.order_id
                    order.client_order_id = result.client_order_id
                    order.status = str(result.status or "submitted")
                    st = order.status.lower()
                    if st in {"canceled", "expired", "rejected"}:
                        order.status = "canceled"
                    elif st in {"closed", "filled"} and result.order_id:
                        filled_qty = _as_float(raw.get("filled"), order.quantity)
                        if filled_qty <= 0:
                            filled_qty = order.quantity
                        fill_px = _as_float(
                            raw.get("average")
                            or raw.get("price")
                            or raw.get("last_filled_price"),
                            order.price,
                        )
                        self.on_execution_report(
                            {
                                "order_id": str(result.order_id),
                                "client_order_id": str(
                                    result.client_order_id
                                    or order.client_order_id
                                    or ""
                                ),
                                "status": "FILLED",
                                "filled_qty": filled_qty,
                                "last_filled_price": fill_px,
                                "trade_time": raw.get("trade_time")
                                or self.state.last_timestamp,
                            }
                        )
            elif result.action == "cancel":
                order = self._find_order(
                    local_id=local_id,
                    exchange_id=result.order_id,
                    client_id=result.client_order_id,
                )
                if order is not None and result.status in {"canceled", "shadow"}:
                    order.status = "canceled"
            elif result.action == "place_protection":
                raw = result.raw or {}
                pos = self._find_position(str(raw.get("leg_id") or ""))
                if pos is not None and result.order_id:
                    pos.protection_order_ids.append(result.order_id)
        # Archive cancelled orders so backfill late-fill lookups still work.
        for o in self.state.pending_orders:
            if o.status == "canceled":
                self._archive_order(o)
        self.state.pending_orders = [
            o for o in self.state.pending_orders if o.status != "canceled"
        ]
        self.save_state()

    def on_reconciliation_report(self, report: ReconciliationReport) -> None:
        missing_ids = {str(o.order_id) for o in report.missing_exchange_orders}
        if missing_ids:
            prunable_ids = {
                str(o.order_id)
                for o in self.state.pending_orders
                if str(o.order_id) in missing_ids
                and bool(o.exchange_order_id or o.client_order_id)
            }
            skipped_ids = sorted(missing_ids - prunable_ids)
            if skipped_ids:
                logger.info(
                    "dual_add_trend: keep %d local-only missing order(s) pending "
                    "(no exchange/client id): %s",
                    len(skipped_ids),
                    skipped_ids[:12],
                )
            if prunable_ids:
                before = len(self.state.pending_orders)
                self.state.pending_orders = [
                    o
                    for o in self.state.pending_orders
                    if str(o.order_id) not in prunable_ids
                ]
                dropped = before - len(self.state.pending_orders)
                if dropped:
                    logger.warning(
                        "dual_add_trend: pruned %d mapped pending order(s) missing on exchange "
                        "(reconcile): %s",
                        dropped,
                        sorted(prunable_ids)[:12],
                    )
        issues: List[str] = []
        issues.extend(
            f"missing_exchange_order:{o.order_id}"
            for o in report.missing_exchange_orders
        )
        issues.extend(
            "orphan_exchange_order:"
            f"{o.get('order_id') or o.get('orderId') or o.get('client_order_id')}"
            for o in report.orphan_exchange_orders
        )
        issues.extend(
            f"position_mismatch:{m.symbol}:{m.side}:{m.local_quantity}->{m.exchange_quantity}"
            for m in report.position_mismatches
        )
        self.state.last_reconciliation_ok = report.ok
        self.state.last_reconciliation_issues = issues
        self.save_state()

    def on_execution_report(self, report: Dict[str, Any]) -> None:
        order = self._find_order(
            exchange_id=str(report.get("order_id") or ""),
            client_id=str(report.get("client_order_id") or ""),
        )
        if order is None:
            return
        status = str(report.get("status") or "").upper()
        filled_qty = _as_float(report.get("filled_qty"), order.filled_quantity)
        last_px = _as_float(report.get("last_filled_price"), order.price)
        if last_px > 0 and order.reference_price > 0:
            if order.side == "BUY":
                slippage_bps = (last_px - order.reference_price) / order.reference_price
            else:
                slippage_bps = (order.reference_price - last_px) / order.reference_price
            report["reference_price"] = order.reference_price
            report["fill_slippage_bps"] = slippage_bps * 10000.0
            report["max_slippage_bps"] = order.max_slippage_bps
        old_filled = float(order.filled_quantity)
        new_filled = max(old_filled, min(filled_qty, order.quantity))
        fill_delta = max(0.0, new_filled - old_filled)
        order.filled_quantity = new_filled
        order.status = status.lower() if status else order.status
        new_position: DualAddPosition | None = None
        if fill_delta > 0:
            pos_side = "LONG" if order.side == "BUY" else "SHORT"
            new_position = DualAddPosition(
                leg_id=f"{order.order_id}_fill{len(self.state.inventory)}",
                symbol=order.symbol,
                side=pos_side,
                entry_price=last_px if last_px > 0 else order.price,
                quantity=fill_delta,
                seq=order.seq,
                entry_time=str(report.get("trade_time") or self.state.last_timestamp),
            )
            self.state.inventory.append(new_position)
        terminal = status in {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}
        if new_position is not None:
            self._pending_actions.extend(
                self._protection_actions(
                    pos=new_position,
                    timestamp=str(
                        report.get("trade_time") or self.state.last_timestamp
                    ),
                )
            )
            # ── Late-fill guard ──
            # When _exit_all ran before the fill report arrived, the segment
            # is already marked inactive but the order was still in
            # pending_orders (the _exit_all fix keeps it there).  The fill
            # creates a position that must be immediately unwound, otherwise
            # the CMS sees a filled entry with no exit.
            if not self.state.active:
                logger.warning(
                    "dual_add_trend late fill after segment exit: symbol=%s "
                    "side=%s qty=%s leg=%s — queueing immediate market_exit",
                    getattr(self.state, "symbol", ""),
                    pos_side,
                    fill_delta,
                    new_position.leg_id,
                )
                self._pending_actions.append(
                    self._market_exit(
                        new_position,
                        last_px if last_px > 0 else order.price,
                        str(report.get("trade_time") or self.state.last_timestamp),
                        "late_fill_cleanup",
                    )
                )
        if terminal or order.filled_quantity >= order.quantity:
            self._archive_order(order)
            self.state.pending_orders = [
                o for o in self.state.pending_orders if o.order_id != order.order_id
            ]
        self.save_state()

    def pop_pending_actions(self) -> List[Dict[str, Any]]:
        actions = list(self._pending_actions)
        self._pending_actions.clear()
        return actions

    def _start_segment(
        self, symbol: str, timestamp: str, close: float, atr: float, trend_side: str
    ) -> List[Dict[str, Any]]:
        self.state = DualAddTrendState(
            segment_id=f"{symbol}_{timestamp}",
            symbol=symbol,
            active=True,
            center=close,
            atr=atr,
            last_add_long=close,
            last_add_short=close,
            trend_side=trend_side,
            last_timestamp=timestamp,
            bar_index=0,
            block_reseed_after_flip=False,
        )
        return self._seed_inventory_orders(close, timestamp, trend_side)

    def _seed_inventory_orders(
        self, close: float, timestamp: str, trend_side: str
    ) -> List[Dict[str, Any]]:
        self.state.last_add_long = close
        self.state.last_add_short = close
        if self.cfg.initial_hedge:
            return [
                self._place_order("BUY", close, timestamp, "initial_long", seq=0),
                self._place_order("SELL", close, timestamp, "initial_short", seq=0),
            ]
        side = "BUY" if trend_side == "LONG" else "SELL"
        return [self._place_order(side, close, timestamp, "initial_trend", seq=0)]

    def _handle_trend_flip(
        self, close: float, timestamp: str, trend_side: str
    ) -> List[Dict[str, Any]]:
        self.state.trend_side = trend_side
        if self.cfg.flip_action == "keep":
            return []
        keep_side = trend_side
        actions: List[Dict[str, Any]] = []
        remaining: List[DualAddPosition] = []
        for pos in self.state.inventory:
            close_pos = pos.side != keep_side
            if self.cfg.flip_action == "close_offside_adds" and pos.seq == 0:
                close_pos = False
            if close_pos:
                actions.append(self._market_exit(pos, close, timestamp, "trend_flip"))
            else:
                remaining.append(pos)
        self.state.inventory = remaining
        return actions

    def _target_exits(
        self, high: float, low: float, close: float, timestamp: str
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        if self.cfg.take_profit_mode == "basket":
            if self.state.inventory and self._basket_pnl_per_capital(
                close
            ) >= self._basket_target_per_capital(close):
                actions.extend(self._exit_all(close, timestamp, reason="basket_tp"))
            return actions
        remaining: List[DualAddPosition] = []
        for pos in self.state.inventory:
            tp = self._tp_distance(pos.entry_price)
            if pos.side == "LONG" and high >= pos.entry_price + tp:
                actions.append(
                    self._market_exit(pos, pos.entry_price + tp, timestamp, "tp")
                )
            elif pos.side == "SHORT" and low <= pos.entry_price - tp:
                actions.append(
                    self._market_exit(pos, pos.entry_price - tp, timestamp, "tp")
                )
            else:
                remaining.append(pos)
        self.state.inventory = remaining
        return actions

    def _trend_adds(
        self, high: float, low: float, timestamp: str, trend_side: str
    ) -> List[Dict[str, Any]]:
        step = self.cfg.step_atr_mult * self.state.atr
        if step <= 0:
            return []
        actions: List[Dict[str, Any]] = []
        if trend_side == "LONG":
            while (
                high >= self.state.last_add_long + step
                and self.state.add_long_count < self.cfg.max_adds_per_side
                and self._can_add("LONG")
            ):
                self.state.last_add_long += step
                self.state.add_long_count += 1
                actions.append(
                    self._place_order(
                        "BUY",
                        self.state.last_add_long,
                        timestamp,
                        "trend_add",
                        seq=self.state.add_long_count,
                    )
                )
        else:
            while (
                low <= self.state.last_add_short - step
                and self.state.add_short_count < self.cfg.max_adds_per_side
                and self._can_add("SHORT")
            ):
                self.state.last_add_short -= step
                self.state.add_short_count += 1
                actions.append(
                    self._place_order(
                        "SELL",
                        self.state.last_add_short,
                        timestamp,
                        "trend_add",
                        seq=self.state.add_short_count,
                    )
                )
        return actions

    def _exit_all(
        self, close: float, timestamp: str, *, reason: str
    ) -> List[Dict[str, Any]]:
        logger.info(
            "dual_add exit_all: symbol=%s reason=%s pending=%d inventory=%d close=%s",
            getattr(self.state, "symbol", ""),
            reason,
            len(self.state.pending_orders),
            len(self.state.inventory),
            close,
        )
        actions = [
            self._market_exit(pos, close, timestamp, reason)
            for pos in self.state.inventory
        ]
        # ── Defer pending_orders clear to on_execution_results ──
        # Previously we cleared pending_orders[] unconditionally BEFORE the
        # cancel actions were executed.  If the exchange rejected a cancel
        # (order already filled), the subsequent fill report could not find
        # the order in pending_orders (on_execution_report → _find_order),
        # the position was never created, and no market_exit was produced.
        # The orphaned filled entry then showed as negative unrealised PnL
        # in the CMS.
        #
        # Now we keep the orders in pending_orders and rely on
        # on_execution_results to prune the ones that were successfully
        # cancelled / filled.  That callback already handles the
        # housekeeping: orders whose status becomes "canceled" or are fully
        # filled get removed from pending_orders at the end of
        # on_execution_results.
        cancel_order_ids = {o.order_id for o in self.state.pending_orders}
        for order in self.state.pending_orders:
            actions.append(
                {
                    "action": "cancel",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "exchange_order_id": order.exchange_order_id,
                    "reason": reason,
                }
            )
        # Keep pending_orders intact; on_execution_results will filter them.
        self.state.inventory = []
        self.state.active = False
        return actions

    def _place_order(
        self, side: str, price: float, timestamp: str, reason: str, *, seq: int
    ) -> Dict[str, Any]:
        qty = self.unit_notional / max(price, 1e-12)
        raw_order_type = (
            self.cfg.add_order_type
            if reason == "trend_add"
            else self.cfg.entry_order_type
        )
        order_type = _normalize_order_type(raw_order_type)
        max_slippage_bps = max(float(self.cfg.max_slippage_bps), 0.0)
        submit_price = float(price)
        time_in_force = None
        if order_type == "marketable_limit":
            slip = max_slippage_bps / 10000.0
            submit_price = (
                price * (1.0 + slip) if side == "BUY" else price * (1.0 - slip)
            )
            time_in_force = "IOC"
        elif order_type == "market":
            submit_price = price
        else:
            order_type = "limit"
        order = DualAddOrder(
            order_id=f"{self.state.segment_id}_{reason}_{side}_{seq}_{len(self.state.pending_orders)}",
            symbol=self.state.symbol,
            side=side,
            price=submit_price,
            quantity=qty,
            reason=reason,
            seq=seq,
            created_at=timestamp,
            created_bar=int(self.state.bar_index),
            reference_price=float(price),
            max_slippage_bps=max_slippage_bps,
        )
        self.state.pending_orders.append(order)
        action = {"action": "place", "order_type": order_type, **asdict(order)}
        if time_in_force:
            action["time_in_force"] = time_in_force
        return action

    def _cancel_stale_pending_orders(self, timestamp: str) -> List[Dict[str, Any]]:
        timeout = int(self.cfg.pending_timeout_bars)
        if timeout <= 0:
            return []
        actions: List[Dict[str, Any]] = []
        kept: List[DualAddOrder] = []
        for order in self.state.pending_orders:
            age = int(self.state.bar_index) - int(order.created_bar)
            if age >= timeout and order.status not in {"filled", "canceled"}:
                order.status = "canceled"
                actions.append(
                    {
                        "action": "cancel",
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "exchange_order_id": order.exchange_order_id,
                        "reason": "pending_timeout",
                        "timestamp": timestamp,
                    }
                )
            else:
                kept.append(order)
        self.state.pending_orders = kept
        if actions:
            logger.info(
                "dual_add pending_timeout: symbol=%s canceled=%d timeout_bars=%s bar_index=%s",
                getattr(self.state, "symbol", ""),
                len(actions),
                timeout,
                self.state.bar_index,
            )
        return actions

    def _market_exit(
        self, pos: DualAddPosition, price: float, timestamp: str, reason: str
    ) -> Dict[str, Any]:
        return {
            "action": "market_exit",
            "order_id": f"{pos.leg_id}_exit_{reason}_{timestamp}",
            "symbol": pos.symbol,
            "side": pos.side,
            "quantity": pos.quantity,
            "exit_price": price,
            "reason": reason,
            "timestamp": timestamp,
        }

    def _protection_actions(
        self, *, pos: DualAddPosition, timestamp: str
    ) -> List[Dict[str, Any]]:
        if self.cfg.protection_stop_mode == "none":
            return []
        tp_dist = self._tp_distance(pos.entry_price)
        # Catastrophic exchange-side stop. Strategy exits can still be faster.
        sl_dist = max(
            tp_dist * self.cfg.catastrophic_stop_tp_mult,
            self.state.atr * max(self.cfg.catastrophic_stop_atr_mult, 1.0),
        )
        if pos.side == "LONG":
            tp = pos.entry_price + tp_dist
            sl = pos.entry_price - sl_dist
        else:
            tp = pos.entry_price - tp_dist
            sl = pos.entry_price + sl_dist
        actions = []
        if self.cfg.take_profit_mode != "basket":
            actions.append(
                {
                    "action": "place_protection",
                    "order_id": f"{pos.leg_id}_tp",
                    "leg_id": pos.leg_id,
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "trigger_price": tp,
                    "protection_type": "take_profit",
                    "timestamp": timestamp,
                }
            )
        actions.append(
            {
                "action": "place_protection",
                "order_id": f"{pos.leg_id}_sl",
                "leg_id": pos.leg_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "trigger_price": sl,
                "protection_type": "stop_loss",
                "timestamp": timestamp,
            }
        )
        return actions

    def _tp_distance(self, entry: float) -> float:
        fee_buffer = 2.0 * (self.cfg.fee_bps / 10000.0) * entry
        net_target = max(
            self.cfg.tp_abs,
            self.cfg.tp_atr_mult * self.state.atr,
            self.cfg.tp_pct * entry,
        )
        return fee_buffer + net_target

    def _position_pnl_pct(self, pos: DualAddPosition, px: float) -> float:
        fee = self.cfg.fee_bps / 10000.0
        if pos.side == "LONG":
            return (px - pos.entry_price) / pos.entry_price - 2.0 * fee
        return (pos.entry_price - px) / pos.entry_price - 2.0 * fee

    def _basket_pnl_per_capital(self, px: float) -> float:
        capital_units = max(2, self.cfg.max_gross_units)
        return (
            sum(self._position_pnl_pct(pos, px) for pos in self.state.inventory)
            / capital_units
        )

    def _basket_target_per_capital(self, px: float) -> float:
        capital_units = max(2, self.cfg.max_gross_units)
        return (self._tp_distance(px) / max(px, 1e-12)) / capital_units

    def _can_add(self, side: str) -> bool:
        long_units = sum(1 for p in self.state.inventory if p.side == "LONG")
        short_units = sum(1 for p in self.state.inventory if p.side == "SHORT")
        if side == "LONG":
            long_units += 1
        else:
            short_units += 1
        return (
            long_units + short_units <= self.cfg.max_gross_units
            and abs(long_units - short_units) <= self.cfg.max_net_units
        )

    def _find_order(
        self,
        *,
        local_id: str = "",
        exchange_id: str = "",
        client_id: str = "",
    ) -> Optional[DualAddOrder]:
        for order in self.state.pending_orders:
            if local_id and order.order_id == local_id:
                return order
            if exchange_id and order.exchange_order_id == exchange_id:
                return order
            if client_id and order.client_order_id == client_id:
                return order
        # ── Retroactive-fill guard ──
        # After a user-stream gap the periodic backfill may detect fills that
        # arrived while the engine was unaware.  By the time the backfill
        # routes the fill back via on_execution_report, the order is no
        # longer in pending_orders.  Keeping a small history lets us match
        # those late fills so the position/exit cycle is completed.
        history = getattr(self.state, "_order_history", None)
        if history:
            for order in history.values():
                if local_id and order.order_id == local_id:
                    return order
                if exchange_id and order.exchange_order_id == exchange_id:
                    return order
                if client_id and order.client_order_id == client_id:
                    return order
        return None

    def _archive_order(self, order: DualAddOrder) -> None:
        """Stash an order so late-fill lookups can still find it."""
        hist = self.state._order_history
        for key in (order.exchange_order_id, order.order_id, order.client_order_id):
            key = str(key or "").strip()
            if key:
                hist[key] = order

    def _find_position(self, leg_id: str) -> Optional[DualAddPosition]:
        if not leg_id:
            return None
        for pos in self.state.inventory:
            if pos.leg_id == leg_id:
                return pos
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_order_type(raw: str) -> str:
    value = str(raw or "limit").strip().lower().replace("-", "_")
    if value in {"market", "limit", "marketable_limit"}:
        return value
    return "limit"
