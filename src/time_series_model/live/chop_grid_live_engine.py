"""Dry-run live engine for the standalone chop_grid strategy.

The engine is intentionally independent from GenericLiveStrategy because a grid
owns multiple limit orders and inventory levels, not one TradeIntent position.
It returns desired order actions and leaves exchange-specific adapters outside.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config.multileg_config import load_multileg_effective_config
from src.config.strategy_layout import resolve_strategy_config_input
from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_reconciliation import (
    LocalOrderSnapshot,
    LocalPositionSnapshot,
    ReconciliationReport,
)
from src.time_series_model.grid.chop_grid_engine import GridEngineConfig


@dataclass
class GridOrder:
    order_id: str
    symbol: str
    side: str
    level: int
    price: float
    quantity: float
    status: str = "pending"
    created_at: str = ""
    exchange_order_id: str = ""
    client_order_id: str = ""
    filled_quantity: float = 0.0


@dataclass
class GridPosition:
    symbol: str
    side: str
    level: int
    entry_price: float
    quantity: float
    entry_time: str
    leg_id: str = ""
    protection_order_ids: List[str] = field(default_factory=list)


@dataclass
class GridState:
    grid_id: str = ""
    symbol: str = ""
    active: bool = False
    center: float = 0.0
    spacing: float = 0.0
    realized_pnl: float = 0.0
    pending_orders: List[GridOrder] = field(default_factory=list)
    inventory: List[GridPosition] = field(default_factory=list)
    last_timestamp: str = ""
    current_regime: str = "idle"
    last_reconciliation_ok: bool = True
    last_reconciliation_issues: List[str] = field(default_factory=list)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_grid_config(path: str | Path) -> GridEngineConfig:
    cfg_path = Path(path)
    config_dir, profile_path, engine_path = resolve_strategy_config_input(cfg_path)
    obj = load_multileg_effective_config(
        config_dir=config_dir,
        strategy_type="grid",
        profile_path=profile_path,
        engine_path=engine_path,
    )
    regime = obj.get("regime", {}) or {}
    inv = obj.get("inventory", {}) or {}
    spacing = inv.get("spacing", {}) or {}
    risk = obj.get("risk", {}) or {}
    live = obj.get("live", {}) or {}
    expected_costs = live.get("expected_costs", {}) if isinstance(live, dict) else {}
    if not isinstance(expected_costs, dict):
        expected_costs = {}
    return GridEngineConfig(
        box_window=int(regime.get("box_window", 120)),
        entry_chop_min=float(regime.get("entry_chop_min", 0.40)),
        exit_chop_below=float(regime.get("exit_chop_below", 0.25)),
        min_segment_bars=int(risk.get("min_segment_bars", 6)),
        max_segment_bars=int(risk.get("max_segment_bars", 120)),
        grid_atr_mult=float(spacing.get("atr_mult", 0.50)),
        grid_min_pct=float(spacing.get("min_pct", 0.004)),
        max_levels_per_side=int(inv.get("max_levels_per_side", 3)),
        fee_bps=float(expected_costs.get("fee_bps", risk.get("fee_bps", 4.0))),
        maker_fee_bps=float(
            expected_costs.get(
                "maker_fee_bps", risk.get("maker_fee_bps", risk.get("fee_bps", 4.0))
            )
        ),
        taker_fee_bps=float(
            expected_costs.get(
                "taker_fee_bps", risk.get("taker_fee_bps", risk.get("fee_bps", 4.0))
            )
        ),
        forced_exit_slippage_bps=float(
            expected_costs.get(
                "forced_exit_slippage_bps",
                risk.get("forced_exit_slippage_bps", 0.0),
            )
        ),
        funding_cost_bps_per_8h=float(
            expected_costs.get(
                "funding_cost_bps_per_8h",
                risk.get("funding_cost_bps_per_8h", 0.0),
            )
        ),
        max_loss_per_grid=risk.get("max_loss_per_grid", 0.03),
        max_open_levels_total=risk.get("max_open_levels_total", 6),
    )


class ChopGridLiveEngine:
    """Dry-run grid engine that produces place/cancel/market_exit actions.

    Deployment packages under ``live/highcap/config/strategies`` intentionally keep
    only ``meta.yaml`` plus ``archetypes/``. Runtime mode, state paths and adapters
    come from the live runner CLI/env, not from research YAML.
    """

    def __init__(
        self,
        *,
        config_path: str | Path = "live/highcap/config/strategies/chop_grid",
        state_path: str | Path = "results/chop_grid/live_state.json",
        level_notional: float = 1.0,
    ) -> None:
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.cfg = _load_grid_config(self.config_path)
        self.level_notional = float(level_notional)
        self.state = self.load_state()
        self._pending_actions: List[Dict[str, Any]] = []

    def load_state(self) -> GridState:
        if not self.state_path.exists():
            return GridState()
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return GridState(
            grid_id=str(raw.get("grid_id", "")),
            symbol=str(raw.get("symbol", "")),
            active=bool(raw.get("active", False)),
            center=_as_float(raw.get("center")),
            spacing=_as_float(raw.get("spacing")),
            realized_pnl=_as_float(raw.get("realized_pnl")),
            pending_orders=[GridOrder(**o) for o in raw.get("pending_orders", [])],
            inventory=[GridPosition(**p) for p in raw.get("inventory", [])],
            last_timestamp=str(raw.get("last_timestamp", "")),
            current_regime=str(raw.get("current_regime", "idle")),
            last_reconciliation_ok=bool(raw.get("last_reconciliation_ok", True)),
            last_reconciliation_issues=[
                str(x) for x in raw.get("last_reconciliation_issues", [])
            ],
        )

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(asdict(self.state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def local_order_snapshots(self) -> List[LocalOrderSnapshot]:
        """Expose pending orders for exchange reconciliation."""
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
        """Expose inventory for exchange position reconciliation."""
        return [
            LocalPositionSnapshot(
                symbol=p.symbol,
                side=p.side,
                quantity=p.quantity,
            )
            for p in self.state.inventory
        ]

    def on_execution_results(self, results: Iterable[GridExecutionResult]) -> None:
        """Persist exchange/client ids returned by the execution adapter."""
        for result in results:
            raw = result.raw or {}
            local_id = str(raw.get("order_id") or "")
            if result.action == "place":
                order = self._find_order(
                    local_id=local_id, client_id=result.client_order_id
                )
                if order is not None:
                    order.exchange_order_id = result.order_id
                    order.client_order_id = result.client_order_id
                    order.status = str(result.status or "submitted")
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
        self.state.pending_orders = [
            o for o in self.state.pending_orders if o.status != "canceled"
        ]
        self.save_state()

    def on_reconciliation_report(self, report: ReconciliationReport) -> None:
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
        """Apply normalized user-stream fill updates to local pending inventory."""
        order = self._find_order(
            exchange_id=str(report.get("order_id") or ""),
            client_id=str(report.get("client_order_id") or ""),
        )
        if order is None:
            return
        status = str(report.get("status") or "").upper()
        filled_qty = _as_float(report.get("filled_qty"), order.filled_quantity)
        last_px = _as_float(report.get("last_filled_price"), order.price)
        order.filled_quantity = max(
            order.filled_quantity, min(filled_qty, order.quantity)
        )
        order.status = status.lower() if status else order.status
        if status == "FILLED" or order.filled_quantity >= order.quantity:
            pos_side = "LONG" if order.side == "BUY" else "SHORT"
            self.state.inventory.append(
                GridPosition(
                    symbol=order.symbol,
                    side=pos_side,
                    level=order.level,
                    entry_price=last_px if last_px > 0 else order.price,
                    quantity=order.filled_quantity or order.quantity,
                    entry_time=str(
                        report.get("trade_time") or self.state.last_timestamp
                    ),
                    leg_id=order.order_id,
                )
            )
            self._pending_actions.extend(
                self._protection_actions(
                    order_id=order.order_id,
                    pos=self.state.inventory[-1],
                    timestamp=str(
                        report.get("trade_time") or self.state.last_timestamp
                    ),
                )
            )
            self.state.pending_orders = [
                o for o in self.state.pending_orders if o.order_id != order.order_id
            ]
        self.save_state()

    def pop_pending_actions(self) -> List[Dict[str, Any]]:
        actions = list(self._pending_actions)
        self._pending_actions.clear()
        return actions

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
        """Process one completed bar and return dry-run order actions."""
        actions: List[Dict[str, Any]] = []
        chop = _as_float(
            features.get("bpc_semantic_chop", features.get("semantic_chop")), 0.0
        )
        is_box = bool(features.get("box_prefilter", False))
        should_enter = chop >= self.cfg.entry_chop_min and not is_box
        should_exit = self.state.active and chop < self.cfg.exit_chop_below

        self.state.last_timestamp = timestamp
        if not self.state.active and should_enter:
            actions.extend(self._start_grid(symbol, timestamp, close, atr))

        if self.state.active and self.state.symbol == symbol:
            actions.extend(self._simulate_fills(timestamp, high, low))
            actions.extend(self._simulate_targets(timestamp, high, low))
            if self._risk_stop(close):
                should_exit = True

        if should_exit:
            actions.extend(
                self._exit_grid(timestamp, close, reason="regime_or_risk_exit")
            )

        self.save_state()
        return actions

    def _start_grid(
        self, symbol: str, timestamp: str, close: float, atr: float
    ) -> List[Dict[str, Any]]:
        spacing = max(self.cfg.grid_atr_mult * atr, self.cfg.grid_min_pct * close)
        self.state = GridState(
            grid_id=f"{symbol}_{timestamp}",
            symbol=symbol,
            active=True,
            center=close,
            spacing=spacing,
            last_timestamp=timestamp,
            current_regime="chop_grid",
        )
        actions: List[Dict[str, Any]] = []
        for level in range(1, self.cfg.max_levels_per_side + 1):
            qty = self.level_notional / max(close, 1e-12)
            buy = GridOrder(
                order_id=f"{self.state.grid_id}_L{level}",
                symbol=symbol,
                side="BUY",
                level=level,
                price=close - spacing * level,
                quantity=qty,
                created_at=timestamp,
            )
            sell = GridOrder(
                order_id=f"{self.state.grid_id}_S{level}",
                symbol=symbol,
                side="SELL",
                level=level,
                price=close + spacing * level,
                quantity=qty,
                created_at=timestamp,
            )
            self.state.pending_orders.extend([buy, sell])
            actions.extend(
                [
                    {
                        "action": "place",
                        "order_type": "limit",
                        "expected_liquidity": "maker",
                        "expected_fee_bps": self._maker_fee_bps(),
                        **asdict(buy),
                    },
                    {
                        "action": "place",
                        "order_type": "limit",
                        "expected_liquidity": "maker",
                        "expected_fee_bps": self._maker_fee_bps(),
                        **asdict(sell),
                    },
                ]
            )
        return actions

    def _simulate_fills(
        self, timestamp: str, high: float, low: float
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        remaining: List[GridOrder] = []
        for order in self.state.pending_orders:
            filled = (order.side == "BUY" and low <= order.price) or (
                order.side == "SELL" and high >= order.price
            )
            if not filled:
                remaining.append(order)
                continue
            pos_side = "LONG" if order.side == "BUY" else "SHORT"
            self.state.inventory.append(
                GridPosition(
                    symbol=order.symbol,
                    side=pos_side,
                    level=order.level,
                    entry_price=order.price,
                    quantity=order.quantity,
                    entry_time=timestamp,
                )
            )
            actions.append(
                {
                    "action": "fill",
                    "order_id": order.order_id,
                    "fill_price": order.price,
                }
            )
        self.state.pending_orders = remaining
        return actions

    def _simulate_targets(
        self, timestamp: str, high: float, low: float
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        remaining: List[GridPosition] = []
        fee = self._maker_fee_bps() / 10000.0
        for pos in self.state.inventory:
            if pos.side == "LONG":
                target = pos.entry_price + self.state.spacing
                hit = high >= target
                pnl = (
                    target - pos.entry_price
                ) * pos.quantity - 2.0 * fee * pos.entry_price * pos.quantity
            else:
                target = pos.entry_price - self.state.spacing
                hit = low <= target
                pnl = (
                    pos.entry_price - target
                ) * pos.quantity - 2.0 * fee * pos.entry_price * pos.quantity
            if hit:
                self.state.realized_pnl += pnl
                actions.append(
                    {
                        "action": "take_profit",
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "level": pos.level,
                        "exit_price": target,
                        "pnl": pnl,
                        "entry_liquidity": "maker",
                        "exit_liquidity": "maker",
                        "fee_bps_charged": 2.0 * self._maker_fee_bps(),
                        "timestamp": timestamp,
                    }
                )
            else:
                remaining.append(pos)
        self.state.inventory = remaining
        return actions

    def _risk_stop(self, close: float) -> bool:
        if self.cfg.max_open_levels_total is not None:
            if len(self.state.inventory) > int(self.cfg.max_open_levels_total):
                return True
        if self.cfg.max_loss_per_grid is None:
            return False
        mtm = self.state.realized_pnl
        for pos in self.state.inventory:
            if pos.side == "LONG":
                mtm += (close - pos.entry_price) * pos.quantity
            else:
                mtm += (pos.entry_price - close) * pos.quantity
        return mtm <= -abs(float(self.cfg.max_loss_per_grid)) * max(
            self.level_notional, 1e-12
        )

    def _exit_grid(
        self, timestamp: str, close: float, *, reason: str
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for order in self.state.pending_orders:
            actions.append(
                {
                    "action": "cancel",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "level": order.level,
                    "price": order.price,
                    "reason": reason,
                }
            )
        for pos in self.state.inventory:
            actions.append(
                {
                    "action": "market_exit",
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "level": pos.level,
                    "quantity": pos.quantity,
                    "exit_price": close,
                    "reason": reason,
                    "entry_liquidity": "maker",
                    "exit_liquidity": "taker",
                    "fee_bps_charged": self._maker_fee_bps() + self._taker_fee_bps(),
                    "slippage_bps_charged": self.cfg.forced_exit_slippage_bps,
                    "timestamp": timestamp,
                }
            )
        self.state.pending_orders = []
        self.state.inventory = []
        self.state.active = False
        self.state.current_regime = "idle"
        return actions

    def _protection_actions(
        self, *, order_id: str, pos: GridPosition, timestamp: str
    ) -> List[Dict[str, Any]]:
        """Create native exchange protection orders for a filled grid leg."""

        if self.state.spacing <= 0:
            return []
        if pos.side == "LONG":
            tp = pos.entry_price + self.state.spacing
            sl = pos.entry_price - self.state.spacing * (
                self.cfg.max_levels_per_side + 1
            )
        else:
            tp = pos.entry_price - self.state.spacing
            sl = pos.entry_price + self.state.spacing * (
                self.cfg.max_levels_per_side + 1
            )
        return [
            {
                "action": "place_protection",
                "order_id": f"{order_id}_tp",
                "leg_id": order_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "price": tp,
                "trigger_price": tp,
                "order_type": "limit",
                "protection_type": "take_profit",
                "reduce_only": True,
                "post_only": True,
                "time_in_force": "GTX",
                "timestamp": timestamp,
            },
            {
                "action": "place_protection",
                "order_id": f"{order_id}_sl",
                "leg_id": order_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "trigger_price": sl,
                "protection_type": "stop_loss",
                "timestamp": timestamp,
            },
        ]

    def _maker_fee_bps(self) -> float:
        if self.cfg.maker_fee_bps is not None:
            return float(self.cfg.maker_fee_bps)
        return float(self.cfg.fee_bps)

    def _taker_fee_bps(self) -> float:
        if self.cfg.taker_fee_bps is not None:
            return float(self.cfg.taker_fee_bps)
        return float(self.cfg.fee_bps)

    def _find_order(
        self,
        *,
        local_id: str = "",
        exchange_id: str = "",
        client_id: str = "",
    ) -> Optional[GridOrder]:
        for order in self.state.pending_orders:
            if local_id and order.order_id == local_id:
                return order
            if exchange_id and order.exchange_order_id == exchange_id:
                return order
            if client_id and order.client_order_id == client_id:
                return order
        return None

    def _find_position(self, leg_id: str) -> Optional[GridPosition]:
        if not leg_id:
            return None
        for pos in self.state.inventory:
            if pos.leg_id == leg_id:
                return pos
        return None
