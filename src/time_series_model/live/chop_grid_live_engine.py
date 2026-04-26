"""Dry-run live engine for the standalone chop_grid strategy.

The engine is intentionally independent from GenericLiveStrategy because a grid
owns multiple limit orders and inventory levels, not one TradeIntent position.
It returns desired order actions and leaves exchange-specific adapters outside.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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


@dataclass
class GridPosition:
    symbol: str
    side: str
    level: int
    entry_price: float
    quantity: float
    entry_time: str


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_grid_config(path: str | Path) -> GridEngineConfig:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    regime = obj.get("regime", {}) or {}
    grid = obj.get("grid", {}) or {}
    spacing = grid.get("spacing", {}) or {}
    risk = obj.get("risk", {}) or {}
    return GridEngineConfig(
        box_window=int(regime.get("box_window", 120)),
        entry_chop_min=float(regime.get("entry_chop_min", 0.40)),
        exit_chop_below=float(regime.get("exit_chop_below", 0.25)),
        min_segment_bars=int(risk.get("min_segment_bars", 6)),
        max_segment_bars=int(risk.get("max_segment_bars", 120)),
        grid_atr_mult=float(spacing.get("atr_mult", 0.50)),
        grid_min_pct=float(spacing.get("min_pct", 0.004)),
        max_levels_per_side=int(grid.get("max_levels_per_side", 3)),
        fee_bps=float(risk.get("fee_bps", 4.0)),
        maker_fee_bps=float(risk.get("maker_fee_bps", risk.get("fee_bps", 4.0))),
        taker_fee_bps=float(risk.get("taker_fee_bps", risk.get("fee_bps", 4.0))),
        forced_exit_slippage_bps=float(risk.get("forced_exit_slippage_bps", 0.0)),
        funding_cost_bps_per_8h=float(risk.get("funding_cost_bps_per_8h", 0.0)),
        max_loss_per_grid=risk.get("max_loss_per_grid", 0.03),
        max_open_levels_total=risk.get("max_open_levels_total", 6),
    )


class ChopGridLiveEngine:
    """Dry-run grid engine that produces place/cancel/market_exit actions."""

    def __init__(
        self,
        *,
        config_path: str | Path = "config/strategies/chop_grid/grid.yaml",
        state_path: str | Path = "results/chop_grid/live_state.json",
        level_notional: float = 1.0,
    ) -> None:
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.cfg = _load_grid_config(self.config_path)
        self.level_notional = float(level_notional)
        self.state = self.load_state()

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
        )

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(asdict(self.state), ensure_ascii=False, indent=2),
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
                {"action": "cancel", "order_id": order.order_id, "reason": reason}
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

    def _maker_fee_bps(self) -> float:
        if self.cfg.maker_fee_bps is not None:
            return float(self.cfg.maker_fee_bps)
        return float(self.cfg.fee_bps)

    def _taker_fee_bps(self) -> float:
        if self.cfg.taker_fee_bps is not None:
            return float(self.cfg.taker_fee_bps)
        return float(self.cfg.fee_bps)
