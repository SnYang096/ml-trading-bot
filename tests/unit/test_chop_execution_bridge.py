"""Unit tests for ChopGridLiveEngine.on_execution_results → on_execution_report bridge.

Covers fatal bug #2: chop engine's ``on_execution_results`` previously only
updated ``order.status`` but never called ``on_execution_report`` to create
``GridPosition`` entries in inventory.  Trend engine already had this bridge.

Fix: on ``action="place"`` + ``status in {"closed","filled"}`` → bridge to
``on_execution_report`` with FILLED status.  Also ``action="market_exit"`` now
removes position from inventory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
    GridOrder,
    GridPosition,
)
from src.order_management.grid_execution_adapter import MultiLegExecutionResult

# ------------------------------------------------------------------ helpers


def _config(tmp_path: Path) -> Path:
    """Minimal chop_grid config for tests."""
    path = tmp_path / "grid.yaml"
    path.write_text(
        """
regime:
  entry_chop_min: 0.40
  exit_chop_below: 0.25
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 3
risk:
  fee_bps: 4.0
""",
        encoding="utf-8",
    )
    return path


def _make_engine(tmp_path: Path) -> ChopGridLiveEngine:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BTCUSDT"
    engine.state.spacing = 100.0
    engine.state.grid_id = "BTCUSDT_2026-01-01 00:00:00+00:00"
    engine.state.last_timestamp = "2026-01-01T01:00:00+00:00"
    return engine


# ------------------------------------------ bridge: filled → inventory created


class TestFilledBridgeCreatesInventory:
    """action='place' + status='filled' should create GridPosition via bridge."""

    def test_filled_limit_creates_inventory(self, tmp_path: Path) -> None:
        """A filled limit order must create a GridPosition in inventory."""
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        # Seed a pending order that will be "filled"
        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="BUY",
                level=1,
                price=50_000.0,
                quantity=0.1,
                status="pending",
            )
        )
        assert len(engine.state.inventory) == 0

        # Simulate execution result: order filled at 50_050
        result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="filled",
            order_id="exch-order-001",
            client_order_id="client-001",
            raw={
                "local_order_id": leg_id,
                "filled_quantity": 0.1,
                "average_price": 50_050.0,
                "trade_time": "2026-01-01T01:00:30+00:00",
            },
        )
        engine.on_execution_results([result])

        # Verify inventory was created
        assert len(engine.state.inventory) == 1
        pos = engine.state.inventory[0]
        assert pos.side == "LONG"
        assert pos.quantity == pytest.approx(0.1)
        assert pos.entry_price == pytest.approx(50_050.0)
        assert pos.symbol == "BTCUSDT"
        assert pos.level == 1

    def test_filled_sell_creates_short_inventory(self, tmp_path: Path) -> None:
        """A filled SELL order must create a SHORT position."""
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_S1"

        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="SELL",
                level=-1,
                price=51_000.0,
                quantity=0.2,
                status="pending",
            )
        )

        result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="filled",
            order_id="exch-order-002",
            raw={
                "local_order_id": leg_id,
                "filled_quantity": 0.2,
                "average_price": 51_050.0,
            },
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 1
        pos = engine.state.inventory[0]
        assert pos.side == "SHORT"
        assert pos.quantity == pytest.approx(0.2)
        assert pos.entry_price == pytest.approx(51_050.0)

    def test_filled_with_closed_status_also_bridges(self, tmp_path: Path) -> None:
        """status='closed' (synonym) should also bridge to inventory."""
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="BUY",
                level=1,
                price=50_000.0,
                quantity=0.1,
                status="pending",
            )
        )

        result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="closed",
            order_id="exch-order-003",
            raw={
                "local_order_id": leg_id,
                "filled": 0.1,
                "price": 50_000.0,
            },
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 1

    def test_filled_uses_order_qty_as_fallback(self, tmp_path: Path) -> None:
        """When raw has no filled qty, should fall back to order.quantity."""
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L2"

        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="BUY",
                level=2,
                price=49_000.0,
                quantity=0.3,
                status="pending",
            )
        )

        result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="filled",
            order_id="exch-order-004",
            raw={
                "local_order_id": leg_id,
                # no filled_quantity or filled
                "average_price": 49_500.0,
            },
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 1
        assert engine.state.inventory[0].quantity == pytest.approx(0.3)

    def test_filled_generates_protection_actions(self, tmp_path: Path) -> None:
        """After bridge, protection actions (TP/SL) should be generated."""
        engine = _make_engine(tmp_path)
        engine.state.center = 50_000.0
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="BUY",
                level=1,
                price=50_000.0,
                quantity=0.1,
                status="pending",
            )
        )

        result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="filled",
            order_id="exch-order-005",
            raw={
                "local_order_id": leg_id,
                "filled_quantity": 0.1,
                "average_price": 50_000.0,
            },
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 1
        # Protection actions should have been generated
        actions = engine.pop_pending_actions()
        # At least 1 protection action (TP or SL depending on price vs center)
        assert len(actions) >= 1
        action_types = {a.get("action") for a in actions}
        assert "place_protection" in action_types


# -------------------------------- bridge: market_exit → inventory removed


class TestMarketExitRemovesInventory:
    """action='market_exit' should remove position from inventory."""

    def test_market_exit_removes_position(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        # Seed inventory with a position
        engine.state.inventory.append(
            GridPosition(
                symbol="BTCUSDT",
                side="LONG",
                level=1,
                entry_price=50_000.0,
                quantity=0.1,
                entry_quantity=0.1,
                entry_time="2026-01-01T01:00:00+00:00",
                leg_id=leg_id,
            )
        )
        assert len(engine.state.inventory) == 1

        # Simulate market_exit
        result = MultiLegExecutionResult(
            action="market_exit",
            symbol="BTCUSDT",
            status="filled",
            raw={
                "leg_id": leg_id,
            },
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 0

    def test_market_exit_skipped_no_position_is_safe(self, tmp_path: Path) -> None:
        """market_exit with status='skipped_no_position' should also remove."""
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_S1"

        engine.state.inventory.append(
            GridPosition(
                symbol="BTCUSDT",
                side="SHORT",
                level=-1,
                entry_price=51_000.0,
                quantity=0.2,
                entry_quantity=0.2,
                entry_time="2026-01-01T01:00:00+00:00",
                leg_id=leg_id,
            )
        )

        result = MultiLegExecutionResult(
            action="market_exit",
            symbol="BTCUSDT",
            status="skipped_no_position",
            raw={"leg_id": leg_id},
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 0

    def test_market_exit_no_position_is_idempotent(self, tmp_path: Path) -> None:
        """market_exit when position already removed should not crash."""
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        # No inventory — position already gone
        result = MultiLegExecutionResult(
            action="market_exit",
            symbol="BTCUSDT",
            status="filled",
            raw={"leg_id": leg_id},
        )
        # Should not raise
        engine.on_execution_results([result])
        assert len(engine.state.inventory) == 0

    def test_market_exit_with_dust_suffix(self, tmp_path: Path) -> None:
        """market_exit with leg_id derived from dust order_id."""
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        engine.state.inventory.append(
            GridPosition(
                symbol="BTCUSDT",
                side="LONG",
                level=1,
                entry_price=50_000.0,
                quantity=0.001,
                entry_quantity=0.001,
                entry_time="2026-01-01T01:00:00+00:00",
                leg_id=leg_id,
            )
        )

        result = MultiLegExecutionResult(
            action="market_exit",
            symbol="BTCUSDT",
            status="filled",
            raw={
                "local_order_id": f"{leg_id}_dust",
            },
        )
        engine.on_execution_results([result])
        assert len(engine.state.inventory) == 0


# ---------------------------------------- no bridge: canceled/rejected orders


class TestCanceledOrdersNoBridge:
    """Canceled/rejected orders should NOT create inventory."""

    def test_canceled_order_not_bridged(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="BUY",
                level=1,
                price=50_000.0,
                quantity=0.1,
                status="pending",
            )
        )

        result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="canceled",
            order_id="exch-order-010",
            raw={"local_order_id": leg_id},
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 0
        # Canceled orders are pruned from pending list at end of on_execution_results
        assert len(engine.state.pending_orders) == 0

    def test_rejected_order_not_bridged(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="BUY",
                level=1,
                price=50_000.0,
                quantity=0.1,
                status="pending",
            )
        )

        result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="rejected",
            order_id="exch-order-011",
            raw={"local_order_id": leg_id},
        )
        engine.on_execution_results([result])

        assert len(engine.state.inventory) == 0


# ------------------------------------------- full round-trip: fill then exit


class TestFullRoundTrip:
    """End-to-end: fill creates inventory → market_exit removes it."""

    def test_fill_then_exit_round_trip(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        leg_id = "BTCUSDT_2026-01-01 00:00:00+00:00_L1"

        # 1. Seed pending order
        engine.state.pending_orders.append(
            GridOrder(
                order_id=leg_id,
                symbol="BTCUSDT",
                side="BUY",
                level=1,
                price=50_000.0,
                quantity=0.1,
                status="pending",
            )
        )

        # 2. Fill it
        fill_result = MultiLegExecutionResult(
            action="place",
            symbol="BTCUSDT",
            status="filled",
            order_id="exch-001",
            client_order_id="client-001",
            raw={
                "local_order_id": leg_id,
                "filled_quantity": 0.1,
                "average_price": 50_050.0,
            },
        )
        engine.on_execution_results([fill_result])
        assert len(engine.state.inventory) == 1
        assert engine.state.inventory[0].side == "LONG"

        # 3. Exit it
        exit_result = MultiLegExecutionResult(
            action="market_exit",
            symbol="BTCUSDT",
            status="filled",
            raw={"leg_id": leg_id},
        )
        engine.on_execution_results([exit_result])
        assert len(engine.state.inventory) == 0
