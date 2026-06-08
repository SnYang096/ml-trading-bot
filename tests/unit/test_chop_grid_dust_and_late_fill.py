from __future__ import annotations

from pathlib import Path

import pytest

from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
    GridPosition,
)


def _config(tmp_path: Path) -> Path:
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


def test_late_entry_fill_ingests_inventory_and_protection(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "SOLUSDT"
    engine.state.spacing = 0.2
    engine.state.grid_id = "SOLUSDT_2026-06-07 08:57:00+00:00"
    engine.state.last_timestamp = "2026-06-07T09:34:00+00:00"

    engine.on_execution_report(
        {
            "leg_id": "SOLUSDT_2026-06-07 08:57:00+00:00_L2",
            "status": "FILLED",
            "filled_qty": 1.13,
            "last_filled_price": 65.32,
            "trade_time": "2026-06-07T09:34:42+00:00",
            "symbol": "SOLUSDT",
        }
    )
    assert len(engine.state.inventory) == 1
    leg = engine.state.inventory[0]
    assert leg.leg_id.endswith("_L2")
    assert leg.quantity == pytest.approx(1.13)
    assert leg.entry_quantity == pytest.approx(1.13)
    follow_ups = engine.pop_pending_actions()
    assert [a["protection_type"] for a in follow_ups] == ["take_profit", "stop_loss"]


def test_sl_partial_fill_reduces_inventory_qty(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "SOLUSDT"
    engine.state.spacing = 0.2
    engine.state.inventory.append(
        GridPosition(
            symbol="SOLUSDT",
            side="LONG",
            level=1,
            entry_price=65.53,
            quantity=1.13,
            entry_quantity=1.13,
            entry_time="2026-06-07T09:00:00+00:00",
            leg_id="SOLUSDT_grid_L1",
            protection_order_ids=["algo_sl_1"],
        )
    )

    engine.on_execution_report(
        {
            "order_id": "algo_sl_1",
            "client_order_id": "cg_test_sl",
            "leg_id": "SOLUSDT_grid_L1_sl",
            "status": "FILLED",
            "filled_qty": 0.37,
            "trade_time": "2026-06-07T09:25:00+00:00",
        }
    )

    assert len(engine.state.inventory) == 1
    assert engine.state.inventory[0].quantity == pytest.approx(0.76, rel=1e-6)
    assert engine.pop_pending_actions() == []


def test_full_sl_close_removes_leg(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "SOLUSDT"
    engine.state.spacing = 0.2
    engine.state.inventory.append(
        GridPosition(
            symbol="SOLUSDT",
            side="LONG",
            level=1,
            entry_price=65.53,
            quantity=1.13,
            entry_quantity=1.13,
            entry_time="2026-06-07T09:00:00+00:00",
            leg_id="SOLUSDT_grid_L1",
            protection_order_ids=["algo_sl_1"],
        )
    )

    engine.on_execution_report(
        {
            "order_id": "algo_sl_1",
            "leg_id": "SOLUSDT_grid_L1_sl",
            "status": "FILLED",
            "filled_qty": 1.13,
            "trade_time": "2026-06-07T09:25:00+00:00",
        }
    )

    assert engine.state.inventory == []


def test_dust_exit_fill_is_not_reingested(tmp_path: Path) -> None:
    """A `_dust` market-exit fill must not be recovered as a fresh entry leg."""
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "SOLUSDT"
    engine.state.spacing = 0.2

    engine.on_execution_report(
        {
            "leg_id": "SOLUSDT_grid_L1_dust",
            "status": "FILLED",
            "filled_qty": 0.02,
            "last_filled_price": 65.78,
            "symbol": "SOLUSDT",
        }
    )

    assert engine.state.inventory == []
    assert engine.pop_pending_actions() == []


def test_sl_partial_fill_detected_via_order_type(tmp_path: Path) -> None:
    """Production user-stream uses STOP_MARKET, not ``_sl`` suffixes."""
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "SOLUSDT"
    engine.state.spacing = 0.2
    engine.state.inventory.append(
        GridPosition(
            symbol="SOLUSDT",
            side="LONG",
            level=1,
            entry_price=65.53,
            quantity=1.13,
            entry_quantity=1.13,
            entry_time="2026-06-07T09:00:00+00:00",
            leg_id="SOLUSDT_grid_L1",
            protection_order_ids=["2000001079676592"],
        )
    )

    engine.on_execution_report(
        {
            "order_id": "2000001079676592",
            "status": "FILLED",
            "filled_qty": 0.37,
            "order_type": "STOP_MARKET",
            "trade_time": "2026-06-07T09:25:00+00:00",
        }
    )

    assert len(engine.state.inventory) == 1
    assert engine.state.inventory[0].quantity == pytest.approx(0.76, rel=1e-6)
    assert engine.pop_pending_actions() == []


def test_ensure_protection_dust_exit_deduped_across_cycles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MLBOT_CHOP_GRID_MIN_NOTIONAL_USD", "5")
    state_path = tmp_path / "state.json"
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "SOLUSDT"
    engine.state.spacing = 0.2
    engine.state.last_timestamp = "2026-06-08T00:00:00+00:00"
    engine.state.inventory.append(
        GridPosition(
            symbol="SOLUSDT",
            side="LONG",
            level=1,
            entry_price=65.53,
            quantity=0.02,
            entry_quantity=1.13,
            entry_time="2026-06-07T09:00:00+00:00",
            leg_id="SOLUSDT_grid_L1",
        )
    )
    exchange_pos = [
        {
            "symbol": "SOL/USDT:USDT",
            "side": "long",
            "size": 0.02,
            "entry_price": 65.53,
        }
    ]

    first = engine.actions_ensure_protection(
        exchange_positions=exchange_pos,
        exchange_orders=[],
    )
    assert len([a for a in first if a.get("action") == "market_exit"]) == 1
    assert "SOLUSDT_grid_L1" in engine.state.pending_dust_exits

    second = engine.actions_ensure_protection(
        exchange_positions=exchange_pos,
        exchange_orders=[],
    )
    assert second == []


def test_ensure_protection_dust_emits_market_exit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLBOT_CHOP_GRID_MIN_NOTIONAL_USD", "5")
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "SOLUSDT"
    engine.state.spacing = 0.2
    engine.state.last_timestamp = "2026-06-08T00:00:00+00:00"
    engine.state.inventory.append(
        GridPosition(
            symbol="SOLUSDT",
            side="LONG",
            level=1,
            entry_price=65.53,
            quantity=0.02,
            entry_quantity=1.13,
            entry_time="2026-06-07T09:00:00+00:00",
            leg_id="SOLUSDT_grid_L1",
        )
    )

    actions = engine.actions_ensure_protection(
        exchange_positions=[
            {
                "symbol": "SOL/USDT:USDT",
                "side": "long",
                "size": 0.02,
                "entry_price": 65.53,
            }
        ],
        exchange_orders=[],
    )
    exits = [a for a in actions if a.get("action") == "market_exit"]
    tps = [a for a in actions if a.get("protection_type") == "take_profit"]
    assert len(exits) == 1
    assert exits[0]["reason"] == "dust_below_min_notional"
    assert float(exits[0]["quantity"]) == pytest.approx(0.02)
    assert tps == []
