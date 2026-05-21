from __future__ import annotations

from pathlib import Path

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
  max_levels_per_side: 1
risk:
  fee_bps: 4.0
""",
        encoding="utf-8",
    )
    return path


def test_ensure_protection_rebuilds_inventory_and_queues_tp(tmp_path: Path) -> None:
    state_path = tmp_path / "bnb.json"
    state_path.write_text(
        """
{
  "grid_id": "BNBUSDT_2026-05-19 08:40:00+00:00",
  "symbol": "BNBUSDT",
  "active": true,
  "center": 643.55,
  "spacing": 6.4355,
  "pending_orders": [],
  "inventory": [],
  "last_timestamp": "2026-05-21T00:00:00+00:00",
  "current_regime": "chop_grid"
}
""",
        encoding="utf-8",
    )
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        bar_simulation=False,
    )
    actions = engine.actions_ensure_protection(
        exchange_positions=[
            {
                "symbol": "BNB/USDT:USDT",
                "side": "long",
                "size": 0.31,
                "entry_price": 637.11,
            }
        ],
        exchange_orders=[],
    )
    assert len(engine.state.inventory) == 1
    assert engine.state.inventory[0].side == "LONG"
    tp_actions = [a for a in actions if a.get("protection_type") == "take_profit"]
    assert len(tp_actions) == 1
    assert tp_actions[0]["order_id"].endswith("_L1_tp")
    assert abs(float(tp_actions[0]["price"]) - 643.5455) < 0.02
    assert tp_actions[0]["post_only"] is False
    assert tp_actions[0]["time_in_force"] == "GTC"


def test_bar_simulation_skips_simulate_targets(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T01:00:00Z",
        high=110.0,
        low=90.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    assert not any(
        a.get("action") == "take_profit" for a in engine.pop_pending_actions()
    )


def test_live_exchange_orders_block_stale_reset_and_new_grid(tmp_path: Path) -> None:
    state_path = tmp_path / "bnb.json"
    state_path.write_text(
        """
{
  "grid_id": "BNBUSDT_old",
  "symbol": "BNBUSDT",
  "active": true,
  "center": 643.55,
  "spacing": 6.4355,
  "pending_orders": [],
  "inventory": [],
  "last_timestamp": "2026-05-21T00:00:00+00:00",
  "current_regime": "chop_grid"
}
""",
        encoding="utf-8",
    )
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        bar_simulation=False,
    )
    engine.sync_live_exchange_state(
        exchange_positions=[],
        exchange_orders=[
            {
                "symbol": "BNB/USDT:USDT",
                "client_order_id": "cg_existing",
                "order_id": "ex_1",
                "side": "sell",
                "price": 656.42,
            }
        ],
    )
    actions = engine.on_bar(
        symbol="BNBUSDT",
        timestamp="2026-05-21T02:00:00+00:00",
        high=654.0,
        low=642.0,
        close=650.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    assert actions == []
    assert engine.state.active is True
    assert engine.state.grid_id == "BNBUSDT_old"


def test_plain_grid_entry_is_not_counted_as_protection(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BNBUSDT"
    engine.state.spacing = 6.4355
    engine.state.inventory.append(
        GridPosition(
            symbol="BNBUSDT",
            side="LONG",
            level=1,
            entry_price=637.11,
            quantity=0.31,
            entry_time="2026-05-21T00:00:00+00:00",
            leg_id="BNBUSDT_grid_L1",
        )
    )
    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[
            {
                "symbol": "BNBUSDT",
                "client_order_id": "cg_grid_s1",
                "side": "sell",
                "price": 643.55,
                "info": {"positionSide": "SHORT", "reduceOnly": "false"},
            }
        ],
    )
    tp_actions = [a for a in actions if a.get("protection_type") == "take_profit"]
    assert len(tp_actions) == 1


def test_stale_protection_id_does_not_block_replacement_tp(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BNBUSDT"
    engine.state.spacing = 6.4355
    engine.state.inventory.append(
        GridPosition(
            symbol="BNBUSDT",
            side="SHORT",
            level=1,
            entry_price=649.99,
            quantity=0.31,
            entry_time="2026-05-21T00:00:00+00:00",
            leg_id="BNBUSDT_grid_S1",
            protection_order_ids=["stale_tp"],
        )
    )

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert engine.state.inventory[0].protection_order_ids == []
    tp_actions = [a for a in actions if a.get("protection_type") == "take_profit"]
    assert len(tp_actions) == 1
    assert tp_actions[0]["side"] == "SHORT"
