from __future__ import annotations

from pathlib import Path

import pytest

from src.order_management.grid_execution_adapter import derive_multileg_client_order_id
from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
    GridPosition,
)
from src.time_series_model.live.segment_lifecycle import SegmentState


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


def _exchange_pos(
    *,
    symbol: str = "BNBUSDT",
    side: str = "long",
    quantity: float = 0.31,
    entry_price: float = 637.11,
) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "size": quantity,
        "entry_price": entry_price,
    }


def test_ensure_protection_prunes_stale_inventory_when_exchange_flat(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BNBUSDT"
    engine.state.spacing = 6.4355
    engine.state.segment_state = (
        SegmentState.ACTIVE.value
    )  # Ensure active persists through save_state
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
    engine.save_state()

    actions = engine.actions_ensure_protection(
        exchange_positions=[],
        exchange_orders=[],
    )

    assert engine.state.inventory == []
    assert actions == []
    persisted = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        bar_simulation=False,
    )
    assert persisted.state.inventory == []


def test_ensure_protection_handles_hedge_both_sides_independently(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BNBUSDT"
    engine.state.spacing = 6.4355
    engine.state.inventory.extend(
        [
            GridPosition(
                symbol="BNBUSDT",
                side="LONG",
                level=1,
                entry_price=637.11,
                quantity=0.31,
                entry_time="2026-05-21T00:00:00+00:00",
                leg_id="BNBUSDT_grid_L1",
            ),
            GridPosition(
                symbol="BNBUSDT",
                side="SHORT",
                level=1,
                entry_price=649.99,
                quantity=0.31,
                entry_time="2026-05-21T00:00:00+00:00",
                leg_id="BNBUSDT_grid_S1",
            ),
        ]
    )
    engine._sync_inventory_from_exchange(
        [_exchange_pos(side="long", quantity=0.31, entry_price=637.11)],
        symbol="BNBUSDT",
    )
    sides = sorted(p.side for p in engine.state.inventory)
    assert sides == ["LONG"]


def test_sync_inventory_does_not_duplicate_when_avg_entry_price_differs(
    tmp_path: Path,
) -> None:
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
    engine._sync_inventory_from_exchange(
        [_exchange_pos(side="long", quantity=0.31, entry_price=637.105)],
        symbol="BNBUSDT",
    )
    assert len(engine.state.inventory) == 1
    leg = engine.state.inventory[0]
    assert leg.entry_price == 637.11
    assert leg.quantity == pytest.approx(0.31, rel=1e-9)


def test_sync_inventory_caps_local_qty_when_exchange_smaller(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BNBUSDT"
    engine.state.spacing = 6.4355
    engine.state.inventory.extend(
        [
            GridPosition(
                symbol="BNBUSDT",
                side="LONG",
                level=1,
                entry_price=637.11,
                quantity=0.31,
                entry_time="2026-05-21T00:00:00+00:00",
                leg_id="BNBUSDT_grid_L1",
            ),
            GridPosition(
                symbol="BNBUSDT",
                side="LONG",
                level=2,
                entry_price=635.50,
                quantity=0.31,
                entry_time="2026-05-21T00:01:00+00:00",
                leg_id="BNBUSDT_grid_L2",
            ),
        ]
    )
    engine._sync_inventory_from_exchange(
        [_exchange_pos(side="long", quantity=0.31, entry_price=636.30)],
        symbol="BNBUSDT",
    )
    assert len(engine.state.inventory) == 1
    assert engine.state.inventory[0].leg_id == "BNBUSDT_grid_L1"
    assert engine.state.inventory[0].quantity == pytest.approx(0.31, rel=1e-9)


def test_sync_inventory_preserves_other_symbol_legs(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BNBUSDT"
    engine.state.spacing = 6.4355
    engine.state.inventory.extend(
        [
            GridPosition(
                symbol="ETHUSDT",
                side="LONG",
                level=1,
                entry_price=2500.0,
                quantity=0.05,
                entry_time="2026-05-21T00:00:00+00:00",
                leg_id="ETHUSDT_grid_L1",
            ),
            GridPosition(
                symbol="BNBUSDT",
                side="LONG",
                level=1,
                entry_price=637.11,
                quantity=0.31,
                entry_time="2026-05-21T00:00:00+00:00",
                leg_id="BNBUSDT_grid_L1",
            ),
        ]
    )
    engine._sync_inventory_from_exchange([], symbol="BNBUSDT")
    sides = [(p.symbol, p.side) for p in engine.state.inventory]
    assert sides == [("ETHUSDT", "LONG")]


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


def test_foreign_exchange_position_does_not_activate_chop_grid(tmp_path: Path) -> None:
    state_path = tmp_path / "btc.json"
    state_path.write_text(
        """
{
  "grid_id": "",
  "symbol": "BTCUSDT",
  "active": false,
  "center": 0.0,
  "spacing": 0.0,
  "pending_orders": [],
  "inventory": [],
  "last_timestamp": "",
  "current_regime": "idle"
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
        exchange_positions=[
            _exchange_pos(side="long", quantity=0.001, entry_price=65000.0)
        ],
        exchange_orders=[],
    )
    assert engine.state.active is False
    assert engine._exchange_open_orders is False
    assert engine.state.inventory == []


def test_foreign_position_with_stale_active_chop_does_not_market_exit(
    tmp_path: Path,
) -> None:
    """Regression: trend-only exchange qty must not be closed via chop exit_grid."""
    state_path = tmp_path / "btc_stale.json"
    state_path.write_text(
        """
{
  "grid_id": "BTCUSDT_stale",
  "symbol": "BTCUSDT",
  "active": true,
  "center": 65000.0,
  "spacing": 650.0,
  "pending_orders": [],
  "inventory": [],
  "last_timestamp": "2026-06-04T00:00:00+00:00",
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
        exchange_positions=[
            _exchange_pos(side="long", quantity=0.001, entry_price=65000.0)
        ],
        exchange_orders=[],
    )
    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-06-04T01:00:00+00:00",
        high=65100.0,
        low=64900.0,
        close=65050.0,
        atr=500.0,
        features={"semantic_chop": 0.1, "box_prefilter": False},
    )
    assert not any(a.get("action") == "market_exit" for a in actions)
    assert engine.state.inventory == []


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
        exchange_positions=[
            _exchange_pos(
                side="long",
                quantity=0.31,
                entry_price=637.11,
            )
        ],
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
        exchange_positions=[
            _exchange_pos(
                side="short",
                quantity=0.31,
                entry_price=649.99,
            )
        ],
        exchange_orders=[],
    )
    assert engine.state.inventory[0].protection_order_ids == []
    tp_actions = [a for a in actions if a.get("protection_type") == "take_profit"]
    assert len(tp_actions) == 1
    assert tp_actions[0]["side"] == "SHORT"


def test_ensure_protection_skips_when_hashed_client_ids_on_exchange(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "BNBUSDT"
    engine.state.spacing = 6.4355
    leg_id = "BNBUSDT_2026-05-19 08:40:00+00:00_S1"
    engine.state.inventory.append(
        GridPosition(
            symbol="BNBUSDT",
            side="LONG",
            level=1,
            entry_price=637.11,
            quantity=0.31,
            entry_time="2026-05-21T00:00:00+00:00",
            leg_id=leg_id,
        )
    )
    template = engine._protection_actions(
        order_id=leg_id,
        pos=engine.state.inventory[0],
        timestamp="2026-05-21T00:00:00+00:00",
    )
    tp_cid = derive_multileg_client_order_id(
        next(a for a in template if a["protection_type"] == "take_profit")
    )
    tp_px = float(template[0]["price"])
    sl_cid = "cg_legacy_sl_open"

    actions = engine.actions_ensure_protection(
        exchange_positions=[
            _exchange_pos(
                side="long",
                quantity=0.31,
                entry_price=637.11,
            )
        ],
        exchange_orders=[
            {
                "symbol": "BNBUSDT",
                "client_order_id": tp_cid,
                "side": "sell",
                "price": tp_px,
                "quantity": 0.31,
                "info": {"positionSide": "LONG", "reduceOnly": "true"},
            },
            {
                "symbol": "BNBUSDT",
                "client_order_id": sl_cid,
                "type": "STOP_MARKET",
                "side": "sell",
                "quantity": 0.31,
                "info": {"positionSide": "LONG", "clientAlgoId": sl_cid},
            },
        ],
    )
    assert [a.get("action") for a in actions] == ["cancel"]
    assert actions[0].get("reason") == "per_leg_stop_loss_disabled"


def test_partial_tp_qty_does_not_count_as_full_protection(tmp_path: Path) -> None:
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
            entry_price=653.205,
            quantity=0.62,
            entry_time="2026-05-21T00:00:00+00:00",
            leg_id="BNBUSDT_2026-05-19 08:40:00+00:00_S1",
        )
    )
    tp_px = engine._tp_price_for_position(engine.state.inventory[0])
    actions = engine.actions_ensure_protection(
        exchange_positions=[
            _exchange_pos(
                side="short",
                quantity=0.62,
                entry_price=653.205,
            )
        ],
        exchange_orders=[
            {
                "symbol": "BNBUSDT",
                "client_order_id": "cg_16738f8fae98",
                "side": "buy",
                "price": tp_px,
                "quantity": 0.31,
                "info": {"positionSide": "SHORT", "reduceOnly": "true"},
            }
        ],
    )
    tp_actions = [a for a in actions if a.get("protection_type") == "take_profit"]
    sl_actions = [a for a in actions if a.get("protection_type") == "stop_loss"]
    # Partial TP already live under the deterministic cg_* client id — do not duplicate.
    assert len(tp_actions) == 0
    assert sl_actions == []
