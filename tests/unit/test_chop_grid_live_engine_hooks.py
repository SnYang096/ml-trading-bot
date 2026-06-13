from __future__ import annotations

from pathlib import Path

from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_reconciliation import (
    PositionMismatch,
    ReconciliationReport,
)
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
  max_loss_per_grid: 0.03
  max_open_levels_total: 2
""",
        encoding="utf-8",
    )
    return path


def test_chop_grid_exposes_order_snapshots_and_maps_execution_result(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )

    snapshots = engine.local_order_snapshots()
    assert len(snapshots) == 2
    first_place = actions[0]
    engine.on_execution_results(
        [
            GridExecutionResult(
                action="place",
                status="open",
                symbol="BTCUSDT",
                order_id="ex_1",
                client_order_id="cg_abc",
                raw=first_place,
            )
        ]
    )

    mapped = engine.local_order_snapshots()[0]
    assert mapped.exchange_order_id == "ex_1"
    assert mapped.client_order_id == "cg_abc"


def test_chop_grid_execution_report_moves_filled_order_to_inventory(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    engine.on_execution_results(
        [
            GridExecutionResult(
                action="place",
                status="open",
                symbol="BTCUSDT",
                order_id="ex_1",
                client_order_id="cg_abc",
                raw=actions[0],
            )
        ]
    )

    engine.on_execution_report(
        {
            "order_id": "ex_1",
            "client_order_id": "cg_abc",
            "status": "FILLED",
            "filled_qty": 1.0,
            "last_filled_price": 99.0,
            "trade_time": "2026-01-01T00:02:00Z",
        }
    )

    assert len(engine.local_order_snapshots()) == 1
    positions = engine.local_position_snapshots()
    assert len(positions) == 1
    assert positions[0].side == "LONG"
    assert positions[0].quantity == 1.0
    follow_ups = engine.pop_pending_actions()
    assert [a["protection_type"] for a in follow_ups] == ["take_profit"]
    assert all(a["action"] == "place_protection" for a in follow_ups)
    tp_action = follow_ups[0]
    assert tp_action["order_type"] == "limit"
    assert tp_action["reduce_only"] is True
    assert tp_action["post_only"] is False
    assert tp_action["time_in_force"] == "GTC"
    assert tp_action["price"] == tp_action["trigger_price"]


def _config_with_tp_mult(tmp_path: Path, mult: float) -> Path:
    path = tmp_path / "grid_tpmult.yaml"
    path.write_text(
        f"""
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
  max_loss_per_grid: 0.03
  max_open_levels_total: 2
  tp_spacing_mult: {mult}
""",
        encoding="utf-8",
    )
    return path


def test_tp_spacing_mult_default_is_one_grid_step(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.state.spacing = 2.0
    pos = GridPosition(
        symbol="BTCUSDT",
        side="LONG",
        level=1,
        entry_price=100.0,
        quantity=1.0,
        entry_time="2026-01-01T00:00:00Z",
    )
    # default mult = 1.0 -> TP one spacing away
    assert engine._tp_price_for_position(pos) == 100.0 + 2.0


def test_tp_spacing_mult_widens_take_profit(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config_with_tp_mult(tmp_path, 3.0),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    assert engine.cfg.tp_spacing_mult == 3.0
    engine.state.spacing = 2.0
    long_pos = GridPosition(
        symbol="BTCUSDT",
        side="LONG",
        level=1,
        entry_price=100.0,
        quantity=1.0,
        entry_time="2026-01-01T00:00:00Z",
    )
    short_pos = GridPosition(
        symbol="BTCUSDT",
        side="SHORT",
        level=1,
        entry_price=100.0,
        quantity=1.0,
        entry_time="2026-01-01T00:00:00Z",
    )
    # mult = 3.0 -> TP three spacings away (entry density unchanged)
    assert engine._tp_price_for_position(long_pos) == 100.0 + 3 * 2.0
    assert engine._tp_price_for_position(short_pos) == 100.0 - 3 * 2.0


def test_chop_grid_order_snapshots_include_protection_ids(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.state.spacing = 1.5
    engine.state.inventory.append(
        GridPosition(
            symbol="BNBUSDT",
            side="SHORT",
            level=1,
            entry_price=645.0,
            quantity=0.31,
            entry_time="2026-05-21T00:00:00Z",
            leg_id="BNBUSDT_grid_S1",
            protection_order_ids=["90489849398"],
        )
    )

    snapshots = engine.local_order_snapshots()

    protection = next(s for s in snapshots if s.exchange_order_id == "90489849398")
    assert protection.order_id == "90489849398"
    assert protection.side == "BUY"
    assert protection.symbol == "BNBUSDT"
    assert protection.quantity == 0.31


def test_chop_grid_keeps_local_only_missing_pending_orders(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    before_ids = [o.order_id for o in engine.state.pending_orders]

    engine.on_reconciliation_report(
        ReconciliationReport(
            missing_exchange_orders=[
                engine.local_order_snapshots()[0],
            ]
        )
    )

    after_ids = [o.order_id for o in engine.state.pending_orders]
    assert before_ids == after_ids


def test_chop_grid_prunes_stale_local_only_pending_after_ttl(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MLBOT_CHOP_GRID_GHOST_PENDING_TTL_S", "1800")
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    stale_snapshot = engine.local_order_snapshots()[0]
    stale_id = stale_snapshot.order_id
    # Advance the engine clock past the TTL; order never received an exchange id.
    engine.state.last_timestamp = "2026-01-01T01:00:00Z"

    engine.on_reconciliation_report(
        ReconciliationReport(missing_exchange_orders=[stale_snapshot])
    )

    assert all(o.order_id != stale_id for o in engine.state.pending_orders)


def test_chop_grid_keeps_local_only_pending_within_ttl(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MLBOT_CHOP_GRID_GHOST_PENDING_TTL_S", "1800")
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    snapshot = engine.local_order_snapshots()[0]
    keep_id = snapshot.order_id
    # Only 5 minutes elapsed -> below TTL, must be kept.
    engine.state.last_timestamp = "2026-01-01T00:05:00Z"

    engine.on_reconciliation_report(
        ReconciliationReport(missing_exchange_orders=[snapshot])
    )

    assert any(o.order_id == keep_id for o in engine.state.pending_orders)


def test_chop_grid_prunes_mapped_missing_pending_orders(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )
    target = engine.state.pending_orders[0]
    target.exchange_order_id = "ex_1"

    engine.on_reconciliation_report(
        ReconciliationReport(
            missing_exchange_orders=[
                engine.local_order_snapshots()[0],
            ]
        )
    )

    assert all(o.order_id != target.order_id for o in engine.state.pending_orders)


def test_chop_grid_prunes_missing_protection_ids(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    engine.state.inventory.append(
        GridPosition(
            symbol="BNBUSDT",
            side="SHORT",
            level=1,
            entry_price=645.0,
            quantity=0.31,
            entry_time="2026-05-21T00:00:00Z",
            leg_id="BNBUSDT_grid_S1",
            protection_order_ids=["stale_tp", "live_sl"],
        )
    )

    engine.on_reconciliation_report(
        ReconciliationReport(
            missing_exchange_orders=[
                engine.local_order_snapshots()[0],
            ]
        )
    )

    assert engine.state.inventory[0].protection_order_ids == ["live_sl"]


def test_chop_grid_records_reconciliation_issues(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )

    engine.on_reconciliation_report(
        ReconciliationReport(
            position_mismatches=[
                PositionMismatch("BTCUSDT", "LONG", 0.0, 0.01),
            ]
        )
    )

    assert engine.state.last_reconciliation_ok is False
    assert engine.state.last_reconciliation_issues == [
        "position_mismatch:BTCUSDT:LONG:0.0->0.01"
    ]


def test_chop_grid_resets_empty_active_state_before_new_entry(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        """
{
  "grid_id": "BTCUSDT_stale",
  "symbol": "BTCUSDT",
  "active": true,
  "center": 100.0,
  "spacing": 1.0,
  "realized_pnl": 0.0,
  "pending_orders": [],
  "inventory": [],
  "last_timestamp": "2026-01-01T00:00:00Z",
  "current_regime": "chop_grid"
}
""",
        encoding="utf-8",
    )
    engine = ChopGridLiveEngine(
        config_path=_config(tmp_path),
        state_path=state_path,
        level_notional=100.0,
    )

    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T02:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={"semantic_chop": 0.8, "box_prefilter": False},
    )

    assert [a["action"] for a in actions] == ["place", "place"]
    assert engine.state.active is True
    assert engine.state.grid_id == "BTCUSDT_2026-01-01T02:00:00Z"
