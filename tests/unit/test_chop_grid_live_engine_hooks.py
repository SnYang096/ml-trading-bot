from __future__ import annotations

from pathlib import Path

from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_reconciliation import (
    PositionMismatch,
    ReconciliationReport,
)
from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "grid.yaml"
    path.write_text(
        """
regime:
  entry_chop_min: 0.40
  exit_chop_below: 0.25
grid:
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
    assert [a["protection_type"] for a in follow_ups] == ["take_profit", "stop_loss"]
    assert all(a["action"] == "place_protection" for a in follow_ups)


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
