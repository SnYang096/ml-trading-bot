"""Jun 16 2026 live safety regressions (mock / in-memory only, no exchange).

Covers:
- late-fill infinite loop guard (winding_down → no protection)
- restart DB wipe (close_absent empty inventory + orchestrator sync gate)
- CMS leg_id _fill suffix + ghost filter (see test_multileg_position_truth)
- kill-switch halt blocks place (see test_multi_leg_kill_switch)

Postmortem: docs/architecture/live_stream/20260616_late_fill_infinite_loop_postmortem_CN.md
Design: docs/architecture/account_safety_gate_CN.md §12
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.order_management.multi_leg_orchestrator import (
    MultiLegLiveOrchestrator,
    _notify_orphan_positions,
)
from src.order_management.multi_leg_reconciliation import (
    MultiLegReconciler,
    ReconciliationPolicy,
)
from src.order_management.multi_leg_risk_governor import (
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)
from src.order_management.multi_leg_storage import MultiLegStorage


@dataclass
class _EmptyInventoryState:
    inventory: list[Any] = field(default_factory=list)


@dataclass
class _EngineEmptyInventory:
    state: _EmptyInventoryState = field(default_factory=_EmptyInventoryState)


def _minimal_orchestrator(
    *,
    engine: Any,
    storage: MultiLegStorage,
) -> MultiLegLiveOrchestrator:
    adapter = MagicMock()
    adapter.sync_open_orders.return_value = []
    adapter.sync_positions.return_value = []
    return MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(max_gross_notional=1_000.0, max_net_notional=1_000.0)
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(
            ReconciliationPolicy(client_id_prefixes={"dat_", "cg_"})
        ),
        storage=storage,
        strategy_name="trend_scalp",
        symbol="XRPUSDT",
        run_id="run_safety",
    )


def test_close_absent_positions_empty_active_does_not_wipe_db(tmp_path) -> None:
    """Jun 16: empty inventory snapshot must not close all open rows (XRP orphan)."""
    db_path = tmp_path / "multi_leg.db"
    storage = MultiLegStorage(str(db_path))
    storage.upsert_position(
        {
            "run_id": "run_1",
            "strategy": "trend_scalp",
            "leg_id": "orphan_leg",
            "symbol": "XRPUSDT",
            "side": "LONG",
            "entry_price": 1.28,
            "quantity": 5931.0,
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "local_order_id": "orphan_entry",
            "run_id": "run_1",
            "strategy": "trend_scalp",
            "symbol": "XRPUSDT",
            "side": "BUY",
            "purpose": "entry",
            "quantity": 5931.0,
            "leg_id": "orphan_leg",
            "status": "filled",
            "filled_quantity": 5931.0,
            "average_price": 1.28,
        }
    )

    changed = storage.close_absent_positions(
        strategy="trend_scalp",
        symbol="XRPUSDT",
        active_leg_ids=[],
        run_id="run_2",
    )
    assert changed == 0

    conn = sqlite3.connect(db_path)
    try:
        pos_status = conn.execute(
            "SELECT status FROM multi_leg_positions WHERE leg_id = ?",
            ("orphan_leg",),
        ).fetchone()[0]
        order_status = conn.execute(
            "SELECT status FROM multi_leg_orders WHERE local_order_id = ?",
            ("orphan_entry",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert pos_status == "open"
    assert order_status == "filled"


def test_orchestrator_skips_close_absent_before_inventory_synced(
    tmp_path,
) -> None:
    """Orchestrator must not call close_absent until first reconcile sets sync flag."""
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    storage.upsert_position(
        {
            "run_id": "run_1",
            "strategy": "trend_scalp",
            "leg_id": "live_leg",
            "symbol": "XRPUSDT",
            "side": "LONG",
            "entry_price": 1.28,
            "quantity": 100.0,
            "status": "open",
        }
    )
    orch = _minimal_orchestrator(engine=_EngineEmptyInventory(), storage=storage)
    assert orch._inventory_synced is False

    orch._persist_positions()

    conn = sqlite3.connect(storage.db_path)
    try:
        status = conn.execute(
            "SELECT status FROM multi_leg_positions WHERE leg_id = ?",
            ("live_leg",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert status == "open"


def test_orchestrator_empty_inventory_after_sync_still_no_wipe(tmp_path) -> None:
    """After sync, empty engine inventory + storage guard → DB rows stay open."""
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    storage.upsert_position(
        {
            "run_id": "run_1",
            "strategy": "trend_scalp",
            "leg_id": "live_leg",
            "symbol": "XRPUSDT",
            "side": "LONG",
            "entry_price": 1.28,
            "quantity": 100.0,
            "status": "open",
        }
    )
    orch = _minimal_orchestrator(engine=_EngineEmptyInventory(), storage=storage)
    orch._inventory_synced = True

    orch._persist_positions()

    conn = sqlite3.connect(storage.db_path)
    try:
        status = conn.execute(
            "SELECT status FROM multi_leg_positions WHERE leg_id = ?",
            ("live_leg",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert status == "open"


def test_notify_orphan_positions_sends_telegram_with_cooldown() -> None:
    """Empty engine inventory + exchange position → TG alert (Jun 16 CMS blind spot)."""
    mismatch = MagicMock(
        symbol="XRPUSDT",
        side="LONG",
        local_quantity=0.0,
        exchange_quantity=5931.0,
    )

    with patch(
        "src.order_management.multi_leg_orchestrator.send_telegram_message"
    ) as send:
        _notify_orphan_positions(
            strategy="trend_scalp",
            symbol="XRPUSDT",
            mismatches=[mismatch],
        )

    send.assert_called_once()
    (message,) = send.call_args[0]
    assert "孤儿仓位" in message
    assert "XRPUSDT" in message
    assert "5931" in message
    assert send.call_args.kwargs["stamp_key"] == "hedge:orphan:XRPUSDT"
    assert send.call_args.kwargs["cooldown_sec"] == 900
