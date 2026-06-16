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
    _notify_phantom_positions,
)
from src.order_management.multi_leg_reconciliation import LocalPositionSnapshot
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


# ── Plan A: phantom positions (engine/DB say open, exchange flat) ────────────


@dataclass
class _Leg:
    leg_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float = 1.0
    protection_order_ids: list = field(default_factory=list)


class _InventoryState:
    def __init__(self, legs: list[_Leg]) -> None:
        self.inventory = list(legs)


class _PhantomEngine:
    """Minimal engine exposing inventory + remove_inventory_legs hook."""

    def __init__(self, legs: list[_Leg]) -> None:
        self.state = _InventoryState(legs)

    def local_position_snapshots(self) -> list[LocalPositionSnapshot]:
        return [
            LocalPositionSnapshot(p.symbol, p.side, p.quantity)
            for p in self.state.inventory
        ]

    def remove_inventory_legs(self, leg_ids) -> int:
        targets = {str(x) for x in leg_ids}
        before = len(self.state.inventory)
        self.state.inventory = [
            p for p in self.state.inventory if p.leg_id not in targets
        ]
        return before - len(self.state.inventory)


def _seed_open_leg(storage: MultiLegStorage, leg_id: str = "dat_1_fill0") -> None:
    storage.upsert_position(
        {
            "run_id": "run_safety",
            "strategy": "trend_scalp",
            "leg_id": leg_id,
            "symbol": "XRPUSDT",
            "side": "LONG",
            "entry_price": 1.28,
            "quantity": 5931.0,
            "status": "open",
        }
    )


def _pos_status(storage: MultiLegStorage, leg_id: str) -> str:
    conn = sqlite3.connect(storage.db_path)
    try:
        return conn.execute(
            "SELECT status FROM multi_leg_positions WHERE leg_id = ?",
            (leg_id,),
        ).fetchone()[0]
    finally:
        conn.close()


def test_close_positions_by_leg_ids_closes_even_when_all_phantom(tmp_path) -> None:
    """Explicit allow-list close works even when engine inventory is empty."""
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)

    changed = storage.close_positions_by_leg_ids(
        strategy="trend_scalp",
        symbol="XRPUSDT",
        leg_ids=["dat_1_fill0"],
        reason="exchange_sync_phantom",
        run_id="run_safety",
    )
    assert changed == 1
    assert _pos_status(storage, "dat_1_fill0") == "closed"

    conn = sqlite3.connect(storage.db_path)
    try:
        raw = conn.execute(
            "SELECT raw_json FROM multi_leg_positions WHERE leg_id = ?",
            ("dat_1_fill0",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert "exchange_sync_phantom" in (raw or "")


def test_phantom_positions_cleaned_after_confirmation(tmp_path, monkeypatch) -> None:
    """Engine holds a leg the exchange reports flat → cleaned after confirm cycles."""
    monkeypatch.setenv("MLBOT_MULTI_LEG_PHANTOM_CONFIRM_CYCLES", "2")
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)
    engine = _PhantomEngine([_Leg("dat_1_fill0", "XRPUSDT", "LONG", 5931.0)])
    orch = _minimal_orchestrator(engine=engine, storage=storage)
    orch._inventory_synced = True

    with patch("src.order_management.multi_leg_orchestrator.send_telegram_message"):
        # Cycle 1: observed phantom once (threshold 2) → no cleanup yet.
        orch.reconcile(exchange_orders=[], exchange_positions=[])
        assert len(engine.state.inventory) == 1
        assert _pos_status(storage, "dat_1_fill0") == "open"

        # Cycle 2: confirmed → engine leg dropped + DB row closed.
        orch.reconcile(exchange_orders=[], exchange_positions=[])

    assert engine.state.inventory == []
    assert _pos_status(storage, "dat_1_fill0") == "closed"


def test_phantom_cleanup_skipped_before_inventory_synced(tmp_path) -> None:
    """No cleanup until the first exchange sync has happened."""
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)
    engine = _PhantomEngine([_Leg("dat_1_fill0", "XRPUSDT", "LONG", 5931.0)])
    orch = _minimal_orchestrator(engine=engine, storage=storage)
    assert orch._inventory_synced is False

    cleaned = orch._sync_phantom_positions([])
    assert cleaned == []
    assert len(engine.state.inventory) == 1
    assert _pos_status(storage, "dat_1_fill0") == "open"


def test_phantom_not_cleaned_when_exchange_has_position(tmp_path, monkeypatch) -> None:
    """A leg the exchange still holds must never be treated as phantom."""
    monkeypatch.setenv("MLBOT_MULTI_LEG_PHANTOM_CONFIRM_CYCLES", "1")
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)
    engine = _PhantomEngine([_Leg("dat_1_fill0", "XRPUSDT", "LONG", 5931.0)])
    orch = _minimal_orchestrator(engine=engine, storage=storage)
    orch._inventory_synced = True

    exchange_positions = [
        {"symbol": "XRPUSDT", "position_side": "LONG", "position_amount": 5931.0}
    ]
    with patch("src.order_management.multi_leg_orchestrator.send_telegram_message"):
        cleaned = orch._sync_phantom_positions(exchange_positions)

    assert cleaned == []
    assert len(engine.state.inventory) == 1
    assert _pos_status(storage, "dat_1_fill0") == "open"


def test_phantom_confirmation_resets_when_position_reappears(
    tmp_path, monkeypatch
) -> None:
    """A single empty snapshot must not wipe; reappearing position resets counter."""
    monkeypatch.setenv("MLBOT_MULTI_LEG_PHANTOM_CONFIRM_CYCLES", "2")
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)
    engine = _PhantomEngine([_Leg("dat_1_fill0", "XRPUSDT", "LONG", 5931.0)])
    orch = _minimal_orchestrator(engine=engine, storage=storage)
    orch._inventory_synced = True
    present = [
        {"symbol": "XRPUSDT", "position_side": "LONG", "position_amount": 5931.0}
    ]

    with patch("src.order_management.multi_leg_orchestrator.send_telegram_message"):
        orch._sync_phantom_positions([])  # transient empty (count=1)
        orch._sync_phantom_positions(present)  # position back → counter reset
        cleaned = orch._sync_phantom_positions([])  # count back to 1, not cleaned

    assert cleaned == []
    assert len(engine.state.inventory) == 1
    assert _pos_status(storage, "dat_1_fill0") == "open"


def test_phantom_sudden_exchange_collapse_skips_one_cycle(
    tmp_path, monkeypatch
) -> None:
    """N>0 → 0 in one step is a suspect API glitch: skip without advancing confirm."""
    monkeypatch.setenv("MLBOT_MULTI_LEG_PHANTOM_CONFIRM_CYCLES", "1")
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)
    engine = _PhantomEngine([_Leg("dat_1_fill0", "XRPUSDT", "LONG", 5931.0)])
    orch = _minimal_orchestrator(engine=engine, storage=storage)
    orch._inventory_synced = True
    present = [
        {"symbol": "XRPUSDT", "position_side": "LONG", "position_amount": 5931.0}
    ]

    with patch("src.order_management.multi_leg_orchestrator.send_telegram_message"):
        orch._sync_phantom_positions(present)  # observe 1 position
        # Sudden collapse to empty → skipped this cycle even at threshold 1.
        cleaned_glitch = orch._sync_phantom_positions([])
        assert cleaned_glitch == []
        assert len(engine.state.inventory) == 1
        assert _pos_status(storage, "dat_1_fill0") == "open"
        # Next empty cycle is no longer a sudden drop (prev=0) → cleaned.
        cleaned_confirmed = orch._sync_phantom_positions([])

    assert cleaned_confirmed == ["dat_1_fill0"]
    assert engine.state.inventory == []
    assert _pos_status(storage, "dat_1_fill0") == "closed"


def test_notify_phantom_positions_sends_telegram_with_cooldown() -> None:
    with patch(
        "src.order_management.multi_leg_orchestrator.send_telegram_message"
    ) as send:
        _notify_phantom_positions(
            strategy="trend_scalp",
            symbol="XRPUSDT",
            leg_ids=["dat_1_fill0"],
        )

    send.assert_called_once()
    (message,) = send.call_args[0]
    assert "幻影仓位" in message
    assert "dat_1_fill0" in message
    assert send.call_args.kwargs["stamp_key"] == "hedge:phantom:XRPUSDT"
    assert send.call_args.kwargs["cooldown_sec"] == 900


def test_phantom_cleanup_cancels_exchange_protection_orders(
    tmp_path, monkeypatch
) -> None:
    """Phantom legs' SL/TP orders must be cancelled on the exchange."""
    monkeypatch.setenv("MLBOT_MULTI_LEG_PHANTOM_CONFIRM_CYCLES", "1")
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)
    leg = _Leg(
        "dat_1_fill0",
        "XRPUSDT",
        "LONG",
        5931.0,
        protection_order_ids=["sl_123", "tp_456"],
    )
    engine = _PhantomEngine([leg])
    orch = _minimal_orchestrator(engine=engine, storage=storage)
    orch._inventory_synced = True

    with patch("src.order_management.multi_leg_orchestrator.send_telegram_message"):
        orch.reconcile(exchange_orders=[], exchange_positions=[])

    # Engine leg should be removed.
    assert engine.state.inventory == []
    # Adapter should have received cancel_protection actions.
    adapter = orch.adapter
    cancel_calls = [
        c
        for c in adapter.execute_actions.call_args_list
        if any(
            a.get("action") == "cancel_protection" for a in (c[0][0] if c[0] else [])
        )
    ]
    assert len(cancel_calls) >= 1
    cancelled_ids = {
        a["order_id"]
        for call in cancel_calls
        for a in call[0][0]
        if a.get("action") == "cancel_protection"
    }
    assert cancelled_ids == {"sl_123", "tp_456"}


def test_phantom_cleanup_skips_cancel_when_no_protection_orders(
    tmp_path, monkeypatch
) -> None:
    """Phantom legs without protection orders must not crash."""
    monkeypatch.setenv("MLBOT_MULTI_LEG_PHANTOM_CONFIRM_CYCLES", "1")
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    _seed_open_leg(storage)
    leg = _Leg("dat_1_fill0", "XRPUSDT", "LONG", 5931.0)  # no protection ids
    engine = _PhantomEngine([leg])
    orch = _minimal_orchestrator(engine=engine, storage=storage)
    orch._inventory_synced = True

    with patch("src.order_management.multi_leg_orchestrator.send_telegram_message"):
        orch.reconcile(exchange_orders=[], exchange_positions=[])

    assert engine.state.inventory == []
    assert _pos_status(storage, "dat_1_fill0") == "closed"
