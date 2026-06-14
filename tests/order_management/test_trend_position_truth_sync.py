"""TrendPositionTruthSync 单元测试

覆盖:
  1. project_to_sqlite 正常写入 Position record
  2. dedup merge：同 symbol+side 已有 OPEN 行时 reuse
  3. status=CLOSED 时正确 update
  4. storage_factory=None 时 graceful fallback
  5. project_position_object 便捷方法
  6. project_position_object CLOSED 透传 exit_price/exit_reason
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

import pytest

from src.order_management.models import Position, PositionSide, PositionStatus
from src.order_management.trend_position_truth_sync import TrendPositionTruthSync


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_storage():
    """创建 mock Storage，关键方法可追踪。"""
    storage = MagicMock()
    storage.get_position.return_value = None
    storage.get_open_positions.return_value = []
    storage.create_position.return_value = True
    storage.update_position.return_value = True
    return storage


def _make_tts(symbol="BTCUSDT", storage=None):
    """创建 TTS 实例，注入 mock storage。"""
    if storage is None:
        storage = _make_storage()
    return (
        TrendPositionTruthSync(
            symbol=symbol,
            storage_factory=lambda: storage,
        ),
        storage,
    )


def _pos_dict(
    side="LONG",
    entry_price=50000.0,
    qty=0.5,
    entry_time=None,
    archetype="tpc",
    stop_loss_price=48000.0,
    take_profit_price=55000.0,
    notes=None,
    unrealized_pnl=None,
    realized_pnl=None,
    initial_size=None,
):
    return {
        "side": side,
        "entry_price": entry_price,
        "qty": qty,
        "entry_time": entry_time or datetime.now(timezone.utc),
        "archetype": archetype,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "notes": notes,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "initial_size": initial_size,
    }


# ─── tests ──────────────────────────────────────────────────────────────────


def test_project_to_sqlite_creates_new_position():
    """正常写入: 不存在同 position_id → create_position。"""
    tts, storage = _make_tts()
    pid = "BTC:001"

    result = tts.project_to_sqlite(pid, _pos_dict())

    assert result == pid
    storage.create_position.assert_called_once()
    record = storage.create_position.call_args[0][0]
    assert record.position_id == pid
    assert record.symbol == "BTCUSDT"
    assert record.side == PositionSide.LONG
    assert record.entry_price == 50000.0
    assert record.current_size == 0.5
    assert record.status == PositionStatus.OPEN


def test_project_to_sqlite_dedup_merge_reuses_existing():
    """dedup merge: 同 symbol+side 已有 OPEN 行 → reuse 而非 create new。"""
    existing = Position(
        position_id="BTC:EXISTING",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(timezone.utc),
        entry_price=49000.0,
        current_size=0.3,
        initial_size=0.3,
        status=PositionStatus.OPEN,
    )
    storage = _make_storage()
    storage.get_position.return_value = None  # new pid not found
    storage.get_open_positions.return_value = [existing]
    tts = TrendPositionTruthSync(symbol="BTCUSDT", storage_factory=lambda: storage)

    result = tts.project_to_sqlite("BTC:NEW", _pos_dict())

    # dedup merge → reuse existing pid
    assert result == "BTC:EXISTING"
    # should NOT create, should UPDATE the existing record
    storage.create_position.assert_not_called()
    storage.update_position.assert_called_once()
    record = storage.update_position.call_args[0][0]
    assert record.position_id == "BTC:EXISTING"


def test_project_to_sqlite_closed_status_updates():
    """status=CLOSED 时正确 update (exit_time, exit_price, exit_reason)。"""
    existing = Position(
        position_id="BTC:001",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(timezone.utc),
        entry_price=50000.0,
        current_size=0.5,
        initial_size=0.5,
        status=PositionStatus.OPEN,
    )
    storage = _make_storage()
    storage.get_position.return_value = existing
    tts = TrendPositionTruthSync(symbol="BTCUSDT", storage_factory=lambda: storage)

    result = tts.project_to_sqlite(
        "BTC:001",
        _pos_dict(),
        status=PositionStatus.CLOSED,
        exit_price=53000.0,
        exit_reason="stop_loss",
    )

    assert result == "BTC:001"
    storage.update_position.assert_called_once()
    record = storage.update_position.call_args[0][0]
    assert record.status == PositionStatus.CLOSED
    assert record.exit_price == 53000.0
    assert record.exit_reason == "stop_loss"
    assert record.current_size == 0.0
    assert record.exit_time is not None


def test_project_to_sqlite_storage_none_fallback():
    """storage_factory 返回 None → 跳过写入，返回原始 position_id。"""
    tts = TrendPositionTruthSync(
        symbol="BTCUSDT",
        storage_factory=lambda: None,
    )
    pid = "BTC:001"

    result = tts.project_to_sqlite(pid, _pos_dict())

    assert result == pid


def test_project_to_sqlite_storage_factory_exception():
    """storage_factory 抛异常 → 视为 None，graceful fallback。"""
    tts = TrendPositionTruthSync(
        symbol="BTCUSDT",
        storage_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    pid = "BTC:001"

    result = tts.project_to_sqlite(pid, _pos_dict())

    assert result == pid


def test_project_position_object_basic():
    """project_position_object 便捷方法: 正常写入。"""
    tts, storage = _make_tts()
    pos = Position(
        position_id="BTC:001",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(timezone.utc),
        entry_price=50000.0,
        initial_size=0.5,
        current_size=0.5,
        total_cost=25000.0,
        status=PositionStatus.OPEN,
        archetype="tpc",
        notes="exchange_sync",
    )

    result = tts.project_position_object(pos)

    assert result == "BTC:001"
    storage.create_position.assert_called_once()
    record = storage.create_position.call_args[0][0]
    assert record.notes == "exchange_sync"


def test_project_position_object_closed_with_exit_fields():
    """project_position_object CLOSED 透传 exit_price/exit_reason。"""
    existing = Position(
        position_id="BTC:001",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(timezone.utc),
        entry_price=50000.0,
        current_size=0.5,
        initial_size=0.5,
        status=PositionStatus.CLOSED,
        exit_price=53000.0,
        exit_reason="exchange_sync_flat",
        realized_pnl=1500.0,
    )
    storage = _make_storage()
    storage.get_position.return_value = existing
    tts = TrendPositionTruthSync(symbol="BTCUSDT", storage_factory=lambda: storage)

    result = tts.project_position_object(
        existing,
        status=PositionStatus.CLOSED,
        exit_price=53000.0,
        exit_reason="exchange_sync_flat",
    )

    assert result == "BTC:001"
    storage.update_position.assert_called_once()
    record = storage.update_position.call_args[0][0]
    assert record.status == PositionStatus.CLOSED
    assert record.exit_price == 53000.0
    assert record.exit_reason == "exchange_sync_flat"
    assert record.current_size == 0.0
    assert record.realized_pnl == 1500.0


def test_project_to_sqlite_qty_zero_open_skips():
    """OPEN status + qty=0 → 跳过写入（无效 OPEN）。"""
    tts, storage = _make_tts()

    result = tts.project_to_sqlite("BTC:001", _pos_dict(qty=0.0))

    assert result == "BTC:001"
    storage.create_position.assert_not_called()
    storage.update_position.assert_not_called()


def test_project_to_sqlite_initial_size_from_pos_dict():
    """pos dict 带 initial_size → TTS 优先使用。"""
    tts, storage = _make_tts()
    pos = _pos_dict(qty=0.3, initial_size=0.5)  # resize-down: initial > current

    result = tts.project_to_sqlite("BTC:001", pos)

    assert result == "BTC:001"
    record = storage.create_position.call_args[0][0]
    assert record.initial_size == 0.5
    assert record.current_size == 0.3


def test_project_to_sqlite_side_short():
    """SHORT side 正确处理。"""
    tts, storage = _make_tts()

    result = tts.project_to_sqlite("BTC:002", _pos_dict(side="SHORT"))

    assert result == "BTC:002"
    record = storage.create_position.call_args[0][0]
    assert record.side == PositionSide.SHORT
