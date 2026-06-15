"""TrendPositionTruthSync 单元测试

覆盖:
  1. project_to_sqlite 正常写入 Position record
  2. dedup merge：同 symbol+side 已有 OPEN 行时 reuse
  3. status=CLOSED 时正确 update
  4. storage_factory=None 时 graceful fallback
  5. project_position_object 便捷方法
  6. project_position_object CLOSED 透传 exit_price/exit_reason
  7. P2: bootstrap_position_from_exchange 保守默认
  8. P2: on_restart 统一恢复入口
  9. P3: periodic_reconcile 各场景
 10. P4: list_open_projections 只读投影
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


# ─── P2: bootstrap_position_from_exchange ─────────────────────────────────


def test_bootstrap_position_from_exchange_conservative():
    """无 execution.yaml → 保守默认 (1% ATR, 1.5R stop, 3R target)。"""
    pid, pos = TrendPositionTruthSync.bootstrap_position_from_exchange(
        symbol="ETHUSDT",
        side="long",
        entry_price=2000.0,
        qty=1.5,
    )
    assert pid.startswith("ETH:live_")
    assert pos["side"] == "LONG"
    assert pos["entry_price"] == 2000.0
    assert pos["qty"] == 1.5
    assert pos["stop_loss_price"] is not None
    assert pos["take_profit_price"] is not None
    assert pos["_bootstrap_from_exchange"] is True
    # Conservative: 1.5R stop = 1.5 * 1% * 2000 = 30
    assert abs(pos["stop_loss_price"] - 1970.0) < 1.0


def test_bootstrap_position_from_exchange_short_side():
    """SHORT side: SL 在上方，TP 在下方。"""
    pid, pos = TrendPositionTruthSync.bootstrap_position_from_exchange(
        symbol="BTCUSDT",
        side="short",
        entry_price=60000.0,
        qty=0.1,
    )
    assert pos["side"] == "SHORT"
    assert pos["stop_loss_price"] > 60000.0  # SL above for short
    assert pos["take_profit_price"] < 60000.0  # TP below for short


def test_bootstrap_position_from_exchange_invalid_raises():
    """无效参数 → ValueError。"""
    with pytest.raises(ValueError):
        TrendPositionTruthSync.bootstrap_position_from_exchange(
            symbol="",
            side="long",
            entry_price=100.0,
            qty=1.0,
        )
    with pytest.raises(ValueError):
        TrendPositionTruthSync.bootstrap_position_from_exchange(
            symbol="BTC",
            side="long",
            entry_price=0.0,
            qty=1.0,
        )


def test_bootstrap_position_from_exchange_writes_json(tmp_path):
    """state_path 指定时写入 JSON 文件。"""
    state_path = tmp_path / "BTCUSDT.json"
    pid, pos = TrendPositionTruthSync.bootstrap_position_from_exchange(
        symbol="BTCUSDT",
        side="long",
        entry_price=50000.0,
        qty=0.5,
        state_path=state_path,
    )
    assert state_path.exists()
    import json

    data = json.loads(state_path.read_text())
    assert data["symbol"] == "BTCUSDT"
    assert data["_bootstrap_from_exchange"] is True
    assert pid in data["positions"]


def test_make_pid_format():
    """pid 格式: {BASE}:live_{uuid12}。"""
    pid = TrendPositionTruthSync._make_pid("ETHUSDT")
    assert pid.startswith("ETH:live_")
    assert len(pid.split("live_")[1]) == 12


# ─── P2: on_restart ───────────────────────────────────────────────────────


def test_on_restart_bootstrap_from_exchange():
    """tracker 空 + 交易所有仓 → bootstrap。"""
    storage = _make_storage()
    tts = TrendPositionTruthSync(symbol="ETHUSDT", storage_factory=lambda: storage)
    api = MagicMock()
    api.get_positions.return_value = [
        {"symbol": "ETH/USDT:USDT", "side": "long", "size": 2.0, "entry_price": 2100.0},
    ]
    tracker = MagicMock()
    tracker.__len__ = Mock(return_value=0)
    tracker._positions = {}
    tracker.restore_from_disk.return_value = 0

    report = tts.on_restart(
        api=api,
        tracker=tracker,
        state_path=MagicMock(),
    )
    assert report["bootstrapped"] == 1
    assert len(tracker._positions) == 1


def test_on_restart_bootstrap_missing_side_only():
    """tracker 有 long + exchange 有 long+short → 只 bootstrap short。"""
    storage = _make_storage()
    tts = TrendPositionTruthSync(symbol="ETHUSDT", storage_factory=lambda: storage)
    api = MagicMock()
    api.get_positions.return_value = [
        {"symbol": "ETH/USDT:USDT", "side": "long", "size": 2.0, "entry_price": 2100.0},
        {
            "symbol": "ETH/USDT:USDT",
            "side": "short",
            "size": 1.0,
            "entry_price": 2150.0,
        },
    ]
    tracker = MagicMock()
    tracker._positions = {
        "ETH:001": {"side": "LONG", "qty": 2.0, "entry_price": 2100.0},
    }
    tracker.restore_from_disk.return_value = 1

    report = tts.on_restart(
        api=api,
        tracker=tracker,
        state_path=MagicMock(),
    )
    assert report["bootstrapped"] == 1
    assert len(tracker._positions) == 2
    sides = {str(p.get("side", "")).upper() for p in tracker._positions.values()}
    assert sides == {"LONG", "SHORT"}
    api.get_positions.assert_called_once()


def test_on_restart_force_exchange_clears_stale_memory():
    """force_exchange 清空旧 tracker 内存后再 bootstrap。"""
    storage = _make_storage()
    tts = TrendPositionTruthSync(symbol="ETHUSDT", storage_factory=lambda: storage)
    api = MagicMock()
    api.get_positions.return_value = [
        {"symbol": "ETH/USDT:USDT", "side": "long", "size": 1.5, "entry_price": 2100.0},
    ]
    tracker = MagicMock()
    tracker._positions = {
        "ETH:stale": {"side": "LONG", "qty": 99.0},
    }

    report = tts.on_restart(
        api=api,
        tracker=tracker,
        state_path=MagicMock(),
        force_exchange=True,
    )
    assert report["bootstrapped"] == 1
    assert "ETH:stale" not in tracker._positions
    assert len(tracker._positions) == 1
    assert tracker._positions[next(iter(tracker._positions))]["qty"] == 1.5
    tracker.restore_from_disk.assert_not_called()


# ─── P3: periodic_reconcile ──────────────────────────────────────────────


def test_periodic_reconcile_no_issues():
    """exchange 和 tracker 一致 → 无 issue。"""
    storage = _make_storage()
    storage.get_open_positions.return_value = [
        Position(
            position_id="ETH:001",
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            entry_time=datetime.now(timezone.utc),
            entry_price=2000.0,
            current_size=2.0,
            initial_size=2.0,
            status=PositionStatus.OPEN,
        ),
    ]
    tts = TrendPositionTruthSync(symbol="ETHUSDT", storage_factory=lambda: storage)
    api = MagicMock()
    api.get_positions.return_value = [
        {"symbol": "ETH/USDT:USDT", "side": "long", "size": 2.0, "entry_price": 2000.0},
    ]
    tracker = MagicMock()
    tracker._positions = {
        "ETH:001": {"side": "LONG", "qty": 2.0},
    }

    issues = tts.periodic_reconcile(api=api, tracker=tracker)
    assert issues == {}


def test_periodic_reconcile_orphan_open():
    """exchange flat + SQLite open → sqlite_orphan_open + auto-close。"""
    storage = _make_storage()
    orphan = Position(
        position_id="ETH:ORPHAN",
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        entry_time=datetime.now(timezone.utc),
        entry_price=2000.0,
        current_size=1.0,
        initial_size=1.0,
        status=PositionStatus.OPEN,
    )
    storage.get_open_positions.return_value = [orphan]
    storage.get_position.return_value = orphan
    tts = TrendPositionTruthSync(symbol="ETHUSDT", storage_factory=lambda: storage)
    api = MagicMock()
    api.get_positions.return_value = []  # exchange flat
    tracker = MagicMock()
    tracker._positions = {}

    issues = tts.periodic_reconcile(api=api, tracker=tracker)
    assert issues.get("sqlite_orphan_open", 0) >= 1
    # Auto-close should call update_position with CLOSED
    storage.update_position.assert_called()


def test_periodic_reconcile_exchange_api_error():
    """exchange 查询失败 → api_error。"""
    tts = TrendPositionTruthSync(
        symbol="ETHUSDT", storage_factory=lambda: _make_storage()
    )
    api = MagicMock()
    api.get_positions.side_effect = RuntimeError("timeout")
    tracker = MagicMock()

    issues = tts.periodic_reconcile(api=api, tracker=tracker)
    assert issues.get("api_error", 0) >= 1


def test_periodic_reconcile_tracker_missing():
    """exchange 有仓但 tracker 空 → bootstrap_from_exchange。"""
    storage = _make_storage()
    storage.get_open_positions.return_value = []
    tts = TrendPositionTruthSync(symbol="ETHUSDT", storage_factory=lambda: storage)
    api = MagicMock()
    api.get_positions.return_value = [
        {"symbol": "ETH/USDT:USDT", "side": "long", "size": 1.0, "entry_price": 2000.0},
    ]
    tracker = MagicMock()
    tracker._positions = {}  # tracker empty

    issues = tts.periodic_reconcile(api=api, tracker=tracker)
    assert issues.get("bootstrap_from_exchange", 0) >= 1


# ─── P4: list_open_projections ────────────────────────────────────────────


def test_list_open_projections_basic():
    """正常读取 open positions。"""
    storage = _make_storage()
    storage.get_open_positions.return_value = [
        Position(
            position_id="ETH:001",
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            entry_time=datetime(2026, 6, 14, tzinfo=timezone.utc),
            entry_price=2000.0,
            current_size=2.0,
            initial_size=2.0,
            status=PositionStatus.OPEN,
            archetype="tpc",
        ),
    ]

    rows = TrendPositionTruthSync.list_open_projections(storage, symbol="ETHUSDT")
    assert len(rows) == 1
    assert rows[0]["position_id"] == "ETH:001"
    assert rows[0]["symbol"] == "ETHUSDT"
    assert rows[0]["side"] == "long"
    assert rows[0]["quantity"] == 2.0
    assert rows[0]["scope"] == "trend"


def test_list_open_projections_empty_storage():
    """storage=None → 空列表。"""
    rows = TrendPositionTruthSync.list_open_projections(None)
    assert rows == []


def test_list_open_projections_symbol_filter():
    """symbol 过滤生效。"""
    storage = _make_storage()
    storage.get_open_positions.return_value = []

    TrendPositionTruthSync.list_open_projections(storage, symbol="BTCUSDT")
    storage.get_open_positions.assert_called_with("BTCUSDT")
