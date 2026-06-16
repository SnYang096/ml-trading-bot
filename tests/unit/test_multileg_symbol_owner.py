"""Unit tests for multileg symbol-owner helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
from unittest.mock import MagicMock

from src.order_management.multileg_symbol_owner import (
    filter_places_for_owner,
    refresh_symbol_owner,
    runtime_holds_symbol_engine,
)


@dataclass
class _Rt:
    name: str
    symbol: str
    engine: Any


def test_runtime_holds_symbol_engine_grid_slot():
    eng = MagicMock()
    eng.holds_real_grid_slot.return_value = True
    eng.local_position_snapshots.return_value = []
    assert runtime_holds_symbol_engine(eng) is True


def test_runtime_holds_symbol_engine_inventory_only():
    eng = MagicMock()
    eng.holds_real_grid_slot.return_value = False
    eng.local_position_snapshots.return_value = [{"side": "LONG", "quantity": 1.0}]
    assert runtime_holds_symbol_engine(eng) is True


def test_refresh_symbol_owner_chop_before_trend():
    chop = MagicMock()
    chop.holds_real_grid_slot.return_value = True
    chop.local_position_snapshots.return_value = []
    trend = MagicMock()
    trend.holds_real_grid_slot.return_value = False
    trend.local_position_snapshots.return_value = []
    runtimes = [
        _Rt("chop_grid", "BTCUSDT", chop),
        _Rt("trend_scalp", "BTCUSDT", trend),
    ]
    owner: dict = {}
    refresh_symbol_owner(runtimes, owner, "BTCUSDT")
    assert owner["BTCUSDT"] == "chop_grid"


def test_filter_places_for_owner_drops_foreign_places():
    actions = [
        {"action": "place", "side": "BUY", "quantity": 1.0},
        {"action": "market_exit", "side": "LONG", "quantity": 1.0},
    ]
    kept, dropped = filter_places_for_owner(
        actions, owner="chop_grid", runtime_name="trend_scalp"
    )
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["action"] == "market_exit"


def test_filter_places_for_owner_allows_owner_places():
    actions = [{"action": "place", "side": "BUY", "quantity": 1.0}]
    kept, dropped = filter_places_for_owner(
        actions, owner="trend_scalp", runtime_name="trend_scalp"
    )
    assert dropped == 0
    assert len(kept) == 1
