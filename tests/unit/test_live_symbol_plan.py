"""Tests for unified live symbol resolution (universe bus + strategy meta filter)."""

from __future__ import annotations

from src.live_data_stream.live_symbol_plan import resolve_live_classic_symbol_plan
from src.live_data_stream.universe_symbols import (
    parse_symbols_csv,
    resolve_bus_symbols,
    resolve_bus_symbols_csv,
)


def test_parse_symbols_csv_dedupes_and_uppercases():
    assert parse_symbols_csv("btcusdt, ETHUSDT ,btcusdt") == ["BTCUSDT", "ETHUSDT"]


def test_resolve_bus_symbols_from_universe():
    syms = resolve_bus_symbols(universe="highcap")
    assert "HYPEUSDT" in syms
    assert "ADAUSDT" not in syms


def test_resolve_bus_symbols_csv_ignores_env():
    csv = resolve_bus_symbols_csv(universe="highcap")
    assert "HYPEUSDT" in csv


def test_classic_plan_tpc_inherits_full_universe_when_include_empty():
    plan = resolve_live_classic_symbol_plan(
        universe="highcap",
        strategies_root="live/highcap/config/strategies",
        enabled_archetypes=["tpc"],
    )
    assert "HYPEUSDT" in plan.bus_symbols
    assert plan.strategy_symbols["tpc"] == plan.bus_symbols
    assert plan.active_union == plan.bus_symbols


def test_classic_plan_env_override_replaces_bus_set():
    plan = resolve_live_classic_symbol_plan(
        universe="highcap",
        strategies_root="live/highcap/config/strategies",
        enabled_archetypes=["tpc"],
        env_symbols="BTCUSDT,ETHUSDT",
    )
    assert plan.bus_symbols == ["BTCUSDT", "ETHUSDT"]
    assert plan.strategy_symbols["tpc"] == ["BTCUSDT", "ETHUSDT"]
