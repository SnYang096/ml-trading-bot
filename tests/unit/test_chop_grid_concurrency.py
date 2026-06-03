from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.order_management.chop_grid_concurrency import ChopGridConcurrencyGate
from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine

_ENTER_FEATURES = {
    "bpc_semantic_chop": 0.8,
    "box_pos_60": 0.50,
}


def _grid_config(tmp_path: Path, name: str) -> Path:
    cfg = tmp_path / f"{name}.yaml"
    cfg.write_text(
        """
regime:
  entry_chop_min: 0.50
  exit_chop_below: 0.32
  exclude_box_prefilter: true
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 1
risk:
  fee_bps: 4.0
  max_open_levels_total: 2
""",
        encoding="utf-8",
    )
    return cfg


class _FakeState:
    def __init__(self, active: bool) -> None:
        self.active = active


class _FakeEngine:
    def __init__(self, active: bool) -> None:
        self.state = _FakeState(active)


def test_chop_grid_concurrency_gate_blocks_fourth_symbol() -> None:
    gate = ChopGridConcurrencyGate(3)
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        gate.register(sym, _FakeEngine(True))
    gate.register("BNBUSDT", _FakeEngine(False))
    assert gate.allow_new_segment("BNBUSDT") is False
    assert gate.allow_new_segment("BTCUSDT") is True


def test_gate_blocks_second_real_engine_on_bar(tmp_path: Path) -> None:
    """Integration: a shared gate (cap=1) blocks the 2nd symbol's _start_grid."""
    gate = ChopGridConcurrencyGate(1)
    eng_a = ChopGridLiveEngine(
        config_path=_grid_config(tmp_path, "a"),
        state_path=tmp_path / "state_a.json",
        level_notional=100.0,
    )
    eng_b = ChopGridLiveEngine(
        config_path=_grid_config(tmp_path, "b"),
        state_path=tmp_path / "state_b.json",
        level_notional=100.0,
    )
    gate.register("BTCUSDT", eng_a)
    gate.register("ETHUSDT", eng_b)

    actions_a = eng_a.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features=dict(_ENTER_FEATURES),
    )
    assert any(a.get("action") == "place" for a in actions_a)
    assert eng_a.state.active is True

    # Second symbol wants to enter but the gate is full (1 active).
    actions_b = eng_b.on_bar(
        symbol="ETHUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=50.0,
        low=50.0,
        close=50.0,
        atr=1.0,
        features=dict(_ENTER_FEATURES),
    )
    assert not any(a.get("action") == "place" for a in actions_b)
    assert eng_b.state.active is False


def test_apply_multi_leg_segment_dd_sizing(tmp_path, monkeypatch) -> None:
    from src.live_data_stream.constitution_config import (
        apply_multi_leg_args_from_constitution,
    )

    y = tmp_path / "constitution.yaml"
    y.write_text(
        """
multi_leg:
  strategies: chop_grid
  sizing:
    chop_grid:
      segment_dd_target: 0.01
      max_loss_per_grid: 0.03
      max_levels_per_side: 3
  account:
    equity_usdt: 10000
  risk_limits:
    max_concurrent_grid_symbols: 3
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MLBOT_CONSTITUTION_YAML", str(y))
    monkeypatch.setenv("MLBOT_STRATEGIES_ROOT", str(tmp_path / "strategies"))
    (tmp_path / "strategies").mkdir()

    args = argparse.Namespace(
        strategies="chop_grid",
        unit_notional=100.0,
        max_gross_notional=2000.0,
        max_net_notional=1000.0,
        max_symbol_gross_notional=800.0,
        max_symbol_net_notional=400.0,
        max_resting_orders=60,
        account_equity_usdt=10000.0,
        max_drawdown_pct=0.12,
        constitution_yaml="",
        max_concurrent_grid_symbols=0,
    )
    apply_multi_leg_args_from_constitution(args)
    assert args.unit_notional == pytest.approx(10000.0 * 0.01 / (0.03 * 6))
    assert args.unit_notional_by_strategy["chop_grid"] == pytest.approx(
        10000.0 * 0.01 / (0.03 * 6)
    )
    assert args.max_concurrent_grid_symbols == 3
