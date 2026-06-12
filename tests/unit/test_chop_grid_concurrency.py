from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.order_management.chop_grid_concurrency import MultiLegConcurrencyGate
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
    def __init__(self, active: bool, holds: bool | None = None) -> None:
        self.state = _FakeState(active)
        self._holds = holds

    def holds_real_grid_slot(self) -> bool:
        if self._holds is None:
            return self.state.active
        return self._holds


def test_chop_grid_concurrency_gate_blocks_fourth_symbol() -> None:
    gate = MultiLegConcurrencyGate(3)
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        gate.register(sym, _FakeEngine(True))
    gate.register("BNBUSDT", _FakeEngine(False))
    assert gate.allow_new_segment("BNBUSDT") is False
    assert gate.allow_new_segment("BTCUSDT") is True


def test_ghost_active_engine_does_not_occupy_a_slot() -> None:
    """An active-but-empty (ghost) segment must not block a new symbol."""
    gate = MultiLegConcurrencyGate(3)
    gate.register("BTCUSDT", _FakeEngine(True, holds=True))
    gate.register("SOLUSDT", _FakeEngine(True, holds=True))
    # active=True but nothing real -> ghost, should not count toward the cap.
    gate.register("XRPUSDT", _FakeEngine(True, holds=False))
    gate.register("ETHUSDT", _FakeEngine(False))

    assert gate.allow_new_segment("ETHUSDT") is True


def test_gate_purges_ghost_active_before_slot_count(tmp_path: Path) -> None:
    """Shared gate clears stale active on other symbols before cap check."""
    gate = MultiLegConcurrencyGate(3)
    eng_btc = ChopGridLiveEngine(
        config_path=_grid_config(tmp_path, "btc"),
        state_path=tmp_path / "state_btc.json",
        level_notional=100.0,
    )
    eng_eth = ChopGridLiveEngine(
        config_path=_grid_config(tmp_path, "eth"),
        state_path=tmp_path / "state_eth.json",
        level_notional=100.0,
    )
    eng_btc.state.symbol = "BTCUSDT"
    eng_btc.state.active = True
    eng_btc.state.grid_id = "BTCUSDT_ghost"
    eng_btc.save_state()

    gate.register("BTCUSDT", eng_btc)
    gate.register("ETHUSDT", eng_eth)

    assert gate.allow_new_segment("ETHUSDT") is True
    assert eng_btc.state.active is False


def test_gate_falls_back_to_state_active_without_holds_hook() -> None:
    class _Bare:
        def __init__(self, active: bool) -> None:
            self.state = _FakeState(active)

    gate = MultiLegConcurrencyGate(1)
    gate.register("BTCUSDT", _Bare(True))
    gate.register("ETHUSDT", _Bare(False))
    assert gate.allow_new_segment("ETHUSDT") is False


def test_gate_blocks_second_real_engine_on_bar(tmp_path: Path) -> None:
    """Integration: a shared gate (cap=1) blocks the 2nd symbol's _start_grid."""
    gate = MultiLegConcurrencyGate(1)
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
    max_concurrent_multi_leg_symbols: 3
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
        max_concurrent_multi_leg_symbols=0,
    )
    apply_multi_leg_args_from_constitution(args)
    assert args.unit_notional == pytest.approx(10000.0 * 0.01 / (0.03 * 6))
    assert args.unit_notional_by_strategy["chop_grid"] == pytest.approx(
        10000.0 * 0.01 / (0.03 * 6)
    )
    assert args.max_concurrent_multi_leg_symbols == 3


# ── cooldown tests ──────────────────────────────────────────────────────────


def test_cooldown_zero_means_no_cooldown() -> None:
    """cooldown_bars=0 disables cooldown entirely."""
    gate = MultiLegConcurrencyGate(6, cooldown_bars=0)
    chop_eng = _FakeEngine(True)
    trend_eng = _FakeEngine(False)
    gate.register("BTCUSDT", chop_eng, strategy="chop_grid")
    gate.register("BTCUSDT", trend_eng, strategy="trend_scalp")

    # chop deactivates, trend takes over
    chop_eng.state.active = False
    gate.notify_deactivation("BTCUSDT", "chop_grid")
    trend_eng.state.active = True

    # chop should be allowed immediately (no cooldown)
    assert gate.allow_new_segment("BTCUSDT", strategy="chop_grid") is True


def test_cooldown_blocks_reactivation_within_window() -> None:
    """After chop deactivates and trend takes over, chop cannot re-activate
    until cooldown_bars × 120 min have elapsed."""
    gate = MultiLegConcurrencyGate(6, cooldown_bars=3)
    chop_eng = _FakeEngine(True)
    trend_eng = _FakeEngine(False)
    gate.register("BTCUSDT", chop_eng, strategy="chop_grid")
    gate.register("BTCUSDT", trend_eng, strategy="trend_scalp")

    # Simulate: chop deactivates, trend takes over
    chop_eng.state.active = False
    gate.notify_deactivation("BTCUSDT", "chop_grid")
    trend_eng.state.active = True

    # chop tries to re-activate immediately → blocked
    assert gate.allow_new_segment("BTCUSDT", strategy="chop_grid") is False
    # trend should still be allowed (it currently holds the slot)
    assert gate.allow_new_segment("BTCUSDT", strategy="trend_scalp") is True


def test_cooldown_allows_reactivation_after_expiry() -> None:
    """After cooldown expires, the deactivated strategy can re-activate."""
    gate = MultiLegConcurrencyGate(6, cooldown_bars=3)
    chop_eng = _FakeEngine(True)
    trend_eng = _FakeEngine(False)
    gate.register("BTCUSDT", chop_eng, strategy="chop_grid")
    gate.register("BTCUSDT", trend_eng, strategy="trend_scalp")

    # Simulate: chop deactivates, trend takes over
    chop_eng.state.active = False
    gate.notify_deactivation("BTCUSDT", "chop_grid")
    trend_eng.state.active = True

    # Fake time passing beyond cooldown (3 bars × 120 min × 60 s = 21600 s)
    import src.order_management.chop_grid_concurrency as mod

    original_monotonic = mod.time.monotonic
    # Shift monotonic clock forward by cooldown + 1 second
    offset = gate._cooldown_seconds + 1
    mod.time.monotonic = lambda: original_monotonic() + offset

    try:
        assert gate.allow_new_segment("BTCUSDT", strategy="chop_grid") is True
    finally:
        mod.time.monotonic = original_monotonic


def test_cooldown_does_not_block_when_same_strategy_holds() -> None:
    """If the same strategy still holds the slot, cooldown does not apply."""
    gate = MultiLegConcurrencyGate(6, cooldown_bars=3)
    chop_eng = _FakeEngine(True)
    gate.register("BTCUSDT", chop_eng, strategy="chop_grid")

    # notify_deactivation called (e.g. from clear_stale_active_if_ghost)
    # but chop still holds the slot
    gate.notify_deactivation("BTCUSDT", "chop_grid")

    # chop should still be allowed (it holds the slot, no other strategy took over)
    assert gate.allow_new_segment("BTCUSDT", strategy="chop_grid") is True


def test_cooldown_does_not_block_when_no_strategy_holds() -> None:
    """If no strategy holds the slot, cooldown does not apply."""
    gate = MultiLegConcurrencyGate(6, cooldown_bars=3)
    chop_eng = _FakeEngine(True)
    trend_eng = _FakeEngine(False)
    gate.register("BTCUSDT", chop_eng, strategy="chop_grid")
    gate.register("BTCUSDT", trend_eng, strategy="trend_scalp")

    # Both deactivate
    chop_eng.state.active = False
    gate.notify_deactivation("BTCUSDT", "chop_grid")
    # trend was never active, so no one holds the slot

    # chop should be allowed (no other strategy holds the slot)
    assert gate.allow_new_segment("BTCUSDT", strategy="chop_grid") is True


def test_cooldown_independent_per_symbol() -> None:
    """Cooldown on BTCUSDT does not affect ETHUSDT."""
    gate = MultiLegConcurrencyGate(6, cooldown_bars=3)
    btc_chop = _FakeEngine(True)
    btc_trend = _FakeEngine(False)
    eth_chop = _FakeEngine(False)
    gate.register("BTCUSDT", btc_chop, strategy="chop_grid")
    gate.register("BTCUSDT", btc_trend, strategy="trend_scalp")
    gate.register("ETHUSDT", eth_chop, strategy="chop_grid")

    # BTC: chop deactivates, trend takes over
    btc_chop.state.active = False
    gate.notify_deactivation("BTCUSDT", "chop_grid")
    btc_trend.state.active = True

    # BTC chop blocked by cooldown
    assert gate.allow_new_segment("BTCUSDT", strategy="chop_grid") is False
    # ETH chop should be fine (no cooldown on ETHUSDT)
    assert gate.allow_new_segment("ETHUSDT", strategy="chop_grid") is True
