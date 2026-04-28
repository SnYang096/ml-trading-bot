from __future__ import annotations

from unittest.mock import MagicMock

from src.time_series_model.live.event_backtest_srb_hooks import SrbEventBacktestHooks


def test_try_from_strategies_none_without_srb() -> None:
    assert SrbEventBacktestHooks.try_from_strategies(["bpc"], {}) is None


def test_reject_wide_entry_false_when_guard_disabled() -> None:
    raw = {"sr_wide_entry_guard": {"enabled": False}}
    h = SrbEventBacktestHooks(
        execution_raw=raw, add_policy=None, wide_entry_guard=raw["sr_wide_entry_guard"]
    )
    sim = MagicMock()
    sim._srb_wide_entry_guard = raw["sr_wide_entry_guard"]
    funnel: dict = {}
    assert (
        h.reject_new_entry_wide_sr_guard(
            arch_lc="srb",
            is_new_entry=True,
            simulator=sim,
            entry_feats={},
            intent=MagicMock(action="BUY"),
            funnel=funnel,
        )
        is False
    )
