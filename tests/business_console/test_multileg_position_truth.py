"""Tests for multileg_position_truth leg-id matching."""

from mlbot_console.services.multileg_position_truth import (
    leg_key_is_pruned_ghost,
    leg_key_matches_open_position_legs,
)


def test_leg_key_matches_exact() -> None:
    active = {"leg_a", "leg_b"}
    assert leg_key_matches_open_position_legs("leg_a", active)
    assert not leg_key_matches_open_position_legs("leg_c", active)


def test_leg_key_matches_trend_scalp_fill_suffix() -> None:
    order_leg = "HYPEUSDT_2026-06-16 03:26:52+00:00_initial_trend_BUY_0_0"
    active = {f"{order_leg}_fill0"}
    assert leg_key_matches_open_position_legs(order_leg, active)


def test_leg_key_matches_empty() -> None:
    assert not leg_key_matches_open_position_legs("", {"leg_a"})
    assert not leg_key_matches_open_position_legs("leg_a", set())


def test_leg_key_is_pruned_ghost_closed_row() -> None:
    assert leg_key_is_pruned_ghost(
        "ghost_leg",
        positions_table_used=True,
        active_leg_ids=set(),
        closed_leg_ids={"ghost_leg"},
    )


def test_leg_key_is_pruned_ghost_restart_fallback() -> None:
    """No open or closed row for leg → show via order-based fallback."""
    assert not leg_key_is_pruned_ghost(
        "orphan_leg",
        positions_table_used=True,
        active_leg_ids=set(),
        closed_leg_ids=set(),
    )


def test_leg_key_is_pruned_ghost_other_legs_open() -> None:
    assert leg_key_is_pruned_ghost(
        "ghost_leg",
        positions_table_used=True,
        active_leg_ids={"active_leg"},
        closed_leg_ids=set(),
    )
