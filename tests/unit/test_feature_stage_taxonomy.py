"""Archetype-driven feature stage taxonomy for business console."""

from __future__ import annotations

from pathlib import Path

from time_series_model.live.feature_stage_taxonomy import (
    build_console_feature_taxonomy,
    extract_strategy_stage_columns,
)

ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = ROOT / "config" / "strategies"


def test_tpc_prefilter_and_gate_columns():
    stages = extract_strategy_stage_columns(STRATEGIES / "tpc" / "archetypes")
    assert "tpc_pullback_depth" in stages["prefilter"]
    assert "tpc_semantic_chop" in stages["gate"]
    assert "box_pos_120" in stages["prefilter"]


def test_spot_weekly_ema_prefilter():
    stages = extract_strategy_stage_columns(
        STRATEGIES / "spot_accum_simple" / "archetypes"
    )
    assert "weekly_ema_200_position" in stages["prefilter"]


def test_chop_grid_regime_and_rules():
    stages = extract_strategy_stage_columns(STRATEGIES / "chop_grid" / "archetypes")
    assert "bpc_semantic_chop" in stages["regime"]
    assert "box_pos_60" in stages["prefilter"]


def test_taxonomy_index_shared_gate_column():
    tax = build_console_feature_taxonomy(STRATEGIES)
    hits = tax["index"]["tpc_semantic_chop"]
    strategies = {h["strategy"] for h in hits}
    assert "tpc" in strategies
    assert "bpc" in strategies
    trend_gates = [h for h in hits if h["account_layer"] == "trend"]
    assert all(h["stage"] == "gate" for h in trend_gates)
    assert any(h["strategy"] == "chop_grid" and h["stage"] == "regime" for h in hits)
