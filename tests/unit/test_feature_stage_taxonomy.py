"""Archetype-driven feature stage taxonomy for business console."""

from __future__ import annotations

from pathlib import Path

from time_series_model.live.feature_stage_taxonomy import (
    build_console_feature_taxonomy,
    extract_strategy_stage_columns,
)

ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = ROOT / "config" / "strategies"


def test_tpc_prefilter_form_and_regime_columns():
    """TPC: form columns stay in prefilter; box/chop slow variables move to regime."""
    stages = extract_strategy_stage_columns(STRATEGIES / "tpc" / "archetypes")
    assert "tpc_pullback_depth" in stages["prefilter"]
    # chop + box 已迁移到 regime.yaml (A/B/C 共用慢变量层)
    assert "tpc_semantic_chop" in stages["regime"]
    assert "box_pos_120" in stages["regime"]


def test_spot_weekly_ema_prefilter():
    stages = extract_strategy_stage_columns(
        STRATEGIES / "spot_accum_simple" / "archetypes"
    )
    assert "weekly_ema_200_position" in stages["prefilter"]


def test_chop_grid_regime_and_rules():
    stages = extract_strategy_stage_columns(STRATEGIES / "chop_grid" / "archetypes")
    # multileg engines still embed regime: inside prefilter.yaml (different schema)
    assert "bpc_semantic_chop" in stages["regime"]
    assert "box_pos_60" in stages["prefilter"]


def test_taxonomy_includes_registry_stub_without_archetypes(tmp_path: Path):
    tax = build_console_feature_taxonomy(
        tmp_path,
        strategies=[
            {"id": "chop_grid", "account_layer": "multi_leg", "title": "Chop Grid"},
            {"id": "trend_scalp", "account_layer": "multi_leg", "title": "Trend Scalp"},
        ],
    )
    ids = {s["id"] for s in tax["strategies"]}
    assert ids == {"chop_grid", "trend_scalp"}


def test_taxonomy_index_shared_chop_column_lives_in_regime_for_trend():
    """tpc_semantic_chop is now a B-system regime variable (was gate)."""
    tax = build_console_feature_taxonomy(STRATEGIES)
    hits = tax["index"]["tpc_semantic_chop"]
    strategies = {h["strategy"] for h in hits}
    assert "tpc" in strategies
    assert "bpc" in strategies
    # B-system trend strategies report chop under regime, not gate
    trend_hits = [h for h in hits if h["account_layer"] == "trend"]
    assert trend_hits, "expected at least one trend hit"
    assert all(h["stage"] == "regime" for h in trend_hits)
    # C-system multileg keeps regime: block inside prefilter.yaml → reported as regime too
    assert any(h["strategy"] == "chop_grid" and h["stage"] == "regime" for h in hits)
