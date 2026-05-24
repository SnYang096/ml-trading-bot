"""Tests for Trade Map feature threshold hints and reference lines."""

from mlbot_console.services.feature_thresholds import (
    build_reference_lines_by_column,
    semantic_hint_for_column,
)


def test_semantic_hint_tpc_chop_high():
    hint = semantic_hint_for_column("tpc_semantic_chop", 0.99)
    assert "regime禁入" in hint
    assert "0.40" in hint


def test_semantic_hint_tpc_chop_low():
    hint = semantic_hint_for_column("tpc_semantic_chop", 0.25)
    assert "可入场" in hint


def test_semantic_hint_compression_pass():
    hint = semantic_hint_for_column("bpc_volume_compression_pct", 0.93)
    assert "通过" in hint


def test_build_reference_lines_builtin():
    refs = build_reference_lines_by_column()
    assert "tpc_semantic_chop" in refs
    assert any(abs(r["y"] - 0.40) < 1e-6 for r in refs["tpc_semantic_chop"])


def test_build_reference_lines_includes_spot_weekly_ema(tmp_path):
    strat = tmp_path / "spot_accum_simple" / "archetypes"
    strat.mkdir(parents=True)
    (strat / "prefilter.yaml").write_text(
        "rules:\n"
        "  - feature: weekly_ema_200_position\n"
        "    operator: <\n"
        "    value: 0.0\n",
        encoding="utf-8",
    )
    refs = build_reference_lines_by_column(tmp_path)
    assert "weekly_ema_200_position" in refs
    assert any(abs(r["y"]) < 1e-9 for r in refs["weekly_ema_200_position"])


def test_semantic_hint_weekly_ema_position():
    assert "深熊" in semantic_hint_for_column("weekly_ema_200_position", -0.05)
    assert "EMA上方" in semantic_hint_for_column("weekly_ema_200_position", 0.02)
