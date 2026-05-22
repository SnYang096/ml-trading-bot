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
