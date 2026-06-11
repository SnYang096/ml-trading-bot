"""Unit tests for constitution-driven monitor strategy resolution."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.monitoring.constitution_strategies import resolve_manifest_strategies

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pcm_layer_filters_to_drift_ready_only():
    manifest = {
        "strategies_source": "constitution",
        "strategies_layer": "pcm",
        "constitution": "live/highcap/config/constitution/constitution.yaml",
    }
    slugs, meta = resolve_manifest_strategies(manifest, repo_root=PROJECT_ROOT)
    assert slugs == ["tpc"]
    assert set(meta.get("skipped_not_ready") or []) >= {"bpc", "me", "srb"}


def test_multi_leg_layer_reads_constitution():
    manifest = {
        "strategies_source": "constitution",
        "strategies_layer": "multi_leg",
        "constitution": "live/highcap/config/constitution/constitution.yaml",
    }
    slugs, meta = resolve_manifest_strategies(manifest, repo_root=PROJECT_ROOT)
    assert slugs == ["chop_grid", "trend_scalp"]
    assert meta["strategies_layer"] == "multi_leg"


def test_legacy_explicit_list_still_filters_pcm(tmp_path: Path):
    manifest = {"strategies": ["bpc", "tpc", "me"]}
    slugs, meta = resolve_manifest_strategies(manifest, repo_root=PROJECT_ROOT)
    assert slugs == ["tpc"]
    assert meta["strategies_source"] == "explicit"
    assert "bpc" in meta["skipped_not_ready"]


def test_weekly_manifest_resolves_tpc_only():
    path = PROJECT_ROOT / "config/monitoring/weekly_rule_stack.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    slugs, _ = resolve_manifest_strategies(manifest, repo_root=PROJECT_ROOT)
    assert slugs == ["tpc"]
