"""Tests for scripts/deploy_config_to_live.py deploy profile + denylist."""

from pathlib import Path

from scripts.deploy_config_to_live import (
    DEPLOY_ROOT_DENYLIST,
    _skip_root_deploy_file,
    deploy_strategy,
    iter_deploy_root_basenames,
    iter_stale_live_root_entries,
    get_strategy_deploy_profile,
    iter_deploy_archetype_basenames,
)


def test_deploy_root_denylist_blocks_research_filenames():
    assert "research.yaml" in DEPLOY_ROOT_DENYLIST
    assert "threshold_search.yaml" in DEPLOY_ROOT_DENYLIST
    assert "features.yaml" in DEPLOY_ROOT_DENYLIST
    assert "training_baseline.json" in DEPLOY_ROOT_DENYLIST
    assert _skip_root_deploy_file("research.yaml") is True
    assert _skip_root_deploy_file("features.yaml") is True
    assert _skip_root_deploy_file("meta.yaml") is False


def test_multileg_profile_all_archetypes():
    prof = get_strategy_deploy_profile("chop_grid")
    assert prof.archetypes_mode == "all"
    assert prof.runtime_yaml is None


def test_classic_profile_whitelist():
    prof = get_strategy_deploy_profile("bpc")
    assert prof.archetypes_mode == "whitelist"
    assert "prefilter.yaml" in prof.archetype_whitelist
    assert prof.runtime_yaml is None


def test_iter_deploy_archetype_basenames_respects_whitelist(tmp_path):
    arch = tmp_path / "archetypes"
    arch.mkdir()
    (arch / "prefilter.yaml").write_text("x: 1\n", encoding="utf-8")
    (arch / "noise.yaml").write_text("x: 1\n", encoding="utf-8")
    prof = get_strategy_deploy_profile("bpc")
    names = iter_deploy_archetype_basenames(prof, arch)
    assert names == ["prefilter.yaml"]


def test_iter_deploy_root_basenames_only_allows_meta(tmp_path: Path):
    (tmp_path / "meta.yaml").write_text("strategy: {}\n", encoding="utf-8")
    (tmp_path / "features.yaml").write_text("feature_pipeline: {}\n", encoding="utf-8")
    (tmp_path / "training_baseline.json").write_text("{}", encoding="utf-8")

    assert iter_deploy_root_basenames(tmp_path) == ["meta.yaml"]


def test_iter_stale_live_root_entries_flags_non_deploy_roots(tmp_path: Path):
    (tmp_path / "meta.yaml").write_text("strategy: {}\n", encoding="utf-8")
    (tmp_path / "features.yaml").write_text("feature_pipeline: {}\n", encoding="utf-8")
    (tmp_path / "research").mkdir()
    (tmp_path / "archetypes").mkdir()

    stale = [p.name for p in iter_stale_live_root_entries(tmp_path)]
    assert stale == ["features.yaml", "research"]


def test_deploy_strategy_prunes_stale_live_research_files(tmp_path: Path, monkeypatch):
    research = tmp_path / "config/strategies"
    live = tmp_path / "live/highcap/config/strategies"
    src = research / "chop_grid"
    dst = live / "chop_grid"

    (src / "archetypes").mkdir(parents=True)
    (src / "meta.yaml").write_text("strategy:\n  name: chop_grid\n", encoding="utf-8")
    (src / "archetypes/prefilter.yaml").write_text("regime: {}\n", encoding="utf-8")
    (src / "features.yaml").write_text("feature_pipeline: {}\n", encoding="utf-8")

    (dst / "research").mkdir(parents=True)
    (dst / "features.yaml").write_text("old: true\n", encoding="utf-8")

    monkeypatch.setattr("scripts.deploy_config_to_live.RESEARCH_STRATEGIES", research)
    monkeypatch.setattr("scripts.deploy_config_to_live.LIVE_STRATEGIES", live)

    changed = deploy_strategy("chop_grid")
    assert changed >= 3
    assert (dst / "meta.yaml").exists()
    assert (dst / "archetypes/prefilter.yaml").exists()
    assert not (dst / "features.yaml").exists()
    assert not (dst / "research").exists()
