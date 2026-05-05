"""Tests for scripts/deploy_config_to_live.py deploy profile + denylist."""

from pathlib import Path

from scripts.deploy_config_to_live import (
    DEPLOY_ROOT_DENYLIST,
    _skip_root_deploy_file,
    get_strategy_deploy_profile,
    iter_deploy_archetype_basenames,
)


def test_deploy_root_denylist_blocks_research_filenames():
    assert "research.yaml" in DEPLOY_ROOT_DENYLIST
    assert "threshold_search.yaml" in DEPLOY_ROOT_DENYLIST
    assert _skip_root_deploy_file("research.yaml") is True
    assert _skip_root_deploy_file("meta.yaml") is False


def test_multileg_profile_all_archetypes():
    prof = get_strategy_deploy_profile("chop_grid")
    assert prof.archetypes_mode == "all"
    assert prof.engine_yaml == "grid.yaml"


def test_classic_profile_whitelist():
    prof = get_strategy_deploy_profile("bpc")
    assert prof.archetypes_mode == "whitelist"
    assert "prefilter.yaml" in prof.archetype_whitelist
    assert prof.engine_yaml is None


def test_iter_deploy_archetype_basenames_respects_whitelist(tmp_path):
    arch = tmp_path / "archetypes"
    arch.mkdir()
    (arch / "prefilter.yaml").write_text("x: 1\n", encoding="utf-8")
    (arch / "noise.yaml").write_text("x: 1\n", encoding="utf-8")
    prof = get_strategy_deploy_profile("bpc")
    names = iter_deploy_archetype_basenames(prof, arch)
    assert names == ["prefilter.yaml"]
