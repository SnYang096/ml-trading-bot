"""Tests for ``enabled_archetypes_from_constitution`` (classic live + PCM backtest single source)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.live_data_stream.constitution_config import (
    enabled_archetypes_from_constitution,
    load_constitution_dict,
)


def _full_default() -> list[str]:
    return [
        "bpc",
        "me",
        "srb",
        "tpc",
        "lv",
        "fbf",
        "msr",
        "fer",
    ]


def test_enabled_archetypes_resource_allocation_precedence_over_root() -> None:
    cfg = {
        "resource_allocation": {"enabled_archetypes": ["bpc", "tpc"]},
        "enabled_archetypes": ["me"],
    }
    assert enabled_archetypes_from_constitution(cfg) == ["bpc", "tpc"]


def test_enabled_archetypes_root_level_when_ra_has_no_key() -> None:
    cfg = {"enabled_archetypes": ["srb", "me"]}
    assert enabled_archetypes_from_constitution(cfg) == ["srb", "me"]


def test_enabled_archetypes_empty_list_under_ra_falls_through_then_full_set() -> None:
    """``[]`` is falsy in ``raw = ra.get(...) or ...``, so empty RA list is ignored; then ``[]`` triggers full default."""
    cfg = {"resource_allocation": {"enabled_archetypes": []}}
    assert enabled_archetypes_from_constitution(cfg) == _full_default()


def test_enabled_archetypes_tuple_accepted() -> None:
    cfg = {"resource_allocation": {"enabled_archetypes": ("bpc", "ME", "  tpc  ")}}
    assert enabled_archetypes_from_constitution(cfg) == ["bpc", "me", "tpc"]


def test_enabled_archetypes_empty_comma_string_returns_full() -> None:
    cfg = {"resource_allocation": {"enabled_archetypes": "  ,  , "}}
    assert enabled_archetypes_from_constitution(cfg) == _full_default()


def test_enabled_archetypes_unknown_type_returns_full() -> None:
    cfg = {"resource_allocation": {"enabled_archetypes": 123}}  # type: ignore[dict-item]
    assert enabled_archetypes_from_constitution(cfg) == _full_default()


def test_load_pcm_enabled_from_constitution_matches_helper(tmp_path) -> None:
    """Step 9.5 loader must use the same normalization as classic live."""
    from scripts.auto_research_pipeline import (
        _load_pcm_enabled_strategies_from_constitution,
    )

    path = tmp_path / "constitution.yaml"
    path.write_text(
        textwrap.dedent(
            """
            resource_allocation:
              enabled_archetypes:
                - bpc
                - tpc
            """
        ).strip(),
        encoding="utf-8",
    )
    loaded = load_constitution_dict(str(path))
    assert enabled_archetypes_from_constitution(loaded) == ["bpc", "tpc"]
    assert _load_pcm_enabled_strategies_from_constitution(path) == ["bpc", "tpc"]


def test_load_pcm_enabled_missing_file_returns_empty() -> None:
    from scripts.auto_research_pipeline import (
        _load_pcm_enabled_strategies_from_constitution,
    )

    assert (
        _load_pcm_enabled_strategies_from_constitution(
            Path("/nonexistent/no_constitution.yaml")
        )
        == []
    )
