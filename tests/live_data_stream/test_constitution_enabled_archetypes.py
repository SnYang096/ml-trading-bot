"""Tests for ``enabled_archetypes_from_constitution`` (classic live + PCM backtest single source)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.live_data_stream.constitution_config import (
    classic_slot_policy_from_constitution,
    enabled_archetypes_from_constitution,
    load_constitution_dict,
    multi_leg_strategies_from_constitution,
    partition_pipeline_strategies_by_type,
    spot_account_equity_anchor_usdt,
    validate_classic_slot_capacity,
    validate_pipeline_constitution_alignment,
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


def test_multi_leg_strategies_from_constitution_supports_list_and_string() -> None:
    assert (
        multi_leg_strategies_from_constitution({"multi_leg": {"strategies": []}}) == []
    )
    assert multi_leg_strategies_from_constitution(
        {"multi_leg": {"strategies": ["chop_grid", "trend_scalp"]}}
    ) == ["chop_grid", "trend_scalp"]
    assert multi_leg_strategies_from_constitution(
        {"multi_leg": {"strategies": "chop_grid, trend_scalp"}}
    ) == ["chop_grid", "trend_scalp"]


def test_partition_pipeline_strategies_by_type() -> None:
    cfg = {
        "strategies": {
            "bpc": {"strategy_type": "single"},
            "me": {},
            "chop_grid": {"strategy_type": "grid"},
            "trend_scalp": {"strategy_type": "trend_scalp"},
        }
    }
    got = partition_pipeline_strategies_by_type(cfg)
    assert got["classic"] == {"bpc", "me"}
    assert got["multi_leg"] == {"chop_grid", "trend_scalp"}


def test_classic_slot_policy_uses_archetype_groups_without_duplicate_list() -> None:
    cfg = {
        "resource_allocation": {
            "slot_policy": {
                "trend_group": "trend",
                "min_trend_slots_per_symbol": 1,
                "max_trend_slots_per_symbol": 1,
            },
            "archetype_groups": {"trend": ["bpc", "tpc", "me"]},
        }
    }
    policy = classic_slot_policy_from_constitution(cfg)
    assert policy["trend_archetypes"] == ["bpc", "tpc", "me"]
    assert policy["max_trend_slots_per_symbol"] == 1


def test_classic_slot_policy_intersects_group_with_enabled_archetypes() -> None:
    """Research: full trend taxonomy ∩ enabled whitelist."""
    cfg = {
        "resource_allocation": {
            "enabled_archetypes": ["tpc"],
            "slot_policy": {"trend_group": "trend"},
            "archetype_groups": {"trend": ["tpc", "bpc", "me", "srb"]},
        }
    }
    policy = classic_slot_policy_from_constitution(cfg)
    assert policy["trend_archetypes"] == ["tpc"]


def test_classic_slot_policy_uses_enabled_when_groups_omitted() -> None:
    """Live constitution: no archetype_groups; trend pool = enabled_archetypes."""
    cfg = {
        "resource_allocation": {
            "enabled_archetypes": ["tpc"],
            "slot_policy": {"trend_group": "trend"},
        }
    }
    policy = classic_slot_policy_from_constitution(cfg)
    assert policy["trend_archetypes"] == ["tpc"]


def test_spot_account_equity_anchor_prefers_equity_usdt() -> None:
    assert spot_account_equity_anchor_usdt({"equity_usdt": 8000}) == 8000.0
    assert spot_account_equity_anchor_usdt({"backtest_equity_usdt": 7000}) == 7000.0
    assert (
        spot_account_equity_anchor_usdt(
            {"equity_usdt": 9000, "backtest_equity_usdt": 7000}
        )
        == 9000.0
    )


def test_validate_classic_slot_capacity_rejects_too_few_slots() -> None:
    cfg = {
        "slots": {"slot_count": 1},
        "resource_allocation": {
            "slot_policy": {"min_trend_slots_per_symbol": 1},
            "archetype_groups": {"trend": ["bpc"]},
        },
    }
    with pytest.raises(ValueError, match="slot_count is too small"):
        validate_classic_slot_capacity(
            constitution_cfg=cfg, symbols=["BTCUSDT", "ETHUSDT"]
        )


def test_validate_classic_slot_capacity_accepts_csv_string_from_pipeline() -> None:
    cfg = {
        "slots": {"slot_count": 10},
        "resource_allocation": {
            "slot_policy": {"min_trend_slots_per_symbol": 1},
            "archetype_groups": {"trend": ["bpc"]},
        },
    }
    got = validate_classic_slot_capacity(
        constitution_cfg=cfg,
        symbols="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT",
    )
    assert got["required_trend_slots"] == 6
    assert len(got["symbols"]) == 6


def test_validate_pipeline_constitution_alignment_ok() -> None:
    cfg = {
        "strategies": {
            "bpc": {},
            "tpc": {},
            "me": {},
            "chop_grid": {"strategy_type": "grid"},
            "trend_scalp": {"strategy_type": "trend_scalp"},
        }
    }
    constitution = {
        "resource_allocation": {"enabled_archetypes": ["bpc", "tpc", "me"]},
        "multi_leg": {"strategies": ["chop_grid", "trend_scalp"]},
    }
    got = validate_pipeline_constitution_alignment(
        pipeline_cfg=cfg, constitution_cfg=constitution, context_label="unit_test"
    )
    assert got["classic"] == ["bpc", "me", "tpc"]
    assert got["multi_leg"] == ["chop_grid", "trend_scalp"]


def test_validate_pipeline_constitution_alignment_allows_constitution_superset() -> (
    None
):
    cfg = {"strategies": {"bpc": {}}}
    constitution = {
        "resource_allocation": {"enabled_archetypes": ["bpc", "me", "tpc"]},
        "multi_leg": {"strategies": ["chop_grid", "trend_scalp"]},
    }
    got = validate_pipeline_constitution_alignment(
        pipeline_cfg=cfg, constitution_cfg=constitution, context_label="unit_test"
    )
    assert got["classic"] == ["bpc"]
    assert got["multi_leg"] == []


def test_validate_pipeline_constitution_alignment_raises_on_mismatch() -> None:
    cfg = {
        "strategies": {
            "bpc": {},
            "fer": {},
            "chop_grid": {"strategy_type": "grid"},
        }
    }
    constitution = {
        "resource_allocation": {"enabled_archetypes": ["bpc", "tpc"]},
        "multi_leg": {"strategies": ["trend_scalp"]},
    }
    with pytest.raises(ValueError, match="pipeline strategy not allowed"):
        validate_pipeline_constitution_alignment(
            pipeline_cfg=cfg, constitution_cfg=constitution, context_label="rolling"
        )
