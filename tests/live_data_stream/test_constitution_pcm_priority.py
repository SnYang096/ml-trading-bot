from __future__ import annotations

from src.live_data_stream.constitution_config import (
    enabled_archetypes_from_constitution,
    intent_archetype_priority_tokens,
    pcm_archetype_priority_for_registry,
    pcm_resolve_registry_key,
)
from src.live_data_stream.strategy_runtime_config import me_enabled_in_allowlist


def test_enabled_archetypes_default_full_set() -> None:
    assert enabled_archetypes_from_constitution({}) == [
        "bpc",
        "me",
        "srb",
        "tpc",
        "lv",
        "fbf",
        "msr",
        "fer",
    ]


def test_enabled_archetypes_explicit_list() -> None:
    cfg = {"resource_allocation": {"enabled_archetypes": ["bpc", "me", "tpc", "srb"]}}
    assert enabled_archetypes_from_constitution(cfg) == ["bpc", "me", "tpc", "srb"]


def test_enabled_archetypes_comma_string() -> None:
    cfg = {"resource_allocation": {"enabled_archetypes": "bpc, me, tpc"}}
    assert enabled_archetypes_from_constitution(cfg) == ["bpc", "me", "tpc"]


def test_pcm_resolve_me_token() -> None:
    assert pcm_resolve_registry_key("me", "me", me_enabled_in_allowlist) == "me"
    assert pcm_resolve_registry_key("me-long", "me", me_enabled_in_allowlist) == "me"


def test_intent_archetype_priority_follows_enabled_when_no_override() -> None:
    cfg = {
        "resource_allocation": {
            "enabled_archetypes": ["tpc", "bpc", "me"],
        }
    }
    assert intent_archetype_priority_tokens(cfg) == ["tpc", "bpc", "me"]


def test_intent_archetype_priority_explicit_override_wins() -> None:
    cfg = {
        "resource_allocation": {
            "enabled_archetypes": ["bpc", "me", "tpc"],
            "intent_selection_policy": {
                "archetype_priority": ["tpc", "me", "bpc"],
            },
        }
    }
    assert intent_archetype_priority_tokens(cfg) == ["tpc", "me", "bpc"]


def test_intent_archetype_priority_comma_string_override() -> None:
    cfg = {
        "resource_allocation": {
            "enabled_archetypes": ["bpc"],
            "intent_selection_policy": {"archetype_priority": "me, tpc, bpc"},
        }
    }
    assert intent_archetype_priority_tokens(cfg) == ["me", "tpc", "bpc"]


def test_pcm_priority_filters_to_registry_keys() -> None:
    cfg = {
        "resource_allocation": {
            "intent_selection_policy": {
                "archetype_priority": ["tpc", "srb", "me", "bpc", "ghost"],
            }
        }
    }
    reg = {"bpc", "me", "tpc"}
    pr = pcm_archetype_priority_for_registry(
        cfg,
        registry_keys=reg,
        me_pkg="me",
        me_enabled_in_allowlist_fn=me_enabled_in_allowlist,
    )
    assert pr == ["tpc", "me", "bpc"]
