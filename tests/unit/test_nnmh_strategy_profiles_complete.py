from pathlib import Path

import pytest

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
    load_strategy_profile,
)


@pytest.mark.unit
def test_nnmh_tc_te_fr_et_profiles_exist_and_valid():
    root = Path("config/nnmultihead/strategies")
    registry = load_execution_archetypes_registry(
        "config/nnmultihead/execution_archetypes_v2.yaml"
    )
    assert registry

    required = [
        "trend_continuation_tc",
        "trend_expansion_te",
        "failure_reversion_fr",
        "exhaustion_turn_et",
    ]
    missing = []
    invalid = []

    for sid in required:
        prof = load_strategy_profile(strategy_id=sid, root_dir=root)
        if prof is None or not prof.archetype:
            missing.append(sid)
            continue
        if prof.archetype not in registry:
            invalid.append(f"{sid} -> {prof.archetype}")

    assert missing == [], f"Missing profile.yaml in: {missing}"
    assert invalid == [], f"Invalid archetype in: {invalid}"
