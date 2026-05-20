"""Publisher merges constitution spot + multi_leg consumers into feature bus FCs."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.live_data_stream.constitution_config import load_constitution_dict
from src.live_data_stream.feature_publisher_stack import (
    _constitution_consumer_strategy_slugs,
    _merge_consumer_features_for_primary,
)
from src.live_data_stream.strategy_runtime_config import load_strategy_timeframe

_ROOT = Path(__file__).resolve().parents[2]
_LIVE_STRATEGIES = _ROOT / "live" / "highcap" / "config" / "strategies"
_LIVE_CONSTITUTION = (
    _ROOT / "live" / "highcap" / "config" / "constitution" / "constitution.yaml"
)


@pytest.mark.skipif(
    not _LIVE_CONSTITUTION.is_file(),
    reason="live/highcap constitution not present",
)
def test_live_constitution_consumer_slugs_include_spot_and_multi_leg() -> None:
    cfg = load_constitution_dict(str(_LIVE_CONSTITUTION))
    slugs = _constitution_consumer_strategy_slugs(cfg)
    assert "spot_accum_simple" in slugs
    assert "chop_grid" in slugs
    assert "trend_scalp" in slugs


@pytest.mark.skipif(
    not (_LIVE_STRATEGIES / "spot_accum_simple" / "archetypes").is_dir(),
    reason="live spot_accum_simple archetypes not present",
)
def test_merge_consumer_features_includes_spot_weekly_ema_on_primary_tf() -> None:
    cfg = load_constitution_dict(str(_LIVE_CONSTITUTION))
    primary_tf = load_strategy_timeframe(str(_LIVE_STRATEGIES), "tpc")
    feat_set, nodes, other_tf = _merge_consumer_features_for_primary(
        cfg=cfg,
        strategies_root=str(_LIVE_STRATEGIES),
        primary_tf=primary_tf,
    )
    assert "weekly_ema_200_position" in feat_set
    assert "weekly_ema_200_position_f" in nodes
    assert other_tf == []


def test_constitution_consumer_slugs_dedupe_spot_and_multi_leg() -> None:
    cfg = {
        "spot": {"strategies": ["spot_accum_simple"]},
        "multi_leg": {"strategies": ["chop_grid", "chop_grid", "trend_scalp"]},
    }
    assert _constitution_consumer_strategy_slugs(cfg) == [
        "spot_accum_simple",
        "chop_grid",
        "trend_scalp",
    ]


def test_merge_consumer_routes_different_timeframe_to_extra_fc(tmp_path: Path) -> None:
    strategies_root = tmp_path / "strategies"
    pkg = strategies_root / "spot_only"
    (pkg / "archetypes").mkdir(parents=True)
    (pkg / "meta.yaml").write_text(
        'strategy:\n  name: spot-only\n  timeframe: "15min"\n',
        encoding="utf-8",
    )
    (pkg / "archetypes" / "prefilter.yaml").write_text(
        "rules:\n  - feature: weekly_ema_200_position\n    operator: <\n    value: 0.0\n",
        encoding="utf-8",
    )
    cfg = {"spot": {"strategies": ["spot_only"]}, "multi_leg": {"strategies": []}}
    feat_set, nodes, other_tf = _merge_consumer_features_for_primary(
        cfg=cfg,
        strategies_root=str(strategies_root),
        primary_tf="120T",
    )
    assert "weekly_ema_200_position" not in feat_set
    assert nodes == []
    assert len(other_tf) == 1
    assert other_tf[0].disk == "spot_only"
    assert other_tf[0].timeframe == "15min"
