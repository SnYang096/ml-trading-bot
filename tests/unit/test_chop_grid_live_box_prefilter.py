from __future__ import annotations

from pathlib import Path

import pytest

from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine
from src.time_series_model.live.multileg_prefilter_rules import (
    features_pass_prefilter_rules,
)
from src.time_series_model.live.segment_lifecycle import SegmentState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_PREFILTER = PROJECT_ROOT / "config/strategies/chop_grid/archetypes/prefilter.yaml"


def _chop_grid_cfg(tmp_path: Path, *, rules_yaml: str = "") -> Path:
    cfg = tmp_path / "prefilter.yaml"
    cfg.write_text(
        f"""
regime:
  entry_chop_min: 0.50
  exit_chop_below: 0.32
  exclude_box_prefilter: false
  box_prefilter:
    stability_min: 0.85
    width_min: 0.04
    width_max: 0.30
    touches_min: 5
{rules_yaml}
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 3
risk:
  fee_bps: 4.0
  max_open_levels_total: 6
""",
        encoding="utf-8",
    )
    return cfg


_BOX_POS_RULES = """
rules:
  - feature: box_pos_60
    operator: ">="
    value: 0.40
  - feature: box_pos_60
    operator: "<="
    value: 0.60
"""


def test_chop_grid_blocks_entry_on_stable_box_from_yaml_thresholds(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_grid_cfg(
            tmp_path,
            rules_yaml="""
rules:
  - all_of:
      - feature: box_pos_60
        operator: ">="
        value: 0.35
""",
        ),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="BTCUSDT",
        timestamp="2026-01-01T00:00:00Z",
        high=100.0,
        low=100.0,
        close=100.0,
        atr=2.0,
        features={
            "bpc_semantic_chop": 0.8,
            "box_stability_60": 0.90,
            "box_width_pct_60": 0.06,
            "box_touches_hi_60": 7.0,
            "box_touches_lo_60": 6.0,
        },
    )
    assert actions == []


def test_chop_grid_blocks_entry_when_box_pos_outside_band(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_grid_cfg(tmp_path, rules_yaml=_BOX_POS_RULES),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="HYPEUSDT",
        timestamp="2026-05-30T22:00:00Z",
        high=68.30,
        low=67.14,
        close=68.27,
        atr=0.5,
        features={
            "bpc_semantic_chop": 0.80,
            "box_pos_60": 0.976,
            "box_stability_60": 0.50,
            "box_width_pct_60": 0.19,
            "box_touches_hi_60": 22.0,
            "box_touches_lo_60": 3.0,
        },
    )
    assert actions == []
    assert engine._last_bar_audit["prefilter_ok"] is False
    assert engine._last_bar_audit["outcome"] == "flat_blocked_prefilter"


def test_chop_grid_exits_active_segment_when_box_pos_leaves_band(
    tmp_path: Path,
) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_grid_cfg(tmp_path, rules_yaml=_BOX_POS_RULES),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
        bar_simulation=True,
    )
    engine.state.active = True
    engine.state.symbol = "HYPEUSDT"
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.grid_id = "HYPEUSDT_2026-05-30T18:00:00Z"
    engine.state.center = 67.80
    engine.state.spacing = 0.22
    from src.time_series_model.live.chop_grid_live_engine import GridPosition

    engine.state.inventory = [
        GridPosition(
            symbol="HYPEUSDT",
            side="SHORT",
            level=1,
            entry_price=68.02,
            quantity=1.0,
            entry_quantity=1.0,
            entry_time="2026-05-30T20:00:00Z",
            leg_id="HYPEUSDT_2026-05-30T18:00:00Z_S1",
        )
    ]
    engine.state.pending_orders = []

    actions = engine.on_bar(
        symbol="HYPEUSDT",
        timestamp="2026-05-30T22:00:00Z",
        high=68.30,
        low=68.05,
        close=68.27,
        atr=0.5,
        features={
            "bpc_semantic_chop": 0.80,
            "box_pos_60": 0.976,
            "box_stability_60": 0.50,
            "box_width_pct_60": 0.19,
            "box_touches_hi_60": 22.0,
            "box_touches_lo_60": 3.0,
        },
    )
    assert any(a.get("action") == "market_exit" for a in actions)
    assert engine._last_bar_audit["should_exit"] is True
    assert engine.state.segment_state == SegmentState.IDLE.value
    assert not any(a.get("action") == "place" for a in actions)


def test_chop_grid_skips_replenish_when_prefilter_fails(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_grid_cfg(tmp_path, rules_yaml=_BOX_POS_RULES),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
        bar_simulation=False,
    )
    engine.state.active = True
    engine.state.symbol = "HYPEUSDT"
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.grid_id = "HYPEUSDT_2026-05-30T18:00:00Z"
    engine.state.center = 67.80
    engine.state.spacing = 0.22
    engine.state.inventory = []
    engine.state.pending_orders = []

    actions = engine.on_bar(
        symbol="HYPEUSDT",
        timestamp="2026-05-30T22:00:00Z",
        high=68.30,
        low=68.05,
        close=68.27,
        atr=0.5,
        features={
            "bpc_semantic_chop": 0.80,
            "box_pos_60": 0.976,
        },
    )
    assert not any(a.get("action") == "place" for a in actions)
    assert engine._last_bar_audit["should_exit"] is True


def test_features_pass_prefilter_box_pos_band() -> None:
    rules = [
        {"feature": "box_pos_60", "operator": ">=", "value": 0.40},
        {"feature": "box_pos_60", "operator": "<=", "value": 0.60},
    ]
    assert features_pass_prefilter_rules({"box_pos_60": 0.50}, rules)
    assert not features_pass_prefilter_rules({"box_pos_60": 0.976}, rules)
    assert not features_pass_prefilter_rules({}, rules)


def test_chop_grid_allows_entry_when_box_pos_in_band(tmp_path: Path) -> None:
    engine = ChopGridLiveEngine(
        config_path=_chop_grid_cfg(tmp_path, rules_yaml=_BOX_POS_RULES),
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="HYPEUSDT",
        timestamp="2026-05-28T18:00:00Z",
        high=61.5,
        low=59.5,
        close=60.97,
        atr=1.0,
        features={
            "bpc_semantic_chop": 0.82,
            "box_pos_60": 0.55,
            "box_stability_60": 0.69,
            "box_width_pct_60": 0.14,
            "box_touches_hi_60": 23.0,
            "box_touches_lo_60": 3.0,
        },
    )
    assert any(a.get("action") == "place" for a in actions)
    assert engine._last_bar_audit["prefilter_ok"] is True
    assert engine._last_bar_audit["wanted_enter"] is True
    assert engine._last_bar_audit["outcome"] == "open_grid_placed"


@pytest.mark.skipif(not PROD_PREFILTER.is_file(), reason="prod prefilter.yaml missing")
def test_chop_grid_prod_prefilter_blocks_hype_rally_top(tmp_path: Path) -> None:
    """Locked prod rules: box_pos_60 in [0.40, 0.60] — HYPE 0.976 must not enter."""
    engine = ChopGridLiveEngine(
        config_path=PROD_PREFILTER,
        state_path=tmp_path / "state.json",
        level_notional=100.0,
    )
    actions = engine.on_bar(
        symbol="HYPEUSDT",
        timestamp="2026-05-30T22:00:00Z",
        high=68.30,
        low=68.05,
        close=68.27,
        atr=0.5,
        features={
            "bpc_semantic_chop": 0.805,
            "box_pos_60": 0.976,
            "box_stability_60": 0.503,
            "box_width_pct_60": 0.195,
            "box_touches_hi_60": 22.0,
            "box_touches_lo_60": 3.0,
        },
    )
    assert actions == []
    assert engine._last_bar_audit["outcome"] == "flat_blocked_prefilter"


@pytest.mark.skipif(not PROD_PREFILTER.is_file(), reason="prod prefilter.yaml missing")
def test_chop_grid_prod_prefilter_rules_loaded() -> None:
    from src.config.multileg_config import load_multileg_effective_config
    from src.config.strategy_layout import resolve_strategy_config_input

    config_dir, profile_path, engine_path = resolve_strategy_config_input(
        PROD_PREFILTER
    )
    cfg = load_multileg_effective_config(
        config_dir=config_dir,
        strategy_type="grid",
        profile_path=profile_path,
        engine_path=engine_path,
    )
    rules = cfg.get("rules") or []
    feats = {
        str(r.get("feature")) for r in rules if isinstance(r, dict) and r.get("feature")
    }
    assert "box_pos_60" in feats
