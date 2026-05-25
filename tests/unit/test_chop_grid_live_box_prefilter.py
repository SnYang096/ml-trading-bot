from __future__ import annotations

from pathlib import Path

from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine


def test_chop_grid_blocks_entry_on_stable_box_from_yaml_thresholds(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "prefilter.yaml"
    cfg.write_text(
        """
regime:
  entry_chop_min: 0.50
  exit_chop_below: 0.32
  exclude_box_prefilter: false
  box_prefilter:
    stability_min: 0.85
    width_min: 0.04
    width_max: 0.30
    touches_min: 5
rules:
  - all_of:
      - feature: box_pos_60
        operator: ">="
        value: 0.35
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 1
risk:
  fee_bps: 4.0
  max_open_levels_total: 2
""",
        encoding="utf-8",
    )
    engine = ChopGridLiveEngine(
        config_path=cfg,
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
