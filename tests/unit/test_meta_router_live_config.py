import pytest

from src.time_series_model.live.meta_router_config import (
    load_meta_router_live_config,
    select_first_enabled_archetype,
)


@pytest.mark.unit
def test_meta_router_live_config_load_and_select(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        """
version: 1
name: "x"
enabled_archetypes:
  TREND: ["A", "B"]
  MEAN: ["C"]
size_multipliers:
  A: 1.0
vol_mean:
  enabled: true
  archetype_id: "V"
  size_multiplier: 0.05
""",
        encoding="utf-8",
    )
    cfg = load_meta_router_live_config(p)
    assert select_first_enabled_archetype(cfg, regime="TREND") == "A"
    assert cfg.vol_mean.enabled is True
    assert cfg.vol_mean.size_multiplier == pytest.approx(0.05)
