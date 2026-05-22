"""Regime.yaml drives box_structure_f in live feature plan (Feature Bus readiness)."""

from __future__ import annotations

from pathlib import Path

from time_series_model.live.live_feature_plan import extract_features_from_archetypes


def test_regime_yaml_pulls_box_structure_node(tmp_path: Path) -> None:
    arch = tmp_path / "archetypes"
    arch.mkdir()
    (arch / "regime.yaml").write_text(
        """
rules:
  - feature: tpc_semantic_chop
    operator: "<="
    value: 0.40
  - any_of:
      - feature: box_pos_120
        operator: "<="
        value: 0.15
      - feature: box_breakout_up
        operator: ">="
        value: 0.5
""",
        encoding="utf-8",
    )
    cols, nodes = extract_features_from_archetypes(
        arch,
        feature_deps_path=Path("config/feature_dependencies.yaml"),
    )
    assert "box_pos_120" in cols
    assert "tpc_semantic_chop" in cols
    assert "box_structure_f" in nodes


def test_live_tpc_has_regime_yaml() -> None:
    live_regime = Path("live/highcap/config/strategies/tpc/archetypes/regime.yaml")
    assert live_regime.is_file(), "deploy regime.yaml to live/highcap first"
