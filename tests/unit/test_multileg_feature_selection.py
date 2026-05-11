from pathlib import Path

import yaml

from scripts.pipeline.multileg_feature_selection import select_multileg_feature_subset


def _write_feature_yaml(path: Path, requested: list[str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": path.stem,
                "feature_pipeline": {"requested_features": requested},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_grid_ablation_prefers_better_scoring_subset(tmp_path):
    cfg_dir = tmp_path / "chop_grid"
    cfg_dir.mkdir()
    requested = ["bpc_soft_phase_f", "box_structure_f", "atr_f", "ema_1200_slope_f"]
    _write_feature_yaml(cfg_dir / "features.yaml", requested)
    _write_feature_yaml(cfg_dir / "features_prefilter.yaml", requested)

    def _eval(name: str, _config_dir: Path, selected_nodes: set[str]) -> dict:
        score_map = {
            "core": 1.20,
            "full_default": 0.50,
        }
        if name.startswith("core_plus_box_structure_f__"):
            score_map[name] = 0.75
        return {"score": score_map.get(name, -1.0), "n_nodes": len(selected_nodes)}

    out = select_multileg_feature_subset(
        strategy="chop_grid",
        strategy_type="grid",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out",
        strategy_cfg={"multileg_feature_selection": {"enabled": True}},
        best_calibration={},
        metrics={},
        evaluate_candidate=_eval,
    )

    selected = yaml.safe_load((cfg_dir / "features_prefilter.yaml").read_text())[
        "feature_pipeline"
    ]["requested_features"]
    assert selected == ["bpc_soft_phase_f", "atr_f"]
    assert out["selected_candidate"] == "core"
    names = {row["name"] for row in out["candidates"]}
    assert {"core", "full_default"} <= names
    assert any(name.startswith("core_plus_box_structure_f__") for name in names)
    assert out["winner"]["score"] == 1.20
    assert Path(out["artifact_path"]).exists()
    assert Path(out["html_report_path"]).exists()
    assert "Multi-leg Feature Selection Report" in Path(
        out["html_report_path"]
    ).read_text(encoding="utf-8")


def test_protected_nodes_always_preserved_in_selected_subset(tmp_path):
    cfg_dir = tmp_path / "dual_add_trend"
    cfg_dir.mkdir()
    requested = [
        "bpc_soft_phase_f",
        "trend_confidence_f",
        "atr_f",
        "box_structure_f",
    ]
    _write_feature_yaml(cfg_dir / "features.yaml", requested)

    out = select_multileg_feature_subset(
        strategy="dual_add_trend",
        strategy_type="dual_add_trend",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out",
        strategy_cfg={
            "multileg_feature_selection": {
                "enabled": True,
                "protected_nodes": ["atr_f"],
            }
        },
        best_calibration={},
        metrics={},
    )

    selected = yaml.safe_load((cfg_dir / "features.yaml").read_text())[
        "feature_pipeline"
    ]["requested_features"]
    assert selected == [
        "bpc_soft_phase_f",
        "trend_confidence_f",
        "atr_f",
    ]
    assert out["selected_candidate"] == "core"


def test_grid_does_not_implicitly_keep_box_from_calibration_flag(tmp_path):
    cfg_dir = tmp_path / "chop_grid_no_box"
    cfg_dir.mkdir()
    requested = ["bpc_soft_phase_f", "box_structure_f", "atr_f"]
    _write_feature_yaml(cfg_dir / "features.yaml", requested)

    out = select_multileg_feature_subset(
        strategy="chop_grid",
        strategy_type="grid",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out_no_box",
        strategy_cfg={"multileg_feature_selection": {"enabled": True}},
        best_calibration={"tuned_candidate": {"exclude_box_prefilter": True}},
        metrics={},
    )

    selected = yaml.safe_load((cfg_dir / "features.yaml").read_text())[
        "feature_pipeline"
    ]["requested_features"]
    assert selected == ["bpc_soft_phase_f", "atr_f"]
    assert "box_structure_f" not in out["winner"]["nodes"]


def test_tie_break_prefers_fewer_features(tmp_path):
    cfg_dir = tmp_path / "tie_break"
    cfg_dir.mkdir()
    requested = ["bpc_soft_phase_f", "box_structure_f", "atr_f"]
    _write_feature_yaml(cfg_dir / "features.yaml", requested)
    _write_feature_yaml(cfg_dir / "features_prefilter.yaml", requested)

    out = select_multileg_feature_subset(
        strategy="chop_grid",
        strategy_type="grid",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out_tie",
        strategy_cfg={"multileg_feature_selection": {"enabled": True}},
        best_calibration={},
        metrics={},
        evaluate_candidate=lambda *_args: {"score": 1.0},
    )

    assert out["selected_candidate"] == "core"
    assert out["tie_breaker"] == "higher_score_then_fewer_features_then_fewer_rules"
    assert out["winner"]["nodes"] == ["bpc_soft_phase_f", "atr_f"]


def test_core_plus_candidate_materializes_prefilter_rules(tmp_path):
    cfg_dir = tmp_path / "rule_materialize"
    (cfg_dir / "archetypes").mkdir(parents=True)
    _write_feature_yaml(
        cfg_dir / "features_prefilter.yaml",
        ["bpc_soft_phase_f", "atr_f", "atr_percentile_f"],
    )
    (cfg_dir / "semantic_polarity.yaml").write_text(
        yaml.safe_dump({"polarity": {"atr_percentile": "lower_is_better"}}),
        encoding="utf-8",
    )
    (cfg_dir / "archetypes" / "prefilter.yaml").write_text(
        yaml.safe_dump({"regime": {"entry_chop_min": 0.4, "exit_chop_below": 0.25}}),
        encoding="utf-8",
    )

    out = select_multileg_feature_subset(
        strategy="chop_grid",
        strategy_type="grid",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out_rules",
        strategy_cfg={"multileg_feature_selection": {"enabled": True}},
        best_calibration={},
        metrics={},
        evaluate_candidate=lambda name, *_args: {
            "score": (
                1.0
                if name == "core_plus_atr_percentile_f__atr_percentile_lte_0p45"
                else 0.0
            )
        },
    )

    by_name = {row["name"]: row for row in out["candidates"]}
    assert by_name["core"]["rule_count"] == 0
    assert (
        by_name["core_plus_atr_percentile_f__atr_percentile_lte_0p45"]["rule_count"]
        >= 1
    )
    adopted = yaml.safe_load((cfg_dir / "archetypes" / "prefilter.yaml").read_text())
    assert len(adopted.get("rules") or []) >= 1


def test_semantic_polarity_generates_generic_threshold_candidates(tmp_path):
    cfg_dir = tmp_path / "chop_grid"
    (cfg_dir / "archetypes").mkdir(parents=True)
    _write_feature_yaml(
        cfg_dir / "features_prefilter.yaml",
        ["bpc_soft_phase_f", "atr_f", "hurst_price_f"],
    )
    (cfg_dir / "semantic_polarity.yaml").write_text(
        yaml.safe_dump({"polarity": {"hurst_price_rolling": "lower_is_better"}}),
        encoding="utf-8",
    )
    (cfg_dir / "archetypes" / "prefilter.yaml").write_text(
        yaml.safe_dump({"rules": []}),
        encoding="utf-8",
    )

    out = select_multileg_feature_subset(
        strategy="chop_grid",
        strategy_type="grid",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out_polarity",
        strategy_cfg={
            "multileg_feature_selection": {
                "enabled": True,
                "auto_rule_thresholds": {"lower_is_better": [0.45]},
            }
        },
        best_calibration={},
        metrics={},
        evaluate_candidate=lambda *_args: {"score": 0.0},
    )

    by_name = {row["name"]: row for row in out["candidates"]}
    name = "core_plus_hurst_price_f__hurst_price_rolling_lte_0p45"
    assert name in by_name
    assert by_name[name]["rule_source"] == "semantic_polarity_threshold_scan"
    assert by_name[name]["prefilter_rules"] == [
        {"feature": "hurst_price_rolling", "operator": "<=", "value": 0.45}
    ]
