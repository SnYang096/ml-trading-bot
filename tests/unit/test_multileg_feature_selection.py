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


def test_grid_feature_selection_drops_unused_box_when_candidate_allows_boxes(tmp_path):
    cfg_dir = tmp_path / "chop_grid"
    cfg_dir.mkdir()
    requested = ["bpc_soft_phase_f", "box_structure_f", "atr_f", "ema_1200_slope_f"]
    _write_feature_yaml(cfg_dir / "features.yaml", requested)
    _write_feature_yaml(cfg_dir / "features_prefilter.yaml", requested)

    out = select_multileg_feature_subset(
        strategy="chop_grid",
        strategy_type="grid",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out",
        strategy_cfg={"multileg_feature_selection": {"enabled": True}},
        best_calibration={
            "tuned_candidate": {"exclude_box_prefilter": False},
        },
        metrics={"n_trades": 10},
    )

    selected = yaml.safe_load((cfg_dir / "features_prefilter.yaml").read_text())[
        "feature_pipeline"
    ]["requested_features"]
    assert selected == ["bpc_soft_phase_f", "atr_f"]
    assert out["files"][0]["removed"] == ["box_structure_f", "ema_1200_slope_f"]
    assert Path(out["artifact_path"]).exists()


def test_dual_add_feature_selection_keeps_trend_chop_box_and_atr(tmp_path):
    cfg_dir = tmp_path / "dual_add_trend"
    cfg_dir.mkdir()
    requested = [
        "bpc_soft_phase_f",
        "box_structure_f",
        "trend_confidence_f",
        "ema_1200_position_f",
        "atr_f",
    ]
    _write_feature_yaml(cfg_dir / "features.yaml", requested)

    select_multileg_feature_subset(
        strategy="dual_add_trend",
        strategy_type="dual_add_trend",
        config_dir=cfg_dir,
        output_dir=tmp_path / "out",
        strategy_cfg={"multileg_feature_selection": {"enabled": True}},
        best_calibration={"tuned_candidate": {"entry_min": 0.75}},
        metrics={"n_trades": 20},
    )

    selected = yaml.safe_load((cfg_dir / "features.yaml").read_text())[
        "feature_pipeline"
    ]["requested_features"]
    assert selected == [
        "bpc_soft_phase_f",
        "box_structure_f",
        "trend_confidence_f",
        "atr_f",
    ]
