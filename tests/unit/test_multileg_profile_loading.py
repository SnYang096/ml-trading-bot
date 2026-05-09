from pathlib import Path

import scripts.auto_research_pipeline as arp


def test_load_multileg_calibration_profiles_from_research_yaml(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "chop_grid"
    (cfg_dir / "research").mkdir(parents=True)
    (cfg_dir / "research" / "turbo.yaml").write_text(
        "\n".join(
            [
                "calibration_profiles:",
                "  - name: p1",
                "    entry_chop_min: 0.4",
                "  - name: p2",
                "    entry_chop_min: 0.5",
            ]
        ),
        encoding="utf-8",
    )
    rows = arp._load_multileg_calibration_profiles(cfg_dir)
    assert len(rows) == 2
    assert rows[0]["name"] == "p1"
    assert rows[1]["entry_chop_min"] == 0.5


def test_multileg_calibration_candidates_prefers_yaml_when_available(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "dual_add_trend"
    (cfg_dir / "research").mkdir(parents=True)
    (cfg_dir / "research" / "turbo.yaml").write_text(
        "\n".join(
            [
                "calibration_profiles:",
                "  - name: from_yaml",
                "    entry_min: 0.81",
            ]
        ),
        encoding="utf-8",
    )
    rows = arp._multileg_calibration_candidates("dual_add_trend", config_dir=cfg_dir)
    assert len(rows) == 1
    assert rows[0]["name"] == "from_yaml"
    assert rows[0]["entry_min"] == 0.81


def test_multileg_calibration_profiles_support_legacy_root_research_yaml(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "chop_grid"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "research.yaml").write_text(
        "calibration_profiles:\n  - name: legacy\n    entry_chop_min: 0.42\n",
        encoding="utf-8",
    )
    rows = arp._load_multileg_calibration_profiles(cfg_dir)
    assert len(rows) == 1
    assert rows[0]["name"] == "legacy"


def test_multileg_calibration_candidates_fallback_defaults() -> None:
    rows = arp._multileg_calibration_candidates("grid", config_dir=None)
    assert len(rows) >= 1
    assert "entry_chop_min" in rows[0]
