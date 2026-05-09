import scripts.auto_research_pipeline as arp


def test_grid_calibration_candidates_are_strategy_owned() -> None:
    rows = arp._multileg_calibration_candidates("grid", config_dir=None)
    assert len(rows) >= 1
    assert "entry_chop_min" in rows[0]
    assert all("box_window" not in row for row in rows)


def test_dual_add_calibration_candidates_are_strategy_owned() -> None:
    rows = arp._multileg_calibration_candidates("dual_add_trend", config_dir=None)
    assert len(rows) >= 1
    assert rows[0]["entry_min"] == 0.75
    assert rows[0]["step_atr_mult"] == 0.75
    assert all("box_window" not in row for row in rows)
