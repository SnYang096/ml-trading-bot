"""Pipeline YAML loading (``extends`` chain) and strategy research paths."""

from pathlib import Path

import pytest

from scripts.pipeline.config import load_pipeline_config


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_bpc_turbo_from_strategy_research():
    cfg = load_pipeline_config(_root() / "config/strategies/bpc/research/turbo.yaml")
    assert "bpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "turbo_fixed_features"
    assert cfg.get("rolling", {}).get("time_split_policy") == "static_holdout"
    assert cfg["rolling"]["windows"]["calibration_months"] == 6
    assert "validation_months" not in cfg["dates"]


def test_dates_calibration_hoist_conflict_raises(tmp_path: Path):
    p = tmp_path / "x.yaml"
    p.write_text(
        "\n".join(
            [
                "dates:",
                '  start_date: "2022-01-01"',
                '  end_date: "2026-03-31"',
                "  holdout_months: 26",
                "  validation_months: 3",
                "  calibration_months: 6",
                "rolling:",
                "  mode: turbo_fixed_features",
                "  windows:",
                "    calibration_months: 5",
                "    structure_lookback_months: 12",
                "  turbo_fixed_features:",
                "    fixed_strategies_root: config/strategies",
                "threshold_calibration:",
                "  enable_model_training: false",
                "strategies:",
                "  x:",
                "    config: config/strategies/bpc",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rolling.windows.calibration_months"):
        load_pipeline_config(p)


def test_load_bpc_slow_from_strategy_research():
    cfg = load_pipeline_config(_root() / "config/strategies/bpc/research/slow.yaml")
    assert "bpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "slow_realistic"
    assert cfg["rolling"]["windows"]["calibration_months"] == 3
    assert "validation_months" not in cfg["dates"]


def test_bpc_turbo_prefilter_locked_fields_are_explicit():
    """turbo：locked_threshold_tuning 单文件内聚；prefilter/entry_filter 与 slow 同一套多打分方法。"""
    cfg = load_pipeline_config(_root() / "config/strategies/bpc/research/turbo.yaml")
    pf = cfg["strategies"]["bpc"]["kpi_gates"]["prefilter"]
    assert pf["locked_threshold_tuning"]["enabled"] is True
    fb = pf.get("scoring_method_fallbacks") or []
    assert len(fb) == 4
    ef = cfg["strategies"]["bpc"]["kpi_gates"]["entry_filter"]
    assert ef["meta_algorithm"] is True
    assert len(ef.get("scoring_method_fallbacks") or []) == 4


def test_bpc_non_rolling_inherits_locked_tune_from_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/non_rolling.yaml"
    )
    assert (
        cfg["strategies"]["bpc"]["kpi_gates"]["prefilter"]["locked_threshold_tuning"][
            "enabled"
        ]
        is True
    )


def test_bpc_turbo_prefilter_lock_fixed_disables_locked_threshold_tuning():
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/turbo_prefilter_lock_fixed.yaml"
    )
    assert (
        cfg["strategies"]["bpc"]["kpi_gates"]["prefilter"]["locked_threshold_tuning"][
            "enabled"
        ]
        is False
    )


def test_bpc_slow_overrides_locked_enabled_true():
    """slow.yaml extends turbo — strategies / threshold_calibration / rolling 深度合并。"""
    cfg = load_pipeline_config(_root() / "config/strategies/bpc/research/slow.yaml")
    kg = cfg["strategies"]["bpc"]["kpi_gates"]
    assert kg["entry_filter"]["meta_algorithm"] is True
    assert kg["entry_filter"]["archetype_plateau"] is True
    assert kg["gate"]["max_hard_gates"] == 4
    assert kg["prefilter"]["locked_threshold_tuning"]["enabled"] is True
    assert cfg["threshold_calibration"]["enable_model_training"] is False
    assert cfg["threshold_calibration"]["prefilter"]["optimize"] is True


def test_load_bpc_non_rolling_extends_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/non_rolling.yaml"
    )
    assert "bpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert "results/bpc/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    d = cfg.get("dates") or {}
    assert d.get("validation_months") == 3
    assert d.get("holdout_months") == 26
    assert d.get("start_date") == "2022-08-01"


def test_load_me_non_rolling_extends_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/me/research/non_rolling.yaml"
    )
    assert "me" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert "results/me/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    d = cfg.get("dates") or {}
    assert d.get("validation_months") == 3
    assert d.get("holdout_months") == 26
    assert d.get("start_date") == "2022-08-01"


def test_load_tpc_non_rolling_extends_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/tpc/research/non_rolling.yaml"
    )
    assert "tpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert "results/tpc/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    d = cfg.get("dates") or {}
    assert d.get("validation_months") == 3
    assert d.get("holdout_months") == 26
    assert d.get("start_date") == "2022-08-01"


def test_multileg_backtest_dates_mismatch_raises(tmp_path: Path):
    bad = tmp_path / "bad_chop.yaml"
    bad.write_text(
        "\n".join(
            [
                "dates:",
                '  start_date: "2022-01-01"',
                '  end_date: "2026-03-31"',
                "  holdout_months: 26",
                "  validation_months: 3",
                "rolling:",
                "  mode: turbo_fixed_features",
                "  windows:",
                "    calibration_months: 6",
                "    structure_lookback_months: 12",
                "  turbo_fixed_features:",
                "    fixed_strategies_root: config/strategies",
                "threshold_calibration:",
                "  enable_model_training: false",
                "grid_backtest:",
                "  enabled: true",
                '  start_date: "2020-01-01"',
                '  end_date: "2026-03-31"',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="grid_backtest.start_date"):
        load_pipeline_config(bad)


def test_load_dual_turbo_contains_study_blocks():
    cfg = load_pipeline_config(
        _root() / "config/strategies/dual_add_trend/research/turbo.yaml"
    )
    assert "dual_add_trend" in (cfg.get("strategies") or {})
    assert isinstance(cfg.get("study"), dict)
    assert isinstance(cfg.get("threshold_search"), dict)


def test_time_split_policy_invalid_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "\n".join(
            [
                "rolling:",
                "  mode: turbo_fixed_features",
                "  time_split_policy: bogus",
                "  windows:",
                "    calibration_months: 3",
                "    structure_lookback_months: 12",
                "  turbo_fixed_features:",
                "    fixed_strategies_root: config/strategies",
                "threshold_calibration:",
                "  enable_model_training: false",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="time_split_policy"):
        load_pipeline_config(bad)
