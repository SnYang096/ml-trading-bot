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
    assert cfg["rolling"]["windows"]["structure_train_window"] == "rolling_window"
    assert cfg["threshold_calibration"]["enable_model_training"] is False
    assert cfg["rolling_calibration"]["enable_model_training"] is False
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
    assert cfg["rolling"]["windows"]["calibration_months"] == 6
    assert cfg["rolling"]["windows"]["structure_lookback_months"] == 24
    assert cfg["rolling"]["windows"]["structure_train_window"] == "full_history"
    assert (cfg.get("dates") or {}).get("start_date") == "2022-01-01"
    assert "validation_months" not in cfg["dates"]
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_me_slow_full_history_from_strategy_research():
    cfg = load_pipeline_config(_root() / "config/strategies/me/research/slow.yaml")
    assert "me" in (cfg.get("strategies") or {})
    w = cfg["rolling"]["windows"]
    assert w["structure_train_window"] == "full_history"
    assert w["structure_lookback_months"] == 24
    assert (cfg.get("dates") or {}).get("start_date") == "2022-01-01"
    assert cfg["threshold_calibration"]["enable_model_training"] is True
    assert cfg["rolling_calibration"]["enable_model_training"] is True


def test_load_tpc_slow_full_history_from_strategy_research():
    cfg = load_pipeline_config(_root() / "config/strategies/tpc/research/slow.yaml")
    assert "tpc" in (cfg.get("strategies") or {})
    w = cfg["rolling"]["windows"]
    assert w["structure_train_window"] == "full_history"
    assert w["structure_lookback_months"] == 24
    assert (cfg.get("dates") or {}).get("start_date") == "2022-01-01"
    assert cfg["threshold_calibration"]["enable_model_training"] is True
    assert cfg["rolling_calibration"]["enable_model_training"] is True


@pytest.mark.parametrize(
    "research_yaml, strategy_key",
    [
        ("bpc/research/slow_recent_24.yaml", "bpc"),
        ("me/research/slow_recent_24.yaml", "me"),
        ("tpc/research/slow_recent_24.yaml", "tpc"),
    ],
)
def test_load_slow_recent_24_backup_uses_rolling_window(
    research_yaml: str, strategy_key: str
):
    cfg = load_pipeline_config(_root() / "config/strategies" / research_yaml)
    assert strategy_key in (cfg.get("strategies") or {})
    w = cfg["rolling"]["windows"]
    assert w["structure_train_window"] == "rolling_window"
    assert w["structure_lookback_months"] == 24


def test_slow_realistic_structure_lookback_must_exceed_calibration(tmp_path: Path):
    bad = tmp_path / "bad_slow.yaml"
    bad.write_text(
        "\n".join(
            [
                "dates:",
                '  start_date: "2022-01-01"',
                '  end_date: "2026-03-31"',
                "  holdout_months: 26",
                "  calibration_months: 12",
                "rolling:",
                "  mode: slow_realistic",
                "  windows:",
                "    structure_lookback_months: 12",
                "  slow_realistic:",
                "    cadence_months: 3",
                "    triggered_retrain_enabled: true",
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
    with pytest.raises(ValueError, match="structure_lookback_months"):
        load_pipeline_config(bad)


def test_slow_realistic_full_history_allows_equal_lookback_to_calibration(
    tmp_path: Path,
):
    ok = tmp_path / "ok_full_hist.yaml"
    ok.write_text(
        "\n".join(
            [
                "dates:",
                '  start_date: "2022-01-01"',
                '  end_date: "2026-03-31"',
                "  holdout_months: 26",
                "  calibration_months: 6",
                "rolling:",
                "  mode: slow_realistic",
                "  windows:",
                "    structure_train_window: full_history",
                "    structure_lookback_months: 6",
                "  slow_realistic:",
                "    cadence_months: 3",
                "    triggered_retrain_enabled: true",
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
    cfg = load_pipeline_config(ok)
    assert cfg["rolling"]["windows"]["structure_train_window"] == "full_history"
    assert cfg["rolling"]["windows"]["structure_lookback_months"] == 6


def test_structure_train_window_invalid_raises(tmp_path: Path):
    bad = tmp_path / "bad_stw.yaml"
    bad.write_text(
        "\n".join(
            [
                "dates:",
                '  start_date: "2022-01-01"',
                '  end_date: "2026-03-31"',
                "  holdout_months: 26",
                "rolling:",
                "  mode: slow_realistic",
                "  windows:",
                "    structure_train_window: bogus",
                "    structure_lookback_months: 12",
                "    calibration_months: 3",
                "  slow_realistic:",
                "    cadence_months: 3",
                "    triggered_retrain_enabled: true",
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
    with pytest.raises(ValueError, match="structure_train_window"):
        load_pipeline_config(bad)


def test_bpc_turbo_prefilter_locked_fields_are_explicit():
    """turbo：locked_threshold + 多打分方法；entry_filter 关闭 meta_algorithm（slow 覆写为 true）。"""
    cfg = load_pipeline_config(_root() / "config/strategies/bpc/research/turbo.yaml")
    pf = cfg["strategies"]["bpc"]["kpi_gates"]["prefilter"]
    assert pf["locked_threshold_tuning"]["enabled"] is True
    fb = pf.get("scoring_method_fallbacks") or []
    assert len(fb) == 4
    ef = cfg["strategies"]["bpc"]["kpi_gates"]["entry_filter"]
    assert len(ef.get("scoring_method_fallbacks") or []) == 4
    assert ef["meta_algorithm"] is False


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
    assert cfg["threshold_calibration"]["enable_model_training"] is True
    assert cfg["rolling_calibration"]["enable_model_training"] is True
    assert cfg["threshold_calibration"]["prefilter"]["optimize"] is True


def test_load_bpc_non_rolling_extends_slow():
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
    assert d.get("start_date") == "2022-01-01"
    assert (
        cfg["strategies"]["bpc"]["kpi_gates"]["entry_filter"]["meta_algorithm"] is True
    )
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_me_non_rolling_extends_slow():
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
    assert d.get("start_date") == "2022-01-01"
    assert (
        cfg["strategies"]["me"]["kpi_gates"]["entry_filter"]["meta_algorithm"] is True
    )
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_tpc_non_rolling_extends_slow():
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
    assert d.get("start_date") == "2022-01-01"
    assert (
        cfg["strategies"]["tpc"]["kpi_gates"]["entry_filter"]["meta_algorithm"] is True
    )
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_chop_grid_non_rolling_extends_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/chop_grid/research/non_rolling.yaml"
    )
    assert "chop_grid" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert cfg.get("rolling", {}).get("time_split_policy") == "static_holdout"
    assert "results/chop_grid/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    assert cfg["grid_backtest"]["enabled"] is True
    assert "non-rolling-full-cycle" in cfg["grid_backtest"]["output_dir"]


def test_load_dual_add_non_rolling_extends_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/dual_add_trend/research/non_rolling.yaml"
    )
    assert "dual_add_trend" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert cfg.get("rolling", {}).get("time_split_policy") == "static_holdout"
    assert "results/dual_add_trend/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    assert cfg["dual_add_backtest"]["enabled"] is True
    assert "non-rolling-full-cycle" in cfg["dual_add_backtest"]["output_dir"]


def test_load_multileg_slow_profiles_extend_turbo_metadata():
    chop_cfg = load_pipeline_config(
        _root() / "config/strategies/chop_grid/research/slow.yaml"
    )
    dual_cfg = load_pipeline_config(
        _root() / "config/strategies/dual_add_trend/research/slow.yaml"
    )

    assert chop_cfg.get("rolling", {}).get("mode") == "slow_realistic"
    assert "study" in chop_cfg
    assert "threshold_search" in chop_cfg
    assert dual_cfg.get("rolling", {}).get("mode") == "slow_realistic"
    assert "study" in dual_cfg
    assert "threshold_search" in dual_cfg


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
