"""Pipeline YAML loading (``extends`` chain) and strategy research paths."""

from pathlib import Path

import pytest

from scripts.pipeline.config import load_pipeline_config, resolve_strategy_dates
from src.config.multileg_config import load_multileg_effective_config


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_bpc_turbo_from_strategy_research():
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/calibrate_roll.default.yaml"
    )
    assert "bpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "turbo_fixed_features"
    assert cfg.get("rolling", {}).get("time_split_policy") == "static_holdout"
    assert cfg["rolling"]["windows"]["calibration_months"] == 6
    assert cfg["rolling"]["windows"]["structure_train_window"] == "rolling_window"
    assert cfg["threshold_calibration"]["enable_model_training"] is False
    assert cfg["rolling_calibration"]["enable_model_training"] is False
    assert "validation_months" not in cfg["dates"]


def test_chop_grid_spacing_candidates_are_config_driven():
    cfg = load_pipeline_config(
        _root() / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
    )
    chop_cfg = cfg["strategies"]["chop_grid"]
    candidates = chop_cfg["multileg_calibration"]["candidates"]
    assert cfg["grid_backtest"]["calibrate_execution"] is True
    assert max(float(c["min_pct"]) for c in candidates) >= 0.012
    assert max(float(c["atr_mult"]) for c in candidates) >= 1.25


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
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/research_roll.features_on.yaml"
    )
    assert "bpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "slow_realistic"
    assert cfg["rolling"]["windows"]["calibration_months"] == 6
    assert cfg["rolling"]["windows"]["structure_lookback_months"] == 24
    assert cfg["rolling"]["windows"]["structure_train_window"] == "full_history"
    assert (cfg.get("dates") or {}).get("start_date") == "2022-01-01"
    assert "validation_months" not in cfg["dates"]
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_me_slow_full_history_from_strategy_research():
    cfg = load_pipeline_config(
        _root() / "config/strategies/me/research/research_roll.features_on.yaml"
    )
    assert "me" in (cfg.get("strategies") or {})
    w = cfg["rolling"]["windows"]
    assert w["structure_train_window"] == "full_history"
    assert w["structure_lookback_months"] == 24
    assert (cfg.get("dates") or {}).get("start_date") == "2022-01-01"
    assert cfg["threshold_calibration"]["enable_model_training"] is True
    assert cfg["rolling_calibration"]["enable_model_training"] is True


def test_load_tpc_slow_full_history_from_strategy_research():
    cfg = load_pipeline_config(
        _root() / "config/strategies/tpc/research/research_roll.features_on.yaml"
    )
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
        ("bpc/research/research_roll.features_on_recent24m.yaml", "bpc"),
        ("me/research/research_roll.features_on_recent24m.yaml", "me"),
        ("tpc/research/research_roll.features_on_recent24m.yaml", "tpc"),
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
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/calibrate_roll.default.yaml"
    )
    pf = cfg["strategies"]["bpc"]["kpi_gates"]["prefilter"]
    assert pf["locked_threshold_tuning"]["enabled"] is True
    fb = pf.get("scoring_method_fallbacks") or []
    assert len(fb) == 4
    ef = cfg["strategies"]["bpc"]["kpi_gates"]["entry_filter"]
    assert len(ef.get("scoring_method_fallbacks") or []) == 4
    assert ef["meta_algorithm"] is False


def test_bpc_non_rolling_inherits_locked_tune_from_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/validate_static.full_study.yaml"
    )
    assert (
        cfg["strategies"]["bpc"]["kpi_gates"]["prefilter"]["locked_threshold_tuning"][
            "enabled"
        ]
        is True
    )


def test_bpc_turbo_prefilter_lock_fixed_disables_locked_threshold_tuning():
    cfg = load_pipeline_config(
        _root()
        / "config/strategies/bpc/research/calibrate_roll.no_prefilter_threshold_search.yaml"
    )
    assert (
        cfg["strategies"]["bpc"]["kpi_gates"]["prefilter"]["locked_threshold_tuning"][
            "enabled"
        ]
        is False
    )


def test_bpc_slow_overrides_locked_enabled_true():
    """slow.yaml extends turbo — strategies / threshold_calibration / rolling 深度合并。"""
    cfg = load_pipeline_config(
        _root() / "config/strategies/bpc/research/research_roll.features_on.yaml"
    )
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
        _root() / "config/strategies/bpc/research/validate_static.full_study.yaml"
    )
    assert "bpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert "results/bpc/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    d = cfg.get("dates") or {}
    assert "validation_months" not in d
    assert d.get("nonrolling_validation_months") == 6
    assert d.get("nonrolling_test_months") == 6
    assert d.get("holdout_months") == 26
    assert d.get("start_date") == "2022-01-01"
    dates = resolve_strategy_dates(
        cfg,
        strategy="bpc",
        default_end_date=cfg["dates"]["end_date"],
    )
    assert dates["holdout_start"] == "2025-05-01"
    assert dates["validation_months"] == 6
    assert dates["test_start"] == "2025-11-01"
    assert (
        cfg["strategies"]["bpc"]["kpi_gates"]["entry_filter"]["meta_algorithm"] is True
    )
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_me_non_rolling_extends_slow():
    cfg = load_pipeline_config(
        _root() / "config/strategies/me/research/validate_static.full_study.yaml"
    )
    assert "me" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert "results/me/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    d = cfg.get("dates") or {}
    assert "validation_months" not in d
    assert d.get("nonrolling_validation_months") == 6
    assert d.get("nonrolling_test_months") == 6
    assert d.get("holdout_months") == 26
    assert d.get("start_date") == "2022-01-01"
    assert (
        cfg["strategies"]["me"]["kpi_gates"]["entry_filter"]["meta_algorithm"] is True
    )
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_tpc_non_rolling_extends_slow():
    cfg = load_pipeline_config(
        _root() / "config/strategies/tpc/research/validate_static.full_study.yaml"
    )
    assert "tpc" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert "results/tpc/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    d = cfg.get("dates") or {}
    assert "validation_months" not in d
    assert d.get("nonrolling_validation_months") == 6
    assert d.get("nonrolling_test_months") == 6
    assert d.get("holdout_months") == 26
    assert d.get("start_date") == "2022-01-01"
    assert (
        cfg["strategies"]["tpc"]["kpi_gates"]["entry_filter"]["meta_algorithm"] is True
    )
    assert (cfg.get("shap_feature_selection") or {}).get("enabled") is True


def test_load_chop_grid_non_rolling_extends_turbo():
    cfg = load_pipeline_config(
        _root() / "config/strategies/chop_grid/research/validate_static.full_study.yaml"
    )
    assert "chop_grid" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert cfg.get("rolling", {}).get("time_split_policy") == "static_holdout"
    assert "results/chop_grid/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    assert cfg["grid_backtest"]["enabled"] is True
    assert "non-rolling-full-cycle" in cfg["grid_backtest"]["output_dir"]
    dates = resolve_strategy_dates(
        cfg,
        strategy="chop_grid",
        default_end_date=cfg["dates"]["end_date"],
    )
    assert dates["holdout_start"] == "2025-05-01"
    assert dates["validation_months"] == 6
    assert dates["test_start"] == "2025-11-01"


def test_load_dual_add_non_rolling_extends_turbo():
    cfg = load_pipeline_config(
        _root()
        / "config/strategies/dual_add_trend/research/validate_static.full_study.yaml"
    )
    assert "dual_add_trend" in (cfg.get("strategies") or {})
    assert cfg.get("rolling", {}).get("mode") == "non_rolling"
    assert cfg.get("rolling", {}).get("time_split_policy") == "static_holdout"
    assert "results/dual_add_trend/non-rolling-sim" in str(
        cfg.get("output", {}).get("history_dir", "")
    )
    assert cfg["dual_add_backtest"]["enabled"] is True
    assert "non-rolling-full-cycle" in cfg["dual_add_backtest"]["output_dir"]
    assert cfg["dual_add_backtest"]["execution_timeframe"] == "1min"
    assert cfg["dual_add_backtest"]["scale_max_loser_hold_to_signal"] is True
    dates = resolve_strategy_dates(
        cfg,
        strategy="dual_add_trend",
        default_end_date=cfg["dates"]["end_date"],
    )
    assert dates["holdout_start"] == "2025-05-01"
    assert dates["validation_months"] == 6
    assert dates["test_start"] == "2025-11-01"


def test_load_multileg_slow_profiles_extend_turbo_metadata():
    chop_cfg = load_pipeline_config(
        _root() / "config/strategies/chop_grid/research/research_roll.features_on.yaml"
    )
    dual_cfg = load_pipeline_config(
        _root()
        / "config/strategies/dual_add_trend/research/research_roll.features_on.yaml"
    )

    assert chop_cfg.get("rolling", {}).get("mode") == "slow_realistic"
    assert chop_cfg["dates"]["end_date"] == "2026-05-01"
    assert "validation_months" not in chop_cfg["dates"]
    assert "study" not in chop_cfg
    assert "threshold_search" not in chop_cfg
    assert (
        chop_cfg["strategies"]["chop_grid"]["kpi_gates"]["backtest"]["min_trades"]
        == 100
    )
    assert chop_cfg["grid_backtest"]["costs"]["maker_fee_bps"] == 20.0
    assert dual_cfg.get("rolling", {}).get("mode") == "slow_realistic"
    assert dual_cfg["dates"]["end_date"] == "2026-05-01"
    assert "validation_months" not in dual_cfg["dates"]
    assert "study" not in dual_cfg
    assert "threshold_search" not in dual_cfg
    assert (
        dual_cfg["strategies"]["dual_add_trend"]["kpi_gates"]["backtest"]["min_trades"]
        == 60
    )
    assert dual_cfg["dual_add_backtest"]["costs"]["fee_bps"] == 20.0
    assert dual_cfg["dual_add_backtest"]["costs"]["market_exit_slippage_bps"] == 5.0
    assert dual_cfg["dual_add_backtest"]["costs"]["intrabar_touch_buffer_bps"] == 5.0
    assert dual_cfg["dual_add_backtest"]["execution_timeframe"] == "1min"
    assert dual_cfg["dual_add_backtest"]["scale_max_loser_hold_to_signal"] is True


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


def test_load_dual_turbo_uses_bpc_style_strategy_blocks():
    cfg = load_pipeline_config(
        _root()
        / "config/strategies/dual_add_trend/research/calibrate_roll.default.yaml"
    )
    assert "dual_add_trend" in (cfg.get("strategies") or {})
    assert "study" not in cfg
    assert "threshold_search" not in cfg
    assert isinstance(
        cfg["strategies"]["dual_add_trend"]["kpi_gates"]["backtest"], dict
    )


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


def test_trend_and_multileg_share_extends_loader_semantics(tmp_path: Path):
    trend_parent = tmp_path / "trend_parent.yaml"
    trend_child = tmp_path / "trend_child.yaml"
    trend_parent.write_text(
        "\n".join(
            [
                "dates:",
                '  start_date: "2022-01-01"',
                '  end_date: "2026-03-31"',
                "  holdout_months: 26",
                "rolling:",
                "  mode: turbo_fixed_features",
                "  windows:",
                "    calibration_months: 3",
                "    structure_lookback_months: 12",
                "  turbo_fixed_features:",
                "    fixed_strategies_root: config/strategies",
                "threshold_calibration:",
                "  enable_model_training: true",
                "strategies:",
                "  x:",
                "    config: config/strategies/bpc",
            ]
        ),
        encoding="utf-8",
    )
    trend_child.write_text(
        "\n".join(
            [
                f"extends: {trend_parent.name}",
                "rolling:",
                "  mode: non_rolling",
                "threshold_calibration:",
                "  enable_model_training: false",
            ]
        ),
        encoding="utf-8",
    )
    trend_cfg = load_pipeline_config(trend_child)
    assert trend_cfg["rolling"]["mode"] == "non_rolling"
    assert trend_cfg["threshold_calibration"]["enable_model_training"] is False

    strat_dir = tmp_path / "config/strategies/chop_grid"
    (strat_dir / "research").mkdir(parents=True, exist_ok=True)
    (strat_dir / "archetypes").mkdir(parents=True, exist_ok=True)
    (strat_dir / "research/base.yaml").write_text(
        "\n".join(
            [
                "strategy_type: grid",
                "status: research",
                "live:",
                "  mode: shadow",
            ]
        ),
        encoding="utf-8",
    )
    (strat_dir / "research/calibrate_roll.default.yaml").write_text(
        "\n".join(
            [
                "extends: base.yaml",
                "status: candidate",
                "live:",
                "  mode: dry_run",
            ]
        ),
        encoding="utf-8",
    )
    (strat_dir / "archetypes/prefilter.yaml").write_text(
        "regime:\n  entry_chop_min: 0.41\n",
        encoding="utf-8",
    )
    (strat_dir / "archetypes/execution.yaml").write_text(
        "inventory:\n  spacing:\n    atr_mult: 0.55\n",
        encoding="utf-8",
    )
    multileg_cfg = load_multileg_effective_config(
        config_dir=strat_dir,
        strategy_type="grid",
    )
    assert multileg_cfg["status"] == "candidate"
    assert multileg_cfg["live"]["mode"] == "dry_run"
    assert multileg_cfg["regime"]["entry_chop_min"] == 0.41
