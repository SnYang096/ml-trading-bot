from pathlib import Path

import pytest

import scripts.auto_research_pipeline as arp


def test_load_pipeline_config_normalizes_fast_loop_defaults(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "rolling:",
                "  mode: turbo_fixed_features",
                "  windows:",
                "    calibration_months: 3",
                "    structure_lookback_months: 12",
                "  turbo_fixed_features:",
                "    fixed_strategies_root: config/strategies",
                "    disable_feature_search: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = arp.load_pipeline_config(cfg_path)
    fast = cfg["fast_loop"]
    assert fast["step_months"] == 1
    assert fast["threshold_calibration"]["enabled"] is True
    assert fast["prefilter"]["optimize"] is True
    assert fast["symbol_threshold_calibration"]["enabled"] is True
    assert fast["execution_opt"]["enabled"] is True
    assert fast["pcm_eval"]["enabled"] is True


def test_load_pipeline_config_preserves_fast_loop_extras(tmp_path):
    """Normalized fast_loop must not strip direction_tuning / disable_model_training."""
    cfg_path = tmp_path / "extra_fast_loop.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dates:",
                '  start_date: "2023-01-01"',
                '  end_date: "2024-01-01"',
                "  holdout_months: 6",
                "universe_group:",
                "  file: config/download/crypto_4h_token_universe_groups.yaml",
                "  universe_set: starter_a",
                "  group: highcap",
                "strategies:",
                "  bpc:",
                "    config: config/strategies/bpc",
                "    timeframe: 120T",
                "    features_gate: features.yaml",
                "    labels_gate: labels.yaml",
                "    has_prefilter: true",
                "    has_direction: true",
                "rolling:",
                "  mode: turbo_fixed_features",
                "  windows:",
                "    calibration_months: 3",
                "    structure_lookback_months: 12",
                "  turbo_fixed_features:",
                "    fixed_strategies_root: config/strategies",
                "    disable_feature_search: true",
                "fast_loop:",
                "  disable_model_training: true",
                "  direction_tuning:",
                "    enabled: true",
                "    compare_features: false",
                "    macro_epsilon_grid:",
                "      enabled: true",
                "      inner_abs_grid: '0.01'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = arp.load_pipeline_config(cfg_path)
    fl = cfg["fast_loop"]
    assert fl["disable_model_training"] is True
    assert fl["direction_tuning"]["enabled"] is True
    assert fl["direction_tuning"]["compare_features"] is False
    assert fl["direction_tuning"]["macro_epsilon_grid"]["enabled"] is True


def test_fast_month_stage_uses_fast_loop_switches(tmp_path, monkeypatch):
    captured = {"rs_calls": [], "exec_opt_calls": 0, "event_calls": 0, "pcm_calls": 0}

    def _fake_run_strategy_pipeline(strategy, cfg, **kwargs):
        captured["rs_calls"].append({"strategy": strategy, **kwargs})
        exp_cfg_dir = kwargs["run_dir"] / "strategies" / strategy
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        return {"exp_config_dir": str(exp_cfg_dir)}

    def _fake_exec_opt_only(*args, **kwargs):
        captured["exec_opt_calls"] += 1
        return {"rc": 0}

    def _fake_event_backtest(*args, **kwargs):
        captured["event_calls"] += 1
        return {"rc": 0, "metrics": {"sharpe_r": 0.0, "n_trades": 0}, "map_path": ""}

    def _fake_pcm(*args, **kwargs):
        captured["pcm_calls"] += 1
        return {}

    monkeypatch.setattr(arp, "run_strategy_pipeline", _fake_run_strategy_pipeline)
    monkeypatch.setattr(arp, "_run_event_execution_opt_only", _fake_exec_opt_only)
    monkeypatch.setattr(arp, "_run_event_backtest_step", _fake_event_backtest)
    monkeypatch.setattr(arp, "_run_pcm_joint_backtest", _fake_pcm)
    monkeypatch.setattr(arp, "resolve_symbols_from_config", lambda cfg: "BTCUSDT")

    cfg = {
        "strategies": {"bpc-long-120T": {"dates": {}}},
        "dates": {"start_date": "2023-01-01"},
        "symbol_policy": {
            "enable_threshold": 0.0,
            "min_symbol_trades_soft": 1,
            "carry_forward_hard_fail_rules": {"min_sharpe_r": -0.25},
        },
        "slot_allocation": {
            "max_symbols_per_side": 2,
            "quality_score_weights": {"history_edge": 0.55, "now_strength": 0.45},
        },
        "fast_loop": {
            "step_months": 1,
            "threshold_calibration": {"enabled": True},
            "prefilter": {"optimize": False},
            "symbol_threshold_calibration": {"enabled": True},
            "execution_opt": {"enabled": False},
            "pcm_eval": {"enabled": False},
        },
        "event_backtest": {"enabled": False},
    }

    summary = arp._run_fast_month_stage(
        cfg=cfg,
        strategies=["bpc-long-120T"],
        history_dir=tmp_path / "history",
        timestamp="20260327_000000",
        month_token="2024-03",
        dry_run=False,
        use_1min=False,
        live_root="live/highcap",
        data_path="data/parquet_data",
        event_sym_r="1.0:0.5:4.0",
        strategies_root=str(tmp_path / "base_strategies"),
        calibration_months=3,
        calibrate_all_layers=True,
        feature_search_enabled=False,
        rolling_mode="turbo_fixed_features",
        config_path="config/prod_train_pipeline_2h_turbo_2024bull.yaml",
    )

    assert len(captured["rs_calls"]) == 1
    assert captured["rs_calls"][0]["threshold_calibration_enabled"] is True
    assert captured["rs_calls"][0]["prefilter_optimization_enabled"] is False
    assert captured["exec_opt_calls"] == 0
    assert captured["event_calls"] == 0
    assert captured["pcm_calls"] == 0
    assert Path(summary["run_root"]).exists()


def test_fast_month_direction_runs_every_month_by_default(tmp_path, monkeypatch):
    """calibration_months is only the prior-window length; direction cadence defaults to 1."""
    captured: list = []

    def _fake_run_strategy_pipeline(strategy, cfg, **kwargs):
        captured.append(kwargs.get("skip_direction_tuning"))
        exp_cfg_dir = kwargs["run_dir"] / "strategies" / strategy
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        return {"exp_config_dir": str(exp_cfg_dir)}

    monkeypatch.setattr(arp, "run_strategy_pipeline", _fake_run_strategy_pipeline)
    monkeypatch.setattr(arp, "_run_event_execution_opt_only", lambda *a, **k: {"rc": 0})
    monkeypatch.setattr(
        arp,
        "_run_event_backtest_step",
        lambda *a, **k: {
            "rc": 0,
            "metrics": {"sharpe_r": 0.0, "n_trades": 0},
            "map_path": "",
        },
    )
    monkeypatch.setattr(arp, "_run_pcm_joint_backtest", lambda *a, **k: {})
    monkeypatch.setattr(arp, "resolve_symbols_from_config", lambda cfg: "BTCUSDT")

    base_cfg = {
        "strategies": {"bpc-long-120T": {"dates": {}}},
        "dates": {"start_date": "2023-01-01"},
        "symbol_policy": {
            "enable_threshold": 0.0,
            "min_symbol_trades_soft": 1,
            "carry_forward_hard_fail_rules": {"min_sharpe_r": -0.25},
        },
        "slot_allocation": {
            "max_symbols_per_side": 2,
            "quality_score_weights": {"history_edge": 0.55, "now_strength": 0.45},
        },
        "fast_loop": {
            "step_months": 1,
            "threshold_calibration": {"enabled": True},
            "prefilter": {"optimize": False},
            "symbol_threshold_calibration": {"enabled": True},
            "execution_opt": {"enabled": False},
            "pcm_eval": {"enabled": False},
            "direction_tuning": {"enabled": True},
        },
        "event_backtest": {"enabled": False},
    }

    for month_idx in (0, 1, 2):
        captured.clear()
        arp._run_fast_month_stage(
            cfg=base_cfg,
            strategies=["bpc-long-120T"],
            history_dir=tmp_path / "history",
            timestamp="20260327_000000",
            month_token="2024-03",
            dry_run=False,
            use_1min=False,
            live_root="live/highcap",
            data_path="data/parquet_data",
            event_sym_r="1.0:0.5:4.0",
            strategies_root=str(tmp_path / "base_strategies"),
            calibration_months=3,
            calibrate_all_layers=True,
            feature_search_enabled=False,
            rolling_mode="turbo_fixed_features",
            config_path="config/prod_train_pipeline_2h_turbo_2024bull.yaml",
            month_index=month_idx,
        )
        assert captured == [False], f"month_index={month_idx} should run direction"


def test_fast_month_direction_cadence_stride(tmp_path, monkeypatch):
    captured: list = []

    def _fake_run_strategy_pipeline(strategy, cfg, **kwargs):
        captured.append(kwargs.get("skip_direction_tuning"))
        exp_cfg_dir = kwargs["run_dir"] / "strategies" / strategy
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        return {"exp_config_dir": str(exp_cfg_dir)}

    monkeypatch.setattr(arp, "run_strategy_pipeline", _fake_run_strategy_pipeline)
    monkeypatch.setattr(arp, "_run_event_execution_opt_only", lambda *a, **k: {"rc": 0})
    monkeypatch.setattr(
        arp,
        "_run_event_backtest_step",
        lambda *a, **k: {
            "rc": 0,
            "metrics": {"sharpe_r": 0.0, "n_trades": 0},
            "map_path": "",
        },
    )
    monkeypatch.setattr(arp, "_run_pcm_joint_backtest", lambda *a, **k: {})
    monkeypatch.setattr(arp, "resolve_symbols_from_config", lambda cfg: "BTCUSDT")

    cfg = {
        "strategies": {"bpc-long-120T": {"dates": {}}},
        "dates": {"start_date": "2023-01-01"},
        "symbol_policy": {
            "enable_threshold": 0.0,
            "min_symbol_trades_soft": 1,
            "carry_forward_hard_fail_rules": {"min_sharpe_r": -0.25},
        },
        "slot_allocation": {
            "max_symbols_per_side": 2,
            "quality_score_weights": {"history_edge": 0.55, "now_strength": 0.45},
        },
        "fast_loop": {
            "step_months": 1,
            "threshold_calibration": {"enabled": True},
            "prefilter": {"optimize": False},
            "symbol_threshold_calibration": {"enabled": True},
            "execution_opt": {"enabled": False},
            "pcm_eval": {"enabled": False},
            "direction_tuning": {"enabled": True, "cadence_months": 3},
        },
        "event_backtest": {"enabled": False},
    }

    for month_idx, expect_skip in ((0, False), (1, True), (2, True), (3, False)):
        captured.clear()
        arp._run_fast_month_stage(
            cfg=cfg,
            strategies=["bpc-long-120T"],
            history_dir=tmp_path / "history2",
            timestamp="20260327_000001",
            month_token="2024-03",
            dry_run=False,
            use_1min=False,
            live_root="live/highcap",
            data_path="data/parquet_data",
            event_sym_r="1.0:0.5:4.0",
            strategies_root=str(tmp_path / "base_strategies2"),
            calibration_months=3,
            calibrate_all_layers=True,
            feature_search_enabled=False,
            rolling_mode="turbo_fixed_features",
            config_path="config/prod_train_pipeline_2h_turbo_2024bull.yaml",
            month_index=month_idx,
        )
        assert captured == [
            expect_skip
        ], f"month_index={month_idx} skip_direction_tuning should be {expect_skip}"


def test_load_pipeline_config_warns_when_slow_loop_present_in_slow_mode(
    tmp_path, capsys
):
    cfg_path = tmp_path / "slow_warn.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "rolling:",
                "  mode: slow_realistic",
                "  windows:",
                "    calibration_months: 3",
                "    structure_lookback_months: 12",
                "  slow_realistic:",
                "    cadence_months: 3",
                "    triggered_retrain_enabled: true",
                "slow_loop:",
                "  cadence_months: 5",
                "  triggered_retrain:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = arp.load_pipeline_config(cfg_path)
    out = capsys.readouterr().out
    assert "rolling.slow_realistic" in out
    assert cfg["rolling"]["slow_realistic"]["cadence_months"] == 3


def test_load_pipeline_config_errors_when_slow_loop_policy_error(tmp_path):
    cfg_path = tmp_path / "slow_error.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "config_contract:",
                "  slow_loop_policy: error",
                "rolling:",
                "  mode: slow_realistic",
                "  windows:",
                "    calibration_months: 3",
                "    structure_lookback_months: 12",
                "  slow_realistic:",
                "    cadence_months: 3",
                "    triggered_retrain_enabled: true",
                "slow_loop:",
                "  cadence_months: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        arp.load_pipeline_config(cfg_path)


def test_load_pipeline_config_warns_event_backtest_enabled_missing(tmp_path, capsys):
    cfg_path = tmp_path / "ev_default.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "rolling:",
                "  mode: turbo_fixed_features",
                "  windows:",
                "    calibration_months: 3",
                "    structure_lookback_months: 12",
                "  turbo_fixed_features:",
                "    fixed_strategies_root: config/strategies",
                "    disable_feature_search: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = arp.load_pipeline_config(cfg_path)
    out = capsys.readouterr().out
    assert "event_backtest.enabled" in out
    assert cfg["event_backtest"]["enabled"] is True


def test_filter_strategies_scope_prefers_explicit_side():
    cfg = {"strategies": {"bpc": {"side": "long"}, "fer": {"side": "short"}}}
    got_long = arp._filter_strategies_by_direction_scope(["bpc", "fer"], "long", cfg)
    got_short = arp._filter_strategies_by_direction_scope(["bpc", "fer"], "short", cfg)
    assert got_long == ["bpc"]
    assert got_short == ["fer"]
