from pathlib import Path

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
