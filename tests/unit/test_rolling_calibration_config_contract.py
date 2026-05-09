from pathlib import Path
import json

import pytest

import scripts.auto_research_pipeline as arp
from scripts.capital_report import write_capital_report_from_trades


def test_load_pipeline_config_normalizes_rolling_calibration_defaults(tmp_path):
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
    fast = cfg["rolling_calibration"]
    assert fast["step_months"] == 1
    assert fast["threshold_calibration"]["enabled"] is True
    assert fast["prefilter"]["optimize"] is True
    assert fast["symbol_threshold_calibration"]["enabled"] is True
    assert fast["execution_opt"]["enabled"] is True
    assert fast["pcm_eval"]["enabled"] is True


def test_load_pipeline_config_preserves_rolling_calibration_extras(tmp_path):
    """Normalized rolling_calibration must not strip direction_tuning / enable_model_training overrides."""
    cfg_path = tmp_path / "extra_rolling_calibration.yaml"
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
                "    prepare_features: features.yaml",
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
                "rolling_calibration:",
                "  enable_model_training: false",
                "  direction_tuning:",
                "    enabled: true",
                "    compare_features: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = arp.load_pipeline_config(cfg_path)
    fl = cfg["rolling_calibration"]
    assert fl["enable_model_training"] is False
    assert fl["direction_tuning"]["enabled"] is True
    assert fl["direction_tuning"]["compare_features"] is False


def test_fast_month_stage_respects_rolling_calibration_switches(tmp_path, monkeypatch):
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
        "rolling_calibration": {
            "step_months": 1,
            "threshold_calibration": {"enabled": True},
            "prefilter": {"optimize": False},
            "symbol_threshold_calibration": {"enabled": True},
            "execution_opt": {"enabled": False},
            "pcm_eval": {"enabled": False},
        },
        "threshold_calibration": {"enable_model_training": False},
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
        config_path="config/strategies/bpc/research/turbo.yaml",
    )

    assert len(captured["rs_calls"]) == 1
    assert captured["rs_calls"][0]["threshold_calibration_enabled"] is True
    assert captured["rs_calls"][0]["enable_model_training"] is False
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
        "rolling_calibration": {
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
            config_path="config/strategies/bpc/research/turbo.yaml",
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
        "rolling_calibration": {
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
            config_path="config/strategies/bpc/research/turbo.yaml",
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


def test_capital_report_r_multiple_explains_money_assumption(tmp_path):
    trades = tmp_path / "event_trades.csv"
    trades.write_text(
        "exit_time,pnl_r\n"
        "2024-01-01T00:00:00+00:00,2.0\n"
        "2024-01-02T00:00:00+00:00,-1.0\n",
        encoding="utf-8",
    )

    report = write_capital_report_from_trades(
        trades_path=trades,
        out_dir=tmp_path,
        unit="r_multiple",
        title="Event Capital Report",
        initial_capital=10000.0,
        risk_per_r=0.01,
        start_date="2024-01-01",
        end_date="2025-01-01",
    )

    assert report["final_capital"] == 10100.0
    assert report["total_r"] == 1.0
    assert "raw sum(pnl_r)" in report["unit_explanation"]
    assert (tmp_path / "capital_report.html").exists()


def test_capital_report_empty_trades_file_is_zero_trade_run(tmp_path):
    trades = tmp_path / "event_trades.csv"
    trades.write_text("", encoding="utf-8")

    report = write_capital_report_from_trades(
        trades_path=trades,
        out_dir=tmp_path / "out",
        unit="r_multiple",
        title="Zero trades",
        initial_capital=10000.0,
        risk_per_r=0.01,
    )

    assert report["trades"] == 0
    assert report["reason"] == "no trades"
    assert report["final_capital"] == 10000.0
    assert (tmp_path / "out" / "capital_report.json").exists()


def _write_dual_add_strategy(root: Path) -> None:
    strat = root / "dual_add_trend"
    strat.mkdir(parents=True, exist_ok=True)
    (strat / "research").mkdir(parents=True, exist_ok=True)
    (strat / "meta.yaml").write_text(
        "strategy:\n  timeframe: '120T'\n  bidirectional: true\n", encoding="utf-8"
    )
    (strat / "research" / "turbo.yaml").write_text(
        "\n".join(
            [
                "strategy_type: dual_add_trend",
                "regime:",
                "  entry_min: 0.80",
                "  exit_below: 0.50",
                "  max_semantic_chop_entry: 0.25",
                "  max_semantic_chop_hold: 0.40",
                "  exclude_box_prefilter: true",
                "inventory:",
                "  add_mode: trend",
                "  flip_action: close_offside_all",
                "  max_adds_per_side: 3",
                "  max_gross_exposure_units: 4",
                "  max_net_exposure_units: 2",
                "  max_loser_hold_bars: 24",
                "add_spacing:",
                "  atr_mult: 0.50",
                "take_profit:",
                "  atr_mult: 0.25",
                "  min_pct: 0.0005",
                "  min_abs: 0.0",
                "risk:",
                "  min_segment_bars: 6",
                "  max_segment_bars: 120",
                "  max_loss_per_segment: 0.01",
                "  diagnostic_fee_bps: 4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (strat / "archetypes").mkdir(parents=True, exist_ok=True)
    (strat / "archetypes" / "prefilter.yaml").write_text(
        "regime: {}\n", encoding="utf-8"
    )
    (strat / "archetypes" / "execution.yaml").write_text(
        "inventory: {}\nadd_spacing: {}\ntake_profit: {}\nrisk: {}\n",
        encoding="utf-8",
    )


def _fake_dual_add_subprocess(cmd, cwd=None, capture_output=True, text=True):
    from types import SimpleNamespace

    out_dir = Path(cmd[cmd.index("--out-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.csv").write_text(
        "\n".join(
            [
                "segments,trades,trade_win_rate,segment_win_rate,sum_pnl_per_capital,worst_segment,median_drawdown,risk_stop_rate,max_gross_units,max_abs_net_units,loser_timeout_rate,tp_rate,forced_rate",
                "2,5,0.8,0.5,0.12,-0.01,-0.001,0.0,4,2,0.0,0.8,0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "dual_add_trades.csv").write_text(
        "symbol,side,entry_time,exit_time,entry_price,exit_price,pnl_pct,exit_reason\n"
        "BTCUSDT,LONG,2024-04-01 00:00:00+00:00,2024-04-01 02:00:00+00:00,1,2,1,tp\n",
        encoding="utf-8",
    )
    (out_dir / "dual_add_segments.csv").write_text(
        "symbol,start,end,direction,pnl_per_capital\n"
        "BTCUSDT,2024-04-01 00:00:00+00:00,2024-04-01 02:00:00+00:00,UP,0.1\n",
        encoding="utf-8",
    )
    return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def test_fast_month_multileg_uses_backtest_adapter(tmp_path, monkeypatch):
    base_root = tmp_path / "strategies"
    _write_dual_add_strategy(base_root)
    captured = {"event_calls": 0, "strategy_calls": 0, "commands": []}

    def _fail_strategy_pipeline(*args, **kwargs):
        captured["strategy_calls"] += 1
        raise AssertionError("multi-leg strategy should not enter TradeIntent pipeline")

    def _fail_event(*args, **kwargs):
        captured["event_calls"] += 1
        raise AssertionError("multi-leg strategy should not enter event_backtest")

    def _fake_run(cmd, cwd=None, capture_output=True, text=True):
        captured["commands"].append(list(cmd))
        return _fake_dual_add_subprocess(cmd, cwd, capture_output, text)

    monkeypatch.setattr(
        arp.pipeline_strategy, "run_strategy_pipeline", _fail_strategy_pipeline
    )
    monkeypatch.setattr(arp.pipeline_events, "run_event_backtest_step", _fail_event)
    monkeypatch.setattr(arp.subprocess, "run", _fake_run)
    monkeypatch.setattr(arp, "resolve_symbols_from_config", lambda cfg: "BTCUSDT")

    cfg = {
        "strategies": {"dual_add_trend": {"strategy_type": "dual_add_trend"}},
        "dates": {"start_date": "2023-01-01", "end_date": "2024-04-30"},
        "rolling_calibration": {
            "threshold_calibration": {"enabled": True},
            "execution_opt": {"enabled": True},
            "pcm_eval": {"enabled": True},
        },
        "event_backtest": {"enabled": True},
        "dual_add_backtest": {"symbols": "BTCUSDT", "exclude_box": True},
    }

    summary = arp._run_fast_month_stage(
        cfg=cfg,
        strategies=["dual_add_trend"],
        history_dir=tmp_path / "history",
        timestamp="20260426_000000",
        month_token="2024-04",
        dry_run=False,
        use_1min=False,
        live_root="live/highcap",
        data_path="data/parquet_data",
        event_sym_r="1.0:0.5:4.0",
        strategies_root=str(base_root),
        calibration_months=3,
        calibrate_all_layers=True,
        feature_search_enabled=False,
        rolling_mode="turbo_fixed_features",
        config_path="config/test.yaml",
    )

    run_root = Path(summary["run_root"])
    assert captured["strategy_calls"] == 0
    assert captured["event_calls"] == 0
    assert captured["commands"]
    assert (
        run_root / "strategies_calibrated/dual_add_trend/research/turbo.yaml"
    ).exists()
    assert (run_root / "dual_add_trend/multileg_summary.json").exists()
    assert summary["trend_pcm_candidates"] == []
    assert summary["multi_leg_pcm_candidates"] == ["dual_add_trend"]
    assert Path(summary["multi_leg_pcm_path"]).exists()
    qobj = json.loads(Path(summary["pcm_candidates_path"]).read_text(encoding="utf-8"))
    rows = qobj.get("candidates", [])
    assert len(rows) == 1
    assert rows[0].get("trend_pcm_candidate") is False
    assert rows[0].get("multi_leg_pcm_candidate") is True


def test_slow_snapshot_multileg_writes_snapshot_config(tmp_path, monkeypatch):
    base_root = tmp_path / "strategies"
    _write_dual_add_strategy(base_root)
    monkeypatch.setattr(arp.subprocess, "run", _fake_dual_add_subprocess)
    monkeypatch.setattr(arp, "resolve_symbols_from_config", lambda cfg: "BTCUSDT")

    cfg = {
        "strategies": {"dual_add_trend": {"strategy_type": "dual_add_trend"}},
        "dates": {"start_date": "2023-01-01", "end_date": "2024-04-30"},
        "dual_add_backtest": {"symbols": "BTCUSDT", "exclude_box": True},
    }

    result = arp._run_slow_structure_snapshot_for_month(
        cfg=cfg,
        strategies=["dual_add_trend"],
        history_dir=tmp_path / "history",
        timestamp="20260426_000001",
        month_token="2024-04",
        data_path="data/parquet_data",
        dry_run=False,
        use_1min=False,
        live_root="live/highcap",
        lookback_months=6,
        structure_holdout_months=3,
        source_strategies_root=str(base_root),
        config_path="config/test.yaml",
    )

    snap_root = Path(result["snapshot_root"])
    assert (snap_root / "strategies/dual_add_trend/research/turbo.yaml").exists()
    assert (snap_root / "slow_snapshot_manifest.json").exists()


def test_slow_snapshot_full_history_uses_dates_start(tmp_path, monkeypatch):
    import json

    captured: dict = {}

    def _fake_run(strategy, cfg, **kwargs):
        captured["start_date"] = kwargs.get("start_date")
        rd = kwargs.get("run_dir")
        if rd is not None:
            ed = Path(rd) / "exp_out"
            ed.mkdir(parents=True, exist_ok=True)
            return {"exp_config_dir": str(ed)}
        return {"exp_config_dir": str(tmp_path / "fallback")}

    monkeypatch.setattr(arp.pipeline_strategy, "run_strategy_pipeline", _fake_run)
    monkeypatch.setattr(arp, "resolve_symbols_from_config", lambda cfg: "BTCUSDT")

    cfg = {
        "dates": {"start_date": "2021-06-15"},
        "symbols": "BTCUSDT",
        "strategies": {"bpc": {}},
        "rolling": {"windows": {"structure_train_window": "full_history"}},
    }
    result = arp._run_slow_structure_snapshot_for_month(
        cfg=cfg,
        strategies=["bpc"],
        history_dir=tmp_path / "history",
        timestamp="fh_001",
        month_token="2024-04",
        data_path="data/parquet_data",
        dry_run=False,
        use_1min=False,
        live_root="live/highcap",
        lookback_months=6,
        structure_holdout_months=3,
        source_strategies_root=str(tmp_path / "src_strat"),
        config_path="config/test.yaml",
    )
    assert captured.get("start_date") == "2021-06-15"
    manifest = json.loads(
        (Path(result["snapshot_root"]) / "slow_snapshot_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest.get("structure_train_window") == "full_history"
    assert manifest.get("structure_start") == "2021-06-15"


def test_multileg_rolling_continuous_map_collects_monthly_artifacts(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "roll/fast_month_2024-04"
    strat_dir = run_root / "dual_add_trend"
    strat_dir.mkdir(parents=True)
    empty_root = tmp_path / "roll/fast_month_2024-05"
    empty_strat_dir = empty_root / "dual_add_trend"
    empty_strat_dir.mkdir(parents=True)
    (empty_strat_dir / "dual_add_trades.csv").write_text("", encoding="utf-8")
    (empty_strat_dir / "dual_add_segments.csv").write_text("", encoding="utf-8")
    (strat_dir / "dual_add_trades.csv").write_text(
        "symbol,side,entry_time,exit_time,entry_price,exit_price,pnl_pct,exit_reason\n"
        "BTCUSDT,LONG,2024-04-01 00:00:00+00:00,2024-04-01 02:00:00+00:00,1,2,1,tp\n",
        encoding="utf-8",
    )
    (strat_dir / "dual_add_segments.csv").write_text(
        "symbol,start,end,direction,pnl_per_capital\n"
        "BTCUSDT,2024-04-01 00:00:00+00:00,2024-04-01 02:00:00+00:00,UP,0.1\n",
        encoding="utf-8",
    )

    def _fake_write_map(**kwargs):
        kwargs["out_path"].write_text("<html>map</html>", encoding="utf-8")

    monkeypatch.setattr(arp, "write_continuous_trading_map", _fake_write_map)
    monkeypatch.setattr(arp, "resolve_symbols_from_config", lambda cfg: "BTCUSDT")
    cfg = {
        "strategies": {"dual_add_trend": {"strategy_type": "dual_add_trend"}},
        "dates": {"start_date": "2024-04-01", "end_date": "2024-04-30"},
    }
    out = tmp_path / "roll/trading_map_continuous.html"

    got = arp._build_multileg_rolling_continuous_map(
        cfg=cfg,
        ledger=[{"run_root": str(run_root)}, {"run_root": str(empty_root)}],
        strategies=["dual_add_trend"],
        roll_root=tmp_path / "roll",
        data_path="data/parquet_data",
        output_path=out,
    )

    assert got == str(out)
    assert out.exists()


def test_build_multi_leg_pcm_artifact_rejects_same_symbol_overlap(tmp_path):
    run_root = tmp_path / "roll/fast_month_2024-04"
    cg = run_root / "chop_grid"
    da = run_root / "dual_add_trend"
    cg.mkdir(parents=True)
    da.mkdir(parents=True)
    (cg / "grid_trades.csv").write_text(
        "symbol,entry_time,exit_time,entry_price,exit_price,pnl_per_capital\n"
        "BTCUSDT,2024-04-01 00:00:00+00:00,2024-04-01 03:00:00+00:00,1,1.1,0.1\n",
        encoding="utf-8",
    )
    (da / "dual_add_trades.csv").write_text(
        "symbol,entry_time,exit_time,entry_price,exit_price,pnl_per_capital\n"
        "BTCUSDT,2024-04-01 01:00:00+00:00,2024-04-01 04:00:00+00:00,1,1.2,0.2\n",
        encoding="utf-8",
    )
    cfg = {
        "strategies": {
            "chop_grid": {"strategy_type": "grid"},
            "dual_add_trend": {"strategy_type": "dual_add_trend"},
        }
    }
    with pytest.raises(ValueError, match="multi-leg same-symbol overlap conflict"):
        arp._build_multi_leg_pcm_artifact(
            cfg=cfg,
            run_root=run_root,
            month_token="2024-04",
            strategies=["chop_grid", "dual_add_trend"],
        )


def test_multileg_standalone_backtest_out_root_defaults_under_history_dir(tmp_path):
    history = tmp_path / "results/chop_grid/turbo-rolling-sim"
    history.mkdir(parents=True)
    got = arp._multileg_standalone_backtest_out_root(
        history_dir=history,
        timestamp="20260427_120000",
        section={},
        nest_dirname="grid_full_window",
    )
    assert got == history / "_rolling_sim/20260427_120000/grid_full_window"
    assert got.is_dir()


def test_multileg_standalone_backtest_out_root_respects_explicit_output_dir(tmp_path):
    history = tmp_path / "hist"
    history.mkdir()
    custom = tmp_path / "custom_out"
    got = arp._multileg_standalone_backtest_out_root(
        history_dir=history,
        timestamp="ts",
        section={"output_dir": str(custom)},
        nest_dirname="ignored",
    )
    assert got.resolve() == custom.resolve()


def test_slow_adoption_gate_soft_undertrade_allows_better_low_frequency_candidate():
    gate_cfg = {
        "adopt_ratio": 1.0,
        "min_improvement": 0.0,
        "cash_score_when_no_trades": 0.0,
        "score_weights": {
            "sharpe_r": 1.0,
            "max_drawdown_r": 0.15,
            "near_stop_rate": 0.5,
        },
        "undertrade": {"target_trades_soft": 10, "penalty": 0.15},
        "hard_reject": {"sharpe_floor": -0.5, "max_drawdown_r": None},
    }
    old_metrics = {
        "n_trades": 18,
        "sharpe_r": 0.35,
        "max_drawdown_r": 2.2,
        "near_stop_rate": 0.14,
    }
    # 低频但质量更高：应允许采纳（而不是被 min_trades 硬门槛拒绝）
    new_metrics = {
        "n_trades": 5,
        "sharpe_r": 0.82,
        "max_drawdown_r": 1.1,
        "near_stop_rate": 0.04,
    }

    d = arp._decide_slow_adoption_for_strategy(old_metrics, new_metrics, gate_cfg)
    assert d["adopt"] is True
    assert d["new"]["undertrade_penalty"] > 0.0
    assert d["new"]["score"] > d["old"]["score"]


def test_slow_adoption_gate_cash_baseline_can_beat_losing_old_structure():
    gate_cfg = {
        "adopt_ratio": 1.0,
        "min_improvement": 0.0,
        "cash_score_when_no_trades": 0.0,
        "score_weights": {
            "sharpe_r": 1.0,
            "max_drawdown_r": 0.15,
            "near_stop_rate": 0.5,
        },
        "undertrade": {"target_trades_soft": 10, "penalty": 0.15},
        "hard_reject": {"sharpe_floor": -0.5, "max_drawdown_r": None},
    }
    old_metrics = {
        "n_trades": 14,
        "sharpe_r": -0.45,
        "max_drawdown_r": 3.8,
        "near_stop_rate": 0.28,
    }
    # 新结构本月不交易，按现金基准分处理
    new_metrics = {"n_trades": 0}

    d = arp._decide_slow_adoption_for_strategy(old_metrics, new_metrics, gate_cfg)
    assert d["adopt"] is True
    assert d["new"]["score"] == 0.0
    assert d["old"]["score"] < 0.0


def test_slow_adoption_gate_hard_reject_blocks_bad_candidate():
    gate_cfg = {
        "adopt_ratio": 1.0,
        "min_improvement": 0.0,
        "cash_score_when_no_trades": 0.0,
        "score_weights": {
            "sharpe_r": 1.0,
            "max_drawdown_r": 0.15,
            "near_stop_rate": 0.5,
        },
        "undertrade": {"target_trades_soft": 10, "penalty": 0.15},
        "hard_reject": {"sharpe_floor": -0.5, "max_drawdown_r": None},
    }
    old_metrics = {
        "n_trades": 12,
        "sharpe_r": 0.42,
        "max_drawdown_r": 1.5,
        "near_stop_rate": 0.10,
    }
    new_metrics = {
        "n_trades": 9,
        "sharpe_r": -0.8,
        "max_drawdown_r": 1.0,
        "near_stop_rate": 0.09,
    }

    d = arp._decide_slow_adoption_for_strategy(old_metrics, new_metrics, gate_cfg)
    assert d["adopt"] is False
    assert "sharpe_floor" in d["reason"]
