import json
from pathlib import Path

import scripts.auto_research_pipeline as arp
from scripts.event_backtest import BacktestResult, _save_json


def test_fast_month_stage_threads_resume_and_dump_paths(tmp_path, monkeypatch):
    calls = []

    def _fake_event_backtest_step(strategy, evidence_dir, run_dir, **kwargs):
        calls.append(
            {
                "strategy": strategy,
                "resume_state_path": kwargs.get("resume_state_path", ""),
                "dump_end_state_path": kwargs.get("dump_end_state_path", ""),
                "keep_open_positions": kwargs.get("keep_open_positions", False),
            }
        )
        Path(kwargs.get("dump_end_state_path", "")).write_text("{}", encoding="utf-8")
        return {
            "rc": 0,
            "metrics": {"sharpe_r": 0.6, "n_trades": 25},
            "map_path": str(run_dir / "map.html"),
            "end_state_path": kwargs.get("dump_end_state_path", ""),
        }

    def _fake_pcm(*args, **kwargs):
        return {"total_r": 1.2, "total_trades": 12}

    monkeypatch.setattr(arp, "_run_event_backtest_step", _fake_event_backtest_step)
    monkeypatch.setattr(arp, "_run_pcm_joint_backtest", _fake_pcm)

    cfg = {
        "strategies": {"fer-short-120T": {}, "me-short-120T": {}},
        "symbol_policy": {"enable_threshold": 0.0, "min_symbol_trades_soft": 1},
        "slot_allocation": {
            "max_symbols_per_side": 2,
            "quality_score_weights": {"history_edge": 0.55, "now_strength": 0.45},
        },
    }
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    summary = arp._run_fast_month_stage(
        cfg=cfg,
        strategies=["fer-short-120T", "me-short-120T"],
        history_dir=history_dir,
        timestamp="20260315_120000",
        month_token="2025-07",
        dry_run=False,
        use_1min=False,
        live_root="live/highcap",
        data_path="data/parquet_data",
        event_sym_r="1.0:0.5:4.0",
        prev_side_state={},
        prev_resume_state_paths={"fer-short-120T": "/tmp/prev_fer_state.json"},
        keep_open_positions=True,
    )

    assert len(calls) == 2
    first = next(c for c in calls if c["strategy"] == "fer-short-120T")
    assert first["resume_state_path"] == "/tmp/prev_fer_state.json"
    assert first["keep_open_positions"] is True
    assert first["dump_end_state_path"].endswith("/fer-short-120T/end_state.json")
    assert summary["end_state_paths"]["fer-short-120T"].endswith(
        "/fer-short-120T/end_state.json"
    )
    assert summary["end_state_paths"]["me-short-120T"].endswith(
        "/me-short-120T/end_state.json"
    )


def test_event_backtest_json_includes_open_positions_and_add_stats(tmp_path):
    result = BacktestResult(strategy="demo")
    result.add_position_stats = {"max_observed_leverage": 5.0, "add_count": 2}
    result.open_positions_end = [
        {
            "symbol": "BTCUSDT",
            "pid": "abc123",
            "position": {"entry_time": "2025-07-01T00:00:00+00:00"},
        }
    ]

    out_path = tmp_path / "event_result.json"
    _save_json(result, str(out_path))

    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["add_position_stats"]["max_observed_leverage"] == 5.0
    assert len(obj["open_positions_end"]) == 1
    assert obj["open_positions_end"][0]["symbol"] == "BTCUSDT"
