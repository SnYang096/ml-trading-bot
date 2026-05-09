from click.testing import CliRunner

from cli import main as cli_main
from cli.main import cli


def test_multileg_validate_forwards_script(monkeypatch):
    runner = CliRunner()
    called = {}

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        called["script_path"] = script_path
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)
    res = runner.invoke(
        cli,
        [
            "multileg",
            "validate-config",
            "--config",
            "config/pipelines/multileg_orchestrate_2h.yaml",
            "--constitution-yaml",
            "config/constitution/constitution.yaml",
        ],
    )
    assert res.exit_code == 0
    assert called["script_path"] == "scripts/multileg_validate_config.py"
    assert "--config" in called["args"]
    assert "--constitution-yaml" in called["args"]


def test_multileg_replay_forwards_args(monkeypatch):
    runner = CliRunner()
    called = {}

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        called["script_path"] = script_path
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)
    res = runner.invoke(
        cli,
        [
            "multileg",
            "replay",
            "--all",
            "--months",
            "2024-01:2024-03",
            "--use-1min",
        ],
    )
    assert res.exit_code == 0
    assert called["script_path"] == "scripts/multileg_replay.py"
    assert "--all" in called["args"]
    assert "--months" in called["args"]
    assert "2024-01:2024-03" in called["args"]
    assert "--use-1min" in called["args"]


def test_multileg_gate_monitor_shadow_live_forward(monkeypatch):
    runner = CliRunner()
    calls = []

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        calls.append((script_path, list(args)))
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)

    r_gate = runner.invoke(
        cli,
        [
            "multileg",
            "gate",
            "--run-dir",
            "results/multi_leg/rolling-sim/_rolling_sim/20260101_000000",
        ],
    )
    assert r_gate.exit_code == 0

    r_monitor = runner.invoke(
        cli,
        ["multileg", "monitor", "--run-id", "20260101_000000"],
    )
    assert r_monitor.exit_code == 0

    r_shadow = runner.invoke(
        cli,
        ["multileg", "shadow", "--once"],
    )
    assert r_shadow.exit_code == 0

    r_live = runner.invoke(
        cli,
        ["multileg", "live", "--mode", "testnet"],
    )
    assert r_live.exit_code == 0

    assert calls[0][0] == "scripts/multileg_gate.py"
    assert calls[1][0] == "scripts/multileg_monitor.py"
    assert calls[2][0] == "scripts/run_multi_leg_live.py"
    assert calls[3][0] == "scripts/run_multi_leg_live.py"
