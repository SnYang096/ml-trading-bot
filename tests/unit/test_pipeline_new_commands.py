from click.testing import CliRunner

from cli import main as cli_main
from cli.main import cli


def test_pipeline_help_includes_new_commands_and_stages():
    runner = CliRunner()
    result = runner.invoke(cli, ["pipeline", "--help"])
    assert result.exit_code == 0
    assert "report-side-state" in result.output
    assert "debug-quality" in result.output

    result_run = runner.invoke(cli, ["pipeline", "run", "--help"])
    assert result_run.exit_code == 0
    assert "slow_snapshot" in result_run.output
    assert "fast_month" in result_run.output
    assert "rolling_sim" in result_run.output
    assert "--month" in result_run.output

    result_list = runner.invoke(cli, ["pipeline", "list", "--help"])
    assert result_list.exit_code == 0
    assert "include-bad-candidates" in result_list.output
    assert "list-all-profiles" in result_list.output


def test_rolling_dashboard_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["rolling-dashboard", "--help"])
    assert r.exit_code == 0
    assert "8008" in r.output


def test_pipeline_run_fast_month_passes_month_arg(monkeypatch):
    runner = CliRunner()
    called = {}

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        called["script_path"] = script_path
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)

    result = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--all",
            "--config",
            "config/pipelines/pcm_orchestrate_2h.yaml",
            "--stage",
            "fast_month",
            "--month",
            "2025-07",
        ],
    )
    assert result.exit_code == 0
    assert called["script_path"] == "scripts/auto_research_pipeline.py"
    assert "--stage" in called["args"]
    assert "fast_month" in called["args"]
    assert "--month" in called["args"]
    assert "2025-07" in called["args"]


def test_pipeline_report_side_state_command(monkeypatch):
    runner = CliRunner()
    called = {}

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        called["script_path"] = script_path
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)

    result = runner.invoke(
        cli,
        [
            "pipeline",
            "report-side-state",
            "--run-id",
            "20260326_120001",
            "--config",
            "config/pipelines/pcm_orchestrate_2h.yaml",
        ],
    )
    assert result.exit_code == 0
    assert called["script_path"] == "scripts/pipeline_report_side_state.py"
    assert called["args"] == [
        "--run-id",
        "20260326_120001",
        "--config",
        "config/pipelines/pcm_orchestrate_2h.yaml",
    ]


def test_pipeline_debug_quality_command(monkeypatch):
    runner = CliRunner()
    called = {}

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        called["script_path"] = script_path
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)

    result = runner.invoke(
        cli,
        [
            "pipeline",
            "debug-quality",
            "--run-id",
            "20260326_120001",
            "--month",
            "2025-07",
            "--config",
            "config/pipelines/pcm_orchestrate_2h.yaml",
        ],
    )
    assert result.exit_code == 0
    assert called["script_path"] == "scripts/pipeline_debug_quality.py"
    assert called["args"] == [
        "--run-id",
        "20260326_120001",
        "--month",
        "2025-07",
        "--config",
        "config/pipelines/pcm_orchestrate_2h.yaml",
    ]


def test_pipeline_list_forwards_include_bad_candidates(monkeypatch):
    runner = CliRunner()
    called = {}

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        called["script_path"] = script_path
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)

    result = runner.invoke(
        cli,
        ["pipeline", "list", "--all", "--include-bad-candidates"],
    )
    assert result.exit_code == 0
    assert called["script_path"] == "scripts/auto_research_pipeline.py"
    assert called["args"] == ["--list", "--all", "--include-bad-candidates"]


def test_pipeline_list_forwards_list_all_profiles(monkeypatch):
    runner = CliRunner()
    called = {}

    def _fake_run_script(script_path, args, docker=False, **kwargs):
        called["script_path"] = script_path
        called["args"] = list(args)
        return 0

    monkeypatch.setattr(cli_main, "run_script", _fake_run_script)

    result = runner.invoke(
        cli,
        ["pipeline", "list", "--all", "--list-all-profiles"],
    )
    assert result.exit_code == 0
    assert called["script_path"] == "scripts/auto_research_pipeline.py"
    assert called["args"] == ["--list", "--all", "--list-all-profiles"]
