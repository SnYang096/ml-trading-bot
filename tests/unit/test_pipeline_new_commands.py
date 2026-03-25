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
            "config/prod_train_pipeline_2h.yaml",
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
            "config/prod_train_pipeline_2h.yaml",
        ],
    )
    assert result.exit_code == 0
    assert called["script_path"] == "scripts/pipeline_report_side_state.py"
    assert called["args"] == [
        "--run-id",
        "20260326_120001",
        "--config",
        "config/prod_train_pipeline_2h.yaml",
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
            "config/prod_train_pipeline_2h.yaml",
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
        "config/prod_train_pipeline_2h.yaml",
    ]
