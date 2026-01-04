"""Unit tests for the CLI module."""

import pytest
from click.testing import CliRunner

from cli.main import cli, get_project_root


class TestCLI:
    """Tests for the CLI main commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that CLI shows help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ML Trading Bot" in result.output
        assert "features" in result.output
        assert "train" in result.output
        assert "nnmultihead" in result.output
        assert "rule" in result.output
        assert "rl" in result.output
        assert "data" in result.output
        assert "server" in result.output
        # unified command family (optional)
        assert "search" in result.output

    def test_cli_version(self, runner):
        """Test that CLI shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.0.2" in result.output


class TestDiagnoseCommands:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_diagnose_help_includes_poolb_semantic_search(self, runner):
        result = runner.invoke(cli, ["diagnose", "--help"])
        assert result.exit_code == 0
        assert "poolb-semantic-search" in result.output
        assert "holdout-eval" in result.output


class TestFeaturesCommands:
    """Tests for the features commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_features_help(self, runner):
        """Test features group help."""
        result = runner.invoke(cli, ["features", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "count" in result.output

    def test_features_list_help(self, runner):
        """Test features list help."""
        result = runner.invoke(cli, ["features", "list", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--category" in result.output
        assert "--search" in result.output


class TestTrainCommands:
    """Tests for the train commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_train_help(self, runner):
        """Test train group help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "sr-reversal-long" in result.output
        assert "sr-reversal-short" in result.output
        assert "rolling" in result.output
        assert "final" in result.output


class TestRLCommands:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_rl_help(self, runner):
        result = runner.invoke(cli, ["rl", "--help"])
        assert result.exit_code == 0
        assert "shadow-eval-3action" in result.output
        assert "counterfactual-eval-3action" in result.output
        assert "fsm-decide" in result.output
        assert "run-e2e-3action" in result.output

    def test_train_sr_reversal_long_help(self, runner):
        """Test sr-reversal-long train help."""
        result = runner.invoke(cli, ["train", "sr-reversal-long", "--help"])
        assert result.exit_code == 0
        assert "--symbol" in result.output
        assert "--timeframe" in result.output
        # training entrypoints are config-driven via default strategy config; this command takes data path instead
        assert "--data-path" in result.output


class TestDataCommands:
    """Tests for the data commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_data_help(self, runner):
        """Test data group help."""
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output
        assert "convert" in result.output
        assert "pipeline" in result.output

    def test_data_download_help(self, runner):
        """Test data download help."""
        result = runner.invoke(cli, ["data", "download", "--help"])
        assert result.exit_code == 0
        assert "--symbols" in result.output
        assert "--start-year" in result.output


class TestDevCommands:
    """Tests for the dev commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_dev_help(self, runner):
        """Test dev group help."""
        result = runner.invoke(cli, ["dev", "--help"])
        assert result.exit_code == 0
        assert "install" in result.output
        assert "format" in result.output
        assert "lint" in result.output
        assert "clean" in result.output


class TestServeResults:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_serve_results_help(self, runner):
        # Backward-compat alias (hidden from top-level help)
        result = runner.invoke(cli, ["serve-results", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--dir" in result.output
        assert "--force" in result.output


class TestServer:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_server_help(self, runner):
        result = runner.invoke(cli, ["server", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--dir" in result.output
        assert "--force" in result.output


class TestProjectRoot:
    """Tests for project root detection."""

    def test_get_project_root(self):
        """Test that project root is correctly detected."""
        root = get_project_root()
        assert root.exists()
        assert (root / "setup.py").exists() or (root / "pyproject.toml").exists()


class TestNNMultiheadCommands:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_nnmultihead_help_includes_render_report(self, runner):
        result = runner.invoke(cli, ["nnmultihead", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "predict" in result.output
        assert "eval" in result.output
        assert "render-report" in result.output
