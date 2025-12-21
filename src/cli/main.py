"""
ML Trading Bot CLI - Unified command-line interface.

This CLI replaces the Makefile for cross-platform compatibility.

Usage:
    mlbot --help                    # Show all commands
    mlbot features list             # List registered features
    mlbot train sr-reversal         # Train SR reversal model
    mlbot data download             # Download Binance data
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import click


# =============================================================================
# Project root detection
# =============================================================================


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for setup.py or pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def _is_in_docker() -> bool:
    """Check if running inside a Docker container."""
    # Check multiple indicators
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("DEV_CONTAINER") == "1":
        return True
    if os.environ.get("DOCKER_CONTAINER") == "1":
        return True
    # Check if /workspace exists (typical Docker mount point in our setup)
    if os.path.exists("/workspace") and os.path.isdir("/workspace"):
        # Additional check: if we're in /workspace and it's a mount point
        try:
            import stat

            workspace_stat = os.stat("/workspace")
            # If /workspace is a mount point, we're likely in Docker
            if os.path.exists("/proc/mounts"):
                with open("/proc/mounts", "r") as f:
                    mounts = f.read()
                    if "/workspace" in mounts:
                        return True
        except (IOError, OSError):
            pass
    # Check cgroup (may not exist in all environments)
    try:
        if os.path.exists("/proc/self/cgroup"):
            with open("/proc/self/cgroup", "r") as f:
                content = f.read()
                if "docker" in content or "containerd" in content:
                    return True
    except (IOError, OSError):
        pass
    return False


def run_python_module(module: str, args: List[str], docker: bool = False, **kwargs):
    """Run a Python module with optional Docker wrapper."""
    if docker and not _is_in_docker():
        # Build Docker command
        docker_image = os.environ.get(
            "DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.7"
        )
        cmd = [
            "docker",
            "run",
            "--rm",
            "-it",
            "--gpus",
            "all",
            "-e",
            "PYTHONPATH=/workspace/src",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-v",
            f"{PROJECT_ROOT}:/workspace",
            "-w",
            "/workspace",
            "--shm-size=8gb",
            docker_image,
            "python3",
            "-m",
            module,
        ] + args
    else:
        cmd = [sys.executable, "-m", module] + args

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    click.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), **kwargs)
    return result.returncode


def run_script(script_path: str, args: List[str], docker: bool = False, **kwargs):
    """Run a Python script with optional Docker wrapper."""
    if docker and not _is_in_docker():
        docker_image = os.environ.get(
            "DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.7"
        )
        cmd = [
            "docker",
            "run",
            "--rm",
            "-it",
            "--gpus",
            "all",
            "-e",
            "PYTHONPATH=/workspace:/workspace/src",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-v",
            f"{PROJECT_ROOT}:/workspace",
            "-w",
            "/workspace",
            "--shm-size=8gb",
            docker_image,
            "python3",
            f"/workspace/{script_path}",
        ] + args
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / script_path)] + args

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    click.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), **kwargs)
    return result.returncode


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.version_option(version="0.0.2", prog_name="mlbot")
def cli():
    """ML Trading Bot - Unified CLI for training, data, and feature management."""
    pass


# =============================================================================
# Features Commands
# =============================================================================


@cli.group()
def features():
    """Feature registry management commands."""
    pass


@features.command("list")
@click.option(
    "--all", "-a", "show_all", is_flag=True, help="Show all features with details"
)
@click.option("--category", "-c", help="Filter by category (e.g., baseline, orderflow)")
@click.option("--search", "-s", help="Search for features by name")
@click.option("--module", "-m", help="Filter by module path")
def features_list(
    show_all: bool,
    category: Optional[str],
    search: Optional[str],
    module: Optional[str],
):
    """List all registered feature functions."""
    args = []
    if show_all:
        args.append("--all")
    if category:
        args.extend(["--category", category])
    if search:
        args.extend(["--search", search])
    if module:
        args.extend(["--module", module])

    sys.exit(run_script("scripts/list_features.py", args))


@features.command("count")
def features_count():
    """Show feature count by category."""
    # Run via the list_features script which handles paths correctly
    args = []
    sys.exit(run_script("scripts/list_features.py", args))


# =============================================================================
# Data Commands
# =============================================================================


@cli.group()
def data():
    """Data download and conversion commands."""
    pass


@data.command("download")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.option("--start-year", default="2023", help="Start year")
@click.option("--start-month", default="1", help="Start month")
@click.option("--end-year", help="End year (default: current)")
@click.option("--end-month", help="End month (default: current)")
@click.option(
    "--data-dir", default="data/agg_data", help="Output directory for ZIP files"
)
@click.option(
    "--parquet-dir", default="data/parquet_data", help="Output directory for Parquet"
)
def data_download(
    symbols, start_year, start_month, end_year, end_month, data_dir, parquet_dir
):
    """Download Binance monthly aggTrades data."""
    args = [
        "--data-dir",
        data_dir,
        "--parquet-dir",
        parquet_dir,
        "--symbols",
        *symbols.split(","),
        "--start-year",
        start_year,
        "--start-month",
        start_month,
    ]
    if end_year:
        args.extend(["--end-year", end_year])
    if end_month:
        args.extend(["--end-month", end_month])

    sys.exit(run_script("src/data_tools/download_training_data.py", args))


@data.command("convert")
@click.option(
    "--cleanup/--no-cleanup", default=True, help="Clean up ZIP files after conversion"
)
def data_convert(cleanup):
    """Convert downloaded ZIPs to Parquet format."""
    args = ["--cleanup", "yes" if cleanup else "no"]
    sys.exit(run_python_module("src.data_tools.zip_to_parquet", args))


@data.command("pipeline")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.pass_context
def data_pipeline(ctx, symbols):
    """Download and convert data (full pipeline)."""
    ctx.invoke(data_download, symbols=symbols)
    ctx.invoke(data_convert)


# =============================================================================
# Train Commands
# =============================================================================


@cli.group()
def train():
    """Model training commands."""
    pass


@train.command("sr-reversal")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option(
    "--timeframe", "-t", default="240T", help="Timeframe (e.g., 15T, 60T, 240T)"
)
@click.option(
    "--config",
    "-c",
    default="config/strategies/sr_reversal",
    help="Strategy config path",
)
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option(
    "--output-root", default="results/strategies/sr_reversal", help="Output directory"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_sr_reversal(
    symbol, timeframe, config, data_path, test_size, output_root, docker
):
    """Train SR Reversal model."""
    args = [
        "--config",
        f"/workspace/{config}" if docker else config,
        "--data-path",
        f"/workspace/{data_path}" if docker else data_path,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--output-root",
        f"/workspace/{output_root}" if docker else output_root,
    ]
    sys.exit(run_script("scripts/train_strategy_pipeline.py", args, docker=docker))


@train.command("sr-reversal-long")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
@click.pass_context
def train_sr_reversal_long(ctx, symbol, timeframe, docker):
    """Train SR Reversal Long-only model."""
    ctx.invoke(
        train_sr_reversal,
        symbol=symbol,
        timeframe=timeframe,
        config="config/strategies/sr_reversal_long",
        output_root="results/strategies/sr_reversal_long",
        docker=docker,
    )


@train.command("sr-reversal-short")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
@click.pass_context
def train_sr_reversal_short(ctx, symbol, timeframe, docker):
    """Train SR Reversal Short-only model."""
    ctx.invoke(
        train_sr_reversal,
        symbol=symbol,
        timeframe=timeframe,
        config="config/strategies/sr_reversal_short",
        output_root="results/strategies/sr_reversal_short",
        docker=docker,
    )


@train.command("rolling")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="15T", help="Timeframe")
@click.option(
    "--config", "-c", default="config/strategies/sr_reversal", help="Strategy config"
)
@click.option("--initial-train-months", default="3", help="Initial training months")
@click.option("--min-train-months", default="3", help="Minimum training months")
@click.option("--start", help="Rolling start date (YYYY-MM-DD)")
@click.option("--end", help="Rolling end date (YYYY-MM-DD)")
@click.option("--update-only", is_flag=True, help="Only update existing models")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_rolling(
    symbol,
    timeframe,
    config,
    initial_train_months,
    min_train_months,
    start,
    end,
    update_only,
    docker,
):
    """Rolling window training (expanding window)."""
    args = [
        "--config",
        f"/workspace/{config}" if docker else config,
        "--symbol",
        symbol,
        "--data-dir",
        "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--timeframe",
        timeframe,
        "--initial-train-months",
        initial_train_months,
        "--min-train-months",
        min_train_months,
        "--output-root",
        "/workspace/results/rolling" if docker else "results/rolling",
    ]
    if start:
        args.extend(["--start", start])
    if end:
        args.extend(["--end", end])
    if update_only:
        args.append("--update-only")

    sys.exit(
        run_script(
            "src/time_series_model/pipeline/rolling/rolling_train.py",
            args,
            docker=docker,
        )
    )


# =============================================================================
# Test Commands
# =============================================================================


@cli.group()
def test():
    """Testing commands."""
    pass


@test.command("unit")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--pattern", "-k", help="Test name pattern to match")
def test_unit(verbose, pattern):
    """Run unit tests."""
    args = ["tests/unit/"]
    if verbose:
        args.append("-v")
    if pattern:
        args.extend(["-k", pattern])

    sys.exit(run_python_module("pytest", args))


@test.command("integration")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--fast", is_flag=True, help="Skip slow tests")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def test_integration(verbose, fast, docker):
    """Run integration tests."""
    args = ["tests/integration/"]
    if verbose:
        args.append("-v")
    if fast:
        args.extend(["-m", "not slow"])

    sys.exit(run_python_module("pytest", args, docker=docker))


@test.command("all")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test_all(verbose):
    """Run all tests (unit + integration)."""
    args = ["tests/"]
    if verbose:
        args.append("-v")

    sys.exit(run_python_module("pytest", args))


# =============================================================================
# Dev Commands
# =============================================================================


@cli.group()
def dev():
    """Development commands."""
    pass


@dev.command("install")
def dev_install():
    """Install project in editable mode."""
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


@dev.command("format")
def dev_format():
    """Format code with black."""
    dirs = [
        "src/time_series_model/",
        "src/cross_sectional/",
        "src/data_tools/",
        "tests/",
        "scripts/",
    ]
    cmd = [sys.executable, "-m", "black"] + dirs
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


@dev.command("lint")
def dev_lint():
    """Lint code with flake8."""
    dirs = [
        "src/time_series_model/",
        "src/cross_sectional/",
        "src/data_tools/",
        "tests/",
        "scripts/",
    ]
    cmd = [sys.executable, "-m", "flake8"] + dirs
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


@dev.command("clean")
def dev_clean():
    """Clean build artifacts."""
    import shutil

    dirs_to_remove = ["build/", "dist/", "*.egg-info/"]
    for d in dirs_to_remove:
        for path in PROJECT_ROOT.glob(d):
            if path.is_dir():
                shutil.rmtree(path)
                click.echo(f"Removed: {path}")

    # Remove __pycache__ and .pyc files
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache)
    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        pyc.unlink()

    click.echo("✅ Cleaned build artifacts")


# =============================================================================
# Docker Commands
# =============================================================================


@cli.group()
def docker():
    """Docker management commands."""
    pass


@docker.command("shell")
def docker_shell():
    """Open interactive shell in Docker container."""
    docker_image = os.environ.get(
        "DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.7"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "--gpus",
        "all",
        "-e",
        "PYTHONPATH=/workspace/src",
        "-v",
        f"{PROJECT_ROOT}:/workspace",
        "-w",
        "/workspace",
        "--shm-size=8gb",
        docker_image,
        "bash",
    ]
    click.echo(f"Opening shell in {docker_image}...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


@docker.command("build")
def docker_build():
    """Build Docker image."""
    script = PROJECT_ROOT / "docker" / "build-gpu.sh"
    if script.exists():
        cmd = [
            "bash",
            str(script),
            "-n",
            "hansenlovefiona017/lightgbm-runtime",
            "-t",
            "v0.0.7",
            "--no-proxy",
            "--no-ssh",
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        sys.exit(result.returncode)
    else:
        click.echo("❌ Build script not found: docker/build-gpu.sh")
        sys.exit(1)


# =============================================================================
# Analysis Commands
# =============================================================================


@cli.group()
def analyze():
    """Analysis and evaluation commands."""
    pass


@analyze.command("feature-eval")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--horizon", default="24", help="Forward horizon bars")
@click.option("--feature-types", default="baseline", help="Feature types to evaluate")
@click.option("--start-date", help="Start date")
@click.option("--end-date", help="End date")
@click.option(
    "--output-dir", default="results/feature_evaluation", help="Output directory"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_feature_eval(
    symbol, timeframe, horizon, feature_types, start_date, end_date, output_dir, docker
):
    """Feature type evaluation (IC ranking + top factors)."""
    args = [
        "--data-path",
        "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--horizon",
        horizon,
        "--feature-types",
        feature_types,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]
    if start_date:
        args.extend(["--train-start", start_date])
    if end_date:
        args.extend(["--train-end", end_date])

    sys.exit(
        run_python_module(
            "time_series_model.pipeline.training.feature_type_evaluator",
            args,
            docker=docker,
        )
    )


@analyze.command("factor-eval")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal_long",
    help="Strategy config directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option(
    "--factors",
    multiple=True,
    help="Factor columns to evaluate (can specify multiple times). If not specified, uses requested_features from strategy config",
)
@click.option("--start-date", help="Start date")
@click.option("--end-date", help="End date")
@click.option(
    "--quantile", type=float, default=0.2, help="Top/Bottom quantile for evaluation"
)
@click.option(
    "--feature-mode",
    type=click.Choice(["strategy", "only", "append"]),
    default="strategy",
    help="How to handle feature pipeline",
)
@click.option(
    "--ic-decay-lags",
    default="1,3,5,10,20",
    help="Comma-separated forward bars for IC decay analysis",
)
@click.option("--output-dir", default="results/factor_ts_eval", help="Output directory")
@click.option(
    "--open-browser",
    is_flag=True,
    default=False,
    help="Automatically open HTML report in browser",
)
@click.option(
    "--remove-correlated",
    is_flag=True,
    default=False,
    help="Remove highly correlated features",
)
@click.option(
    "--correlation-threshold",
    type=float,
    default=0.9,
    help="Correlation threshold for removing redundant features (default: 0.9)",
)
@click.option(
    "--filter-by-best-lag",
    is_flag=True,
    default=False,
    help="Filter features by best lag (only keep features with best lag matching target lag)",
)
@click.option(
    "--target-lag",
    type=int,
    default=None,
    help="Target lag for filtering (if not specified, will be inferred from label config max_holding_bars)",
)
@click.option(
    "--lag-tolerance",
    type=int,
    default=5,
    help="Tolerance for target lag matching (default: 5)",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_factor_eval(
    strategy_config,
    symbol,
    timeframe,
    factors,
    start_date,
    end_date,
    quantile,
    feature_mode,
    ic_decay_lags,
    output_dir,
    open_browser,
    remove_correlated,
    correlation_threshold,
    filter_by_best_lag,
    target_lag,
    lag_tolerance,
    docker,
):
    """Time-series factor IC / win-rate evaluation (single asset)."""
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if docker else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--timeframe",
        timeframe,
        "--quantile",
        str(quantile),
        "--feature-mode",
        feature_mode,
        "--ic-decay-lags",
        ic_decay_lags,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]
    if factors:
        args.extend(["--factors"] + list(factors))
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if open_browser:
        args.append("--open-browser")
    if remove_correlated:
        args.append("--remove-correlated")
        args.extend(["--correlation-threshold", str(correlation_threshold)])
    if filter_by_best_lag or target_lag is not None:
        args.append("--filter-by-best-lag")
        if target_lag is not None:
            args.extend(["--target-lag", str(target_lag)])
        args.extend(["--lag-tolerance", str(lag_tolerance)])

    # When running inside Docker (via Makefile), docker=False
    # When running locally, docker=True will spawn Docker container
    # The Makefile handles Docker setup, so we just run the module directly
    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.factor_ts_eval",
            args,
            docker=False,  # Makefile already runs us in Docker, don't nest
        )
    )


@analyze.command("dim-compare")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="15T", help="Timeframe")
@click.option(
    "--config", "-c", default="config/strategies/sr_reversal", help="Strategy config"
)
@click.option("--start-date", help="Start date")
@click.option("--end-date", help="End date")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_dim_compare(symbol, timeframe, config, start_date, end_date, docker):
    """Dimensionality comparison & feature selection."""
    args = [
        "--config",
        f"/workspace/{config}" if docker else config,
        "--symbol",
        symbol,
        "--data-path",
        "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--timeframe",
        timeframe,
    ]
    if start_date:
        args.extend(["--train-start", start_date])
    if end_date:
        args.extend(["--train-end", end_date])

    sys.exit(
        run_python_module(
            "src.time_series_model.pipeline.dimensionality.dimensionality_comparison",
            args,
            docker=docker,
        )
    )


@analyze.command("strategy-feature-compare")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal",
    help="Strategy config directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option(
    "--output-dir",
    default="results/strategy_compare",
    help="Output directory",
)
@click.option(
    "--feature-overrides",
    help="Feature config overrides (format: name=path,name2=path2)",
)
@click.option("--run-rolling", is_flag=True, help="Run rolling window evaluation")
@click.option(
    "--rolling-train-bars", default="1000", help="Rolling training window size"
)
@click.option("--rolling-test-bars", default="200", help="Rolling test window size")
@click.option("--rolling-step-bars", default="100", help="Rolling step size")
@click.option("--rolling-max-windows", default="10", help="Maximum rolling windows")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_strategy_feature_compare(
    strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    output_dir,
    feature_overrides,
    run_rolling,
    rolling_train_bars,
    rolling_test_bars,
    rolling_step_bars,
    rolling_max_windows,
    docker,
):
    """Ablation Study: Compare multiple feature configs for a strategy."""
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if docker else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
        "--rolling-train-bars",
        rolling_train_bars,
        "--rolling-test-bars",
        rolling_test_bars,
        "--rolling-step-bars",
        rolling_step_bars,
        "--rolling-max-windows",
        rolling_max_windows,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if feature_overrides:
        args.extend(["--feature-overrides", feature_overrides])
    if run_rolling:
        args.append("--run-rolling")

    sys.exit(
        run_script(
            "src/time_series_model/strategies/evaluation/strategy_feature_compare.py",
            args,
            docker=docker,
        )
    )


@analyze.command("timeframe-comparison")
@click.option(
    "--output-dir",
    default="results/model_comparison",
    help="Output directory",
)
@click.option(
    "--results-1h",
    default="results/model_comparison/comparison_results.csv",
    help="1h timeframe results CSV",
)
@click.option(
    "--results-4h",
    default="results/model_comparison_240h/comparison_results.csv",
    help="4h timeframe results CSV",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_timeframe_comparison(output_dir, results_1h, results_4h, docker):
    """Generate comprehensive comparison between 1h and 4h timeframes."""
    args = [
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
        "--results-1h",
        f"/workspace/{results_1h}" if docker else results_1h,
        "--results-4h",
        f"/workspace/{results_4h}" if docker else results_4h,
    ]

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.generate_timeframe_comparison_report",
            args,
            docker=docker,
        )
    )


# =============================================================================
# Diagnostic Commands
# =============================================================================


@cli.group()
def diagnose():
    """Diagnostic and analysis commands."""
    pass


@diagnose.command("rule-baseline")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal",
    help="Strategy config directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--data-path",
    default="data/parquet_data",
    help="Data directory",
)
@click.option(
    "--ticks-dir",
    default="data/parquet_data",
    help="Tick data directory",
)
@click.option(
    "--ticks-lookback-minutes",
    default="60",
    help="VPIN calculation lookback minutes",
)
@click.option(
    "--max-holding-bars",
    type=int,
    help="Maximum holding bars (overrides config)",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_rule_baseline(
    strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    data_path,
    ticks_dir,
    ticks_lookback_minutes,
    max_holding_bars,
    docker,
):
    """Test pure rule-based SR+RR strategy (no ML)."""
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if docker else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        f"/workspace/{data_path}" if docker else data_path,
        "--timeframe",
        timeframe,
        "--ticks-dir",
        f"/workspace/{ticks_dir}" if docker else ticks_dir,
        "--ticks-lookback-minutes",
        ticks_lookback_minutes,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if max_holding_bars:
        args.extend(["--max-holding-bars", str(max_holding_bars)])

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.sr_reversal_rule_baseline",
            args,
            docker=docker,
        )
    )


@diagnose.command("test-vpin-thresholds")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal",
    help="Strategy config directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--data-path",
    default="data/parquet_data",
    help="Data directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_test_vpin_thresholds(
    strategy_config, symbol, timeframe, start_date, end_date, data_path, docker
):
    """Test different VPIN thresholds for SR Reversal."""
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if docker else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        f"/workspace/{data_path}" if docker else data_path,
        "--timeframe",
        timeframe,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.test_vpin_thresholds",
            args,
            docker=docker,
        )
    )


@diagnose.command("ml-volatility")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_ml_volatility(docker):
    """Analyze ML+Volatility Model Performance Issues."""
    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.analyze_ml_volatility_model",
            [],
            docker=docker,
        )
    )


@diagnose.command("dtw-volatility")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_dtw_volatility(docker):
    """Analyze DTW Features and Volatility Model."""
    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.analyze_dtw_and_volatility",
            [],
            docker=docker,
        )
    )


@diagnose.command("model-comparison")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal",
    help="Strategy config directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option(
    "--output-dir",
    default="results/model_comparison",
    help="Output directory (will append timeframe)",
)
@click.option(
    "--data-path",
    default="data/parquet_data",
    help="Data directory",
)
@click.option(
    "--ticks-dir",
    default="data/parquet_data",
    help="Tick data directory",
)
@click.option(
    "--ticks-lookback-minutes",
    default="60",
    help="VPIN calculation lookback minutes",
)
@click.option(
    "--rule-params",
    help="Rule optimization results CSV (optional)",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_model_comparison(
    strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    output_dir,
    data_path,
    ticks_dir,
    ticks_lookback_minutes,
    rule_params,
    docker,
):
    """Compare Rule-based vs ML vs ML+Volatility models."""
    # Append timeframe to output dir
    output_dir_full = f"{output_dir}/{timeframe}"

    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if docker else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        f"/workspace/{data_path}" if docker else data_path,
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--output-dir",
        f"/workspace/{output_dir_full}" if docker else output_dir_full,
        "--ticks-dir",
        f"/workspace/{ticks_dir}" if docker else ticks_dir,
        "--ticks-lookback-minutes",
        ticks_lookback_minutes,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if rule_params:
        args.extend(
            ["--rule-params", f"/workspace/{rule_params}" if docker else rule_params]
        )

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.sr_reversal_model_comparison",
            args,
            docker=docker,
        )
    )


# =============================================================================
# Optimization Commands
# =============================================================================


@cli.group()
def optimize():
    """Optimization commands."""
    pass


@optimize.command("rule")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal",
    help="Strategy config directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--data-path",
    default="data/parquet_data",
    help="Data directory",
)
@click.option(
    "--output-dir",
    default="results/rule_optimization",
    help="Output directory",
)
@click.option(
    "--search-type",
    type=click.Choice(["grid", "random", "optuna"]),
    default="random",
    help="Search type",
)
@click.option(
    "--n-trials",
    default="100",
    help="Number of optimization trials",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def optimize_rule(
    strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    data_path,
    output_dir,
    search_type,
    n_trials,
    docker,
):
    """Find parameter plateaus for rule-based strategy."""
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if docker else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        f"/workspace/{data_path}" if docker else data_path,
        "--timeframe",
        timeframe,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
        "--search-type",
        search_type,
        "--n-trials",
        n_trials,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.sr_reversal_rule_optimization",
            args,
            docker=docker,
        )
    )


@optimize.command("rule-plateau-charts")
@click.option(
    "--results-csv",
    default="results/rule_optimization/optimization_results.csv",
    help="Optimization results CSV",
)
@click.option(
    "--report-html",
    default="results/rule_optimization/optimization_report.html",
    help="Report HTML file",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def optimize_rule_plateau_charts(results_csv, report_html, docker):
    """Generate rule plateau heatmaps and scatter charts."""
    args = [
        "--results-csv",
        f"/workspace/{results_csv}" if docker else results_csv,
        "--report-html",
        f"/workspace/{report_html}" if docker else report_html,
    ]

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.generate_rule_plateau_charts",
            args,
            docker=docker,
        )
    )


@optimize.command("ml-param-sweep")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal",
    help="Strategy config directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option(
    "--output-dir",
    default="results/model_comparison",
    help="Output directory (will append timeframe)",
)
@click.option(
    "--data-path",
    default="data/parquet_data",
    help="Data directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def optimize_ml_param_sweep(
    strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    output_dir,
    data_path,
    docker,
):
    """Run ML parameter sweep for plateau analysis."""
    output_dir_full = f"{output_dir}/{timeframe}"

    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if docker else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        f"/workspace/{data_path}" if docker else data_path,
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--output-dir",
        f"/workspace/{output_dir_full}" if docker else output_dir_full,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.sr_reversal_ml_parameter_sweep",
            args,
            docker=docker,
        )
    )


@optimize.command("ml-plateau-charts")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option(
    "--results-csv",
    help="ML parameter sweep CSV (default: results/model_comparison/{timeframe}/ml_param_sweep.csv)",
)
@click.option(
    "--report-html",
    help="Report HTML file (default: results/model_comparison/{timeframe}/comparison_report.html)",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def optimize_ml_plateau_charts(timeframe, results_csv, report_html, docker):
    """Generate ML plateau heatmaps and scatter charts."""
    if not results_csv:
        results_csv = f"results/model_comparison/{timeframe}/ml_param_sweep.csv"
    if not report_html:
        report_html = f"results/model_comparison/{timeframe}/comparison_report.html"

    args = [
        "--results-csv",
        f"/workspace/{results_csv}" if docker else results_csv,
        "--report-html",
        f"/workspace/{report_html}" if docker else report_html,
    ]

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.generate_ml_plateau_charts",
            args,
            docker=docker,
        )
    )


# =============================================================================
# Backtest Commands
# =============================================================================


@cli.group()
def backtest():
    """Backtesting commands."""
    pass


@backtest.command("vectorbot")
@click.option(
    "--model",
    help="Model path (optional, if not provided uses default)",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def backtest_vectorbot(model, symbol, start, end, docker):
    """Run VectorBot risk-managed backtest."""
    args = []
    if model:
        args.extend(["--model", model])
    args.extend(["--symbol", symbol])
    if start:
        args.extend(["--start", start])
    if end:
        args.extend(["--end", end])

    sys.exit(
        run_python_module(
            "time_series_model.backtesting.vectorbot",
            args,
            docker=docker,
        )
    )


@backtest.command("nautilus")
@click.option(
    "--data-dir",
    default="data/parquet_data",
    help="Data directory",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def backtest_nautilus(data_dir, symbol, docker):
    """Run Nautilus Trader backtest."""
    args = [
        "--data-dir",
        data_dir if not docker else f"/workspace/{data_dir}",
        "--symbol",
        symbol,
    ]

    sys.exit(
        run_python_module(
            "time_series_model.backtesting.nautilus_dim",
            args,
            docker=docker,
        )
    )


# =============================================================================
# Cross-Sectional Commands
# =============================================================================


@cli.group()
def cross_section():
    """Cross-sectional analysis commands."""
    pass


@cross_section.command("build-panel")
@click.option("--symbols", "-s", help="Comma-separated symbols (default: from config)")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--output-dir",
    default="data/cross_sectional_panels",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_build_panel(symbols, start_date, end_date, output_dir, docker):
    """Generate multi-asset factor panels for CS modelling."""
    args = [
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]
    if symbols:
        args.extend(["--symbols"] + symbols.split(","))
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/generate_panel.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("report")
@click.option(
    "--panel-path",
    default="data/cross_sectional_panels/panel.parquet",
    help="Panel data path",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_report(panel_path, output_dir, docker):
    """Generate Fama-MacBeth + Newey-West + IC/IR markdown report."""
    args = [
        "--panel-path",
        f"/workspace/{panel_path}" if docker else panel_path,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_famacbeth_report.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("train")
@click.option(
    "--panel-path",
    default="data/cross_sectional_panels/panel.parquet",
    help="Panel data path",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_train(panel_path, output_dir, docker):
    """Train cross-sectional models (boosting/Fama-MacBeth)."""
    args = [
        "--panel-path",
        f"/workspace/{panel_path}" if docker else panel_path,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/train_cross_sectional_model.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("catalog")
@click.option(
    "--panel-path",
    default="data/cross_sectional_panels/panel.parquet",
    help="Panel data path",
)
@click.option(
    "--output-path",
    default="results/cross_sectional/factor_catalog.json",
    help="Output catalog JSON path",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_catalog(panel_path, output_path, docker):
    """Export factor catalog (IC/IR summary)."""
    args = [
        "--panel-path",
        f"/workspace/{panel_path}" if docker else panel_path,
        "--output-path",
        f"/workspace/{output_path}" if docker else output_path,
    ]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/export_factor_catalog.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("select")
@click.option(
    "--panel-path",
    default="data/cross_sectional_panels/panel.parquet",
    help="Panel data path",
)
@click.option(
    "--output-path",
    default="results/cross_sectional/selected_factors.json",
    help="Output selected factors JSON path",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_select(panel_path, output_path, docker):
    """Auto-select factors using correlation and IC filtering."""
    args = [
        "--panel-path",
        f"/workspace/{panel_path}" if docker else panel_path,
        "--output-path",
        f"/workspace/{output_path}" if docker else output_path,
    ]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/auto_select_factors.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("shap")
@click.option(
    "--model-path",
    default="results/cross_sectional/model.pkl",
    help="Trained model path",
)
@click.option(
    "--panel-path",
    default="data/cross_sectional_panels/panel.parquet",
    help="Panel data path",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional/shap",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_shap(model_path, panel_path, output_dir, docker):
    """Run SHAP analysis on cross-sectional model."""
    args = [
        "--model-path",
        f"/workspace/{model_path}" if docker else model_path,
        "--panel-path",
        f"/workspace/{panel_path}" if docker else panel_path,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_shap_analysis.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("logic-check")
@click.option(
    "--panel-path",
    default="data/cross_sectional_panels/panel.parquet",
    help="Panel data path",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_logic_check(panel_path, output_dir, docker):
    """Run factor logic consistency checks."""
    args = [
        "--panel-path",
        f"/workspace/{panel_path}" if docker else panel_path,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_factor_logic_check.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("shap-drift")
@click.option(
    "--model-path",
    default="results/cross_sectional/model.pkl",
    help="Trained model path",
)
@click.option(
    "--panel-path",
    default="data/cross_sectional_panels/panel.parquet",
    help="Panel data path",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional/shap_drift",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_shap_drift(model_path, panel_path, output_dir, docker):
    """Monitor SHAP value drift over time."""
    args = [
        "--model-path",
        f"/workspace/{model_path}" if docker else model_path,
        "--panel-path",
        f"/workspace/{panel_path}" if docker else panel_path,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_shap_drift_monitor.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("factor-eval")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_factor_eval(symbol, timeframe, start_date, end_date, docker):
    """Cross-sectional factor evaluation (IC, decay, quantile spread)."""
    args = []
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.cross_sectional_eval",
            args,
            docker=docker,
        )
    )


# =============================================================================
# Visualization Commands
# =============================================================================


@cli.group()
def visualize():
    """Visualization commands."""
    pass


@visualize.command("feature-indicators")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--config",
    default="config/visualization/feature_indicators.yaml",
    help="Visualization config file",
)
@click.option(
    "--output-dir",
    default="results/feature_indicators",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def visualize_feature_indicators(
    symbol, timeframe, start_date, end_date, config, output_dir, docker
):
    """Generate feature indicators visualization."""
    args = [
        "--data-path",
        "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--config",
        f"/workspace/{config}" if docker else config,
        "--output-dir",
        f"/workspace/{output_dir}" if docker else output_dir,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])

    sys.exit(
        run_script(
            "src/time_series_model/visualization/feature_indicator_visualizer.py",
            args,
            docker=docker,
        )
    )


@analyze.command("timeframe-forward-report")
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option(
    "--timeframes",
    default="60T,240T",
    help="Comma-separated timeframes (e.g., 60T,240T)",
)
@click.option(
    "--forward-bars",
    default="1,3,5,10,20",
    help="Comma-separated forward bars (e.g., 1,3,5,10,20)",
)
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option(
    "--data-dir",
    default="data/parquet_data",
    help="Data directory",
)
@click.option(
    "--output-dir",
    default="results/timeframe_analysis",
    help="Output directory",
)
@click.option("--max-lag", default="20", help="Maximum lag")
@click.option("--min-samples", default="500", help="Minimum samples")
@click.option("--top-k", default="10", help="Top K features")
@click.option(
    "--feature-type",
    default="baseline",
    help="Feature type to analyze",
)
@click.option("--extra-features", help="Extra feature config path (optional)")
@click.option("--run-tag", help="Run tag for organizing results (optional)")
@click.option(
    "--pearson-threshold",
    default="0.03",
    help="Pearson correlation threshold for config generation",
)
@click.option(
    "--pvalue-threshold",
    default="1e-5",
    help="P-value threshold for config generation",
)
@click.option(
    "--config-min-samples",
    default="500",
    help="Minimum samples for config generation",
)
@click.option(
    "--top-features-per-symbol",
    default="5",
    help="Top features per symbol for config generation",
)
@click.option(
    "--top-features-per-group",
    default="10",
    help="Top features per group for config generation",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_timeframe_forward_report(
    symbols,
    timeframes,
    forward_bars,
    start,
    end,
    data_dir,
    output_dir,
    max_lag,
    min_samples,
    top_k,
    feature_type,
    extra_features,
    run_tag,
    pearson_threshold,
    pvalue_threshold,
    config_min_samples,
    top_features_per_symbol,
    top_features_per_group,
    docker,
):
    """Timeframe vs forward-bar correlation analysis."""
    # Convert comma-separated strings to space-separated for the script
    symbols_list = symbols.split(",")
    timeframes_list = timeframes.split(",")
    forward_bars_list = forward_bars.split(",")

    args = (
        [
            "--data-dir",
            f"/workspace/{data_dir}" if docker else data_dir,
            "--output-dir",
            f"/workspace/{output_dir}" if docker else output_dir,
            "--symbols",
        ]
        + symbols_list
        + [
            "--timeframes",
        ]
        + timeframes_list
        + [
            "--forward-bars",
        ]
        + forward_bars_list
        + [
            "--max-lag",
            max_lag,
            "--min-samples",
            min_samples,
            "--top-k",
            top_k,
            "--feature-type",
            feature_type,
        ]
    )
    if start:
        args.extend(["--start", start])
    if end:
        args.extend(["--end", end])
    if extra_features:
        args.extend(["--extra-features", extra_features])
    if run_tag:
        args.extend(["--run-tag", run_tag])

    result = run_python_module(
        "time_series_model.analysis.timeframe_forward_correlation",
        args,
        docker=docker,
    )

    # If run-tag is provided and first command succeeded, generate config
    if run_tag and result == 0:
        details_csv = f"{output_dir}/{run_tag}/timeframe_forward_details.csv"
        config_output_dir = f"{output_dir}/{run_tag}/config"
        config_args = [
            "--details-csv",
            f"/workspace/{details_csv}" if docker else details_csv,
            "--output-dir",
            f"/workspace/{config_output_dir}" if docker else config_output_dir,
            "--pearson-threshold",
            pearson_threshold,
            "--pvalue-threshold",
            pvalue_threshold,
            "--min-samples",
            config_min_samples,
            "--top-features-per-symbol",
            top_features_per_symbol,
            "--top-features-per-group",
            top_features_per_group,
        ]
        run_python_module(
            "time_series_model.analysis.timeframe_feature_selector",
            config_args,
            docker=docker,
        )

    sys.exit(result)


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
