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


def run_python_module(module: str, args: List[str], docker: bool = False, **kwargs):
    """Run a Python module with optional Docker wrapper."""
    if docker and not os.environ.get("DEV_CONTAINER"):
        # Build Docker command
        docker_image = os.environ.get("DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.7")
        cmd = [
            "docker", "run", "--rm", "-it",
            "--gpus", "all",
            "-e", "PYTHONPATH=/workspace/src",
            "-e", "PYTHONUNBUFFERED=1",
            "-v", f"{PROJECT_ROOT}:/workspace",
            "-w", "/workspace",
            "--shm-size=8gb",
            docker_image,
            "python3", "-m", module,
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
    if docker and not os.environ.get("DEV_CONTAINER"):
        docker_image = os.environ.get("DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.7")
        cmd = [
            "docker", "run", "--rm", "-it",
            "--gpus", "all",
            "-e", "PYTHONPATH=/workspace:/workspace/src",
            "-e", "PYTHONUNBUFFERED=1",
            "-v", f"{PROJECT_ROOT}:/workspace",
            "-w", "/workspace",
            "--shm-size=8gb",
            docker_image,
            "python3", f"/workspace/{script_path}",
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
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all features with details")
@click.option("--category", "-c", help="Filter by category (e.g., baseline, orderflow)")
@click.option("--search", "-s", help="Search for features by name")
@click.option("--module", "-m", help="Filter by module path")
def features_list(show_all: bool, category: Optional[str], search: Optional[str], module: Optional[str]):
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
@click.option("--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols")
@click.option("--start-year", default="2023", help="Start year")
@click.option("--start-month", default="1", help="Start month")
@click.option("--end-year", help="End year (default: current)")
@click.option("--end-month", help="End month (default: current)")
@click.option("--data-dir", default="data/agg_data", help="Output directory for ZIP files")
@click.option("--parquet-dir", default="data/parquet_data", help="Output directory for Parquet")
def data_download(symbols, start_year, start_month, end_year, end_month, data_dir, parquet_dir):
    """Download Binance monthly aggTrades data."""
    args = [
        "--data-dir", data_dir,
        "--parquet-dir", parquet_dir,
        "--symbols", *symbols.split(","),
        "--start-year", start_year,
        "--start-month", start_month,
    ]
    if end_year:
        args.extend(["--end-year", end_year])
    if end_month:
        args.extend(["--end-month", end_month])
    
    sys.exit(run_script("src/data_tools/download_training_data.py", args))


@data.command("convert")
@click.option("--cleanup/--no-cleanup", default=True, help="Clean up ZIP files after conversion")
def data_convert(cleanup):
    """Convert downloaded ZIPs to Parquet format."""
    args = ["--cleanup", "yes" if cleanup else "no"]
    sys.exit(run_python_module("src.data_tools.zip_to_parquet", args))


@data.command("pipeline")
@click.option("--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols")
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
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 15T, 60T, 240T)")
@click.option("--config", "-c", default="config/strategies/sr_reversal", help="Strategy config path")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option("--output-root", default="results/strategies/sr_reversal", help="Output directory")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_sr_reversal(symbol, timeframe, config, data_path, test_size, output_root, docker):
    """Train SR Reversal model."""
    args = [
        "--config", f"/workspace/{config}" if docker else config,
        "--data-path", f"/workspace/{data_path}" if docker else data_path,
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--test-size", test_size,
        "--output-root", f"/workspace/{output_root}" if docker else output_root,
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
@click.option("--config", "-c", default="config/strategies/sr_reversal", help="Strategy config")
@click.option("--initial-train-months", default="3", help="Initial training months")
@click.option("--min-train-months", default="3", help="Minimum training months")
@click.option("--start", help="Rolling start date (YYYY-MM-DD)")
@click.option("--end", help="Rolling end date (YYYY-MM-DD)")
@click.option("--update-only", is_flag=True, help="Only update existing models")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_rolling(symbol, timeframe, config, initial_train_months, min_train_months, start, end, update_only, docker):
    """Rolling window training (expanding window)."""
    args = [
        "--config", f"/workspace/{config}" if docker else config,
        "--symbol", symbol,
        "--data-dir", "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--timeframe", timeframe,
        "--initial-train-months", initial_train_months,
        "--min-train-months", min_train_months,
        "--output-root", "/workspace/results/rolling" if docker else "results/rolling",
    ]
    if start:
        args.extend(["--start", start])
    if end:
        args.extend(["--end", end])
    if update_only:
        args.append("--update-only")
    
    sys.exit(run_script("src/time_series_model/pipeline/rolling/rolling_train.py", args, docker=docker))


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
    dirs = ["src/time_series_model/", "src/cross_sectional/", "src/data_tools/", "tests/", "scripts/"]
    cmd = [sys.executable, "-m", "black"] + dirs
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


@dev.command("lint")
def dev_lint():
    """Lint code with flake8."""
    dirs = ["src/time_series_model/", "src/cross_sectional/", "src/data_tools/", "tests/", "scripts/"]
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
    docker_image = os.environ.get("DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.7")
    cmd = [
        "docker", "run", "--rm", "-it",
        "--gpus", "all",
        "-e", "PYTHONPATH=/workspace/src",
        "-v", f"{PROJECT_ROOT}:/workspace",
        "-w", "/workspace",
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
        cmd = ["bash", str(script), "-n", "hansenlovefiona017/lightgbm-runtime", "-t", "v0.0.7", "--no-proxy", "--no-ssh"]
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
@click.option("--output-dir", default="results/feature_evaluation", help="Output directory")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_feature_eval(symbol, timeframe, horizon, feature_types, start_date, end_date, output_dir, docker):
    """Feature type evaluation (IC ranking + top factors)."""
    args = [
        "--data-path", "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--horizon", horizon,
        "--feature-types", feature_types,
        "--output-dir", f"/workspace/{output_dir}" if docker else output_dir,
    ]
    if start_date:
        args.extend(["--train-start", start_date])
    if end_date:
        args.extend(["--train-end", end_date])
    
    sys.exit(run_python_module("time_series_model.pipeline.training.feature_type_evaluator", args, docker=docker))


@analyze.command("dim-compare")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="15T", help="Timeframe")
@click.option("--config", "-c", default="config/strategies/sr_reversal", help="Strategy config")
@click.option("--start-date", help="Start date")
@click.option("--end-date", help="End date")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def analyze_dim_compare(symbol, timeframe, config, start_date, end_date, docker):
    """Dimensionality comparison & feature selection."""
    args = [
        "--config", f"/workspace/{config}" if docker else config,
        "--symbol", symbol,
        "--data-path", "/workspace/data/parquet_data" if docker else "data/parquet_data",
        "--timeframe", timeframe,
    ]
    if start_date:
        args.extend(["--train-start", start_date])
    if end_date:
        args.extend(["--train-end", end_date])
    
    sys.exit(run_python_module("src.time_series_model.pipeline.dimensionality.dimensionality_comparison", args, docker=docker))


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

