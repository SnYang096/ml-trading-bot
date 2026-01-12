"""
ML Trading Bot CLI - Unified command-line interface.

This CLI replaces the Makefile for cross-platform compatibility.

Usage:
    mlbot --help                    # Show all commands
    mlbot features list             # List registered features
    mlbot train sr-reversal-long    # Train SR Reversal Long-only model
    mlbot train sr-reversal-short   # Train SR Reversal Short-only model
    mlbot data download             # Download Binance data
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
import socket
from datetime import datetime, timezone
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

# Ensure repo root is importable so `import src.*` works when running `mlbot --no-docker`
# from arbitrary working directories / environments.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    env_overrides = kwargs.pop("env_overrides", None) or {}
    if docker and not _is_in_docker():
        # Build Docker command
        docker_image = os.environ.get(
            "DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.7"
        )
        extra_env_flags: List[str] = []
        for k, v in (env_overrides or {}).items():
            if v is None:
                continue
            extra_env_flags.extend(["-e", f"{k}={v}"])
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
        ] + extra_env_flags + [
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
    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                continue
            env[str(k)] = str(v)
    # Match Docker layout (PYTHONPATH includes project root + src/)
    pythonpath_parts = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
    if "PYTHONPATH" in env and env["PYTHONPATH"]:
        pythonpath_parts.insert(0, env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    # Ensure real-time logs even when stdout is redirected (e.g. nohup > file)
    env["PYTHONUNBUFFERED"] = "1"

    click.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), **kwargs)
    return result.returncode


def run_script(script_path: str, args: List[str], docker: bool = False, **kwargs):
    """Run a Python script with optional Docker wrapper."""
    env_overrides = kwargs.pop("env_overrides", None) or {}
    if docker and not _is_in_docker():
        docker_image = os.environ.get(
            "DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.9"
        )
        extra_env_flags: List[str] = []
        for k, v in (env_overrides or {}).items():
            if v is None:
                continue
            extra_env_flags.extend(["-e", f"{k}={v}"])
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
        ] + extra_env_flags + [
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
    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                continue
            env[str(k)] = str(v)
    # Match Docker layout (PYTHONPATH includes project root + src/)
    pythonpath_parts = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
    if "PYTHONPATH" in env and env["PYTHONPATH"]:
        pythonpath_parts.insert(0, env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    # Ensure real-time logs even when stdout is redirected (e.g. nohup > file)
    env["PYTHONUNBUFFERED"] = "1"

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
# Local Serving Commands (HTML reports, etc.)
# =============================================================================


def _port_is_in_use(port: int, bind: str = "0.0.0.0") -> bool:
    """Best-effort check whether (bind, port) is already bound by another process."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((bind, int(port)))
        return False
    except OSError:
        return True
    finally:
        try:
            s.close()
        except Exception:
            pass


def _find_listening_pids(port: int) -> List[int]:
    """Find process IDs listening on a TCP port (best-effort; requires psutil)."""
    try:
        import psutil  # type: ignore
    except Exception:
        return []

    pids = set()
    try:
        for conn in psutil.net_connections(kind="inet"):
            if not conn.laddr:
                continue
            if int(conn.laddr.port) != int(port):
                continue
            # Only consider listeners
            if getattr(conn, "status", None) != getattr(
                psutil, "CONN_LISTEN", "LISTEN"
            ):
                continue
            if conn.pid:
                pids.add(int(conn.pid))
    except Exception:
        # If psutil can't enumerate, just return empty and let caller handle
        return []

    return sorted(pids)


def _kill_pids(pids: List[int], timeout_s: float = 2.0) -> List[int]:
    """Terminate (then kill) PIDs. Returns the list of PIDs actually killed."""
    if not pids:
        return []

    try:
        import psutil  # type: ignore
    except Exception:
        return []

    killed: List[int] = []
    procs = []
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            procs.append(psutil.Process(pid))
        except Exception:
            continue

    # First: terminate
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass

    gone, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for p in gone:
        killed.append(p.pid)

    # Escalate: kill
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    gone2, _alive2 = psutil.wait_procs(alive, timeout=timeout_s)
    for p in gone2:
        killed.append(p.pid)

    return sorted(set(killed))


def _serve_static_dir(*, port: int, directory: str, bind: str, force: bool) -> None:
    port = int(port)
    directory_path = (PROJECT_ROOT / directory).resolve()
    if not directory_path.exists():
        raise click.ClickException(f"Directory not found: {directory_path}")

    if _port_is_in_use(port, bind=bind):
        if not force:
            raise click.ClickException(
                f"Port {port} is already in use. "
                f"Use --force to kill the owning process, or choose another port with --port."
            )

        pids = _find_listening_pids(port)
        if not pids:
            raise click.ClickException(
                f"Port {port} is in use but failed to identify owning PID(s). "
                f"Try another port (e.g. --port {port + 1})."
            )

        click.echo(f"⚠️  Port {port} in use by PID(s): {pids}. Killing (--force)...")
        killed = _kill_pids(pids)
        if not killed:
            raise click.ClickException(
                f"Failed to kill process(es) on port {port}: {pids}"
            )

        # Wait briefly for socket to release
        for _ in range(20):
            if not _port_is_in_use(port, bind=bind):
                break
            time.sleep(0.1)
        if _port_is_in_use(port, bind=bind):
            raise click.ClickException(
                f"Port {port} still in use after killing {killed}. Try another port."
            )

    url = f"http://localhost:{port}/"
    click.echo(f"🌐 Serving {directory_path} on {bind}:{port}")
    click.echo(f"   Open: {url}")
    click.echo("   (In devcontainer: forward the port in the Ports panel.)")

    cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(port),
        "--directory",
        str(directory_path),
        "--bind",
        str(bind),
    ]
    # Foreground (Ctrl+C to stop)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


@cli.command("server")
@click.option("--port", "-p", type=int, default=8008, show_default=True, help="Port")
@click.option(
    "--dir",
    "-d",
    "directory",
    default="results",
    show_default=True,
    help="Directory to serve",
)
@click.option(
    "--bind",
    default="0.0.0.0",
    show_default=True,
    help="Bind address (use 0.0.0.0 for devcontainer port forwarding)",
)
@click.option(
    "--force",
    is_flag=True,
    help="If port is in use, kill the process listening on the port and retry",
)
def server(port: int, directory: str, bind: str, force: bool) -> None:
    """Serve a directory via a local static server (HTML reports)."""
    _serve_static_dir(port=port, directory=directory, bind=bind, force=force)


@cli.command("serve-results", hidden=True)
@click.option("--port", "-p", type=int, default=8008, show_default=True, help="Port")
@click.option(
    "--dir",
    "-d",
    "directory",
    default="results",
    show_default=True,
    help="Directory to serve",
)
@click.option(
    "--bind",
    default="0.0.0.0",
    show_default=True,
    help="Bind address (use 0.0.0.0 for devcontainer port forwarding)",
)
@click.option(
    "--force",
    is_flag=True,
    help="If port is in use, kill the process listening on the port and retry",
)
def serve_results(port: int, directory: str, bind: str, force: bool) -> None:
    """DEPRECATED (use `mlbot server`). Kept for backward compatibility."""
    click.echo("⚠️  DEPRECATED: use `mlbot server` (this alias will be removed later).")
    _serve_static_dir(port=port, directory=directory, bind=bind, force=force)


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


def _data_download_impl(
    *,
    symbols: str,
    universe_config: Optional[str],
    universe_set: str,
    universe_groups: Optional[str],
    start_year: str,
    start_month: str,
    end_year: Optional[str],
    end_month: Optional[str],
    data_dir: str,
    parquet_dir: str,
    docker: bool,
) -> int:
    if universe_config:
        from src.data_tools.universe_config import load_universe_config

        cfg = load_universe_config(universe_config)
        groups = (
            [g.strip() for g in str(universe_groups).split(",") if g.strip()]
            if universe_groups
            else None
        )
        resolved = cfg.resolve_symbols_usdt(
            universe_set=str(universe_set), groups=groups
        )
        symbols = ",".join(resolved)

    args = [
        "--data-dir",
        data_dir,
        "--parquet-dir",
        parquet_dir,
        "--symbols",
        *[s for s in symbols.split(",") if s.strip()],
        "--start-year",
        str(start_year),
        "--start-month",
        str(start_month),
        "--yes",
    ]
    if end_year:
        args.extend(["--end-year", str(end_year)])
    if end_month:
        args.extend(["--end-month", str(end_month)])

    return run_script("src/data_tools/download_training_data.py", args, docker=docker)


def _data_convert_impl(
    *,
    cleanup: bool,
    input_dir: Optional[str],
    output_dir: Optional[str],
    backup_dir: Optional[str],
    pattern: Optional[str],
    force: bool,
    docker: bool,
) -> int:
    args = ["--cleanup", "yes" if cleanup else "no"]
    if pattern:
        args.extend(["--pattern", str(pattern)])
    if input_dir:
        args.extend(["--input-dir", input_dir])
    if output_dir:
        args.extend(["--output-dir", output_dir])
    if backup_dir:
        args.extend(["--backup-dir", backup_dir])
    if force:
        args.append("--force")
    return run_python_module("src.data_tools.zip_to_parquet", args, docker=docker)


@data.command("download")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.option(
    "--universe-config",
    default=None,
    help="YAML universe config (if set, overrides --symbols). Example: config/download/crypto_4h_token_universe_groups.yaml",
)
@click.option(
    "--universe-set",
    default="starter_a",
    help="universe_sets key inside universe config (e.g. starter_a/expanded_b).",
)
@click.option(
    "--universe-groups",
    default=None,
    help="Optional comma-separated groups to include (e.g. highcap,alt,meme). Default: all groups in the set.",
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
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def data_download(
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    start_year,
    start_month,
    end_year,
    end_month,
    data_dir,
    parquet_dir,
    docker,
):
    """Download Binance monthly aggTrades data."""
    code = _data_download_impl(
        symbols=symbols,
        universe_config=universe_config,
        universe_set=universe_set,
        universe_groups=universe_groups,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        data_dir=data_dir,
        parquet_dir=parquet_dir,
        docker=docker,
    )
    sys.exit(code)


@data.command("download-funding-rate")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.option(
    "--universe-config",
    default=None,
    help="YAML universe config (if set, overrides --symbols). Example: config/download/crypto_4h_token_universe_groups.yaml",
)
@click.option("--universe-set", default="starter_a")
@click.option("--universe-groups", default=None)
@click.option("--start-year", default="2023", help="Start year")
@click.option("--start-month", default="1", help="Start month")
@click.option("--end-year", help="End year (default: current)")
@click.option("--end-month", help="End month (default: current)")
@click.option(
    "--data-dir",
    default="data/funding_rate/zip",
    help="Output directory for fundingRate ZIP files",
)
@click.option(
    "--parquet-dir",
    default="data/funding_rate/parquet",
    help="Output directory for fundingRate Parquet",
)
@click.option(
    "--sleep-sec",
    type=float,
    default=0.2,
    show_default=True,
    help="Sleep between requests (rate-limit friendly)",
)
@click.option(
    "--progress-every",
    type=int,
    default=25,
    show_default=True,
    help="Print progress every N tasks (0 disables)",
)
@click.option("--force/--no-force", default=False, show_default=True)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def data_download_funding_rate(
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    start_year,
    start_month,
    end_year,
    end_month,
    data_dir,
    parquet_dir,
    sleep_sec,
    progress_every,
    force,
    docker,
):
    """Download Binance monthly fundingRate data."""
    if universe_config:
        from src.data_tools.universe_config import load_universe_config

        cfg = load_universe_config(universe_config)
        groups = (
            [g.strip() for g in str(universe_groups).split(",") if g.strip()]
            if universe_groups
            else None
        )
        resolved = cfg.resolve_symbols_usdt(
            universe_set=str(universe_set), groups=groups
        )
        symbols = ",".join(resolved)

    args = [
        "--data-dir",
        data_dir,
        "--parquet-dir",
        parquet_dir,
        "--symbols",
        *[s for s in symbols.split(",") if s.strip()],
        "--start-year",
        str(start_year),
        "--start-month",
        str(start_month),
        "--sleep-sec",
        str(sleep_sec),
        "--progress-every",
        str(progress_every),
    ]
    if force:
        args.append("--force")
    if end_year:
        args.extend(["--end-year", str(end_year)])
    if end_month:
        args.extend(["--end-month", str(end_month)])

    sys.exit(run_script("src/data_tools/download_funding_rate.py", args))


@data.command("update-market-cap")
@click.option("--config", default="config/data/market_cap.yaml", show_default=True)
@click.option(
    "--symbols",
    default="",
    help="Optional comma-separated symbols (default: universe+config)",
)
@click.option(
    "--output-dir", default="", help="Override output dir (default: config.data_dir)"
)
@click.option("--write-manifest/--no-write-manifest", default=True, show_default=True)
@click.option("--force/--no-force", default=False, show_default=True)
@click.option("--max-age-days", type=int, default=1, show_default=True)
@click.option("--sleep-sec", type=float, default=0.1, show_default=True)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def data_update_market_cap(
    config,
    symbols,
    output_dir,
    write_manifest,
    force,
    max_age_days,
    sleep_sec,
    docker,
):
    """Update market cap snapshots (static) or series (daily) per config, with skip-on-fresh behavior."""
    args = [
        "--config",
        config,
        "--max-age-days",
        str(max_age_days),
        "--sleep-sec",
        str(sleep_sec),
    ]
    if symbols:
        args.extend(["--symbols", symbols])
    if output_dir:
        args.extend(["--output-dir", output_dir])
    if write_manifest:
        args.append("--write-manifest")
    if force:
        args.append("--force")

    sys.exit(run_script("scripts/update_market_cap.py", args))


@data.command("convert")
@click.option(
    "--cleanup/--no-cleanup", default=True, help="Clean up ZIP files after conversion"
)
@click.option(
    "--pattern",
    default=None,
    help="Optional ZIP glob pattern to convert a subset (example: BNBUSDT-aggTrades-2024-*.zip).",
)
@click.option(
    "--input-dir", default=None, help="ZIP input directory (default: data/agg_data)"
)
@click.option(
    "--output-dir",
    default=None,
    help="Parquet output directory (default: data/parquet_data)",
)
@click.option(
    "--backup-dir",
    default=None,
    help="Optional backup directory for ZIPs (default: disabled; avoid disk blowups).",
)
@click.option("--force/--no-force", default=False, show_default=True)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def data_convert(cleanup, pattern, input_dir, output_dir, backup_dir, force, docker):
    """Convert downloaded ZIPs to Parquet format."""
    code = _data_convert_impl(
        cleanup=cleanup,
        pattern=pattern,
        input_dir=input_dir,
        output_dir=output_dir,
        backup_dir=backup_dir,
        force=force,
        docker=docker,
    )
    sys.exit(code)


@data.command("pipeline")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
@click.pass_context
def data_pipeline(ctx, symbols, docker):
    """Download and convert data (full pipeline)."""
    code = _data_download_impl(
        symbols=symbols,
        universe_config=None,
        universe_set="starter_a",
        universe_groups=None,
        start_year="2023",
        start_month="1",
        end_year=None,
        end_month=None,
        data_dir="data/agg_data",
        parquet_dir="data/parquet_data",
        docker=docker,
    )
    if code != 0:
        sys.exit(code)
    code = _data_convert_impl(
        cleanup=True,
        pattern=None,
        input_dir=None,
        output_dir=None,
        backup_dir=None,
        force=False,
        docker=docker,
    )
    sys.exit(code)


@data.command("pipeline-universe")
@click.option(
    "--universe-config",
    default="config/download/crypto_4h_token_universe_groups.yaml",
    help="Universe YAML config to drive download+convert.",
)
@click.option("--universe-set", default="starter_a")
@click.option("--universe-groups", default=None)
@click.option("--start-year", default="2023")
@click.option("--start-month", default="1")
@click.option("--end-year", default=None)
@click.option("--end-month", default=None)
@click.option("--data-dir", default="data/agg_data")
@click.option("--parquet-dir", default="data/parquet_data")
@click.option("--cleanup/--no-cleanup", default=True)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def data_pipeline_universe(
    universe_config,
    universe_set,
    universe_groups,
    start_year,
    start_month,
    end_year,
    end_month,
    data_dir,
    parquet_dir,
    cleanup,
    docker,
):
    """Download+convert using universe config (non-interactive)."""
    code = _data_download_impl(
        symbols="",
        universe_config=universe_config,
        universe_set=universe_set,
        universe_groups=universe_groups,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        data_dir=data_dir,
        parquet_dir=parquet_dir,
        docker=docker,
    )
    if code != 0:
        sys.exit(code)
    code = _data_convert_impl(
        cleanup=cleanup,
        pattern=None,
        input_dir=data_dir,
        output_dir=parquet_dir,
        backup_dir=None,
        force=False,
        docker=docker,
    )
    sys.exit(code)


@data.command("check-month-coverage")
@click.option(
    "--symbol",
    default="",
    help="Optional symbol like BNBUSDT (default: all symbols summary)",
)
@click.option(
    "--start",
    "start_ym",
    default="2023-01",
    show_default=True,
    help="Start YYYY-MM (inclusive)",
)
@click.option(
    "--end",
    "end_ym",
    default=datetime.now(timezone.utc).strftime("%Y-%m"),
    show_default=True,
    help="End YYYY-MM (inclusive)",
)
@click.option(
    "--zip-dir", default="data/agg_data", show_default=True, help="ZIP directory"
)
@click.option(
    "--parquet-dir",
    default="data/parquet_data",
    show_default=True,
    help="Parquet directory",
)
@click.option("--show-missing/--no-show-missing", default=False, show_default=True)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def data_check_month_coverage(
    symbol: str,
    start_ym: str,
    end_ym: str,
    zip_dir: str,
    parquet_dir: str,
    show_missing: bool,
    docker: bool,
):
    """Check YYYY-MM coverage for monthly aggTrades ZIP + Parquet datasets."""
    args = [
        "--start",
        str(start_ym),
        "--end",
        str(end_ym),
        "--zip-dir",
        str(zip_dir),
        "--parquet-dir",
        str(parquet_dir),
    ]
    if symbol and str(symbol).strip():
        args.extend(["--symbol", str(symbol).strip()])
    if show_missing:
        args.append("--show-missing")
    sys.exit(run_script("scripts/check_month_coverage.py", args, docker=docker))


# =============================================================================
# Train Commands
# =============================================================================


@cli.group()
def train():
    """Model training commands."""
    pass


@cli.group()
def nnmultihead():
    """NN multi-head base model (path primitives) commands."""
    pass


@cli.group("feature-store")
def feature_store():
    """Shared FeatureStore (monthly partitioned) commands."""
    pass


@feature_store.command("build")
@click.option(
    "--config",
    "-c",
    required=True,
    help="Config directory containing features.yaml (tree or nn).",
)
@click.option(
    "--symbols",
    "-s",
    default=None,
    help="Comma-separated symbols. If not provided, use --universe-config to load all symbols.",
)
@click.option(
    "--universe-config",
    default=None,
    help="Path to universe config YAML (e.g., config/download/crypto_4h_token_universe_groups.yaml). "
    "If provided and --symbols is not set, will load all symbols from the config.",
)
@click.option(
    "--universe-set",
    default="starter_a",
    help="Universe set name to use from universe config (default: starter_a).",
)
@click.option(
    "--universe-groups",
    default=None,
    help="Comma-separated groups to include (e.g., 'highcap,alt'). If not specified, includes all groups.",
)
@click.option("--timeframe", "-t", required=True, help="Timeframe (e.g., 240T).")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD) optional")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD) optional")
@click.option(
    "--root",
    "feature_store_root",
    default="feature_store",
    help="FeatureStore root dir.",
)
@click.option(
    "--layer",
    default=None,
    help="FeatureStore layer (dataset id). If not specified, auto-generated from config content. "
    "You can pass a versioned name like heavy_v6/base_v6 for manual invalidation/rebuild.",
)
@click.option(
    "--warmup-months", type=int, default=0, help="Warmup calendar months (optional)."
)
@click.option(
    "--warmup-bars",
    type=int,
    default=0,
    help="Fallback warmup by bars if warmup-months=0.",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def feature_store_build(
    config,
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    timeframe,
    data_path,
    start_date,
    end_date,
    feature_store_root,
    layer,
    warmup_months,
    warmup_bars,
    docker,
):
    """Build monthly FeatureStore from a config directory (shared infra for tree+nn)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--timeframe",
        timeframe,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--root",
        (
            f"/workspace/{feature_store_root}"
            if use_workspace_prefix
            else feature_store_root
        ),
        "--warmup-months",
        str(int(warmup_months)),
        "--warmup-bars",
        str(int(warmup_bars)),
    ]
    if symbols:
        args.extend(["--symbols", symbols])
    if universe_config:
        args.extend(
            [
                "--universe-config",
                (
                    f"/workspace/{universe_config}"
                    if use_workspace_prefix
                    else universe_config
                ),
            ]
        )
        args.extend(["--universe-set", universe_set])
        if universe_groups:
            args.extend(["--universe-groups", universe_groups])
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    # Only pass --layer if explicitly provided (None means auto-generate in script)
    if layer is not None:
        args.extend(["--layer", layer])
    sys.exit(
        run_script("scripts/build_feature_store_from_config.py", args, docker=docker)
    )


@cli.group()
def rule():
    """Rule router commands (3-action: NO_TRADE/MEAN/TREND)."""
    pass


@cli.group()
def rl():
    """RL-ready router tooling (3-action BC/shadow/counterfactual/FSM) using logs."""
    pass


# =============================================================================
# Unified Search Workflows (Tree + nnmultihead)
# =============================================================================


@cli.group()
def search():
    """Unified search workflows (tree strategies and nnmultihead primitives)."""
    pass


@search.command("tree")
@click.option(
    "--strategies",
    default="sr_reversal_rr_reg_long,sr_breakout,compression_breakout,trend_following",
    show_default=True,
    help="Comma-separated strategy directory names under config/strategies/ (can be a single strategy).",
)
@click.option("--tag", default=None, help="Tag for outputs.")
@click.option(
    "--symbol", "-s", default="BTCUSDT", show_default=True, help="Trading symbol"
)
@click.option("--timeframe", "-t", default="240T", show_default=True, help="Timeframe")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.3", show_default=True, help="Test set ratio")
@click.option(
    "--seeds", default="1,2,3,4,5", show_default=True, help="Comma-separated seeds"
)
@click.option(
    "--objective", default="Sharpe_mean", show_default=True, help="Objective metric"
)
@click.option(
    "--min-trades", default="10", show_default=True, help="Min trades_mean constraint"
)
@click.option(
    "--max-steps",
    default="5",
    show_default=True,
    help="Max steps (beam depth / greedy steps)",
)
@click.option(
    "--search-algo",
    default="pipeline",
    type=click.Choice(["greedy", "halving", "beam", "sffs", "pipeline"]),
    show_default=True,
    help="Search algorithm. Recommended: pipeline (SH prefilter -> Beam -> SFFS prune).",
)
@click.option("--expand-semantic-singletons", is_flag=True, default=False)
@click.option("--regen-poolb", is_flag=True, default=False)
@click.option("--rerun-search", is_flag=True, default=False)
@click.option("--report-only", is_flag=True, default=False)
def search_tree(
    strategies,
    tag,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    seeds,
    objective,
    min_trades,
    max_steps,
    search_algo,
    expand_semantic_singletons,
    regen_poolb,
    rerun_search,
    report_only,
):
    """One-shot tree workflow: PoolB(factor-eval) + feature-group-search + writeback + report."""
    script = PROJECT_ROOT / "scripts" / "run_poolb_semantic_search.py"
    cmd = [
        sys.executable,
        str(script),
        "--strategies",
        str(strategies),
        "--symbol",
        str(symbol),
        "--timeframe",
        str(timeframe),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--test-size",
        str(test_size),
        "--seeds",
        str(seeds),
        "--objective",
        str(objective),
        "--min-trades",
        str(min_trades),
        "--max-steps",
        str(max_steps),
        "--search-algo",
        str(search_algo),
    ]
    if tag:
        cmd.extend(["--tag", str(tag)])
    if expand_semantic_singletons:
        cmd.append("--expand-semantic-singletons")
    if regen_poolb:
        cmd.append("--regen-poolb")
    if rerun_search:
        cmd.append("--rerun-search")
    if report_only:
        cmd.append("--report-only")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


@search.command("nn")
@click.option(
    "--config",
    "config_dir",
    default="config/nnmultihead/path_primitives_4h_80h_min",
    show_default=True,
    help="Base nnmultihead config directory.",
)
@click.option(
    "--symbols",
    default="BTCUSDT,ETHUSDT",
    show_default=True,
    help="Comma-separated symbols",
)
@click.option("--timeframe", default="240T", show_default=True)
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
@click.option("--features-store-root", default="feature_store", show_default=True)
@click.option("--features-store-layer", required=True)
@click.option("--tag", default=None, help="Tag for outputs (default: auto).")
@click.option(
    "--objective",
    default="dir_auc",
    show_default=True,
    help="nn objective metric (metrics.json key)",
)
@click.option(
    "--search-algo",
    default="pipeline",
    type=click.Choice(["greedy", "halving", "beam", "sffs", "pipeline"]),
    show_default=True,
)
@click.option("--epochs", type=int, default=10, show_default=True)
@click.option(
    "--exclude-columns",
    default=None,
    help="Comma-separated columns to exclude from MLP input. If omitted, use config's feature_pipeline.exclude_columns (recommended).",
)
@click.option("--expand-semantic-singletons", is_flag=True, default=False)
@click.option(
    "--run-train/--no-run-train",
    default=True,
    show_default=True,
    help="Train best config after search",
)
def search_nn(
    config_dir,
    symbols,
    timeframe,
    start_date,
    end_date,
    features_store_root,
    features_store_layer,
    tag,
    objective,
    search_algo,
    epochs,
    exclude_columns,
    expand_semantic_singletons,
    run_train,
):
    """One-shot nn workflow: primitives PoolB + nn feature-group-search (+ optional train)."""
    script = PROJECT_ROOT / "scripts" / "run_nnmultihead_search.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config_dir),
        "--symbols",
        str(symbols),
        "--timeframe",
        str(timeframe),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--features-store-root",
        str(features_store_root),
        "--features-store-layer",
        str(features_store_layer),
        "--objective",
        str(objective),
        "--search-algo",
        str(search_algo),
        "--epochs",
        str(int(epochs)),
    ]
    # CLI override only (preferred default lives in config/features.yaml)
    if exclude_columns is not None:
        cmd.extend(["--exclude-columns", str(exclude_columns)])
    if tag:
        cmd.extend(["--tag", str(tag)])
    if expand_semantic_singletons:
        cmd.append("--expand-semantic-singletons")
    if run_train:
        cmd.append("--run-train")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


@rl.group("exec")
def rl_exec():
    """Execution control tooling (invariants/kill-switch/chaos) on Router logs."""
    pass


@rl_exec.command("control-check")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Logs .csv/.parquet (symbol,timestamp,mode,ret_mean,ret_trend,...)",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Output directory for artifacts (report/metrics/csv).",
)
@click.option("--entry-delay", type=int, default=1)
@click.option("--cost-per-turnover", type=float, default=0.0002)
@click.option("--slippage-bps", type=float, default=0.0)
@click.option("--max-dd", type=float, default=0.35)
@click.option("--max-turnover-mean", type=float, default=0.35)
@click.option("--max-turnover-p95", type=float, default=1.0)
@click.option("--max-cost-mean", type=float, default=0.002)
@click.option("--max-cost-p95", type=float, default=0.01)
@click.option("--max-nan-ratio", type=float, default=0.001)
@click.option("--max-abs-return", type=float, default=0.5)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_exec_control_check(
    logs_path,
    out_dir,
    entry_delay,
    cost_per_turnover,
    slippage_bps,
    max_dd,
    max_turnover_mean,
    max_turnover_p95,
    max_cost_mean,
    max_cost_p95,
    max_nan_ratio,
    max_abs_return,
    docker,
):
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--out",
        f"/workspace/{out_dir}" if use_workspace_prefix else out_dir,
        "--entry-delay",
        str(int(entry_delay)),
        "--cost-per-turnover",
        str(float(cost_per_turnover)),
        "--slippage-bps",
        str(float(slippage_bps)),
        "--max-dd",
        str(float(max_dd)),
        "--max-turnover-mean",
        str(float(max_turnover_mean)),
        "--max-turnover-p95",
        str(float(max_turnover_p95)),
        "--max-cost-mean",
        str(float(max_cost_mean)),
        "--max-cost-p95",
        str(float(max_cost_p95)),
        "--max-nan-ratio",
        str(float(max_nan_ratio)),
        "--max-abs-return",
        str(float(max_abs_return)),
    ]
    sys.exit(run_script("scripts/rl_exec_control_check.py", args, docker=docker))


@rl_exec.command("chaos-test")
@click.option("--logs", "logs_path", required=True, help="Logs .csv/.parquet")
@click.option(
    "--out", "out_dir", required=True, help="Output directory root (baseline/chaos)."
)
@click.option("--seed", type=int, default=0)
@click.option("--nan-ratio", type=float, default=0.0)
@click.option("--return-scale", type=float, default=1.0)
@click.option("--slippage-bps", type=float, default=0.0)
@click.option("--cost-per-turnover", type=float, default=0.0002)
@click.option("--entry-delay", type=int, default=1)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_exec_chaos_test(
    logs_path,
    out_dir,
    seed,
    nan_ratio,
    return_scale,
    slippage_bps,
    cost_per_turnover,
    entry_delay,
    docker,
):
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--out",
        f"/workspace/{out_dir}" if use_workspace_prefix else out_dir,
        "--seed",
        str(int(seed)),
        "--nan-ratio",
        str(float(nan_ratio)),
        "--return-scale",
        str(float(return_scale)),
        "--slippage-bps",
        str(float(slippage_bps)),
        "--cost-per-turnover",
        str(float(cost_per_turnover)),
        "--entry-delay",
        str(int(entry_delay)),
    ]
    sys.exit(run_script("scripts/rl_exec_chaos_test.py", args, docker=docker))


@rl.group("router")
def rl_router():
    """Router diagnostics utilities (multi-symbol, drift, consistency)."""
    pass


@rl_router.command("diagnose")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Logs .csv/.parquet (symbol,timestamp,mode,ret_mean,ret_trend,...)",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Output directory for artifacts (report/metrics/csv).",
)
@click.option(
    "--rolling-window",
    type=int,
    default=300,
    help="Rolling window (steps) for drift metrics.",
)
@click.option(
    "--rolling-min-periods",
    type=int,
    default=60,
    help="Min periods for rolling drift metrics.",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_router_diagnose(logs_path, out_dir, rolling_window, rolling_min_periods, docker):
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--out",
        f"/workspace/{out_dir}" if use_workspace_prefix else out_dir,
        "--rolling-window",
        str(int(rolling_window)),
        "--rolling-min-periods",
        str(int(rolling_min_periods)),
    ]
    sys.exit(run_script("scripts/rl_router_diagnose.py", args, docker=docker))


@rl_router.command("embed-eval")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Logs .csv/.parquet with mode + ret_mean/ret_trend + head_*",
)
@click.option(
    "--out", "out_dir", required=True, help="Output directory root for A/B artifacts."
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.7,
    help="Train ratio per symbol (time-ordered).",
)
@click.option(
    "--regime-buckets",
    type=int,
    default=4,
    help="Number of regime buckets for one-hot embedding.",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_router_embed_eval(logs_path, out_dir, train_ratio, regime_buckets, docker):
    """A/B eval: BC baseline vs +regime(one-hot) embedding."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--out",
        f"/workspace/{out_dir}" if use_workspace_prefix else out_dir,
        "--train_ratio",
        str(float(train_ratio)),
        "--regime-buckets",
        str(int(regime_buckets)),
    ]
    sys.exit(run_script("scripts/rl_router_embed_eval.py", args, docker=docker))


@rl.command("build-logs-3action")
@click.option(
    "--preds",
    "preds_path",
    required=True,
    help="Preds file/dir from nnmultihead predict (preds_*.parquet)",
)
@click.option(
    "--mode",
    "mode_path",
    default=None,
    help="Optional mode file/dir from mlbot rule mode-3action",
)
@click.option(
    "--symbols",
    "-s",
    default=None,
    help="Optional symbols filter (comma-separated). If omitted, infer from preds.",
)
@click.option("--data-path", default="data/parquet_data", help="Raw data directory")
@click.option("--timeframe", default="240T", help="Timeframe (must match preds)")
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
@click.option(
    "--model",
    "model_path",
    default=None,
    help="Optional model.pt to infer preds_in_log1p",
)
@click.option(
    "--preds-in-log1p",
    type=click.Choice(["yes", "no"]),
    default=None,
    help="Override preds space (yes=log1p)",
)
@click.option(
    "--returns-source",
    type=click.Choice(["momentum_proxy", "rr_execution", "vectorbt_execution"]),
    default="momentum_proxy",
    help="How to build ret_mean/ret_trend",
)
@click.option(
    "--momentum-lookback",
    type=int,
    default=5,
    help="Lookback for momentum proxy used in ret_mean/ret_trend",
)
@click.option(
    "--vbt-top-quantile",
    type=float,
    default=0.05,
    help="vectorbt: top quantile for long entries (regression score)",
)
@click.option(
    "--vbt-bottom-quantile",
    type=float,
    default=0.05,
    help="vectorbt: bottom quantile for short entries (regression score)",
)
@click.option(
    "--vbt-entry-mode",
    type=click.Choice(["level", "cross"]),
    default="cross",
    help="vectorbt: entry mode",
)
@click.option(
    "--vbt-fee", type=float, default=0.0004, help="vectorbt: fee per trade (fraction)"
)
@click.option(
    "--vbt-slippage", type=float, default=0.0001, help="vectorbt: slippage (fraction)"
)
@click.option("--vbt-freq", default="4H", help="vectorbt: freq string, e.g. 4H/1H/15T")
@click.option(
    "--symbol-profiles-json",
    default=None,
    help='Per-symbol profile mapping JSON, e.g. {"BTCUSDT":"btc","DOGEUSDT":"meme"}',
)
@click.option("--default-profile", default="standard", help="Default market profile.")
@click.option(
    "--rr-profile-overrides-json",
    default=None,
    help='RR profile overrides JSON, e.g. {"meme":{"take_profit_r":2.5}}',
)
@click.option(
    "--vbt-profile-overrides-json",
    default=None,
    help='vectorbt profile overrides JSON, e.g. {"btc":{"fee":0.0002}}',
)
@click.option(
    "--output", "output_path", required=True, help="Output logs path (.parquet/.csv)"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_build_logs_3action(
    preds_path,
    mode_path,
    symbols,
    data_path,
    timeframe,
    start_date,
    end_date,
    model_path,
    preds_in_log1p,
    returns_source,
    momentum_lookback,
    vbt_top_quantile,
    vbt_bottom_quantile,
    vbt_entry_mode,
    vbt_fee,
    vbt_slippage,
    vbt_freq,
    symbol_profiles_json,
    default_profile,
    rr_profile_overrides_json,
    vbt_profile_overrides_json,
    output_path,
    docker,
):
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--preds",
        f"/workspace/{preds_path}" if use_workspace_prefix else preds_path,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--timeframe",
        str(timeframe),
        "--output",
        f"/workspace/{output_path}" if use_workspace_prefix else output_path,
        "--momentum-lookback",
        str(int(momentum_lookback)),
        "--returns-source",
        str(returns_source),
        "--vbt-top-quantile",
        str(float(vbt_top_quantile)),
        "--vbt-bottom-quantile",
        str(float(vbt_bottom_quantile)),
        "--vbt-entry-mode",
        str(vbt_entry_mode),
        "--vbt-fee",
        str(float(vbt_fee)),
        "--vbt-slippage",
        str(float(vbt_slippage)),
        "--vbt-freq",
        str(vbt_freq),
        "--default-profile",
        str(default_profile),
    ]
    if symbol_profiles_json:
        args.extend(["--symbol-profiles-json", str(symbol_profiles_json)])
    if rr_profile_overrides_json:
        args.extend(["--rr-profile-overrides-json", str(rr_profile_overrides_json)])
    if vbt_profile_overrides_json:
        args.extend(["--vbt-profile-overrides-json", str(vbt_profile_overrides_json)])
    if mode_path:
        args.extend(
            ["--mode", f"/workspace/{mode_path}" if use_workspace_prefix else mode_path]
        )
    if symbols:
        args.extend(["--symbols", str(symbols)])
    if start_date:
        args.extend(["--start-date", str(start_date)])
    if end_date:
        args.extend(["--end-date", str(end_date)])
    if model_path:
        args.extend(
            [
                "--model",
                f"/workspace/{model_path}" if use_workspace_prefix else model_path,
            ]
        )
    if preds_in_log1p:
        args.extend(["--preds-in-log1p", str(preds_in_log1p)])

    sys.exit(run_script("scripts/rl_build_logs_3action.py", args, docker=docker))


@rule.command("mode-3action")
@click.option(
    "--preds",
    required=True,
    help="Preds file (.parquet/.csv) or directory of per-symbol preds_*.parquet",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    help="Optional model.pt to infer whether preds are log1p targets",
)
@click.option(
    "--preds-in-log1p",
    type=click.Choice(["yes", "no"]),
    default=None,
    help="Override preds space (yes=log1p)",
)
@click.option(
    "--output", "output_path", required=True, help="Output path (.parquet or .csv)"
)
@click.option("--mfe-min", type=float, default=None)
@click.option("--eff-min", type=float, default=None)
@click.option("--dir-conf-trend-min", type=float, default=None)
@click.option("--mfe-trend-min", type=float, default=None)
@click.option("--ttm-trend-min", type=float, default=None)
@click.option("--eff-mean-min", type=float, default=None)
@click.option("--ttm-mean-max", type=float, default=None)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rule_mode_3action(
    preds,
    model_path,
    preds_in_log1p,
    output_path,
    mfe_min,
    eff_min,
    dir_conf_trend_min,
    mfe_trend_min,
    ttm_trend_min,
    eff_mean_min,
    ttm_mean_max,
    docker,
):
    """Generate mode labels (NO_TRADE/MEAN/TREND) from nnmultihead heads."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--preds",
        f"/workspace/{preds}" if use_workspace_prefix else preds,
        "--output",
        f"/workspace/{output_path}" if use_workspace_prefix else output_path,
    ]
    if model_path:
        args.extend(
            [
                "--model",
                f"/workspace/{model_path}" if use_workspace_prefix else model_path,
            ]
        )
    if preds_in_log1p:
        args.extend(["--preds-in-log1p", preds_in_log1p])
    # thresholds
    if mfe_min is not None:
        args.extend(["--mfe-min", str(mfe_min)])
    if eff_min is not None:
        args.extend(["--eff-min", str(eff_min)])
    if dir_conf_trend_min is not None:
        args.extend(["--dir-conf-trend-min", str(dir_conf_trend_min)])
    if mfe_trend_min is not None:
        args.extend(["--mfe-trend-min", str(mfe_trend_min)])
    if ttm_trend_min is not None:
        args.extend(["--ttm-trend-min", str(ttm_trend_min)])
    if eff_mean_min is not None:
        args.extend(["--eff-mean-min", str(eff_mean_min)])
    if ttm_mean_max is not None:
        args.extend(["--ttm-mean-max", str(ttm_mean_max)])

    sys.exit(run_script("scripts/rule_mode_3action.py", args, docker=docker))


@rl.command("shadow-eval-3action")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Logs .csv/.parquet with columns: symbol,timestamp,mode,head_*",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Output directory for artifacts (metrics/report).",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.7,
    help="Train ratio per symbol (time-ordered).",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_shadow_eval_3action(logs_path, out_dir, train_ratio, docker):
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--out",
        f"/workspace/{out_dir}" if use_workspace_prefix else out_dir,
        "--train_ratio",
        str(float(train_ratio)),
    ]
    sys.exit(run_script("scripts/rl_shadow_eval_3action.py", args, docker=docker))


@rl.command("counterfactual-eval-3action")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Logs .csv/.parquet with mode + ret_mean/ret_trend + head_*",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Output directory for artifacts (metrics/report).",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.7,
    help="Train ratio per symbol (time-ordered).",
)
@click.option("--entry-delay", type=int, default=0, help="Entry delay steps for sim.")
@click.option(
    "--cost-per-turnover", type=float, default=0.0, help="Cost per turnover unit."
)
@click.option(
    "--slippage-bps", type=float, default=0.0, help="Slippage bps per abs exposure."
)
@click.option(
    "--preds-in-log1p/--preds-not-in-log1p",
    default=True,
    help="Whether head_mfe/head_mae/head_t_to_mfe are in log1p space (affects Router diagnostics only).",
)
@click.option(
    "--router-mfe-min",
    type=float,
    default=None,
    help="Router threshold override: mfe_min (for counterfactual report diagnostics).",
)
@click.option(
    "--router-eff-min",
    type=float,
    default=None,
    help="Router threshold override: eff_min (for counterfactual report diagnostics).",
)
@click.option(
    "--router-dir-conf-trend-min",
    type=float,
    default=None,
    help="Router threshold override: dir_conf_trend_min (for counterfactual report diagnostics).",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_counterfactual_eval_3action(
    logs_path,
    out_dir,
    train_ratio,
    entry_delay,
    cost_per_turnover,
    slippage_bps,
    preds_in_log1p,
    router_mfe_min,
    router_eff_min,
    router_dir_conf_trend_min,
    docker,
):
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--out",
        f"/workspace/{out_dir}" if use_workspace_prefix else out_dir,
        "--train_ratio",
        str(float(train_ratio)),
        "--entry_delay",
        str(int(entry_delay)),
        "--cost_per_turnover",
        str(float(cost_per_turnover)),
        "--slippage_bps",
        str(float(slippage_bps)),
        "--preds-in-log1p",
        "1" if preds_in_log1p else "0",
    ]
    if router_mfe_min is not None:
        args.extend(["--router-mfe-min", str(float(router_mfe_min))])
    if router_eff_min is not None:
        args.extend(["--router-eff-min", str(float(router_eff_min))])
    if router_dir_conf_trend_min is not None:
        args.extend(["--router-dir-conf-trend-min", str(float(router_dir_conf_trend_min))])
    sys.exit(
        run_script("scripts/rl_counterfactual_eval_3action.py", args, docker=docker)
    )


@rl.command("fsm-decide")
@click.option(
    "--metrics",
    "metrics_path",
    required=True,
    help="metrics.json produced by counterfactual-eval-3action",
)
@click.option(
    "--state",
    default="RL_CANDIDATE",
    help="Initial FSM state: RULE/RL_CANDIDATE/RL_ACTIVE/RL_SUSPENDED",
)
@click.option(
    "--promote-days",
    type=int,
    default=10,
    help="Consecutive ok windows required to promote.",
)
@click.option(
    "--cooldown-days", type=int, default=20, help="Cooldown windows after suspension."
)
@click.option(
    "--dd-ratio-max",
    type=float,
    default=1.2,
    help="Hard gate: dd_RL > dd_Rule * dd_ratio_max",
)
@click.option(
    "--switch-ratio-max",
    type=float,
    default=2.0,
    help="Hard gate: switch_RL > switch_Rule * switch_ratio_max",
)
@click.option(
    "--pnl-dd-margin",
    type=float,
    default=0.15,
    help="Drift gate: (PnL/DD)_RL < (PnL/DD)_Rule * (1 - margin)",
)
@click.option(
    "--sharpe-ratio-min",
    type=float,
    default=0.8,
    help="Hard gate: sharpe_RL < sharpe_Rule * sharpe_ratio_min",
)
@click.option(
    "--sharpe-min-abs",
    type=float,
    default=None,
    help="Hard gate: sharpe_RL < sharpe_min_abs (optional)",
)
@click.option(
    "--sortino-ratio-min",
    type=float,
    default=0.8,
    help="Hard gate: sortino_RL < sortino_Rule * sortino_ratio_min",
)
@click.option(
    "--sortino-min-abs",
    type=float,
    default=None,
    help="Hard gate: sortino_RL < sortino_min_abs (optional)",
)
@click.option(
    "--ann-vol-ratio-max",
    type=float,
    default=2.0,
    help="Hard gate: ann_vol_RL > ann_vol_Rule * ann_vol_ratio_max",
)
@click.option(
    "--out", "out_path", default=None, help="Optional path to write decision json."
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_fsm_decide(
    metrics_path,
    state,
    promote_days,
    cooldown_days,
    dd_ratio_max,
    switch_ratio_max,
    pnl_dd_margin,
    sharpe_ratio_min,
    sharpe_min_abs,
    sortino_ratio_min,
    sortino_min_abs,
    ann_vol_ratio_max,
    out_path,
    docker,
):
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--metrics",
        f"/workspace/{metrics_path}" if use_workspace_prefix else metrics_path,
        "--state",
        str(state),
        "--promote_days",
        str(int(promote_days)),
        "--cooldown_days",
        str(int(cooldown_days)),
        "--dd_ratio_max",
        str(float(dd_ratio_max)),
        "--switch_ratio_max",
        str(float(switch_ratio_max)),
        "--pnl_dd_margin",
        str(float(pnl_dd_margin)),
        "--sharpe_ratio_min",
        str(float(sharpe_ratio_min)),
        "--sortino_ratio_min",
        str(float(sortino_ratio_min)),
        "--ann_vol_ratio_max",
        str(float(ann_vol_ratio_max)),
    ]
    if sharpe_min_abs is not None:
        args.extend(["--sharpe_min_abs", str(float(sharpe_min_abs))])
    if sortino_min_abs is not None:
        args.extend(["--sortino_min_abs", str(float(sortino_min_abs))])
    if out_path:
        args.extend(
            ["--out", f"/workspace/{out_path}" if use_workspace_prefix else out_path]
        )
    sys.exit(run_script("scripts/rl_fsm_decide.py", args, docker=docker))


@rl.command("run-e2e-3action")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Logs .csv/.parquet with mode + heads (+ ret_mean/ret_trend for counterfactual).",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Output directory root. Will create shadow/ counterfactual/ fsm_decision.json",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.7,
    help="Train ratio per symbol (time-ordered).",
)
@click.option("--entry-delay", type=int, default=0, help="Entry delay steps for sim.")
@click.option(
    "--cost-per-turnover", type=float, default=0.0, help="Cost per turnover unit."
)
@click.option(
    "--slippage-bps", type=float, default=0.0, help="Slippage bps per abs exposure."
)
@click.option(
    "--preds-in-log1p/--preds-not-in-log1p",
    default=True,
    help="Whether head_mfe/head_mae/head_t_to_mfe are in log1p space (affects Router diagnostics only).",
)
@click.option(
    "--router-mfe-min",
    type=float,
    default=None,
    help="Router threshold override: mfe_min (for counterfactual report diagnostics).",
)
@click.option(
    "--router-eff-min",
    type=float,
    default=None,
    help="Router threshold override: eff_min (for counterfactual report diagnostics).",
)
@click.option(
    "--router-dir-conf-trend-min",
    type=float,
    default=None,
    help="Router threshold override: dir_conf_trend_min (for counterfactual report diagnostics).",
)
@click.option("--fsm-state", default="RL_CANDIDATE", help="Initial FSM state.")
@click.option("--promote-days", type=int, default=10)
@click.option("--cooldown-days", type=int, default=20)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_run_e2e_3action(
    logs_path,
    out_dir,
    train_ratio,
    entry_delay,
    cost_per_turnover,
    slippage_bps,
    preds_in_log1p,
    router_mfe_min,
    router_eff_min,
    router_dir_conf_trend_min,
    fsm_state,
    promote_days,
    cooldown_days,
    docker,
):
    """
    Convenience wrapper:
      1) shadow-eval-3action -> {out}/shadow
      2) counterfactual-eval-3action -> {out}/counterfactual
      3) fsm-decide -> {out}/fsm_decision.json
    """
    use_workspace_prefix = docker and not _is_in_docker()
    logs_arg = f"/workspace/{logs_path}" if use_workspace_prefix else logs_path
    out_root = f"/workspace/{out_dir}" if use_workspace_prefix else out_dir
    shadow_out = f"{out_root}/shadow"
    cf_out = f"{out_root}/counterfactual"
    fsm_out = f"{out_root}/fsm_decision.json"

    rc = run_script(
        "scripts/rl_shadow_eval_3action.py",
        [
            "--logs",
            logs_arg,
            "--out",
            shadow_out,
            "--train_ratio",
            str(float(train_ratio)),
        ],
        docker=docker,
    )
    if rc != 0:
        sys.exit(rc)

    cf_args = [
        "--logs",
        logs_arg,
        "--out",
        cf_out,
        "--train_ratio",
        str(float(train_ratio)),
        "--entry_delay",
        str(int(entry_delay)),
        "--cost_per_turnover",
        str(float(cost_per_turnover)),
        "--slippage_bps",
        str(float(slippage_bps)),
        "--preds-in-log1p",
        "1" if preds_in_log1p else "0",
    ]
    if router_mfe_min is not None:
        cf_args.extend(["--router-mfe-min", str(float(router_mfe_min))])
    if router_eff_min is not None:
        cf_args.extend(["--router-eff-min", str(float(router_eff_min))])
    if router_dir_conf_trend_min is not None:
        cf_args.extend(
            ["--router-dir-conf-trend-min", str(float(router_dir_conf_trend_min))]
        )
    rc = run_script("scripts/rl_counterfactual_eval_3action.py", cf_args, docker=docker)
    if rc != 0:
        sys.exit(rc)

    metrics_path = f"{cf_out}/metrics.json"
    rc = run_script(
        "scripts/rl_fsm_decide.py",
        [
            "--metrics",
            metrics_path,
            "--state",
            str(fsm_state),
            "--promote_days",
            str(int(promote_days)),
            "--cooldown_days",
            str(int(cooldown_days)),
            "--out",
            fsm_out,
        ],
        docker=docker,
    )
    sys.exit(rc)


@nnmultihead.command("train")
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--config",
    "-c",
    default="config/nnmultihead/path_primitives_4h_80h_min",
    help="NN multihead config directory (features.yaml + labels.yaml + model.yaml)",
)
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD) optional")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD) optional")
@click.option(
    "--horizon-hours",
    type=float,
    default=80.0,
    help="Future horizon in hours (e.g., 80H)",
)
@click.option(
    "--bar-hours", type=float, default=4.0, help="Bar duration in hours (4H => 4)"
)
@click.option("--epochs", type=int, default=30, help="Training epochs")
@click.option("--batch-size", type=int, default=512, help="Batch size")
@click.option("--lr", type=float, default=2e-4, help="Learning rate")
@click.option("--hidden", type=int, default=256, help="MLP hidden size")
@click.option("--depth", type=int, default=2, help="MLP depth")
@click.option("--dropout", type=float, default=0.1, help="Dropout")
@click.option("--device", default=None, help="cpu|cuda (default auto)")
@click.option(
    "--feature-store-layer",
    default=None,
    help="FeatureStore layer name (e.g. heavy_v6). If not specified, auto-generated from config hash.",
)
@click.option(
    "--feature-store-root",
    default="feature_store",
    help="FeatureStore root dir (default: feature_store). Usually no need to change.",
)
@click.option(
    "--output-dir",
    default="results/nnmultihead",
    help="Output root directory for artifacts",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_train(
    symbols,
    timeframe,
    data_path,
    config,
    start_date,
    end_date,
    horizon_hours,
    bar_hours,
    epochs,
    batch_size,
    lr,
    hidden,
    depth,
    dropout,
    device,
    feature_store_layer,
    feature_store_root,
    output_dir,
    docker,
):
    """Train NN multi-head path primitives MLP and save report.html artifacts."""
    # Note: Layer name auto-generation is handled by the script itself.
    # CLI just passes the parameter as-is (None = auto-generate).
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--symbols",
        symbols,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--timeframe",
        timeframe,
        "--horizon-hours",
        str(horizon_hours),
        "--bar-hours",
        str(bar_hours),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--hidden",
        str(hidden),
        "--depth",
        str(depth),
        "--dropout",
        str(dropout),
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if device:
        args.extend(["--device", device])
    # Always use FeatureStore (monthly) for nnmultihead (fast + consistent).
    # NOTE: underlying scripts use '--features-store-layer/--features-store-root'.
    # Only pass --features-store-layer if explicitly provided (None means auto-generate in script)
    if feature_store_layer is not None:
        args.extend(["--features-store-layer", feature_store_layer])
    args.extend(
        [
            "--features-store-root",
            (
                f"/workspace/{feature_store_root}"
                if use_workspace_prefix
                else feature_store_root
            ),
        ]
    )

    sys.exit(run_script("scripts/train_path_primitives_mlp.py", args, docker=docker))


@nnmultihead.command("predict")
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--config",
    "-c",
    default="config/nnmultihead/path_primitives_4h_80h_min",
    help="NN multihead config directory (features.yaml + labels.yaml + model.yaml)",
)
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD) optional")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD) optional")
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Path to model.pt produced by nnmultihead train",
)
@click.option(
    "--output", "output_path", required=True, help="Output path (.parquet or .csv)"
)
@click.option("--device", default=None, help="cpu|cuda (default auto)")
@click.option(
    "--feature-store-layer",
    default=None,
    help="FeatureStore layer name (e.g. heavy_v6). If not specified, auto-generated from config hash.",
)
@click.option(
    "--feature-store-root",
    default="feature_store",
    help="FeatureStore root dir (default: feature_store). Usually no need to change.",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_predict(
    symbols,
    timeframe,
    data_path,
    config,
    start_date,
    end_date,
    model_path,
    output_path,
    device,
    feature_store_layer,
    feature_store_root,
    docker,
):
    """Run inference and save heads/preds for downstream Router/RL."""
    # Note: Layer name auto-generation is handled by the script itself.
    # CLI just passes the parameter as-is (None = auto-generate).
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--symbols",
        symbols,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--timeframe",
        timeframe,
        "--model",
        f"/workspace/{model_path}" if use_workspace_prefix else model_path,
        "--output",
        f"/workspace/{output_path}" if use_workspace_prefix else output_path,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if device:
        args.extend(["--device", device])
    # Always use FeatureStore (monthly) for nnmultihead (fast + consistent).
    # Only pass --features-store-layer if explicitly provided (None means auto-generate in script)
    if feature_store_layer is not None:
        args.extend(["--features-store-layer", feature_store_layer])
    args.extend(
        [
            "--features-store-root",
            (
                f"/workspace/{feature_store_root}"
                if use_workspace_prefix
                else feature_store_root
            ),
        ]
    )

    sys.exit(run_script("scripts/predict_path_primitives_mlp.py", args, docker=docker))


@nnmultihead.command("pipeline-3action-e2e")
@click.option(
    "--config",
    "-c",
    default="config/nnmultihead/path_primitives_4h_80h_min",
    help="NN multihead config directory (features.yaml + labels.yaml + model.yaml)",
)
@click.option(
    "--symbols",
    "-s",
    required=True,
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Path to model.pt produced by nnmultihead train",
)
@click.option(
    "--feature-store-root",
    default="feature_store",
    show_default=True,
    help="FeatureStore root dir",
)
@click.option(
    "--feature-store-layer",
    default=None,
    help="FeatureStore layer id (leave empty to auto-generate from config-dir)",
)
@click.option(
    "--data-path",
    default="data/parquet_data",
    show_default=True,
    help="Raw parquet data directory (used by build-logs)",
)
@click.option(
    "--returns-source",
    default="rr_execution",
    show_default=True,
    help="Execution assumption used to build ret_mean/ret_trend (e.g., rr_execution, momentum_proxy).",
)
@click.option("--out", "out_dir", required=True, help="Output directory root for this pipeline run.")
@click.option(
    "--task-spec",
    default=None,
    help="Optional TaskSpec YAML (v1). If provided, will inject MLBOT_TASK_ID / MLBOT_CONSTITUTION_YAML / MLBOT_KPI_GATE_YAML into downstream steps.",
)
@click.option("--mfe-min", type=float, default=None)
@click.option("--eff-min", type=float, default=None)
@click.option("--dir-conf-trend-min", type=float, default=None)
@click.option("--mfe-trend-min", type=float, default=None)
@click.option("--ttm-trend-min", type=float, default=None)
@click.option("--eff-mean-min", type=float, default=None)
@click.option("--ttm-mean-max", type=float, default=None)
@click.option("--train-ratio", type=float, default=0.7, show_default=True)
@click.option("--entry-delay", type=int, default=0, show_default=True)
@click.option("--cost-per-turnover", type=float, default=0.0, show_default=True)
@click.option("--slippage-bps", type=float, default=0.0, show_default=True)
@click.option(
    "--preds-in-log1p/--preds-not-in-log1p",
    default=True,
    help="Whether head_mfe/head_mae/head_t_to_mfe are in log1p space (affects Router diagnostics only).",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_pipeline_3action_e2e(
    config,
    symbols,
    timeframe,
    start_date,
    end_date,
    model_path,
    feature_store_root,
    feature_store_layer,
    data_path,
    returns_source,
    out_dir,
    task_spec,
    mfe_min,
    eff_min,
    dir_conf_trend_min,
    mfe_trend_min,
    ttm_trend_min,
    eff_mean_min,
    ttm_mean_max,
    train_ratio,
    entry_delay,
    cost_per_turnover,
    slippage_bps,
    preds_in_log1p,
    docker,
):
    """
    One-command mainline pipeline:
      nnmultihead predict -> rule mode-3action -> rl build-logs-3action -> rl run-e2e-3action

    Motivation:
    - Keep existing family commands (nnmultihead/rule/rl) for clarity and modularity.
    - Provide a smooth, single entrypoint for the recommended mainline (Rule + Execution),
      while BC/RL/FSM remain optional modules.
    """
    use_workspace_prefix = docker and not _is_in_docker()
    out_root = f"/workspace/{out_dir}" if use_workspace_prefix else out_dir
    preds_dir = f"{out_root}/preds"
    mode_path = f"{out_root}/mode_3action.parquet"
    logs_path = f"{out_root}/logs_3action.parquet"
    e2e_out = f"{out_root}/e2e"

    # -------------------------------------------------------------------------
    # P0: TaskSpec-driven enforcement injection (research/live unified)
    # -------------------------------------------------------------------------
    env_overrides = {}
    effective_config = config
    def _materialize_cfg_from_task_spec(*, ts_obj: dict, out_root: str, base_config_dir: str) -> str:
        import yaml
        import shutil
        from pathlib import Path

        fp = ts_obj.get("feature_plan") or {}
        tiers_enabled = fp.get("tiers_enabled") or []
        tier_feature_files = fp.get("tier_feature_files") or {}
        if not (isinstance(tiers_enabled, list) and isinstance(tier_feature_files, dict)):
            return base_config_dir

        # Collect feature nodes from enabled tier files
        tier_nodes = []
        for tier_name in tiers_enabled:
            k = str(tier_name).strip()
            fpath = tier_feature_files.get(k)
            if not fpath:
                continue
            p = Path(str(fpath))
            if not p.exists():
                raise click.ClickException(
                    f"TaskSpec tier_feature_files[{k}] not found: {p}"
                )
            obj = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(obj, list):
                raise click.ClickException(f"Tier file must be a YAML list: {p}")
            tier_nodes.extend([str(x).strip() for x in obj if str(x).strip()])

        if not tier_nodes:
            return base_config_dir

        base_dir = Path(base_config_dir)
        if not base_dir.exists():
            raise click.ClickException(f"Config dir not found: {base_dir}")

        derived_dir = Path(out_root) / "derived_config_from_task_spec"
        if derived_dir.exists():
            shutil.rmtree(derived_dir)
        shutil.copytree(base_dir, derived_dir)

        # Load base features.yaml and override required list.
        feat_path = derived_dir / "features.yaml"
        feat_obj = yaml.safe_load(feat_path.read_text(encoding="utf-8")) or {}
        fp2 = feat_obj.get("feature_pipeline") or {}
        req = fp2.get("requested_features") or {}
        if not isinstance(req, dict):
            req = {}
        req["required"] = sorted(set(tier_nodes))
        fp2["requested_features"] = req
        feat_obj["feature_pipeline"] = fp2

        # Rebuild minimal_required_cols based on selected required nodes (avoid contract mismatch).
        deps_path = Path("config/feature_dependencies.yaml")
        deps = yaml.safe_load(deps_path.read_text(encoding="utf-8")) or {}
        feats = deps.get("features") or {}
        out_cols = []
        for node in req["required"]:
            meta = feats.get(str(node)) if isinstance(feats, dict) else None
            cols = meta.get("output_columns") if isinstance(meta, dict) else None
            if isinstance(cols, list) and cols:
                out_cols.extend([str(c).strip() for c in cols if str(c).strip()])
        out_cols = sorted(set(out_cols))
        fc = feat_obj.get("feature_contract") or {}
        if not isinstance(fc, dict):
            fc = {}
        fc["minimal_required_cols"] = out_cols
        feat_obj["feature_contract"] = fc

        feat_path.write_text(
            yaml.safe_dump(feat_obj, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return str(derived_dir)
    if task_spec:
        import yaml
        import shutil
        from pathlib import Path

        ts_path = Path(task_spec)
        ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}
        task_id = str(ts_obj.get("task_id") or "").strip()
        if task_id:
            env_overrides["MLBOT_TASK_ID"] = task_id

        enf = ts_obj.get("enforcement") or {}
        constitution_yaml = str(enf.get("constitution_yaml") or "").strip()
        kpi_gate_yaml = str(enf.get("kpi_gate_yaml") or "").strip()

        def _ws(p: str) -> str:
            return f"/workspace/{p}" if use_workspace_prefix else p

        if constitution_yaml:
            env_overrides["MLBOT_CONSTITUTION_YAML"] = _ws(constitution_yaml)
        if kpi_gate_yaml:
            env_overrides["MLBOT_KPI_GATE_YAML"] = _ws(kpi_gate_yaml)

        # Optional: Portfolio Assets v1 contract (diagnostic artifacts / PCM wiring).
        pa = ts_obj.get("portfolio_assets_plan") or {}
        try:
            pa_enabled = bool(pa.get("enabled", False))
        except Exception:
            pa_enabled = False
        pa_cfg = str(pa.get("config_file") or "").strip()
        if pa_enabled and pa_cfg:
            env_overrides["MLBOT_PORTFOLIO_ASSETS_YAML"] = _ws(pa_cfg)
        effective_config = _materialize_cfg_from_task_spec(
            ts_obj=ts_obj, out_root=out_root, base_config_dir=config
        )

    # [1/4] nnmultihead predict
    args_pred = [
        "--config",
        f"/workspace/{effective_config}" if use_workspace_prefix else effective_config,
        "--symbols",
        str(symbols),
        "--timeframe",
        str(timeframe),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--model",
        f"/workspace/{model_path}" if use_workspace_prefix else model_path,
        "--output",
        preds_dir,
        "--features-store-root",
        f"/workspace/{feature_store_root}" if use_workspace_prefix else feature_store_root,
    ]
    if feature_store_layer is not None:
        args_pred.extend(["--features-store-layer", str(feature_store_layer)])
    rc = run_script(
        "scripts/predict_path_primitives_mlp.py",
        args_pred,
        docker=docker,
        env_overrides=env_overrides,
    )
    if rc != 0:
        sys.exit(rc)

    # [2/4] rule mode-3action
    args_mode = [
        "--preds",
        preds_dir,
        "--model",
        f"/workspace/{model_path}" if use_workspace_prefix else model_path,
        "--output",
        mode_path,
    ]
    if mfe_min is not None:
        args_mode.extend(["--mfe-min", str(float(mfe_min))])
    if eff_min is not None:
        args_mode.extend(["--eff-min", str(float(eff_min))])
    if dir_conf_trend_min is not None:
        args_mode.extend(["--dir-conf-trend-min", str(float(dir_conf_trend_min))])
    if mfe_trend_min is not None:
        args_mode.extend(["--mfe-trend-min", str(float(mfe_trend_min))])
    if ttm_trend_min is not None:
        args_mode.extend(["--ttm-trend-min", str(float(ttm_trend_min))])
    if eff_mean_min is not None:
        args_mode.extend(["--eff-mean-min", str(float(eff_mean_min))])
    if ttm_mean_max is not None:
        args_mode.extend(["--ttm-mean-max", str(float(ttm_mean_max))])
    rc = run_script(
        "scripts/rule_mode_3action.py",
        args_mode,
        docker=docker,
        env_overrides=env_overrides,
    )
    if rc != 0:
        sys.exit(rc)

    # [3/4] rl build-logs-3action
    args_logs = [
        "--preds",
        preds_dir,
        "--mode",
        mode_path,
        "--model",
        f"/workspace/{model_path}" if use_workspace_prefix else model_path,
        "--symbols",
        str(symbols),
        "--timeframe",
        str(timeframe),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--returns-source",
        str(returns_source),
        "--output",
        logs_path,
    ]
    rc = run_script(
        "scripts/rl_build_logs_3action.py",
        args_logs,
        docker=docker,
        env_overrides=env_overrides,
    )
    if rc != 0:
        sys.exit(rc)

    # [4/4] rl run-e2e-3action (shadow + counterfactual + fsm decision)
    shadow_out = f"{e2e_out}/shadow"
    cf_out = f"{e2e_out}/counterfactual"
    fsm_out = f"{e2e_out}/fsm_decision.json"

    rc = run_script(
        "scripts/rl_shadow_eval_3action.py",
        [
            "--logs",
            logs_path,
            "--out",
            shadow_out,
            "--train_ratio",
            str(float(train_ratio)),
        ],
        docker=docker,
        env_overrides=env_overrides,
    )
    if rc != 0:
        sys.exit(rc)

    cf_args = [
        "--logs",
        logs_path,
        "--out",
        cf_out,
        "--train_ratio",
        str(float(train_ratio)),
        "--entry_delay",
        str(int(entry_delay)),
        "--cost_per_turnover",
        str(float(cost_per_turnover)),
        "--slippage_bps",
        str(float(slippage_bps)),
        "--preds-in-log1p",
        "1" if preds_in_log1p else "0",
    ]
    # For unified Router diagnostics in counterfactual report
    if mfe_min is not None:
        cf_args.extend(["--router-mfe-min", str(float(mfe_min))])
    if eff_min is not None:
        cf_args.extend(["--router-eff-min", str(float(eff_min))])
    if dir_conf_trend_min is not None:
        cf_args.extend(["--router-dir-conf-trend-min", str(float(dir_conf_trend_min))])

    rc = run_script(
        "scripts/rl_counterfactual_eval_3action.py",
        cf_args,
        docker=docker,
        env_overrides=env_overrides,
    )
    if rc != 0:
        sys.exit(rc)

    rc = run_script(
        "scripts/rl_fsm_decide.py",
        [
            "--metrics",
            f"{cf_out}/metrics.json",
            "--state",
            "RL_CANDIDATE",
            "--promote_days",
            "10",
            "--cooldown_days",
            "20",
            "--out",
            fsm_out,
        ],
        docker=docker,
        env_overrides=env_overrides,
    )
    sys.exit(rc)


@nnmultihead.command("materialize-config-from-task-spec")
@click.option(
    "--task-spec",
    required=True,
    help="TaskSpec YAML (v1). Uses feature_plan.tiers_enabled + tier_feature_files to build a derived nnmultihead config.",
)
@click.option(
    "--base-config",
    default="config/nnmultihead/path_primitives_4h_80h_min",
    show_default=True,
    help="Base nnmultihead config directory to copy from.",
)
@click.option(
    "--out-config",
    required=True,
    help="Output directory for the derived nnmultihead config (will be overwritten).",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_materialize_config_from_task_spec(task_spec, base_config, out_config, docker):
    """
    Generate a concrete config directory so tiers are *real* (not just metadata).

    Typical usage:
      - generate tier0 config -> train model -> eval
      - generate tier0+1 config -> train model -> eval
      - compare A-layer + system reports
    """
    use_workspace_prefix = docker and not _is_in_docker()
    ts_path = f"/workspace/{task_spec}" if use_workspace_prefix else task_spec
    base_dir = f"/workspace/{base_config}" if use_workspace_prefix else base_config
    out_dir = f"/workspace/{out_config}" if use_workspace_prefix else out_config

    import yaml
    import shutil
    from pathlib import Path

    ts_obj = yaml.safe_load(Path(ts_path).read_text(encoding="utf-8")) or {}

    # Use the same materialization logic as pipeline (copy base config and rewrite features.yaml required list).
    fp = ts_obj.get("feature_plan") or {}
    tiers_enabled = fp.get("tiers_enabled") or []
    tier_feature_files = fp.get("tier_feature_files") or {}
    if not (isinstance(tiers_enabled, list) and isinstance(tier_feature_files, dict)):
        raise click.ClickException("TaskSpec missing feature_plan.tiers_enabled or tier_feature_files")

    tier_nodes = []
    for tier_name in tiers_enabled:
        k = str(tier_name).strip()
        fpath = tier_feature_files.get(k)
        if not fpath:
            continue
        p = Path(fpath)
        if not p.is_absolute():
            # When running in docker, ts_path is /workspace/..., repo root also /workspace
            p = Path("/workspace") / p
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise click.ClickException(f"Tier file must be a YAML list: {p}")
        tier_nodes.extend([str(x).strip() for x in obj if str(x).strip()])
    tier_nodes = sorted(set(tier_nodes))
    if not tier_nodes:
        raise click.ClickException("No tier feature nodes collected (tiers_enabled/tier_feature_files mismatch).")

    outp = Path(out_dir)
    if outp.exists():
        shutil.rmtree(outp)
    shutil.copytree(Path(base_dir), outp)

    feat_path = outp / "features.yaml"
    feat_obj = yaml.safe_load(feat_path.read_text(encoding="utf-8")) or {}
    fp2 = feat_obj.get("feature_pipeline") or {}
    req = fp2.get("requested_features") or {}
    if not isinstance(req, dict):
        req = {}
    req["required"] = tier_nodes
    fp2["requested_features"] = req
    feat_obj["feature_pipeline"] = fp2

    deps_path = Path("/workspace/config/feature_dependencies.yaml") if use_workspace_prefix else Path("config/feature_dependencies.yaml")
    deps = yaml.safe_load(deps_path.read_text(encoding="utf-8")) or {}
    feats = deps.get("features") or {}
    out_cols = []
    for node in tier_nodes:
        meta = feats.get(str(node)) if isinstance(feats, dict) else None
        cols = meta.get("output_columns") if isinstance(meta, dict) else None
        if isinstance(cols, list) and cols:
            out_cols.extend([str(c).strip() for c in cols if str(c).strip()])
    out_cols = sorted(set(out_cols))
    fc = feat_obj.get("feature_contract") or {}
    if not isinstance(fc, dict):
        fc = {}
    fc["minimal_required_cols"] = out_cols
    feat_obj["feature_contract"] = fc

    feat_path.write_text(
        yaml.safe_dump(feat_obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    click.echo(f"✅ Derived config written: {out_dir}")


@nnmultihead.command("build-feature-store")
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--config",
    "-c",
    default="config/nnmultihead/path_primitives_4h_80h_min",
    help="NN multihead config directory (features.yaml + labels.yaml + model.yaml)",
)
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD) optional")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD) optional")
@click.option(
    "--feature-store-root",
    default="feature_store",
    help="FeatureStore root dir (default: feature_store).",
)
@click.option(
    "--layer",
    default="nnmultihead_v1",
    help="FeatureStore layer name for nnmultihead features.",
)
@click.option(
    "--warmup-bars",
    type=int,
    default=512,
    help="Warmup bars to prepend when computing each month (stateful/ticks features).",
)
@click.option(
    "--warmup-months",
    type=int,
    default=1,
    help="Warmup calendar months to prepend when computing each month (recommended).",
)
@click.option(
    "--feature-monthly-workers",
    type=int,
    default=1,
    show_default=True,
    help="Opt-in: parallelize per-feature monthly cache-miss computation (FEATURE_MONTHLY_WORKERS). Use small values (2-4) for tick-heavy features.",
)
@click.option(
    "--feature-monthly-backend",
    type=click.Choice(["process", "thread"]),
    default="process",
    show_default=True,
    help="Backend for monthly parallelism (FEATURE_MONTHLY_BACKEND).",
)
@click.option(
    "--fast-features",
    is_flag=True,
    default=False,
    help="Enable faster feature computation for DTW/Spectrum where supported (FEATURE_FAST_MODE=1).",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_build_feature_store(
    symbols,
    timeframe,
    data_path,
    config,
    start_date,
    end_date,
    feature_store_root,
    layer,
    warmup_bars,
    warmup_months,
    feature_monthly_workers,
    feature_monthly_backend,
    fast_features,
    docker,
):
    """Build monthly FeatureStore for nnmultihead features (shared infra; default path works for tree+nn)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--symbols",
        symbols,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--timeframe",
        timeframe,
        "--output-dir",
        (
            f"/workspace/{feature_store_root}"
            if use_workspace_prefix
            else feature_store_root
        ),
        "--output-format",
        "monthly",
        "--layer",
        layer,
        "--warmup-bars",
        str(int(warmup_bars)),
        "--warmup-months",
        str(int(warmup_months)),
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    env_overrides = {
        "FEATURE_MONTHLY_WORKERS": str(int(feature_monthly_workers)),
        "FEATURE_MONTHLY_BACKEND": str(feature_monthly_backend),
    }
    if fast_features:
        env_overrides["FEATURE_FAST_MODE"] = "1"
    sys.exit(
        run_script(
            "scripts/build_feature_store_nnmultihead.py",
            args,
            docker=docker,
            env_overrides=env_overrides,
        )
    )


@nnmultihead.command("eval")
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--config",
    "-c",
    default="config/nnmultihead/path_primitives_4h_80h_min",
    help="NN multihead config directory (features.yaml + labels.yaml + model.yaml)",
)
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD) optional")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD) optional")
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Path to model.pt produced by nnmultihead train",
)
@click.option(
    "--horizon-hours",
    type=float,
    default=80.0,
    help="Future horizon in hours (e.g., 80H)",
)
@click.option(
    "--bar-hours", type=float, default=4.0, help="Bar duration in hours (4H => 4)"
)
@click.option("--device", default=None, help="cpu|cuda (default auto)")
@click.option(
    "--output-dir",
    default="results/nnmultihead_eval",
    help="Output directory for eval artifacts",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_eval(
    symbols,
    timeframe,
    data_path,
    config,
    start_date,
    end_date,
    model_path,
    horizon_hours,
    bar_hours,
    device,
    output_dir,
    docker,
):
    """Evaluate a trained nnmultihead model and generate report.html artifacts."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--symbols",
        symbols,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--timeframe",
        timeframe,
        "--model",
        f"/workspace/{model_path}" if use_workspace_prefix else model_path,
        "--horizon-hours",
        str(horizon_hours),
        "--bar-hours",
        str(bar_hours),
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if device:
        args.extend(["--device", device])

    sys.exit(run_script("scripts/evaluate_path_primitives_mlp.py", args, docker=docker))


@nnmultihead.command("render-report")
@click.option(
    "--run-dir",
    required=True,
    help="Existing nnmultihead training run directory (contains meta.json + metrics.json).",
)
@click.option(
    "--out-html",
    default=None,
    help="Optional output HTML path (default: <run-dir>/report.html)",
)
@click.option(
    "--out-summary",
    default=None,
    help="Optional output summary markdown path (default: <run-dir>/metrics_summary.md)",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_render_report(run_dir, out_html, out_summary, docker):
    """
    Re-render nnmultihead training report artifacts (report.html + metrics_summary.md)
    without retraining. Useful after updating report templates/summary logic.

    Note: for new training runs, `mlbot nnmultihead train` already writes these artifacts.
    """
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--run-dir",
        f"/workspace/{run_dir}" if use_workspace_prefix else run_dir,
    ]
    if out_html:
        args.extend(
            [
                "--out-html",
                f"/workspace/{out_html}" if use_workspace_prefix else out_html,
            ]
        )
    if out_summary:
        args.extend(
            [
                "--out-summary",
                f"/workspace/{out_summary}" if use_workspace_prefix else out_summary,
            ]
        )
    sys.exit(
        run_script("scripts/render_path_primitives_report.py", args, docker=docker)
    )


@nnmultihead.command("factor-eval")
@click.option(
    "--config-dir",
    required=True,
    help="nnmultihead config dir (for provenance and default output routing)",
)
@click.option(
    "--candidates-yaml",
    required=True,
    help="YAML with feature_pipeline.requested_features (e.g., config/strategies/*/features_all.yaml)",
)
@click.option(
    "--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT"
)
@click.option("--timeframe", default="240T", show_default=True, help="Timeframe")
@click.option(
    "--features-store-root",
    default="feature_store",
    show_default=True,
    help="FeatureStore root",
)
@click.option(
    "--features-store-layer",
    default=None,
    help="FeatureStore layer id (leave empty to auto-generate from config-dir)",
)
@click.option("--start-date", default=None, help="Start date (optional)")
@click.option("--end-date", default=None, help="End date (optional)")
@click.option(
    "--horizon-hours",
    type=float,
    default=80.0,
    show_default=True,
    help="Horizon in hours",
)
@click.option(
    "--bar-hours",
    type=float,
    default=4.0,
    show_default=True,
    help="Bar duration in hours",
)
@click.option(
    "--min-samples-per-group",
    type=int,
    default=200,
    show_default=True,
    help="Min samples per (symbol,month) group for IC/AUC",
)
@click.option(
    "--max-nan-rate",
    type=float,
    default=0.5,
    show_default=True,
    help="Max NaN rate for factor",
)
@click.option(
    "--min-abs-ir",
    type=float,
    default=0.05,
    show_default=True,
    help="Min |IR| for qualification",
)
@click.option(
    "--min-abs-tstat",
    type=float,
    default=1.96,
    show_default=True,
    help="Min |t-stat| for qualification",
)
@click.option("--output-dir", default=None, help="Override output directory")
@click.option("--export-yaml", default=None, help="Override export YAML path")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_factor_eval(
    config_dir,
    candidates_yaml,
    symbols,
    timeframe,
    features_store_root,
    features_store_layer,
    start_date,
    end_date,
    horizon_hours,
    bar_hours,
    min_samples_per_group,
    max_nan_rate,
    min_abs_ir,
    min_abs_tstat,
    output_dir,
    export_yaml,
    docker,
):
    """Factor-eval for nnmultihead: score candidate features vs path primitives labels; export Pool B YAML."""
    from src.feature_store.layer_naming import resolve_layer_name

    use_workspace_prefix = docker and not _is_in_docker()
    layer = resolve_layer_name(features_store_layer, Path(config_dir).resolve())

    args = [
        "--config-dir",
        f"/workspace/{config_dir}" if use_workspace_prefix else config_dir,
        "--candidates-yaml",
        f"/workspace/{candidates_yaml}" if use_workspace_prefix else candidates_yaml,
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--features-store-root",
        (
            f"/workspace/{features_store_root}"
            if use_workspace_prefix
            else features_store_root
        ),
        "--features-store-layer",
        layer,
        "--horizon-hours",
        str(horizon_hours),
        "--bar-hours",
        str(bar_hours),
        "--min-samples-per-group",
        str(min_samples_per_group),
        "--max-nan-rate",
        str(max_nan_rate),
        "--min-abs-ir",
        str(min_abs_ir),
        "--min-abs-tstat",
        str(min_abs_tstat),
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if output_dir:
        args.extend(
            [
                "--output-dir",
                f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
            ]
        )
    if export_yaml:
        args.extend(
            [
                "--export-yaml",
                f"/workspace/{export_yaml}" if use_workspace_prefix else export_yaml,
            ]
        )

    sys.exit(
        run_python_module(
            "time_series_model.diagnostics.factor_primitives_eval",
            args,
            docker=docker,
        )
    )


@nnmultihead.command("feature-group-search")
@click.option("--base-config", required=True, help="Base nnmultihead config dir")
@click.option(
    "--base-features-yaml",
    default=None,
    help="Optional base feature funcs YAML (Pool A). If omitted, will try <base-config>/features_base.yaml.",
)
@click.option(
    "--groups-yaml",
    default=None,
    help="Optional semantic groups YAML (same schema as config/feature_groups.yaml).",
)
@click.option(
    "--expand-semantic-singletons",
    is_flag=True,
    default=False,
    help="Expand semantic nodes into singleton output-column groups (finer-grained selection).",
)
@click.option("--symbols", required=True, help="Comma-separated symbols")
@click.option("--timeframe", default="240T", show_default=True)
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
@click.option("--features-store-root", default="feature_store", show_default=True)
@click.option("--features-store-layer", required=True)
@click.option(
    "--pool-b-yaml", required=True, help="PoolB YAML (features_pool_b_primitives.yaml)"
)
@click.option(
    "--objective",
    default="dir_auc",
    show_default=True,
    help="metrics.json key to maximize",
)
@click.option("--max-steps", type=int, default=6, show_default=True)
@click.option(
    "--preset",
    default="",
    type=click.Choice(["", "A", "B", "C"]),
    show_default=True,
    help="Budget preset: A=fast screen, B=medium, C=full verify. Overrides budget knobs.",
)
@click.option(
    "--search-algo",
    type=click.Choice(["greedy", "halving", "beam", "sffs", "pipeline"]),
    default="greedy",
    show_default=True,
)
@click.option(
    "--run-abc",
    is_flag=True,
    default=False,
    help="Run A->B->C orchestration into <output-dir>/{A,B,C} with shortlists and a summary.md.",
)
@click.option("--epochs", type=int, default=10, show_default=True)
@click.option("--batch-size", type=int, default=512, show_default=True)
@click.option("--lr", type=float, default=2e-4, show_default=True)
@click.option("--hidden", type=int, default=256, show_default=True)
@click.option("--depth", type=int, default=2, show_default=True)
@click.option("--dropout", type=float, default=0.1, show_default=True)
@click.option(
    "--exclude-columns",
    default=None,
    help="Comma-separated columns to exclude from MLP input (still computed for labels). If omitted, use base-config feature_pipeline.exclude_columns.",
)
@click.option("--device", default=None)
@click.option("--halving-stages", default="3,6,10", show_default=True)
@click.option("--halving-top-fraction", type=float, default=0.25, show_default=True)
@click.option("--halving-min-survivors", type=int, default=5, show_default=True)
@click.option("--beam-width", type=int, default=3, show_default=True)
@click.option("--sffs-max-backward-per-step", type=int, default=2, show_default=True)
@click.option("--pipeline-survivors", type=int, default=30, show_default=True)
@click.option(
    "--export-shortlist-yaml",
    default=None,
    help="Optional: export shortlisted groups YAML (plain dict; tree-compatible).",
)
@click.option(
    "--export-shortlist-mode",
    type=click.Choice(["prefilter_survivors", "beam_selected", "selected_groups"]),
    default="prefilter_survivors",
    show_default=True,
)
@click.option("--export-shortlist-max-groups", type=int, default=0, show_default=True)
@click.option("--output-dir", required=True)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def nnmultihead_feature_group_search(
    base_config,
    base_features_yaml,
    groups_yaml,
    expand_semantic_singletons,
    symbols,
    timeframe,
    start_date,
    end_date,
    features_store_root,
    features_store_layer,
    pool_b_yaml,
    objective,
    max_steps,
    preset,
    search_algo,
    run_abc,
    epochs,
    batch_size,
    lr,
    hidden,
    depth,
    dropout,
    exclude_columns,
    device,
    halving_stages,
    halving_top_fraction,
    halving_min_survivors,
    beam_width,
    sffs_max_backward_per_step,
    pipeline_survivors,
    export_shortlist_yaml,
    export_shortlist_mode,
    export_shortlist_max_groups,
    output_dir,
    docker,
):
    """Feature-group-search for nnmultihead (primitives objective)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--base-config",
        f"/workspace/{base_config}" if use_workspace_prefix else base_config,
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--features-store-root",
        (
            f"/workspace/{features_store_root}"
            if use_workspace_prefix
            else features_store_root
        ),
        "--features-store-layer",
        features_store_layer,
        "--pool-b-yaml",
        f"/workspace/{pool_b_yaml}" if use_workspace_prefix else pool_b_yaml,
        "--objective",
        objective,
        "--max-steps",
        str(int(max_steps)),
        "--search-algo",
        str(search_algo),
        "--run-abc" if run_abc else "",
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--hidden",
        str(int(hidden)),
        "--depth",
        str(int(depth)),
        "--dropout",
        str(float(dropout)),
    ]
    # CLI override only (preferred default lives in base-config/features.yaml)
    if exclude_columns is not None:
        args.extend(["--exclude-columns", str(exclude_columns)])
    args.extend(
        [
            "--halving-stages",
            str(halving_stages),
            "--halving-top-fraction",
            str(float(halving_top_fraction)),
            "--halving-min-survivors",
            str(int(halving_min_survivors)),
            "--beam-width",
            str(int(beam_width)),
            "--sffs-max-backward-per-step",
            str(int(sffs_max_backward_per_step)),
            "--pipeline-survivors",
            str(int(pipeline_survivors)),
            "--output-dir",
            f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
        ]
    )
    # Only include --preset if explicitly provided (non-empty).
    # (Empty preset would otherwise leave a dangling '--preset' after empty-token filtering.)
    if str(preset or "").strip():
        args.extend(["--preset", str(preset)])
    if base_features_yaml:
        args.extend(
            [
                "--base-features-yaml",
                (
                    f"/workspace/{base_features_yaml}"
                    if use_workspace_prefix
                    else base_features_yaml
                ),
            ]
        )
    if groups_yaml:
        args.extend(
            [
                "--groups-yaml",
                f"/workspace/{groups_yaml}" if use_workspace_prefix else groups_yaml,
            ]
        )
    if expand_semantic_singletons:
        args.append("--expand-semantic-singletons")
    if device:
        args.extend(["--device", device])

    if export_shortlist_yaml:
        args.extend(
            [
                "--export-shortlist-yaml",
                (
                    f"/workspace/{export_shortlist_yaml}"
                    if use_workspace_prefix
                    else export_shortlist_yaml
                ),
                "--export-shortlist-mode",
                str(export_shortlist_mode),
                "--export-shortlist-max-groups",
                str(int(export_shortlist_max_groups)),
            ]
        )

    # Remove any empty tokens (from conditional flags)
    args = [x for x in args if str(x).strip()]
    sys.exit(
        run_python_module(
            "time_series_model.diagnostics.nn_feature_group_search", args, docker=docker
        )
    )


def _train_strategy_pipeline(
    symbol,
    timeframe,
    config,
    data_path,
    test_size,
    output_root,
    docker,
    *,
    feature_store_dir: str,
    feature_store_layer: str | None,
):
    """Shared implementation for strategy training (train_strategy_pipeline.py)."""
    # Only add /workspace prefix if we're launching a new Docker container (not already inside one)
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--output-root",
        f"/workspace/{output_root}" if use_workspace_prefix else output_root,
    ]
    # FeatureStore is always enabled by train_strategy_pipeline.py.
    # We keep feature-store-dir/layer so users can override location/layer if needed.
    args.extend(
        [
            "--feature-store-dir",
            (
                f"/workspace/{feature_store_dir}"
                if use_workspace_prefix
                else feature_store_dir
            ),
        ]
    )
    # Only pass --feature-store-layer if explicitly provided (None means auto-generate in script)
    if feature_store_layer is not None:
        args.extend(["--feature-store-layer", feature_store_layer])
    sys.exit(run_script("scripts/train_strategy_pipeline.py", args, docker=docker))


@train.command("sr-reversal-long")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option(
    "--timeframe", "-t", default="240T", help="Timeframe (e.g., 15T, 60T, 240T)"
)
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option("--feature-store-dir", default="feature_store")
@click.option("--feature-store-layer", default=None)
@click.option(
    "--output-root",
    default="results/strategies/sr_reversal_long",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_sr_reversal_long(
    symbol,
    timeframe,
    data_path,
    test_size,
    feature_store_dir,
    feature_store_layer,
    output_root,
    docker,
):
    """Train SR Reversal Long-only model (direction-fixed)."""
    _train_strategy_pipeline(
        symbol=symbol,
        timeframe=timeframe,
        config="config/strategies/sr_reversal_long",
        data_path=data_path,
        test_size=test_size,
        output_root=output_root,
        docker=docker,
        feature_store_dir=str(feature_store_dir),
        feature_store_layer=feature_store_layer,  # None means auto-generate in script
    )


@train.command("sr-reversal-short")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option(
    "--timeframe", "-t", default="240T", help="Timeframe (e.g., 15T, 60T, 240T)"
)
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option("--feature-store-dir", default="feature_store")
@click.option("--feature-store-layer", default=None)
@click.option(
    "--output-root",
    default="results/strategies/sr_reversal_short",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_sr_reversal_short(
    symbol,
    timeframe,
    data_path,
    test_size,
    feature_store_dir,
    feature_store_layer,
    output_root,
    docker,
):
    """Train SR Reversal Short-only model (direction-fixed)."""
    _train_strategy_pipeline(
        symbol=symbol,
        timeframe=timeframe,
        config="config/strategies/sr_reversal_short",
        data_path=data_path,
        test_size=test_size,
        output_root=output_root,
        docker=docker,
        feature_store_dir=str(feature_store_dir),
        feature_store_layer=feature_store_layer,  # None means auto-generate in script
    )


@train.command("rolling")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="15T", help="Timeframe")
@click.option(
    "--config",
    "-c",
    default="config/strategies/sr_reversal_long",
    help="Strategy config (direction-fixed; train long/short separately)",
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


@train.command("final")
@click.option(
    "--symbol", "-s", default="BTCUSDT", show_default=True, help="Trading symbol"
)
@click.option("--timeframe", "-t", default="240T", show_default=True, help="Timeframe")
@click.option(
    "--config",
    "-c",
    default="config/strategies/sr_reversal_long",
    show_default=True,
    help="Strategy config directory",
)
@click.option("--start-date", required=True, help="Train start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="Train end date (YYYY-MM-DD)")
@click.option(
    "--seed", default="42", show_default=True, help="Seed for reproducibility"
)
@click.option(
    "--output-root",
    default="models",
    show_default=True,
    help="Root dir for final model outputs (ModelArtifact saved under <output-root>/<strategy_name>/).",
)
@click.option(
    "--data-path", default="data/parquet_data", show_default=True, help="Data directory"
)
@click.option("--feature-store-dir", default="feature_store", show_default=True)
@click.option("--feature-store-layer", default=None)
@click.option(
    "--deterministic/--non-deterministic", default=True, help="Deterministic training"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_final(
    symbol,
    timeframe,
    config,
    start_date,
    end_date,
    seed,
    output_root,
    data_path,
    feature_store_dir,
    feature_store_layer,
    deterministic,
    docker,
):
    """Train a final (deployable) model on the full training window and save a ModelArtifact."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--seed",
        str(seed),
        "--output-root",
        f"/workspace/{output_root}" if use_workspace_prefix else output_root,
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--train-all",
    ]
    args.extend(
        [
            "--feature-store-dir",
            (
                f"/workspace/{feature_store_dir}"
                if use_workspace_prefix
                else feature_store_dir
            ),
        ]
    )
    if feature_store_layer is not None:
        args.extend(["--feature-store-layer", feature_store_layer])
    if deterministic:
        args.append("--deterministic")
    sys.exit(run_script("scripts/train_strategy_pipeline.py", args, docker=docker))


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
        "DOCKER_IMAGE", "hansenlovefiona017/lightgbm-runtime:v0.0.9"
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
            "v0.0.9",
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
    "--export-yaml",
    default=None,
    help="Write the exported features.yaml (with invert_features) to this path",
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
    export_yaml,
    remove_correlated,
    correlation_threshold,
    filter_by_best_lag,
    target_lag,
    lag_tolerance,
    docker,
):
    """Time-series factor IC / win-rate evaluation (single asset)."""
    # If already in Docker, don't add /workspace prefix
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if use_workspace_prefix else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        "/workspace/data/parquet_data" if use_workspace_prefix else "data/parquet_data",
        "--timeframe",
        timeframe,
        "--quantile",
        str(quantile),
        "--feature-mode",
        feature_mode,
        "--ic-decay-lags",
        ic_decay_lags,
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
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
    if export_yaml:
        args.extend(["--export-yaml", export_yaml])

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


@analyze.command("strategy-feature-compare")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal_long",
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
@click.option(
    "--calibrate-proba",
    type=click.Choice(["none", "platt", "isotonic"], case_sensitive=False),
    default="none",
    help="Probability calibration method (none/platt/isotonic)",
)
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
    calibrate_proba,
    docker,
):
    """Ablation Study: Compare multiple feature configs for a strategy."""
    # Only add /workspace prefix if we're launching a new Docker container (not already inside one)
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if use_workspace_prefix else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        "/workspace/data/parquet_data" if use_workspace_prefix else "data/parquet_data",
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
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
        # Split by whitespace to support multiple overrides
        override_list = feature_overrides.split()
        args.append("--feature-overrides")
        args.extend(override_list)
    if run_rolling:
        args.append("--run-rolling")
    if calibrate_proba and calibrate_proba != "none":
        args.extend(["--calibrate-proba", calibrate_proba])

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


@diagnose.command("feature-contract")
@click.option(
    "--feature-deps",
    default="config/feature_dependencies.yaml",
    help="Path to feature_dependencies.yaml",
)
@click.option(
    "--mode",
    default="error",
    type=click.Choice(["error", "warn"]),
    help="error: non-zero exit on violations; warn: always exit 0 but print report",
)
@click.option(
    "--out-json",
    default=None,
    help="Optional output JSON path for the report",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_feature_contract(feature_deps, mode, out_json, docker):
    """
    Feature contract checks (normalization + semantic safety).

    This is the unified entrypoint for:
    - normalization contract completeness (no missing methods)
    - mixed-output normalization maps (e.g., usd vs unitless)
    - scale-column protection (e.g., atr must be price_unit)
    """
    args = [
        "--feature-deps",
        f"/workspace/{feature_deps}" if docker else feature_deps,
        "--mode",
        mode,
    ]
    if out_json:
        args.extend(["--out-json", f"/workspace/{out_json}" if docker else out_json])

    sys.exit(
        run_python_module(
            "src.features.normalization.feature_contract_checks",
            args,
            docker=docker,
        )
    )


@diagnose.command("kpi-gate")
@click.option(
    "--metrics-json",
    required=True,
    help="Path to metrics.json (e.g. produced by rl counterfactual-eval-3action).",
)
@click.option(
    "--gate-yaml",
    required=True,
    help="Path to kpi_gate.yaml defining hard_fail/warn thresholds.",
)
@click.option(
    "--out-json",
    default=None,
    help="Optional output path for gate result json (ok/hard_failures/warnings).",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_kpi_gate(metrics_json, gate_yaml, out_json, docker):
    """
    KPI gate checker.

    CI usage: exit code 2 when hard_fail triggers (blocks promotion / merge).
    """
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--metrics-json",
        f"/workspace/{metrics_json}" if use_workspace_prefix else metrics_json,
        "--gate-yaml",
        f"/workspace/{gate_yaml}" if use_workspace_prefix else gate_yaml,
    ]
    if out_json:
        args.extend(["--out-json", f"/workspace/{out_json}" if use_workspace_prefix else out_json])
    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.kpi_gate_cli",
            args,
            docker=docker,
        )
    )


@diagnose.command("rule-baseline")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal_long",
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
    default="config/strategies/sr_reversal_long",
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
    default="config/strategies/sr_reversal_long",
    help="Comma-separated strategy config directories (strategy-agnostic).",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option("--seed", default="42", help="Random seed (forwarded to train pipeline)")
@click.option(
    "--deterministic",
    is_flag=True,
    default=False,
    help="Force deterministic training (slower but reproducible).",
)
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
def diagnose_model_comparison(
    strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    seed,
    deterministic,
    output_dir,
    data_path,
    docker,
):
    """Compare multiple strategy configs under identical settings (strategy-agnostic)."""
    # Append timeframe to output dir
    output_dir_full = f"{output_dir}/{timeframe}"

    # Only add /workspace prefix if we're launching a new Docker container (not already inside one)
    use_workspace_prefix = docker and not _is_in_docker()

    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if use_workspace_prefix else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--seed",
        seed,
        "--output-dir",
        f"/workspace/{output_dir_full}" if use_workspace_prefix else output_dir_full,
    ]
    if deterministic:
        args.append("--deterministic")
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])

    # Note: previously this command was SR-reversal-specific. It is now strategy-agnostic:
    # it runs the unified train pipeline for each strategy config and summarizes results.
    # (The legacy SR reversal comparison remains available as diagnose sr-reversal-model-comparison.)
    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.strategy_model_comparison",
            args,
            docker=docker,
        )
    )


@diagnose.command("sr-reversal-model-comparison")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/sr_reversal_long",
    help="Strategy config directory (SR reversal family)",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option("--seed", default="42", help="Random seed (forwarded to train pipeline)")
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
@click.option(
    "--rule-based-entry",
    default=None,
    help="Python module path for rule-based strategy entry point",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_sr_reversal_model_comparison(
    strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    seed,
    output_dir,
    data_path,
    ticks_dir,
    ticks_lookback_minutes,
    rule_params,
    rule_based_entry,
    docker,
):
    """Legacy SR-reversal-specific model comparison (kept for backward compatibility)."""
    output_dir_full = f"{output_dir}/{timeframe}"
    use_workspace_prefix = docker and not _is_in_docker()

    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if use_workspace_prefix else strategy_config,
        "--symbol",
        symbol,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--timeframe",
        timeframe,
        "--test-size",
        test_size,
        "--seed",
        seed,
        "--output-dir",
        f"/workspace/{output_dir_full}" if use_workspace_prefix else output_dir_full,
        "--ticks-dir",
        f"/workspace/{ticks_dir}" if use_workspace_prefix else ticks_dir,
        "--ticks-lookback-minutes",
        ticks_lookback_minutes,
    ]
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if rule_params:
        args.extend(
            [
                "--rule-params",
                f"/workspace/{rule_params}" if use_workspace_prefix else rule_params,
            ]
        )
    if rule_based_entry:
        args.extend(["--rule-based-entry", rule_based_entry])

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.sr_reversal_model_comparison",
            args,
            docker=docker,
        )
    )


@diagnose.command("feature-group-search")
@click.option(
    "--base-strategy-config",
    "-c",
    required=True,
    help="Base strategy config directory (single strategy). The tool will create temp variants.",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.3", help="Test set ratio")
@click.option("--seeds", default="1,2,3", help="Comma-separated seeds")
@click.option(
    "--objective", default="Sharpe_mean", help="Objective metric (e.g. Sharpe_mean)"
)
@click.option("--min-trades", default="10", help="Min trades_mean constraint")
@click.option("--max-steps", default="6", help="Max greedy steps")
@click.option(
    "--preset",
    default="",
    type=click.Choice(["", "A", "B", "C"]),
    show_default=True,
    help="Budget preset for feature-group-search. A=fast screening, B=medium, C=final verification.",
)
@click.option(
    "--invert-eval",
    default="conservative",
    type=click.Choice(["none", "conservative", "all"]),
    show_default=True,
    help="Validate Pool-B inverted candidates (output columns) by trying raw vs inverted. Use all to always validate/pick better sign.",
)
@click.option(
    "--fast-features",
    is_flag=True,
    default=False,
    help="Enable faster feature computation for search runs (sets FEATURE_FAST_MODE=1 in training subprocess).",
)
@click.option(
    "--search-algo",
    default="greedy",
    type=click.Choice(["greedy", "halving", "beam", "sffs", "pipeline"]),
    show_default=True,
    help="Search algorithm: greedy / halving / beam / sffs / pipeline (halving->beam->sffs).",
)
@click.option(
    "--halving-stages",
    default="1,3,5",
    show_default=True,
    help="Comma-separated seed counts as halving budgets (tool will append full seeds if missing).",
)
@click.option(
    "--halving-top-fraction",
    default="0.25",
    show_default=True,
    help="Fraction of candidates to keep at each halving stage (0,1].",
)
@click.option(
    "--halving-min-survivors",
    default="5",
    show_default=True,
    help="Minimum survivors to keep at each halving stage.",
)
@click.option(
    "--pipeline-survivors",
    default="30",
    show_default=True,
    help="Pipeline only: target survivors after halving prefilter.",
)
@click.option(
    "--beam-width",
    default="3",
    show_default=True,
    help="Beam width (top-K paths to keep) for beam/pipeline.",
)
@click.option(
    "--sffs-max-backward-per-step",
    default="2",
    show_default=True,
    help="SFFS backward removal budget (used by sffs/pipeline).",
)
@click.option(
    "--groups-json",
    default=None,
    help="Optional JSON file path overriding default feature groups.",
)
@click.option(
    "--groups-yaml",
    default=None,
    help="Optional YAML file path overriding default feature groups.",
)
@click.option(
    "--pool-b-yaml",
    default=None,
    help=(
        "Optional Pool B YAML exported by factor-eval (features_pool_b.yaml). "
        "If provided, the tool will auto-generate extra singleton groups for any "
        "feature_pipeline.requested_features not already present in groups."
    ),
)
@click.option(
    "--feature-blacklist",
    default="",
    help="Comma-separated requested_feature nodes to exclude from BOTH base and candidate groups.",
)
@click.option(
    "--base-features-yaml",
    default=None,
    help="Optional YAML list file path for base requested_features (default empty).",
)
@click.option(
    "--writeback-yaml",
    default=None,
    help="Optional output path to write a features_suggested.yaml (requested_features=final_features) with provenance metadata.",
)
@click.option(
    "--invert-candidates-yaml",
    default=None,
    help=(
        "Optional YAML path providing invert candidates. Accepts either a YAML list, or a full "
        "features config containing feature_pipeline.invert_features. On writeback, the tool "
        "will set invert_features = invert_candidates ∩ final_requested_features."
    ),
)
@click.option(
    "--expand-semantic-singletons",
    is_flag=True,
    default=False,
    help=(
        "Expand semantic feature nodes (e.g., vpin_scene_semantic_scores_f) into singleton groups, "
        "one per output column (e.g., vpin_compression_scene_score, vpin_ignition_scene_score). "
        "This allows fine-grained selection of individual semantic scores."
    ),
)
@click.option(
    "--output-dir",
    default="results/feature_group_search",
    help="Output directory",
)
@click.option(
    "--deterministic/--non-deterministic", default=True, help="Deterministic training"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_feature_group_search(
    base_strategy_config,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    seeds,
    objective,
    min_trades,
    max_steps,
    preset,
    invert_eval,
    fast_features,
    search_algo,
    halving_stages,
    halving_top_fraction,
    halving_min_survivors,
    pipeline_survivors,
    beam_width,
    sffs_max_backward_per_step,
    groups_json,
    groups_yaml,
    pool_b_yaml,
    feature_blacklist,
    base_features_yaml,
    writeback_yaml,
    invert_candidates_yaml,
    expand_semantic_singletons,
    output_dir,
    deterministic,
    docker,
):
    """Greedy forward selection over feature groups for one base strategy."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--base-strategy-config",
        (
            f"/workspace/{base_strategy_config}"
            if use_workspace_prefix
            else base_strategy_config
        ),
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--test-size",
        str(test_size),
        "--seeds",
        str(seeds),
        "--objective",
        str(objective),
        "--min-trades",
        str(min_trades),
        "--max-steps",
        str(max_steps),
        "--preset",
        str(preset),
        "--invert-eval",
        str(invert_eval),
        "--fast-features" if fast_features else "",
        "--search-algo",
        str(search_algo),
        "--halving-stages",
        str(halving_stages),
        "--halving-top-fraction",
        str(halving_top_fraction),
        "--halving-min-survivors",
        str(halving_min_survivors),
        "--pipeline-survivors",
        str(pipeline_survivors),
        "--beam-width",
        str(beam_width),
        "--sffs-max-backward-per-step",
        str(sffs_max_backward_per_step),
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
        "--no-docker",
    ]
    # Remove empty entries added by optional flags
    args = [a for a in args if a]
    if deterministic:
        args.append("--deterministic")
    if groups_json:
        args.extend(
            [
                "--groups-json",
                f"/workspace/{groups_json}" if use_workspace_prefix else groups_json,
            ]
        )
    if groups_yaml:
        args.extend(
            [
                "--groups-yaml",
                f"/workspace/{groups_yaml}" if use_workspace_prefix else groups_yaml,
            ]
        )
    if pool_b_yaml:
        args.extend(
            [
                "--pool-b-yaml",
                f"/workspace/{pool_b_yaml}" if use_workspace_prefix else pool_b_yaml,
            ]
        )
    if feature_blacklist:
        args.extend(["--feature-blacklist", str(feature_blacklist)])
    if base_features_yaml:
        args.extend(
            [
                "--base-features-yaml",
                (
                    f"/workspace/{base_features_yaml}"
                    if use_workspace_prefix
                    else base_features_yaml
                ),
            ]
        )
    if writeback_yaml:
        args.extend(
            [
                "--writeback-yaml",
                (
                    f"/workspace/{writeback_yaml}"
                    if use_workspace_prefix
                    else writeback_yaml
                ),
            ]
        )
    if invert_candidates_yaml:
        args.extend(
            [
                "--invert-candidates-yaml",
                (
                    f"/workspace/{invert_candidates_yaml}"
                    if use_workspace_prefix
                    else invert_candidates_yaml
                ),
            ]
        )
    if expand_semantic_singletons:
        args.append("--expand-semantic-singletons")

    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.feature_group_search",
            args,
            docker=docker,
        )
    )


@diagnose.command("poolb-semantic-search")
@click.option(
    "--strategies",
    default="sr_reversal_rr_reg_long,sr_breakout,compression_breakout,trend_following",
    show_default=True,
    help="Comma-separated strategy directory names under config/strategies/ (can be a single strategy).",
)
@click.option(
    "--tag",
    default=None,
    help="Tag for all outputs (Pool-B dir, search output, writeback YAML, report).",
)
@click.option(
    "--symbol", "-s", default="BTCUSDT", show_default=True, help="Trading symbol"
)
@click.option("--timeframe", "-t", default="240T", show_default=True, help="Timeframe")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--test-size", default="0.3", show_default=True, help="Test set ratio")
@click.option(
    "--min-trades", default="10", show_default=True, help="Min trades_mean constraint"
)
@click.option(
    "--search-algo",
    default="pipeline",
    type=click.Choice(["greedy", "halving", "beam", "sffs", "pipeline"]),
    show_default=True,
    help="Search algorithm used in all stages. Recommended: pipeline (SH prefilter -> Beam -> SFFS prune).",
)
@click.option(
    "--shortlist-max-groups-a",
    default="30",
    show_default=True,
    help="Stage A shortlist size (exported and used as Stage B candidate groups).",
)
@click.option(
    "--shortlist-max-groups-b",
    default="20",
    show_default=True,
    help="Stage B shortlist size (exported and used as Stage C candidate groups).",
)
@click.option(
    "--expand-semantic-singletons",
    is_flag=True,
    default=False,
    help="Expand semantic nodes into singleton output-column groups for finer-grained selection.",
)
@click.option(
    "--regen-poolb", is_flag=True, default=False, help="Force regenerate Pool-B YAML"
)
@click.option(
    "--rerun-search",
    is_flag=True,
    default=False,
    help="Force rerun feature-group-search even if result exists",
)
@click.option(
    "--report-only",
    is_flag=True,
    default=False,
    help="Only generate report (requires result JSON present)",
)
@click.option(
    "--skip-report",
    is_flag=True,
    default=False,
    help="Skip writing the markdown summary report (useful for parallel per-strategy runs).",
)
@click.option(
    "--feature-blacklist",
    default="",
    help="Comma-separated requested_feature nodes to exclude from BOTH base and candidate groups during feature-group-search (applied to A/B/C).",
)
def diagnose_poolb_semantic_search(
    strategies,
    tag,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    min_trades,
    search_algo,
    shortlist_max_groups_a,
    shortlist_max_groups_b,
    expand_semantic_singletons,
    regen_poolb,
    rerun_search,
    report_only,
    skip_report,
    feature_blacklist,
):
    """Best workflow: generate Pool-B + run staged feature-group-search (A->B->C with shortlist) + writeback YAMLs + report."""
    script = PROJECT_ROOT / "scripts" / "run_poolb_semantic_search.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--strategies",
        str(strategies),
        "--symbol",
        str(symbol),
        "--timeframe",
        str(timeframe),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--test-size",
        str(test_size),
        "--min-trades",
        str(min_trades),
        "--search-algo",
        str(search_algo),
        "--shortlist-max-groups-a",
        str(shortlist_max_groups_a),
        "--shortlist-max-groups-b",
        str(shortlist_max_groups_b),
    ]
    if tag:
        cmd.extend(["--tag", str(tag)])
    if expand_semantic_singletons:
        cmd.append("--expand-semantic-singletons")
    if regen_poolb:
        cmd.append("--regen-poolb")
    if rerun_search:
        cmd.append("--rerun-search")
    if report_only:
        cmd.append("--report-only")
    if skip_report:
        cmd.append("--skip-report")
    if feature_blacklist:
        cmd.extend(["--feature-blacklist", feature_blacklist])

    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


@diagnose.command("export-fgs-shortlist")
@click.option(
    "--base-strategy-config",
    "-c",
    required=True,
    help="Base strategy config directory (single strategy), e.g. config/strategies/sr_breakout",
)
@click.option(
    "--result-json",
    required=True,
    help="Path to feature_group_search_result.json from a previous run",
)
@click.option(
    "--output-yaml",
    required=True,
    help="Output shortlist groups YAML path",
)
@click.option(
    "--mode",
    default="prefilter_survivors",
    type=click.Choice(["selected_groups", "prefilter_survivors", "beam_selected"]),
    show_default=True,
    help="Which group-name list to export from result JSON",
)
@click.option(
    "--pool-b-yaml",
    default="",
    show_default=True,
    help="Optional features_pool_b.yaml used in the run (to reproduce poolb__* singleton groups).",
)
@click.option(
    "--expand-semantic-singletons",
    is_flag=True,
    default=False,
    help="Apply the same semantic singleton expansion before filtering (must match your run).",
)
@click.option(
    "--max-groups",
    default="0",
    show_default=True,
    help="If >0, keep only the first N names from the chosen list",
)
def diagnose_export_fgs_shortlist(
    base_strategy_config: str,
    result_json: str,
    output_yaml: str,
    mode: str,
    pool_b_yaml: str,
    expand_semantic_singletons: bool,
    max_groups: str,
):
    """Export a shortlisted groups YAML from a previous feature-group-search run."""
    script = PROJECT_ROOT / "scripts" / "fgs_export_shortlist_groups_yaml.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--base-strategy-config",
        str(base_strategy_config),
        "--result-json",
        str(result_json),
        "--output-yaml",
        str(output_yaml),
        "--mode",
        str(mode),
        "--max-groups",
        str(max_groups),
    ]
    if str(pool_b_yaml).strip():
        cmd.extend(["--pool-b-yaml", str(pool_b_yaml)])
    if expand_semantic_singletons:
        cmd.append("--expand-semantic-singletons")

    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


@diagnose.command("holdout-eval")
@click.option(
    "--config",
    "-c",
    default="config/strategies/sr_reversal_long",
    show_default=True,
    help="Strategy config directory",
)
@click.option(
    "--symbol", "-s", default="BTCUSDT", show_default=True, help="Trading symbol"
)
@click.option("--timeframe", "-t", default="240T", show_default=True, help="Timeframe")
@click.option("--train-start-date", required=True, help="Train start date (YYYY-MM-DD)")
@click.option(
    "--holdout-start-date", required=True, help="Holdout start date (YYYY-MM-DD)"
)
@click.option("--holdout-end-date", required=True, help="Holdout end date (YYYY-MM-DD)")
@click.option(
    "--seed", default="42", show_default=True, help="Seed for reproducibility"
)
@click.option(
    "--output-root",
    default="results/holdout_eval",
    show_default=True,
    help="Output root (will write to <output-root>/<strategy_name>/).",
)
@click.option(
    "--data-path", default="data/parquet_data", show_default=True, help="Data directory"
)
@click.option("--feature-store-dir", default="feature_store", show_default=True)
@click.option("--feature-store-layer", default=None)
@click.option(
    "--deterministic/--non-deterministic", default=True, help="Deterministic training"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def diagnose_holdout_eval(
    config,
    symbol,
    timeframe,
    train_start_date,
    holdout_start_date,
    holdout_end_date,
    seed,
    output_root,
    data_path,
    feature_store_dir,
    feature_store_layer,
    deterministic,
    docker,
):
    """Train on [train_start_date, holdout_start_date) and evaluate on [holdout_start_date, holdout_end_date]."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config}" if use_workspace_prefix else config,
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--seed",
        str(seed),
        "--output-root",
        f"/workspace/{output_root}" if use_workspace_prefix else output_root,
        "--start-date",
        str(train_start_date),
        "--end-date",
        str(holdout_end_date),
        "--holdout-start-date",
        str(holdout_start_date),
        "--holdout-end-date",
        str(holdout_end_date),
    ]
    args.extend(
        [
            "--feature-store-dir",
            (
                f"/workspace/{feature_store_dir}"
                if use_workspace_prefix
                else feature_store_dir
            ),
        ]
    )
    if feature_store_layer is not None:
        args.extend(["--feature-store-layer", feature_store_layer])
    if deterministic:
        args.append("--deterministic")
    sys.exit(run_script("scripts/train_strategy_pipeline.py", args, docker=docker))


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
    default="config/strategies/sr_reversal_long",
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
    default="config/strategies/sr_reversal_long",
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


@backtest.command("strategy")
@click.option(
    "--strategy",
    "-c",
    required=True,
    help="Strategy name (matches config/strategies/<name>)",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 60T, 240T)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--model-path", help="Path to trained model file")
@click.option("--data-path", help="Path to feature data file (Parquet)")
@click.option(
    "--output-dir",
    default="results/backtest",
    help="Output directory for results",
)
@click.option(
    "--mode",
    type=click.Choice(["vectorized", "event-driven"]),
    default="vectorized",
    help="Backtest mode (vectorized is faster, event-driven is more realistic)",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def backtest_strategy(
    strategy,
    symbol,
    timeframe,
    start_date,
    end_date,
    model_path,
    data_path,
    output_dir,
    mode,
    docker,
):
    """
    Run strategy backtest with trained model.

    Example:
        mlbot backtest strategy -c sr_reversal_rr_reg_long -s BTCUSDT \\
            --start-date 2024-01-01 --end-date 2024-12-31 --no-docker
    """
    args = [
        "--strategy",
        strategy,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--mode",
        mode,
        "--output-dir",
        output_dir if not docker else f"/workspace/{output_dir}",
    ]
    if model_path:
        args.extend(
            ["--model-path", model_path if not docker else f"/workspace/{model_path}"]
        )
    if data_path:
        args.extend(
            ["--data-path", data_path if not docker else f"/workspace/{data_path}"]
        )

    sys.exit(
        run_python_module(
            "time_series_model.backtesting.nautilus_backtest_runner",
            args,
            docker=docker,
        )
    )


@backtest.command("visualize")
@click.option(
    "--strategy",
    "-c",
    required=True,
    help="Strategy name (matches config/strategies/<name>)",
)
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--data-path", required=True, help="Path to OHLCV parquet file")
@click.option("--trades-path", required=True, help="Path to trades JSON file")
@click.option("--model-path", help="Path to ModelArtifact directory (for SHAP)")
@click.option(
    "--output-path",
    default="results/backtest/report.html",
    help="Output HTML report path",
)
def backtest_visualize(
    strategy, symbol, data_path, trades_path, model_path, output_path
):
    """
    Generate interactive backtest visualization report.

    Creates an HTML report with:
    - Candlestick chart with trade markers
    - Trade list with PnL and exit reasons
    - SHAP feature importance (if model provided)

    Example:
        mlbot backtest visualize -c sr_reversal_rr_reg_long \\
            --data-path data/parquet_data/BTCUSDT/combined.parquet \\
            --trades-path results/backtest/sr_reversal/trades.json \\
            --model-path models/sr_reversal_rr_reg_long
    """
    from pathlib import Path
    import json
    import pandas as pd

    print(f"\n📊 Generating Backtest Visualization Report")
    print(f"   Strategy: {strategy}")
    print(f"   Symbol: {symbol}")
    print(f"   Data: {data_path}")
    print(f"   Trades: {trades_path}")

    try:
        from src.time_series_model.visualization.backtest_visualizer import (
            BacktestVisualizer,
        )

        # Load OHLCV data
        ohlcv_df = pd.read_parquet(data_path)

        # Load trades
        with open(trades_path) as f:
            trades_data = json.load(f)

        # Handle different trade formats
        if isinstance(trades_data, dict):
            trades = trades_data.get("trades", [])
        else:
            trades = trades_data

        # Load model artifact if provided
        model_artifact = None
        if model_path:
            try:
                from src.time_series_model.strategies.models.model_artifact import (
                    ModelArtifact,
                )

                model_artifact = ModelArtifact.load(Path(model_path))
                print(f"   ✅ Loaded ModelArtifact: {model_path}")
            except Exception as e:
                print(f"   ⚠️ Failed to load model: {e}")

        # Generate report
        visualizer = BacktestVisualizer(
            ohlcv_df=ohlcv_df,
            trades=trades,
            model_artifact=model_artifact,
            strategy_name=strategy,
            symbol=symbol,
        )

        report_path = visualizer.generate_report(output_path)
        print(f"\n✅ Report generated: {report_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Cross-Sectional Commands
# =============================================================================


@cli.group()
def cross_section():
    """Cross-sectional analysis commands."""
    pass


@cross_section.command("build-store")
@click.option(
    "--symbols",
    "-s",
    required=True,
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT).",
)
@click.option(
    "--timeframe",
    "-t",
    default="240T",
    show_default=True,
    help="Timeframe (e.g., 240T)",
)
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--data-path",
    default="data/parquet_data",
    show_default=True,
    help="Raw parquet root",
)
@click.option("--factor-set-yaml", required=True, help="YAML containing factor_sets")
@click.option("--factor-set", required=True, help="Factor set name to compute")
@click.option(
    "--feature-deps",
    default="config/feature_dependencies.yaml",
    show_default=True,
    help="Feature dependencies YAML",
)
@click.option(
    "--features-store-root",
    default="feature_store",
    show_default=True,
    help="FeatureStore root",
)
@click.option(
    "--features-store-layer", default=None, help="Optional layer name (default: hashed)"
)
@click.option(
    "--warmup-bars",
    default=600,
    show_default=True,
    help="Warmup bars before each month",
)
@click.option(
    "--include-ohlcv/--no-include-ohlcv",
    default=True,
    show_default=True,
    help="Include OHLCV in store",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing month files",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_build_store(
    symbols,
    timeframe,
    start_date,
    end_date,
    data_path,
    factor_set_yaml,
    factor_set,
    feature_deps,
    features_store_root,
    features_store_layer,
    warmup_bars,
    include_ohlcv,
    overwrite,
    docker,
):
    """Build monthly FeatureStore partitions for CS workflows (cacheable, no ticks)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--symbols",
        str(symbols),
        "--timeframe",
        str(timeframe),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--data-path",
        f"/workspace/{data_path}" if use_workspace_prefix else data_path,
        "--factor-set-yaml",
        f"/workspace/{factor_set_yaml}" if use_workspace_prefix else factor_set_yaml,
        "--factor-set",
        str(factor_set),
        "--feature-deps",
        f"/workspace/{feature_deps}" if use_workspace_prefix else feature_deps,
        "--features-store-root",
        (
            f"/workspace/{features_store_root}"
            if use_workspace_prefix
            else features_store_root
        ),
        "--warmup-bars",
        str(int(warmup_bars)),
    ]
    if features_store_layer:
        args.extend(["--features-store-layer", str(features_store_layer)])
    if not include_ohlcv:
        args.append("--no-include-ohlcv")
    if overwrite:
        args.append("--overwrite")

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/build_feature_store.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("report")
@click.option(
    "--input",
    "inputs",
    multiple=True,
    default=None,
    help="Input parquet/csv files or glob patterns (repeatable).",
)
@click.option(
    "--panel-path",
    default=None,
    help="(Deprecated) Alias for --input <panel.parquet>.",
)
@click.option(
    "--output",
    default="results/cross_sectional/fama_macbeth_report.md",
    show_default=True,
    help="Output markdown path",
)
@click.option("--symbols", default=None, help="Comma-separated symbols filter")
@click.option(
    "--horizon", default=12, show_default=True, help="Forward return horizon in bars"
)
@click.option(
    "--max-lag", default=5, show_default=True, help="Newey-West truncation lag"
)
@click.option(
    "--periods-per-year",
    default="auto",
    show_default=True,
    help="Annualisation factor or 'auto'",
)
@click.option(
    "--winsor",
    default=3.0,
    show_default=True,
    help="Sigma winsorisation (<=0 disables)",
)
@click.option(
    "--zscore/--no-zscore",
    default=True,
    show_default=True,
    help="Cross-sectional z-score per timestamp",
)
@click.option(
    "--crypto-factors/--no-crypto-factors",
    default=True,
    show_default=True,
    help="Add built-in crypto CS factors",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_report(
    inputs,
    panel_path,
    output,
    symbols,
    horizon,
    max_lag,
    periods_per_year,
    winsor,
    zscore,
    crypto_factors,
    docker,
):
    """Generate Fama-MacBeth + Newey-West + IC/IR markdown report."""
    use_workspace_prefix = docker and not _is_in_docker()
    in_list = list(inputs or [])
    if panel_path:
        in_list.append(panel_path)
    if not in_list:
        raise ValueError("Must pass --input (repeatable) or --panel-path.")
    args = []
    for x in in_list:
        args.extend(["--input", f"/workspace/{x}" if use_workspace_prefix else x])
    args.extend(
        ["--output", f"/workspace/{output}" if use_workspace_prefix else output]
    )
    args.extend(
        [
            "--horizon",
            str(horizon),
            "--max-lag",
            str(max_lag),
            "--periods-per-year",
            str(periods_per_year),
            "--winsor",
            str(winsor),
        ]
    )
    if symbols:
        args.extend(["--symbols", str(symbols)])
    if not zscore:
        args.append("--no-zscore")
    if not crypto_factors:
        args.append("--no-crypto-factors")

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_famacbeth_report.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("train")
@click.option(
    "--input",
    "inputs",
    multiple=True,
    default=None,
    help="Input parquet/csv files or glob patterns (repeatable).",
)
@click.option(
    "--panel-path",
    default=None,
    help="(Deprecated) Alias for --input <panel.parquet>.",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional/models",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--model",
    type=click.Choice(["boosting", "fama_macbeth"]),
    default="boosting",
    show_default=True,
    help="Model type",
)
@click.option("--symbols", default=None, help="Comma-separated symbols filter")
@click.option(
    "--horizon", default=12, show_default=True, help="Forward return horizon in bars"
)
@click.option(
    "--winsor",
    default=3.0,
    show_default=True,
    help="Sigma winsorisation (<=0 disables)",
)
@click.option(
    "--periods-per-year",
    default="auto",
    show_default=True,
    help="Annualisation factor or 'auto'",
)
@click.option(
    "--model-name",
    default="cs_boosting.joblib",
    show_default=True,
    help="Saved model filename",
)
@click.option(
    "--predictions-name",
    default="predictions.parquet",
    show_default=True,
    help="Saved predictions filename",
)
@click.option(
    "--metrics-name",
    default="metrics.json",
    show_default=True,
    help="Saved metrics filename",
)
@click.option(
    "--feature-cols", default=None, help="Optional comma-separated feature list"
)
@click.option(
    "--feature-file", default=None, help="Optional feature file (one per line)"
)
@click.option(
    "--auto-select/--no-auto-select",
    default=False,
    show_default=True,
    help="Auto-select factors via IC/IR",
)
@click.option(
    "--select-topk",
    default=0,
    show_default=True,
    help="Keep only top-K factors (0 disables)",
)
@click.option(
    "--ic-threshold", default=None, help="Minimum abs(IC mean) to keep a factor"
)
@click.option(
    "--ir-threshold", default=None, help="Minimum abs(IC IR) to keep a factor"
)
@click.option(
    "--selection-stat",
    type=click.Choice(["ic", "ir"]),
    default="ic",
    show_default=True,
    help="Rank stat for selection",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_train(
    inputs,
    panel_path,
    output_dir,
    model,
    symbols,
    horizon,
    winsor,
    periods_per_year,
    model_name,
    predictions_name,
    metrics_name,
    feature_cols,
    feature_file,
    auto_select,
    select_topk,
    ic_threshold,
    ir_threshold,
    selection_stat,
    docker,
):
    """Train cross-sectional models (boosting/Fama-MacBeth)."""
    use_workspace_prefix = docker and not _is_in_docker()
    in_list = list(inputs or [])
    if panel_path:
        in_list.append(panel_path)
    if not in_list:
        raise ValueError("Must pass --input (repeatable) or --panel-path.")
    args = []
    for x in in_list:
        args.extend(["--input", f"/workspace/{x}" if use_workspace_prefix else x])
    args.extend(
        [
            "--output-dir",
            f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
        ]
    )
    args.extend(
        [
            "--model",
            str(model),
            "--horizon",
            str(horizon),
            "--winsor",
            str(winsor),
            "--periods-per-year",
            str(periods_per_year),
            "--model-name",
            str(model_name),
            "--predictions-name",
            str(predictions_name),
            "--metrics-name",
            str(metrics_name),
        ]
    )
    if symbols:
        args.extend(["--symbols", str(symbols)])
    if feature_file:
        args.extend(
            [
                "--feature-file",
                f"/workspace/{feature_file}" if use_workspace_prefix else feature_file,
            ]
        )
    if feature_cols:
        args.extend(["--feature-cols", str(feature_cols)])
    if auto_select:
        args.append("--auto-select")
    if int(select_topk or 0) > 0:
        args.extend(["--select-topk", str(int(select_topk))])
    if ic_threshold is not None:
        args.extend(["--ic-threshold", str(ic_threshold)])
    if ir_threshold is not None:
        args.extend(["--ir-threshold", str(ir_threshold)])
    if selection_stat:
        args.extend(["--selection-stat", str(selection_stat)])

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/train_cross_sectional_model.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("catalog")
@click.option(
    "--input",
    "input_path",
    default="results/feature_exports/*.parquet",
    show_default=True,
    help="Input panel parquet/csv file or glob",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional/factor_sets",
    show_default=True,
    help="Output directory for exported factor sets",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_catalog(input_path, output_dir, docker):
    """Export factor catalog (IC/IR summary)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--input",
        f"/workspace/{input_path}" if use_workspace_prefix else input_path,
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
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
    "--input",
    "input_path",
    default="results/feature_exports/*.parquet",
    show_default=True,
    help="Input panel parquet/csv file or glob",
)
@click.option(
    "--output",
    default="results/cross_sectional/selected_factors.txt",
    show_default=True,
    help="Output selected factors text path",
)
@click.option(
    "--output-json",
    default="results/cross_sectional/selection_summary.json",
    show_default=True,
    help="Output selection summary JSON path",
)
@click.option("--target", default=None, help="Target column (default inferred)")
@click.option(
    "--min-assets", default=4, show_default=True, help="Minimum assets per timestamp"
)
@click.option(
    "--per-category-top", default=2, show_default=True, help="Top per category"
)
@click.option("--global-top", default=12, show_default=True, help="Global top-K")
@click.option("--ic-threshold", default=None, help="Minimum abs(IC mean) to keep")
@click.option("--ir-threshold", default=None, help="Minimum abs(IC IR) to keep")
@click.option(
    "--ranking-stat",
    type=click.Choice(["ic", "ir"]),
    default="ic",
    show_default=True,
    help="Ranking statistic",
)
@click.option(
    "--include-categories", default=None, help="Comma-separated categories to include"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_select(
    input_path,
    output,
    output_json,
    target,
    min_assets,
    per_category_top,
    global_top,
    ic_threshold,
    ir_threshold,
    ranking_stat,
    include_categories,
    docker,
):
    """Auto-select factors using correlation and IC filtering."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--input",
        f"/workspace/{input_path}" if use_workspace_prefix else input_path,
        "--min-assets",
        str(min_assets),
        "--per-category-top",
        str(per_category_top),
        "--global-top",
        str(global_top),
        "--ranking-stat",
        str(ranking_stat),
        "--output",
        f"/workspace/{output}" if use_workspace_prefix else output,
        "--output-json",
        f"/workspace/{output_json}" if use_workspace_prefix else output_json,
    ]
    if target:
        args.extend(["--target", str(target)])
    if ic_threshold is not None:
        args.extend(["--ic-threshold", str(ic_threshold)])
    if ir_threshold is not None:
        args.extend(["--ir-threshold", str(ir_threshold)])
    if include_categories:
        args.extend(["--include-categories", str(include_categories)])

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/auto_select_factors.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("shap")
@click.option(
    "--model",
    "model_path",
    default="results/cross_sectional/models/cs_boosting.joblib",
    show_default=True,
    help="Trained model joblib/pkl path",
)
@click.option(
    "--panel",
    "panel_path",
    default="results/feature_exports/cs_panel.parquet",
    show_default=True,
    help="Cross-sectional panel parquet/csv path",
)
@click.option(
    "--feature-file",
    default=None,
    help="Optional feature file (one feature per line) to restrict SHAP columns",
)
@click.option(
    "--target", default=None, help="Optional target column (default auto-detect)"
)
@click.option(
    "--topk", default=10, show_default=True, help="Top-K features for dependence plots"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
@click.option(
    "--output-dir",
    default="results/cross_sectional/shap_reports",
    show_default=True,
    help="Output directory for SHAP artifacts",
)
@click.option(
    "--max-samples",
    default=2000,
    show_default=True,
    help="Max samples for SHAP computation",
)
@click.option(
    "--interaction/--no-interaction",
    default=True,
    show_default=True,
    help="Compute SHAP interaction plot",
)
def cross_section_shap(
    model_path,
    panel_path,
    feature_file,
    target,
    topk,
    output_dir,
    max_samples,
    interaction,
    docker,
):
    """Run SHAP analysis on cross-sectional model."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--model",
        f"/workspace/{model_path}" if use_workspace_prefix else model_path,
        "--panel",
        f"/workspace/{panel_path}" if use_workspace_prefix else panel_path,
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
        "--topk",
        str(topk),
        "--max-samples",
        str(max_samples),
    ]
    if feature_file:
        args.extend(
            [
                "--feature-file",
                f"/workspace/{feature_file}" if use_workspace_prefix else feature_file,
            ]
        )
    if target:
        args.extend(["--target", str(target)])
    if not interaction:
        args.append("--no-interaction")

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_shap_analysis.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("logic-check")
@click.option(
    "--shap-manifest",
    default="results/cross_sectional/shap_reports/manifest.json",
    show_default=True,
    help="SHAP manifest.json produced by `mlbot cross-section shap`",
)
@click.option(
    "--expectations",
    default=None,
    help="Optional expectations YAML/JSON file for economic-logic checks",
)
@click.option(
    "--tolerance",
    default=0.0,
    show_default=True,
    help="Tolerance for expectation checks",
)
@click.option(
    "--output",
    default="results/cross_sectional/shap_logic_report.md",
    show_default=True,
    help="Output markdown report path",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_logic_check(shap_manifest, expectations, tolerance, output, docker):
    """Run factor logic consistency checks."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--shap-manifest",
        f"/workspace/{shap_manifest}" if use_workspace_prefix else shap_manifest,
        "--tolerance",
        str(tolerance),
        "--output",
        f"/workspace/{output}" if use_workspace_prefix else output,
    ]
    if expectations:
        args.extend(
            [
                "--expectations",
                f"/workspace/{expectations}" if use_workspace_prefix else expectations,
            ]
        )

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_factor_logic_check.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("shap-drift")
@click.option(
    "--current",
    default="results/cross_sectional/shap_reports/manifest.json",
    show_default=True,
    help="Current SHAP manifest.json",
)
@click.option(
    "--baseline",
    default="results/cross_sectional/shap_baseline.json",
    show_default=True,
    help="Baseline SHAP metrics JSON",
)
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    help="Alert threshold for drift scoring",
)
@click.option(
    "--output",
    default="results/cross_sectional/shap_drift_report.md",
    show_default=True,
    help="Output drift markdown report path",
)
@click.option(
    "--update-baseline",
    is_flag=True,
    default=False,
    help="Overwrite baseline with current metrics if no alerts",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_shap_drift(
    current, baseline, threshold, output, update_baseline, docker
):
    """Monitor SHAP value drift over time."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--current",
        f"/workspace/{current}" if use_workspace_prefix else current,
        "--baseline",
        f"/workspace/{baseline}" if use_workspace_prefix else baseline,
        "--threshold",
        str(threshold),
        "--output",
        f"/workspace/{output}" if use_workspace_prefix else output,
    ]
    if update_baseline:
        args.append("--update-baseline")

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/run_shap_drift_monitor.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("factor-eval")
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Optional YAML config for factor eval (recommended).",
)
@click.option(
    "--input",
    "input_path",
    default=None,
    help="Panel parquet/csv path. If provided, evaluate factors on this panel.",
)
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated symbols (required for FeatureStore source when --input is omitted).",
)
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option(
    "--features-store-root",
    default="feature_store",
    show_default=True,
    help="FeatureStore root",
)
@click.option(
    "--features-store-layer", default=None, help="FeatureStore layer (features_xxx)"
)
@click.option("--timeframe", "-t", default="240T", show_default=True, help="Timeframe")
@click.option(
    "--columns",
    default=None,
    help="Comma-separated FeatureStore columns to load (optional).",
)
@click.option(
    "--factors", default=None, help="Comma-separated factor columns to evaluate."
)
@click.option(
    "--factors-file", default=None, help="Path to factor list text file (one per line)."
)
@click.option(
    "--factor-set-yaml", default=None, help="YAML path containing factor_sets."
)
@click.option(
    "--factor-set", default=None, help="Factor set name in --factor-set-yaml."
)
@click.option(
    "--target",
    default=None,
    help="Target column (default: infer future_return_<horizon>).",
)
@click.option(
    "--horizon", default=12, show_default=True, help="Forward return horizon in bars."
)
@click.option(
    "--min-assets", default=4, show_default=True, help="Minimum assets per timestamp."
)
@click.option(
    "--quantiles", default=5, show_default=True, help="Quantiles for long/short."
)
@click.option(
    "--fee-bps", default=0.0, show_default=True, help="Fee (bps) applied to turnover."
)
@click.option(
    "--output-dir",
    default="results/cross_sectional/factor_eval",
    show_default=True,
    help="Output directory.",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_factor_eval(
    config_path,
    input_path,
    symbols,
    start_date,
    end_date,
    features_store_root,
    features_store_layer,
    timeframe,
    columns,
    factors,
    factors_file,
    factor_set_yaml,
    factor_set,
    target,
    horizon,
    min_assets,
    quantiles,
    fee_bps,
    output_dir,
    docker,
):
    """Cross-sectional factor evaluation (IC + long/short backtest)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
        "--timeframe",
        str(timeframe),
        "--horizon",
        str(int(horizon)),
        "--min-assets",
        str(int(min_assets)),
        "--quantiles",
        str(int(quantiles)),
        "--fee-bps",
        str(float(fee_bps)),
    ]
    if config_path:
        args.extend(
            [
                "--config",
                f"/workspace/{config_path}" if use_workspace_prefix else config_path,
            ]
        )
    if input_path:
        args.extend(
            [
                "--input",
                f"/workspace/{input_path}" if use_workspace_prefix else input_path,
            ]
        )
    if symbols:
        args.extend(["--symbols", str(symbols)])
    if start_date:
        args.extend(["--start-date", str(start_date)])
    if end_date:
        args.extend(["--end-date", str(end_date)])
    if features_store_root:
        args.extend(
            [
                "--features-store-root",
                (
                    f"/workspace/{features_store_root}"
                    if use_workspace_prefix
                    else features_store_root
                ),
            ]
        )
    if features_store_layer:
        args.extend(["--features-store-layer", str(features_store_layer)])
    if columns:
        args.extend(["--columns", str(columns)])
    if factors:
        args.extend(["--factors", str(factors)])
    if factors_file:
        args.extend(
            [
                "--factors-file",
                f"/workspace/{factors_file}" if use_workspace_prefix else factors_file,
            ]
        )
    if factor_set_yaml:
        args.extend(
            [
                "--factor-set-yaml",
                (
                    f"/workspace/{factor_set_yaml}"
                    if use_workspace_prefix
                    else factor_set_yaml
                ),
            ]
        )
    if factor_set:
        args.extend(["--factor-set", str(factor_set)])
    if target:
        args.extend(["--target", str(target)])

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/factor_eval.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("pipeline")
@click.option(
    "--config", "config_path", required=True, help="Pipeline YAML config path"
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_pipeline(config_path, docker):
    """Run end-to-end cross-sectional pipeline from a YAML config."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config_path}" if use_workspace_prefix else config_path,
    ]
    sys.exit(
        run_script(
            "src/cross_sectional/scripts/pipeline.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("rank")
@click.option("--date", required=True, help="Date (YYYY-MM-DD)")
@click.option("--factor", required=True, help="Factor/feature column name to rank by")
@click.option(
    "--factor-set-yaml",
    default=None,
    help="Optional YAML containing factor_sets (validate --factor).",
)
@click.option(
    "--factor-set", default=None, help="Factor set name in --factor-set-yaml."
)
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT). If set, overrides universe config.",
)
@click.option(
    "--universe-config",
    default=None,
    help="Universe config YAML (e.g., config/download/crypto_4h_token_universe_groups.yaml).",
)
@click.option(
    "--universe-set", default="starter_a", show_default=True, help="Universe set name"
)
@click.option(
    "--universe-groups",
    default=None,
    help="Comma-separated groups to include (e.g., highcap,alt). Default: all groups.",
)
@click.option(
    "--features-store-root",
    default="feature_store",
    show_default=True,
    help="FeatureStore root dir",
)
@click.option(
    "--features-store-layer",
    required=True,
    help="FeatureStore layer (e.g., features_83f12ecc5e)",
)
@click.option(
    "--timeframe",
    default="240T",
    show_default=True,
    help="Timeframe (e.g., 240T for 4H)",
)
@click.option(
    "--bar",
    type=click.Choice(["last", "first"]),
    default="last",
    show_default=True,
    help="Which bar within the day to use.",
)
@click.option(
    "--ascending",
    is_flag=True,
    default=False,
    help="Sort ascending (default: descending)",
)
@click.option("--top", default=50, show_default=True, help="Top-N rows to print")
@click.option("--output", default=None, help="Optional output path (.csv or .json)")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_rank(
    date,
    factor,
    factor_set_yaml,
    factor_set,
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    features_store_root,
    features_store_layer,
    timeframe,
    bar,
    ascending,
    top,
    output,
    docker,
):
    """Rank tokens cross-sectionally on a given date using a FeatureStore factor column."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--date",
        str(date),
        "--factor",
        str(factor),
        "--features-store-root",
        (
            f"/workspace/{features_store_root}"
            if use_workspace_prefix
            else features_store_root
        ),
        "--features-store-layer",
        str(features_store_layer),
        "--timeframe",
        str(timeframe),
        "--bar",
        str(bar),
        "--top",
        str(top),
    ]
    if factor_set_yaml:
        args.extend(
            [
                "--factor-set-yaml",
                (
                    f"/workspace/{factor_set_yaml}"
                    if use_workspace_prefix
                    else factor_set_yaml
                ),
            ]
        )
    if factor_set:
        args.extend(["--factor-set", str(factor_set)])
    if ascending:
        args.append("--ascending")
    if symbols:
        args.extend(["--symbols", str(symbols)])
    if universe_config:
        args.extend(
            [
                "--universe-config",
                (
                    f"/workspace/{universe_config}"
                    if use_workspace_prefix
                    else universe_config
                ),
                "--universe-set",
                str(universe_set),
            ]
        )
        if universe_groups:
            args.extend(["--universe-groups", str(universe_groups)])
    if output:
        args.extend(
            ["--output", f"/workspace/{output}" if use_workspace_prefix else output]
        )

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/rank_tokens.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("workflow")
@click.option(
    "--config",
    "config_path",
    default=None,
    help="YAML config path for config-driven workflow (recommended).",
)
@click.option(
    "--symbols",
    "-s",
    required=False,
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT,...)",
)
@click.option(
    "--timeframe",
    "-t",
    default="240T",
    show_default=True,
    help="Timeframe (e.g., 240T)",
)
@click.option(
    "--horizon", default=12, show_default=True, help="Forward return horizon in bars"
)
@click.option("--start-date", required=False, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=False, help="End date (YYYY-MM-DD)")
@click.option(
    "--feature-type",
    type=click.Choice(["baseline", "comprehensive"]),
    default="baseline",
    show_default=True,
    help="Feature recipe for CS panel generation.",
)
@click.option(
    "--data-path", default=None, help="Optional data root (default: data/parquet_data)"
)
@click.option(
    "--panel-out",
    default="results/feature_exports/cs_panel_workflow.parquet",
    show_default=True,
    help="Output panel parquet path",
)
@click.option(
    "--report-out",
    default="results/cross_sectional/fama_macbeth_report_workflow.md",
    show_default=True,
    help="Output markdown report path",
)
@click.option(
    "--train-out-dir",
    default="results/cross_sectional/models_workflow",
    show_default=True,
    help="Output directory for trained model artifacts",
)
@click.option(
    "--model",
    type=click.Choice(["boosting", "fama_macbeth"]),
    default="boosting",
    show_default=True,
    help="Model type",
)
@click.option(
    "--winsor",
    default=3.0,
    show_default=True,
    help="Sigma winsorisation (<=0 disables)",
)
@click.option(
    "--periods-per-year",
    default="auto",
    show_default=True,
    help="Annualisation factor or 'auto'",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_workflow(
    config_path,
    symbols,
    timeframe,
    horizon,
    start_date,
    end_date,
    feature_type,
    data_path,
    panel_out,
    report_out,
    train_out_dir,
    model,
    winsor,
    periods_per_year,
    docker,
):
    """End-to-end CS workflow. Requires --config (runs YAML pipeline including eval/select/report/train/backtest)."""
    use_workspace_prefix = docker and not _is_in_docker()

    if not config_path:
        raise ValueError("cross-section workflow now requires --config.")
    args = [
        "--config",
        f"/workspace/{config_path}" if use_workspace_prefix else config_path,
    ]
    sys.exit(
        run_script(
            "src/cross_sectional/scripts/pipeline.py",
            args,
            docker=docker,
        )
    )


@cross_section.command("nautilus-backtest")
@click.option(
    "--panel",
    required=True,
    help="Panel parquet/csv (e.g., output_root/panel_from_feature_store.parquet)",
)
@click.option(
    "--output-dir",
    default="results/cross_sectional/nautilus_backtest",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--signal",
    type=click.Choice(["model", "factor_combo"]),
    default="model",
    show_default=True,
    help="Signal source",
)
@click.option(
    "--model-path",
    default=None,
    help="Path to trained CS model joblib (for signal=model)",
)
@click.option(
    "--feature-file", default=None, help="Text file with feature columns (one per line)"
)
@click.option("--feature-cols", default=None, help="Comma-separated feature columns")
@click.option(
    "--mode",
    default="market_neutral",
    show_default=True,
    help="long_only | market_neutral",
)
@click.option("--holding", default=12, show_default=True, help="Holding period in bars")
@click.option("--lag", default=1, show_default=True, help="Execution lag in bars")
@click.option("--topk", default=10, show_default=True, help="Top-K longs")
@click.option(
    "--bottomk", default=10, show_default=True, help="Bottom-K shorts (market_neutral)"
)
@click.option(
    "--gross-leverage", default=1.0, show_default=True, help="Gross leverage cap"
)
@click.option(
    "--max-weight", default=0.10, show_default=True, help="Max abs weight per asset"
)
@click.option(
    "--turnover-limit", default=None, help="Optional turnover limit per rebalance"
)
@click.option(
    "--cash-buffer", default=0.10, show_default=True, help="Cash buffer fraction"
)
@click.option(
    "--equity-mode", default="compound", show_default=True, help="simple|compound|log"
)
@click.option("--fee-bps", default=2.0, show_default=True, help="Fee bps on turnover")
@click.option(
    "--slippage-bps", default=0.0, show_default=True, help="Slippage bps on turnover"
)
@click.option(
    "--funding-bps-per-bar",
    default=0.0,
    show_default=True,
    help="Funding bps per bar (short exposure)",
)
@click.option(
    "--borrow-bps-per-bar",
    default=0.0,
    show_default=True,
    help="Borrow bps per bar (short exposure)",
)
@click.option(
    "--min-assets", default=12, show_default=True, help="Min assets per timestamp"
)
@click.option(
    "--periods-per-year", default=None, help="Annualisation factor (bars/year)"
)
@click.option(
    "--html",
    default="report.html",
    show_default=True,
    help="HTML report filename under output-dir",
)
@click.option(
    "--max-trades",
    default=300,
    show_default=True,
    help="Max trades to show inline in HTML",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def cross_section_nautilus_backtest(
    panel,
    output_dir,
    signal,
    model_path,
    feature_file,
    feature_cols,
    mode,
    holding,
    lag,
    topk,
    bottomk,
    gross_leverage,
    max_weight,
    turnover_limit,
    cash_buffer,
    equity_mode,
    fee_bps,
    slippage_bps,
    funding_bps_per_bar,
    borrow_bps_per_bar,
    min_assets,
    periods_per_year,
    html,
    max_trades,
    docker,
):
    """CS portfolio backtest in a Nautilus-style bar-driven execution loop (re-run from panel + model/factor-combo)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--panel",
        f"/workspace/{panel}" if use_workspace_prefix else panel,
        "--output-dir",
        f"/workspace/{output_dir}" if use_workspace_prefix else output_dir,
        "--signal",
        str(signal),
        "--mode",
        str(mode),
        "--holding",
        str(int(holding)),
        "--lag",
        str(int(lag)),
        "--topk",
        str(int(topk)),
        "--bottomk",
        str(int(bottomk)),
        "--gross-leverage",
        str(float(gross_leverage)),
        "--max-weight",
        str(float(max_weight)),
        "--cash-buffer",
        str(float(cash_buffer)),
        "--equity-mode",
        str(equity_mode),
        "--fee-bps",
        str(float(fee_bps)),
        "--slippage-bps",
        str(float(slippage_bps)),
        "--funding-bps-per-bar",
        str(float(funding_bps_per_bar)),
        "--borrow-bps-per-bar",
        str(float(borrow_bps_per_bar)),
        "--min-assets",
        str(int(min_assets)),
        "--html",
        str(html),
        "--max-trades",
        str(int(max_trades)),
    ]
    if model_path:
        args += [
            "--model-path",
            f"/workspace/{model_path}" if use_workspace_prefix else model_path,
        ]
    if feature_file:
        args += [
            "--feature-file",
            f"/workspace/{feature_file}" if use_workspace_prefix else feature_file,
        ]
    if feature_cols:
        args += ["--feature-cols", str(feature_cols)]
    if turnover_limit is not None:
        args += ["--turnover-limit", str(turnover_limit)]
    if periods_per_year is not None:
        args += ["--periods-per-year", str(periods_per_year)]

    sys.exit(
        run_script(
            "src/cross_sectional/scripts/nautilus_backtest.py",
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
