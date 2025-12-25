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
    # Ensure PYTHONPATH includes project root for src imports
    pythonpath_parts = [str(PROJECT_ROOT / "src")]
    if "PYTHONPATH" in env:
        pythonpath_parts.insert(0, env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

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
            if getattr(conn, "status", None) != getattr(psutil, "CONN_LISTEN", "LISTEN"):
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


@cli.command("serve-results")
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
    """Serve the results/ directory via a local static server (HTML reports)."""
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


@cli.group()
def nnmultihead():
    """NN multi-head base model (path primitives) commands."""
    pass


@cli.group()
def rule():
    """Rule router commands (3-action: NO_TRADE/MEAN/TREND)."""
    pass


@cli.group()
def rl():
    """RL-ready router tooling (3-action BC/shadow/counterfactual/FSM) using logs."""
    pass


@rl.command("build-logs-3action")
@click.option("--preds", "preds_path", required=True, help="Preds file/dir from nnmultihead predict (preds_*.parquet)")
@click.option("--mode", "mode_path", default=None, help="Optional mode file/dir from mlbot rule mode-3action")
@click.option("--symbols", "-s", default=None, help="Optional symbols filter (comma-separated). If omitted, infer from preds.")
@click.option("--data-path", default="data/parquet_data", help="Raw data directory")
@click.option("--timeframe", default="240T", help="Timeframe (must match preds)")
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
@click.option("--model", "model_path", default=None, help="Optional model.pt to infer preds_in_log1p")
@click.option("--preds-in-log1p", type=click.Choice(["yes", "no"]), default=None, help="Override preds space (yes=log1p)")
@click.option("--returns-source", type=click.Choice(["momentum_proxy", "rr_execution"]), default="momentum_proxy", help="How to build ret_mean/ret_trend")
@click.option("--momentum-lookback", type=int, default=5, help="Lookback for momentum proxy used in ret_mean/ret_trend")
@click.option("--output", "output_path", required=True, help="Output logs path (.parquet/.csv)")
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
    ]
    if mode_path:
        args.extend(["--mode", f"/workspace/{mode_path}" if use_workspace_prefix else mode_path])
    if symbols:
        args.extend(["--symbols", str(symbols)])
    if start_date:
        args.extend(["--start-date", str(start_date)])
    if end_date:
        args.extend(["--end-date", str(end_date)])
    if model_path:
        args.extend(["--model", f"/workspace/{model_path}" if use_workspace_prefix else model_path])
    if preds_in_log1p:
        args.extend(["--preds-in-log1p", str(preds_in_log1p)])

    sys.exit(run_script("scripts/rl_build_logs_3action.py", args, docker=docker))


@rule.command("mode-3action")
@click.option("--preds", required=True, help="Preds file (.parquet/.csv) or directory of per-symbol preds_*.parquet")
@click.option("--model", "model_path", default=None, help="Optional model.pt to infer whether preds are log1p targets")
@click.option("--preds-in-log1p", type=click.Choice(["yes", "no"]), default=None, help="Override preds space (yes=log1p)")
@click.option("--output", "output_path", required=True, help="Output path (.parquet or .csv)")
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
        args.extend(["--model", f"/workspace/{model_path}" if use_workspace_prefix else model_path])
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
@click.option("--logs", "logs_path", required=True, help="Logs .csv/.parquet with columns: symbol,timestamp,mode,head_*")
@click.option("--out", "out_dir", required=True, help="Output directory for artifacts (metrics/report).")
@click.option("--train-ratio", type=float, default=0.7, help="Train ratio per symbol (time-ordered).")
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
@click.option("--logs", "logs_path", required=True, help="Logs .csv/.parquet with mode + ret_mean/ret_trend + head_*")
@click.option("--out", "out_dir", required=True, help="Output directory for artifacts (metrics/report).")
@click.option("--train-ratio", type=float, default=0.7, help="Train ratio per symbol (time-ordered).")
@click.option("--entry-delay", type=int, default=0, help="Entry delay steps for sim.")
@click.option("--cost-per-turnover", type=float, default=0.0, help="Cost per turnover unit.")
@click.option("--slippage-bps", type=float, default=0.0, help="Slippage bps per abs exposure.")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_counterfactual_eval_3action(
    logs_path,
    out_dir,
    train_ratio,
    entry_delay,
    cost_per_turnover,
    slippage_bps,
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
    ]
    sys.exit(run_script("scripts/rl_counterfactual_eval_3action.py", args, docker=docker))


@rl.command("fsm-decide")
@click.option("--metrics", "metrics_path", required=True, help="metrics.json produced by counterfactual-eval-3action")
@click.option("--state", default="RL_CANDIDATE", help="Initial FSM state: RULE/RL_CANDIDATE/RL_ACTIVE/RL_SUSPENDED")
@click.option("--promote-days", type=int, default=10, help="Consecutive ok windows required to promote.")
@click.option("--cooldown-days", type=int, default=20, help="Cooldown windows after suspension.")
@click.option("--out", "out_path", default=None, help="Optional path to write decision json.")
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def rl_fsm_decide(metrics_path, state, promote_days, cooldown_days, out_path, docker):
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
    ]
    if out_path:
        args.extend(["--out", f"/workspace/{out_path}" if use_workspace_prefix else out_path])
    sys.exit(run_script("scripts/rl_fsm_decide.py", args, docker=docker))


@rl.command("run-e2e-3action")
@click.option("--logs", "logs_path", required=True, help="Logs .csv/.parquet with mode + heads (+ ret_mean/ret_trend for counterfactual).")
@click.option("--out", "out_dir", required=True, help="Output directory root. Will create shadow/ counterfactual/ fsm_decision.json")
@click.option("--train-ratio", type=float, default=0.7, help="Train ratio per symbol (time-ordered).")
@click.option("--entry-delay", type=int, default=0, help="Entry delay steps for sim.")
@click.option("--cost-per-turnover", type=float, default=0.0, help="Cost per turnover unit.")
@click.option("--slippage-bps", type=float, default=0.0, help="Slippage bps per abs exposure.")
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
        ["--logs", logs_arg, "--out", shadow_out, "--train_ratio", str(float(train_ratio))],
        docker=docker,
    )
    if rc != 0:
        sys.exit(rc)

    rc = run_script(
        "scripts/rl_counterfactual_eval_3action.py",
        [
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
        ],
        docker=docker,
    )
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
@click.option("--symbols", "-s", default="BTCUSDT", help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)")
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
@click.option("--horizon-hours", type=float, default=80.0, help="Future horizon in hours (e.g., 80H)")
@click.option("--bar-hours", type=float, default=4.0, help="Bar duration in hours (4H => 4)")
@click.option("--epochs", type=int, default=30, help="Training epochs")
@click.option("--batch-size", type=int, default=512, help="Batch size")
@click.option("--lr", type=float, default=2e-4, help="Learning rate")
@click.option("--hidden", type=int, default=256, help="MLP hidden size")
@click.option("--depth", type=int, default=2, help="MLP depth")
@click.option("--dropout", type=float, default=0.1, help="Dropout")
@click.option("--device", default=None, help="cpu|cuda (default auto)")
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
    output_dir,
    docker,
):
    """Train NN multi-head path primitives MLP and save report.html artifacts."""
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

    sys.exit(run_script("scripts/train_path_primitives_mlp.py", args, docker=docker))


@nnmultihead.command("predict")
@click.option("--symbols", "-s", default="BTCUSDT", help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)")
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
@click.option("--model", "model_path", required=True, help="Path to model.pt produced by nnmultihead train")
@click.option("--output", "output_path", required=True, help="Output path (.parquet or .csv)")
@click.option("--device", default=None, help="cpu|cuda (default auto)")
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
    docker,
):
    """Run inference and save heads/preds for downstream Router/RL."""
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

    sys.exit(run_script("scripts/predict_path_primitives_mlp.py", args, docker=docker))


@nnmultihead.command("eval")
@click.option("--symbols", "-s", default="BTCUSDT", help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)")
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
@click.option("--model", "model_path", required=True, help="Path to model.pt produced by nnmultihead train")
@click.option("--horizon-hours", type=float, default=80.0, help="Future horizon in hours (e.g., 80H)")
@click.option("--bar-hours", type=float, default=4.0, help="Bar duration in hours (4H => 4)")
@click.option("--device", default=None, help="cpu|cuda (default auto)")
@click.option("--output-dir", default="results/nnmultihead_eval", help="Output directory for eval artifacts")
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

def _train_strategy_pipeline(symbol, timeframe, config, data_path, test_size, output_root, docker):
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
    sys.exit(run_script("scripts/train_strategy_pipeline.py", args, docker=docker))


@train.command("sr-reversal-long")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option(
    "--timeframe", "-t", default="240T", help="Timeframe (e.g., 15T, 60T, 240T)"
)
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option(
    "--output-root",
    default="results/strategies/sr_reversal_long",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_sr_reversal_long(symbol, timeframe, data_path, test_size, output_root, docker):
    """Train SR Reversal Long-only model (direction-fixed)."""
    _train_strategy_pipeline(
        symbol=symbol,
        timeframe=timeframe,
        config="config/strategies/sr_reversal_long",
        data_path=data_path,
        test_size=test_size,
        output_root=output_root,
        docker=docker,
    )


@train.command("sr-reversal-short")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option(
    "--timeframe", "-t", default="240T", help="Timeframe (e.g., 15T, 60T, 240T)"
)
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option("--test-size", default="0.15", help="Test set ratio")
@click.option(
    "--output-root",
    default="results/strategies/sr_reversal_short",
    help="Output directory",
)
@click.option("--docker/--no-docker", default=True, help="Run in Docker")
def train_sr_reversal_short(symbol, timeframe, data_path, test_size, output_root, docker):
    """Train SR Reversal Short-only model (direction-fixed)."""
    _train_strategy_pipeline(
        symbol=symbol,
        timeframe=timeframe,
        config="config/strategies/sr_reversal_short",
        data_path=data_path,
        test_size=test_size,
        output_root=output_root,
        docker=docker,
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
    help="Strategy config directory",
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
def diagnose_model_comparison(
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
    """Compare Rule-based vs ML vs ML+Volatility models."""
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
            ["--rule-params", f"/workspace/{rule_params}" if use_workspace_prefix else rule_params]
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
