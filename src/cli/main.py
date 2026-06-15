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
from typing import Optional, List, Dict, Any

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

from src.config.strategy_layout import PACKAGED_PROFILE_DEFAULT_STEM


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
        cmd = (
            [
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
            ]
            + extra_env_flags
            + [
                "-v",
                f"{PROJECT_ROOT}:/workspace",
                "-w",
                "/workspace",
                "--shm-size=8gb",
                docker_image,
                "python3",
                "-m",
                module,
            ]
            + args
        )
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
        cmd = (
            [
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
            ]
            + extra_env_flags
            + [
                "-v",
                f"{PROJECT_ROOT}:/workspace",
                "-w",
                "/workspace",
                "--shm-size=8gb",
                docker_image,
                "python3",
                f"/workspace/{script_path}",
            ]
            + args
        )
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
# TaskSpec -> derived nnmultihead config (single entrypoint)
# =============================================================================


def materialize_nnmh_config_from_task_spec(
    *,
    task_spec_path: str,
    base_config_dir: str,
    out_config_dir: str,
) -> str:
    """
    Materialize a concrete nnmultihead config directory from TaskSpec tiers.

    Single-source-of-truth rule (enforced by design):
    - required feature nodes come ONLY from TaskSpec tiers (tier files).
    - optional block enablement comes ONLY from TaskSpec optional_blocks_enabled.
    - base config's features.yaml is treated as a template (schema + block library).
    """
    import yaml
    import shutil
    import json as _json2

    ts_path = Path(task_spec_path)
    if not ts_path.is_absolute():
        ts_path = (PROJECT_ROOT / ts_path).resolve()
    if not ts_path.exists():
        raise click.ClickException(f"TaskSpec not found: {ts_path}")
    ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}

    # Feature plan sources (in priority order):
    # 1) feature_plan_ref (recommended): load nnmultihead-owned feature plan file
    # 2) feature_plan_overrides: per-task overrides
    # 3) feature_plan (inline): legacy/temporary compatibility
    fp: Dict[str, Any] = {}
    plan_ref = str(ts_obj.get("feature_plan_ref") or "").strip()
    if plan_ref:
        p = Path(plan_ref)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise click.ClickException(f"feature_plan_ref not found: {p}")
        objp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if isinstance(objp, dict):
            fp = objp.get("feature_plan") or {}
    # Apply overrides if present
    fp_over = ts_obj.get("feature_plan_overrides") or {}
    if isinstance(fp_over, dict) and fp_over:
        fp = {**fp, **fp_over}
    # Apply inline as last (compat)
    fp_inline = ts_obj.get("feature_plan") or {}
    if isinstance(fp_inline, dict) and fp_inline:
        fp = {**fp, **fp_inline}
    tiers_enabled = fp.get("tiers_enabled") or []
    tier_feature_files = fp.get("tier_feature_files") or {}
    if not (isinstance(tiers_enabled, list) and isinstance(tier_feature_files, dict)):
        raise click.ClickException(
            "TaskSpec missing feature_plan.tiers_enabled or tier_feature_files"
        )

    base_dir = Path(base_config_dir)
    if not base_dir.is_absolute():
        base_dir = (PROJECT_ROOT / base_dir).resolve()
    if not base_dir.exists():
        raise click.ClickException(f"Base config dir not found: {base_dir}")

    outp = Path(out_config_dir)
    if not outp.is_absolute():
        outp = (PROJECT_ROOT / outp).resolve()
    if outp.exists():
        shutil.rmtree(outp)
    shutil.copytree(base_dir, outp)

    # AUTO-DETECT TIER FEATURES: 自动检测并添加gate规则需要的tier features
    # 在复制之后修改out_config_dir中的tier文件，这样不会影响base_config_dir
    added_tier_nodes: List[str] = []
    try:
        from src.cli.auto_detect_compute_requirements import (
            extract_required_features_from_execution_archetypes,
            map_features_to_tier_nodes,
            ensure_tier_features,
        )

        deps_path = (PROJECT_ROOT / "config/feature_dependencies.yaml").resolve()
        deps = yaml.safe_load(deps_path.read_text(encoding="utf-8")) or {}

        # 提取gate规则需要的特征
        required_features = extract_required_features_from_execution_archetypes(
            PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml"
        )

        if required_features:
            # 映射到feature nodes
            required_nodes = map_features_to_tier_nodes(required_features, deps)

            if required_nodes:
                # 对每个启用的tier，检查并添加缺失的nodes
                for tier_name in tiers_enabled:
                    tier_file_rel = tier_feature_files.get(str(tier_name))
                    if not tier_file_rel:
                        continue
                    tier_file_path = Path(tier_file_rel)
                    if not tier_file_path.is_absolute():
                        tier_file_path = outp / tier_file_rel
                    else:
                        # 如果是绝对路径，需要转换为out_config_dir中的路径
                        # 假设tier文件在base_config_dir的某个子目录中
                        tier_file_path = outp / tier_file_rel

                    if tier_file_path.exists():
                        added = ensure_tier_features(
                            required_nodes, tier_file_path, deps
                        )
                        added_tier_nodes.extend(added)

                if added_tier_nodes:
                    click.echo(
                        f"🔍 Auto-added tier features: {sorted(set(added_tier_nodes))} "
                        f"(gate/regime needs)",
                        err=True,
                    )
    except Exception as e:
        # 如果自动推导失败，不影响正常流程（向后兼容）
        click.echo(
            f"⚠️  Auto-detect tier features failed: {e}. Using existing tier files only.",
            err=True,
        )

    # Collect feature nodes from enabled tier files (after auto-adding)
    # 从out_config_dir读取tier文件，因为可能已经被修改
    tier_nodes: List[str] = []
    for tier_name in tiers_enabled:
        k = str(tier_name).strip()
        fpath = tier_feature_files.get(k)
        if not fpath:
            continue
        # 优先从out_config_dir读取（可能已被修改）
        p = outp / fpath
        if not p.exists():
            # 回退到从PROJECT_ROOT读取（如果out_config_dir中没有）
            p = Path(str(fpath))
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise click.ClickException(
                f"TaskSpec tier_feature_files[{k}] not found: {p}"
            )
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise click.ClickException(f"Tier file must be a YAML list: {p}")
        tier_nodes.extend([str(x).strip() for x in obj if str(x).strip()])
    tier_nodes = sorted(set(tier_nodes))
    if not tier_nodes:
        raise click.ClickException(
            "No tier feature nodes collected (tiers_enabled/tier_feature_files mismatch)."
        )

    base_dir = Path(base_config_dir)
    if not base_dir.is_absolute():
        base_dir = (PROJECT_ROOT / base_dir).resolve()
    if not base_dir.exists():
        raise click.ClickException(f"Base config dir not found: {base_dir}")

    outp = Path(out_config_dir)
    if not outp.is_absolute():
        outp = (PROJECT_ROOT / outp).resolve()
    if outp.exists():
        shutil.rmtree(outp)
    shutil.copytree(base_dir, outp)

    feat_path = outp / "features.yaml"
    feat_obj = yaml.safe_load(feat_path.read_text(encoding="utf-8")) or {}
    fp2 = feat_obj.get("feature_pipeline") or {}
    # TaskSpec can override exclude_columns (still computed, but excluded from MLP inputs)
    exc = fp.get("exclude_columns", None)
    if exc is not None:
        if not isinstance(exc, list):
            raise click.ClickException(
                "TaskSpec feature_plan.exclude_columns must be a list when provided"
            )
        fp2["exclude_columns"] = [str(x).strip() for x in exc if str(x).strip()]
    req = fp2.get("requested_features") or {}
    if not isinstance(req, dict):
        req = {}

    # Required features come ONLY from tiers
    req["required"] = tier_nodes

    # Optional blocks are enabled/disabled ONLY by FeaturePlan (via TaskSpec feature_plan_ref / overrides).
    # Block definitions (library) also live ONLY in FeaturePlan to avoid duplicated definitions.
    ob_library = fp.get("optional_blocks_library", {}) or {}
    if not isinstance(ob_library, dict):
        raise click.ClickException(
            "TaskSpec feature_plan.optional_blocks_library must be a dict when provided"
        )
    ob_enabled = fp.get("optional_blocks_enabled", [])
    if not isinstance(ob_enabled, list):
        raise click.ClickException(
            "TaskSpec feature_plan.optional_blocks_enabled must be a list"
        )
    enabled_keys = {str(x).strip() for x in ob_enabled if str(x).strip()}

    # AUTO-DETECT: 自动推导gate/regime需要的blocks（方案3：自动推导计算需求）
    try:
        from src.cli.auto_detect_compute_requirements import (
            auto_detect_compute_requirements,
        )

        auto_detected_blocks = auto_detect_compute_requirements(
            task_spec_path=ts_path,
            execution_archetypes_path=PROJECT_ROOT
            / "config/nnmultihead/execution_archetypes.yaml",
            feature_dependencies_path=PROJECT_ROOT / "config/feature_dependencies.yaml",
        )
        if auto_detected_blocks:
            # 合并自动推导的blocks（不覆盖用户显式指定的）
            enabled_keys = enabled_keys | auto_detected_blocks
            click.echo(
                f"🔍 Auto-detected compute requirements: {sorted(auto_detected_blocks)} "
                f"(gate/regime needs). Total enabled: {sorted(enabled_keys)}",
                err=True,
            )
    except Exception as e:
        # 如果自动推导失败，不影响正常流程（向后兼容）
        click.echo(
            f"⚠️  Auto-detect compute requirements failed: {e}. Using manual config only.",
            err=True,
        )
    if not enabled_keys:
        req["optional_blocks"] = {}
    else:
        missing = sorted([k for k in enabled_keys if k not in ob_library])
        if missing:
            raise click.ClickException(
                f"TaskSpec optional_blocks_enabled references missing blocks in optional_blocks_library: {missing}"
            )
        req["optional_blocks"] = {k: ob_library.get(k) for k in sorted(enabled_keys)}

    fp2["requested_features"] = req
    feat_obj["feature_pipeline"] = fp2

    # Rebuild feature_contract from FeaturePlan:
    # - FeaturePlan.feature_contract provides stable contract semantics (baseline minimal cols + missingness_policy)
    # - minimal_required_cols overwritten by (baseline minimal + tier output_columns union)
    deps_path = (PROJECT_ROOT / "config/feature_dependencies.yaml").resolve()
    deps = yaml.safe_load(deps_path.read_text(encoding="utf-8")) or {}
    feats = deps.get("features") or {}
    out_cols: List[str] = []
    for node in tier_nodes:
        meta = feats.get(str(node)) if isinstance(feats, dict) else None
        cols = meta.get("output_columns") if isinstance(meta, dict) else None
        if isinstance(cols, list) and cols:
            out_cols.extend([str(c).strip() for c in cols if str(c).strip()])
    out_cols = sorted(set(out_cols))

    # Baseline contract comes from FeaturePlan (kept in the feature_plan_ref file).
    fc0 = fp.get("feature_contract") if isinstance(fp, dict) else None
    fc0 = fc0 if isinstance(fc0, dict) else {}

    baseline_min = fc0.get("minimal_required_cols") if isinstance(fc0, dict) else None
    if not isinstance(baseline_min, list):
        baseline_min = []
    minimal_required_cols = sorted(
        set([str(x).strip() for x in baseline_min if str(x).strip()] + out_cols)
    )

    fc_new: Dict[str, Any] = {}
    if fc0:
        # Keep semantics from FeaturePlan.feature_contract (missingness policy only).
        if "missingness_policy" in fc0 and isinstance(
            fc0.get("missingness_policy"), dict
        ):
            fc_new["missingness_policy"] = fc0.get("missingness_policy")

    # Build feature_contract.optional_blocks from enabled optional blocks (node-level -> output columns).
    # This avoids duplicated definitions and keeps block mapping tied to actual enabled feature nodes.
    block_cols_patterns: Dict[str, List[str]] = {}
    for bname in sorted(enabled_keys):
        nodes = ob_library.get(bname)
        if not isinstance(nodes, list):
            continue
        cols_acc: List[str] = []
        for node in nodes:
            meta = feats.get(str(node)) if isinstance(feats, dict) else None
            cols = meta.get("output_columns") if isinstance(meta, dict) else None
            if isinstance(cols, list) and cols:
                cols_acc.extend([str(c).strip() for c in cols if str(c).strip()])
        # De-dup while preserving order
        seen = set()
        cols_acc = [c for c in cols_acc if not (c in seen or seen.add(c))]
        if cols_acc:
            # Use exact columns (not wildcards) to avoid accidentally capturing core columns.
            block_cols_patterns[str(bname)] = cols_acc
    if block_cols_patterns:
        fc_new["optional_blocks"] = block_cols_patterns
    fc_new["minimal_required_cols"] = minimal_required_cols
    feat_obj["feature_contract"] = fc_new

    feat_path.write_text(
        yaml.safe_dump(feat_obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    marker = {
        "kind": "nnmultihead_config_derived_from_task_spec",
        "task_id": str(ts_obj.get("task_id") or "").strip(),
        "base_config_dir": str(base_config_dir),
        "tiers_enabled": list(tiers_enabled),
        "optional_blocks_enabled": list(ob_enabled),
        "tier_nodes": list(tier_nodes),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outp / "derived_from_task_spec.json").write_text(
        _json2.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return str(outp)


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
    """DEPRECATED: use ``mlbot rolling-dashboard`` (results static + /browse + /rd)."""
    click.echo(
        "⚠️  DEPRECATED: `mlbot server` 请改用 `mlbot rolling-dashboard` "
        "（含 /browse 与 /rd 实验管理）。"
    )
    from scripts.rolling_dashboard_server import run_from_project

    run_from_project(
        PROJECT_ROOT,
        bind=bind,
        port=int(port),
        results_rel=directory,
        force=force,
    )


@cli.command("rolling-dashboard")
@click.option("--port", "-p", type=int, default=8008, show_default=True)
@click.option(
    "--dir",
    "-d",
    "directory",
    default="results",
    show_default=True,
    help="Directory to serve (same layout as mlbot server)",
)
@click.option(
    "--bind",
    default="127.0.0.1",
    show_default=True,
    help="Bind address",
)
@click.option(
    "--force",
    is_flag=True,
    help="If port is in use, kill the listener (requires psutil)",
)
def rolling_dashboard(port: int, directory: str, bind: str, force: bool) -> None:
    """本地研发：results 静态 + /browse + /rd 实验管理（与实盘 CMS 分离）。"""
    from scripts.rolling_dashboard_server import run_from_project

    run_from_project(
        PROJECT_ROOT,
        bind=bind,
        port=int(port),
        results_rel=directory,
        force=force,
    )


@cli.command("console")
@click.option("--port", "-p", type=int, default=8800, show_default=True, help="Port")
@click.option(
    "--bind",
    default="127.0.0.1",
    show_default=True,
    help="Bind address (use 0.0.0.0 for devcontainer port forwarding)",
)
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev)")
@click.option(
    "--force",
    is_flag=True,
    help="If port is in use, kill the process listening on the port and retry",
)
def console_cmd(port: int, bind: str, reload: bool, force: bool) -> None:
    """实盘 Business CMS（Trade Map / orders / account）；不含本地实验管理。"""
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError as exc:
        raise click.ClickException(
            "uvicorn not installed. Run: pip install -r deploy/business-console/requirements.txt"
        ) from exc

    if force and _port_is_in_use(port, bind=bind):
        pids = _find_listening_pids(port)
        if pids:
            click.echo(f"⚠️  Port {port} in use by PID(s) {pids}; terminating (--force)...")
            _kill_pids(pids)
            import time as _time

            for _ in range(30):
                if not _port_is_in_use(port, bind=bind):
                    break
                _time.sleep(0.1)

    click.echo("🌐 MLBot Business Console — 实盘 CMS (FastAPI)")
    click.echo(f"   bind:    http://{bind}:{port}/")
    click.echo(f"   Trade Map: http://{bind if bind not in ('0.0.0.0', '::') else 'localhost'}:{port}/trade-map")
    click.echo("   本地 R&D 实验: mlbot rolling-dashboard → /rd")
    click.echo("   Ctrl+C 停止")

    src_root = PROJECT_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from mlbot_console.main import app

    uvicorn.run(
        app,
        host=bind,
        port=int(port),
        reload=bool(reload),
    )


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
    input_dir: Optional[str],
    output_dir: Optional[str],
    pattern: Optional[str],
    symbols: Optional[str],
    force: bool,
    aggregate_freq: Optional[str],
    docker: bool,
) -> int:
    args = []
    if pattern:
        args.extend(["--pattern", str(pattern)])
    if symbols:
        args.extend(["--symbols", str(symbols)])
    if input_dir:
        args.extend(["--input-dir", input_dir])
    if output_dir:
        args.extend(["--output-dir", output_dir])
    if force:
        args.append("--force")
    if aggregate_freq:
        args.extend(["--aggregate-freq", aggregate_freq])
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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


@data.command("download-open-interest")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.option(
    "--universe-config",
    default=None,
    help="YAML universe config (if set, overrides --symbols).",
)
@click.option("--universe-set", default="starter_a")
@click.option("--universe-groups", default=None)
@click.option("--start-year", default="2023", help="Start year")
@click.option("--start-month", default="1", help="Start month")
@click.option("--end-year", help="End year (default: current)")
@click.option("--end-month", help="End month (default: current)")
@click.option(
    "--period",
    default="5m",
    type=click.Choice(["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]),
    show_default=True,
    help="OI aggregation period",
)
@click.option(
    "--parquet-dir",
    default="data/open_interest/parquet",
    help="Output directory for OI Parquet",
)
@click.option(
    "--sleep-sec",
    type=float,
    default=0.35,
    show_default=True,
    help="Sleep between API calls (rate-limit friendly)",
)
@click.option(
    "--progress-every",
    type=int,
    default=25,
    show_default=True,
    help="Print progress every N tasks (0 disables)",
)
@click.option("--force/--no-force", default=False, show_default=True)
def data_download_open_interest(
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    start_year,
    start_month,
    end_year,
    end_month,
    period,
    parquet_dir,
    sleep_sec,
    progress_every,
    force,
):
    """Download Binance futures Open Interest history (API → Parquet)."""
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
        "--parquet-dir",
        parquet_dir,
        "--period",
        period,
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

    sys.exit(run_script("src/data_tools/download_open_interest.py", args))


@data.command("download-open-interest-vision")
@click.option(
    "--symbols", "-s", default="HYPEUSDT", help="Comma-separated symbols"
)
@click.option(
    "--universe-config",
    default=None,
    help="YAML universe config (if set, overrides --symbols).",
)
@click.option("--universe-set", default="starter_a")
@click.option("--universe-groups", default=None)
@click.option("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
@click.option("--end-date", default=None, help="YYYY-MM-DD inclusive (default: today UTC)")
@click.option(
    "--data-dir",
    default="data/open_interest/vision_zip",
    help="Cache directory for daily metrics ZIP files",
)
@click.option(
    "--parquet-dir",
    default="data/open_interest/parquet",
    help="Output directory for monthly OI Parquet",
)
@click.option("--sleep-sec", type=float, default=0.2, show_default=True)
@click.option("--progress-every", type=int, default=25, show_default=True)
@click.option("--force/--no-force", default=False, show_default=True)
def data_download_open_interest_vision(
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    start_date,
    end_date,
    data_dir,
    parquet_dir,
    sleep_sec,
    progress_every,
    force,
):
    """Download Binance Vision daily metrics → monthly OI Parquet (5m, e.g. HYPE)."""
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
        "--start-date",
        str(start_date),
        "--sleep-sec",
        str(sleep_sec),
        "--progress-every",
        str(progress_every),
    ]
    if end_date:
        args.extend(["--end-date", str(end_date)])
    if force:
        args.append("--force")

    sys.exit(run_script("src/data_tools/download_open_interest_vision.py", args))


@data.command("download-book-depth")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.option(
    "--universe-config",
    default=None,
    help="YAML universe config (if set, overrides --symbols).",
)
@click.option("--universe-set", default="starter_a")
@click.option("--universe-groups", default=None)
@click.option("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
@click.option("--end-date", default=None, help="YYYY-MM-DD inclusive (default: today UTC)")
@click.option(
    "--data-dir",
    default="data/book_depth/zip",
    help="Cache directory for daily bookDepth ZIP files",
)
@click.option(
    "--parquet-dir",
    default="data/book_depth/parquet",
    help="Output directory for daily wall snapshot Parquet",
)
@click.option("--sleep-sec", type=float, default=0.2, show_default=True)
@click.option("--progress-every", type=int, default=25, show_default=True)
@click.option("--force/--no-force", default=False, show_default=True)
def data_download_book_depth(
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    start_date,
    end_date,
    data_dir,
    parquet_dir,
    sleep_sec,
    progress_every,
    force,
):
    """Download Binance Vision daily bookDepth → wall snapshot Parquet (T5α history)."""
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
        "--start-date",
        str(start_date),
        "--sleep-sec",
        str(sleep_sec),
        "--progress-every",
        str(progress_every),
    ]
    if end_date:
        args.extend(["--end-date", str(end_date)])
    if force:
        args.append("--force")

    sys.exit(run_script("src/data_tools/download_book_depth_vision.py", args))


@data.command("download-depth-snapshots")
@click.option(
    "--symbols", "-s", default="BTCUSDT", help="Comma-separated symbols"
)
@click.option("--poll-count", type=int, default=1, show_default=True)
@click.option("--poll-interval-sec", type=float, default=60.0, show_default=True)
@click.option(
    "--parquet-dir",
    default="data/orderbook/parquet",
    help="Output directory for REST depth snapshot Parquet",
)
@click.option("--bucket-pct", type=float, default=0.005, show_default=True)
def data_download_depth_snapshots(
    symbols, poll_count, poll_interval_sec, parquet_dir, bucket_pct
):
    """Poll live REST depth snapshots (incremental; Vision has historical bookDepth)."""
    args = [
        "--parquet-dir",
        parquet_dir,
        "--symbols",
        *[s for s in symbols.split(",") if s.strip()],
        "--poll-count",
        str(poll_count),
        "--poll-interval-sec",
        str(poll_interval_sec),
        "--bucket-pct",
        str(bucket_pct),
    ]
    sys.exit(run_script("src/data_tools/download_depth_snapshots.py", args))


@data.command("update-market-cap")
@click.option(
    "--config", default="config/market_cap/market_cap.yaml", show_default=True
)
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    "--pattern",
    default=None,
    help="Optional ZIP glob pattern to convert a subset (example: BNBUSDT-aggTrades-2024-*.zip).",
)
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated list of symbols to convert (e.g., BTCUSDT,ETHUSDT,BNBUSDT). "
    "If not specified, all matching files will be converted.",
)
@click.option(
    "--input-dir", default=None, help="ZIP input directory (default: data/agg_data)"
)
@click.option(
    "--output-dir",
    default=None,
    help="Parquet output directory (default: data/parquet_data)",
)
@click.option("--force/--no-force", default=False, show_default=True)
@click.option(
    "--aggregate-freq",
    default="1min",
    help="Aggregation frequency for tick data (default: 1min). "
    "Examples: '1s' (1 second), '1T' (1 minute), '5T' (5 minutes). "
    "Uses pandas resample frequency strings.",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def data_convert(
    pattern, symbols, input_dir, output_dir, force, aggregate_freq, docker
):
    """Convert downloaded ZIPs to Parquet format (preserves source ZIPs)."""
    code = _data_convert_impl(
        pattern=pattern,
        symbols=symbols,
        input_dir=input_dir,
        output_dir=output_dir,
        force=force,
        aggregate_freq=aggregate_freq,
        docker=docker,
    )
    sys.exit(code)


@data.command("convert-1min")
@click.option(
    "--input-dir",
    default=None,
    help="Input parquet directory (default: data/parquet_data)",
)
@click.option(
    "--output-dir",
    default=None,
    help="Output parquet directory (default: data/parquet_data_1min)",
)
@click.option(
    "--pattern",
    default="*.parquet",
    help="File pattern to match (default: *.parquet)",
)
@click.option(
    "--symbol",
    default=None,
    help="Optional: convert only specific symbol (e.g., BTCUSDT)",
)
@click.option(
    "--force/--no-force", default=False, help="Force re-convert even if output exists"
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def data_convert_1min(input_dir, output_dir, pattern, symbol, force, docker):
    """Convert tick/1s parquet data to 1-minute aggregated orderflow parquet."""
    args = []
    if input_dir:
        args.extend(["--input-dir", input_dir])
    if output_dir:
        args.extend(["--output-dir", output_dir])
    if pattern:
        args.extend(["--pattern", pattern])
    if symbol:
        args.extend(["--symbol", symbol])
    if force:
        args.append("--force")
    code = run_python_module(
        "src.data_tools.convert_to_1min_orderflow", args, docker=docker
    )
    sys.exit(code)


@data.command("pipeline")
@click.option(
    "--symbols", "-s", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
        pattern=None,
        symbols=symbols,
        input_dir=None,
        output_dir=None,
        force=False,
        aggregate_freq=None,
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    # 解析 universe symbols 传给 convert
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
        convert_symbols = ",".join(resolved)
    else:
        convert_symbols = None
    code = _data_convert_impl(
        pattern=None,
        symbols=convert_symbols,
        input_dir=data_dir,
        output_dir=parquet_dir,
        force=False,
        aggregate_freq=None,
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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


@train.command("export-rules-to-readme")
@click.option(
    "--strategy-config",
    required=True,
    help="config/strategies/<strategy> directory",
)
@click.option(
    "--rules-md",
    default=None,
    help="Path to existing rules.md file (if not provided and --generate-rules, will generate)",
)
@click.option(
    "--generate-rules",
    is_flag=True,
    default=False,
    help="Generate rules first using export_tree_rules_imodels.py",
)
@click.option(
    "--features-yaml",
    default=None,
    help="Features YAML path (required if --generate-rules)",
)
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="240T", help="Timeframe")
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--test-size", type=float, default=0.3, help="Test set size")
@click.option("--max-rules", type=int, default=20, help="Maximum rules to export")
@click.option(
    "--min-support", type=float, default=0.01, help="Minimum support threshold"
)
@click.option(
    "--max-conditions", type=int, default=3, help="Maximum conditions per rule"
)
@click.option(
    "--max-rule-len", type=int, default=120, help="Maximum rule string length"
)
@click.option("--random-state", type=int, default=42, help="Random state")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def train_export_rules_to_readme(
    strategy_config,
    rules_md,
    generate_rules,
    features_yaml,
    symbol,
    timeframe,
    start_date,
    end_date,
    test_size,
    max_rules,
    min_support,
    max_conditions,
    max_rule_len,
    random_state,
    docker,
):
    """Export tree model rules to strategy README.md (using imodels or existing rules.md)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--strategy-config",
        f"/workspace/{strategy_config}" if use_workspace_prefix else strategy_config,
    ]
    if rules_md:
        args.extend(
            [
                "--rules-md",
                f"/workspace/{rules_md}" if use_workspace_prefix else rules_md,
            ]
        )
    if generate_rules:
        args.append("--generate-rules")
        if not features_yaml:
            raise click.BadParameter("--features-yaml required when --generate-rules")
        if not start_date or not end_date:
            raise click.BadParameter(
                "--start-date and --end-date required when --generate-rules"
            )
        args.extend(
            [
                "--features-yaml",
                (
                    f"/workspace/{features_yaml}"
                    if use_workspace_prefix
                    else features_yaml
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
                "--max-rules",
                str(max_rules),
                "--min-support",
                str(min_support),
                "--max-conditions",
                str(max_conditions),
                "--max-rule-len",
                str(max_rule_len),
                "--random-state",
                str(random_state),
            ]
        )
    sys.exit(
        run_script("scripts/export_strategy_rules_to_readme.py", args, docker=docker)
    )


@train.command("export-rules")
@click.option(
    "--model-dir",
    required=True,
    help="Directory containing model.pkl (e.g., results/train_final_xxx/bpc)",
)
@click.option(
    "--strategy",
    required=True,
    help="Strategy name (e.g., bpc)",
)
@click.option(
    "--max-splits",
    type=int,
    default=30,
    help="Maximum number of split conditions to export",
)
@click.option(
    "--generate-risk-gate/--no-generate-risk-gate",
    default=True,
    help="Generate risk_gate_draft.yaml from tree splits (default: enabled)",
)
@click.option(
    "--risk-gate-output",
    default=None,
    help="Output path for risk_gate_draft.yaml (default: <model-dir>/risk_gate_draft.yaml)",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def train_export_rules(
    model_dir,
    strategy,
    max_splits,
    generate_risk_gate,
    risk_gate_output,
    docker,
):
    """Export LightGBM tree rules from model.pkl to <model-dir>/<strategy>_tree_rules.md."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--model-dir",
        f"/workspace/{model_dir}" if use_workspace_prefix else model_dir,
        "--strategy",
        strategy,
        "--max-splits",
        str(max_splits),
    ]
    if generate_risk_gate:
        args.append("--generate-risk-gate")
    if risk_gate_output:
        args.extend(
            [
                "--risk-gate-output",
                (
                    f"/workspace/{risk_gate_output}"
                    if use_workspace_prefix
                    else risk_gate_output
                ),
            ]
        )
    sys.exit(
        run_script("scripts/export_lightgbm_rules_to_readme.py", args, docker=docker)
    )


@train.command("export-monthly")
@click.option(
    "--results-dir",
    required=True,
    help="Base directory containing strategy results (e.g., results/fixed_long or results/fixed_short)",
)
@click.option(
    "--strategy",
    required=True,
    help="Strategy name (e.g., bpc)",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def train_export_monthly(
    results_dir,
    strategy,
    docker,
):
    """Export monthly OOS results to strategy README.md."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--results-dir",
        f"/workspace/{results_dir}" if use_workspace_prefix else results_dir,
        "--strategy",
        strategy,
    ]
    sys.exit(
        run_script("scripts/export_monthly_results_to_readme.py", args, docker=docker)
    )


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
@click.option(
    "--force-rebuild",
    is_flag=True,
    default=False,
    help="Delete existing layer data and rebuild from scratch. Without this flag, existing months are skipped.",
)
@click.option(
    "--features-yaml",
    default=None,
    help="Feature manifest (e.g. config/strategies/_shared/features_all.yaml for ~940 cols).",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    force_rebuild,
    features_yaml,
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
    if force_rebuild:
        args.append("--force-rebuild")
    if features_yaml:
        args.extend(
            [
                "--features-yaml",
                (
                    f"/workspace/{features_yaml}"
                    if use_workspace_prefix
                    else features_yaml
                ),
            ]
        )
    sys.exit(
        run_script("scripts/build_feature_store_from_config.py", args, docker=docker)
    )


@cli.group()
def rule():
    """Rule router commands (3-action: NO_TRADE/MEAN/TREND)."""
    pass


@cli.group()
def rl():
    """BC/RL research tooling (shadow/BC/RL/FSM). Mainline eval lives in nnmultihead."""
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
    default="bpc,me,fer,lv",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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


@rl.command("build-execution-logs")
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
    help="Optional mode file/dir (deprecated, not used)",
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
    "--calibration-json",
    default=None,
    help="Optional router calibration JSON (dir_prob + linear bias for heads).",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rl_build_execution_logs(
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
    click.echo(
        "Note: `mlbot rl build-execution-logs` is deprecated for v0 mainline. "
        "Use `mlbot nnmultihead build-execution-logs` instead."
    )
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

    sys.exit(run_script("scripts/rl_build_execution_logs.py", args, docker=docker))


@rule.command("plot-router-modes-kline")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="logs_3action or physics_regime parquet (must contain regime column)",
)
@click.option(
    "--feature-store-root",
    default="feature_store",
    show_default=True,
    help="FeatureStore root dir",
)
@click.option(
    "--feature-store-layer",
    required=True,
    help="FeatureStore layer id (used to load OHLC)",
)
@click.option("--symbol", default=None, help="Single symbol to plot")
@click.option(
    "--all-symbols",
    is_flag=True,
    default=False,
    help="Plot all symbols found in mode_3action",
)
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--out", "out_path", required=True, help="Output PNG or output dir")
@click.option(
    "--gate-only",
    type=click.Choice(["all", "allow", "veto"]),
    default="all",
    show_default=True,
    help="If gate_decision exists, filter plotted points.",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_plot_router_modes_kline(
    mode_path,
    feature_store_root,
    feature_store_layer,
    symbol,
    all_symbols,
    start_date,
    end_date,
    out_path,
    gate_only,
    docker,
):
    """Plot router modes on OHLC close series."""
    use_workspace_prefix = docker
    args = [
        "--mode",
        f"/workspace/{mode_path}" if use_workspace_prefix else mode_path,
        "--feature-store-root",
        (
            f"/workspace/{feature_store_root}"
            if use_workspace_prefix
            else feature_store_root
        ),
        "--feature-store-layer",
        str(feature_store_layer),
        "--out",
        f"/workspace/{out_path}" if use_workspace_prefix else out_path,
        "--gate-only",
        str(gate_only),
    ]
    if all_symbols:
        args.append("--all-symbols")
    if symbol:
        args.extend(["--symbol", str(symbol)])
    if start_date:
        args.extend(["--start-date", str(start_date)])
    if end_date:
        args.extend(["--end-date", str(end_date)])
    sys.exit(run_script("scripts/plot_router_modes_kline.py", args, docker=docker))


@rule.command("physics-regime")
@click.option(
    "--preds",
    required=True,
    help="Preds file (.parquet/.csv) or directory of per-symbol preds_*.parquet",
)
@click.option(
    "--feature-store-root",
    default=None,
    help="Optional FeatureStore root (if preds lack required features).",
)
@click.option("--layer", default="tier0", show_default=True, help="FeatureStore layer.")
@click.option(
    "--timeframe", default="240T", show_default=True, help="Timeframe (e.g., 240T)."
)
@click.option("--output", "output_path", required=True, help="Output parquet/csv path.")
@click.option(
    "--stats-output",
    default=None,
    help="Optional JSON output for Symbol × Regime frequency stats.",
)
@click.option(
    "--scan-physics-score-pct",
    default=None,
    help="Comma-separated physics_score_min_pct values for scan (e.g., 0.8,0.85,0.9).",
)
@click.option(
    "--scan-output",
    default=None,
    help="JSON output path for scan report.",
)
@click.option(
    "--scan-md-output",
    default=None,
    help="Markdown output path for scan summary.",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_physics_regime(
    preds,
    feature_store_root,
    layer,
    timeframe,
    output_path,
    stats_output,
    scan_physics_score_pct,
    scan_output,
    scan_md_output,
    docker,
):
    """[DEPRECATED] Classify Physics/Regimes and (optionally) scan physics_score_min_pct.

    ⚠️ DEPRECATED: Regime classification has been migrated to gate rules in execution_archetypes.yaml.
    Physical features are now computed in FeatureStore and checked directly by gate rules.
    This command is kept for backward compatibility and diagnostics only.
    """
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--preds",
        f"/workspace/{preds}" if use_workspace_prefix else preds,
        "--output",
        f"/workspace/{output_path}" if use_workspace_prefix else output_path,
    ]
    if feature_store_root:
        args.extend(
            [
                "--feature-store-root",
                (
                    f"/workspace/{feature_store_root}"
                    if use_workspace_prefix
                    else feature_store_root
                ),
            ]
        )
    if layer:
        args.extend(["--layer", str(layer)])
    if timeframe:
        args.extend(["--timeframe", str(timeframe)])
    if stats_output:
        args.extend(
            [
                "--stats-output",
                f"/workspace/{stats_output}" if use_workspace_prefix else stats_output,
            ]
        )
    if scan_physics_score_pct:
        args.extend(["--scan-physics-score-pct", str(scan_physics_score_pct)])
    if scan_output:
        args.extend(
            [
                "--scan-output",
                f"/workspace/{scan_output}" if use_workspace_prefix else scan_output,
            ]
        )
    if scan_md_output:
        args.extend(
            [
                "--scan-md-output",
                (
                    f"/workspace/{scan_md_output}"
                    if use_workspace_prefix
                    else scan_md_output
                ),
            ]
        )
    sys.exit(run_script("scripts/physics_regime_classifier.py", args, docker=docker))


@rule.command("diagnose-tc-regime-execution")
@click.option("--logs", "logs_path", required=True, help="logs_3action.parquet")
@click.option("--regime", "regime_path", required=True, help="physics_regime parquet")
@click.option("--output-json", required=True, help="Output JSON path")
@click.option("--output-md", required=True, help="Output Markdown path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_diagnose_tc_regime_execution(
    logs_path,
    regime_path,
    output_json,
    output_md,
    docker,
):
    """Diagnose execution KPIs within TC_REGIME subset only."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--regime",
        f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
        "--output-json",
        f"/workspace/{output_json}" if use_workspace_prefix else output_json,
        "--output-md",
        f"/workspace/{output_md}" if use_workspace_prefix else output_md,
    ]
    sys.exit(run_script("scripts/diagnose_tc_world_execution.py", args, docker=docker))


@rule.command("diagnose-fr-et-filtering")
@click.option("--preds", required=True, help="Predictions parquet directory or file")
@click.option("--output-md", default=None, help="Output Markdown report path")
@click.option(
    "--relax-router", is_flag=True, help="Relax router thresholds for testing"
)
@click.option(
    "--relax-regime", is_flag=True, help="Relax regime classification for testing"
)
def rule_diagnose_fr_et_filtering(
    preds,
    output_md,
    relax_router,
    relax_regime,
    docker,
):
    """Diagnose FR/ET filtering at each layer."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--preds",
        f"/workspace/{preds}" if use_workspace_prefix else preds,
    ]
    if output_md:
        args.extend(
            [
                "--output-md",
                f"/workspace/{output_md}" if use_workspace_prefix else output_md,
            ]
        )
    if relax_router:
        args.append("--relax-router")
    if relax_regime:
        args.append("--relax-regime")
    sys.exit(run_script("scripts/diagnose_fr_et_filtering.py", args, docker=docker))


@rule.command("diagnose-e2e-kpi")
@click.option("--logs", "logs_path", required=True, help="logs_3action.parquet")
@click.option(
    "--regime", "regime_path", default=None, help="physics_regime parquet (optional)"
)
@click.option(
    "--gate",
    "gate_path",
    default=None,
    help="gate output parquet with archetype info (optional)",
)
@click.option(
    "--output-json",
    default=None,
    help="Output JSON path (default: results/e2e_kpi/e2e_kpi_report.json)",
)
@click.option(
    "--output-md",
    default=None,
    help="Output Markdown path (default: results/e2e_kpi/e2e_kpi_report.md)",
)
@click.option(
    "--no-regime-filter",
    is_flag=True,
    help="Generate comparison report without regime filtering",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_diagnose_e2e_kpi(
    logs_path,
    regime_path,
    gate_path,
    output_json,
    output_md,
    no_regime_filter,
    docker,
):
    """E2E KPI diagnostics for Router/Gate/Execution."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
    ]
    if output_json:
        args.extend(
            [
                "--output-json",
                f"/workspace/{output_json}" if use_workspace_prefix else output_json,
            ]
        )
    if output_md:
        args.extend(
            [
                "--output-md",
                f"/workspace/{output_md}" if use_workspace_prefix else output_md,
            ]
        )
    if regime_path:
        args.extend(
            [
                "--regime",
                f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
            ]
        )
    if gate_path:
        args.extend(
            [
                "--gate",
                f"/workspace/{gate_path}" if use_workspace_prefix else gate_path,
            ]
        )
    if no_regime_filter:
        args.append("--no-regime-filter")
    sys.exit(run_script("scripts/diagnose_e2e_kpi.py", args, docker=docker))


@rule.command("diagnose-gate-label-loosen")
@click.option(
    "--labels",
    "labels_path",
    required=True,
    help="labeled parquet from generate_fbf_labels.py",
)
@click.option(
    "--config",
    "config_path",
    default="config/nnmultihead/execution_archetypes.yaml",
    show_default=True,
    help="execution_archetypes.yaml path",
)
@click.option("--archetype", required=True, help="archetype name")
@click.option("--quantiles", default=None, help="evidence_quantiles.json path")
@click.option("--feature-store-root", default=None, help="FeatureStore root")
@click.option("--feature-store-layer", default=None, help="FeatureStore layer")
@click.option("--timeframe", default=None, help="timeframe for FeatureStore")
@click.option(
    "--label-col", default="fbf_label", show_default=True, help="label column"
)
@click.option("--label-value", default="FBF", show_default=True, help="label value")
@click.option(
    "--target-fbf-pass",
    default=0.6,
    show_default=True,
    help="target pass rate on labels",
)
@click.option(
    "--target-fbf-veto",
    default=0.1,
    show_default=True,
    help="target veto rate on labels",
)
@click.option(
    "--min-samples",
    default=200,
    show_default=True,
    help="min samples to suggest thresholds",
)
@click.option("--out", "out_path", required=True, help="output json path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_diagnose_gate_label_loosen(
    labels_path,
    config_path,
    archetype,
    quantiles,
    feature_store_root,
    feature_store_layer,
    timeframe,
    label_col,
    label_value,
    target_fbf_pass,
    target_fbf_veto,
    min_samples,
    out_path,
    docker,
):
    """Diagnose which rules veto labels and suggest looser thresholds."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--labels",
        f"/workspace/{labels_path}" if use_workspace_prefix else labels_path,
        "--config",
        f"/workspace/{config_path}" if use_workspace_prefix else config_path,
        "--archetype",
        archetype,
        "--label-col",
        label_col,
        "--label-value",
        label_value,
        "--target-fbf-pass",
        str(target_fbf_pass),
        "--target-fbf-veto",
        str(target_fbf_veto),
        "--min-samples",
        str(min_samples),
        "--out",
        f"/workspace/{out_path}" if use_workspace_prefix else out_path,
    ]
    if quantiles:
        args.extend(
            [
                "--quantiles",
                f"/workspace/{quantiles}" if use_workspace_prefix else quantiles,
            ]
        )
    if feature_store_root:
        args.extend(["--feature-store-root", feature_store_root])
    if feature_store_layer:
        args.extend(["--feature-store-layer", feature_store_layer])
    if timeframe:
        args.extend(["--timeframe", timeframe])
    sys.exit(run_script("scripts/diagnose_gate_label_loosen.py", args, docker=docker))


@rule.command("diagnose-gate-application")
@click.option("--logs", "logs_path", required=True, help="logs parquet")
@click.option(
    "--execution-archetypes",
    "config_path",
    default="config/nnmultihead/execution_archetypes.yaml",
    show_default=True,
    help="execution_archetypes.yaml path",
)
@click.option("--output", "out_path", required=True, help="output json path")
@click.option("--feature-store-root", default="feature_store", show_default=True)
@click.option("--feature-store-layer", default=None)
@click.option("--evidence-quantiles", default=None)
@click.option("--timeframe", default="240T", show_default=True)
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_diagnose_gate_application(
    logs_path,
    config_path,
    out_path,
    feature_store_root,
    feature_store_layer,
    evidence_quantiles,
    timeframe,
    start_date,
    end_date,
    docker,
):
    """Diagnose gate application and veto sources."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--execution-archetypes",
        f"/workspace/{config_path}" if use_workspace_prefix else config_path,
        "--output",
        f"/workspace/{out_path}" if use_workspace_prefix else out_path,
        "--feature-store-root",
        feature_store_root,
        "--timeframe",
        timeframe,
    ]
    if feature_store_layer:
        args.extend(["--feature-store-layer", feature_store_layer])
    if evidence_quantiles:
        args.extend(
            [
                "--evidence-quantiles",
                (
                    f"/workspace/{evidence_quantiles}"
                    if use_workspace_prefix
                    else evidence_quantiles
                ),
            ]
        )
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    sys.exit(run_script("scripts/diagnose_gate_application.py", args, docker=docker))


@rule.command("build-evidence-quantiles")
@click.option("--feature-store-root", default="feature_store", show_default=True)
@click.option("--layer", required=True, help="FeatureStore layer name")
@click.option("--symbols", required=True, help="comma-separated symbols")
@click.option("--timeframe", default="240T", show_default=True)
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
@click.option("--keys", required=True, help="comma-separated feature keys")
@click.option(
    "--quantiles",
    default="0.0,0.05,0.2,0.25,0.3,0.4,0.5,0.55,0.6,0.7,0.75,0.85,0.9,0.95,1.0",
    show_default=True,
)
@click.option("--out", "out_path", required=True, help="output json path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_build_evidence_quantiles(
    feature_store_root,
    layer,
    symbols,
    timeframe,
    start_date,
    end_date,
    keys,
    quantiles,
    out_path,
    docker,
):
    """Build evidence quantiles for quantile_* gate rules."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--feature-store-root",
        feature_store_root,
        "--layer",
        layer,
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--keys",
        keys,
        "--quantiles",
        quantiles,
        "--out",
        f"/workspace/{out_path}" if use_workspace_prefix else out_path,
    ]
    sys.exit(run_script("scripts/build_evidence_quantiles.py", args, docker=docker))


@rule.command("extract-evidence-keys")
@click.option(
    "--config",
    "config_path",
    default="config/nnmultihead/execution_archetypes.yaml",
    show_default=True,
    help="execution_archetypes.yaml path",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_extract_evidence_keys(config_path, docker):
    """Extract quantile_* evidence keys from when_then_rules."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--config",
        f"/workspace/{config_path}" if use_workspace_prefix else config_path,
    ]
    sys.exit(run_script("scripts/extract_evidence_keys.py", args, docker=docker))


@rule.command("apply-tree-gate")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="logs_3action or physics_regime parquet (must contain regime column)",
)
@click.option("--out", "out_path", required=True, help="output gated mode file")
@click.option(
    "--db-path",
    "db_path",
    default="data/order_management.db",
    show_default=True,
    help="Order management DB path",
)
@click.option(
    "--features-store-root",
    default="feature_store",
    show_default=True,
    help="FeatureStore root dir",
)
@click.option(
    "--features-store-layer",
    "features_store_layer",
    required=True,
    help="FeatureStore layer (required)",
)
@click.option(
    "--timeframe",
    default="240T",
    show_default=True,
    help="Timeframe (e.g., 240T)",
)
@click.option(
    "--start-date",
    default=None,
    help="Start date (YYYY-MM-DD, optional)",
)
@click.option(
    "--end-date",
    default=None,
    help="End date (YYYY-MM-DD, optional)",
)
@click.option(
    "--physics-regime",
    "physics_regime_path",
    default=None,
    help="physics_regime parquet file (adds tc/te semantic scores)",
)
@click.option(
    "--semantic-score-floors",
    default=None,
    help="JSON file with semantic score floors (optional)",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_apply_tree_gate(
    logs_path,
    out_path,
    db_path,
    features_store_root,
    features_store_layer,
    timeframe,
    start_date,
    end_date,
    physics_regime_path,
    semantic_score_floors,
    docker,
):
    """Apply tree gate based on regime and archetype (Gate filtering step in pipeline)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--out",
        f"/workspace/{out_path}" if use_workspace_prefix else out_path,
        "--db-path",
        f"/workspace/{db_path}" if use_workspace_prefix else db_path,
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
    ]
    if start_date:
        args.extend(["--start-date", str(start_date)])
    if end_date:
        args.extend(["--end-date", str(end_date)])
    if physics_regime_path:
        args.extend(
            [
                "--physics-regime",
                (
                    f"/workspace/{physics_regime_path}"
                    if use_workspace_prefix
                    else physics_regime_path
                ),
            ]
        )
    if semantic_score_floors:
        args.extend(
            [
                "--semantic-score-floors",
                (
                    f"/workspace/{semantic_score_floors}"
                    if use_workspace_prefix
                    else semantic_score_floors
                ),
            ]
        )
    sys.exit(run_script("scripts/apply_archetype_gate.py", args, docker=docker))


@rule.command("diagnose-gate-filtering")
@click.option("--logs", "logs_path", required=True, help="logs_3action.parquet")
@click.option("--regime", "regime_path", required=True, help="physics_regime parquet")
@click.option(
    "--db-path",
    "db_path",
    required=True,
    help="Order management DB path",
)
@click.option("--output-md", required=True, help="Output Markdown path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_diagnose_gate_filtering(
    logs_path,
    regime_path,
    db_path,
    output_md,
    docker,
):
    """Diagnose Gate filtering effects on trades."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--regime",
        f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
        "--db-path",
        f"/workspace/{db_path}" if use_workspace_prefix else db_path,
        "--output-md",
        f"/workspace/{output_md}" if use_workspace_prefix else output_md,
    ]
    sys.exit(run_script("scripts/diagnose_gate_filtering.py", args, docker=docker))


@rule.command("diagnose-e2e-symbol-regime-archetype")
@click.option("--logs", "logs_path", required=True, help="logs_3action.parquet")
@click.option("--regime", "regime_path", required=True, help="physics_regime parquet")
@click.option("--output-json", required=True, help="Output JSON path")
@click.option("--output-md", required=True, help="Output Markdown path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_diagnose_e2e_symbol_regime_archetype(
    logs_path,
    regime_path,
    output_json,
    output_md,
    docker,
):
    """E2E KPI by Symbol × Regime × Archetype (sharpe, trade_count, win_rate, profit_loss_ratio)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--regime",
        f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
        "--output-json",
        f"/workspace/{output_json}" if use_workspace_prefix else output_json,
        "--output-md",
        f"/workspace/{output_md}" if use_workspace_prefix else output_md,
    ]
    sys.exit(
        run_script(
            "scripts/diagnose_e2e_symbol_regime_archetype.py", args, docker=docker
        )
    )


@rule.command("optimize-gate-plateau")
@click.option(
    "--gated-logs",
    required=True,
    help="Gated logs parquet file (from apply-tree-gate)",
)
@click.option(
    "--raw-logs",
    default=None,
    help="Raw logs parquet file (optional, for re-applying gate rules)",
)
@click.option(
    "--execution-archetypes",
    default="config/nnmultihead/execution_archetypes.yaml",
    show_default=True,
    help="execution_archetypes.yaml path",
)
@click.option(
    "--output",
    default="results/gate_optimization.json",
    show_default=True,
    help="Output JSON path",
)
@click.option(
    "--min-trade-rate",
    type=float,
    default=0.005,
    show_default=True,
    help="Minimum trade rate constraint",
)
@click.option(
    "--min-sharpe-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum Sharpe for plateau detection",
)
@click.option(
    "--threshold-step",
    type=float,
    default=0.05,
    show_default=True,
    help="Threshold scan step size",
)
@click.option(
    "--archetype-filter",
    default=None,
    help="Filter to specific archetype (e.g., BPC, HTF)",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_optimize_gate_plateau(
    gated_logs,
    raw_logs,
    execution_archetypes,
    output,
    min_trade_rate,
    min_sharpe_threshold,
    threshold_step,
    archetype_filter,
    docker,
):
    """
    Optimize Gate rule thresholds using plateau search (backup tool).

    DEPRECATED: scripts/optimize_gate_plateau.py removed.
    Use: mlbot research plateau ... OR python scripts/optimize_gate_unified.py
    """
    click.echo(
        "DEPRECATED: mlbot rule optimize-gate-plateau → use "
        "'mlbot research plateau' or 'scripts/optimize_gate_unified.py'",
        err=True,
    )
    sys.exit(2)
    use_workspace_prefix = docker and not _is_in_docker()
    script = "scripts/optimize_gate_plateau.py"

    args = [
        "--gated-logs",
        f"/workspace/{gated_logs}" if use_workspace_prefix else gated_logs,
        "--execution-archetypes",
        (
            f"/workspace/{execution_archetypes}"
            if use_workspace_prefix
            else execution_archetypes
        ),
        "--output",
        f"/workspace/{output}" if use_workspace_prefix else output,
        "--min-trade-rate",
        str(min_trade_rate),
        "--min-sharpe-threshold",
        str(min_sharpe_threshold),
        "--threshold-step",
        str(threshold_step),
    ]

    if raw_logs:
        args.extend(
            [
                "--raw-logs",
                f"/workspace/{raw_logs}" if use_workspace_prefix else raw_logs,
            ]
        )

    if archetype_filter:
        args.extend(["--archetype-filter", archetype_filter])

    sys.exit(run_script(script, args, docker=docker))


@cli.group("experiment")
def experiment():
    """Experiment commands for testing architecture changes."""
    pass


# =============================================================================
# Pipeline commands (mlbot pipeline run/list/adopt/delete/diff)
# =============================================================================


@cli.group()
def pipeline():
    """研究管线: 训练、管理、对比实验."""
    pass


@pipeline.command()
@click.option("--strategy", help="策略名 (如 bpc-long, fer-short, me)")
@click.option("--all", "run_all", is_flag=True, help="执行所有策略")
@click.option("--end-date", help="数据截止日期 (默认自动检测)")
@click.option("--dry-run", is_flag=True, help="打印命令但不执行")
@click.option("--no-adopt", is_flag=True, help="禁止自动采纳, 仅保存实验结果")
@click.option(
    "--skip-shap",
    is_flag=True,
    help="跳过 SHAP 特征筛选；Gate 训练导出亦跳过 TreeSHAP/交互（gain-only，快速迭代）",
)
@click.option("--compare-only", is_flag=True, help="只对比, 不重训")
@click.option("--use-1min", is_flag=True, help="使用 1min bar 精细模拟")
@click.option("--live-root", default="live/highcap", help="1min bar 数据根目录")
@click.option("--config", "config_path", default=None, help="pipeline 配置文件路径")
@click.option(
    "--stage",
    type=click.Choice(
        [
            "full",
            "prefilter",
            "gate",
            "entry_filter",
            "slow_snapshot",
            "execution_opt",
            "event_backtest",
            "fast_month",
            "rolling_sim",
            "grid_backtest",
            "dual_add_backtest",
            "pcm_joint",
            "pcm_slot_grid",
        ]
    ),
    default="full",
    show_default=True,
    help=(
        "运行阶段: full/prefilter/gate/entry_filter/slow_snapshot/"
        "execution_opt/event_backtest/fast_month/rolling_sim/grid_backtest/"
        "dual_add_backtest/pcm_joint/pcm_slot_grid"
    ),
)
@click.option(
    "--month",
    default=None,
    help="月份 YYYY-MM，fast_month 必填；多个月用逗号/空格分隔，如 2024-07,2024-08",
)
@click.option(
    "--event-backtest",
    is_flag=True,
    help="训练后运行事件回测 execution 优化 (sym-r grid search + 交易地图)",
)
@click.option(
    "--event-sym-r",
    default="1.0:0.5:4.0",
    help="事件回测 execution 优化 sym-r 范围 (default: 1.0:0.5:4.0)",
)
def run(
    strategy,
    run_all,
    end_date,
    dry_run,
    no_adopt,
    skip_shap,
    compare_only,
    use_1min,
    live_root,
    config_path,
    stage,
    month,
    event_backtest,
    event_sym_r,
):
    """执行研究管线."""
    args = []
    if strategy:
        args.extend(["--strategy", strategy])
    if run_all:
        args.append("--all")
    if end_date:
        args.extend(["--end-date", end_date])
    if dry_run:
        args.append("--dry-run")
    if no_adopt:
        args.append("--no-adopt")
    if skip_shap:
        args.append("--skip-shap")
    if compare_only:
        args.append("--compare-only")
    if use_1min:
        args.append("--use-1min")
    if live_root != "live/highcap":
        args.extend(["--live-root", live_root])
    if config_path:
        args.extend(["--config", config_path])
    if stage != "full":
        args.extend(["--stage", stage])
    if month:
        args.extend(["--month", month])
    if event_backtest:
        args.append("--event-backtest")
    if event_sym_r != "1.0:0.5:4.0":
        args.extend(["--event-sym-r", event_sym_r])

    sys.exit(run_script("scripts/auto_research_pipeline.py", args))


@pipeline.command("list")
@click.option("--strategy", help="策略名")
@click.option("--all", "list_all", is_flag=True, help="列出所有策略")
@click.option(
    "--include-bad-candidates",
    is_flag=True,
    help="与 --all 联用：同时列出 config/strategies/bad-candidates/<pkg>/ 下各包",
)
@click.option(
    "--list-all-profiles",
    "list_all_profiles",
    is_flag=True,
    help="与 --all 且未指定 --config 联用：turbo/slow/pipeline 各入口各列一遍",
)
@click.option("--config", "config_path", default=None, help="pipeline 配置文件路径")
def pipeline_list(
    strategy, list_all, include_bad_candidates, list_all_profiles, config_path
):
    """列出历史实验及其 metrics."""
    args = ["--list"]
    if strategy:
        args.extend(["--strategy", strategy])
    if list_all:
        args.append("--all")
    if include_bad_candidates:
        args.append("--include-bad-candidates")
    if list_all_profiles:
        args.append("--list-all-profiles")
    if config_path:
        args.extend(["--config", config_path])

    sys.exit(run_script("scripts/auto_research_pipeline.py", args))


@pipeline.command()
@click.option("--strategy", required=True, help="策略名")
@click.argument("timestamp")
@click.option("--config", "config_path", default=None, help="pipeline 配置文件路径")
def adopt(strategy, timestamp, config_path):
    """手动采纳指定时间戳的实验."""
    args = ["--strategy", strategy, "--adopt", timestamp]
    if config_path:
        args.extend(["--config", config_path])

    sys.exit(run_script("scripts/auto_research_pipeline.py", args))


@pipeline.command()
@click.option("--strategy", required=True, help="策略名")
@click.argument("ts1")
@click.argument("ts2")
@click.option("--config", "config_path", default=None, help="pipeline 配置文件路径")
def diff(strategy, ts1, ts2, config_path):
    """对比两次实验的 archetypes 差异."""
    args = ["--strategy", strategy, "--diff", ts1, ts2]
    if config_path:
        args.extend(["--config", config_path])

    sys.exit(run_script("scripts/auto_research_pipeline.py", args))


@pipeline.command("deploy")
@click.option("--diff", "show_diff", is_flag=True, help="只查看研究仓与 live/highcap 差异")
@click.option("--deploy", "do_deploy", is_flag=True, help="执行部署 (对比 + 复制)")
@click.option("--rollback", is_flag=True, help="显示 live/highcap 回滚指引")
@click.option("--strategy", multiple=True, help="指定策略 (可重复；默认脚本内置策略集)")
@click.option("--yes", "-y", is_flag=True, help="非交互模式, 跳过部署确认")
@click.option("--git-commit", is_flag=True, help="部署后自动 git commit live/ 变更")
def pipeline_deploy(show_diff, do_deploy, rollback, strategy, yes, git_commit):
    """研究仓配置同步到 live/highcap."""
    args = []
    if show_diff:
        args.append("--diff")
    if do_deploy:
        args.append("--deploy")
    if rollback:
        args.append("--rollback")
    if strategy:
        args.append("--strategy")
        args.extend(strategy)
    if yes:
        args.append("--yes")
    if git_commit:
        args.append("--git-commit")

    sys.exit(run_script("scripts/deploy_config_to_live.py", args))


@pipeline.command()
@click.option("--strategy", required=True, help="策略名")
@click.option("--timestamp", multiple=True, help="指定时间戳 (可多次)")
@click.option("--status", help="按状态筛选 (error/keep/adopt/alert)")
@click.option("--all", "delete_all", is_flag=True, help="删除全部历史实验")
@click.option("--dry-run", is_flag=True, help="预览要删除的实验")
@click.option("--date-range", nargs=2, help="日期范围 (YYYY-MM-DD YYYY-MM-DD)")
@click.option(
    "--config",
    "config_path",
    default=None,
    help="pipeline YAML；用于 output.history_dir（与 list/adopt 一致，否则按默认 research 解析）",
)
def delete(strategy, timestamp, status, delete_all, dry_run, date_range, config_path):
    """删除实验 (集成 cleanup_old_experiments 功能)."""
    # Import core functions from cleanup script
    _scripts_dir = str(PROJECT_ROOT / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    try:
        from cleanup_old_experiments import (
            get_experiment_dirs,
            get_experiment_status,
            should_delete_based_on_date,
        )
    except ImportError:
        click.echo("❌ 无法导入 cleanup_old_experiments 模块")
        sys.exit(1)

    from pathlib import Path as _Path

    from scripts.pipeline import config as pipeline_config
    from src.config.strategy_layout import resolve_default_pipeline_config

    import shutil

    if config_path:
        cfg_p = _Path(config_path).resolve()
    else:
        cfg_p, _warns = resolve_default_pipeline_config(
            PROJECT_ROOT, str(strategy).strip(), None
        )
    cfg = pipeline_config.load_pipeline_config(cfg_p)
    history_root = (PROJECT_ROOT / cfg["output"]["history_dir"]).resolve()
    try:
        rel_cfg = cfg_p.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_cfg = cfg_p
    click.echo(f"配置: {rel_cfg}")
    click.echo(f"实验子目录根: {history_root / strategy}")
    click.echo(f"（另会尝试遗留路径 results/research_history/{strategy}/）")

    experiment_dirs = get_experiment_dirs(
        strategy, history_root=history_root, project_root=PROJECT_ROOT
    )
    if not experiment_dirs:
        click.echo(f"没有找到 {strategy} 策略的实验目录")
        return

    dirs_to_delete = []

    if delete_all:
        dirs_to_delete = experiment_dirs
    elif timestamp:
        for ts in timestamp:
            for exp_dir in experiment_dirs:
                if exp_dir.name == ts:
                    dirs_to_delete.append(exp_dir)
                    break
            else:
                click.echo(f"警告: 找不到时间戳为 {ts} 的实验")
    elif status:
        for exp_dir in experiment_dirs:
            exp_status = get_experiment_status(exp_dir)
            if exp_status == status.lower():
                dirs_to_delete.append(exp_dir)
    elif date_range:
        for exp_dir in experiment_dirs:
            if should_delete_based_on_date(exp_dir, date_range[0], date_range[1]):
                dirs_to_delete.append(exp_dir)
    else:
        click.echo("请指定: --timestamp, --all, --status, 或 --date-range")
        return

    if not dirs_to_delete:
        click.echo("没有找到匹配条件的实验")
        return

    click.echo(f"将要删除 {len(dirs_to_delete)} 个实验:")
    for exp_dir in dirs_to_delete:
        exp_status = get_experiment_status(exp_dir)
        click.echo(f"  - {exp_dir.name} (状态: {exp_status})")

    if dry_run:
        click.echo("\n这是 dry-run 模式，实际文件不会被删除")
        return

    if not click.confirm(f"\n确认删除这 {len(dirs_to_delete)} 个实验?"):
        click.echo("取消操作")
        return

    deleted_count = 0
    for exp_dir in dirs_to_delete:
        try:
            shutil.rmtree(exp_dir)
            click.echo(f"已删除: {exp_dir}")
            deleted_count += 1
        except Exception as e:
            click.echo(f"删除失败 {exp_dir}: {e}")

    click.echo(f"\n完成: 成功删除 {deleted_count} 个实验目录")


@pipeline.command("event-backtest")
@click.option("--strategy", required=True, help="策略名 (如 me-long, bpc-short)")
@click.option(
    "--hash",
    "exp_hash",
    default=None,
    help="实验时闳戳 hash (如 20260313_234448), 默认使用最新实验",
)
@click.option(
    "--sym-r",
    "sym_r",
    default=None,
    help="开启 Execution 优化, 指定 sym-r 范围 (如 1.0:0.5:4.0)",
)
@click.option("--promote", is_flag=True, help="将优化结果写入实验 execution.yaml")
@click.option("--fast", is_flag=True, default=True, help="快速模式 (60T bar, 默认开启)")
@click.option("--no-fast", "fast", flag_value=False, help="使用 1min bar 精细模式")
@click.option("--data-path", default="data/parquet_data", help="研究数据目录")
@click.option("--config", "config_path", default=None, help="pipeline 配置文件路径")
@click.option(
    "--map-extra-months",
    default=None,
    type=int,
    help=(
        "交易地图 VWAP/EMA 向前多取的月数；默认读 pipeline 配置 event_backtest.map_extra_months，缺省 12"
    ),
)
def pipeline_event_backtest(
    strategy, exp_hash, sym_r, promote, fast, data_path, config_path, map_extra_months
):
    """对指定实验运行事件回测 + 可选 execution 优化, 输出交易地图到 research_history."""
    import json
    import subprocess

    # 加载 pipeline 配置读取 history_dir
    _cfg_path = config_path or "config/pipelines/pcm_orchestrate_2h.yaml"
    _cfg: dict = {}
    try:
        import yaml as _yaml

        _cfg = _yaml.safe_load(Path(_cfg_path).read_text(encoding="utf-8")) or {}
        history_dir = PROJECT_ROOT / _cfg["output"]["history_dir"]
    except Exception:
        history_dir = PROJECT_ROOT / "results/research_history"

    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        click.echo(f"❌ 找不到 {strategy} 策略的 research_history: {strat_dir}")
        sys.exit(1)

    # 确定实验目录
    if exp_hash:
        exp_dir = strat_dir / exp_hash
        if not exp_dir.exists():
            click.echo(f"❌ 找不到实验 {exp_hash}")
            sys.exit(1)
    else:
        runs = sorted(
            [
                d
                for d in strat_dir.iterdir()
                if d.is_dir() and (d / "report.json").exists()
            ],
            reverse=True,
        )
        if not runs:
            click.echo(f"❌ {strategy} 没有可用实验")
            sys.exit(1)
        exp_dir = runs[0]
        click.echo(f"📂 使用最新实验: {exp_dir.name}")

    # 读取 report.json 获取日期范围
    report = json.loads((exp_dir / "report.json").read_text(encoding="utf-8"))
    holdout_start = report["data_range"]["holdout_start"]
    end_date = report["data_range"]["end_date"]
    strategies_root = str(exp_dir / "strategies")
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)

    click.echo(f"📅 回测区间: {holdout_start} ~ {end_date}")
    click.echo(f"📁 实验目录: {exp_dir}")

    rc = 0

    # Step 1 (可选): Execution 参数优化
    if sym_r:
        click.echo(f"\n🔧 Step 1: Execution 优化 (sym-r={sym_r})")
        opt_args = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/optimize_event_execution.py"),
            "--strategy",
            strategy,
            "--start-date",
            holdout_start,
            "--end-date",
            end_date,
            "--sym-r",
            sym_r,
            "--strategies-root",
            strategies_root,
            "--data-path",
            data_path,
            "--output",
            str(results_dir / "event_exec_opt.json"),
        ]
        if promote:
            opt_args.append("--promote")
        rc_opt = subprocess.run(opt_args, cwd=str(PROJECT_ROOT)).returncode
        if rc_opt != 0:
            click.echo("⚠️  Execution 优化有异常, 继续使用当前配置")

    # Step 2: 事件回测 + 交易地图
    click.echo(f"\n🎯 Step 2: 事件回测 + 交易地图")
    map_path = str(results_dir / f"trading_map_{strategy}_event.html")
    export_path = str(results_dir / f"event_trades_{strategy}.csv")
    ev_args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/event_backtest.py"),
        "--strategy",
        strategy,
        "--start-date",
        holdout_start,
        "--end-date",
        end_date,
        "--strategies-root",
        strategies_root,
        "--data-path",
        data_path,
        "--trading-map",
        map_path,
        "--trades-csv",
        export_path,
    ]
    if fast:
        ev_args.append("--fast")
    _mem = map_extra_months
    if _mem is None:
        try:
            _mem = int((_cfg.get("event_backtest") or {}).get("map_extra_months", 12))
        except (TypeError, ValueError):
            _mem = 12
    if _mem >= 0:
        ev_args.extend(["--map-extra-months", str(_mem)])
    rc = subprocess.run(ev_args, cwd=str(PROJECT_ROOT)).returncode
    click.echo(f"\n🗺️  交易地图: {map_path}")
    click.echo(f"📄 交易明细: {export_path}")
    sys.exit(rc)


@pipeline.command("report-side-state")
@click.option("--run-id", required=True, help="rolling_sim run id (timestamp)")
@click.option("--config", "config_path", default=None, help="pipeline 配置文件路径")
def pipeline_report_side_state(run_id, config_path):
    """汇总 rolling_sim 每月 PCM 候选池（不再使用 symbol_side_state）。"""
    args = ["--run-id", str(run_id)]
    if config_path:
        args.extend(["--config", config_path])
    sys.exit(run_script("scripts/pipeline_report_side_state.py", args))


@pipeline.command("debug-pcm-candidates")
@click.option("--run-id", required=True, help="rolling_sim run id (timestamp)")
@click.option("--month", required=True, help="月份 YYYY-MM")
@click.option("--config", "config_path", default=None, help="pipeline 配置文件路径")
def pipeline_debug_pcm_candidates(run_id, month, config_path):
    """查看指定月份 PCM 候选池明细."""
    args = ["--run-id", str(run_id), "--month", str(month)]
    if config_path:
        args.extend(["--config", config_path])
    sys.exit(run_script("scripts/pipeline_debug_pcm_candidates.py", args))


@cli.group()
def multileg():
    """多腿管线: 研究、回放、门禁、监控、影子/测试实盘."""
    pass


def _multileg_default_config(profile: str, strategy: str = "") -> str:
    """Packaged research profile stem/filename → path under ``.../<slug>/research/``."""
    from pathlib import Path as _Path

    from src.config.strategy_layout import (
        packaged_research_yaml_name,
        strategy_packaged_root,
    )

    fname = packaged_research_yaml_name(str(profile).strip() or None)
    s = str(strategy or "").strip()
    if s:
        root = _Path.cwd().resolve()
        pkg = strategy_packaged_root(root, s)
        prof = pkg / "research" / fname
        try:
            return str(prof.relative_to(root))
        except ValueError:
            return str(prof)
    return "config/pipelines/multileg_orchestrate_2h.yaml"


def _multileg_stage_for_strategy(strategy: str) -> str:
    s = str(strategy or "").strip().lower()
    if s == "chop_grid":
        return "grid_backtest"
    if s in ("dual_add_trend", "trend_scalp"):
        return "dual_add_backtest"
    return "rolling_sim"


def _multileg_auto_stage(profile: str, strategy: str, run_all: bool) -> str:
    if run_all or not str(strategy).strip():
        return "rolling_sim"
    from src.config.strategy_layout import (
        packaged_profile_yaml_is_validate_static,
        packaged_research_yaml_name,
    )

    fname = packaged_research_yaml_name(str(profile).strip() or None)
    if packaged_profile_yaml_is_validate_static(fname):
        return _multileg_stage_for_strategy(strategy)
    return "rolling_sim"


@multileg.command("validate-config")
@click.option(
    "--config",
    "config_path",
    default="config/pipelines/multileg_orchestrate_2h.yaml",
    show_default=True,
)
@click.option("--constitution-yaml", default="", help="可选：覆盖宪法 YAML 路径")
def multileg_validate_config(config_path: str, constitution_yaml: str):
    """校验多腿 pipeline + constitution 对齐与风险字段完整性."""
    args = ["--config", config_path]
    if str(constitution_yaml).strip():
        args.extend(["--constitution-yaml", str(constitution_yaml).strip()])
    sys.exit(run_script("scripts/multileg_validate_config.py", args))


@multileg.command("research")
@click.option("--strategy", default="", help="单策略 (chop_grid/trend_scalp)")
@click.option("--all", "run_all", is_flag=True, help="跑 config 里的全部多腿策略")
@click.option(
    "--profile",
    default=PACKAGED_PROFILE_DEFAULT_STEM,
    show_default=True,
    type=click.Choice(
        [
            "calibrate_roll.default",
            "research_roll.features_on",
            "validate_static.full_study",
            "validate_static.constrained",
        ]
    ),
)
@click.option("--config", "config_path", default="", help="可选：指定 pipeline YAML")
@click.option("--stage", default="auto", help="auto|rolling_sim|grid_backtest|dual_add_backtest")
@click.option("--month", default="", help="可选：YYYY-MM 或逗号分隔多个月")
@click.option("--end-date", default="", help="可选：截止日期")
@click.option("--dry-run", is_flag=True, help="打印命令不执行")
@click.option("--no-adopt", is_flag=True, help="禁止自动采纳")
@click.option("--use-1min", is_flag=True, help="使用 1min bar 精细模拟")
@click.option("--live-root", default="live/highcap", show_default=True)
def multileg_research(
    strategy: str,
    run_all: bool,
    profile: str,
    config_path: str,
    stage: str,
    month: str,
    end_date: str,
    dry_run: bool,
    no_adopt: bool,
    use_1min: bool,
    live_root: str,
):
    """运行多腿研究阶段（复用 auto_research_pipeline stage）。"""
    cfg = str(config_path).strip() or _multileg_default_config(profile, strategy)
    args: List[str] = ["--config", cfg]
    if run_all:
        args.append("--all")
    elif str(strategy).strip():
        args.extend(["--strategy", str(strategy).strip()])
    else:
        args.append("--all")
    stg = str(stage).strip().lower()
    if stg == "auto":
        stg = _multileg_auto_stage(profile, strategy, run_all)
    if stg != "full":
        args.extend(["--stage", stg])
    if str(month).strip():
        args.extend(["--month", str(month).strip()])
    if str(end_date).strip():
        args.extend(["--end-date", str(end_date).strip()])
    if dry_run:
        args.append("--dry-run")
    if no_adopt:
        args.append("--no-adopt")
    if use_1min:
        args.append("--use-1min")
    if str(live_root).strip() and str(live_root).strip() != "live/highcap":
        args.extend(["--live-root", str(live_root).strip()])
    sys.exit(run_script("scripts/auto_research_pipeline.py", args))


@multileg.command("replay")
@click.option(
    "--config",
    "config_path",
    default="config/pipelines/multileg_orchestrate_2h.yaml",
    show_default=True,
)
@click.option("--months", default="", help="YYYY-MM,YYYY-MM 或 YYYY-MM:YYYY-MM")
@click.option("--strategy", default="", help="单策略（可选）")
@click.option("--all", "run_all", is_flag=True, help="跑配置中的全部策略")
@click.option("--end-date", default="", help="可选截止日期")
@click.option("--dry-run", is_flag=True)
@click.option("--use-1min", is_flag=True)
@click.option("--live-root", default="live/highcap", show_default=True)
def multileg_replay(
    config_path: str,
    months: str,
    strategy: str,
    run_all: bool,
    end_date: str,
    dry_run: bool,
    use_1min: bool,
    live_root: str,
):
    """无前视多腿回放（rolling_sim 封装 + 汇总报告）。"""
    args = ["--config", config_path]
    if str(months).strip():
        args.extend(["--months", str(months).strip()])
    if str(strategy).strip():
        args.extend(["--strategy", str(strategy).strip()])
    if run_all:
        args.append("--all")
    if str(end_date).strip():
        args.extend(["--end-date", str(end_date).strip()])
    if dry_run:
        args.append("--dry-run")
    if use_1min:
        args.append("--use-1min")
    if str(live_root).strip() and str(live_root).strip() != "live/highcap":
        args.extend(["--live-root", str(live_root).strip()])
    sys.exit(run_script("scripts/multileg_replay.py", args))


@multileg.command("gate")
@click.option("--run-dir", required=True, help="rolling run root: .../_rolling_sim/<run_id>")
@click.option(
    "--config",
    "config_path",
    default="config/pipelines/multileg_orchestrate_2h.yaml",
    show_default=True,
)
@click.option("--out-json", default="", help="可选输出 JSON")
@click.option("--out-html", default="", help="可选输出 HTML")
def multileg_gate(run_dir: str, config_path: str, out_json: str, out_html: str):
    """多腿上线门禁评估（READY_SHADOW/RESEARCH_ONLY/RETUNE_THRESHOLDS/OFFLINE）。"""
    args = ["--run-dir", run_dir, "--config", config_path]
    if str(out_json).strip():
        args.extend(["--out-json", str(out_json).strip()])
    if str(out_html).strip():
        args.extend(["--out-html", str(out_html).strip()])
    sys.exit(run_script("scripts/multileg_gate.py", args))


@multileg.command("monitor")
@click.option(
    "--config",
    "config_path",
    default="config/pipelines/multileg_orchestrate_2h.yaml",
    show_default=True,
)
@click.option("--run-id", default="", help="可选 rolling_sim run id")
@click.option("--lookback-months", type=int, default=6, show_default=True)
@click.option("--out-json", default="", help="可选输出 JSON")
@click.option("--out-html", default="", help="可选输出 HTML")
def multileg_monitor(
    config_path: str, run_id: str, lookback_months: int, out_json: str, out_html: str
):
    """多腿健康监控（regime/feature/threshold/risk 信号）。"""
    args = ["--config", config_path, "--lookback-months", str(int(lookback_months))]
    if str(run_id).strip():
        args.extend(["--run-id", str(run_id).strip()])
    if str(out_json).strip():
        args.extend(["--out-json", str(out_json).strip()])
    if str(out_html).strip():
        args.extend(["--out-html", str(out_html).strip()])
    sys.exit(run_script("scripts/multileg_monitor.py", args))


@multileg.command("shadow")
@click.option("--strategies", default="chop_grid,trend_scalp", show_default=True)
@click.option("--bar-source", default="feature-store", show_default=True)
@click.option("--once", is_flag=True, help="仅跑一次")
@click.option("--poll-seconds", type=float, default=60.0, show_default=True)
@click.option("--config", "constitution_yaml", default="", help="可选：宪法路径覆盖")
def multileg_shadow(
    strategies: str,
    bar_source: str,
    once: bool,
    poll_seconds: float,
    constitution_yaml: str,
):
    """启动多腿影子运行（run_multi_leg_live.py --mode shadow）。"""
    args = [
        "--mode",
        "shadow",
        "--strategies",
        strategies,
        "--bar-source",
        bar_source,
        "--poll-seconds",
        str(float(poll_seconds)),
    ]
    if once:
        args.append("--once")
    if str(constitution_yaml).strip():
        args.extend(["--constitution-yaml", str(constitution_yaml).strip()])
    sys.exit(run_script("scripts/run_multi_leg_live.py", args))


@multileg.command("live")
@click.option(
    "--mode",
    default="testnet",
    show_default=True,
    type=click.Choice(["testnet", "mainnet"]),
)
@click.option("--strategies", default="chop_grid,trend_scalp", show_default=True)
@click.option("--bar-source", default="feature-store", show_default=True)
@click.option("--poll-seconds", type=float, default=60.0, show_default=True)
@click.option("--allow-shared-account", is_flag=True)
@click.option(
    "--no-orders",
    is_flag=True,
    help="只观测不下单（testnet/mainnet；同 MLBOT_MULTI_LEG_NO_ORDERS=1）",
)
@click.option("--config", "constitution_yaml", default="", help="可选：宪法路径覆盖")
def multileg_live(
    mode: str,
    strategies: str,
    bar_source: str,
    poll_seconds: float,
    allow_shared_account: bool,
    no_orders: bool,
    constitution_yaml: str,
):
    """启动多腿测试网/主网运行（run_multi_leg_live.py）。"""
    args = [
        "--mode",
        str(mode),
        "--strategies",
        strategies,
        "--bar-source",
        bar_source,
        "--poll-seconds",
        str(float(poll_seconds)),
    ]
    if allow_shared_account:
        args.append("--allow-shared-account")
    if no_orders:
        args.append("--no-orders")
    if str(constitution_yaml).strip():
        args.extend(["--constitution-yaml", str(constitution_yaml).strip()])
    sys.exit(run_script("scripts/run_multi_leg_live.py", args))


@experiment.command("regime-gate")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Input logs file (must contain symbol, timestamp, regime, ret_mean, ret_trend)",
)
@click.option(
    "--output-dir",
    "output_dir",
    default="results/experiments",
    help="Output directory for experiment results",
)
@click.option(
    "--features-store-root",
    "features_store_root",
    default="feature_store",
    help="FeatureStore root directory",
)
@click.option(
    "--features-store-layer",
    "features_store_layer",
    required=True,
    help="FeatureStore layer name",
)
@click.option("--symbols", "symbols", default=None, help="Comma-separated symbols")
@click.option("--timeframe", "timeframe", default="240T")
@click.option("--start-date", "start_date", default=None)
@click.option("--end-date", "end_date", default=None)
@click.option(
    "--execution-archetypes",
    "execution_archetypes",
    default="config/nnmultihead/execution_archetypes.yaml",
)
@click.option(
    "--db-path",
    "db_path",
    default="data/order_management.db",
)
@click.option("--evidence-quantiles", "evidence_quantiles", default=None)
@click.option("--physics-regime", "physics_regime", default=None)
@click.option("--semantic-score-floors", "semantic_score_floors", default=None)
@click.option("--ret-mean-col", "ret_mean_col", default="ret_mean")
@click.option("--ret-trend-col", "ret_trend_col", default="ret_trend")
@click.pass_context
def experiment_regime_gate(
    ctx,
    logs_path,
    output_dir,
    features_store_root,
    features_store_layer,
    symbols,
    timeframe,
    start_date,
    end_date,
    execution_archetypes,
    db_path,
    evidence_quantiles,
    physics_regime,
    semantic_score_floors,
    ret_mean_col,
    ret_trend_col,
):
    """Run regime and gate experiments with 4 configurations."""
    docker = ctx.obj.get("docker", False) if ctx.obj else False
    args = [
        "--logs",
        logs_path,
        "--output-dir",
        output_dir,
        "--features-store-root",
        features_store_root,
        "--features-store-layer",
        features_store_layer,
        "--timeframe",
        timeframe,
        "--execution-archetypes",
        execution_archetypes,
        "--db-path",
        db_path,
        "--ret-mean-col",
        ret_mean_col,
        "--ret-trend-col",
        ret_trend_col,
    ]
    if symbols:
        args.extend(["--symbols", symbols])
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if evidence_quantiles:
        args.extend(["--evidence-quantiles", evidence_quantiles])
    if physics_regime:
        args.extend(["--physics-regime", physics_regime])
    if semantic_score_floors:
        args.extend(["--semantic-score-floors", semantic_score_floors])
    sys.exit(run_script("scripts/experiment_regime_gate.py", args, docker=docker))


@click.option("--logs", "logs_path", required=True, help="logs_3action.parquet")
@click.option("--regime", "regime_path", required=True, help="physics_regime parquet")
@click.option("--output-json", required=True, help="Output JSON path")
@click.option("--output-md", required=True, help="Output Markdown path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rule_diagnose_e2e_symbol_regime_archetype(
    logs_path,
    regime_path,
    output_json,
    output_md,
    docker,
):
    """E2E KPI by Symbol × Regime × Archetype (sharpe, trade_count, win_rate, profit_loss_ratio)."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs_path}" if use_workspace_prefix else logs_path,
        "--regime",
        f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
        "--output-json",
        f"/workspace/{output_json}" if use_workspace_prefix else output_json,
        "--output-md",
        f"/workspace/{output_md}" if use_workspace_prefix else output_md,
    ]
    sys.exit(run_script("scripts/diagnose_e2e_kpi.py", args, docker=docker))


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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def rl_shadow_eval_3action(logs_path, out_dir, train_ratio, docker):
    click.echo(
        "Note: `mlbot rl shadow-eval-3action` is deprecated for v0 mainline. "
        "Use `mlbot nnmultihead shadow-eval-3action` instead."
    )
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
        args.extend(
            ["--router-dir-conf-trend-min", str(float(router_dir_conf_trend_min))]
        )
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    click.echo(
        "Note: `mlbot rl run-e2e-3action` is deprecated for v0 mainline. "
        "Use `mlbot nnmultihead run-e2e-3action` instead."
    )
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
    "--task-spec",
    required=True,
    help="TaskSpec YAML (v1). REQUIRED: nnmultihead is TaskSpec-only (no legacy config mode).",
)
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--base-config",
    default=None,
    help="Override base nnmultihead config dir (default: TaskSpec.model_plan.base_config_dir).",
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
@click.option(
    "--emit-evidence-quantiles/--no-emit-evidence-quantiles",
    default=None,
    help="If set, write evidence_quantiles.json based on training window features.",
)
@click.option(
    "--evidence-quantiles-out",
    default=None,
    help="Output path for evidence_quantiles.json (default: <run_dir>/evidence_quantiles.json).",
)
@click.option(
    "--evidence-quantiles-keys",
    default=None,
    help="Comma-separated feature keys to include in quantiles.",
)
@click.option(
    "--evidence-quantiles-prefixes",
    default=None,
    help="Comma-separated prefixes to include in quantiles (in addition to keys).",
)
@click.option(
    "--evidence-quantiles",
    default=None,
    help="Comma-separated quantiles (e.g., 0.1,0.5,0.9).",
)
@click.option(
    "--evidence-quantiles-global/--evidence-quantiles-per-symbol",
    default=None,
    help="If set, pool all symbols into GLOBAL quantiles.",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_train(
    task_spec,
    symbols,
    timeframe,
    data_path,
    base_config,
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
    emit_evidence_quantiles,
    evidence_quantiles_out,
    evidence_quantiles_keys,
    evidence_quantiles_prefixes,
    evidence_quantiles,
    evidence_quantiles_global,
    docker,
):
    """Train NN multi-head path primitives MLP and save report.html artifacts."""
    # Note: Layer name auto-generation is handled by the script itself.
    # CLI just passes the parameter as-is (None = auto-generate).
    use_workspace_prefix = docker and not _is_in_docker()
    import yaml

    ts_path = Path(task_spec)
    if not ts_path.is_absolute():
        ts_path = (PROJECT_ROOT / ts_path).resolve()
    ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}
    task_id = str(ts_obj.get("task_id") or "").strip() or "TASKSPEC"
    training_cfg = (ts_obj.get("model_plan") or {}).get("training") or {}
    base_cfg = (
        str(base_config).strip()
        if base_config
        else str((ts_obj.get("model_plan") or {}).get("base_config_dir") or "").strip()
    ) or "config/nnmultihead/path_primitives_4h_80h_min"

    # Default windows from TaskSpec if not explicitly provided
    win_train = (ts_obj.get("windows") or {}).get("train") or {}
    if not start_date:
        start_date = str(win_train.get("start") or "").strip() or None
    if not end_date:
        end_date = str(win_train.get("end") or "").strip() or None

    derived_rel = f"results/derived_configs/{task_id}/nnmh_config_train"
    materialize_nnmh_config_from_task_spec(
        task_spec_path=str(ts_path),
        base_config_dir=base_cfg,
        out_config_dir=derived_rel,
    )
    effective_config = derived_rel

    # FeatureStore defaults come from TaskSpec unless explicitly overridden.
    fp = ts_obj.get("feature_plan") or {}
    fs = fp.get("feature_store") or {}
    if feature_store_layer is None:
        v = str(fs.get("layer") or "").strip()
        if v:
            feature_store_layer = v
    if feature_store_root == "feature_store":
        v = str(fs.get("root") or "").strip()
        if v:
            feature_store_root = v

    if emit_evidence_quantiles is None:
        emit_evidence_quantiles = bool(
            training_cfg.get("emit_evidence_quantiles", False)
        )
    if evidence_quantiles_out is None:
        evidence_quantiles_out = training_cfg.get("evidence_quantiles_out")
    if evidence_quantiles_keys is None:
        evidence_quantiles_keys = training_cfg.get(
            "evidence_quantiles_keys", "vpin,cvd_change_5"
        )
    if evidence_quantiles_prefixes is None:
        evidence_quantiles_prefixes = training_cfg.get(
            "evidence_quantiles_prefixes", ""
        )
    if evidence_quantiles is None:
        evidence_quantiles = training_cfg.get("evidence_quantiles", "0.1,0.5,0.9")
    if evidence_quantiles_global is None:
        evidence_quantiles_global = bool(
            training_cfg.get("evidence_quantiles_global", False)
        )

    args = [
        "--config",
        f"/workspace/{effective_config}" if use_workspace_prefix else effective_config,
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
    if emit_evidence_quantiles:
        args.append("--emit-evidence-quantiles")
        if evidence_quantiles_out:
            args.extend(["--evidence-quantiles-out", evidence_quantiles_out])
        if evidence_quantiles_keys:
            args.extend(["--evidence-quantiles-keys", evidence_quantiles_keys])
        if evidence_quantiles_prefixes:
            args.extend(["--evidence-quantiles-prefixes", evidence_quantiles_prefixes])
        if evidence_quantiles:
            args.extend(["--evidence-quantiles", evidence_quantiles])
        if evidence_quantiles_global:
            args.append("--evidence-quantiles-global")
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
    "--task-spec",
    required=True,
    help="TaskSpec YAML (v1). REQUIRED: nnmultihead is TaskSpec-only (no legacy config mode).",
)
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--base-config",
    default=None,
    help="Override base nnmultihead config dir (default: TaskSpec.model_plan.base_config_dir).",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_predict(
    task_spec,
    symbols,
    timeframe,
    data_path,
    base_config,
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
    import yaml

    ts_path = Path(task_spec)
    if not ts_path.is_absolute():
        ts_path = (PROJECT_ROOT / ts_path).resolve()
    ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}
    task_id = str(ts_obj.get("task_id") or "").strip() or "TASKSPEC"
    base_cfg = (
        str(base_config).strip()
        if base_config
        else str((ts_obj.get("model_plan") or {}).get("base_config_dir") or "").strip()
    ) or "config/nnmultihead/path_primitives_4h_80h_min"

    # Default window from TaskSpec if not explicitly provided (prefer oos, else holdout, else train)
    wins = ts_obj.get("windows") or {}
    win = wins.get("oos") or wins.get("holdout") or wins.get("train") or {}
    if not start_date:
        start_date = str(win.get("start") or "").strip() or None
    if not end_date:
        end_date = str(win.get("end") or "").strip() or None

    derived_rel = f"results/derived_configs/{task_id}/nnmh_config_predict"
    materialize_nnmh_config_from_task_spec(
        task_spec_path=str(ts_path),
        base_config_dir=base_cfg,
        out_config_dir=derived_rel,
    )
    effective_config = derived_rel

    # FeatureStore defaults come from TaskSpec unless explicitly overridden.
    fp = ts_obj.get("feature_plan") or {}
    fs = fp.get("feature_store") or {}
    if feature_store_layer is None:
        v = str(fs.get("layer") or "").strip()
        if v:
            feature_store_layer = v
    if feature_store_root == "feature_store":
        v = str(fs.get("root") or "").strip()
        if v:
            feature_store_root = v
    args = [
        "--config",
        f"/workspace/{effective_config}" if use_workspace_prefix else effective_config,
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
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Output directory root for this pipeline run.",
)
@click.option(
    "--task-spec",
    required=True,
    help="TaskSpec YAML (v1). REQUIRED: pipeline is TaskSpec-only (no legacy config mode).",
)
@click.option(
    "--base-config",
    default=None,
    help="Override base nnmultihead config dir (default: TaskSpec.model_plan.base_config_dir).",
)
@click.option("--mfe-min", type=float, default=None)
@click.option("--eff-min", type=float, default=None)
@click.option("--dir-conf-trend-min", type=float, default=None)
@click.option("--mfe-trend-min", type=float, default=None)
@click.option("--ttm-trend-min", type=float, default=None)
@click.option("--eff-mean-min", type=float, default=None)
@click.option("--ttm-mean-max", type=float, default=None)
@click.option(
    "--router-thresholds-json",
    default=None,
    help="Optional JSON file with 7 router thresholds to one-click apply in this pipeline. Any explicit --mfe-min/--eff-min/... flags take precedence.",
)
@click.option("--train-ratio", type=float, default=0.7, show_default=True)
@click.option("--entry-delay", type=int, default=0, show_default=True)
@click.option("--cost-per-turnover", type=float, default=0.0, show_default=True)
@click.option("--slippage-bps", type=float, default=0.0, show_default=True)
@click.option(
    "--preds-in-log1p/--preds-not-in-log1p",
    default=True,
    help="Whether head_mfe/head_mae/head_t_to_mfe are in log1p space (affects Router diagnostics only).",
)
@click.option(
    "--emit-exec-log-stages/--no-emit-exec-log-stages",
    default=True,
    help="Emit split-stage execution logs from pipeline outputs.",
)
@click.option(
    "--exec-log-stage-dir",
    default=None,
    help="Output dir for stage logs (default: <out>/exec_logs).",
)
@click.option(
    "--emit-exec-log-canonical/--no-emit-exec-log-canonical",
    default=False,
    help="Also aggregate canonical execution log from stage logs.",
)
@click.option(
    "--exec-log-canonical-path",
    default=None,
    help="Output canonical jsonl (default: <out>/execution_log.jsonl).",
)
@click.option(
    "--emit-router-plots/--no-emit-router-plots",
    default=True,
    show_default=True,
    help="Emit router mode plots on OHLC for all symbols.",
)
@click.option(
    "--router-plot-dir",
    default=None,
    help="Output dir for router plots (default: <out>/router_plots).",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_pipeline_3action_e2e(
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
    base_config,
    mfe_min,
    eff_min,
    dir_conf_trend_min,
    mfe_trend_min,
    ttm_trend_min,
    eff_mean_min,
    ttm_mean_max,
    router_thresholds_json,
    train_ratio,
    entry_delay,
    cost_per_turnover,
    slippage_bps,
    preds_in_log1p,
    emit_exec_log_stages,
    exec_log_stage_dir,
    emit_exec_log_canonical,
    exec_log_canonical_path,
    emit_router_plots,
    router_plot_dir,
    docker,
):
    """
    One-command mainline pipeline:
      nnmultihead predict -> rule physics-regime -> nnmultihead build-execution-logs -> nnmultihead run-e2e-3action

    Motivation:
    - Keep existing family commands (nnmultihead/rule/rl) for clarity and modularity.
    - Provide a smooth, single entrypoint for the recommended mainline (Regime + Execution),
      while BC/RL/FSM remain optional modules.
    """
    use_workspace_prefix = docker and not _is_in_docker()
    out_root = f"/workspace/{out_dir}" if use_workspace_prefix else out_dir
    preds_dir = f"{out_root}/preds"
    regime_path = f"{out_root}/physics_regime.parquet"
    logs_path = f"{out_root}/logs_execution.parquet"
    e2e_out = f"{out_root}/e2e"

    # -------------------------------------------------------------------------
    # Convenience: always materialize a baseline router thresholds JSON in out_dir
    # so plateau tuning can be run without hand-writing the file.
    # -------------------------------------------------------------------------
    try:
        import json as _json
        from src.time_series_model.rule.router_3action import Rule3ActionConfig as _R3

        cfg0 = _R3()
        # Optional: load thresholds JSON (best/baseline). Explicit CLI flags win.
        loaded = {}
        try:
            if router_thresholds_json:
                pp = Path(str(router_thresholds_json))
                if not pp.is_absolute():
                    pp = (PROJECT_ROOT / pp).resolve()
                loaded = _json.loads(pp.read_text(encoding="utf-8")) or {}
        except Exception:
            loaded = {}

        def _pick(name: str, cur):
            if cur is not None:
                return float(cur)
            if (
                isinstance(loaded, dict)
                and name in loaded
                and loaded.get(name) is not None
            ):
                return float(loaded.get(name))
            return float(getattr(cfg0, name))

        mfe_min = _pick("mfe_min", mfe_min)
        eff_min = _pick("eff_min", eff_min)
        dir_conf_trend_min = _pick("dir_conf_trend_min", dir_conf_trend_min)
        mfe_trend_min = _pick("mfe_trend_min", mfe_trend_min)
        ttm_trend_min = _pick("ttm_trend_min", ttm_trend_min)
        eff_mean_min = _pick("eff_mean_min", eff_mean_min)
        ttm_mean_max = _pick("ttm_mean_max", ttm_mean_max)

        baseline = {
            "mfe_min": float(mfe_min),
            "eff_min": float(eff_min),
            "dir_conf_trend_min": float(dir_conf_trend_min),
            "mfe_trend_min": float(mfe_trend_min),
            "ttm_trend_min": float(ttm_trend_min),
            "eff_mean_min": float(eff_mean_min),
            "ttm_mean_max": float(ttm_mean_max),
        }
        p = Path(out_dir)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / "router_thresholds_baseline.json").write_text(
            _json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # TaskSpec (mandatory): enforcement injection + config materialization (no legacy mode)
    # -------------------------------------------------------------------------
    import yaml

    ts_path = Path(task_spec)
    if not ts_path.is_absolute():
        ts_path = (PROJECT_ROOT / ts_path).resolve()
    ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}
    task_id = str(ts_obj.get("task_id") or "").strip() or "TASKSPEC"

    env_overrides = {"MLBOT_TASK_ID": task_id}

    enf = ts_obj.get("enforcement") or {}
    constitution_yaml = str(enf.get("constitution_yaml") or "").strip()
    kpi_gate_yaml = str(enf.get("kpi_gate_yaml") or "").strip()

    def _ws(p: str) -> str:
        return f"/workspace/{p}" if use_workspace_prefix else p

    if constitution_yaml:
        env_overrides["MLBOT_CONSTITUTION_YAML"] = _ws(constitution_yaml)
    if kpi_gate_yaml:
        env_overrides["MLBOT_KPI_GATE_YAML"] = _ws(kpi_gate_yaml)

    base_cfg = (
        str(base_config).strip()
        if base_config
        else str((ts_obj.get("model_plan") or {}).get("base_config_dir") or "").strip()
    ) or "config/nnmultihead/path_primitives_4h_80h_min"

    derived_rel = f"results/derived_configs/{task_id}/nnmh_config_pipeline"
    materialize_nnmh_config_from_task_spec(
        task_spec_path=str(ts_path),
        base_config_dir=base_cfg,
        out_config_dir=derived_rel,
    )
    effective_config = derived_rel

    # FeatureStore defaults come from TaskSpec unless explicitly overridden.
    fp = ts_obj.get("feature_plan") or {}
    fs = fp.get("feature_store") or {}
    if feature_store_layer is None:
        v = str(fs.get("layer") or "").strip()
        if v:
            feature_store_layer = v
    if feature_store_root == "feature_store":
        v = str(fs.get("root") or "").strip()
        if v:
            feature_store_root = v

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
        (
            f"/workspace/{feature_store_root}"
            if use_workspace_prefix
            else feature_store_root
        ),
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

    # [2/4] physics-regime (Router removed, regime classification is done here)
    # Note: Router (mode-3action) has been removed. Regime classification is now done in physics-regime step.
    regime_path_raw = regime_path
    args_regime = [
        "--preds",
        preds_dir,
        "--output",
        f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
        "--timeframe",
        str(timeframe),
    ]
    if feature_store_root:
        args_regime.extend(
            [
                "--feature-store-root",
                (
                    f"/workspace/{feature_store_root}"
                    if use_workspace_prefix
                    else feature_store_root
                ),
            ]
        )
    if feature_store_layer:
        args_regime.extend(["--layer", str(feature_store_layer)])
    rc = run_script(
        "scripts/physics_regime_classifier.py",
        args_regime,
        docker=docker,
        env_overrides=env_overrides,
    )
    if rc != 0:
        sys.exit(rc)

    # Optional: tree gate veto (TaskSpec gate_plan)
    gate_plan = ts_obj.get("gate_plan") or {}
    try:
        gate_enabled = bool(gate_plan.get("enabled", False))
    except Exception:
        gate_enabled = False
    gate_kind = str(gate_plan.get("kind") or "").strip()
    if gate_enabled and gate_kind == "tree_gate_veto":
        gate_logs_path = f"{out_root}/logs_execution_gated.parquet"
        gate_args = [
            "--logs",
            f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
            "--out",
            gate_logs_path,
            "--features-store-root",
            (
                f"/workspace/{feature_store_root}"
                if use_workspace_prefix
                else feature_store_root
            ),
            "--features-store-layer",
            str(feature_store_layer),
            "--symbols",
            str(symbols),
            "--timeframe",
            str(timeframe),
            "--start-date",
            str(start_date),
            "--end-date",
            str(end_date),
        ]
        gate_exec = str(
            gate_plan.get("execution_archetypes_yaml")
            or "config/nnmultihead/execution_archetypes.yaml"
        )
        gate_args.extend(
            [
                "--execution-archetypes",
                f"/workspace/{gate_exec}" if use_workspace_prefix else gate_exec,
                "--db-path",
                (
                    f"/workspace/{env_overrides.get('MLBOT_ORDER_MANAGEMENT_DB_PATH')}"
                    if use_workspace_prefix
                    and env_overrides.get("MLBOT_ORDER_MANAGEMENT_DB_PATH")
                    else env_overrides.get(
                        "MLBOT_ORDER_MANAGEMENT_DB_PATH", "data/order_management.db"
                    )
                ),
            ]
        )
        eq_path = str(gate_plan.get("evidence_quantiles_json") or "").strip()
        if eq_path:
            gate_args.extend(
                [
                    "--evidence-quantiles",
                    f"/workspace/{eq_path}" if use_workspace_prefix else eq_path,
                ]
            )
        rc = run_script(
            "scripts/apply_archetype_gate.py",
            gate_args,
            docker=docker,
            env_overrides=env_overrides,
        )
        if rc != 0:
            sys.exit(rc)
        # Gate output contains logs with gate_archetype, use it for downstream steps
        logs_path = gate_logs_path

    # Optional: router plot on OHLC (all symbols, default on)
    if emit_router_plots:
        plot_dir = router_plot_dir or f"{out_root}/router_plots"
        plot_args = [
            "--logs",
            f"/workspace/{regime_path}" if use_workspace_prefix else regime_path,
            "--feature-store-root",
            (
                f"/workspace/{feature_store_root}"
                if use_workspace_prefix
                else feature_store_root
            ),
            "--feature-store-layer",
            str(feature_store_layer),
            "--all-symbols",
            "--out",
            plot_dir,
        ]
        if start_date:
            plot_args.extend(["--start-date", str(start_date)])
        if end_date:
            plot_args.extend(["--end-date", str(end_date)])
        _ = run_script(
            "scripts/plot_router_modes_kline.py",
            plot_args,
            docker=docker,
            env_overrides=env_overrides,
        )

    # [3/4] build-execution-logs
    args_logs = [
        "--preds",
        preds_dir,
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
        "scripts/rl_build_execution_logs.py",
        args_logs,
        docker=docker,
        env_overrides=env_overrides,
    )
    if rc != 0:
        sys.exit(rc)

    # Optional: emit split-stage execution logs (pipeline parity with live).
    if emit_exec_log_stages:
        stage_dir = exec_log_stage_dir or f"{out_root}/exec_logs"
        run_id = Path(out_root).name
        stage_args = [
            "--preds",
            preds_dir,
            "--logs",
            logs_path,
            "--out-dir",
            stage_dir,
            "--run-id",
            run_id,
            "--timeframe",
            str(timeframe),
            "--strategy-name",
            "pipeline-3action-e2e",
        ]
        rc = run_script(
            "scripts/build_execution_log_stages.py",
            stage_args,
            docker=docker,
            env_overrides=env_overrides,
        )
        if rc != 0:
            sys.exit(rc)
        if emit_exec_log_canonical:
            canonical_path = (
                exec_log_canonical_path or f"{out_root}/execution_log.jsonl"
            )
            rc = run_script(
                "scripts/aggregate_execution_log_stages.py",
                ["--stage-dir", stage_dir, "--out", canonical_path],
                docker=docker,
                env_overrides=env_overrides,
            )
            if rc != 0:
                sys.exit(rc)

    # [4/4] run-e2e-3action (shadow + counterfactual + fsm decision)
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

    # ---------------------------------------------------------------------
    # Per-symbol primitives KPI (default-on diagnostic):
    # - produces per_symbol_kpi/(baseline|tuned)/per_symbol_kpi.(md|csv|json)
    # - kpi_journal.html will auto-embed these tables when present
    # ---------------------------------------------------------------------
    try:
        per_sym_root = f"{out_root}/per_symbol_kpi"

        # Baseline thresholds (always materialized at out_dir/router_thresholds_baseline.json)
        rc2 = run_script(
            "scripts/nnmh_per_symbol_primitives_kpi.py",
            [
                "--model",
                str(model_path),
                "--symbols",
                str(symbols),
                "--timeframe",
                str(timeframe),
                "--start-date",
                str(start_date),
                "--end-date",
                str(end_date),
                "--features-store-root",
                str(feature_store_root),
                "--features-store-layer",
                str(feature_store_layer),
                "--router-thresholds-json",
                f"{out_root}/router_thresholds_baseline.json",
                "--out-dir",
                f"{per_sym_root}/baseline",
                "--max-rows-per-symbol",
                "4000",
            ],
            docker=docker,
            env_overrides=env_overrides,
        )
        _ = rc2  # non-fatal

        # Tuned thresholds (if explicitly provided to pipeline)
        if router_thresholds_json:
            rc3 = run_script(
                "scripts/nnmh_per_symbol_primitives_kpi.py",
                [
                    "--model",
                    str(model),
                    "--symbols",
                    str(symbols),
                    "--timeframe",
                    str(timeframe),
                    "--start-date",
                    str(start_date),
                    "--end-date",
                    str(end_date),
                    "--features-store-root",
                    str(feature_store_root),
                    "--features-store-layer",
                    str(feature_store_layer),
                    "--router-thresholds-json",
                    str(router_thresholds_json),
                    "--out-dir",
                    f"{per_sym_root}/tuned",
                    "--max-rows-per-symbol",
                    "4000",
                ],
                docker=docker,
                env_overrides=env_overrides,
            )
            _ = rc3  # non-fatal
    except Exception:
        pass

    # KPI journal (append-only): summarize layer KPIs in one place per run dir.
    try:
        from src.time_series_model.diagnostics.kpi_journal import write_kpi_journal

        # `out_dir` is the user-provided run root for this pipeline invocation.
        write_kpi_journal(run_dir=str(out_dir), stage="pipeline")
    except Exception:
        pass

    sys.exit(rc)


@nnmultihead.command("build-execution-logs")
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
    help="Optional mode file/dir (deprecated, not used)",
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
    default="rr_execution",
    show_default=True,
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_build_execution_logs(
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
    """Build logs_3action.parquet for mainline evaluation (v0)."""
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
    sys.exit(run_script("scripts/rl_build_execution_logs.py", args, docker=docker))


@nnmultihead.command("run-e2e-3action")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_run_e2e_3action(
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
    """Mainline e2e evaluation (shadow + counterfactual + fsm decision)."""
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

    rc = run_script(
        "scripts/rl_counterfactual_eval_3action.py",
        cf_args,
        docker=docker,
    )
    if rc != 0:
        sys.exit(rc)

    fsm_args = [
        "--metrics",
        f"{cf_out}/metrics.json",
        "--state",
        str(fsm_state),
        "--promote_days",
        str(int(promote_days)),
        "--cooldown_days",
        str(int(cooldown_days)),
        "--out",
        fsm_out,
    ]
    sys.exit(run_script("scripts/rl_fsm_decide.py", fsm_args, docker=docker))


@nnmultihead.command("shadow-eval-3action")
@click.option(
    "--logs",
    "logs_path",
    required=True,
    help="Logs .csv/.parquet with mode + heads (from nnmultihead build-logs-3action).",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Output directory for shadow report.",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.7,
    help="Train ratio per symbol (time-ordered).",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_shadow_eval_3action(logs_path, out_dir, train_ratio, docker):
    """Mainline shadow evaluation (behavioral consistency)."""
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_materialize_config_from_task_spec(
    task_spec, base_config, out_config, docker
):
    """
    Generate a concrete config directory so tiers are *real* (not just metadata).

    Typical usage:
      - generate tier0 config -> train model -> eval
      - generate tier0+1 config -> train model -> eval
      - compare A-layer + system reports
    """
    use_workspace_prefix = docker and not _is_in_docker()
    # Materialize on host filesystem; docker wrapper will mount repo to /workspace anyway.
    materialize_nnmh_config_from_task_spec(
        task_spec_path=task_spec,
        base_config_dir=base_config,
        out_config_dir=out_config,
    )
    click.echo(
        f"✅ Derived config written: {('/workspace/' + out_config) if use_workspace_prefix else out_config}"
    )


@nnmultihead.command("build-feature-store")
@click.option(
    "--task-spec",
    required=True,
    help="TaskSpec YAML (v1). REQUIRED: nnmultihead is TaskSpec-only (no legacy config mode).",
)
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--base-config",
    default=None,
    help="Override base nnmultihead config dir (default: TaskSpec.model_plan.base_config_dir).",
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
    default=None,
    help="FeatureStore layer name for nnmultihead features (default: TaskSpec.feature_plan.feature_store.layer).",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_build_feature_store(
    task_spec,
    symbols,
    timeframe,
    data_path,
    base_config,
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
    import yaml

    ts_path = Path(task_spec)
    if not ts_path.is_absolute():
        ts_path = (PROJECT_ROOT / ts_path).resolve()
    ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}
    task_id = str(ts_obj.get("task_id") or "").strip() or "TASKSPEC"
    base_cfg = (
        str(base_config).strip()
        if base_config
        else str((ts_obj.get("model_plan") or {}).get("base_config_dir") or "").strip()
    ) or "config/nnmultihead/path_primitives_4h_80h_min"

    # Default window from TaskSpec if not explicitly provided (prefer train)
    win_train = (ts_obj.get("windows") or {}).get("train") or {}
    if not start_date:
        start_date = str(win_train.get("start") or "").strip() or None
    if not end_date:
        end_date = str(win_train.get("end") or "").strip() or None

    derived_rel = f"results/derived_configs/{task_id}/nnmh_config_feature_store"
    materialize_nnmh_config_from_task_spec(
        task_spec_path=str(ts_path),
        base_config_dir=base_cfg,
        out_config_dir=derived_rel,
    )
    effective_config = derived_rel

    # FeatureStore defaults come from TaskSpec unless explicitly overridden.
    fp = ts_obj.get("feature_plan") or {}
    fs = fp.get("feature_store") or {}
    if layer is None:
        v = str(fs.get("layer") or "").strip()
        if v:
            layer = v
    if feature_store_root == "feature_store":
        v = str(fs.get("root") or "").strip()
        if v:
            feature_store_root = v

    args = [
        "--config",
        f"/workspace/{effective_config}" if use_workspace_prefix else effective_config,
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
        str(layer),
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
    "--task-spec",
    required=True,
    help="TaskSpec YAML (v1). REQUIRED: nnmultihead is TaskSpec-only (no legacy config mode).",
)
@click.option(
    "--symbols",
    "-s",
    default="BTCUSDT",
    help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
)
@click.option("--timeframe", "-t", default="240T", help="Timeframe (e.g., 240T for 4H)")
@click.option("--data-path", default="data/parquet_data", help="Data directory")
@click.option(
    "--base-config",
    default=None,
    help="Override base nnmultihead config dir (default: TaskSpec.model_plan.base_config_dir).",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_eval(
    task_spec,
    symbols,
    timeframe,
    data_path,
    base_config,
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
    import yaml

    ts_path = Path(task_spec)
    if not ts_path.is_absolute():
        ts_path = (PROJECT_ROOT / ts_path).resolve()
    ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}
    task_id = str(ts_obj.get("task_id") or "").strip() or "TASKSPEC"
    base_cfg = (
        str(base_config).strip()
        if base_config
        else str((ts_obj.get("model_plan") or {}).get("base_config_dir") or "").strip()
    ) or "config/nnmultihead/path_primitives_4h_80h_min"

    # Default window from TaskSpec if not explicitly provided (prefer oos, else holdout, else train)
    wins = ts_obj.get("windows") or {}
    win = wins.get("oos") or wins.get("holdout") or wins.get("train") or {}
    if not start_date:
        start_date = str(win.get("start") or "").strip() or None
    if not end_date:
        end_date = str(win.get("end") or "").strip() or None

    derived_rel = f"results/derived_configs/{task_id}/nnmh_config_eval"
    materialize_nnmh_config_from_task_spec(
        task_spec_path=str(ts_path),
        base_config_dir=base_cfg,
        out_config_dir=derived_rel,
    )
    effective_config = derived_rel

    args = [
        "--config",
        f"/workspace/{effective_config}" if use_workspace_prefix else effective_config,
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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


@nnmultihead.command("compare-feature-sets")
@click.option(
    "--task-spec",
    type=str,
    required=True,
    help="TaskSpec YAML (source of tiers + feature_plan_ref/overrides).",
)
@click.option(
    "--base-config",
    type=str,
    required=True,
    help="Base nnmultihead config dir (template dir used for materialization).",
)
@click.option(
    "--poolb-yaml",
    type=str,
    required=True,
    help="PoolB YAML exported by nnmultihead factor-eval / feature-group-search (list of feature nodes).",
)
@click.option(
    "--out",
    type=str,
    required=False,
    default=None,
    help="Output directory for report JSON/MD. Default: results/feature_compare/<task_id>__<poolb_stem>",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_compare_feature_sets(task_spec, base_config, poolb_yaml, out, docker):
    """
    Compare feature sets: TaskSpec-tier required nodes vs PoolB suggestion YAML.

    Output:
      - features_compare_summary.json
      - features_compare_summary.md
    """
    import json
    import shutil
    from pathlib import Path
    import yaml

    use_workspace_prefix = docker and not _is_in_docker()

    # Default output under results/feature_compare/...
    if out is None or str(out).strip() == "":
        ts_obj = yaml.safe_load(Path(task_spec).read_text(encoding="utf-8")) or {}
        task_id = ts_obj.get("task_id") if isinstance(ts_obj, dict) else None
        task_id = str(task_id).strip() if task_id else "TASK"
        poolb_stem = Path(poolb_yaml).stem
        base_out = Path("results") / "feature_compare" / f"{task_id}__{poolb_stem}"
        out_dir = base_out
        # Avoid clobbering: add suffix if already exists
        if out_dir.exists():
            for i in range(1, 1000):
                cand = Path(f"{str(base_out)}__{i}")
                if not cand.exists():
                    out_dir = cand
                    break
    else:
        out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Materialize a derived config and read its resolved required nodes list.
    tmp_cfg = out_dir / "_derived_cfg_for_compare"
    if tmp_cfg.exists():
        shutil.rmtree(tmp_cfg)

    materialize_nnmh_config_from_task_spec(
        task_spec_path=Path(task_spec),
        base_config_dir=Path(base_config),
        out_config_dir=tmp_cfg,
    )

    feat_yaml = (
        yaml.safe_load((tmp_cfg / "features.yaml").read_text(encoding="utf-8")) or {}
    )
    req_nodes = (
        (feat_yaml.get("feature_pipeline") or {})
        .get("requested_features", {})
        .get("required", [])
    )
    if not isinstance(req_nodes, list):
        req_nodes = []
    tier_required = sorted({str(x).strip() for x in req_nodes if str(x).strip()})

    # PoolB YAML: allow either plain list OR {feature_pipeline: {requested_features: [..]}}
    obj = yaml.safe_load(Path(poolb_yaml).read_text(encoding="utf-8")) or {}
    if isinstance(obj, list):
        poolb = obj
    else:
        poolb = (
            (obj.get("feature_pipeline") or {}).get("requested_features")
            or obj.get("requested_features")
            or []
        )
    if not isinstance(poolb, list):
        poolb = []
    poolb_raw = [str(x).strip() for x in poolb if str(x).strip()]

    # Normalize PoolB entries:
    # - If user provided output column names (e.g. "volume_profile_vpvr"), map them to feature funcs using feature_dependencies.yaml.
    deps_path = (PROJECT_ROOT / "config/feature_dependencies.yaml").resolve()
    deps_obj = yaml.safe_load(deps_path.read_text(encoding="utf-8")) or {}
    feats = deps_obj.get("features") if isinstance(deps_obj, dict) else None
    feats = feats if isinstance(feats, dict) else {}
    out2func = {}
    for func, meta in feats.items():
        cols = (meta or {}).get("output_columns") if isinstance(meta, dict) else None
        if isinstance(cols, list):
            for c in cols:
                out2func[str(c)] = str(func)

    poolb_norm = []
    for item in poolb_raw:
        if item in feats:
            # common alias convention: foo -> foo_f
            if (not item.endswith("_f")) and (f"{item}_f" in feats):
                poolb_norm.append(f"{item}_f")
            else:
                poolb_norm.append(item)
        elif item in out2func:
            poolb_norm.append(out2func[item])
        else:
            poolb_norm.append(item)
    poolb_nodes = sorted(set(poolb_norm))

    # Build tier membership map (so we can explain "only_poolb" items are actually in tier files but not enabled).
    # We do NOT assume tier2 is enabled; we just show where each node is declared.
    ts_obj = yaml.safe_load(Path(task_spec).read_text(encoding="utf-8")) or {}
    fp_ref = ts_obj.get("feature_plan_ref") if isinstance(ts_obj, dict) else None
    fp_overrides = (
        ts_obj.get("feature_plan_overrides") if isinstance(ts_obj, dict) else None
    )
    fp_overrides = fp_overrides if isinstance(fp_overrides, dict) else {}
    fp_obj = yaml.safe_load(Path(fp_ref).read_text(encoding="utf-8")) if fp_ref else {}
    fp = (fp_obj.get("feature_plan") if isinstance(fp_obj, dict) else None) or {}
    # apply shallow overrides for tier_feature_files/tiers_enabled
    if "tiers_enabled" in fp_overrides:
        fp["tiers_enabled"] = fp_overrides.get("tiers_enabled")
    if "tier_feature_files" in fp_overrides:
        fp["tier_feature_files"] = fp_overrides.get("tier_feature_files")

    tiers_enabled = fp.get("tiers_enabled") if isinstance(fp, dict) else None
    tiers_enabled = (
        [str(x) for x in tiers_enabled] if isinstance(tiers_enabled, list) else []
    )
    tier_files = fp.get("tier_feature_files") if isinstance(fp, dict) else None
    tier_files = tier_files if isinstance(tier_files, dict) else {}

    tier_membership: Dict[str, str] = {}
    for tier_name, rel_path in tier_files.items():
        try:
            items = (
                yaml.safe_load(
                    (PROJECT_ROOT / str(rel_path)).read_text(encoding="utf-8")
                )
                or []
            )
            if isinstance(items, list):
                for node in items:
                    node = str(node).strip()
                    if node and node not in tier_membership:
                        tier_membership[node] = str(tier_name)
        except Exception:
            continue

    s_tier = set(tier_required)
    s_poolb = set(poolb_nodes)
    overlap = sorted(s_tier & s_poolb)
    only_tier = sorted(s_tier - s_poolb)
    only_poolb = sorted(s_poolb - s_tier)

    summary = {
        "task_spec": str(Path(task_spec).resolve()),
        "base_config": str(Path(base_config).resolve()),
        "poolb_yaml": str(Path(poolb_yaml).resolve()),
        "derived_config_dir": str(tmp_cfg.resolve()),
        "tier_required_n": len(tier_required),
        "poolb_n": len(poolb_nodes),
        "overlap_n": len(overlap),
        "only_tier_n": len(only_tier),
        "only_poolb_n": len(only_poolb),
        "overlap": overlap,
        "only_tier": only_tier,
        "only_poolb": only_poolb,
        "tiers_enabled": tiers_enabled,
        "only_poolb_tier_hint": [
            {
                "node": n,
                "declared_in_tier": tier_membership.get(n),
                "tier_enabled": (
                    bool(tier_membership.get(n) in tiers_enabled)
                    if tier_membership.get(n)
                    else False
                ),
            }
            for n in only_poolb
        ],
    }
    (out_dir / "features_compare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = []
    md.append("# nnmultihead feature set compare\n")
    md.append(f"- task_spec: `{summary['task_spec']}`\n")
    md.append(f"- base_config: `{summary['base_config']}`\n")
    md.append(f"- poolb_yaml: `{summary['poolb_yaml']}`\n")
    md.append(f"- derived_config_dir: `{summary['derived_config_dir']}`\n")
    md.append("\n## counts\n")
    md.append(f"- tier_required_n: **{summary['tier_required_n']}**\n")
    md.append(f"- poolb_n: **{summary['poolb_n']}**\n")
    md.append(f"- overlap_n: **{summary['overlap_n']}**\n")
    md.append(f"- only_tier_n: **{summary['only_tier_n']}**\n")
    md.append(f"- only_poolb_n: **{summary['only_poolb_n']}**\n")
    md.append("\n## overlap\n")
    md.extend([f"- `{x}`\n" for x in overlap[:200]])
    if len(overlap) > 200:
        md.append(f"- ... ({len(overlap)-200} more)\n")
    md.append("\n## only in tier_required\n")
    md.extend([f"- `{x}`\n" for x in only_tier[:200]])
    if len(only_tier) > 200:
        md.append(f"- ... ({len(only_tier)-200} more)\n")
    md.append("\n## only in poolb\n")
    for x in only_poolb[:200]:
        tname = tier_membership.get(x)
        if tname:
            md.append(
                f"- `{x}`  (declared_in={tname}, enabled={str(tname in tiers_enabled).lower()})\n"
            )
        else:
            md.append(f"- `{x}`\n")
    if len(only_poolb) > 200:
        md.append(f"- ... ({len(only_poolb)-200} more)\n")
    (out_dir / "features_compare_summary.md").write_text("".join(md), encoding="utf-8")

    # A small UX hint for docker users (paths)
    _ = use_workspace_prefix

    click.echo(f"✅ Wrote: {out_dir / 'features_compare_summary.json'}")
    click.echo(f"✅ Wrote: {out_dir / 'features_compare_summary.md'}")


@nnmultihead.command("compare-runs")
@click.option(
    "--runs",
    required=True,
    help="Comma-separated run directories (produced by nnmultihead train/pipeline-3action-e2e), e.g. results/runs/runA,results/runs/runB",
)
@click.option(
    "--out",
    default=None,
    help="Output directory (default: results/compare/nnmh_runs/<timestamp>/)",
)
def nnmultihead_compare_runs(runs, out):
    """
    Compare nnmultihead run directories (model params + key metrics + e2e counterfactual sharpe/dd).

    Produces:
      - report.md (human-readable)
      - summary.json (machine-readable)
    """
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    def _as_abs(p: str) -> _Path:
        pp = _Path(p)
        if not pp.is_absolute():
            pp = (PROJECT_ROOT / pp).resolve()
        return pp

    run_dirs = [s.strip() for s in str(runs).split(",") if s.strip()]
    if len(run_dirs) < 2:
        raise click.ClickException("--runs must contain at least 2 run dirs")

    out_dir = _as_abs(
        out or f"results/compare/nnmh_runs/{_time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def _safe_read_json(p: _Path) -> dict:
        try:
            return _json.loads(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _find_latest_model_pt(root: _Path) -> _Path | None:
        candidates = list(root.glob("**/model.pt"))
        candidates = [p for p in candidates if "feature_store" not in p.parts]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _extract_model_meta(model_pt: _Path) -> dict:
        meta_p = model_pt.parent / "meta.json"
        if meta_p.exists():
            return _safe_read_json(meta_p)
        try:
            import torch  # type: ignore

            payload = torch.load(str(model_pt), map_location="cpu")
            if isinstance(payload, dict):
                return payload.get("meta") or {}
        except Exception:
            pass
        return {}

    def _extract_train_metrics(model_pt: _Path) -> dict:
        metrics_p = model_pt.parent / "metrics.json"
        return _safe_read_json(metrics_p) if metrics_p.exists() else {}

    def _extract_counterfactual_metrics(run_dir: _Path) -> dict:
        p = run_dir / "e2e" / "counterfactual" / "metrics.json"
        return _safe_read_json(p) if p.exists() else {}

    def _extract_thresholds(run_dir: _Path) -> dict:
        outp = {}
        p0 = run_dir / "router_thresholds_baseline.json"
        if p0.exists():
            outp["baseline"] = _safe_read_json(p0)
        p1 = run_dir / "threshold_plateau" / "router_thresholds_best.json"
        if p1.exists():
            outp["plateau_best"] = _safe_read_json(p1)
        p2 = run_dir / "threshold_plateau" / "summary.json"
        if p2.exists():
            outp["plateau_summary"] = _safe_read_json(p2)
        p3 = run_dir / "threshold_plateau" / "report.html"
        if p3.exists():
            outp["plateau_report_html"] = str(p3)
        return outp

    entries = []
    for rd in run_dirs:
        rdir = _as_abs(rd)
        if not rdir.exists():
            raise click.ClickException(f"run dir not found: {rdir}")
        model_pt = _find_latest_model_pt(rdir)
        if model_pt is None:
            raise click.ClickException(f"model.pt not found under: {rdir}")
        meta = _extract_model_meta(model_pt)
        train_metrics = _extract_train_metrics(model_pt)
        cf = _extract_counterfactual_metrics(rdir)
        th = _extract_thresholds(rdir)

        feature_cols = meta.get("feature_cols") or []
        train_cfg = meta.get("train_cfg") or {}

        entries.append(
            {
                "run_dir": str(rdir),
                "model_pt": str(model_pt),
                "feature_cols_n": (
                    int(len(feature_cols)) if isinstance(feature_cols, list) else None
                ),
                "train_cfg": {
                    k: train_cfg.get(k)
                    for k in [
                        "hidden",
                        "depth",
                        "dropout",
                        "batch_size",
                        "lr",
                        "epochs",
                        "seed",
                    ]
                },
                "task_id": meta.get("task_id"),
                "n_samples": meta.get("n_samples"),
                "train_metrics": train_metrics,
                "counterfactual": cf,
                "thresholds": th,
            }
        )

    summary = {
        "kind": "nnmultihead_run_compare_v1",
        "created_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "runs": entries,
    }
    (out_dir / "summary.json").write_text(
        _json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def _fmt(x):
        try:
            if x is None:
                return "null"
            if isinstance(x, float):
                return f"{x:.6g}"
            return str(x)
        except Exception:
            return str(x)

    def _pick(m: dict, k: str):
        return m.get(k) if isinstance(m, dict) else None

    md = []
    md.append("# nnmultihead run comparison\n\n")
    md.append(f"- out_dir: `{out_dir}`\n")
    md.append(f"- n_runs: **{len(entries)}**\n\n")

    md.append("## Summary table (counterfactual)\n\n")
    md.append(
        "| run | rule_sharpe_mean | pred_sharpe_mean | rule_avg_max_dd | pred_avg_max_dd | rule_avg_total_return | pred_avg_total_return | router_trade_n | router_trade_rate |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for e in entries:
        cf = e.get("counterfactual") or {}
        md.append(
            "| "
            + f"`{_Path(e['run_dir']).name}`"
            + " | "
            + " | ".join(
                _fmt(_pick(cf, k))
                for k in [
                    "rule_sharpe_mean",
                    "pred_sharpe_mean",
                    "rule_avg_max_dd",
                    "pred_avg_max_dd",
                    "rule_avg_total_return",
                    "pred_avg_total_return",
                    "router_diag__trade_n",
                    "router_diag__trade_rate",
                ]
            )
            + " |\n"
        )
    md.append("\n")

    md.append("## Model params / feature size\n\n")
    md.append(
        "| run | feature_cols_n | hidden | depth | dropout | batch_size | lr | epochs | seed |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for e in entries:
        tc = e.get("train_cfg") or {}
        md.append(
            "| "
            + f"`{_Path(e['run_dir']).name}`"
            + " | "
            + " | ".join(
                _fmt(x)
                for x in [
                    e.get("feature_cols_n"),
                    tc.get("hidden"),
                    tc.get("depth"),
                    tc.get("dropout"),
                    tc.get("batch_size"),
                    tc.get("lr"),
                    tc.get("epochs"),
                    tc.get("seed"),
                ]
            )
            + " |\n"
        )
    md.append("\n")

    md.append("## Threshold tuning (plateau)\n\n")
    for e in entries:
        rname = _Path(e["run_dir"]).name
        md.append(f"### `{rname}`\n\n")
        th = e.get("thresholds") or {}
        ps = th.get("plateau_summary") or {}
        if ps:
            md.append(
                f"- plateau_frac_ge_95pct: **{_fmt(ps.get('plateau_frac_ge_95pct'))}**\n"
            )
            md.append(
                f"- best.robust_score: {_fmt((ps.get('best') or {}).get('robust_score'))}\n"
            )
            if th.get("plateau_report_html"):
                md.append(f"- plateau report: `{th.get('plateau_report_html')}`\n")
        else:
            md.append("- (no threshold_plateau artifacts found)\n")
        md.append("\n")

    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")
    click.echo(f"✅ Wrote: {out_dir / 'report.md'}")
    click.echo(f"✅ Wrote: {out_dir / 'summary.json'}")


@nnmultihead.command("feature-group-search")
@click.option(
    "--task-spec",
    required=True,
    help="TaskSpec YAML (v1). REQUIRED: nnmultihead is TaskSpec-only (no legacy config mode).",
)
@click.option(
    "--base-config",
    default=None,
    help="Override base nnmultihead config dir (default: TaskSpec.model_plan.base_config_dir).",
)
@click.option(
    "--base-features-yaml",
    default=None,
    help="Optional base feature funcs YAML (Pool A). If omitted, will use Tier0 file from TaskSpec (recommended).",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def nnmultihead_feature_group_search(
    task_spec,
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
    import yaml

    ts_path = Path(task_spec)
    if not ts_path.is_absolute():
        ts_path = (PROJECT_ROOT / ts_path).resolve()
    ts_obj = yaml.safe_load(ts_path.read_text(encoding="utf-8")) or {}
    base_cfg = (
        str(base_config).strip()
        if base_config
        else str((ts_obj.get("model_plan") or {}).get("base_config_dir") or "").strip()
    ) or "config/nnmultihead/path_primitives_4h_80h_min"

    # Pool-A default: Tier0 file from TaskSpec (single-source-of-truth).
    if not str(base_features_yaml or "").strip():
        fp = ts_obj.get("feature_plan") or {}
        tier_files = fp.get("tier_feature_files") or {}
        tier0_path = (
            tier_files.get("TIER0_OHLCV_LIGHT")
            or tier_files.get("Tier0")
            or next(
                (v for k, v in tier_files.items() if "TIER0" in str(k).upper()), None
            )
        )
        if not tier0_path:
            raise click.ClickException(
                "TaskSpec missing Tier0 tier_feature_files entry. "
                "Set feature_plan.tier_feature_files.TIER0_OHLCV_LIGHT or pass --base-features-yaml."
            )
        base_features_yaml = str(tier0_path)

    args = [
        "--base-config",
        f"/workspace/{base_cfg}" if use_workspace_prefix else base_cfg,
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
    # Pool-A base feature nodes list
    if str(base_features_yaml or "").strip():
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


@train.command("rolling")
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", "-t", default="15T", help="Timeframe")
@click.option(
    "--config",
    "-c",
    default="config/strategies/tpc",
    help="Strategy config directory",
)
@click.option("--initial-train-months", default="3", help="Initial training months")
@click.option("--min-train-months", default="3", help="Minimum training months")
@click.option("--start", help="Rolling start date (YYYY-MM-DD)")
@click.option("--end", help="Rolling end date (YYYY-MM-DD)")
@click.option("--update-only", is_flag=True, help="Only update existing models")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    default="config/strategies/tpc",
    show_default=True,
    help="Strategy config directory",
)
@click.option("--start-date", required=True, help="Train start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="Train end date (YYYY-MM-DD)")
@click.option(
    "--holdout-start-date",
    default=None,
    help="OOS holdout start (YYYY-MM-DD). If set with --holdout-end-date, train on [start, holdout_start) and test on [holdout_start, holdout_end].",
)
@click.option(
    "--holdout-end-date",
    default=None,
    help="OOS holdout end (YYYY-MM-DD). Requires --holdout-start-date.",
)
@click.option(
    "--seed", default="42", show_default=True, help="Seed for reproducibility"
)
@click.option(
    "--output-root",
    default=None,
    help=(
        "Root dir for outputs. Default: results/train_final/<strategy>/train_final_<timestamp>_<label>/ "
        "(strategy from --config dirname; auto-generated). "
        "Legacy dirs results/<strategy>/train_final_* and results/train_final_* remain discoverable."
    ),
)
@click.option(
    "--data-path", default="data/parquet_data", show_default=True, help="Data directory"
)
@click.option("--feature-store-dir", default="feature_store", show_default=True)
@click.option("--feature-store-layer", default=None)
@click.option(
    "--deterministic/--non-deterministic", default=True, help="Deterministic training"
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
@click.option(
    "--labels",
    default=None,
    help="Override labels config file (e.g. config/strategies/tpc/labels_rr_extreme.yaml)",
)
@click.option(
    "--features",
    default=None,
    help="Override features config file (e.g. config/strategies/tpc/features_gate.yaml)",
)
@click.option(
    "--prepare-only",
    is_flag=True,
    default=False,
    help="Only run feature pipeline + label generation, save features_labeled.parquet. Skips model training.",
)
@click.option(
    "--archetype-prefilter",
    default=None,
    help="Path to archetypes/prefilter.yaml. Filters training data by archetype prerequisites before model training.",
)
@click.option(
    "--skip-gate-shap",
    is_flag=True,
    default=False,
    help="Skip TreeSHAP in risk_gate_draft statistical export (gain-only; faster iterations).",
)
def train_final(
    symbol,
    timeframe,
    config,
    start_date,
    end_date,
    holdout_start_date,
    holdout_end_date,
    seed,
    output_root,
    data_path,
    feature_store_dir,
    feature_store_layer,
    deterministic,
    docker,
    labels,
    features,
    prepare_only,
    archetype_prefilter,
    skip_gate_shap,
):
    """Train a final model and save ModelArtifact. With --holdout-*: train/test split by date (no overlap); without: train on full window (--train-all)."""
    from datetime import datetime
    from pathlib import Path

    # 自动生成带时间戳和 label 名称的输出目录（集中到 results/train_final/<策略>/ 便于清理）
    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _cfg = Path(config)
        strategy_key = _cfg.parent.name if _cfg.is_file() else _cfg.name
        # 提取 label 名称（如 labels_no_opportunity.yaml -> no_opportunity）
        label_suffix = ""
        if labels:
            label_name = Path(labels).stem  # e.g. "labels_no_opportunity"
            if label_name.startswith("labels_"):
                label_suffix = f"_{label_name[7:]}"  # 去掉 "labels_" 前缀
            elif label_name != "labels":
                label_suffix = f"_{label_name}"
        output_root = (
            f"results/train_final/{strategy_key}/"
            f"train_final_{timestamp}{label_suffix}"
        )
        click.echo(f"📂 Output directory: {output_root}")

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
    ]
    if holdout_start_date and holdout_end_date:
        args.extend(
            [
                "--holdout-start-date",
                str(holdout_start_date),
                "--holdout-end-date",
                str(holdout_end_date),
            ]
        )
    else:
        args.append("--train-all")
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
    if labels:
        args.extend(
            ["--labels", f"/workspace/{labels}" if use_workspace_prefix else labels]
        )
    if features:
        args.extend(
            [
                "--features",
                f"/workspace/{features}" if use_workspace_prefix else features,
            ]
        )
    if prepare_only:
        args.append("--prepare-only")
    if archetype_prefilter:
        args.extend(
            [
                "--archetype-prefilter",
                (
                    f"/workspace/{archetype_prefilter}"
                    if use_workspace_prefix
                    else archetype_prefilter
                ),
            ]
        )
    if skip_gate_shap:
        args.append("--skip-gate-shap")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
# Research Commands (layer-agnostic R&D stat kernels)
# =============================================================================


@cli.group()
def research():
    """Layer-agnostic research tools (scan / ic / plateau / fit / promote).

    Boundaries:
      mlbot research      — single-subject stat kernels (B + tree R&D)
      mlbot multileg research — multi-leg orchestrate
      mlbot pipeline run  — legacy bundle (ROUTINE_R&D_DEPRECATED for discovery)
      mlbot train final   — production ModelArtifact training
    """
    pass


def _research_forward(module: str, argv: list):
    import importlib

    mod = importlib.import_module(f"scripts.research.{module}")
    sys.exit(mod.main(argv))


_RESEARCH_CTX = {"ignore_unknown_options": True}


@research.command("scan", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_scan(args):
    """Label / condition scan (wraps scripts/research/scan.py)."""
    _research_forward("scan", list(args))


@research.command("ic", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_ic(args):
    """IC decay with horizon shift (wraps scripts/research/ic.py)."""
    _research_forward("ic", list(args))


@research.command("ic-prune", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_ic_prune(args):
    """Holdout IC prune for tree strategies (writeback requested_features)."""
    _research_forward("ic_prune", list(args))


@research.command("plateau", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_plateau(args):
    """Threshold plateau scan (wraps scripts/research/plateau.py)."""
    _research_forward("plateau", list(args))


@research.command("segment", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_segment(args):
    """Bucket-by segmented scan (wraps scripts/research/segment.py)."""
    _research_forward("segment", list(args))


@research.command("compare", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_compare(args):
    """Compare research JSON artifacts."""
    _research_forward("compare", list(args))


@research.command("robustness", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_robustness(args):
    """Temporal fold or gate robustness score (wraps scripts/research/robustness.py)."""
    _research_forward("robustness", list(args))


@research.command("calibrate", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_calibrate(args):
    """Write draft yaml from plateau json (no auto-promote)."""
    _research_forward("calibrate", list(args))


@research.command("fit", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_fit(args):
    """Exploratory LightGBM fit (any layer; not train final)."""
    _research_forward("fit", list(args))


@research.command("promote", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_promote(args):
    """Explicit promote draft yaml to archetypes (--yes required)."""
    _research_forward("promote", list(args))


@research.command("init")
@click.argument("topic")
@click.option("--strategy", default="tpc")
@click.option("--layers", default="regime", help="Layer hint for README (comma-separated)")
@click.option("--segment", default="recent_6m_oos")
@click.option("--force", is_flag=True, help="Overwrite existing experiment directory")
def research_init(topic, strategy, layers, segment, force):
    """Scaffold config/experiments/<topic>/ with rd_loop phase1 + monitor_bundle draft."""
    from scripts.research.init_experiment import init_experiment

    try:
        out = init_experiment(
            topic,
            strategy=strategy,
            layers=layers,
            segment=segment,
            force=force,
        )
    except FileExistsError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(3)
    click.echo(f"created {out}")
    click.echo(f"  Phase 1: PYTHONPATH=src:scripts python scripts/rd_loop.py \\")
    click.echo(f"    --hypothesis-yaml {out / f'rd_loop_{topic}_phase1.yaml'}")


@research.command("promote-baseline")
@click.option(
    "--experiment-dir",
    default=None,
    type=click.Path(),
    help="Experiment dir containing monitor_bundle/bundle.json",
)
@click.option("--strategy", default=None)
@click.option("--layer", default="regime")
@click.option(
    "--parquet",
    default=None,
    type=click.Path(),
    help="One-shot: export+promote from calibration parquet (no prior draft)",
)
@click.option("--enable-drift-ready", is_flag=True)
@click.option("--dry-run", is_flag=True)
def research_promote_baseline(
    experiment_dir, strategy, layer, parquet, enable_drift_ready, dry_run
):
    """Phase 5: write monitor_bundle to git baselines (watchdog / PSI / regime_shares)."""
    from scripts.research.promote_baseline import promote_baseline_main

    argv = []
    if experiment_dir:
        argv += ["--experiment-dir", experiment_dir]
    if strategy:
        argv += ["--strategy", strategy]
    if layer:
        argv += ["--layer", layer]
    if parquet:
        argv += ["--parquet", parquet]
    if enable_drift_ready:
        argv.append("--enable-drift-ready")
    if dry_run:
        argv.append("--dry-run")
    sys.exit(promote_baseline_main(argv))


# =============================================================================
# Monitor Commands (drift / regime health — local + remote)
# =============================================================================


@cli.group()
def monitor():
    """Drift monitoring for live and R&D (does not promote or edit archetypes).

    Authoritative guide: docs/strategy/漂移监控_mlbot_monitor_CN.md

    Boundaries:
      mlbot monitor     — regime_watchdog, plateau drift, contract checks
      mlbot multileg monitor — multi-leg C-system rolling health
      mlbot research    — hypothesis discovery (not cron monitoring)
    """
    pass


_MONITOR_CTX = {"ignore_unknown_options": True}


@monitor.command("watchdog", context_settings=_MONITOR_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def monitor_watchdog(args):
    """Regime / gate health, PSI, IC sign-flip (scripts/regime_watchdog.py)."""
    sys.exit(run_script("scripts/regime_watchdog.py", list(args)))


@monitor.command("drift", context_settings=_MONITOR_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def monitor_drift(args):
    """Regime plateau drift vs last_calibration (scripts/regime_drift_monitor.py)."""
    sys.exit(run_script("scripts/regime_drift_monitor.py", list(args)))


@monitor.command("contract", context_settings=_MONITOR_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def monitor_contract(args):
    """Pre-deploy contract + plateau stability (scripts/pre_deploy_contract_checks.py)."""
    sys.exit(run_script("scripts/pre_deploy_contract_checks.py", list(args)))


@monitor.command("segments")
@click.option(
    "--config",
    "config_path",
    default="config/market_segment.yaml",
    show_default=True,
    help="market_segment.yaml path",
)
def monitor_segments(config_path: str):
    """List canonical calendar segments (config/market_segment.yaml)."""
    from scripts.event_backtest.market_segment import load_market_segments

    root = get_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    segments = load_market_segments(path)
    click.echo(f"market_segment: {path} ({len(segments)} segments)\n")
    for sid in sorted(segments):
        row = segments[sid]
        click.echo(
            f"  {sid}: {row.get('start_date')} → {row.get('end_date')}  "
            f"[{row.get('label', '')}] {row.get('purpose', '')}"
        )


@monitor.command("export-window")
@click.option(
    "--bus-root",
    envvar="MLBOT_FEATURE_BUS_ROOT",
    default="live/shared_feature_bus",
    show_default=True,
)
@click.option("--timeframe", default="120T", show_default=True)
@click.option("--lookback-days", default=7, show_default=True, type=int)
@click.option("--symbols", default="", help="Comma-separated symbols (default: bus listing)")
@click.option("--output", required=True, help="Output parquet path")
def monitor_export_window(
    bus_root: str, timeframe: str, lookback_days: int, symbols: str, output: str
):
    """Export feature-bus window parquet (scripts/monitoring/export_feature_bus_window.py)."""
    args = [
        "--bus-root",
        bus_root,
        "--timeframe",
        timeframe,
        "--lookback-days",
        str(lookback_days),
        "--output",
        output,
    ]
    if str(symbols).strip():
        args.extend(["--symbols", str(symbols).strip()])
    sys.exit(run_script("scripts/monitoring/export_feature_bus_window.py", args))


@monitor.command("archive-batch")
@click.option("--strategy", default="tpc", show_default=True)
@click.option("--segment", default="recent_6m_oos", show_default=True)
@click.option(
    "--market-segment",
    default="config/market_segment.yaml",
    show_default=True,
)
@click.option("--output", required=True, help="Output parquet path")
@click.option("--symbol", default="BTCUSDT", show_default=True)
@click.option("--timeframe", default="120T", show_default=True)
def monitor_archive_batch(
    strategy: str,
    segment: str,
    market_segment: str,
    output: str,
    symbol: str,
    timeframe: str,
):
    """Long-window parquet via train final --prepare-only (archive_batch_window.py)."""
    sys.exit(
        run_script(
            "scripts/monitoring/archive_batch_window.py",
            [
                "--strategy",
                strategy,
                "--segment",
                segment,
                "--market-segment",
                market_segment,
                "--output",
                output,
                "--symbol",
                symbol,
                "--timeframe",
                timeframe,
            ],
        )
    )


@monitor.command("schedule")
@click.option(
    "--cadence",
    default="",
    help="weekly|monthly|quarterly|yearly (see config/monitoring/schedules.yaml)",
)
@click.option("--all", "run_all", is_flag=True, help="Run every cadence in schedules.yaml")
@click.option(
    "--schedules",
    default="config/monitoring/schedules.yaml",
    show_default=True,
)
@click.option("--run-ts", default="", help="Override {run_ts} in manifest paths")
@click.option("--dry-run", is_flag=True)
@click.option("--list", "list_only", is_flag=True, help="List cadence names")
def monitor_schedule(
    cadence: str, run_all: bool, schedules: str, run_ts: str, dry_run: bool, list_only: bool
):
    """Scheduled monitor: run manifest + update index.json + rd_registry.sqlite (CMS)."""
    args = ["--schedules", schedules]
    if list_only:
        args.append("--list")
    elif run_all:
        args.append("--all")
    else:
        if not str(cadence).strip():
            raise click.ClickException("Specify --cadence or --all (or --list)")
        args.extend(["--cadence", str(cadence).strip()])
    if str(run_ts).strip():
        args.extend(["--run-ts", str(run_ts).strip()])
    if dry_run:
        args.append("--dry-run")
    sys.exit(run_script("scripts/monitoring/monitor_scheduler.py", args))


@monitor.command("rebalance-check")
@click.option("--symbol", default="BTCUSDT", show_default=True)
@click.option("--window-days", default=7, show_default=True, type=int)
@click.option("--dry-run", is_flag=True)
@click.option("--skip-telegram", is_flag=True)
def monitor_rebalance_check(symbol: str, window_days: int, dry_run: bool, skip_telegram: bool):
    """Regime Cockpit NAV rebalance check → monitor_event + optional TG (T2d)."""
    args = ["--symbol", symbol, "--window-days", str(window_days)]
    if dry_run:
        args.append("--dry-run")
    if skip_telegram:
        args.append("--skip-telegram")
    sys.exit(run_script("scripts/monitoring/rebalance_cockpit_check.py", args))


@monitor.command("check-staleness")
@click.option(
    "--schedules",
    default="config/monitoring/schedules.yaml",
    show_default=True,
)
@click.option("--dry-run", is_flag=True, help="Report only, no Telegram")
def monitor_check_staleness(schedules: str, dry_run: bool):
    """Telegram alert when any cadence missed its expected run (缺勤)."""
    args = ["--schedules", schedules]
    if dry_run:
        args.append("--dry-run")
    sys.exit(run_script("scripts/monitoring/check_monitor_staleness.py", args))


@monitor.command("catalog")
@click.option(
    "--root",
    default="results",
    show_default=True,
    help="Search root for features_labeled.parquet",
)
@click.option(
    "--path",
    default="",
    help="Inspect a single parquet (skip discovery)",
)
@click.option(
    "--strategy",
    default="",
    help="Filter by strategy slug in path (e.g. tpc, bpc)",
)
@click.option(
    "--name",
    default="features_labeled.parquet",
    show_default=True,
    help="Filename to match",
)
@click.option("--limit", default=30, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True, help="JSON output with full metadata")
def monitor_catalog(
    root: str, path: str, strategy: str, name: str, limit: int, as_json: bool
):
    """List labeled parquet files + monitor metadata (rows, symbols, date range, IC cols)."""
    args = ["--root", root, "--name", name, "--limit", str(limit)]
    if str(path).strip():
        args.extend(["--path", str(path).strip()])
    if str(strategy).strip():
        args.extend(["--strategy", str(strategy).strip()])
    if as_json:
        args.append("--json")
    sys.exit(run_script("scripts/monitoring/catalog_labeled_parquets.py", args))


@monitor.command("run")
@click.option(
    "--config",
    "config_path",
    default="config/monitoring/weekly_rule_stack.yaml",
    show_default=True,
    help="Monitor manifest YAML",
)
@click.option("--run-ts", default="", help="Override {run_ts} in manifest paths")
@click.option("--dry-run", is_flag=True, help="Print steps without executing")
def monitor_run(config_path: str, run_ts: str, dry_run: bool):
    """Execute monitor manifest (export-window → archive-batch → watchdog → drift)."""
    args = ["--config", config_path]
    if str(run_ts).strip():
        args.extend(["--run-ts", str(run_ts).strip()])
    if dry_run:
        args.append("--dry-run")
    sys.exit(run_script("scripts/monitoring/run_monitor_manifest.py", args))


@monitor.command("weekly")
@click.option(
    "--window-parquet",
    envvar="WATCHDOG_PARQUET",
    default="",
    help="Short-window parquet for watchdog (env: WATCHDOG_PARQUET)",
)
@click.option(
    "--drift-parquet",
    envvar="DRIFT_PARQUET",
    default="",
    help="Long-window parquet for drift plateau (env: DRIFT_PARQUET)",
)
@click.option(
    "--manifest",
    default="",
    help="If set, run mlbot monitor run --config <manifest> instead of run_weekly.sh",
)
@click.option(
    "--auto-window/--no-auto-window",
    default=True,
    help="When parquets unset, export 7d bus + 6m archive-batch before watchdog/drift",
)
def monitor_weekly(
    window_parquet: str, drift_parquet: str, manifest: str, auto_window: bool
):
    """Weekly monitor: manifest run, or shell bundle with optional auto window production."""
    import os

    if str(manifest).strip():
        args = ["--config", str(manifest).strip()]
        sys.exit(run_script("scripts/monitoring/run_monitor_manifest.py", args))

    if str(window_parquet).strip():
        os.environ["WATCHDOG_PARQUET"] = str(window_parquet).strip()
    if str(drift_parquet).strip():
        os.environ["DRIFT_PARQUET"] = str(drift_parquet).strip()
    if auto_window:
        os.environ["MLBOT_MONITOR_AUTO_WINDOW"] = "1"
    sys.exit(run_script("scripts/monitoring/run_weekly.sh", []))


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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    default="config/strategies/tpc",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    """Time-series factor IC / win-rate evaluation (single asset).

    IC decay kernel also lives in ``mlbot research ic`` (horizon shift fixed in
    ``src/research/stat_kernels/ic.py``). This entry remains for Pool-B export flows.
    """
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
    default="config/strategies/tpc",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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


@analyze.command("archetype-performance")
@click.option("--logs", required=True, help="Input logs file (parquet)")
@click.option("--output", required=True, help="Output markdown report path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def analyze_archetype_performance(logs, output, docker):
    """Analyze archetype performance metrics."""
    args = [
        "--logs",
        f"/workspace/{logs}" if docker and not _is_in_docker() else logs,
        "--output",
        f"/workspace/{output}" if docker and not _is_in_docker() else output,
    ]
    sys.exit(
        run_script("scripts/analyze_archetype_performance.py", args, docker=docker)
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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


@analyze.command("gate-residual")
@click.option(
    "--model-dir",
    "-m",
    required=True,
    help="模型目录（如 results/train_final_xxx/bpc）",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.8,
    help="Gate 阈值（success_prob >= threshold 视为通过）",
)
@click.option(
    "--split",
    type=click.Choice(["train", "holdout", "all"]),
    default="holdout",
    help="数据集划分",
)
@click.option(
    "--direction",
    type=click.Choice(["long", "short"]),
    default="long",
    help="交易方向",
)
@click.option(
    "--horizon",
    type=int,
    default=50,
    help="持仓窗口（bars）",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="输出详细 CSV 路径（可选）",
)
def analyze_gate_residual(model_dir, threshold, split, direction, horizon, output):
    """
    Gate 剩余失败归因分析。

    分析 Gate 模型通过后剩余失败的特征分布，判断是否来自 Evidence/Execution 层。

    示例：
        mlbot analyze gate-residual \\
            --model-dir results/train_final_20260205_011545_rr_extreme/bpc \\
            --threshold 0.8 \\
            --split holdout
    """
    args = [
        "--model-dir",
        model_dir,
        "--threshold",
        str(threshold),
        "--split",
        split,
        "--direction",
        direction,
        "--horizon",
        str(horizon),
    ]

    if output:
        args.extend(["--output", output])

    sys.exit(
        run_script("scripts/analyze_gate_residual_failures.py", args, docker=False)
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
        args.extend(
            [
                "--out-json",
                f"/workspace/{out_json}" if use_workspace_prefix else out_json,
            ]
        )
    sys.exit(
        run_python_module(
            "src.time_series_model.diagnostics.kpi_gate_cli",
            args,
            docker=docker,
        )
    )


@diagnose.command("kpi-journal")
@click.option(
    "--run-dir", required=True, help="nnmultihead run dir (results/runs/<RUN_ID>)."
)
@click.option(
    "--stage",
    default="all",
    show_default=True,
    type=click.Choice(["all", "train", "pipeline", "threshold_plateau"]),
    help="Which KPI layer snapshot(s) to append (best-effort).",
)
@click.option(
    "--docker/--no-docker",
    default=True,
    help="Run in Docker (not required; this command reads local artifacts).",
)
def diagnose_kpi_journal(run_dir, stage, docker):
    """
    Append a KPI snapshot section into <run_dir>/kpi_journal.md and write kpi_latest.(md|json).

    This is a lightweight “single pane of glass” for evaluating improvements when you:
      - add features
      - change labels
      - tune router thresholds
    """
    _ = docker  # purely local; kept for CLI consistency
    from pathlib import Path

    from src.time_series_model.diagnostics.kpi_journal import write_kpi_journal

    p = Path(run_dir)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    outp = write_kpi_journal(run_dir=str(p), stage=str(stage))
    click.echo(f"✅ Wrote: {outp}")
    click.echo(f"✅ Wrote: {p / 'kpi_latest.md'}")
    click.echo(f"✅ Wrote: {p / 'kpi_latest.json'}")
    click.echo(f"✅ Wrote: {p / 'kpi_journal.html'}")
    click.echo(f"✅ Wrote: {p / 'kpi_latest.html'}")


@diagnose.command("evidence-quantiles")
@click.option("--feature-store-root", default="feature_store")
@click.option("--layer", required=True, help="FeatureStore layer id")
@click.option("--symbols", required=True, help="Comma-separated symbols")
@click.option("--timeframe", default="240T")
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
@click.option("--keys", default="vpin,cvd_change_5")
@click.option("--prefixes", default="")
@click.option("--quantiles", default="0.1,0.5,0.9")
@click.option("--out", required=True, help="Output JSON path")
@click.option("--global/--per-symbol", "global_pool", default=False)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_evidence_quantiles(
    feature_store_root,
    layer,
    symbols,
    timeframe,
    start_date,
    end_date,
    keys,
    prefixes,
    quantiles,
    out,
    global_pool,
    docker,
):
    """Build evidence_quantiles.json for execution evidence rules."""
    args = [
        "--feature-store-root",
        feature_store_root,
        "--layer",
        layer,
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--keys",
        keys,
        "--prefixes",
        prefixes,
        "--quantiles",
        quantiles,
        "--out",
        out,
    ]
    if global_pool:
        args.append("--global")
    sys.exit(run_script("scripts/build_evidence_quantiles.py", args, docker=docker))


@diagnose.command("evidence-quantiles-plateau")
@click.option("--feature-store-root", default="feature_store")
@click.option("--layer", required=True, help="FeatureStore layer id")
@click.option("--symbols", required=True, help="Comma-separated symbols")
@click.option("--timeframe", default="240T")
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
@click.option(
    "--registry",
    default="config/nnmultihead/execution_archetypes.yaml",
    help="Execution archetypes registry yaml",
)
@click.option("--archetype", required=True, help="Archetype to evaluate")
@click.option("--sweep-key", default="vpin")
@click.option("--q-grid", default="0.55,0.60,0.65,0.70,0.75,0.80")
@click.option("--quantiles", default="0.1,0.5,0.9")
@click.option("--quantiles-json", default=None)
@click.option("--logs", default=None)
@click.option("--out", required=True)
@click.option(
    "--gate-yaml",
    default="config/kpi_gates/nnmh_execution_layer.yaml",
    help="KPI gate yaml for auto selection",
)
@click.option("--require-gate/--no-require-gate", default=False)
@click.option("--plateau-frac", default=0.05, type=float)
@click.option("--score-key", default="gate_exec_score")
@click.option("--global/--per-symbol", "global_pool", default=False)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_evidence_quantiles_plateau(
    feature_store_root,
    layer,
    symbols,
    timeframe,
    start_date,
    end_date,
    registry,
    archetype,
    sweep_key,
    q_grid,
    quantiles,
    quantiles_json,
    logs,
    out,
    gate_yaml,
    require_gate,
    plateau_frac,
    score_key,
    global_pool,
    docker,
):
    """Plateau sweep for evidence quantile thresholds."""
    click.echo(
        "DEPRECATED (tree gate R&D): prefer mlbot research plateau; "
        "this diagnose command targets nnmultihead evidence quantiles.",
        err=True,
    )
    args = [
        "--feature-store-root",
        feature_store_root,
        "--layer",
        layer,
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--registry",
        registry,
        "--archetype",
        archetype,
        "--sweep-key",
        sweep_key,
        "--q-grid",
        q_grid,
        "--quantiles",
        quantiles,
        "--out",
        out,
        "--gate-yaml",
        gate_yaml,
        "--plateau-frac",
        str(plateau_frac),
        "--score-key",
        score_key,
    ]
    if quantiles_json:
        args.extend(["--quantiles-json", quantiles_json])
    if logs:
        args.extend(["--logs", logs])
    if require_gate:
        args.append("--require-gate")
    if global_pool:
        args.append("--global")
    sys.exit(
        run_script(
            "scripts/diagnose_evidence_quantiles_plateau.py", args, docker=docker
        )
    )


@diagnose.command("execution-gate-plateau")
@click.option("--feature-store-root", default="feature_store")
@click.option("--layer", required=True, help="FeatureStore layer id")
@click.option("--symbols", required=True, help="Comma-separated symbols")
@click.option("--timeframe", default="240T")
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
@click.option("--mode", required=True, help="mode_3action parquet/csv")
@click.option("--logs", required=True, help="logs_3action parquet/csv")
@click.option(
    "--registry",
    default="config/nnmultihead/execution_archetypes.yaml",
    help="Execution archetypes registry yaml",
)
@click.option(
    "--db-path",
    default="data/order_management.db",
    help="Order management DB path",
)
@click.option("--sweep-key", default="vpin")
@click.option("--q-grid", default="0.55,0.60,0.65,0.70,0.75,0.80")
@click.option("--quantiles", default="0.1,0.5,0.9")
@click.option("--evidence-quantiles", default=None)
@click.option(
    "--gate-yaml",
    default="config/kpi_gates/nnmh_execution_layer.yaml",
    help="KPI gate yaml for auto selection",
)
@click.option("--require-gate/--no-require-gate", default=False)
@click.option("--plateau-frac", default=0.05, type=float)
@click.option("--score-key", default="gate_exec_score")
@click.option("--out", required=True)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_execution_gate_plateau(
    feature_store_root,
    layer,
    symbols,
    timeframe,
    start_date,
    end_date,
    mode,
    logs,
    registry,
    db_path,
    sweep_key,
    q_grid,
    quantiles,
    evidence_quantiles,
    gate_yaml,
    require_gate,
    plateau_frac,
    score_key,
    out,
    docker,
):
    """Execution-layer plateau sweep (joint gate + execution KPIs)."""
    click.echo(
        "DEPRECATED (tree gate R&D): prefer mlbot research plateau; "
        "this diagnose command targets nnmultihead execution gate KPIs.",
        err=True,
    )
    args = [
        "--feature-store-root",
        feature_store_root,
        "--layer",
        layer,
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--mode",
        mode,
        "--logs",
        logs,
        "--registry",
        registry,
        "--db-path",
        db_path,
        "--sweep-key",
        sweep_key,
        "--q-grid",
        q_grid,
        "--quantiles",
        quantiles,
        "--out",
        out,
        "--gate-yaml",
        gate_yaml,
        "--plateau-frac",
        str(plateau_frac),
        "--score-key",
        score_key,
    ]
    if evidence_quantiles:
        args.extend(["--evidence-quantiles", evidence_quantiles])
    if require_gate:
        args.append("--require-gate")
    sys.exit(
        run_script("scripts/diagnose_execution_gate_plateau.py", args, docker=docker)
    )


@diagnose.command("execution-constraints-plateau")
@click.option("--logs", required=True, help="logs_3action parquet/csv")
@click.option("--min-interval-grid", default="0,60,120,240,360")
@click.option(
    "--gate-yaml",
    default="config/kpi_gates/nnmh_execution_layer.yaml",
    help="KPI gate yaml for auto selection",
)
@click.option("--require-gate/--no-require-gate", default=False)
@click.option("--plateau-frac", default=0.05, type=float)
@click.option("--score-key", default="gate_exec_score")
@click.option("--out", required=True)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_execution_constraints_plateau(
    logs,
    min_interval_grid,
    gate_yaml,
    require_gate,
    plateau_frac,
    score_key,
    out,
    docker,
):
    """Execution constraints plateau (min_order_interval only, proxy KPIs)."""
    click.echo(
        "DEPRECATED (tree gate R&D): prefer mlbot research plateau; "
        "this diagnose command targets nnmultihead execution constraints.",
        err=True,
    )
    args = [
        "--logs",
        logs,
        "--min-interval-grid",
        min_interval_grid,
        "--out",
        out,
        "--gate-yaml",
        gate_yaml,
        "--plateau-frac",
        str(plateau_frac),
        "--score-key",
        score_key,
    ]
    if require_gate:
        args.append("--require-gate")
    sys.exit(
        run_script(
            "scripts/diagnose_execution_constraints_plateau.py", args, docker=docker
        )
    )


@diagnose.command("archetype-trade-counts")
@click.option("--mode", required=True, help="mode_3action_gate parquet/csv")
@click.option("--out", required=True)
@click.option("--symbol-col", default="symbol")
@click.option("--timestamp-col", default="timestamp")
@click.option("--mode-col", default="mode")
@click.option("--archetype-col", default="gate_archetype")
@click.option("--gate-decision-col", default="gate_decision")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_archetype_trade_counts(
    mode,
    out,
    symbol_col,
    timestamp_col,
    mode_col,
    archetype_col,
    gate_decision_col,
    docker,
):
    """Count archetype trade entries from gated mode outputs."""
    args = [
        "--mode",
        mode,
        "--out",
        out,
        "--symbol-col",
        symbol_col,
        "--timestamp-col",
        timestamp_col,
        "--mode-col",
        mode_col,
        "--archetype-col",
        archetype_col,
        "--gate-decision-col",
        gate_decision_col,
    ]
    sys.exit(
        run_script("scripts/diagnose_archetype_trade_counts.py", args, docker=docker)
    )


@diagnose.command("execution-log-stages")
@click.option("--preds", required=True, help="preds file/dir (preds_*.parquet)")
@click.option("--mode", default=None, help="mode_3action file/dir (optional)")
@click.option("--logs", default=None, help="logs_3action file/dir (optional)")
@click.option("--out-dir", required=True, help="Output base dir for stage logs")
@click.option("--run-id", default=None)
@click.option("--timeframe", default=None)
@click.option("--strategy-name", default="pipeline")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_execution_log_stages(
    preds, mode, logs, out_dir, run_id, timeframe, strategy_name, docker
):
    """Build split-stage execution logs from pipeline outputs."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--preds",
        f"/workspace/{preds}" if use_workspace_prefix else preds,
        "--out-dir",
        f"/workspace/{out_dir}" if use_workspace_prefix else out_dir,
        "--strategy-name",
        str(strategy_name),
    ]
    if mode:
        args.extend(["--mode", f"/workspace/{mode}" if use_workspace_prefix else mode])
    if logs:
        args.extend(["--logs", f"/workspace/{logs}" if use_workspace_prefix else logs])
    if run_id:
        args.extend(["--run-id", str(run_id)])
    if timeframe:
        args.extend(["--timeframe", str(timeframe)])
    sys.exit(run_script("scripts/build_execution_log_stages.py", args, docker=docker))


@diagnose.command("execution-log-aggregate")
@click.option("--stage-dir", required=True, help="Base dir with stage subdirs")
@click.option("--out", required=True, help="Output canonical jsonl path")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_execution_log_aggregate(stage_dir, out, docker):
    """Aggregate stage logs into canonical execution log."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--stage-dir",
        f"/workspace/{stage_dir}" if use_workspace_prefix else stage_dir,
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
    ]
    sys.exit(
        run_script("scripts/aggregate_execution_log_stages.py", args, docker=docker)
    )


@diagnose.command("backtest-time-windows")
@click.option("--trades", required=True, help="Trades file (json/csv/parquet)")
@click.option("--out", required=True, help="Output JSON path")
@click.option("--entry-col", default=None)
@click.option("--exit-col", default=None)
@click.option("--symbol-col", default=None)
@click.option("--default-symbol", default=None)
@click.option("--pre-minutes", default=480, show_default=True)
@click.option("--post-minutes", default=480, show_default=True)
@click.option("--max-windows", default=None)
@click.option("--merge-overlap", is_flag=True, default=False)
@click.option("--merge-gap-minutes", default=0, show_default=True)
@click.option("--negative-ratio", default=0.0, show_default=True)
@click.option("--timeline-parquet", default=None)
@click.option("--timeline-ts-col", default="timestamp", show_default=True)
@click.option("--timeline-symbol-col", default="symbol", show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_backtest_time_windows(
    trades,
    out,
    entry_col,
    exit_col,
    symbol_col,
    default_symbol,
    pre_minutes,
    post_minutes,
    max_windows,
    merge_overlap,
    merge_gap_minutes,
    negative_ratio,
    timeline_parquet,
    timeline_ts_col,
    timeline_symbol_col,
    seed,
    docker,
):
    """Build time windows JSON for Nautilus event-sampling backtest."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--trades",
        f"/workspace/{trades}" if use_workspace_prefix else trades,
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
        "--pre-minutes",
        str(int(pre_minutes)),
        "--post-minutes",
        str(int(post_minutes)),
        "--merge-gap-minutes",
        str(int(merge_gap_minutes)),
        "--negative-ratio",
        str(float(negative_ratio)),
        "--seed",
        str(int(seed)),
    ]
    if entry_col:
        args.extend(["--entry-col", entry_col])
    if exit_col:
        args.extend(["--exit-col", exit_col])
    if symbol_col:
        args.extend(["--symbol-col", symbol_col])
    if default_symbol:
        args.extend(["--default-symbol", default_symbol])
    if max_windows is not None:
        args.extend(["--max-windows", str(int(max_windows))])
    if merge_overlap:
        args.append("--merge-overlap")
    if timeline_parquet:
        args.extend(
            [
                "--timeline-parquet",
                (
                    f"/workspace/{timeline_parquet}"
                    if use_workspace_prefix
                    else timeline_parquet
                ),
            ]
        )
    if timeline_ts_col:
        args.extend(["--timeline-ts-col", timeline_ts_col])
    if timeline_symbol_col:
        args.extend(["--timeline-symbol-col", timeline_symbol_col])
    sys.exit(run_script("scripts/build_backtest_time_windows.py", args, docker=docker))


@diagnose.command("export-vectorbt-trades")
@click.option("--artifacts-dir", default=None, help="Dir with backtest artifacts")
@click.option("--meta", default=None, help="Path to backtest_artifacts_meta.json")
@click.option("--df", default=None, help="Path to backtest_df_test.parquet")
@click.option("--preds", default=None, help="Path to backtest_preds.npy")
@click.option("--out", required=True, help="Output trades JSON path")
@click.option("--max-trades", default=None)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_export_vectorbt_trades(
    artifacts_dir,
    meta,
    df,
    preds,
    out,
    max_trades,
    docker,
):
    """Export vectorbt trades JSON from saved backtest artifacts."""
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
    ]
    if artifacts_dir:
        args.extend(
            [
                "--artifacts-dir",
                (
                    f"/workspace/{artifacts_dir}"
                    if use_workspace_prefix
                    else artifacts_dir
                ),
            ]
        )
    if meta:
        args.extend(["--meta", f"/workspace/{meta}" if use_workspace_prefix else meta])
    if df:
        args.extend(["--df", f"/workspace/{df}" if use_workspace_prefix else df])
    if preds:
        args.extend(
            ["--preds", f"/workspace/{preds}" if use_workspace_prefix else preds]
        )
    if max_trades is not None:
        args.extend(["--max-trades", str(int(max_trades))])
    sys.exit(run_script("scripts/export_vectorbt_trades.py", args, docker=docker))


@diagnose.command("threshold-plateau")
@click.option(
    "--preds",
    required=True,
    help="preds dir/file (preds_*.parquet) from nnmultihead predict or pipeline-3action-e2e.",
)
@click.option(
    "--logs",
    required=True,
    help="logs_3action.parquet (must contain ret_mean/ret_trend), typically from nnmultihead build-logs-3action.",
)
@click.option(
    "--model",
    required=True,
    help="model.pt (used to infer preds_in_log1p so threshold semantics match).",
)
@click.option(
    "--baseline-json",
    required=True,
    help="Baseline router thresholds JSON (7 keys: mfe_min,eff_min,dir_conf_trend_min,mfe_trend_min,ttm_trend_min,eff_mean_min,ttm_mean_max).",
)
@click.option(
    "--out",
    required=True,
    help="Output directory (will write candidates.csv/summary.json/report.md).",
)
@click.option("--n-candidates", default=300, type=int, show_default=True)
@click.option("--n-windows", default=6, type=int, show_default=True)
@click.option("--min-days-per-window", default=25, type=int, show_default=True)
@click.option("--n-bootstrap", default=30, type=int, show_default=True)
@click.option("--rel-sigma", default=0.05, type=float, show_default=True)
@click.option("--abs-sigma", default=0.01, type=float, show_default=True)
@click.option("--lambda", "lam", default=1.0, type=float, show_default=True)
@click.option("--mu", default=0.5, type=float, show_default=True)
@click.option("--entry-delay", default=0, type=int, show_default=True)
@click.option("--cost-per-turnover", default=0.0, type=float, show_default=True)
@click.option("--slippage-bps", default=0.0, type=float, show_default=True)
@click.option("--trade-rate-target", default=None, type=float, show_default=True)
@click.option("--trade-rate-tol", default=0.06, type=float, show_default=True)
@click.option("--trade-rate-min", default=None, type=float, show_default=True)
@click.option("--trade-rate-max", default=None, type=float, show_default=True)
@click.option("--trade-rate-penalty", default=1.5, type=float, show_default=True)
@click.option("--trend-rate-target", default=None, type=float, show_default=True)
@click.option("--trend-rate-tol", default=0.04, type=float, show_default=True)
@click.option("--trend-rate-min", default=0.10, type=float, show_default=True)
@click.option("--trend-rate-max", default=0.60, type=float, show_default=True)
@click.option("--trend-rate-penalty", default=1.0, type=float, show_default=True)
@click.option("--mean-rate-min", default=0.05, type=float, show_default=True)
@click.option("--mean-rate-max", default=0.40, type=float, show_default=True)
@click.option("--no-trade-rate-min", default=0.10, type=float, show_default=True)
@click.option("--no-trade-rate-max", default=0.70, type=float, show_default=True)
@click.option(
    "--disable-dist-rate-constraints",
    is_flag=True,
    default=False,
    show_default=True,
    help="Disable mean/no_trade distribution range constraints.",
)
@click.option("--trend-correct-horizon", default=24, type=int, show_default=True)
@click.option("--heuristic-bounds/--no-heuristic-bounds", default=False)
@click.option("--heuristic-qmin", default=0.05, type=float, show_default=True)
@click.option("--heuristic-qmax", default=0.95, type=float, show_default=True)
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_threshold_plateau(
    preds,
    logs,
    model,
    baseline_json,
    out,
    n_candidates,
    n_windows,
    min_days_per_window,
    n_bootstrap,
    rel_sigma,
    abs_sigma,
    lam,
    mu,
    entry_delay,
    cost_per_turnover,
    slippage_bps,
    trade_rate_target,
    trade_rate_tol,
    trade_rate_min,
    trade_rate_max,
    trade_rate_penalty,
    trend_rate_target,
    trend_rate_tol,
    trend_rate_min,
    trend_rate_max,
    trend_rate_penalty,
    mean_rate_min,
    mean_rate_max,
    no_trade_rate_min,
    no_trade_rate_max,
    disable_dist_rate_constraints,
    trend_correct_horizon,
    heuristic_bounds,
    heuristic_qmin,
    heuristic_qmax,
    seed,
    docker,
):
    """
    Threshold plateau tuning (Rule Router 3-action).

    This wraps `scripts/plateau_tune_rule_router_3action.py` and implements a robust
    "flat plateau" protocol (multi-window + bootstrap + local perturbations).

    Detailed guide: docs/architecture/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md
    """
    click.echo(
        "DEPRECATED (tree gate R&D): prefer mlbot research plateau; "
        "this diagnose command targets nnmultihead rule-router thresholds.",
        err=True,
    )
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--preds",
        f"/workspace/{preds}" if use_workspace_prefix else preds,
        "--logs",
        f"/workspace/{logs}" if use_workspace_prefix else logs,
        "--model",
        f"/workspace/{model}" if use_workspace_prefix else model,
        "--baseline-json",
        f"/workspace/{baseline_json}" if use_workspace_prefix else baseline_json,
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
        "--n-candidates",
        str(int(n_candidates)),
        "--n-windows",
        str(int(n_windows)),
        "--min-days-per-window",
        str(int(min_days_per_window)),
        "--n-bootstrap",
        str(int(n_bootstrap)),
        "--rel-sigma",
        str(float(rel_sigma)),
        "--abs-sigma",
        str(float(abs_sigma)),
        "--lambda",
        str(float(lam)),
        "--mu",
        str(float(mu)),
        "--entry-delay",
        str(int(entry_delay)),
        "--cost-per-turnover",
        str(float(cost_per_turnover)),
        "--slippage-bps",
        str(float(slippage_bps)),
        "--trade-rate-tol",
        str(float(trade_rate_tol)),
        "--trade-rate-penalty",
        str(float(trade_rate_penalty)),
        "--trend-rate-tol",
        str(float(trend_rate_tol)),
        "--trend-rate-penalty",
        str(float(trend_rate_penalty)),
        "--mean-rate-min",
        str(float(mean_rate_min)),
        "--mean-rate-max",
        str(float(mean_rate_max)),
        "--no-trade-rate-min",
        str(float(no_trade_rate_min)),
        "--no-trade-rate-max",
        str(float(no_trade_rate_max)),
        "--trend-correct-horizon",
        str(int(trend_correct_horizon)),
        "--heuristic-qmin",
        str(float(heuristic_qmin)),
        "--heuristic-qmax",
        str(float(heuristic_qmax)),
        "--seed",
        str(int(seed)),
    ]
    if trade_rate_target is not None:
        args.extend(["--trade-rate-target", str(float(trade_rate_target))])
    if trade_rate_min is not None:
        args.extend(["--trade-rate-min", str(float(trade_rate_min))])
    if trade_rate_max is not None:
        args.extend(["--trade-rate-max", str(float(trade_rate_max))])
    if trend_rate_target is not None:
        args.extend(["--trend-rate-target", str(float(trend_rate_target))])
    if trend_rate_min is not None:
        args.extend(["--trend-rate-min", str(float(trend_rate_min))])
    if trend_rate_max is not None:
        args.extend(["--trend-rate-max", str(float(trend_rate_max))])
    if heuristic_bounds:
        args.append("--heuristic-bounds")
    if disable_dist_rate_constraints:
        args.append("--disable-dist-rate-constraints")
    sys.exit(
        run_script("scripts/plateau_tune_rule_router_3action.py", args, docker=docker)
    )


@diagnose.command("extinction-replay-3action")
@click.option(
    "--logs",
    required=True,
    help="logs_3action.parquet (must contain symbol,timestamp,mode,ret_mean,ret_trend).",
)
@click.option(
    "--out",
    required=True,
    help="Output directory (report.json/sim.parquet/labels.parquet).",
)
@click.option(
    "--ood-config",
    default=None,  # OOD removed; safety handled by constitution only
    help="[DEPRECATED] OOD config YAML (optional, research only). Safety is handled by constitution/slots.",
)
@click.option(
    "--ood-score-col",
    default=None,
    help="Optional column name for ood_score (if present in logs).",
)
@click.option(
    "--survival-prob-col",
    default=None,
    help="Optional column name for survival_prob (if present in logs).",
)
@click.option("--survival-horizon-bars", default=50, type=int, show_default=True)
@click.option("--equity-floor-frac", default=0.5, type=float, show_default=True)
@click.option("--dd-floor", default=0.5, type=float, show_default=True)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_extinction_replay_3action(
    logs,
    out,
    ood_config,
    ood_score_col,
    survival_prob_col,
    survival_horizon_bars,
    equity_floor_frac,
    dd_floor,
    docker,
):
    """
    [DEPRECATED - Research only] Extinction replay runner for 3-action logs.
    OOD/survival removed; safety is handled by constitution/slots only.

    Produces:
      - report.json: extinction_rate/max_dd per symbol
      - sim.parquet: equity/drawdown/exposure timeline
      - labels.parquet: y_surv labels (for Survival Head training)
    """
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs}" if use_workspace_prefix else logs,
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
        "--survival-horizon-bars",
        str(int(survival_horizon_bars)),
        "--equity-floor-frac",
        str(float(equity_floor_frac)),
        "--dd-floor",
        str(float(dd_floor)),
    ]
    if ood_config:
        args.extend(
            [
                "--ood-config",
                f"/workspace/{ood_config}" if use_workspace_prefix else ood_config,
            ]
        )
    if ood_score_col:
        args.extend(["--ood-score-col", str(ood_score_col)])
    if survival_prob_col:
        args.extend(["--survival-prob-col", str(survival_prob_col)])
    sys.exit(run_script("scripts/extinction_replay_3action.py", args, docker=docker))


@diagnose.command("survival-head-train")  # [DEPRECATED - Research only]
@click.option(
    "--logs",
    required=True,
    help="logs_3action.parquet (from build-logs/run-e2e).",
)
@click.option(
    "--labels",
    required=True,
    help="labels.parquet (from diagnose extinction-replay-3action).",
)
@click.option(
    "--out",
    required=True,
    help="Output directory (model.pt/survival_preds.parquet/report.html).",
)
@click.option(
    "--config",
    "config_yaml",
    default=None,  # OOD removed; safety handled by constitution only
    show_default=True,
    help="Survival head config YAML.",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_survival_head_train(logs, labels, out, config_yaml, docker):
    """
    [DEPRECATED - Research only] Train Survival Head (tiny MLP) from extinction replay labels.
    OOD/survival removed; safety is handled by constitution/slots only.

    Produces model.pt + survival_preds.parquet + metrics/curves/report.
    """
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs}" if use_workspace_prefix else logs,
        "--labels",
        f"/workspace/{labels}" if use_workspace_prefix else labels,
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
        "--config",
        (
            f"/workspace/{config_yaml}"
            if use_workspace_prefix and config_yaml
            else (config_yaml or "")
        ),
    ]
    sys.exit(run_script("scripts/train_survival_head_mlp.py", args, docker=docker))


@diagnose.command("ood-to-archetype-weights")  # [DEPRECATED - Research only]
@click.option(
    "--logs",
    required=True,
    help="logs_3action.parquet (must contain ood_score + active_archetype).",
)
@click.option(
    "--labels",
    required=True,
    help="labels.parquet (y_surv) from extinction-replay-3action.",
)
@click.option(
    "--out", required=True, help="Output directory (survival_table.csv/weights.yaml)."
)
@click.option(
    "--config",
    "config_yaml",
    default=None,  # OOD removed; safety handled by constitution only
    show_default=True,
    help="Config YAML (bins/archetypes/temperature/min_samples).",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_ood_to_archetype_weights(logs, labels, out, config_yaml, docker):
    """
    [DEPRECATED - Research only] Learn OOD -> Archetype weights via Conditional Survival Table (baseline).
    OOD/survival removed; safety is handled by constitution/slots only.
    """
    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs}" if use_workspace_prefix else logs,
        "--labels",
        f"/workspace/{labels}" if use_workspace_prefix else labels,
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
    ]
    if config_yaml:
        args.extend(
            [
                "--config",
                f"/workspace/{config_yaml}" if use_workspace_prefix else config_yaml,
            ]
        )
    sys.exit(
        run_script(
            "scripts/learn_ood_to_archetype_weights_table.py", args, docker=docker
        )
    )


@diagnose.command("model-comparison")
@click.option(
    "--strategy-config",
    "-c",
    default="config/strategies/tpc",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
    default="bpc,me,fer,lv",
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


@diagnose.command("e2e-kpi")
@click.option("--logs", required=True, help="Input logs file (parquet)")
@click.option("--output-md", default=None, help="Output Markdown report path")
@click.option("--output-json", default=None, help="Output JSON report path")
@click.option("--ret-mean-col", default="ret_mean", help="Column name for mean returns")
@click.option(
    "--ret-trend-col", default="ret_trend", help="Column name for trend returns"
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_e2e_kpi(logs, output_md, output_json, ret_mean_col, ret_trend_col, docker):
    """Generate E2E KPI diagnostics report."""
    args = [
        "--logs",
        f"/workspace/{logs}" if docker and not _is_in_docker() else logs,
        "--ret-mean-col",
        ret_mean_col,
        "--ret-trend-col",
        ret_trend_col,
    ]
    if output_md:
        args.extend(
            [
                "--output-md",
                (
                    f"/workspace/{output_md}"
                    if docker and not _is_in_docker()
                    else output_md
                ),
            ]
        )
    if output_json:
        args.extend(
            [
                "--output-json",
                (
                    f"/workspace/{output_json}"
                    if docker and not _is_in_docker()
                    else output_json
                ),
            ]
        )
    sys.exit(run_script("scripts/diagnose_e2e_kpi.py", args, docker=docker))


@diagnose.command("pcm-performance")
@click.option("--logs", required=True, help="Input logs file (parquet)")
@click.option(
    "--baseline", default=None, help="Baseline logs file for comparison (optional)"
)
@click.option("--output", required=True, help="Output report path (markdown)")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_pcm_performance(logs, baseline, output, docker):
    """Diagnose PCM (Portfolio Capital Management) layer performance."""
    args = [
        "--logs",
        f"/workspace/{logs}" if docker and not _is_in_docker() else logs,
        "--output",
        f"/workspace/{output}" if docker and not _is_in_docker() else output,
    ]
    if baseline:
        args.extend(
            [
                "--baseline",
                (
                    f"/workspace/{baseline}"
                    if docker and not _is_in_docker()
                    else baseline
                ),
            ]
        )
    sys.exit(run_script("scripts/diagnose_pcm_performance.py", args, docker=docker))


@diagnose.command("production-attribution")
@click.option("--production-logs", required=True, help="Production logs file (parquet)")
@click.option("--baseline-logs", required=True, help="Baseline logs file (parquet)")
@click.option("--output-dir", required=True, help="Output directory for diagnostics")
@click.option(
    "--alert-thresholds",
    default='{"consecutive_losses": 5, "sharpe_drop": -0.5, "trade_count_drop": 0.2}',
    help="JSON string with alert thresholds",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_production_attribution(
    production_logs, baseline_logs, output_dir, alert_thresholds, docker
):
    """Comprehensive production attribution analysis across all layers."""
    args = [
        "--production-logs",
        (
            f"/workspace/{production_logs}"
            if docker and not _is_in_docker()
            else production_logs
        ),
        "--baseline-logs",
        (
            f"/workspace/{baseline_logs}"
            if docker and not _is_in_docker()
            else baseline_logs
        ),
        "--output-dir",
        f"/workspace/{output_dir}" if docker and not _is_in_docker() else output_dir,
        "--alert-thresholds",
        alert_thresholds,
    ]
    sys.exit(
        run_script("scripts/diagnose_production_attribution.py", args, docker=docker)
    )


@diagnose.command("outcome-attribution")
@click.option("--logs", required=True, help="Input logs file (parquet)")
@click.option(
    "--baseline", default=None, help="Baseline logs file for comparison (optional)"
)
@click.option("--output", required=True, help="Output report path (markdown)")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def diagnose_outcome_attribution(logs, baseline, output, docker):
    """Diagnose outcome/attribution layer performance."""
    args = [
        "--logs",
        f"/workspace/{logs}" if docker and not _is_in_docker() else logs,
        "--output",
        f"/workspace/{output}" if docker and not _is_in_docker() else output,
    ]
    if baseline:
        args.extend(
            [
                "--baseline",
                (
                    f"/workspace/{baseline}"
                    if docker and not _is_in_docker()
                    else baseline
                ),
            ]
        )
    sys.exit(run_script("scripts/diagnose_outcome_attribution.py", args, docker=docker))


@diagnose.command("export-fgs-shortlist")
@click.option(
    "--base-strategy-config",
    "-c",
    required=True,
    help="Base strategy config directory (single strategy), e.g. config/strategies/tpc",
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
    default="config/strategies/tpc",
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
def gate():
    """Gate (archetype filtering) commands."""
    pass


@gate.command("apply-archetype")
@click.option("--logs", required=True, help="Input logs file (parquet)")
@click.option(
    "--out",
    default=None,
    help="Output gated logs file (auto: latest train dir if not specified)",
)
@click.option(
    "--features-store-layer",
    default=None,
    help="FeatureStore layer name (auto-detect latest if not specified)",
)
@click.option(
    "--features-store-root", default="feature_store", help="FeatureStore root directory"
)
@click.option(
    "--strategy",
    default=None,
    help="Strategy name (e.g., bpc, htf). Loads from config/strategies/{strategy}/archetypes/",
)
@click.option(
    "--strategies-root",
    default="config/strategies",
    help="Root directory for strategy configs",
)
@click.option(
    "--gate-path",
    default=None,
    help="Custom gate YAML path (e.g., config/strategies/fer/gate_draft.yaml)",
)
@click.option(
    "--docker/--no-docker", default=False, help="Run in Docker (default: no-docker)"
)
def gate_apply_archetype(
    logs,
    out,
    features_store_layer,
    features_store_root,
    strategy,
    strategies_root,
    gate_path,
    docker,
):
    """Apply archetype gate rules to logs."""
    from src.feature_store.layer_naming import (
        detect_layer_for_strategy,
        detect_layer_timeframe,
    )

    # --- 尝试从 predictions 旁边的 metadata 获取 timeframe ---
    _inferred_timeframe: str | None = None
    try:
        from pathlib import Path as _P
        import json as _json

        _meta_path = _P(logs).parent / "model_artifact_metadata.json"
        if _meta_path.exists():
            with open(_meta_path) as _f:
                _inferred_timeframe = _json.load(_f).get("timeframe")
    except Exception:
        pass

    # --- Fallback: 从策略 meta.yaml 读取 timeframe ---
    if not _inferred_timeframe and strategy:
        try:
            import re as _re
            from pathlib import Path as _P2

            for _root in (strategies_root or "config/strategies",):
                _meta_yaml = _P2(_root) / strategy / "meta.yaml"
                if _meta_yaml.exists():
                    _tf_match = _re.search(
                        r'timeframe:\s*["\']?([\w]+)["\']?',
                        _meta_yaml.read_text(),
                    )
                    if _tf_match:
                        _inferred_timeframe = _tf_match.group(1)
                        break
        except Exception:
            pass

    # 自动检测匹配 strategy (+timeframe) 的最新 feature store layer
    if features_store_layer is None:
        features_store_layer = detect_layer_for_strategy(
            strategy=strategy,
            features_store_root=features_store_root,
            timeframe=_inferred_timeframe,
        )
        if features_store_layer:
            click.echo(
                f"ℹ️ Auto-detected feature store layer for {strategy or 'all'}: {features_store_layer}"
            )
        else:
            click.echo(
                f"❌ No feature store layers found in {features_store_root}", err=True
            )
            sys.exit(1)

    # 从 FS meta 读取 timeframe（优先用 metadata 推断结果）
    _layer_tf = detect_layer_timeframe(features_store_layer, features_store_root)
    timeframe = _inferred_timeframe or _layer_tf or "240T"
    if _layer_tf:
        click.echo(f"ℹ️ Timeframe from FS layer meta: {_layer_tf}")

    # 自动检测输出目录：放到最新的训练结果目录下
    if out is None:
        from pathlib import Path

        results_dir = Path("results")
        if results_dir.exists():
            # 查找匹配 strategy 的最新训练目录
            # 现行: results/train_final/<strategy>/train_final_*/<strategy>/
            # 仍兼容: results/<strategy>/train_final_*/<strategy>/
            # 旧布局: results/train_final_*/<strategy>/
            train_dirs = []
            for d in results_dir.glob("train_final_*"):
                if d.is_dir():
                    if strategy:
                        strategy_dir = d / strategy
                        if strategy_dir.is_dir():
                            train_dirs.append(strategy_dir)
                    else:
                        train_dirs.append(d)
            if strategy:
                bucket = results_dir / "train_final" / strategy
                if bucket.is_dir():
                    for d in bucket.glob("train_final_*"):
                        if d.is_dir():
                            sd = d / strategy
                            if sd.is_dir():
                                train_dirs.append(sd)
                strat_root = results_dir / strategy
                if strat_root.is_dir():
                    for d in strat_root.glob("train_final_*"):
                        if d.is_dir():
                            sd = d / strategy
                            if sd.is_dir():
                                train_dirs.append(sd)

            if train_dirs:
                latest_train = max(train_dirs, key=lambda p: p.stat().st_mtime)
                out = str(latest_train / "logs_gated.parquet")
                click.echo(f"ℹ️ Auto output to: {out}")
            else:
                out = "results/logs_gated.parquet"
                click.echo(f"⚠️ No train dir found, output to: {out}")
        else:
            out = "results/logs_gated.parquet"
            click.echo(f"⚠️ Results dir not found, output to: {out}")

    use_workspace_prefix = docker and not _is_in_docker()
    args = [
        "--logs",
        f"/workspace/{logs}" if use_workspace_prefix else logs,
        "--out",
        f"/workspace/{out}" if use_workspace_prefix else out,
        "--features-store-layer",
        features_store_layer,
        "--features-store-root",
        (
            f"/workspace/{features_store_root}"
            if use_workspace_prefix
            else features_store_root
        ),
        "--strategies-root",
        f"/workspace/{strategies_root}" if use_workspace_prefix else strategies_root,
        "--timeframe",
        timeframe,
    ]
    if strategy:
        args.extend(["--strategy", strategy])
    if gate_path:
        args.extend(
            [
                "--gate-path",
                f"/workspace/{gate_path}" if use_workspace_prefix else gate_path,
            ]
        )
    sys.exit(run_script("scripts/apply_archetype_gate.py", args, docker=docker))


@cli.group()
def optimize():
    """Optimization commands."""
    pass


@optimize.command("gate-plateau")
@click.option(
    "--archetype", required=True, help="Archetype name (e.g., TrendContinuationTC)"
)
@click.option("--rule-name", required=True, help="Gate rule name to optimize")
@click.option("--gated-logs", required=True, help="Gated logs file (parquet)")
@click.option("--raw-logs", default=None, help="Raw logs file (parquet, optional)")
@click.option(
    "--output", required=True, help="Output JSON path for optimization results"
)
@click.option(
    "--min-trade-rate", type=float, default=0.005, help="Minimum trade rate threshold"
)
@click.option(
    "--min-trades-per-bucket", type=int, default=10, help="Minimum trades per bucket"
)
@click.option(
    "--min-sharpe-threshold",
    type=float,
    default=0.5,
    help="Minimum Sharpe threshold for plateau",
)
@click.option(
    "--threshold-step",
    type=float,
    default=0.05,
    help="Threshold step size for scanning",
)
@click.option(
    "--execution-archetypes",
    default="config/nnmultihead/execution_archetypes.yaml",
    help="Path to execution_archetypes.yaml",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def optimize_gate_plateau(
    archetype,
    rule_name,
    gated_logs,
    raw_logs,
    output,
    min_trade_rate,
    min_trades_per_bucket,
    min_sharpe_threshold,
    threshold_step,
    execution_archetypes,
    docker,
):
    """Optimize a single gate rule threshold using plateau method.

    DEPRECATED: scripts/optimize_gate_plateau.py removed.
    Use: mlbot research plateau ... OR python scripts/optimize_gate_unified.py
    """
    click.echo(
        "DEPRECATED: mlbot optimize gate-plateau → use "
        "'mlbot research plateau' or 'scripts/optimize_gate_unified.py'",
        err=True,
    )
    sys.exit(2)
    args = [
        "--archetype",
        archetype,
        "--rule-name",
        rule_name,
        "--gated-logs",
        f"/workspace/{gated_logs}" if docker and not _is_in_docker() else gated_logs,
        "--output",
        f"/workspace/{output}" if docker and not _is_in_docker() else output,
        "--min-trade-rate",
        str(min_trade_rate),
        "--min-trades-per-bucket",
        str(min_trades_per_bucket),
        "--min-sharpe-threshold",
        str(min_sharpe_threshold),
        "--threshold-step",
        str(threshold_step),
        "--execution-archetypes",
        (
            f"/workspace/{execution_archetypes}"
            if docker and not _is_in_docker()
            else execution_archetypes
        ),
    ]
    if raw_logs:
        args.extend(
            [
                "--raw-logs",
                (
                    f"/workspace/{raw_logs}"
                    if docker and not _is_in_docker()
                    else raw_logs
                ),
            ]
        )
    sys.exit(run_script("scripts/optimize_gate_plateau.py", args, docker=docker))


@optimize.command("gate-plateau-all")
@click.option("--gated-logs", required=True, help="Gated logs file (parquet)")
@click.option("--raw-logs", default=None, help="Raw logs file (parquet, optional)")
@click.option(
    "--output-dir", required=True, help="Output directory for optimization results"
)
@click.option(
    "--min-trade-rate", type=float, default=0.005, help="Minimum trade rate threshold"
)
@click.option(
    "--min-trades-per-bucket", type=int, default=10, help="Minimum trades per bucket"
)
@click.option(
    "--execution-archetypes",
    default="config/nnmultihead/execution_archetypes.yaml",
    help="Path to execution_archetypes.yaml",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def optimize_gate_plateau_all(
    gated_logs,
    raw_logs,
    output_dir,
    min_trade_rate,
    min_trades_per_bucket,
    execution_archetypes,
    docker,
):
    """Optimize all gate rules for all archetypes using plateau method."""
    click.echo(
        "DEPRECATED: mlbot optimize gate-plateau-all → use "
        "'scripts/optimize_gate_unified.py' per strategy",
        err=True,
    )
    sys.exit(2)
    args = [
        "--gated-logs",
        f"/workspace/{gated_logs}" if docker and not _is_in_docker() else gated_logs,
        "--output-dir",
        f"/workspace/{output_dir}" if docker and not _is_in_docker() else output_dir,
        "--min-trade-rate",
        str(min_trade_rate),
        "--min-trades-per-bucket",
        str(min_trades_per_bucket),
        "--execution-archetypes",
        (
            f"/workspace/{execution_archetypes}"
            if docker and not _is_in_docker()
            else execution_archetypes
        ),
    ]
    if raw_logs:
        args.extend(
            [
                "--raw-logs",
                (
                    f"/workspace/{raw_logs}"
                    if docker and not _is_in_docker()
                    else raw_logs
                ),
            ]
        )
    sys.exit(
        run_script("scripts/optimize_all_archetypes_plateau.py", args, docker=docker)
    )


@optimize.command("gate-experiments")
@click.option("--gated-logs", required=True, help="Gated logs file (parquet)")
@click.option("--raw-logs", required=True, help="Raw logs file (parquet)")
@click.option(
    "--execution-archetypes",
    default="config/nnmultihead/execution_archetypes.yaml",
    help="execution_archetypes.yaml path",
)
@click.option("--output-dir", required=True, help="Output directory")
@click.option(
    "--feature-store-root", default="feature_store", help="FeatureStore root directory"
)
@click.option("--feature-store-layer", default=None, help="FeatureStore layer name")
@click.option("--timeframe", default="240T", help="Timeframe (e.g., 240T)")
@click.option("--start-date", default=None, help="Start date (optional)")
@click.option("--end-date", default=None, help="End date (optional)")
@click.option(
    "--experiments",
    multiple=True,
    default=["all"],
    help="Experiments to run (baseline, progressive, hard_gate, all)",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def optimize_gate_experiments(
    gated_logs,
    raw_logs,
    execution_archetypes,
    output_dir,
    feature_store_root,
    feature_store_layer,
    timeframe,
    start_date,
    end_date,
    experiments,
    docker,
):
    """Run gate optimization experiments (baseline, progressive, hard-gate)."""
    args = [
        "--gated-logs",
        f"/workspace/{gated_logs}" if docker and not _is_in_docker() else gated_logs,
        "--raw-logs",
        f"/workspace/{raw_logs}" if docker and not _is_in_docker() else raw_logs,
        "--execution-archetypes",
        (
            f"/workspace/{execution_archetypes}"
            if docker and not _is_in_docker()
            else execution_archetypes
        ),
        "--output-dir",
        f"/workspace/{output_dir}" if docker and not _is_in_docker() else output_dir,
    ]
    if feature_store_layer:
        args.extend(
            [
                "--feature-store-root",
                feature_store_root,
                "--feature-store-layer",
                feature_store_layer,
                "--timeframe",
                timeframe,
            ]
        )
        if start_date:
            args.extend(["--start-date", start_date])
        if end_date:
            args.extend(["--end-date", end_date])
    if experiments:
        args.extend(["--experiments"] + list(experiments))
    sys.exit(
        run_script("scripts/run_gate_optimization_experiments.py", args, docker=docker)
    )


@optimize.command("gate-compare")
@click.option("--results-file", required=True, help="Experiment results JSON file")
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def optimize_gate_compare(
    results_file,
    output_dir,
    docker,
):
    """Compare gate optimization experiment results and generate report."""
    args = [
        "--results-file",
        (
            f"/workspace/{results_file}"
            if docker and not _is_in_docker()
            else results_file
        ),
        "--output-dir",
        f"/workspace/{output_dir}" if docker and not _is_in_docker() else output_dir,
    ]
    sys.exit(
        run_script(
            "scripts/compare_gate_optimization_experiments.py", args, docker=docker
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
):
    """Run strategy backtest with trained model (train+backtest via pipeline)."""
    use_workspace_prefix = False  # CLI typically run with --no-docker
    from pathlib import Path as _Path

    from src.config.strategy_layout import strategy_packaged_root

    root = _Path.cwd().resolve()
    pkg = strategy_packaged_root(root, str(strategy))
    cfg_arg = pkg.relative_to(root).as_posix()
    args = [
        "--config",
        (f"/workspace/{cfg_arg}" if use_workspace_prefix else cfg_arg),
        "--symbol",
        symbol,
        "--data-path",
        data_path or "data/parquet_data",
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--output-root",
        output_dir,
    ]
    # pipeline 暂无 --model-path；仅用日期范围做 train+backtest
    sys.exit(run_script("scripts/train_strategy_pipeline.py", args, docker=False))


# NOTE: `mlbot backtest visualize` removed — trading map now integrated into
# backtest_execution_layer.py (_generate_trading_map_html)


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
@click.option(
    "--strategy-config",
    default=None,
    help="Strategy config dir (e.g. config/strategies/tpc). When set, run feature pipeline so Hurst/Hilbert/WPT/Spectral etc. appear in the report.",
)
@click.option(
    "--feature-store-dir",
    default="feature_store",
    help="FeatureStore root when using --strategy-config",
)
@click.option(
    "--use-cache/--no-cache",
    default=False,
    help="Use FeatureStore cache (default: compute fresh)",
)
@click.option(
    "--force-rebuild",
    is_flag=True,
    default=False,
    help="Force rebuild FeatureStore cache (requires --use-cache)",
)
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
def visualize_feature_indicators(
    symbol,
    timeframe,
    start_date,
    end_date,
    config,
    output_dir,
    strategy_config,
    feature_store_dir,
    use_cache,
    force_rebuild,
    docker,
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
    if strategy_config:
        args.extend(["--strategy-config", strategy_config])
        args.extend(["--feature-store-dir", feature_store_dir])
        if use_cache:
            args.append("--use-cache")
        if force_rebuild:
            args.append("--force-rebuild")

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
@click.option("--docker/--no-docker", default=False, help="Run in Docker")
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
