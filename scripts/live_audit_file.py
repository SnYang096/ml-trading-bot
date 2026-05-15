"""Shared rotating file audit logging for ``run_live`` / ``run_multi_leg_live``.

Attaches ``TimedRotatingFileHandler`` to the root logger (once per resolved path).
"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_audit_logger = logging.getLogger(__name__)

_ATTACHED: set[str] = set()


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def retention_days_from_env(env_name: str, default: int = 30) -> int:
    raw = os.getenv(env_name, str(default)).strip()
    try:
        v = int(raw)
    except ValueError:
        return default
    if v <= 0:
        return default
    return min(v, 500)


def prune_rotated_audit_files(
    log_dir: Path, log_filename: str, max_age_days: int
) -> None:
    """Remove ``{log_filename}`` and ``{log_filename}.*`` older than max_age_days by mtime."""
    if max_age_days <= 0 or not log_dir.is_dir():
        return
    cutoff = time.time() - float(max_age_days) * 86400.0
    stem = Path(log_filename).name
    for path in log_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name != stem and not name.startswith(f"{stem}."):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def attach_timed_rotating_audit(
    *,
    log_file: Path,
    retention_days: int,
    banner: str,
) -> None:
    key = str(log_file.resolve())
    if key in _ATTACHED:
        return

    log_dir = log_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    prune_rotated_audit_files(log_dir, log_file.name, retention_days)

    backup_count = max(1, retention_days)
    fh = TimedRotatingFileHandler(
        str(log_file),
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8",
        utc=False,
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(fh)
    _ATTACHED.add(key)
    _audit_logger.info(
        "%s: path=%s rotation=daily backupCount=%d retention_days=%d",
        banner,
        log_file,
        backup_count,
        retention_days,
    )


def configure_audit_from_env_defaults(
    *,
    default_log_file: Path,
    disable_env: str,
    path_env: str,
    retention_env: str,
    banner: str,
) -> None:
    """If not disabled via env, attach audit file (``default_log_file`` when path env unset)."""
    if _truthy_env(disable_env):
        _audit_logger.info("audit file disabled (%s)", disable_env)
        return

    raw = os.getenv(path_env, "").strip()
    if raw.lower() in ("0", "off", "false", "no"):
        _audit_logger.info("audit file disabled (%s)", path_env)
        return

    if not raw or raw.lower() == "default":
        log_file = default_log_file
    else:
        log_file = Path(raw)

    retention = retention_days_from_env(retention_env, 30)
    attach_timed_rotating_audit(
        log_file=log_file,
        retention_days=retention,
        banner=banner,
    )
