"""Shared rotating file audit logging for ``run_live`` / ``run_multi_leg_live``.

Attaches ``TimedRotatingFileHandler`` to the root logger (once per resolved path).

Rotation defaults to **hourly**; set env ``MLBOT_AUDIT_ROTATION=day`` (or the
per-runner ``*_AUDIT_ROTATION``) for legacy daily (midnight) rollover.
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


def _rotation_style_from_env(
    rotation_env: str,
    *,
    global_fallback: str = "MLBOT_AUDIT_ROTATION",
) -> tuple[str, int]:
    """Return (TimedRotatingFileHandler *when*, *interval*) for hourly vs daily rollover."""
    raw = os.getenv(rotation_env, "").strip().lower()
    if not raw:
        raw = os.getenv(global_fallback, "hour").strip().lower()
    if raw in ("day", "daily", "midnight", "d"):
        return "midnight", 1
    # default: hourly
    return "H", 1


def _backup_count_for_rotation(retention_days: int, *, when: str, interval: int) -> int:
    """``backupCount`` limits rotated archives kept by the handler (plus prune on startup)."""
    if when == "midnight" and interval >= 1:
        return max(1, retention_days)
    # hourly: keep roughly one file per hour for retention_days
    return max(1, min(retention_days * 24, 12000))


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
    rotation_env: str,
) -> None:
    key = str(log_file.resolve())
    if key in _ATTACHED:
        return

    log_dir = log_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    prune_rotated_audit_files(log_dir, log_file.name, retention_days)

    when, interval = _rotation_style_from_env(rotation_env)
    backup_count = _backup_count_for_rotation(
        retention_days, when=when, interval=interval
    )
    fh = TimedRotatingFileHandler(
        str(log_file),
        when=when,
        interval=interval,
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
    rot_desc = "hourly" if when == "H" else "daily_midnight"
    _audit_logger.info(
        "%s: path=%s rotation=%s when=%s interval=%s backupCount=%d retention_days=%d",
        banner,
        log_file,
        rot_desc,
        when,
        interval,
        backup_count,
        retention_days,
    )


def configure_audit_from_env_defaults(
    *,
    default_log_file: Path,
    disable_env: str,
    path_env: str,
    retention_env: str,
    rotation_env: str,
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
        rotation_env=rotation_env,
    )
