"""Atomic parquet writes and safe reads for live storage / feature bus."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

MIN_PARQUET_BYTES = 8

_logger = logging.getLogger(__name__)


def is_unreadable_parquet(path: Path) -> bool:
    """True if path is missing, too small, or cannot be read as parquet."""
    if not path.is_file():
        return False
    try:
        if path.stat().st_size < MIN_PARQUET_BYTES:
            return True
        pd.read_parquet(path)
        return False
    except Exception:
        return True


def quarantine_corrupt_parquet(
    path: Path,
    *,
    logger: Optional[logging.Logger] = None,
    reason: str = "unreadable parquet",
) -> bool:
    """Remove a corrupt parquet file. Returns True if a file was removed."""
    log = logger or _logger
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    try:
        path.unlink()
    except OSError as exc:
        log.warning("parquet quarantine failed path=%s: %s", path, exc)
        return False
    log.warning(
        "parquet quarantine: removed corrupt file path=%s size=%s reason=%s",
        path,
        size,
        reason,
    )
    return True


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically via temp file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_parquet_safe(
    path: Path,
    *,
    empty: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Read parquet or quarantine and return empty frame on failure."""
    path = Path(path)
    if not path.exists():
        return empty.copy() if empty is not None else pd.DataFrame()
    if is_unreadable_parquet(path):
        quarantine_corrupt_parquet(path, logger=logger, reason="read failed")
        return empty.copy() if empty is not None else pd.DataFrame()
    return pd.read_parquet(path)
