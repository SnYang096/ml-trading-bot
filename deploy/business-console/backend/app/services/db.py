"""Read-only SQLite helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Read-only open for live DBs on Docker :ro mounts (no WAL journal creation)."""
    resolved = path.resolve()
    attempts = (
        f"file:{resolved.as_posix()}?mode=ro&immutable=1",
        f"file:{resolved.as_posix()}?mode=ro&nolock=1",
        f"file:{resolved.as_posix()}?mode=ro",
    )
    last_err: sqlite3.OperationalError | None = None
    for uri in attempts:
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
            conn.execute("SELECT 1")
            return conn
        except sqlite3.OperationalError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise sqlite3.OperationalError("unable to open database file")


def db_status(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "readable": False}
    try:
        conn = _connect_ro(path)
        conn.close()
        return {
            "path": str(path),
            "exists": True,
            "readable": True,
            "size_bytes": path.stat().st_size,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "readable": False,
            "error": str(exc),
        }


def query_rows(
    path: Path,
    sql: str,
    params: tuple = (),
) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    conn = _connect_ro(path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def query_one(
    path: Path,
    sql: str,
    params: tuple = (),
) -> Optional[Dict[str, Any]]:
    rows = query_rows(path, sql, params)
    return rows[0] if rows else None
