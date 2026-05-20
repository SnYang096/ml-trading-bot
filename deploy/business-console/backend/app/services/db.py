"""Read-only SQLite helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def db_status(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "readable": False}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("SELECT 1")
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
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
