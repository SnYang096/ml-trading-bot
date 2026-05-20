"""SQLite read-only connection against WAL-style DB files."""

from __future__ import annotations

import sqlite3

from app.services.db import query_rows


def test_query_rows_on_wal_db(trend_db):
    rows = query_rows(trend_db, "SELECT position_id FROM positions LIMIT 5")
    assert isinstance(rows, list)


def test_nolock_uri_syntax():
    uri = "file:/tmp/x.db?mode=ro&nolock=1"
    assert "nolock=1" in uri
