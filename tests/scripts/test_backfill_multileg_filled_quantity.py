from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.backfill_multileg_filled_quantity import (
    backfill,
    extract_fill_fields_from_raw,
)


def test_extract_fill_fields_from_raw_json() -> None:
    raw = {
        "filled": 0.126,
        "average_price": 1685.77,
        "info": {"executedQty": "0.126", "avgPrice": "1685.77000"},
    }
    filled, avg = extract_fill_fields_from_raw(raw)
    assert filled == 0.126
    assert avg == 1685.77


def test_backfill_updates_zero_filled_column(tmp_path: Path) -> None:
    db = tmp_path / "multi_leg.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE multi_leg_orders (
            local_order_id TEXT PRIMARY KEY,
            strategy TEXT,
            symbol TEXT,
            filled_quantity REAL,
            average_price REAL,
            raw_json TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO multi_leg_orders (
            local_order_id, strategy, symbol, filled_quantity, average_price, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "trend_entry",
            "trend_scalp",
            "ETHUSDT",
            0.0,
            None,
            '{"filled": 0.126, "average_price": 1685.77}',
        ),
    )
    conn.commit()
    conn.close()

    stats = backfill(db, dry_run=False)
    assert stats["updated"] == 1

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT filled_quantity, average_price FROM multi_leg_orders WHERE local_order_id = ?",
        ("trend_entry",),
    ).fetchone()
    conn.close()
    assert float(row[0]) == 0.126
    assert float(row[1]) == 1685.77
