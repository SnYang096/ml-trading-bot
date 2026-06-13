"""Unified multi-leg entry→exit pairing (TP / SL / market_exit / regime)."""

from __future__ import annotations

import pytest

from mlbot_console.services.multileg_leg_pnl import (
    exit_kind_for_multileg_row,
    pair_multileg_entry_exits,
)
from mlbot_console.services.trade_links import multi_leg_trade_links


def test_pair_multileg_covers_trend_regime_and_chop_tp(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid", "trend_scalp"],
        symbols=["BNBUSDT"],
        run_id="pair_mix",
    )
    group = "BNBUSDT_grid"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 650.0,
            "filled_at": "2026-06-05 10:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_sl",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "stop_loss",
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 655.0,
            "filled_at": "2026-06-05 11:00:00+00:00",
        }
    )
    segment = "BNBUSDT_2026-06-05 12:00:00+00:00"
    entry_id = f"{segment}_initial_trend_BUY_0_0"
    exit_id = f"{entry_id}_fill0_exit_regime_exit_2026-06-05 13:00:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": entry_id,
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "entry",
            "status": "filled",
            "filled_quantity": 0.1,
            "average_price": 640.0,
            "filled_at": "2026-06-05 12:05:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": exit_id,
            "symbol": "BNBUSDT",
            "side": "SELL",
            "position_side": "LONG",
            "purpose": "market_exit",
            "status": "filled",
            "filled_quantity": 0.1,
            "average_price": 635.0,
            "filled_at": "2026-06-05 13:00:00+00:00",
        }
    )

    from mlbot_console.services.multileg_leg_pnl import _MULTILEG_ORDER_SQL
    from mlbot_console.services.db import query_rows
    from mlbot_console.services.multileg_order_links import hydrate_multileg_fill_fields

    rows = list(query_rows(multi_leg_db, _MULTILEG_ORDER_SQL, ("BNBUSDT",)))
    for row in rows:
        hydrate_multileg_fill_fields(row)

    pairs = pair_multileg_entry_exits(rows)
    assert len(pairs) == 2
    kinds = {exit_kind_for_multileg_row(ex) for _en, ex in pairs}
    assert kinds == {"stop_loss", "regime_exit"}

    links, _ = multi_leg_trade_links(multi_leg_db, "BNBUSDT")
    assert len(links) == 2
    chop = next(lk for lk in links if lk["strategy"] == "chop_grid")
    trend = next(lk for lk in links if lk["strategy"] == "trend_scalp")
    assert chop["exit_kind"] == "stop_loss"
    assert chop["pnl_usdt"] == pytest.approx((650.0 - 655.0) * 0.5, rel=1e-4)
    assert trend["exit_kind"] == "regime_exit"
    assert trend["pnl_usdt"] == pytest.approx((635.0 - 640.0) * 0.1, rel=1e-4)
