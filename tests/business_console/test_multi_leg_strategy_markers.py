"""Markers distinguish multi-leg strategy slugs (chop_grid vs trend_scalp)."""

from __future__ import annotations

from mlbot_console.services.trade_markers import _multi_leg_event, multi_leg_markers


def test_multi_leg_event_take_profit_is_tp() -> None:
    assert _multi_leg_event("take_profit", "LIMIT") == "tp"


def test_multi_leg_markers_carry_strategy_slug(multi_leg_db) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid", "trend_scalp"],
        symbols=["BNBUSDT"],
        run_id="mlr_strat_split",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "bnb_cg",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "entry",
            "quantity": 0.1,
            "price": 600.0,
            "status": "filled",
            "filled_quantity": 0.1,
            "average_price": 600.0,
            "filled_at": "2026-05-19T12:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "trend_scalp",
            "local_order_id": "bnb_ts",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "entry",
            "quantity": 0.1,
            "price": 610.0,
            "status": "filled",
            "filled_quantity": 0.1,
            "average_price": 610.0,
            "filled_at": "2026-05-20T12:00:00+00:00",
        }
    )
    markers = multi_leg_markers(multi_leg_db, "BNBUSDT")
    strategies = {m["strategy"] for m in markers if m["scope"] == "multi_leg"}
    assert "chop_grid" in strategies
    assert "trend_scalp" in strategies
