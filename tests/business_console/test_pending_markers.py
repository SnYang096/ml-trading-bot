from mlbot_console.services.trade_markers import collect_markers


def test_pending_excluded_by_default(trend_db, spot_db, multi_leg_db):
    m = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
        include_pending=False,
    )
    assert all(x.get("status") != "pending" for x in m)


def test_pending_included(trend_db, spot_db, multi_leg_db):
    m = collect_markers(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        symbol="ETHUSDT",
        scopes=["trend", "spot"],
        include_pending=True,
    )
    assert any(x.get("status") == "pending" for x in m)


def test_expired_multileg_not_shown_as_pending(multi_leg_db):
    from src.order_management.multi_leg_storage import MultiLegStorage

    from mlbot_console.services.trade_markers import multi_leg_markers

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="mlr_exp_pending",
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "bnb_open",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.1,
            "price": 656.0,
            "status": "open",
            "filled_quantity": 0.0,
            "created_at": "2026-05-20T12:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "bnb_expired",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.1,
            "price": 657.0,
            "status": "expired",
            "filled_quantity": 0.0,
            "created_at": "2026-05-19T08:00:00+00:00",
        }
    )
    with_open = multi_leg_markers(multi_leg_db, "BNBUSDT", include_open_orders=True)
    ids = {
        (m.get("detail") or {}).get("local_order_id")
        for m in with_open
        if m.get("status") == "pending"
    }
    assert "bnb_open" in ids
    assert "bnb_expired" not in ids
