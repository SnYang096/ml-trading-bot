"""Account summary aggregation for console."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mlbot_console.services.account_summary import (
    _link_pnl_usdt,
    aggregate_weekly_realized,
    build_account_summary,
    build_order_pnl_maps,
    build_wallet_equity_curves,
    cumulative_realized_curve,
    latest_close_prices,
)


def test_build_wallet_equity_curves_lookback_without_snapshots_reconstructs() -> None:
    daily = [
        {"date": "2026-06-10", "pnl": 10.0},
        {"date": "2026-06-11", "pnl": -5.0},
    ]
    curves = build_wallet_equity_curves(
        daily,
        wallet_usdt=1005.0,
        equity_usdt=1020.0,
        lookback_days=30,
    )
    assert len(curves["balance"]) >= 2
    assert curves["balance"][-1]["value_usdt"] == pytest.approx(1005.0)
    assert curves["equity"][-1]["value_usdt"] == pytest.approx(1020.0)
    assert curves["balance"][0]["value_usdt"] == pytest.approx(1010.0)


def test_build_wallet_equity_curves_reconstructs_balance_and_live_equity() -> None:
    daily = [
        {"date": "2026-06-10", "pnl": 10.0},
        {"date": "2026-06-11", "pnl": -5.0},
    ]
    curves = build_wallet_equity_curves(
        daily,
        wallet_usdt=1005.0,
        equity_usdt=1020.0,
    )
    assert curves["balance"][-1]["value_usdt"] == pytest.approx(1005.0)
    assert curves["equity"][-1]["value_usdt"] == pytest.approx(1020.0)
    assert curves["balance"][0]["value_usdt"] == pytest.approx(1010.0)
    assert curves["equity"][0]["value_usdt"] == pytest.approx(1010.0)


def test_build_account_summary(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=3650,
    )
    assert data["symbol"] == "ETHUSDT"
    totals = data["totals"]
    assert totals["closed_trades"] >= 1
    trend_scope = next(s for s in data["scopes"] if s["scope"] == "trend")
    assert trend_scope["realized_pnl"] == 12.5
    assert any(s["scope"] == "spot" for s in data["scopes"])
    assert "daily_realized" in data


def test_account_summary_lookback_zero_includes_all_history(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=0,
    )
    trend_scope = next(s for s in data["scopes"] if s["scope"] == "trend")
    assert trend_scope["realized_pnl"] == 12.5
    assert any("all historical" in n for n in data["notes"])
    spot_strats = next(s for s in data["strategies"] if s["scope"] == "spot")
    assert spot_strats["strategy"] in {"spot_accum_simple"}


def test_account_summary_lookback_filters_old_trend_exit(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=7,
    )
    trend_scope = next(s for s in data["scopes"] if s["scope"] == "trend")
    assert trend_scope["realized_pnl"] == 0.0
    assert trend_scope["closed_trades"] == 0


def test_trend_stats_excludes_exchange_sync_closes(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    recent_exit = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'sync1', 'ETHUSDT', 'long',
            '2024-01-01T10:00:00+00:00', ?, 100.0, 100.0, 0.0,
            'closed', 'tpc', 98.5, 106.0, 2.5, 'exchange_sync_flat'
        )
        """,
        (recent_exit,),
    )
    conn.commit()
    conn.close()

    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=7,
    )
    trend_scope = next(s for s in data["scopes"] if s["scope"] == "trend")
    assert trend_scope.get("sync_cleanup_closed") == 1
    assert trend_scope["realized_pnl"] == 0.0
    assert trend_scope["closed_trades"] == 0


def test_account_summary_recent_exit_included(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    recent_exit = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        UPDATE positions
        SET exit_time = ?, realized_pnl = 99.0, status = 'closed'
        WHERE position_id = 'p1'
        """,
        (recent_exit,),
    )
    conn.commit()
    conn.close()

    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=7,
    )
    trend_scope = next(s for s in data["scopes"] if s["scope"] == "trend")
    assert trend_scope["realized_pnl"] == pytest.approx(99.0)
    assert trend_scope["closed_trades"] == 1


def test_build_order_pnl_maps_trend_exit(
    trend_db, spot_db, multi_leg_db, bus_root
) -> None:
    trend_map, spot_map, _ml = build_order_pnl_maps(
        trend_db=trend_db,
        spot_db=spot_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
    )
    assert trend_map["p1:exit"]["realized_pnl"] == 12.5
    assert trend_map["p1:exit"]["pnl_hint"] == "已实现"
    assert "p1" not in trend_map
    assert "s1" in spot_map


def test_link_pnl_usdt_long_and_short() -> None:
    entry = {"side": "BUY", "filled_quantity": 2.0, "average_price": 100.0}
    exit_row = {"side": "SELL", "filled_quantity": 2.0, "average_price": 110.0}
    assert _link_pnl_usdt(entry, exit_row) == pytest.approx(20.0)

    short_entry = {"side": "SELL", "filled_quantity": 1.0, "average_price": 200.0}
    short_exit = {"side": "BUY", "filled_quantity": 1.0, "average_price": 180.0}
    assert _link_pnl_usdt(short_entry, short_exit) == pytest.approx(20.0)


def test_latest_close_prices_reads_bus(bus_root) -> None:
    marks = latest_close_prices(bus_root, ["ETHUSDT", "MISSING"])
    assert "ETHUSDT" in marks
    assert marks["ETHUSDT"] > 0
    assert "MISSING" not in marks


def test_account_summary_api(client) -> None:
    r = client.get(
        "/api/account/summary", params={"symbol": "ETHUSDT", "lookback_days": 0}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "totals" in body["data"]
    assert len(body["data"]["scopes"]) == 3
    assert body["meta"]["lookback_days"] == 0
    assert "recent_realized" in body["data"]
    assert "weekly_realized" in body["data"]
    assert "cumulative_realized" in body["data"]
    assert "this_week_pnl" in body["data"]["recent_realized"]


def test_aggregate_weekly_and_cumulative_curve() -> None:
    daily = [
        {"date": "2026-05-19", "pnl": 1.0},
        {"date": "2026-05-20", "pnl": 2.0},
        {"date": "2026-05-22", "pnl": -0.5},
        {"date": "2026-05-26", "pnl": 4.0},
    ]
    weekly = aggregate_weekly_realized(daily)
    assert len(weekly) == 2
    assert weekly[0]["week_start"] == "2026-05-18"  # Mon of week containing 05-19
    assert weekly[0]["pnl"] == pytest.approx(2.5)
    assert weekly[1]["week_start"] == "2026-05-25"
    assert weekly[1]["pnl"] == pytest.approx(4.0)
    curve = cumulative_realized_curve(daily)
    assert [c["cumulative"] for c in curve] == pytest.approx([1.0, 3.0, 2.5, 6.5])


def test_account_summary_multileg_realized(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="acct_ml_pnl",
    )
    group = "BNBUSDT_2026-05-20 12:00:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SELL",
            "purpose": "entry",
            "quantity": 0.5,
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 700.0,
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "SHORT",
            "purpose": "take_profit",
            "quantity": 0.5,
            "status": "filled",
            "filled_quantity": 0.5,
            "average_price": 680.0,
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="BNBUSDT",
        lookback_days=0,
    )
    ml_scope = next(s for s in data["scopes"] if s["scope"] == "multi_leg")
    assert ml_scope["realized_pnl"] == pytest.approx(10.0, rel=1e-4)
    assert ml_scope["closed_trades"] >= 1
    chop = next(
        s
        for s in data["strategies"]
        if s["scope"] == "multi_leg" and s["strategy"] == "chop_grid"
    )
    assert chop["realized_pnl"] == pytest.approx(10.0, rel=1e-4)


def test_account_summary_filters_by_scopes(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    """scopes filter limits totals/strategies/daily aggregation to selected scopes."""
    full = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=0,
    )
    trend_only = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=0,
        scopes=["trend"],
    )
    assert {s["scope"] for s in trend_only["scopes"]} == {"trend"}
    assert all(s["scope"] == "trend" for s in trend_only["strategies"])
    assert trend_only["totals"]["realized_pnl"] == pytest.approx(
        next(s for s in full["scopes"] if s["scope"] == "trend")["realized_pnl"]
    )
    # Global ledger fields stay unfiltered (account-wide).
    assert trend_only["totals"].get("equity_usdt") == full["totals"].get("equity_usdt")


def test_account_summary_seeds_registry_trend_strategies_without_local_trades(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    """B·Trend rows appear even when local DB has no positions for the symbol."""
    from mlbot_console.services.strategy_registry import get_live_console_strategies

    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="BTCUSDT",
        lookback_days=0,
    )
    trend = [s for s in data["strategies"] if s["scope"] == "trend"]
    expected = {
        m["id"]
        for m in get_live_console_strategies()
        if m.get("account_layer") == "trend"
    }
    assert expected <= {s["strategy"] for s in trend}
    tpc = next(s for s in trend if s["strategy"] == "tpc")
    assert tpc["realized_pnl"] == 0.0
    assert tpc["closed_trades"] == 0
    assert tpc["strategy_title"] == "TPC"
    assert tpc["scope_label"] == "B·Trend"


def test_trend_open_position_unrealized_uses_entry_qty_when_size_missing(
    trend_db, spot_db, spot_ledger_db, multi_leg_db, bus_root
) -> None:
    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p_open2', 'ETHUSDT', 'long',
            '2026-05-19T08:00:00+00:00', NULL,
            2100.0, NULL, NULL, 'open', 'tpc', 2095.0, NULL, NULL, NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_open2', 'ETHUSDT', 'BUY', 'filled', 'limit',
            0.5, 2100.0, NULL,
            '2026-05-19T08:00:00+00:00', '2026-05-19T08:00:00+00:00',
            '2026-05-19T08:00:00+00:00', 2100.0, 0.5, 'p_open2'
        )
        """
    )
    conn.commit()
    conn.close()

    data = build_account_summary(
        trend_db=trend_db,
        spot_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        feature_bus_root=bus_root,
        symbol="ETHUSDT",
        lookback_days=0,
    )
    trend_scope = next(s for s in data["scopes"] if s["scope"] == "trend")
    assert trend_scope["open_positions"] >= 1
    assert trend_scope["unrealized_pnl"] != 0.0


def test_orders_list_api_includes_pnl_on_trend_exit(client) -> None:
    r = client.get(
        "/api/orders/list",
        params={"symbol": "ETHUSDT", "scopes": "trend", "limit": 50},
    )
    assert r.status_code == 200
    rows = r.json()["data"]
    exit_row = next(r for r in rows if r.get("order_id") == "p1:exit")
    assert exit_row["pnl_usdt"] == 12.5
    assert exit_row["pnl_hint"] == "已实现"
