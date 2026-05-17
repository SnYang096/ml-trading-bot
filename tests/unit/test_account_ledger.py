from __future__ import annotations

from datetime import datetime, timezone

from scripts.account_ledger import AccountLedger


def test_spot_open_merge_close_realized_pnl() -> None:
    ld = AccountLedger(account="spot", initial_cash_usdt=10_000.0)
    ok, n1, why = ld.open_lot(
        lot_id="btc",
        strategy="spot_accum",
        symbol="BTCUSDT",
        side="LONG",
        notional_usdt=500.0,
        entry_price=50_000.0,
        fee_rate=0.001,
        opened_at=datetime.now(timezone.utc),
        cash_mode="cash_notional",
        allow_scale_down=False,
    )
    assert ok and why == ""
    assert round(n1, 6) == 500.0
    cash_after_open = ld.cash_usdt
    assert cash_after_open < 10_000.0

    ok2, n2, why2 = ld.merge_lot(
        lot_id="btc",
        add_notional_usdt=250.0,
        add_price=40_000.0,
        fee_rate=0.001,
        allow_scale_down=False,
    )
    assert ok2 and why2 == ""
    assert round(n2, 6) == 250.0

    res = ld.close_lot(lot_id="btc", exit_price=60_000.0, fee_rate=0.001)
    assert res is not None
    assert res.qty_base > 0.0
    assert res.entry_notional_usdt == 750.0
    assert res.exit_notional_usdt > 0.0
    assert res.realized_pnl_usdt > 0.0
    # Cash returns to initial plus realized pnl for cash_notional mode.
    assert abs(ld.cash_usdt - (10_000.0 + res.realized_pnl_usdt)) < 1e-6


def test_fee_only_mode_keeps_notional_out_of_cash() -> None:
    ld = AccountLedger(account="trend", initial_cash_usdt=1_000.0)
    ok, n1, why = ld.open_lot(
        lot_id="trend1",
        strategy="bpc",
        symbol="BTCUSDT",
        side="LONG",
        notional_usdt=5_000.0,
        entry_price=50_000.0,
        fee_rate=0.001,
        opened_at=datetime.now(timezone.utc),
        cash_mode="fee_only",
        allow_scale_down=False,
    )
    assert ok and why == ""
    # only entry fee (5 USDT) deducted
    assert abs(ld.cash_usdt - 995.0) < 1e-9
    res = ld.close_lot(lot_id="trend1", exit_price=55_000.0, fee_rate=0.001)
    assert res is not None
    assert res.realized_pnl_usdt > 0.0
