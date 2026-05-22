"""Tests for spot ledger book."""

from pathlib import Path

import pytest

from mlbot_console.services.spot_ledger_book import fetch_spot_ledger_holdings


def test_fetch_spot_ledger_holdings(spot_ledger_db: Path) -> None:
    import sqlite3
    import json

    conn = sqlite3.connect(spot_ledger_db)
    positions = {
        "lot1": {
            "symbol": "BTCUSDT",
            "qty_base": 0.5,
            "vwap_entry": 60000.0,
            "entry_notional_usdt": 30000.0,
        },
        "lot2": {
            "symbol": "ETHUSDT",
            "qty_base": 10.0,
            "vwap_entry": 3000.0,
            "entry_notional_usdt": 30000.0,
        },
    }
    conn.execute(
        "INSERT INTO state_kv (k, v) VALUES (?, ?)",
        ("positions", json.dumps(positions)),
    )
    conn.commit()
    conn.close()

    mark_prices = {"BTCUSDT": 65000.0, "ETHUSDT": 2500.0}
    res = fetch_spot_ledger_holdings(spot_ledger_db, mark_prices)

    assert len(res["holdings"]) == 2
    btc = next(h for h in res["holdings"] if h["asset"] == "BTC")
    assert btc["qty"] == 0.5
    assert btc["price_usdt"] == 65000.0
    assert btc["value_usdt"] == 32500.0
    assert btc["unrealized_pnl_usdt"] == 2500.0

    eth = next(h for h in res["holdings"] if h["asset"] == "ETH")
    assert eth["qty"] == 10.0
    assert eth["price_usdt"] == 2500.0
    assert eth["value_usdt"] == 25000.0
    assert eth["unrealized_pnl_usdt"] == -5000.0

    assert res["holdings_value_usdt"] == 57500.0


def test_fetch_spot_ledger_holdings_live_runner_format(spot_ledger_db: Path) -> None:
    """run_spot_accum_live stores symbol-keyed positions with _qty_base."""
    import sqlite3
    import json

    conn = sqlite3.connect(spot_ledger_db)
    positions = {
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "_qty_base": 0.0587412,
            "_entry_notional_usdt": 145.0,
            "_spot_quote_deployed": 145.0,
            "structural_exit": "spot_simple_profit_ladder",
        },
    }
    conn.execute(
        "INSERT INTO state_kv (k, v) VALUES (?, ?)",
        ("positions", json.dumps(positions)),
    )
    conn.commit()
    conn.close()

    res = fetch_spot_ledger_holdings(spot_ledger_db, {"ETHUSDT": 2500.0})
    assert len(res["holdings"]) == 1
    eth = res["holdings"][0]
    assert eth["asset"] == "ETH"
    assert eth["qty"] == pytest.approx(0.0587412)
    assert eth["deploy_usdt"] == 145.0
    assert eth["value_usdt"] == pytest.approx(0.0587412 * 2500.0)
