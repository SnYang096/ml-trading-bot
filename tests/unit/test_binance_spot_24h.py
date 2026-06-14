from __future__ import annotations

from src.market_momentum.binance_spot_24h import (
    fetch_usdt_24h_gainers,
    is_tradable_satellite_usdt_pair,
    weekly_deploy_usdt,
)


def test_is_tradable_satellite_usdt_pair_filters_stables_and_leveraged():
    assert is_tradable_satellite_usdt_pair("SOLUSDT")
    assert not is_tradable_satellite_usdt_pair("USDCUSDT")
    assert not is_tradable_satellite_usdt_pair("BTCUPUSDT")
    assert not is_tradable_satellite_usdt_pair("ETHDOWNUSDT")
    assert not is_tradable_satellite_usdt_pair("BTCBUSD")


def test_fetch_usdt_24h_gainers_sorts_and_filters(monkeypatch):
    sample = [
        {
            "symbol": "AAAUSDT",
            "priceChangePercent": "5.0",
            "quoteVolume": "2000000",
            "lastPrice": "1.0",
        },
        {
            "symbol": "BBBUSDT",
            "priceChangePercent": "50.0",
            "quoteVolume": "500000",
            "lastPrice": "2.0",
        },
        {
            "symbol": "CCCUSDT",
            "priceChangePercent": "30.0",
            "quoteVolume": "3000000",
            "lastPrice": "3.0",
        },
        {
            "symbol": "USDCUSDT",
            "priceChangePercent": "99.0",
            "quoteVolume": "9000000",
            "lastPrice": "1.0",
        },
    ]

    monkeypatch.setattr(
        "src.market_momentum.binance_spot_24h.fetch_ticker_24hr",
        lambda **_: sample,
    )
    rows = fetch_usdt_24h_gainers(limit=5, min_quote_volume_usdt=1_000_000)
    assert [r.symbol for r in rows] == ["CCCUSDT", "AAAUSDT"]
    assert rows[0].rank == 1
    assert rows[0].price_change_pct == 30.0


def test_weekly_deploy_usdt_one_percent_and_caps():
    assert weekly_deploy_usdt(10_000, deploy_frac=0.01) == 100.0
    assert weekly_deploy_usdt(10_000, deploy_frac=0.01, single_coin_cap_usdt=50) == 50.0
    assert weekly_deploy_usdt(-100, deploy_frac=0.01) == 0.0
