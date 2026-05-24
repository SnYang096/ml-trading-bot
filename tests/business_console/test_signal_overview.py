"""Signal overview table API."""

from __future__ import annotations


def test_trade_map_signals(client):
    r = client.get(
        "/api/trade-map/signals", params={"timeframe": "2h", "lookback_days": 7}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    rows = body["data"]
    assert len(rows) >= 1
    row = next(x for x in rows if x["symbol"] == "ETHUSDT")
    assert "strategies" in row
    assert "trend" in row["strategies"]
    assert "spot" in row["strategies"]
    assert "multi_leg" in row["strategies"]
    assert row["map_href"].startswith("/trade-map")
    assert "summary" in row["strategies"]["spot"]
    spot_by = row["strategies"]["spot"].get("by_strategy") or {}
    assert "spot_accum_simple" in spot_by or row["strategies"]["spot"]["summary"] != "—"
    trend_by = row["strategies"]["trend"].get("by_strategy")
    assert trend_by is not None


def test_signals_page_served(client):
    r = client.get("/signals")
    assert r.status_code == 200
    assert "策略信号" in r.text
    assert "signals-page.js" in r.text
