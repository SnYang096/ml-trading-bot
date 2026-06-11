def test_orders_list_api(client):
    r = client.get(
        "/api/orders/list",
        params={"symbol": "ETHUSDT", "scopes": "trend,spot"},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert isinstance(data, list)
    assert len(data) >= 1


def test_trend_orders_api_includes_stop_loss_and_pnl(client, trend_db):
    import sqlite3

    conn = sqlite3.connect(trend_db)
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'api_exit', 'ETHUSDT', 'SELL', 'filled', 'limit',
            0.1, 105.0, NULL,
            '2024-01-01T14:00:00+00:00', '2024-01-01T13:30:00+00:00',
            '2024-01-01T14:00:00+00:00', 105.0, 0.1, 'p1'
        )
        """
    )
    conn.commit()
    conn.close()

    r = client.get("/api/trend/orders", params={"symbol": "ETHUSDT", "limit": "50"})
    assert r.status_code == 200
    rows = r.json()["data"]
    row = next(x for x in rows if x["order_id"] == "api_exit")
    assert row["stop_loss_price"] == 98.5
    assert row["pnl_usdt"] == 12.5


def test_multileg_orders_api_returns_history_when_not_excluded(client, multi_leg_db):
    r = client.get(
        "/api/multileg/orders",
        params={"symbol": "ETHUSDT", "limit": "50"},
    )
    assert r.status_code == 200
    ids = {x["order_id"] for x in r.json()["data"]}
    assert "ml_eth_entry" in ids
    assert "ml_eth_l2_expired" in ids


def test_multileg_orders_api_hides_expired_when_requested(client):
    r = client.get(
        "/api/multileg/orders",
        params={
            "symbol": "ETHUSDT",
            "limit": "50",
            "exclude_status": "expired,canceled,rejected",
        },
    )
    assert r.status_code == 200
    ids = {x["order_id"] for x in r.json()["data"]}
    assert "ml_eth_l2_expired" not in ids
    assert "ml_eth_open_tp" in ids


def test_orders_list_all_symbols(client):
    r = client.get(
        "/api/orders/list",
        params={"symbol": "*", "scopes": "trend,spot", "limit": 500},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["meta"]["symbol"] == "ALL"
    assert isinstance(payload["data"], list)


def test_orders_trade_links_api(client, multi_leg_db):
    r = client.get(
        "/api/orders/trade-links",
        params={"symbol": "ETHUSDT", "scopes": "multi_leg", "limit": 50},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["meta"]["symbol"] == "ETHUSDT"
    assert isinstance(payload["data"], list)


def test_orders_trade_links_all_symbols_empty(client):
    r = client.get(
        "/api/orders/trade-links",
        params={"symbol": "*", "scopes": "multi_leg"},
    )
    assert r.status_code == 200
    assert r.json()["data"] == []
