def test_orders_list_api(client):
    r = client.get(
        "/api/orders/list",
        params={"symbol": "ETHUSDT", "scopes": "trend,spot"},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert isinstance(data, list)
    assert len(data) >= 1


def test_orders_list_all_symbols(client):
    r = client.get(
        "/api/orders/list",
        params={"symbol": "*", "scopes": "trend,spot", "limit": 500},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["meta"]["symbol"] == "ALL"
    assert isinstance(payload["data"], list)
