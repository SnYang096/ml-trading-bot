def test_orders_list_api(client):
    r = client.get(
        "/api/orders/list",
        params={"symbol": "ETHUSDT", "scopes": "trend,spot"},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert isinstance(data, list)
    assert len(data) >= 1
