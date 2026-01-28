import sys
import types

from src.live_data_stream.order_manager_factory import init_order_manager_from_env


def test_init_order_manager_from_env_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MLBOT_ORDER_MANAGER_ENABLED", raising=False)
    assert init_order_manager_from_env() is None


def test_init_order_manager_from_env_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLBOT_ORDER_MANAGER_ENABLED", "true")
    monkeypatch.setenv("MLBOT_ORDER_MANAGER_TESTNET", "true")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "key")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", "secret")
    monkeypatch.setenv(
        "MLBOT_ORDER_MANAGEMENT_DB_PATH", str(tmp_path / "order_mgmt.db")
    )

    storage_module = types.SimpleNamespace()
    binance_module = types.SimpleNamespace()
    order_manager_module = types.SimpleNamespace()

    class DummyStorage:
        def __init__(self, db_path: str) -> None:
            self.db_path = db_path

    class DummyBinanceAPI:
        def __init__(self, api_key: str, api_secret: str, testnet: bool, use_proxy):
            self.api_key = api_key
            self.api_secret = api_secret
            self.testnet = testnet
            self.use_proxy = use_proxy

    class DummyOrderManager:
        def __init__(self, storage, binance_api) -> None:
            self.storage = storage
            self.binance_api = binance_api

    storage_module.Storage = DummyStorage
    binance_module.BinanceAPI = DummyBinanceAPI
    order_manager_module.OrderManager = DummyOrderManager

    monkeypatch.setitem(sys.modules, "src.order_management.storage", storage_module)
    monkeypatch.setitem(sys.modules, "src.order_management.binance_api", binance_module)
    monkeypatch.setitem(
        sys.modules, "src.order_management.order_manager", order_manager_module
    )

    om = init_order_manager_from_env()
    assert isinstance(om, DummyOrderManager)
    assert om.storage.db_path.endswith("order_mgmt.db")
    assert om.binance_api.testnet is True
