from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.order_management.binance_api import BinanceAPI
from src.order_management.mock_binance_api import MockBinanceAPI


def test_make_api_shadow() -> None:
    from scripts.run_multi_leg_live import _make_api

    api = _make_api("shadow", allow_shared_account=False)
    assert isinstance(api, MockBinanceAPI)


def test_make_api_mainnet_dedicated_keys() -> None:
    from scripts.run_multi_leg_live import _make_api

    with patch.dict(
        os.environ,
        {
            "MULTI_LEG_BINANCE_FUTURES_API_KEY": "k1",
            "MULTI_LEG_BINANCE_FUTURES_API_SECRET": "s1",
            "BINANCE_API_KEY": "",
            "BINANCE_API_SECRET": "",
        },
        clear=False,
    ):
        api = _make_api("mainnet", allow_shared_account=False)
    assert isinstance(api, BinanceAPI)
    assert api.testnet is False
    assert api.api_key == "k1"


def test_make_api_mainnet_shared_fallback() -> None:
    from scripts.run_multi_leg_live import _make_api

    with patch.dict(
        os.environ,
        {
            "MULTI_LEG_BINANCE_FUTURES_API_KEY": "",
            "MULTI_LEG_BINANCE_FUTURES_API_SECRET": "",
            "BINANCE_API_KEY": "bk",
            "BINANCE_API_SECRET": "bs",
        },
        clear=False,
    ):
        api = _make_api("mainnet", allow_shared_account=True)
    assert isinstance(api, BinanceAPI)
    assert api.testnet is False
    assert api.api_key == "bk"


def test_make_api_mainnet_missing_keys_raises() -> None:
    from scripts.run_multi_leg_live import _make_api

    with patch.dict(
        os.environ,
        {
            "MULTI_LEG_BINANCE_FUTURES_API_KEY": "",
            "MULTI_LEG_BINANCE_FUTURES_API_SECRET": "",
            "BINANCE_API_KEY": "",
            "BINANCE_API_SECRET": "",
        },
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="mainnet mode requires"):
            _make_api("mainnet", allow_shared_account=False)
