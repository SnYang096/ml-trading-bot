"""Tests for Binance USD-M ``POST /fapi/v1/positionSide/dual`` helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.order_management.binance_api import BinanceAPI


def _minimal_api() -> BinanceAPI:
    api = BinanceAPI.__new__(BinanceAPI)
    api.api_key = "test_key"
    api.api_secret = "test_secret_must_be_known"
    api.testnet = False
    api.use_proxy = False
    api.time_offset = 0
    return api


def test_set_dual_side_position_posts_to_fapi() -> None:
    api = _minimal_api()
    with patch("src.order_management.binance_api.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 200, "msg": "success"}
        mock_post.return_value = mock_resp

        out = api.set_dual_side_position(True)

        mock_post.assert_called_once()
        call_kw = mock_post.call_args
        url = call_kw[0][0]
        body = call_kw[1]["data"]

        assert "fapi.binance.com/fapi/v1/positionSide/dual" in url
        assert "dualSidePosition=true" in body
        assert "timestamp=" in body
        assert "signature=" in body
        assert "test_key" == call_kw[1]["headers"]["X-MBX-APIKEY"]

        assert out == {"code": 200, "msg": "success"}


def test_set_dual_side_position_raises_on_binance_error() -> None:
    api = _minimal_api()
    with patch("src.order_management.binance_api.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": -4059, "msg": "cannot switch"}
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Binance futures API error"):
            api.set_dual_side_position(False)
