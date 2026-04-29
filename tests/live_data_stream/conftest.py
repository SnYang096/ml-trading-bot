"""live_data_stream test collection hooks."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_binance_ws: opens real Binance WebSocket (set MLBOT_RUN_LIVE_WS_TESTS=1)",
    )
