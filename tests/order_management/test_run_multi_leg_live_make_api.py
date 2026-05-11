from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
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


def test_build_daemon_applies_per_strategy_symbol_filters(tmp_path: Path, monkeypatch):
    import scripts.run_multi_leg_live as m

    chop_dir = tmp_path / "chop_grid"
    dual_dir = tmp_path / "dual_add_trend"
    (chop_dir / "research").mkdir(parents=True)
    (dual_dir / "research").mkdir(parents=True)
    (chop_dir / "research" / "calibrate_roll.default.yaml").write_text(
        "", encoding="utf-8"
    )
    (dual_dir / "research" / "calibrate_roll.default.yaml").write_text(
        "", encoding="utf-8"
    )
    (chop_dir / "meta.yaml").write_text(
        """
strategy:
  symbol_include: [BTCUSDT,ETHUSDT]
""".strip(),
        encoding="utf-8",
    )
    (dual_dir / "meta.yaml").write_text(
        """
strategy:
  symbol_include: [BTCUSDT,XRPUSDT]
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(m, "apply_multi_leg_args_from_constitution", lambda args: None)
    monkeypatch.setattr(
        m, "_make_api", lambda mode, allow_shared_account=False: MockBinanceAPI()
    )
    monkeypatch.setattr(
        m,
        "_make_engine",
        lambda strategy, symbol, args: object(),
    )
    args = Namespace(
        mode="shadow",
        strategies="chop_grid,dual_add_trend",
        symbols="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT",
        allow_shared_account=False,
        multi_leg_db_path="",
        max_gross_notional=2000.0,
        max_net_notional=1000.0,
        max_symbol_gross_notional=800.0,
        max_symbol_net_notional=400.0,
        max_resting_orders=60,
        account_equity_usdt=10000.0,
        max_drawdown_pct=0.12,
        bar_source="parquet",
        data_dir="data/parquet_data",
        timeframe="2h",
        lookback_days=180,
        feature_bus_root="live/shared_feature_bus",
        feature_store_timeframe="2h",
        feature_store_execution_timeframe="1min",
        live_storage_base="data/live_storage",
        feature_compute_interval_minutes=15,
        memory_window_hours=4.0,
        orderflow_window_minutes=None,
        feature_4h_interval_hours=4,
        warmup_days=0,
        poll_seconds=60.0,
        unit_notional=100.0,
        state_dir=str(tmp_path / "state"),
        chop_grid_config=str(chop_dir / "research" / "calibrate_roll.default.yaml"),
        dual_add_config=str(dual_dir / "research" / "calibrate_roll.default.yaml"),
    )

    daemon, _api, _storage, _run_id = m.build_daemon(args)
    pairs = {(rt.name, rt.symbol) for rt in daemon.runtimes}
    assert ("chop_grid", "BTCUSDT") in pairs
    assert ("chop_grid", "ETHUSDT") in pairs
    assert ("chop_grid", "XRPUSDT") not in pairs
    assert ("dual_add_trend", "XRPUSDT") in pairs
    assert ("dual_add_trend", "ETHUSDT") not in pairs
