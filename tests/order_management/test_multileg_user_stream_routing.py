"""Multileg user-stream → orchestrator routing (mock, no WebSocket).

Mirrors ``run_multi_leg_live.on_execution_report`` callback: symbol-scoped
dispatch to the matching ``StrategyRuntime`` only, with optional storage persist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from src.order_management.binance_user_stream import BinanceUserStream
from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator


@dataclass
class _FakeEngine:
    reports: list[dict[str, Any]] = field(default_factory=list)

    def on_execution_report(self, report: dict[str, Any]) -> None:
        self.reports.append(dict(report))


def _orchestrator(
    engine: _FakeEngine, *, strategy: str, symbol: str
) -> MultiLegLiveOrchestrator:
    return MultiLegLiveOrchestrator(
        engine=engine,
        governor=MagicMock(),
        adapter=MagicMock(),
        reconciler=MagicMock(),
        strategy_name=strategy,
        symbol=symbol,
        run_id="run_us",
    )


def _make_routing_callback(runtimes, *, storage=None, run_id="run_us"):
    """Same routing semantics as ``scripts/run_multi_leg_live.py``."""

    def on_execution_report(exec_report: dict[str, Any]) -> None:
        sym = str(exec_report.get("symbol") or "").upper().strip()
        if not sym:
            return
        for rt in runtimes:
            if rt.symbol.upper() == sym:
                rt.orchestrator.on_execution_report(exec_report)
                if storage is not None:
                    storage.record_execution_report(
                        {
                            **dict(exec_report),
                            "run_id": run_id,
                            "strategy": rt.name,
                            "raw": dict(exec_report),
                        }
                    )
                break

    return on_execution_report


def test_user_stream_routes_fill_to_matching_symbol_runtime_only() -> None:
    btc_engine = _FakeEngine()
    eth_engine = _FakeEngine()
    btc_rt = MagicMock()
    btc_rt.name = "trend_scalp"
    btc_rt.symbol = "BTCUSDT"
    btc_rt.orchestrator = _orchestrator(
        btc_engine, strategy="trend_scalp", symbol="BTCUSDT"
    )
    eth_rt = MagicMock()
    eth_rt.name = "chop_grid"
    eth_rt.symbol = "ETHUSDT"
    eth_rt.orchestrator = _orchestrator(
        eth_engine, strategy="chop_grid", symbol="ETHUSDT"
    )

    stream = BinanceUserStream.__new__(BinanceUserStream)
    stream.on_execution_report = _make_routing_callback([btc_rt, eth_rt])
    stream.on_account_update = MagicMock()

    msg = json.dumps(
        {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1710000000000,
            "o": {
                "s": "BTCUSDT",
                "i": 123,
                "c": "dat_btc_fill",
                "S": "BUY",
                "o": "LIMIT",
                "X": "FILLED",
                "x": "TRADE",
                "l": "0.01",
                "z": "0.01",
                "L": "50000.5",
                "ap": "50000.5",
                "T": 1710000000123,
            },
        }
    )
    stream._handle_message(msg)

    assert len(btc_engine.reports) == 1
    assert btc_engine.reports[0]["client_order_id"] == "dat_btc_fill"
    assert eth_engine.reports == []


def test_user_stream_routing_persists_execution_report_to_storage() -> None:
    engine = _FakeEngine()
    rt = MagicMock()
    rt.name = "trend_scalp"
    rt.symbol = "XRPUSDT"
    rt.orchestrator = _orchestrator(engine, strategy="trend_scalp", symbol="XRPUSDT")
    storage = MagicMock()

    callback = _make_routing_callback([rt], storage=storage, run_id="run_xrp")
    report = {
        "symbol": "XRPUSDT",
        "order_id": "ex_42",
        "client_order_id": "dat_xrp_1",
        "status": "PARTIALLY_FILLED",
        "filled_qty": 100.0,
    }
    callback(report)

    storage.record_execution_report.assert_called_once()
    payload = storage.record_execution_report.call_args.args[0]
    assert payload["run_id"] == "run_xrp"
    assert payload["strategy"] == "trend_scalp"
    assert payload["raw"]["filled_qty"] == 100.0
    assert len(engine.reports) == 1


def test_user_stream_routing_ignores_empty_symbol() -> None:
    engine = _FakeEngine()
    rt = MagicMock()
    rt.name = "trend_scalp"
    rt.symbol = "BTCUSDT"
    rt.orchestrator = _orchestrator(engine, strategy="trend_scalp", symbol="BTCUSDT")

    callback = _make_routing_callback([rt])
    callback({"symbol": "", "status": "FILLED"})

    assert engine.reports == []
