from __future__ import annotations

from unittest.mock import MagicMock

from src.time_series_model.live.decision_chain_debug import (
    chain_debug_enabled,
    log_trend_no_intent,
)


def test_chain_debug_enabled_scope(monkeypatch):
    monkeypatch.delenv("MLBOT_CHAIN_DEBUG", raising=False)
    monkeypatch.delenv("MLBOT_TREND_CHAIN_DEBUG", raising=False)
    assert not chain_debug_enabled("trend")
    monkeypatch.setenv("MLBOT_TREND_CHAIN_DEBUG", "1")
    assert chain_debug_enabled("trend")


def test_log_trend_no_intent_pcm_trace(monkeypatch):
    monkeypatch.setenv("MLBOT_TREND_CHAIN_DEBUG", "1")
    pcm = MagicMock()
    pcm._last_decide_trace = {"drop_slot": 1}
    strat = MagicMock()
    strat._last_funnel = {"direction": False}
    pcm._strategies = {"bpc": strat}
    log_trend_no_intent("BTCUSDT", pcm, {"timestamp": "2026-01-01"})
