"""Tests for multi-leg base symbol resolution (universe.yaml alignment with trend)."""

from __future__ import annotations

import argparse

import scripts.run_multi_leg_live as m


def test_read_universe_highcap_sorted_keys():
    syms = m._read_universe_yaml_symbols("highcap")
    assert syms == sorted(syms)
    assert "BTCUSDT" in syms


def test_resolve_cli_overrides_universe(monkeypatch):
    monkeypatch.delenv("MLBOT_MULTI_LEG_SYMBOLS", raising=False)
    ns = argparse.Namespace(symbols="DOGEUSDT", universe="highcap")
    assert m.resolve_multi_leg_base_symbols_csv(ns) == "DOGEUSDT"


def test_resolve_universe_when_no_cli_or_env(monkeypatch):
    monkeypatch.delenv("MLBOT_MULTI_LEG_SYMBOLS", raising=False)
    ns = argparse.Namespace(symbols=None, universe="highcap")
    expected = ",".join(m._read_universe_yaml_symbols("highcap"))
    assert m.resolve_multi_leg_base_symbols_csv(ns) == expected


def test_resolve_env_when_universe_file_missing(monkeypatch):
    def _raise(_universe: str) -> list[str]:
        raise FileNotFoundError("no yaml")

    monkeypatch.setattr(m, "_read_universe_yaml_symbols", _raise)
    monkeypatch.setenv("MLBOT_MULTI_LEG_SYMBOLS", "AAUSDT,BBUSDT")
    ns = argparse.Namespace(symbols=None, universe="no_such")
    assert m.resolve_multi_leg_base_symbols_csv(ns) == "AAUSDT,BBUSDT"


def test_resolve_fallback_when_universe_empty_and_no_env(monkeypatch):
    monkeypatch.setattr(m, "_read_universe_yaml_symbols", lambda _u: [])
    monkeypatch.delenv("MLBOT_MULTI_LEG_SYMBOLS", raising=False)
    ns = argparse.Namespace(symbols=None, universe="highcap")
    assert m.resolve_multi_leg_base_symbols_csv(ns) == m._FALLBACK_MULTI_LEG_SYMBOLS
