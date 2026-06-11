"""Tests for live symbol resolution from universe.yaml."""

from __future__ import annotations

import argparse

import pytest

from src.live_data_stream.universe_symbols import (
    read_universe_symbols,
    resolve_symbols_csv,
)


def test_read_universe_highcap_includes_hype():
    syms = read_universe_symbols("highcap")
    assert "HYPEUSDT" in syms
    assert "ADAUSDT" not in syms


def test_resolve_cli_overrides_universe():
    csv = resolve_symbols_csv(cli_symbols="DOGEUSDT", universe="highcap")
    assert csv == "DOGEUSDT"


def test_resolve_universe_when_no_cli():
    expected = ",".join(read_universe_symbols("highcap"))
    csv = resolve_symbols_csv(cli_symbols=None, universe="highcap", env_symbols="")
    assert csv == expected


def test_resolve_env_when_universe_missing(tmp_path, monkeypatch):
    csv = resolve_symbols_csv(
        cli_symbols=None,
        universe="missing",
        env_symbols="LINKUSDT",
        project_root=tmp_path,
    )
    assert csv == "LINKUSDT"


def test_publisher_resolve_symbols(monkeypatch):
    import scripts.run_market_feature_publisher as pub

    ns = argparse.Namespace(symbols=None, universe="highcap")
    syms = pub._resolve_symbols(ns)
    assert "HYPEUSDT" in syms


def test_resolve_raises_when_all_missing(tmp_path):
    with pytest.raises(ValueError, match="no symbols"):
        resolve_symbols_csv(
            cli_symbols=None,
            universe="missing",
            env_symbols="",
            project_root=tmp_path,
        )
