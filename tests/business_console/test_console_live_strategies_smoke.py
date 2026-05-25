"""Smoke expectations for constitution-driven console live strategies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mlbot_console.config import SETTINGS
from mlbot_console.services.strategy_registry import get_live_console_strategies


@pytest.fixture(autouse=True)
def _clear_live_cache():
    get_live_console_strategies.cache_clear()
    yield
    get_live_console_strategies.cache_clear()


def test_live_strategies_from_repo_constitution():
    assert SETTINGS.constitution_yaml.is_file()
    live = get_live_console_strategies()
    ids = [s["id"] for s in live]
    assert "tpc" in ids
    assert "chop_grid" in ids
    assert "trend_scalp" in ids
    assert "spot_accum_simple" in ids
    assert "bpc" not in ids
    by_id = {s["id"]: s["account_layer"] for s in live}
    assert by_id["chop_grid"] == "multi_leg"
    assert by_id["trend_scalp"] == "multi_leg"


def test_live_strategies_empty_when_constitution_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    missing = tmp_path / "no_constitution.yaml"
    assert not missing.is_file()

    monkeypatch.setattr(
        "mlbot_console.config.SETTINGS",
        replace(SETTINGS, constitution_yaml=missing),
    )
    assert get_live_console_strategies() == []
