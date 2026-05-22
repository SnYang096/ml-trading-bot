"""Console strategy registry from constitution.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from mlbot_console.services.strategy_registry import (
    default_spot_strategy_id,
    get_console_strategies,
    spot_strategy_ids,
    strategies_from_constitution_cfg,
)


def test_strategies_from_constitution_matches_yaml(tmp_path: Path) -> None:
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "config/constitution/constitution.yaml"
        ).read_text(encoding="utf-8")
    )
    rows = strategies_from_constitution_cfg(cfg)
    ids = {r["id"] for r in rows}
    assert {"tpc", "bpc", "me", "srb"}.issubset(ids)
    assert "chop_grid" in ids
    assert "trend_scalp" in ids
    assert "spot_accum_simple" in ids


def test_get_console_strategies_has_layer_titles() -> None:
    rows = get_console_strategies()
    by_id = {r["id"]: r for r in rows}
    assert by_id["chop_grid"]["account_layer"] == "multi_leg"
    assert by_id["tpc"]["account_layer"] == "trend"
    assert by_id["chop_grid"]["title"] == "Chop Grid"


def test_spot_strategy_ids_from_constitution_yaml(tmp_path: Path) -> None:
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "config/constitution/constitution.yaml"
        ).read_text(encoding="utf-8")
    )
    ids = [
        s["id"]
        for s in strategies_from_constitution_cfg(cfg)
        if s["account_layer"] == "spot"
    ]
    assert ids == ["spot_accum_simple"]
    assert spot_strategy_ids()  # uses SETTINGS or fallback; non-empty
    assert default_spot_strategy_id() == "spot_accum_simple"
