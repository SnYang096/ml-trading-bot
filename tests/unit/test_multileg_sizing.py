from __future__ import annotations

from pathlib import Path

import pytest

from src.config.multileg_sizing import (
    grid_capital_units,
    resolve_chop_grid_unit_notional,
    resolve_multi_leg_unit_notional,
    resolve_multi_leg_unit_notionals,
    resolve_trend_scalp_unit_notional,
    trend_capital_units,
    unit_notional_from_segment_dd,
    unit_notional_from_trend_segment_dd,
)


def test_grid_capital_units_dense_3l() -> None:
    assert grid_capital_units(3) == 6


def test_trend_capital_units() -> None:
    assert trend_capital_units(4) == 4


def test_unit_notional_from_segment_dd_1pct() -> None:
    unit = unit_notional_from_segment_dd(
        equity_usdt=10000.0,
        segment_dd_target=0.01,
        max_loss_per_grid=0.03,
        max_levels_per_side=3,
    )
    assert unit == pytest.approx(10000.0 * 0.01 / (0.03 * 6))


def test_unit_notional_from_trend_segment_dd_1pct() -> None:
    unit = unit_notional_from_trend_segment_dd(
        equity_usdt=10000.0,
        segment_dd_target=0.01,
        max_loss_per_segment=0.02,
        max_gross_exposure_units=4,
    )
    assert unit == pytest.approx(10000.0 * 0.01 / (0.02 * 4))


def test_resolve_prefers_explicit_unit_notional() -> None:
    ml = {"unit_notional": 123.0, "sizing": {"segment_dd_target": 0.01}}
    assert resolve_multi_leg_unit_notional(ml, equity_usdt=10000.0) == 123.0


def test_resolve_from_segment_dd_and_execution_yaml(tmp_path: Path) -> None:
    exe = tmp_path / "execution.yaml"
    exe.write_text(
        "inventory:\n  max_levels_per_side: 3\n" "risk:\n  max_loss_per_grid: 0.03\n",
        encoding="utf-8",
    )
    ml = {"sizing": {"segment_dd_target": 0.01}}
    unit = resolve_chop_grid_unit_notional(
        ml, equity_usdt=10000.0, chop_grid_execution_path=exe
    )
    assert unit == pytest.approx(555.555, rel=1e-3)


def test_per_strategy_sizing_blocks() -> None:
    ml = {
        "sizing": {
            "chop_grid": {
                "segment_dd_target": 0.025,
                "max_loss_per_grid": 0.03,
                "max_levels_per_side": 3,
            },
            "trend_scalp": {
                "segment_dd_target": 0.01,
                "max_loss_per_segment": 0.02,
                "max_gross_exposure_units": 4,
            },
        }
    }
    units = resolve_multi_leg_unit_notionals(ml, equity_usdt=10000.0)
    assert units["chop_grid"] == pytest.approx(10000.0 * 0.025 / (0.03 * 6))
    assert units["trend_scalp"] == pytest.approx(10000.0 * 0.01 / (0.02 * 4))


def test_trend_resolve_uses_own_block() -> None:
    ml = {
        "sizing": {
            "chop_grid": {"segment_dd_target": 0.01},
            "trend_scalp": {
                "segment_dd_target": 0.012,
                "max_loss_per_segment": 0.02,
                "max_gross_exposure_units": 4,
            },
        }
    }
    unit = resolve_trend_scalp_unit_notional(ml, equity_usdt=10000.0)
    assert unit == pytest.approx(10000.0 * 0.012 / (0.02 * 4))
