from __future__ import annotations

import argparse


def test_apply_multi_leg_args_from_constitution(tmp_path, monkeypatch) -> None:
    from src.live_data_stream.constitution_config import (
        apply_multi_leg_args_from_constitution,
    )

    y = tmp_path / "constitution.yaml"
    y.write_text(
        """
multi_leg:
  strategies: "chop_grid"
  unit_notional: 50
  account:
    equity_usdt: 1000
    max_drawdown_pct: 0.2
  risk_limits:
    max_gross_notional_pct: 0.099
    max_resting_orders: 3
"""
    )
    monkeypatch.setenv("MLBOT_CONSTITUTION_YAML", str(y))
    monkeypatch.setenv("MLBOT_STRATEGIES_ROOT", str(tmp_path / "strategies"))
    (tmp_path / "strategies").mkdir()

    args = argparse.Namespace(
        strategies="trend_scalp",
        unit_notional=100.0,
        max_gross_notional=2000.0,
        max_net_notional=1000.0,
        max_symbol_gross_notional=800.0,
        max_symbol_net_notional=400.0,
        max_resting_orders=60,
        account_equity_usdt=10000.0,
        max_drawdown_pct=0.12,
        constitution_yaml="",
    )
    apply_multi_leg_args_from_constitution(args)
    assert args.strategies == "chop_grid"
    assert args.unit_notional == 50.0
    assert args.max_gross_notional == 99.0
    assert args.max_resting_orders == 3
    assert args.account_equity_usdt == 1000.0
    assert args.max_drawdown_pct == 0.2


def test_apply_multi_leg_empty_section_noop(tmp_path, monkeypatch) -> None:
    from src.live_data_stream.constitution_config import (
        apply_multi_leg_args_from_constitution,
    )

    y = tmp_path / "c.yaml"
    y.write_text("version: 1\n")
    monkeypatch.setenv("MLBOT_CONSTITUTION_YAML", str(y))
    monkeypatch.setenv("MLBOT_STRATEGIES_ROOT", str(tmp_path / "strategies"))
    (tmp_path / "strategies").mkdir()

    args = argparse.Namespace(
        strategies="chop_grid",
        unit_notional=100.0,
        max_gross_notional=2000.0,
        max_net_notional=1000.0,
        max_symbol_gross_notional=800.0,
        max_symbol_net_notional=400.0,
        max_resting_orders=60,
        account_equity_usdt=10000.0,
        max_drawdown_pct=0.12,
        constitution_yaml="",
    )
    apply_multi_leg_args_from_constitution(args)
    assert args.strategies == "chop_grid"


def test_apply_multi_leg_strategies_as_yaml_list(tmp_path, monkeypatch) -> None:
    from src.live_data_stream.constitution_config import (
        apply_multi_leg_args_from_constitution,
    )

    y = tmp_path / "constitution.yaml"
    y.write_text(
        """
multi_leg:
  strategies:
    - chop_grid
    - trend_scalp
  unit_notional: 10
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MLBOT_CONSTITUTION_YAML", str(y))
    monkeypatch.setenv("MLBOT_STRATEGIES_ROOT", str(tmp_path / "strategies"))
    (tmp_path / "strategies").mkdir()

    args = argparse.Namespace(
        strategies="solo",
        unit_notional=100.0,
        max_gross_notional=2000.0,
        max_net_notional=1000.0,
        max_symbol_gross_notional=800.0,
        max_symbol_net_notional=400.0,
        max_resting_orders=60,
        account_equity_usdt=10000.0,
        max_drawdown_pct=0.12,
        constitution_yaml="",
    )
    apply_multi_leg_args_from_constitution(args)
    assert args.strategies == "chop_grid,trend_scalp"
    assert args.unit_notional == 10.0
