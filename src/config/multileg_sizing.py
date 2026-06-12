"""Derive per-strategy ``unit_notional`` from segment drawdown budgets.

Chop grid::

    segment_loss = max_loss_per_grid × (2 × max_levels_per_side) × unit_notional

Trend scalp::

    segment_loss = max_loss_per_segment × max_gross_exposure_units × unit_notional

Each strategy can set its own ``segment_dd_target`` under ``multi_leg.sizing``::

    sizing:
      segment_dd_target: 0.01          # legacy fallback for both
      chop_grid:
        segment_dd_target: 0.025
        max_loss_per_grid: 0.03
        max_levels_per_side: 3
      trend_scalp:
        segment_dd_target: 0.01
        max_loss_per_segment: 0.02
        max_gross_exposure_units: 4

Portfolio overlap is capped for both strategies together
(``max_concurrent_multi_leg_symbols``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

StrategyKey = str  # chop_grid | trend_scalp


def grid_capital_units(max_levels_per_side: int) -> int:
    return max(1, 2 * int(max_levels_per_side))


def trend_capital_units(max_gross_exposure_units: int) -> int:
    return max(1, int(max_gross_exposure_units))


def unit_notional_from_segment_dd(
    *,
    equity_usdt: float,
    segment_dd_target: float,
    max_loss_per_grid: float,
    max_levels_per_side: int,
) -> float:
    """USDT per chop grid level from a per-segment equity drawdown budget."""
    cap = grid_capital_units(max_levels_per_side)
    loss = abs(float(max_loss_per_grid))
    if loss <= 0.0:
        raise ValueError(f"max_loss_per_grid must be > 0, got {max_loss_per_grid!r}")
    if float(segment_dd_target) <= 0.0:
        raise ValueError(
            f"segment_dd_target must be > 0, got {segment_dd_target!r}"
        )
    equity = float(equity_usdt)
    if equity <= 0.0:
        raise ValueError(f"equity_usdt must be > 0, got {equity_usdt!r}")
    return equity * float(segment_dd_target) / (loss * cap)


def unit_notional_from_trend_segment_dd(
    *,
    equity_usdt: float,
    segment_dd_target: float,
    max_loss_per_segment: float,
    max_gross_exposure_units: int,
) -> float:
    """USDT per trend scalp leg from a per-segment equity drawdown budget."""
    cap = trend_capital_units(max_gross_exposure_units)
    loss = abs(float(max_loss_per_segment))
    if loss <= 0.0:
        raise ValueError(
            f"max_loss_per_segment must be > 0, got {max_loss_per_segment!r}"
        )
    if float(segment_dd_target) <= 0.0:
        raise ValueError(
            f"segment_dd_target must be > 0, got {segment_dd_target!r}"
        )
    equity = float(equity_usdt)
    if equity <= 0.0:
        raise ValueError(f"equity_usdt must be > 0, got {equity_usdt!r}")
    return equity * float(segment_dd_target) / (loss * cap)


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def chop_grid_execution_sizing_params(
    execution_yaml: Path,
) -> Dict[str, float]:
    """Read ``max_loss_per_grid`` and ``max_levels_per_side`` from execution.yaml."""
    doc = _load_yaml_mapping(execution_yaml)
    inv = doc.get("inventory") if isinstance(doc.get("inventory"), dict) else {}
    risk = doc.get("risk") if isinstance(doc.get("risk"), dict) else {}
    return {
        "max_loss_per_grid": float(risk.get("max_loss_per_grid", 0.03)),
        "max_levels_per_side": float(inv.get("max_levels_per_side", 3)),
    }


def trend_scalp_execution_sizing_params(
    execution_yaml: Path,
) -> Dict[str, float]:
    doc = _load_yaml_mapping(execution_yaml)
    inv = doc.get("inventory") if isinstance(doc.get("inventory"), dict) else {}
    risk = doc.get("risk") if isinstance(doc.get("risk"), dict) else {}
    return {
        "max_loss_per_segment": float(risk.get("max_loss_per_segment", 0.02)),
        "max_gross_exposure_units": float(inv.get("max_gross_exposure_units", 4)),
    }


def _segment_dd_for_strategy(
    sizing: Mapping[str, Any],
    strategy: StrategyKey,
) -> Optional[float]:
    block = sizing.get(strategy)
    if isinstance(block, dict) and block.get("segment_dd_target") is not None:
        return float(block["segment_dd_target"])
    if sizing.get("segment_dd_target") is not None:
        return float(sizing["segment_dd_target"])
    return None


def resolve_chop_grid_unit_notional(
    ml: Mapping[str, Any],
    *,
    equity_usdt: float,
    chop_grid_execution_path: Optional[Path] = None,
) -> float:
    if ml.get("unit_notional") is not None and not isinstance(ml.get("sizing"), dict):
        return float(ml["unit_notional"])

    sizing = ml.get("sizing")
    if not isinstance(sizing, dict):
        raise ValueError(
            "multi_leg requires unit_notional or sizing.segment_dd_target for chop_grid"
        )
    segment_dd = _segment_dd_for_strategy(sizing, "chop_grid")
    if segment_dd is None:
        raise ValueError(
            "multi_leg.sizing.chop_grid.segment_dd_target (or sizing.segment_dd_target) required"
        )

    cg = sizing.get("chop_grid")
    if isinstance(cg, dict) and cg:
        max_loss = float(cg.get("max_loss_per_grid", 0.03))
        max_levels = int(cg.get("max_levels_per_side", 3))
    elif chop_grid_execution_path is not None:
        params = chop_grid_execution_sizing_params(chop_grid_execution_path)
        max_loss = float(params["max_loss_per_grid"])
        max_levels = int(params["max_levels_per_side"])
    else:
        max_loss = 0.03
        max_levels = 3

    return unit_notional_from_segment_dd(
        equity_usdt=float(equity_usdt),
        segment_dd_target=float(segment_dd),
        max_loss_per_grid=max_loss,
        max_levels_per_side=max_levels,
    )


def resolve_trend_scalp_unit_notional(
    ml: Mapping[str, Any],
    *,
    equity_usdt: float,
    trend_scalp_execution_path: Optional[Path] = None,
) -> float:
    if ml.get("unit_notional") is not None and not isinstance(ml.get("sizing"), dict):
        return float(ml["unit_notional"])

    sizing = ml.get("sizing")
    if not isinstance(sizing, dict):
        raise ValueError(
            "multi_leg requires unit_notional or sizing.segment_dd_target for trend_scalp"
        )
    segment_dd = _segment_dd_for_strategy(sizing, "trend_scalp")
    if segment_dd is None:
        raise ValueError(
            "multi_leg.sizing.trend_scalp.segment_dd_target (or sizing.segment_dd_target) required"
        )

    ts = sizing.get("trend_scalp")
    if isinstance(ts, dict) and ts:
        max_loss = float(ts.get("max_loss_per_segment", 0.02))
        max_units = int(ts.get("max_gross_exposure_units", 4))
    elif trend_scalp_execution_path is not None:
        params = trend_scalp_execution_sizing_params(trend_scalp_execution_path)
        max_loss = float(params["max_loss_per_segment"])
        max_units = int(params["max_gross_exposure_units"])
    else:
        max_loss = 0.02
        max_units = 4

    return unit_notional_from_trend_segment_dd(
        equity_usdt=float(equity_usdt),
        segment_dd_target=float(segment_dd),
        max_loss_per_segment=max_loss,
        max_gross_exposure_units=max_units,
    )


def resolve_multi_leg_unit_notionals(
    ml: Mapping[str, Any],
    *,
    equity_usdt: float,
    chop_grid_execution_path: Optional[Path] = None,
    trend_scalp_execution_path: Optional[Path] = None,
    strategies: Optional[list[str]] = None,
) -> Dict[str, float]:
    """Return per-strategy leg notional (USDT)."""
    if ml.get("unit_notional") is not None:
        unit = float(ml["unit_notional"])
        keys = strategies or ["chop_grid", "trend_scalp"]
        return {k: unit for k in keys}

    out: Dict[str, float] = {}
    want = set(strategies or ["chop_grid", "trend_scalp"])
    if "chop_grid" in want:
        out["chop_grid"] = resolve_chop_grid_unit_notional(
            ml,
            equity_usdt=float(equity_usdt),
            chop_grid_execution_path=chop_grid_execution_path,
        )
    if "trend_scalp" in want:
        out["trend_scalp"] = resolve_trend_scalp_unit_notional(
            ml,
            equity_usdt=float(equity_usdt),
            trend_scalp_execution_path=trend_scalp_execution_path,
        )
    return out


def resolve_multi_leg_unit_notional(
    ml: Mapping[str, Any],
    *,
    equity_usdt: float,
    chop_grid_execution_path: Optional[Path] = None,
    trend_scalp_execution_path: Optional[Path] = None,
    strategy: StrategyKey = "chop_grid",
) -> float:
    """Resolve leg notional for one strategy (backward compatible)."""
    if ml.get("unit_notional") is not None:
        return float(ml["unit_notional"])

    if strategy == "trend_scalp":
        return resolve_trend_scalp_unit_notional(
            ml,
            equity_usdt=float(equity_usdt),
            trend_scalp_execution_path=trend_scalp_execution_path,
        )
    return resolve_chop_grid_unit_notional(
        ml,
        equity_usdt=float(equity_usdt),
        chop_grid_execution_path=chop_grid_execution_path,
    )


def max_concurrent_multi_leg_symbols_from_ml(ml: Mapping[str, Any]) -> Optional[int]:
    rs = ml.get("risk_limits")
    if not isinstance(rs, dict):
        return None
    raw = rs.get("max_concurrent_multi_leg_symbols")
    if raw is None:
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None
