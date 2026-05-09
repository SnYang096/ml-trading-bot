from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


_PREFILTER_KEYS_BY_TYPE = {
    "grid": {
        "entry_chop_min",
        "exit_chop_below",
        "exclude_box_prefilter",
    },
    "dual_add_trend": {
        "entry_min",
        "exit_below",
        "max_semantic_chop_entry",
        "max_semantic_chop_hold",
    },
}

_EXECUTION_KEYS_BY_TYPE = {
    "grid": {
        "atr_mult",
        "min_pct",
        "max_levels_per_side",
        "max_open_levels_total",
        "max_segment_bars",
    },
    "dual_add_trend": {
        "step_atr_mult",
        "tp_atr_mult",
        "tp_pct",
        "flip_action",
        "max_gross_exposure",
        "max_net_exposure",
        "max_loss_per_segment",
    },
}


@dataclass(frozen=True)
class MultilegLayerSettings:
    strategy_type: str
    has_prefilter: bool
    has_gate: bool
    has_entry_filter: bool
    prefilter_optimize: bool
    gate_optimize: bool
    entry_filter_optimize: bool
    execution_optimize: bool

    @property
    def calibrate_any(self) -> bool:
        return bool(self.prefilter_optimize or self.execution_optimize)


def _bool_from_section(section: Any, key: str, default: bool = False) -> bool:
    if isinstance(section, Mapping):
        return bool(section.get(key, default))
    return bool(default)


def resolve_multileg_layer_settings(
    *,
    strategy_type: str,
    strategy_cfg: Mapping[str, Any],
    threshold_cfg: Mapping[str, Any],
    default_prefilter_optimize: bool,
    default_gate_optimize: bool,
    default_entry_filter_optimize: bool,
    default_execution_optimize: bool,
) -> MultilegLayerSettings:
    scfg = strategy_cfg or {}
    tcfg = threshold_cfg or {}
    has_prefilter = bool(scfg.get("has_prefilter", False))
    has_gate = bool(scfg.get("has_gate", False))
    has_entry_filter = bool(scfg.get("has_entry_filter", False))
    prefilter_optimize = has_prefilter and bool(
        _bool_from_section(
            tcfg.get("prefilter"), "optimize", default_prefilter_optimize
        )
    )
    gate_optimize = has_gate and bool(
        _bool_from_section(tcfg.get("gate"), "optimize", default_gate_optimize)
    )
    entry_filter_optimize = has_entry_filter and bool(
        _bool_from_section(
            tcfg.get("entry_filter"), "optimize", default_entry_filter_optimize
        )
    )
    execution_optimize = bool(
        _bool_from_section(
            tcfg.get("execution_opt"), "enabled", default_execution_optimize
        )
    )
    return MultilegLayerSettings(
        strategy_type=str(strategy_type or "").strip().lower(),
        has_prefilter=has_prefilter,
        has_gate=has_gate,
        has_entry_filter=has_entry_filter,
        prefilter_optimize=prefilter_optimize,
        gate_optimize=gate_optimize,
        entry_filter_optimize=entry_filter_optimize,
        execution_optimize=execution_optimize,
    )


def candidate_for_enabled_layers(
    *,
    strategy_type: str,
    candidate: Mapping[str, Any],
    settings: MultilegLayerSettings,
) -> Dict[str, Any]:
    st = str(strategy_type or "").strip().lower()
    base = dict(candidate or {})
    if not base:
        return {}
    keep = set()
    if settings.prefilter_optimize:
        keep.update(_PREFILTER_KEYS_BY_TYPE.get(st, set()))
    if settings.execution_optimize:
        keep.update(_EXECUTION_KEYS_BY_TYPE.get(st, set()))
    if not keep:
        return {}
    return {k: v for k, v in base.items() if k in keep}


def score_candidate_with_constraints(
    *,
    metrics: Mapping[str, Any],
    kpi_backtest: Mapping[str, Any] | None = None,
) -> float:
    m = metrics or {}
    total = float(m.get("total_r", 0.0) or 0.0)
    worst = float(m.get("worst_segment", 0.0) or 0.0)
    forced = float(m.get("forced_rate", 0.0) or 0.0)
    risk_stop = float(m.get("risk_stop_rate", m.get("near_stop_rate", 0.0)) or 0.0)
    score = total + 5.0 * worst - 0.25 * forced - 0.50 * risk_stop
    if m.get("cost_coverage_ratio") is not None:
        coverage = float(m.get("cost_coverage_ratio", 0.0) or 0.0)
        # For high-turnover grid engines, prefer spacing profiles whose gross edge
        # clears configured all-in costs instead of merely increasing trade count.
        score += max(min(coverage - 1.0, 2.0), -2.0) * 0.02
    gates = kpi_backtest or {}
    min_trades = int(gates.get("min_trades", gates.get("target_trades_min", 0)) or 0)
    if min_trades > 0 and int(m.get("n_trades", 0) or 0) < min_trades:
        score -= 1e6
    max_forced = gates.get("max_forced_exit_rate", gates.get("max_forced_rate", None))
    if max_forced is not None and forced > float(max_forced):
        score -= 1e6
    max_dd_floor = gates.get("max_drawdown_floor")
    if max_dd_floor is not None and float(m.get("max_drawdown_r", 0.0) or 0.0) < float(
        max_dd_floor
    ):
        score -= 1e6
    return score
