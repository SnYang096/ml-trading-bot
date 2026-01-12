from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class EscalationEligibility:
    min_months_in_control: int = 3
    max_recent_dd: float = 0.12
    min_profit_factor: float = 1.2
    require_no_kill_switch_trigger: bool = True
    require_equity_ath: bool = True


@dataclass(frozen=True)
class LateBullLeverageConfig:
    phase_required: int = 3
    min_trend_duration: int = 30  # bars
    min_trend_stability: float = 0.7
    require_equity_ath: bool = True


def _get_float(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def eligible_for_escalation(
    metrics: Dict[str, Any],
    *,
    cfg: EscalationEligibility = EscalationEligibility(),
    now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """
    Best-effort eligibility check based on available metrics.
    This is intentionally low-freedom and can be tightened later.
    """
    _ = now or datetime.now(timezone.utc)
    # Required proxy keys (these should be produced by portfolio layer in future; for now use best-effort).
    recent_dd = _get_float(
        metrics, "rule_pcm_avg_max_dd", _get_float(metrics, "rule_avg_max_dd", 1.0)
    )
    profit_factor = _get_float(metrics, "portfolio_profit_factor", 0.0)
    equity_ath = bool(metrics.get("portfolio_equity_ath", False))
    kill = bool(metrics.get("kill_switch_triggered", False))

    if cfg.require_no_kill_switch_trigger and kill:
        return False, "kill_switch_triggered"
    if cfg.require_equity_ath and not equity_ath:
        return False, "equity_not_ath"
    if recent_dd > float(cfg.max_recent_dd):
        return False, "recent_dd_too_high"
    if profit_factor < float(cfg.min_profit_factor):
        return False, "profit_factor_too_low"

    # min_months_in_control is left as an external governance input; accept if unknown.
    return True, "eligible"


def late_bull_leverage_allowed(
    metrics: Dict[str, Any],
    *,
    cfg: LateBullLeverageConfig = LateBullLeverageConfig(),
) -> Tuple[bool, str]:
    """
    Contract:
    - leverage is only allowed in mature trend (phase=3)
    - never allow leverage for new entries (this function is only for *scaling* existing exposure)
    """
    phase = int(metrics.get("bull_phase", 0) or 0)
    trend_dur = int(metrics.get("trend_duration_bars", 0) or 0)
    trend_stab = _get_float(metrics, "trend_stability", 0.0)
    equity_ath = bool(metrics.get("portfolio_equity_ath", False))

    if phase < int(cfg.phase_required):
        return False, "phase_not_mature"
    if trend_dur < int(cfg.min_trend_duration):
        return False, "trend_duration_too_short"
    if trend_stab < float(cfg.min_trend_stability):
        return False, "trend_stability_too_low"
    if cfg.require_equity_ath and not equity_ath:
        return False, "equity_not_ath"
    return True, "allowed"
