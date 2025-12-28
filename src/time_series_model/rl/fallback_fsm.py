from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class RouterControlState(str, Enum):
    RULE = "RULE"
    RL_CANDIDATE = "RL_CANDIDATE"  # shadow mode only
    RL_ACTIVE = "RL_ACTIVE"  # allowed to control execution
    RL_SUSPENDED = "RL_SUSPENDED"  # forced back to RULE for cooldown


@dataclass(frozen=True)
class GateConfig:
    """
    Production gates for RL rollout / fallback.

    Metrics are expected to be computed from shadow/counterfactual evaluation outputs
    over a rolling window (e.g., last N days/weeks).
    """

    # Hard safety gates (trigger immediate suspension)
    dd_ratio_max: float = 1.2  # max_dd_RL > max_dd_Rule * dd_ratio_max
    worst5_ratio_min: float = 0.8  # worst5_RL < worst5_Rule * worst5_ratio_min
    switch_ratio_max: float = (
        2.0  # switch_rate_RL > switch_rate_Rule * switch_ratio_max
    )

    # Hard: risk-adjusted performance deterioration (optional but recommended)
    sharpe_ratio_min: float = 0.8  # sharpe_RL < sharpe_Rule * sharpe_ratio_min
    sharpe_min_abs: Optional[float] = (
        None  # if set: sharpe_RL < sharpe_min_abs triggers hard
    )
    sortino_ratio_min: float = 0.8
    sortino_min_abs: Optional[float] = None

    # Hard: realized volatility blow-up (optional)
    ann_vol_ratio_max: float = 2.0  # vol_RL > vol_Rule * ann_vol_ratio_max

    # Drift gate on efficiency (slow variable)
    pnl_dd_margin: float = 0.15  # (PnL/DD)_RL < (PnL/DD)_Rule * (1 - pnl_dd_margin)

    # Candidate promotion thresholds
    promote_min_days: int = 10  # consecutive ok windows to promote to ACTIVE

    # Suspension cooldown
    cooldown_days: int = 20


@dataclass(frozen=True)
class GateInputs:
    """
    Minimal set of inputs for gates (per rolling window).
    """

    # Core risk/quality
    max_dd_rule: float
    max_dd_rl: float
    worst5_rule: Optional[float] = None
    worst5_rl: Optional[float] = None

    # Behavior
    switch_rate_rule: float = 0.0
    switch_rate_rl: float = 0.0

    # Efficiency
    pnl_dd_rule: Optional[float] = None
    pnl_dd_rl: Optional[float] = None

    # Risk-adjusted performance (optional)
    sharpe_rule: Optional[float] = None
    sharpe_rl: Optional[float] = None
    sortino_rule: Optional[float] = None
    sortino_rl: Optional[float] = None
    ann_vol_rule: Optional[float] = None
    ann_vol_rl: Optional[float] = None


def evaluate_gates(inp: GateInputs, *, cfg: GateConfig) -> Tuple[bool, Dict[str, bool]]:
    """
    Return (hard_triggered, flags).
    """
    flags: Dict[str, bool] = {}

    # Hard: drawdown deterioration
    dd_bad = False
    if inp.max_dd_rule > 0:
        dd_bad = float(inp.max_dd_rl) > float(inp.max_dd_rule) * float(cfg.dd_ratio_max)
    flags["hard_dd"] = bool(dd_bad)

    # Hard: tail deterioration (optional)
    tail_bad = False
    if inp.worst5_rule is not None and inp.worst5_rl is not None:
        # worst5 is negative; "worse" means more negative
        tail_bad = float(inp.worst5_rl) < float(inp.worst5_rule) * float(
            cfg.worst5_ratio_min
        )
    flags["hard_tail"] = bool(tail_bad)

    # Hard: excessive switching
    switch_bad = False
    if inp.switch_rate_rule > 0:
        switch_bad = float(inp.switch_rate_rl) > float(inp.switch_rate_rule) * float(
            cfg.switch_ratio_max
        )
    flags["hard_switch"] = bool(switch_bad)

    # Hard: Sharpe / Sortino deterioration (optional)
    sharpe_bad = False
    if inp.sharpe_rl is not None:
        if cfg.sharpe_min_abs is not None:
            sharpe_bad = float(inp.sharpe_rl) < float(cfg.sharpe_min_abs)
        if (
            (not sharpe_bad)
            and inp.sharpe_rule is not None
            and float(inp.sharpe_rule) != 0.0
        ):
            sharpe_bad = float(inp.sharpe_rl) < float(inp.sharpe_rule) * float(
                cfg.sharpe_ratio_min
            )
    flags["hard_sharpe"] = bool(sharpe_bad)

    sortino_bad = False
    if inp.sortino_rl is not None:
        if cfg.sortino_min_abs is not None:
            sortino_bad = float(inp.sortino_rl) < float(cfg.sortino_min_abs)
        if (
            (not sortino_bad)
            and inp.sortino_rule is not None
            and float(inp.sortino_rule) != 0.0
        ):
            sortino_bad = float(inp.sortino_rl) < float(inp.sortino_rule) * float(
                cfg.sortino_ratio_min
            )
    flags["hard_sortino"] = bool(sortino_bad)

    vol_bad = False
    if (
        inp.ann_vol_rule is not None
        and inp.ann_vol_rl is not None
        and float(inp.ann_vol_rule) > 0
    ):
        vol_bad = float(inp.ann_vol_rl) > float(inp.ann_vol_rule) * float(
            cfg.ann_vol_ratio_max
        )
    flags["hard_vol"] = bool(vol_bad)

    hard_triggered = bool(
        dd_bad or tail_bad or switch_bad or sharpe_bad or sortino_bad or vol_bad
    )
    flags["hard_triggered"] = hard_triggered

    # Drift gate (slow): pnl/dd efficiency drop (optional)
    drift_bad = False
    if (
        inp.pnl_dd_rule is not None
        and inp.pnl_dd_rl is not None
        and float(inp.pnl_dd_rule) != 0.0
    ):
        drift_bad = float(inp.pnl_dd_rl) < float(inp.pnl_dd_rule) * (
            1.0 - float(cfg.pnl_dd_margin)
        )
    flags["drift_bad"] = bool(drift_bad)

    return hard_triggered, flags


@dataclass
class FallbackFSM:
    """
    Stateful rollout controller for RL Router.

    - RULE: execute rule router
    - RL_CANDIDATE: shadow only; evaluate gates and accumulate "ok streak"
    - RL_ACTIVE: execute RL; if hard gate triggers -> SUSPENDED
    - RL_SUSPENDED: forced back to RULE for cooldown; after cooldown -> CANDIDATE
    """

    cfg: GateConfig = GateConfig()
    state: RouterControlState = RouterControlState.RULE

    # internal counters
    ok_streak_days: int = 0
    cooldown_left_days: int = 0

    def step(self, inp: GateInputs) -> Dict[str, object]:
        hard, flags = evaluate_gates(inp, cfg=self.cfg)

        transition_reason: Optional[str] = None

        if self.state == RouterControlState.RL_ACTIVE:
            if hard or flags.get("drift_bad", False):
                self.state = RouterControlState.RL_SUSPENDED
                self.cooldown_left_days = int(self.cfg.cooldown_days)
                self.ok_streak_days = 0
                transition_reason = "suspend_hard" if hard else "suspend_drift"

        elif self.state == RouterControlState.RL_SUSPENDED:
            self.cooldown_left_days = max(0, int(self.cooldown_left_days) - 1)
            if self.cooldown_left_days <= 0:
                self.state = RouterControlState.RL_CANDIDATE
                transition_reason = "cooldown_done"

        elif self.state == RouterControlState.RL_CANDIDATE:
            if hard or flags.get("drift_bad", False):
                # stay candidate but reset streak; candidate never executes
                self.ok_streak_days = 0
                transition_reason = "candidate_not_ok"
            else:
                self.ok_streak_days += 1
                if self.ok_streak_days >= int(self.cfg.promote_min_days):
                    self.state = RouterControlState.RL_ACTIVE
                    transition_reason = "promoted"

        elif self.state == RouterControlState.RULE:
            # RULE can be moved into candidate by operator. If operator already did, no-op here.
            self.ok_streak_days = 0
            self.cooldown_left_days = 0

        return {
            "state": self.state.value,
            "ok_streak_days": int(self.ok_streak_days),
            "cooldown_left_days": int(self.cooldown_left_days),
            "transition_reason": transition_reason,
            "gates": flags,
        }
