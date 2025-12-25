from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RewardConfig:
    """
    Risk-adjusted per-step reward.

    This is intentionally simple and decomposable for debugging.
    """

    pnl_weight: float = 1.0
    cost_weight: float = 1.0
    turnover_weight: float = 0.0
    drawdown_weight: float = 0.0
    vol_weight: float = 0.0

    # Drawdown control (optional)
    dd_limit: float = 1.0  # normalized ratio (e.g., current_dd / dd_budget)
    dd_power: float = 2.0  # quadratic by default

    # Action stability (optional)
    action_change_weight: float = 0.0

    # Strategy diversity / anti-collapse (optional)
    diversity_weight: float = 0.0

    # Optional: cap extreme per-step reward
    clip_abs: Optional[float] = None


def compute_step_reward(
    *,
    pnl: float,
    cost: float = 0.0,
    turnover: float = 0.0,
    drawdown_increment: float = 0.0,
    realized_vol: float = 0.0,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """
    Compute per-step reward for Router/Allocator RL.

    Convention:
    - pnl should already be net-of-fees/slippage if possible; if not, include in cost.
    - drawdown_increment should be >=0 when drawdown worsens, else 0.
    """
    r = (
        cfg.pnl_weight * float(pnl)
        - cfg.cost_weight * float(cost)
        - cfg.turnover_weight * float(turnover)
        - cfg.drawdown_weight * float(drawdown_increment)
        - cfg.vol_weight * float(realized_vol)
    )
    if cfg.clip_abs is not None:
        ca = float(cfg.clip_abs)
        if ca > 0:
            r = max(min(r, ca), -ca)
    return float(r)


def compute_drawdown_penalty(
    *,
    dd_ratio: float,
    dd_limit: float,
    dd_weight: float,
    dd_power: float = 2.0,
) -> float:
    """
    Penalty when drawdown ratio exceeds limit.
    dd_ratio: current_dd / allowed_dd (>=0)
    """
    if dd_weight == 0.0:
        return 0.0
    x = max(0.0, float(dd_ratio) - float(dd_limit))
    return float(dd_weight * (x ** float(dd_power)))


def compute_action_change_penalty(
    *,
    prev_action: dict,
    next_action: dict,
    weight: float,
) -> float:
    """
    L1 change penalty for action stability. Intended for continuous weights/multipliers.
    Uses shared keys and treats missing as 0.
    """
    if weight == 0.0:
        return 0.0

    # Flatten only numeric leaves for stability
    def _flat(a: dict) -> dict:
        out = {}
        for k, v in (a or {}).items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    try:
                        out[f"{k}.{kk}"] = float(vv)
                    except Exception:
                        out[f"{k}.{kk}"] = 0.0
            else:
                try:
                    out[str(k)] = float(v)
                except Exception:
                    out[str(k)] = 0.0
        return out

    a0 = _flat(prev_action or {})
    a1 = _flat(next_action or {})
    keys = set(a0.keys()) | set(a1.keys())
    l1 = 0.0
    for k in keys:
        l1 += abs(float(a1.get(k, 0.0)) - float(a0.get(k, 0.0)))
    return float(weight * l1)


def compute_diversity_penalty(
    *,
    weights: dict,
    weight: float,
    eps: float = 1e-12,
) -> float:
    """
    Penalize strategy collapse using negative entropy of normalized weights.
    If weights sum to 0, no penalty (OFF state).
    """
    if weight == 0.0:
        return 0.0
    w = []
    for v in (weights or {}).values():
        try:
            w.append(max(0.0, float(v)))
        except Exception:
            w.append(0.0)
    s = float(sum(w))
    if s <= eps:
        return 0.0
    p = [x / s for x in w if x > eps]
    # entropy in [0, logK]; penalty uses (logK - entropy) to penalize collapse
    import math

    ent = -sum(pi * math.log(pi + eps) for pi in p)
    k = max(2, len(p))
    max_ent = math.log(k + eps)
    collapse = max(0.0, max_ent - ent)
    return float(weight * collapse)


def compute_router_reward_from_step(
    *,
    pnl: float,
    cost: float = 0.0,
    turnover: float = 0.0,
    dd_ratio: float = 0.0,
    realized_vol: float = 0.0,
    action_prev: Optional[dict] = None,
    action_next: Optional[dict] = None,
    action_weights_key: str = "weights",
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """
    Production-oriented reward for Router/Allocator layer.

    - Base: risk-adjusted pnl (optionally)
    - Penalties: drawdown, turnover/cost, action change, strategy collapse
    """
    base = compute_step_reward(
        pnl=pnl,
        cost=cost,
        turnover=turnover,
        drawdown_increment=0.0,  # handled via dd_ratio
        realized_vol=realized_vol,
        cfg=cfg,
    )

    dd_pen = compute_drawdown_penalty(
        dd_ratio=dd_ratio,
        dd_limit=cfg.dd_limit,
        dd_weight=cfg.drawdown_weight,
        dd_power=cfg.dd_power,
    )

    ac_pen = 0.0
    if (
        cfg.action_change_weight != 0.0
        and action_prev is not None
        and action_next is not None
    ):
        ac_pen = compute_action_change_penalty(
            prev_action=action_prev,
            next_action=action_next,
            weight=cfg.action_change_weight,
        )

    div_pen = 0.0
    if cfg.diversity_weight != 0.0 and action_next is not None:
        w = (
            action_next.get(action_weights_key, {})
            if isinstance(action_next, dict)
            else {}
        )
        if isinstance(w, dict):
            div_pen = compute_diversity_penalty(weights=w, weight=cfg.diversity_weight)

    r = float(base - dd_pen - ac_pen - div_pen)
    if cfg.clip_abs is not None:
        ca = float(cfg.clip_abs)
        if ca > 0:
            r = max(min(r, ca), -ca)
    return float(r)
