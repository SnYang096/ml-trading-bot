"""spot_accum_simple: weekly deep-bear DCA, deploy decay, profit-multiple sell ladder."""

from __future__ import annotations

import math
from datetime import datetime, time, timezone
from typing import Any, Dict, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

SPOT_ACCUM_ARCHETYPES = frozenset({"spot_accum", "spot_accum_simple"})


def is_spot_accum_archetype(name: str) -> bool:
    return str(name or "").strip().lower() in SPOT_ACCUM_ARCHETYPES


def simple_accumulation_policy(raw_execution: Mapping[str, Any]) -> Dict[str, Any]:
    policy = raw_execution.get("simple_accumulation_policy") or {}
    return dict(policy) if isinstance(policy, dict) else {}


def deploy_schedule_policy(raw_execution: Mapping[str, Any]) -> Dict[str, Any]:
    policy = raw_execution.get("deploy_schedule") or {}
    return dict(policy) if isinstance(policy, dict) else {}


def _parse_hhmm(token: Any) -> Optional[time]:
    raw = str(token or "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        if len(parts) == 2:
            return time(int(parts[0]), int(parts[1]))
        if len(parts) == 3:
            return time(int(parts[0]), int(parts[1]), int(parts[2]))
    except (TypeError, ValueError):
        return None
    return None


def deploy_schedule_allows_new_buy(
    now: datetime,
    schedule: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Whether a new limit/market deploy may be submitted at ``now``.

    Config (execution.deploy_schedule):
      enabled, timezone (IANA, e.g. Europe/London),
      new_order_local_start / new_order_local_end (HH:MM, inclusive window in local TZ).

    Pending cancel age uses pending_max_age_hours (fallback: env MLBOT_SPOT_PENDING_BUY_MAX_HOURS).
    """
    if not schedule.get("enabled", False):
        return True, ""
    tz_name = str(schedule.get("timezone") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return True, f"invalid_timezone:{tz_name}"
    start_t = _parse_hhmm(schedule.get("new_order_local_start"))
    end_t = _parse_hhmm(schedule.get("new_order_local_end"))
    if start_t is None or end_t is None:
        return True, "schedule_window_unset"
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(tz)
    local_t = local_now.time()
    if start_t <= end_t:
        in_window = start_t <= local_t <= end_t
    else:
        # overnight window, e.g. 22:00–06:00
        in_window = local_t >= start_t or local_t <= end_t
    if not in_window:
        return (
            False,
            f"outside_deploy_window local={local_now.strftime('%H:%M')} "
            f"window={start_t.strftime('%H:%M')}-{end_t.strftime('%H:%M')} tz={tz_name}",
        )
    return True, ""


def pending_buy_max_age_hours(
    schedule: Mapping[str, Any], *, default: float = 24.0
) -> float:
    try:
        v = float(schedule.get("pending_max_age_hours", default) or default)
    except (TypeError, ValueError):
        v = default
    return max(1.0, v)


def profit_take_ladder_cfg(pos: Mapping[str, Any]) -> Dict[str, Any]:
    raw = pos.get("profit_take_ladder")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def deep_bear_allows_buy(
    features: Mapping[str, Any], policy: Mapping[str, Any]
) -> Tuple[bool, Optional[float]]:
    if not policy.get("enabled", False):
        return True, None
    feat = str(policy.get("regime_feature") or "weekly_ema_200_position").strip()
    try:
        pos_val = float(features.get(feat))
    except (TypeError, ValueError):
        return False, None
    if pos_val != pos_val:
        return False, pos_val
    # (close - ema) / close < 0  => price below weekly EMA200
    max_pos = float(policy.get("deep_bear_max_position", 0.0) or 0.0)
    return bool(pos_val < max_pos), pos_val


def deploy_decay_multiplier(
    deployed_quote_usdt: float,
    symbol_budget_usdt: float,
    decay_cfg: Optional[Mapping[str, Any]],
) -> float:
    if not isinstance(decay_cfg, dict) or not decay_cfg.get("enabled", False):
        return 1.0
    budget = float(symbol_budget_usdt or 0.0)
    if budget <= 0.0:
        return 1.0
    deployed_pct = 100.0 * max(0.0, float(deployed_quote_usdt)) / budget
    tiers = decay_cfg.get("tiers") or []
    if not isinstance(tiers, list):
        return 1.0
    ordered = sorted(
        tiers,
        key=lambda t: float(
            (t or {}).get("max_deployed_pct_exclusive", 999.0) or 999.0
        ),
    )
    for tier in ordered:
        if not isinstance(tier, dict):
            continue
        bound = tier.get("max_deployed_pct_exclusive")
        if bound is None:
            continue
        try:
            if deployed_pct < float(bound):
                return max(0.0, float(tier.get("unit_multiplier", 1.0) or 1.0))
        except (TypeError, ValueError):
            continue
    return 1.0


def resolve_min_profit_multiple(symbol: str, ladder: Mapping[str, Any]) -> float:
    per = ladder.get("per_symbol_min_profit_multiple") or {}
    sym = str(symbol or "").strip().upper()
    if isinstance(per, dict) and sym in per:
        try:
            return max(1.0, float(per[sym]))
        except (TypeError, ValueError):
            pass
    try:
        return max(1.0, float(ladder.get("min_profit_multiple", 5.0) or 5.0))
    except (TypeError, ValueError):
        return 5.0


def profit_ladder_speed_multiplier(
    mtm_multiple: float, trigger_multiple: float, accel: Mapping[str, Any]
) -> float:
    """Accelerate sells as mark-to-cost multiple rises above trigger.

    Default ``type: power``:
        speed = min(max_mult, (mtm / trigger) ** exponent)

    ``type: exponential`` (optional):
        speed = min(max_mult, exp(k * max(0, mtm/trigger - 1)))

    At trigger (mtm==trigger) speed is 1.0; above trigger sells faster.
    """
    if mtm_multiple < trigger_multiple or trigger_multiple <= 0.0:
        return 0.0
    try:
        max_mult = max(1.0, float(accel.get("max_speed_multiplier", 4.0) or 4.0))
    except (TypeError, ValueError):
        max_mult = 4.0
    ratio = float(mtm_multiple) / float(trigger_multiple)
    kind = str(accel.get("type") or "power").strip().lower()
    if kind == "exponential":
        try:
            k = float(accel.get("k", 0.35) or 0.35)
        except (TypeError, ValueError):
            k = 0.35
        return min(max_mult, float(math.exp(k * max(0.0, ratio - 1.0))))
    try:
        exp = float(accel.get("exponent", 0.75) or 0.75)
    except (TypeError, ValueError):
        exp = 0.75
    return min(max_mult, float(ratio**exp))


def maybe_spot_simple_partial_sell(
    pos: Dict[str, Any],
    *,
    price_close: float,
    now: datetime,
) -> Optional[Tuple[float, str]]:
    """Return (qty_to_sell, reason) when profit ladder allows a UTC-day slice."""
    if (
        str(pos.get("structural_exit") or "").strip().lower()
        != "spot_simple_profit_ladder"
    ):
        return None
    ladder = profit_take_ladder_cfg(pos)
    if not ladder.get("enabled", True):
        return None

    qty = float(pos.get("_qty_base", 0.0) or 0.0)
    cost = float(pos.get("_entry_notional_usdt", 0.0) or 0.0)
    if qty <= 0.0 or cost <= 0.0 or price_close <= 0.0:
        return None

    mtm = (qty * float(price_close)) / cost
    sym = str(pos.get("symbol") or "")
    trigger = resolve_min_profit_multiple(sym, ladder)
    if mtm < trigger:
        return None

    day_key = pd.Timestamp(now).strftime("%Y-%m-%d")
    if str(pos.get("_profit_ladder_last_sell_day") or "") == day_key:
        return None

    accel = ladder.get("acceleration") or {}
    if not isinstance(accel, dict):
        accel = {}
    speed = profit_ladder_speed_multiplier(mtm, trigger, accel)
    if speed <= 0.0:
        return None

    try:
        base_frac = float(ladder.get("base_daily_sell_fraction", 0.05) or 0.05)
    except (TypeError, ValueError):
        base_frac = 0.05
    base_frac = max(0.0, min(1.0, base_frac))
    sell_frac = min(1.0, base_frac * speed)
    sell_qty = qty * sell_frac
    min_qty = float(ladder.get("min_sell_qty", 0.0) or 0.0)
    if sell_qty < min_qty and sell_qty < qty:
        return None
    if sell_qty <= 0.0:
        return None

    reason = (
        f"spot_simple_profit_ladder|mtm={mtm:.2f}x|trigger={trigger:.1f}x|"
        f"speed={speed:.2f}|frac={sell_frac:.3f}"
    )
    return float(sell_qty), reason


def apply_partial_sell_to_position(
    pos: Dict[str, Any], *, sell_qty: float, exit_price: float
) -> None:
    """Reduce spot mother lot after partial take-profit (cost basis ∝ remaining qty)."""
    qty = float(pos.get("_qty_base", 0.0) or 0.0)
    cost = float(pos.get("_entry_notional_usdt", 0.0) or 0.0)
    sell_qty = min(max(0.0, sell_qty), qty)
    if sell_qty <= 0.0 or qty <= 0.0:
        return
    keep_ratio = (qty - sell_qty) / qty
    pos["_qty_base"] = float(qty - sell_qty)
    pos["_entry_notional_usdt"] = float(cost * keep_ratio)
    pos["_spot_quote_deployed"] = (
        float(pos.get("_spot_quote_deployed", cost) or cost) * keep_ratio
    )
    pos["_entry_fee_usdt"] = float(pos.get("_entry_fee_usdt", 0.0) or 0.0) * keep_ratio
    pos["_size_multiplier"] = (
        float(pos.get("_size_multiplier", 1.0) or 1.0) * keep_ratio
    )
    pos["_deploy_leg_count"] = max(1, int(pos.get("_deploy_leg_count", 1) or 1))
    pos["_last_deploy_price"] = float(exit_price)
