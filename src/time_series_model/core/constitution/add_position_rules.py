from __future__ import annotations

from typing import Any, Dict, Mapping


def _strategy_keys(archetype: str) -> list[str]:
    key = str(archetype or "").strip().lower()
    if not key:
        return []
    parts = [p for p in key.split("-") if p]
    keys: list[str] = []
    seen: set[str] = set()

    def _push(k: str) -> None:
        kk = str(k or "").strip().lower()
        if not kk or kk in seen:
            return
        seen.add(kk)
        keys.append(kk)

    _push(key)

    # strip timeframe suffix: "*-60t" / "*-240t"
    if parts and parts[-1].endswith("t") and parts[-1][:-1].isdigit():
        parts_no_tf = parts[:-1]
        _push("-".join(parts_no_tf))
    else:
        parts_no_tf = parts

    # family-direction key: "<family>-<long|short>"
    if len(parts_no_tf) >= 2 and parts_no_tf[1] in {"long", "short"}:
        _push("-".join(parts_no_tf[:2]))

    # family only
    if parts_no_tf:
        _push(parts_no_tf[0])

    return keys


def _as_dict(obj: Any) -> Dict[str, Any]:
    return dict(obj) if isinstance(obj, dict) else {}


def _value_by_add_number(
    raw: Any,
    add_number: int,
    default: float,
) -> float:
    if add_number <= 0:
        return default
    vals = raw
    if isinstance(vals, (int, float)):
        vals = [vals]
    if not isinstance(vals, list) or not vals:
        return default
    idx = min(add_number - 1, len(vals) - 1)
    try:
        return float(vals[idx])
    except Exception:
        return default


def resolve_strategy_add_position_config(
    *,
    archetype: str,
    add_position_rules: Mapping[str, Any] | None,
    per_strategy_limits: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Merge global add rules with strategy-specific overrides."""
    base = _as_dict(add_position_rules)
    limits = _as_dict(per_strategy_limits)
    merged = dict(base)
    strat_cfg: Dict[str, Any] = {}
    for key in _strategy_keys(archetype):
        cand = _as_dict(limits.get(key))
        if cand:
            strat_cfg = cand
            break

    strat_add = _as_dict(strat_cfg.get("add_position"))
    for field in ("max_add_times", "add_size_multipliers", "min_current_r_by_add"):
        if field in strat_cfg and field not in strat_add:
            strat_add[field] = strat_cfg[field]

    trigger_cfg = _as_dict(base.get("trigger"))
    trigger_cfg.update(_as_dict(strat_add.get("trigger")))
    if trigger_cfg:
        merged["trigger"] = trigger_cfg

    merged.update({k: v for k, v in strat_add.items() if k != "trigger"})
    return merged


def resolve_float_r_ladder_only(add_position_cfg: Mapping[str, Any] | None) -> bool:
    """True iff ``add_position.trigger.type == "float_r_ladder_only"`` (事件回测浮盈阶梯路径)."""
    trig = _as_dict(_as_dict(add_position_cfg).get("trigger"))
    return str(trig.get("type", "")).strip().lower() == "float_r_ladder_only"


def resolve_add_position_size_multiplier(
    add_position_cfg: Mapping[str, Any] | None,
    add_number: int,
    signal: Mapping[str, Any] | None = None,
) -> float:
    if add_number <= 0:
        return 1.0
    cfg = _as_dict(add_position_cfg)
    sizing_mode = str(cfg.get("sizing_mode", "fixed_multiplier")).strip().lower()
    if sizing_mode == "target_leverage_gap":
        sig = dict(signal or {})
        target_lev = _value_by_add_number(
            cfg.get("target_leverage_by_add"), add_number, 0.0
        )
        current_lev = 0.0
        try:
            current_lev = max(0.0, float(sig.get("current_leverage", 0.0) or 0.0))
        except Exception:
            current_lev = 0.0
        base_lev = 1.0
        try:
            base_lev = max(
                1e-6,
                float(sig.get("base_leverage_unit", 1.0) or 1.0),
            )
        except Exception:
            base_lev = 1.0
        gap = max(0.0, target_lev - current_lev)
        mult = gap / base_lev
        max_total_lev = _value_by_add_number(
            cfg.get("max_total_leverage"), add_number, 0.0
        )
        if max_total_lev > 0:
            lev_room = max(0.0, max_total_lev - current_lev)
            mult = min(mult, lev_room / base_lev)
        max_step = _value_by_add_number(
            cfg.get("max_add_leverage_step"), add_number, 0.0
        )
        if max_step > 0:
            mult = min(mult, max_step / base_lev)
        max_notional_frac = _value_by_add_number(
            cfg.get("max_add_notional_frac"), add_number, 0.0
        )
        if max_notional_frac > 0:
            try:
                current_notional_frac = max(
                    0.0, float(sig.get("current_notional_frac", 0.0) or 0.0)
                )
                add_frac_cap = max(0.0, max_notional_frac - current_notional_frac)
                base_frac = max(
                    1e-9, float(sig.get("base_notional_frac", max_notional_frac))
                )
                mult = min(mult, add_frac_cap / base_frac)
            except Exception:
                pass
        min_add_usd = _value_by_add_number(
            cfg.get("min_add_notional_usd"), add_number, 0.0
        )
        if min_add_usd > 0:
            try:
                equity = max(0.0, float(sig.get("equity_usd", 0.0) or 0.0))
                base_frac = max(
                    1e-9, float(sig.get("base_notional_frac", max_notional_frac))
                )
                min_mult = min_add_usd / max(equity * base_frac, 1e-9)
                mult = max(mult, min_mult)
            except Exception:
                pass
        if mult <= 0:
            mult = _value_by_add_number(
                cfg.get("add_size_multipliers"), add_number, 1.0
            )
    else:
        mult = _value_by_add_number(cfg.get("add_size_multipliers"), add_number, 1.0)
    return mult if mult > 0 else 1.0


def resolve_add_position_max_times(
    add_position_cfg: Mapping[str, Any] | None,
) -> int:
    cfg = _as_dict(add_position_cfg)
    try:
        return max(0, int(cfg.get("max_add_times", 1)))
    except Exception:
        return 1


def resolve_add_position_min_current_r(
    add_position_cfg: Mapping[str, Any] | None,
    add_number: int,
    signal: Mapping[str, Any] | None = None,
) -> float:
    if add_number <= 0:
        return 0.0
    cfg = _as_dict(add_position_cfg)
    threshold_raw = max(
        0.0, _value_by_add_number(cfg.get("min_current_r_by_add"), add_number, 0.0)
    )

    # Unit semantics for min_current_r_by_add:
    # - initial_r (default): threshold is in current_r units
    # - atr: threshold is in ATR units and converted to current_r by parent_initial_r
    unit = (
        str(
            cfg.get(
                "min_current_r_unit", cfg.get("min_current_by_add_unit", "initial_r")
            )
            or "initial_r"
        )
        .strip()
        .lower()
    )
    if unit not in {"atr", "initial_r"}:
        unit = "initial_r"
    if unit != "atr":
        return threshold_raw

    sig = _as_dict(signal)
    try:
        parent_initial_r = float(sig.get("parent_initial_r", 0.0) or 0.0)
    except Exception:
        parent_initial_r = 0.0
    if parent_initial_r <= 0:
        # Safe fallback: if ATR conversion basis is unavailable, keep old behavior.
        return threshold_raw
    return max(0.0, threshold_raw / parent_initial_r)


def _get_num(signal: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in signal and signal.get(name) is not None:
            try:
                return float(signal.get(name))
            except Exception:
                continue
    return None


def validate_add_position_trigger(
    *,
    archetype: str,
    direction: int,
    signal: Mapping[str, Any],
    add_position_cfg: Mapping[str, Any] | None,
    current_r: float,
) -> bool:
    cfg = _as_dict(add_position_cfg)
    add_seq = int(signal.get("add_position_seq", 1) or 1)
    if current_r < resolve_add_position_min_current_r(cfg, add_seq, signal):
        return False
    trigger = _as_dict(cfg.get("trigger"))
    if not trigger:
        return True

    trig_type = str(trigger.get("type", "")).strip().lower()
    # 与 resolve_float_r_ladder_only 一致：阶梯模式由事件回测单独路径处理，此处不附加特征条件。
    if trig_type == "float_r_ladder_only":
        return True
    # BPC: 加仓触发 — breakout 方向与持仓一致（0 视为中性）。
    if trig_type in {"bpc_follow_signal", "follow_signal"}:
        breakout_dir = _get_num(signal, "bpc_breakout_direction")
        if breakout_dir is not None and int(breakout_dir) not in (0, direction):
            return False
        return True

    if trig_type == "me_momentum_expand":
        atr_pct = _get_num(signal, "atr_percentile")
        if atr_pct is not None:
            if atr_pct < float(trigger.get("atr_percentile_min", 0.20)):
                return False
            if atr_pct > float(trigger.get("atr_percentile_max", 0.95)):
                return False
        decay = _get_num(signal, "recent_compression_decay")
        if decay is not None:
            if decay < float(trigger.get("recent_compression_decay_min", 0.01)):
                return False
            if decay > float(trigger.get("recent_compression_decay_max", 0.35)):
                return False
        comp = _get_num(signal, "compression_duration")
        if comp is not None and comp < float(
            trigger.get("compression_duration_min", 0.03)
        ):
            return False
        oi_comp = _get_num(signal, "oi_compression_score")
        if oi_comp is not None and oi_comp < float(
            trigger.get("oi_compression_score_min", 0.35)
        ):
            return False
        dual_ign = _get_num(signal, "dual_ignition_score")
        if dual_ign is not None and dual_ign < float(
            trigger.get("dual_ignition_score_min", 0.40)
        ):
            return False

        max_recent_extreme = float(trigger.get("breakout_recent_extreme_max", 0.10))
        if direction > 0:
            recent_high = _get_num(signal, "bars_since_local_high")
            if recent_high is not None:
                return recent_high <= max_recent_extreme
            return (_get_num(signal, "roc_5") or 0.0) > 0 and (
                _get_num(signal, "macd_atr") or 0.0
            ) > 0
        if direction < 0:
            recent_low = _get_num(signal, "bars_since_local_low")
            if recent_low is not None:
                return recent_low <= max_recent_extreme
            return (_get_num(signal, "roc_5") or 0.0) < 0 and (
                _get_num(signal, "macd_atr") or 0.0
            ) < 0
        return False

    if trig_type == "me_atr_step_add":
        init_r = _get_num(signal, "parent_initial_r")
        if init_r is None or init_r <= 0:
            init_r = float(trigger.get("fallback_initial_r", 2.0))
        atr_step = float(trigger.get("atr_step", 0.5))
        add_seq = int(signal.get("add_position_seq", 1) or 1)
        required_r = max(0.0, atr_step * max(1, add_seq) / init_r)
        if current_r < required_r:
            return False

        if bool(trigger.get("require_momentum_sign", True)):
            roc5 = _get_num(signal, "roc_5")
            macd_atr = _get_num(signal, "macd_atr")
            if direction > 0:
                if roc5 is not None and roc5 <= 0:
                    return False
                if macd_atr is not None and macd_atr <= 0:
                    return False
            elif direction < 0:
                if roc5 is not None and roc5 >= 0:
                    return False
                if macd_atr is not None and macd_atr >= 0:
                    return False

        # 优先用 bars_since_local_high/low 约束“新高/新低附近”触发，缺失时仅用 step 条件
        max_recent_extreme = float(trigger.get("breakout_recent_extreme_max", 0.10))
        if direction > 0:
            recent_high = _get_num(signal, "bars_since_local_high")
            if recent_high is not None and recent_high > max_recent_extreme:
                return False
        elif direction < 0:
            recent_low = _get_num(signal, "bars_since_local_low")
            if recent_low is not None and recent_low > max_recent_extreme:
                return False
        return True

    return True
