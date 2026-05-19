"""Read constitution YAML paths and shared sections (trend/fat-tail + hedge multi-leg + publisher)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Union

logger = logging.getLogger(__name__)
MULTI_LEG_STRATEGY_TYPES = frozenset({"grid", "dual_add_trend", "trend_scalp"})
SPOT_STRATEGY_TYPES = frozenset({"spot", "spot_accum"})


def resolve_constitution_yaml(
    strategies_root: str, *, override: Optional[str] = None
) -> str:
    if override and str(override).strip():
        return str(override).strip()
    config_root = os.path.join(strategies_root, "..")
    return os.getenv(
        "MLBOT_CONSTITUTION_YAML",
        os.path.join(config_root, "constitution", "constitution.yaml"),
    )


def load_constitution_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    import yaml

    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("读取 constitution 失败 %s: %s", path, exc)
        return {}


def enabled_archetypes_from_constitution(cfg: Dict[str, Any]) -> List[str]:
    """PCM 联合回测白名单 + trend/fat-tail LivePCM 注册候选（同一列表）。

    配置在 ``resource_allocation.enabled_archetypes``（或根级同名键）。
    推荐 YAML 显式 ``- bpc`` 列表；也接受逗号分隔字符串（与 ``multi_leg.strategies`` 一致）。
    缺省或空则返回内置全集（历史「未写白名单即全开」语义）。
    """
    _ALL = ["bpc", "me", "srb", "tpc", "lv", "fbf", "msr", "fer"]
    raw = (
        (cfg.get("resource_allocation") or {}).get("enabled_archetypes")
        or cfg.get("enabled_archetypes")
        or []
    )
    if raw is None or raw == "":
        return [a.lower() for a in _ALL]
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        return parts or [a.lower() for a in _ALL]
    if isinstance(raw, (list, tuple)):
        if not raw:
            return [a.lower() for a in _ALL]
        return [str(a).lower().strip() for a in raw if str(a).strip()]
    return [a.lower() for a in _ALL]


def enabled_archetypes_key_present(cfg: Dict[str, Any]) -> bool:
    """True when YAML explicitly sets enabled_archetypes (live/research whitelist)."""
    ra = cfg.get("resource_allocation") or {}
    return "enabled_archetypes" in ra or "enabled_archetypes" in cfg


def intent_archetype_priority_tokens(cfg: Dict[str, Any]) -> List[str]:
    """Tokens for PCM archetype ordering (same-bar intent sort + LivePCM.register order helper).

    If ``resource_allocation.intent_selection_policy.archetype_priority`` is set and non-empty,
    use it. Otherwise use ``enabled_archetypes`` list order (single source with membership).
    """
    ra = cfg.get("resource_allocation") or {}
    isp = ra.get("intent_selection_policy") or {}
    raw = isp.get("archetype_priority")
    if raw is not None and raw != "":
        if isinstance(raw, str):
            parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
            if parts:
                return parts
        elif isinstance(raw, (list, tuple)) and raw:
            return [str(x).lower().strip() for x in raw if str(x).strip()]
    return list(enabled_archetypes_from_constitution(cfg))


def multi_leg_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sec = cfg.get("multi_leg")
    return sec if isinstance(sec, dict) else {}


def multi_leg_strategies_from_constitution(cfg: Dict[str, Any]) -> List[str]:
    raw = (multi_leg_section(cfg) or {}).get("strategies")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    return []


def spot_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sec = cfg.get("spot")
    return sec if isinstance(sec, dict) else {}


def spot_strategies_from_constitution(cfg: Dict[str, Any]) -> List[str]:
    raw = (spot_section(cfg) or {}).get("strategies")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    return []


def spot_account_from_constitution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    account = (spot_section(cfg) or {}).get("account")
    return dict(account) if isinstance(account, dict) else {}


def spot_account_equity_anchor_usdt(
    account: Optional[Dict[str, Any]],
    *,
    default: float = 10000.0,
) -> float:
    """Offline equity anchor for spot backtest / deploy pct (live uses exchange sync).

    Accepts ``equity_usdt`` (canonical, same key as multi_leg.account) or legacy
    ``backtest_equity_usdt``.
    """
    if not isinstance(account, dict):
        return float(default)
    for key in ("equity_usdt", "backtest_equity_usdt"):
        raw = account.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return float(default)


def spot_strategy_limits_from_constitution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = (spot_section(cfg) or {}).get("strategy_limits")
    return dict(raw) if isinstance(raw, dict) else {}


def archetype_groups_from_constitution(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    ra = cfg.get("resource_allocation") or {}
    groups = ra.get("archetype_groups") or {}
    if not isinstance(groups, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for group, raw in groups.items():
        if isinstance(raw, str):
            vals = [p.strip().lower() for p in raw.split(",") if p.strip()]
        elif isinstance(raw, (list, tuple)):
            vals = [str(x).strip().lower() for x in raw if str(x).strip()]
        else:
            vals = []
        if vals:
            out[str(group).strip().lower()] = vals
    return out


def classic_slot_policy_from_constitution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ra = cfg.get("resource_allocation") or {}
    raw = ra.get("slot_policy") or {}
    policy = dict(raw) if isinstance(raw, dict) else {}
    groups = archetype_groups_from_constitution(cfg)
    group_name = str(policy.get("trend_group", "trend") or "trend").strip().lower()
    trend = groups.get(group_name, [])
    # Backward-compatible fallback for older local test fixtures only.
    if not trend and isinstance(policy.get("trend_archetypes"), (list, tuple, str)):
        legacy = policy.get("trend_archetypes")
        if isinstance(legacy, str):
            trend = [p.strip().lower() for p in legacy.split(",") if p.strip()]
        else:
            trend = [str(x).strip().lower() for x in legacy if str(x).strip()]
    # Trend slot pool: archetype_groups.trend ∩ enabled_archetypes, or enabled-only when
    # groups omitted (live constitution).
    if trend:
        enabled_set = set(enabled_archetypes_from_constitution(cfg))
        trend = [a for a in trend if a in enabled_set]
    elif enabled_archetypes_key_present(cfg):
        trend = list(enabled_archetypes_from_constitution(cfg))
    policy["trend_archetypes"] = trend
    policy["min_trend_slots_per_symbol"] = int(
        policy.get("min_trend_slots_per_symbol", 1) or 1
    )
    policy["max_trend_slots_per_symbol"] = int(
        policy.get(
            "max_trend_slots_per_symbol",
            1 if bool(policy.get("enforce_single_trend_per_symbol", False)) else 0,
        )
        or 0
    )
    return policy


def normalize_symbols_for_slot_validation(
    symbols: Optional[Union[str, Iterable[str]]],
) -> List[str]:
    """Coerce universe input to uppercase symbol tokens.

    Callers historically pass either a ``list[str]`` (live) or a comma-separated
    ``str`` from ``resolve_symbols_from_config`` (research); iterating a string
    byte-by-character would wrongly inflate slot requirements.
    """
    if symbols is None:
        return []
    if isinstance(symbols, str):
        raw = symbols.replace("|", ",").replace(";", ",")
        out: List[str] = []
        for chunk in raw.split(","):
            t = chunk.strip().upper()
            if t:
                out.append(t)
        return out
    return [str(s).strip().upper() for s in symbols if str(s).strip()]


def validate_classic_slot_capacity(
    *,
    constitution_cfg: Dict[str, Any],
    symbols: Optional[Union[str, Iterable[str]]],
) -> Dict[str, Any]:
    policy = classic_slot_policy_from_constitution(constitution_cfg)
    clean_symbols = sorted(set(normalize_symbols_for_slot_validation(symbols)))
    slots = constitution_cfg.get("slots") or {}
    slot_count = int(slots.get("slot_count", 0) or 0)
    min_per_symbol = int(policy.get("min_trend_slots_per_symbol", 1) or 1)
    required = len(clean_symbols) * max(min_per_symbol, 0)
    if required > 0 and slot_count < required:
        raise ValueError(
            "constitution slots.slot_count is too small for classic trend policy: "
            f"symbols={len(clean_symbols)} min_trend_slots_per_symbol={min_per_symbol} "
            f"required={required} slot_count={slot_count}"
        )
    return {
        "symbols": clean_symbols,
        "slot_count": slot_count,
        "required_trend_slots": required,
        "policy": policy,
    }


def _strategy_type_from_pipeline_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("strategy_type", "") or "").strip().lower()


def partition_pipeline_strategies_by_type(
    pipeline_cfg: Dict[str, Any],
) -> Dict[str, Set[str]]:
    raw = pipeline_cfg.get("strategies") or {}
    if not isinstance(raw, dict):
        return {"classic": set(), "multi_leg": set(), "spot": set()}
    classic: Set[str] = set()
    multi_leg: Set[str] = set()
    spot: Set[str] = set()
    for key, scfg in raw.items():
        name = str(key or "").strip().lower()
        if not name:
            continue
        stype = _strategy_type_from_pipeline_entry(scfg)
        if stype in MULTI_LEG_STRATEGY_TYPES:
            multi_leg.add(name)
        elif stype in SPOT_STRATEGY_TYPES:
            spot.add(name)
        else:
            classic.add(name)
    return {"classic": classic, "multi_leg": multi_leg, "spot": spot}


def validate_pipeline_constitution_alignment(
    *,
    pipeline_cfg: Dict[str, Any],
    constitution_cfg: Dict[str, Any],
    context_label: str = "rolling_research",
) -> Dict[str, List[str]]:
    """Ensure every strategy in the pipeline YAML is authorized by the constitution.

    Subset semantics (single-strategy research): ``enabled_archetypes`` /
    ``multi_leg`` / ``spot`` may list more than the pipeline; only
    **pipeline ⊂ constitution** is required.
    Violations: a classic name not in ``enabled_archetypes``, or a multi-leg name not
    in ``multi_leg.strategies``, or a spot name not in ``spot.strategies``.
    """
    parts = partition_pipeline_strategies_by_type(pipeline_cfg)
    pipeline_classic = set(parts["classic"])
    pipeline_multi_leg = set(parts["multi_leg"])
    pipeline_spot = set(parts.get("spot") or set())
    const_classic = set(enabled_archetypes_from_constitution(constitution_cfg))
    const_multi_leg = set(multi_leg_strategies_from_constitution(constitution_cfg))
    const_spot = set(spot_strategies_from_constitution(constitution_cfg))

    # Backward-compatible inference: if strategy_type is omitted in pipeline YAML,
    # spot strategies can still be recognized by constitution spot allowlist.
    inferred_spot = pipeline_classic & const_spot
    if inferred_spot:
        pipeline_spot |= inferred_spot
        pipeline_classic -= inferred_spot

    classic_missing_in_const = sorted(pipeline_classic - const_classic)
    multi_missing_in_const = sorted(pipeline_multi_leg - const_multi_leg)
    spot_missing_in_const = sorted(pipeline_spot - const_spot)

    if classic_missing_in_const or multi_missing_in_const or spot_missing_in_const:
        msg_lines = [
            f"{context_label}: pipeline strategy not allowed by constitution "
            "(pipeline must be a subset of constitution lists)",
            f"  classic pipeline={sorted(pipeline_classic)}",
            f"  classic constitution={sorted(const_classic)}",
            f"  multi_leg pipeline={sorted(pipeline_multi_leg)}",
            f"  multi_leg constitution={sorted(const_multi_leg)}",
            f"  spot pipeline={sorted(pipeline_spot)}",
            f"  spot constitution={sorted(const_spot)}",
        ]
        if classic_missing_in_const:
            msg_lines.append(
                "  not in constitution.enabled_archetypes="
                f"{classic_missing_in_const}"
            )
        if multi_missing_in_const:
            msg_lines.append(
                "  not in constitution.multi_leg.strategies="
                f"{multi_missing_in_const}"
            )
        if spot_missing_in_const:
            msg_lines.append(
                "  not in constitution.spot.strategies="
                f"{spot_missing_in_const}"
            )
        raise ValueError("\n".join(msg_lines))

    return {
        "classic": sorted(pipeline_classic),
        "multi_leg": sorted(pipeline_multi_leg),
        "spot": sorted(pipeline_spot),
    }


def _float_or_none(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _literal_from_env(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return _float_or_none(raw)


def load_multi_leg_backtest_risk_context(
    *,
    strategies_root: Optional[str] = None,
    constitution_yaml: Optional[str] = None,
    initial_capital: Optional[float] = None,
):
    """Build BacktestAccountRiskTracker + unit_notional for multi-leg research scripts."""
    from src.time_series_model.core.constitution.account_risk_guard import (
        BacktestAccountRiskTracker,
    )

    sr = strategies_root or os.getenv("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies")
    path = resolve_constitution_yaml(sr, override=constitution_yaml)
    ml = multi_leg_section(load_constitution_dict(path))
    account = ml.get("account") or {}
    equity = float(initial_capital if initial_capital is not None else account.get("equity_usdt", 10000.0) or 10000.0)
    unit = float(ml.get("unit_notional", 0.0) or 0.0)
    tracker = BacktestAccountRiskTracker(
        limits=dict(ml.get("account_risk_limits") or {}),
        equity_usdt=equity,
    )
    return tracker, unit


def resolve_multi_leg_risk_limits_from_constitution(
    cfg: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    ml = multi_leg_section(cfg)
    account = ml.get("account") or {}
    if not isinstance(account, dict):
        account = {}
    rs = ml.get("risk_limits") or {}
    if not isinstance(rs, dict):
        rs = {}
    equity = _literal_from_env("MULTI_LEG_ACCOUNT_EQUITY_USDT")
    if equity is None:
        equity = _float_or_none(account.get("equity_usdt"))

    def _resolve_abs(key: str) -> Optional[float]:
        direct = _float_or_none(rs.get(key))
        if direct is not None:
            return direct
        pct = _float_or_none(rs.get(f"{key}_pct"))
        if pct is not None and equity is not None:
            return float(equity) * float(pct)
        return None

    return {
        "account_equity_usdt": equity,
        "max_drawdown_pct": (
            _literal_from_env("MULTI_LEG_MAX_DRAWDOWN_PCT")
            if _literal_from_env("MULTI_LEG_MAX_DRAWDOWN_PCT") is not None
            else _float_or_none(account.get("max_drawdown_pct"))
        ),
        "max_gross_notional": _resolve_abs("max_gross_notional"),
        "max_net_notional": _resolve_abs("max_net_notional"),
        "max_symbol_gross_notional": _resolve_abs("max_symbol_gross_notional"),
        "max_symbol_net_notional": _resolve_abs("max_symbol_net_notional"),
        "max_resting_orders": _float_or_none(rs.get("max_resting_orders")),
        "account_risk_limits": dict(ml.get("account_risk_limits") or {}),
    }


def pcm_resolve_registry_key(
    archetype_token: str, _me_logical: str, me_enabled_in_allowlist_fn
) -> str:
    """Map a constitution archetype token to the key used in ``LivePCM.register``."""
    del _me_logical  # legacy signature; ME always registers as logical ``me``
    tl = str(archetype_token).lower().strip()
    if not tl:
        return ""
    if me_enabled_in_allowlist_fn([tl]):
        return "me"
    return tl.split("-", 1)[0]


def pcm_archetype_priority_for_registry(
    cfg: Dict[str, Any],
    *,
    registry_keys: set[str],
    me_pkg: str,
    me_enabled_in_allowlist_fn,
) -> List[str]:
    """Resolve PCM registry key order from YAML (override or ``enabled_archetypes`` order)."""
    raw = intent_archetype_priority_tokens(cfg)
    if not raw:
        raw = ["bpc", "tpc", "srb", "me", "fbf", "msr", "lv"]
    out: List[str] = []
    seen_rk: set[str] = set()
    for p in raw:
        token = str(p).strip()
        if not token:
            continue
        rk = pcm_resolve_registry_key(token, me_pkg, me_enabled_in_allowlist_fn)
        if not rk or rk not in registry_keys or rk in seen_rk:
            continue
        seen_rk.add(rk)
        out.append(rk)
    if not out:
        for lk in ("tpc", "srb", "me", "bpc", "lv"):
            if lk in registry_keys and lk not in seen_rk:
                seen_rk.add(lk)
                out.append(lk)
    return out


def apply_multi_leg_args_from_constitution(args: Any) -> None:
    """Fill ``run_multi_leg_live`` argparse defaults from ``multi_leg:`` in constitution."""
    sr = os.getenv("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies")
    ov = getattr(args, "constitution_yaml", None)
    if isinstance(ov, str) and not str(ov).strip():
        ov = None
    path = resolve_constitution_yaml(sr, override=ov)
    ml = multi_leg_section(load_constitution_dict(path))
    if not ml:
        return
    strat = ml.get("strategies")
    if strat:
        if isinstance(strat, (list, tuple)):
            args.strategies = ",".join(str(x).strip() for x in strat if str(x).strip())
        else:
            args.strategies = str(strat).strip()
    limits = resolve_multi_leg_risk_limits_from_constitution({"multi_leg": ml})
    for key in (
        "max_gross_notional",
        "max_net_notional",
        "max_symbol_gross_notional",
        "max_symbol_net_notional",
    ):
        if limits.get(key) is not None:
            setattr(args, key, float(limits[key] or 0.0))
    if limits.get("max_resting_orders") is not None:
        args.max_resting_orders = int(limits["max_resting_orders"] or 0)
    if limits.get("account_equity_usdt") is not None:
        setattr(args, "account_equity_usdt", float(limits["account_equity_usdt"] or 0.0))
    if limits.get("max_drawdown_pct") is not None:
        setattr(args, "max_drawdown_pct", float(limits["max_drawdown_pct"] or 0.0))
    if "unit_notional" in ml:
        args.unit_notional = float(ml["unit_notional"])
