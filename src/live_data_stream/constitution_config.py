"""Read constitution YAML paths and shared sections (classic + multi-leg + publisher)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)
MULTI_LEG_STRATEGY_TYPES = frozenset({"grid", "dual_add_trend"})


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
    """PCM 联合回测白名单 + 经典 LivePCM 注册候选（同一列表）。

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


def validate_classic_slot_capacity(
    *, constitution_cfg: Dict[str, Any], symbols: Iterable[str]
) -> Dict[str, Any]:
    policy = classic_slot_policy_from_constitution(constitution_cfg)
    clean_symbols = sorted({str(s).upper().strip() for s in symbols if str(s).strip()})
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
        return {"classic": set(), "multi_leg": set()}
    classic: Set[str] = set()
    multi_leg: Set[str] = set()
    for key, scfg in raw.items():
        name = str(key or "").strip().lower()
        if not name:
            continue
        stype = _strategy_type_from_pipeline_entry(scfg)
        if stype in MULTI_LEG_STRATEGY_TYPES:
            multi_leg.add(name)
        else:
            classic.add(name)
    return {"classic": classic, "multi_leg": multi_leg}


def validate_pipeline_constitution_alignment(
    *,
    pipeline_cfg: Dict[str, Any],
    constitution_cfg: Dict[str, Any],
    context_label: str = "rolling_research",
) -> Dict[str, List[str]]:
    parts = partition_pipeline_strategies_by_type(pipeline_cfg)
    pipeline_classic = set(parts["classic"])
    pipeline_multi_leg = set(parts["multi_leg"])
    const_classic = set(enabled_archetypes_from_constitution(constitution_cfg))
    const_multi_leg = set(multi_leg_strategies_from_constitution(constitution_cfg))

    classic_missing_in_const = sorted(pipeline_classic - const_classic)
    classic_extra_in_const = sorted(const_classic - pipeline_classic)
    multi_missing_in_const = sorted(pipeline_multi_leg - const_multi_leg)
    multi_extra_in_const = sorted(const_multi_leg - pipeline_multi_leg)

    if (
        classic_missing_in_const
        or classic_extra_in_const
        or multi_missing_in_const
        or multi_extra_in_const
    ):
        msg_lines = [
            f"{context_label}: pipeline/constitution strategy mismatch",
            f"  classic pipeline={sorted(pipeline_classic)}",
            f"  classic constitution={sorted(const_classic)}",
            f"  multi_leg pipeline={sorted(pipeline_multi_leg)}",
            f"  multi_leg constitution={sorted(const_multi_leg)}",
        ]
        if classic_missing_in_const:
            msg_lines.append(
                "  missing in constitution.enabled_archetypes="
                f"{classic_missing_in_const}"
            )
        if classic_extra_in_const:
            msg_lines.append(
                "  extra in constitution.enabled_archetypes="
                f"{classic_extra_in_const}"
            )
        if multi_missing_in_const:
            msg_lines.append(
                "  missing in constitution.multi_leg.strategies="
                f"{multi_missing_in_const}"
            )
        if multi_extra_in_const:
            msg_lines.append(
                "  extra in constitution.multi_leg.strategies="
                f"{multi_extra_in_const}"
            )
        raise ValueError("\n".join(msg_lines))

    return {
        "classic": sorted(pipeline_classic),
        "multi_leg": sorted(pipeline_multi_leg),
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
    }


def pcm_resolve_registry_key(
    archetype_token: str, me_pkg: str, me_enabled_in_allowlist_fn
) -> str:
    """Map a constitution archetype token to the key used in ``LivePCM.register``."""
    tl = str(archetype_token).lower().strip()
    if not tl:
        return ""
    if me_enabled_in_allowlist_fn([tl]):
        return me_pkg
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
        for lk in ("tpc", "srb", me_pkg, "bpc", "lv"):
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
