"""Read constitution YAML paths and shared sections (classic + multi-leg + publisher)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    rs = ml.get("risk_limits") or {}
    for key, attr in (
        ("max_gross_notional", "max_gross_notional"),
        ("max_net_notional", "max_net_notional"),
        ("max_symbol_gross_notional", "max_symbol_gross_notional"),
        ("max_symbol_net_notional", "max_symbol_net_notional"),
    ):
        if key in rs:
            setattr(args, attr, float(rs[key]))
    if "max_resting_orders" in rs:
        args.max_resting_orders = int(rs["max_resting_orders"])
    if "unit_notional" in ml:
        args.unit_notional = float(ml["unit_notional"])
