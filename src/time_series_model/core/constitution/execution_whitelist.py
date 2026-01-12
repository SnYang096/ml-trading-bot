from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .violation import ConstitutionViolation


@dataclass(frozen=True)
class RegimeWhitelist:
    allowed_strategies: List[str]
    forbidden_keywords: List[str]
    required_evidence_by_strategy: Dict[str, List[str]]


@dataclass(frozen=True)
class ExecutionWhitelistConfig:
    version: int
    name: str
    regimes: Dict[str, RegimeWhitelist]


def load_execution_whitelist_config(path: str | Path) -> ExecutionWhitelistConfig:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    regimes_raw = obj.get("regimes") or {}
    regimes: Dict[str, RegimeWhitelist] = {}
    if isinstance(regimes_raw, dict):
        for k, v in regimes_raw.items():
            if not isinstance(v, dict):
                continue
            req_map: Dict[str, List[str]] = {}
            req_raw = (
                v.get("required_evidence_by_strategy")
                or v.get("strategy_requirements")
                or {}
            )
            if isinstance(req_raw, dict):
                for sid, rr in req_raw.items():
                    if isinstance(rr, dict):
                        lst = rr.get("required_evidence") or []
                    else:
                        lst = rr
                    if isinstance(lst, list):
                        req_map[str(sid)] = [str(x) for x in lst]
            regimes[str(k).upper()] = RegimeWhitelist(
                allowed_strategies=[
                    str(x) for x in (v.get("allowed_strategies") or [])
                ],
                forbidden_keywords=[
                    str(x) for x in (v.get("forbidden_keywords") or [])
                ],
                required_evidence_by_strategy=req_map,
            )
    return ExecutionWhitelistConfig(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "execution_whitelist")),
        regimes=regimes,
    )


def validate_execution_whitelist(
    *,
    cfg: ExecutionWhitelistConfig,
    regime: str,
    strategy_id: str,
    tags: Optional[Sequence[str]] = None,
    evidence: Optional[Dict[str, bool]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    r = str(regime).upper().strip()
    sid = str(strategy_id).strip()
    ctx: Dict[str, Any] = {"regime": r, "strategy_id": sid, "tags": list(tags or [])}

    if not sid:
        return False, "missing_strategy_id", ctx
    rw = cfg.regimes.get(r)
    if rw is None:
        return False, "unknown_regime", ctx

    if r == "NO_TRADE":
        # No strategy should be executed in NO_TRADE.
        return False, "no_trade_forbids_execution", ctx

    if sid not in set(rw.allowed_strategies):
        ctx["allowed_strategies"] = list(rw.allowed_strategies)
        return False, "strategy_not_allowed", ctx

    req = (rw.required_evidence_by_strategy or {}).get(sid) or []
    if req:
        ev = dict(evidence or {})
        missing = [k for k in req if k not in ev]
        bad = [k for k in req if (k in ev and not bool(ev[k]))]
        if missing or bad:
            ctx["required_evidence"] = list(req)
            ctx["evidence_missing"] = missing
            ctx["evidence_false"] = bad
            return False, "required_evidence_not_satisfied", ctx

    hay = " ".join([sid] + [str(x) for x in (tags or [])])
    for kw in rw.forbidden_keywords:
        if kw and (kw in hay):
            ctx["forbidden_keyword"] = kw
            return False, "forbidden_keyword", ctx

    return True, "ok", ctx


def enforce_execution_whitelist(
    *,
    cfg: ExecutionWhitelistConfig,
    regime: str,
    strategy_id: str,
    tags: Optional[Sequence[str]] = None,
    evidence: Optional[Dict[str, bool]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    ok, reason, ctx = validate_execution_whitelist(
        cfg=cfg, regime=regime, strategy_id=strategy_id, tags=tags, evidence=evidence
    )
    if ok:
        return
    raise ConstitutionViolation(
        code="EXEC_WHITELIST",
        message=reason,
        context={**(meta or {}), **ctx},
    )
