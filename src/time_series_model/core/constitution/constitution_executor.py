from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .state import ConstitutionState
from .runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
    EscalationRuntimeState,
    SlotRecord,
)
from .replacement_judge import (
    ReplacementDecision,
    ReplacementInputs,
    decide_replacement_v1,
)
from .state_store import ConstitutionStatePaths, append_jsonl, read_json, write_json
from .violation import ConstitutionViolation
from .execution_whitelist import (
    ExecutionWhitelistConfig,
    enforce_execution_whitelist,
    load_execution_whitelist_config,
)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConstitutionConfig:
    """
    Runtime-ready view of constitution.yaml.
    This is intentionally a minimal subset for V1.1 enforcement.
    """

    version: int
    name: str
    constitution_hash: str

    # Kill-switch (hard stops)
    kill_enabled: bool
    daily_loss_limit: float
    weekly_loss_limit: float
    monthly_loss_limit: float
    max_dd: float
    kill_on_any_hard_violation: bool


def load_constitution_config(path: str | Path) -> ConstitutionConfig:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    obj = yaml.safe_load(raw) or {}
    ks = obj.get("kill_switch") or {}
    return ConstitutionConfig(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "Constitution_v1")),
        constitution_hash=_sha256_text(raw),
        kill_enabled=bool(ks.get("enabled", True)),
        daily_loss_limit=float(ks.get("daily_loss_limit", 0.04)),
        weekly_loss_limit=float(ks.get("weekly_loss_limit", 0.08)),
        monthly_loss_limit=float(ks.get("monthly_loss_limit", 0.12)),
        max_dd=float(ks.get("max_dd", 0.20)),
        kill_on_any_hard_violation=bool(ks.get("kill_on_any_hard_violation", True)),
    )


def _infer_base_dir(constitution_yaml: str | Path) -> Path:
    p = Path(constitution_yaml).resolve()
    # Try to infer repo root by locating ".../config/..." in the path.
    for parent in p.parents:
        if (parent / "config").exists() and (parent / "src").exists():
            return parent
    # Fallback: config/constitution/constitution_v1.yaml -> go up 2
    try:
        return p.parents[2]
    except Exception:
        return p.parent


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


class ConstitutionExecutor:
    """
    Single enforcement entry point.

    Rule: any capital/position/execution instruction must be validated here
    before it can be applied downstream.
    """

    def __init__(self, *, constitution_yaml: str | Path):
        self.constitution_yaml = str(constitution_yaml)
        self.cfg = load_constitution_config(constitution_yaml)
        self._base_dir = _infer_base_dir(constitution_yaml)
        self._raw_obj = (
            yaml.safe_load(Path(constitution_yaml).read_text(encoding="utf-8")) or {}
        )
        self._paths = self._load_state_paths()
        self._exec_whitelist_cfg = self._load_execution_whitelist_cfg()

    def meta(self) -> Dict[str, Any]:
        return {
            "constitution_yaml": str(self.constitution_yaml),
            "constitution_version": int(self.cfg.version),
            "constitution_name": str(self.cfg.name),
            "constitution_hash": str(self.cfg.constitution_hash),
        }

    def _load_execution_whitelist_cfg(self) -> Optional[ExecutionWhitelistConfig]:
        obj = self._raw_obj or {}
        ew = obj.get("execution_whitelist") or {}
        if not isinstance(ew, dict):
            return None
        if not bool(ew.get("enabled", False)):
            return None
        cfg_file = ew.get("config_file")
        if not cfg_file:
            return None
        base = Path(self._base_dir).resolve()
        p = Path(str(cfg_file))
        if not p.is_absolute():
            p = (base / p).resolve()
        return load_execution_whitelist_config(p)

    def validate_execution_strategy(
        self,
        *,
        regime: str,
        strategy_id: str,
        tags: Optional[List[str]] = None,
        evidence: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Hard contract: Router decides regime; execution MUST be in regime allow-list.
        """
        if self._exec_whitelist_cfg is None:
            return
        enforce_execution_whitelist(
            cfg=self._exec_whitelist_cfg,
            regime=str(regime),
            strategy_id=str(strategy_id),
            tags=tags,
            evidence=evidence,
            meta=self.meta(),
        )

    def _load_state_paths(self) -> ConstitutionStatePaths:
        obj = self._raw_obj or {}
        slots = obj.get("slots") or {}
        addp = obj.get("add_position") or {}
        rep = obj.get("replacement_policy") or {}
        esc = obj.get("capital_escalation") or {}
        esc_ad = esc.get("auto_degradation") or {}
        xt = obj.get("extreme_tail") or {}

        slots_p = (slots.get("slot_state_tracking") or {}).get("persist_to") or None
        addp_p = (addp.get("state_tracking") or {}).get("persist_to") or None
        esc_p = ((esc_ad.get("state_persistence") or {}).get("persist_to")) or None
        rep_dir = (rep.get("auditability") or {}).get("log_path") or None
        xt_p = (xt.get("state_tracking") or {}).get("persist_to") or None

        base = Path(self._base_dir).resolve()
        tmp = ConstitutionStatePaths(base_dir=base)
        return ConstitutionStatePaths(
            base_dir=base,
            slots_path=tmp.resolve(str(slots_p)) if slots_p else None,
            add_position_path=tmp.resolve(str(addp_p)) if addp_p else None,
            escalation_path=tmp.resolve(str(esc_p)) if esc_p else None,
            replacement_log_dir=tmp.resolve(str(rep_dir)) if rep_dir else None,
            extreme_tail_path=tmp.resolve(str(xt_p)) if xt_p else None,
        )

    # -------------------------------------------------------------------------
    # Runtime state persistence (V1.1): slots / add-position / escalation
    # -------------------------------------------------------------------------
    def load_runtime_state(self) -> ConstitutionRuntimeState:
        st = ConstitutionRuntimeState()

        # Slots
        if self._paths.slots_path:
            obj = read_json(self._paths.slots_path)
            active = (obj.get("active") or {}) if isinstance(obj, dict) else {}
            if isinstance(active, dict):
                for pid, rec in active.items():
                    if not pid:
                        continue
                    r = rec or {}
                    st.slots.active[str(pid)] = SlotRecord(
                        position_id=str(pid),
                        symbol=(
                            str(r.get("symbol"))
                            if r.get("symbol") is not None
                            else None
                        ),
                        mode=str(r.get("mode")) if r.get("mode") is not None else None,
                        opened_at=(
                            str(r.get("opened_at"))
                            if r.get("opened_at") is not None
                            else None
                        ),
                        closed_at=(
                            str(r.get("closed_at"))
                            if r.get("closed_at") is not None
                            else None
                        ),
                        close_reason=(
                            str(r.get("close_reason"))
                            if r.get("close_reason") is not None
                            else None
                        ),
                    )

        # Add-position
        if self._paths.add_position_path:
            obj = read_json(self._paths.add_position_path)
            pos = (obj.get("positions") or {}) if isinstance(obj, dict) else {}
            if isinstance(pos, dict):
                for pid, rec in pos.items():
                    if not pid:
                        continue
                    r = rec or {}
                    st.add_position.positions[str(pid)] = AddPositionRecord(
                        position_id=str(pid),
                        add_count=int(r.get("add_count", 0)),
                        locked_profit=bool(r.get("locked_profit", False)),
                        current_r=(
                            float(r["current_r"])
                            if r.get("current_r") is not None
                            else None
                        ),
                        updated_at=(
                            str(r.get("updated_at"))
                            if r.get("updated_at") is not None
                            else None
                        ),
                    )

        # Escalation
        if self._paths.escalation_path:
            obj = read_json(self._paths.escalation_path)
            if isinstance(obj, dict):
                st.escalation = EscalationRuntimeState(
                    is_escalated=bool(obj.get("is_escalated", False)),
                    escalation_entry_time=(
                        str(obj.get("escalation_entry_time"))
                        if obj.get("escalation_entry_time") is not None
                        else None
                    ),
                    escalation_entry_equity=(
                        float(obj["escalation_entry_equity"])
                        if obj.get("escalation_entry_equity") is not None
                        else None
                    ),
                    locked_until=(
                        str(obj.get("locked_until"))
                        if obj.get("locked_until") is not None
                        else None
                    ),
                    last_exit_reason=(
                        str(obj.get("last_exit_reason"))
                        if obj.get("last_exit_reason") is not None
                        else None
                    ),
                    last_exit_time=(
                        str(obj.get("last_exit_time"))
                        if obj.get("last_exit_time") is not None
                        else None
                    ),
                )

        # Extreme tail (event optionality) state (raw dict for v1)
        if self._paths.extreme_tail_path:
            obj = read_json(self._paths.extreme_tail_path)
            st.extreme_tail = dict(obj or {}) if isinstance(obj, dict) else {}

        return st

    def save_runtime_state(self, st: ConstitutionRuntimeState) -> None:
        if self._paths.slots_path:
            write_json(self._paths.slots_path, st.slots.as_dict())
        if self._paths.add_position_path:
            write_json(self._paths.add_position_path, st.add_position.as_dict())
        if self._paths.escalation_path:
            write_json(self._paths.escalation_path, st.escalation.as_dict())
        if self._paths.extreme_tail_path:
            write_json(self._paths.extreme_tail_path, dict(st.extreme_tail or {}))

    def reserve_slot(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        symbol: Optional[str] = None,
        mode: Optional[str] = None,
        opened_at: Optional[str] = None,
    ) -> None:
        slots = (self._raw_obj or {}).get("slots") or {}
        if not bool(slots.get("enabled", True)):
            return
        slot_count = int(slots.get("slot_count", 2))
        pid = str(position_id).strip()
        if not pid:
            raise ConstitutionViolation(
                code="SLOT_BAD_ID", message="position_id is empty", context=self.meta()
            )
        if pid in st.slots.active:
            return
        if st.slots.active_count() >= slot_count:
            raise ConstitutionViolation(
                code="SLOT_FULL",
                message=f"Slot capacity exceeded: active={st.slots.active_count()} slot_count={slot_count}",
                context={"active_slots": list(st.slots.active.keys()), **self.meta()},
            )
        st.slots.active[pid] = SlotRecord(
            position_id=pid,
            symbol=str(symbol) if symbol is not None else None,
            mode=str(mode) if mode is not None else None,
            opened_at=opened_at or _iso_now(),
        )

    def release_slot(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        reason: str,
        closed_at: Optional[str] = None,
    ) -> None:
        pid = str(position_id).strip()
        if not pid:
            return
        rec = st.slots.active.get(pid)
        if not rec:
            return
        # Closed record not persisted separately in v1; we just free the slot.
        st.slots.active.pop(pid, None)

    def append_replacement_audit(self, *, event: Dict[str, Any]) -> None:
        rep = (self._raw_obj or {}).get("replacement_policy") or {}
        aud = rep.get("auditability") or {}
        if not bool(rep.get("enabled", True)) or not bool(
            aud.get("log_every_replacement", True)
        ):
            return
        req = aud.get("required_fields") or []
        missing = []
        for k in req:
            kk = str(k)
            if kk and (event or {}).get(kk) in (None, "", []):
                missing.append(kk)
        if missing:
            raise ConstitutionViolation(
                code="REPLACEMENT_AUDIT_MISSING_FIELDS",
                message=f"Replacement audit missing required fields: {missing}",
                context={"missing": missing, "event": event, **self.meta()},
            )
        log_dir = (
            self._paths.replacement_log_dir
            or (Path(self._base_dir) / "logs/replacements").resolve()
        )
        log_path = Path(log_dir) / "replacements.jsonl"
        evt = dict(event or {})
        evt.setdefault("timestamp", _iso_now())
        append_jsonl(log_path, evt)

    def decide_replacement(
        self,
        *,
        st: ConstitutionRuntimeState,
        old_position_id: str,
        old_remaining_rr: float,
        old_failure_reasons: List[str],
        new_signal_id: str,
        new_expected_rr: float,
    ) -> Dict[str, Any]:
        """
        Constitution-level replacement judge (v1):
        - Uses replacement_policy.min_expected_rr_improvement => beta = 1 + min_expected_rr_improvement
        - Enforces 'replacement must be guilty' and forbids multi-dimensional scoring
        - Writes audit log when decision==REPLACE
        """
        rep = (self._raw_obj or {}).get("replacement_policy") or {}
        min_imp = float(rep.get("min_expected_rr_improvement", 0.25))
        beta = float(1.0 + max(0.0, min_imp))
        has_free_slot = st.slots.active_count() < int(
            ((self._raw_obj or {}).get("slots") or {}).get("slot_count", 2)
        )

        res = decide_replacement_v1(
            ReplacementInputs(
                has_free_slot=bool(has_free_slot),
                old_position_id=str(old_position_id),
                old_remaining_rr=float(old_remaining_rr),
                old_failure_reasons=list(old_failure_reasons or []),
                new_signal_id=str(new_signal_id),
                new_expected_rr=float(new_expected_rr),
                beta=beta,
            )
        )
        out = {
            "decision": res.decision.value,
            "reason": res.reason,
            "context": dict(res.context or {}),
        }

        if res.decision == ReplacementDecision.REPLACE and not has_free_slot:
            # Audit required for any eviction-style replacement.
            self.append_replacement_audit(
                event={
                    "closed_position_id": str(old_position_id),
                    "close_reason": str(res.reason),
                    "new_position_signal": str(new_signal_id),
                    "expected_rr_improvement": float(
                        float(new_expected_rr) - float(old_remaining_rr)
                    ),
                }
            )
        return out

    def is_escalation_locked(
        self, *, st: ConstitutionRuntimeState, now_iso: Optional[str] = None
    ) -> bool:
        now = _parse_iso(now_iso) or datetime.now(timezone.utc)
        until = _parse_iso(st.escalation.locked_until)
        return bool(until and now < until)

    def record_escalation_exit(
        self,
        *,
        st: ConstitutionRuntimeState,
        exit_reason: str,
        equity_at_exit: Optional[float] = None,
        exited_at: Optional[str] = None,
    ) -> None:
        esc = (self._raw_obj or {}).get("capital_escalation") or {}
        ad = esc.get("auto_degradation") or {}
        dur_days = 0
        for act in ad.get("on_exit") or []:
            if (
                isinstance(act, dict)
                and str(act.get("action")) == "lock_new_escalation"
            ):
                dur_days = int(act.get("duration_days", 0))
                break
        exited_at = exited_at or _iso_now()
        t = _parse_iso(exited_at) or datetime.now(timezone.utc)
        locked_until = (
            (t + timedelta(days=int(max(0, dur_days))))
            .replace(microsecond=0)
            .isoformat()
        )
        st.escalation = EscalationRuntimeState(
            is_escalated=False,
            escalation_entry_time=st.escalation.escalation_entry_time,
            escalation_entry_equity=st.escalation.escalation_entry_equity,
            locked_until=locked_until,
            last_exit_reason=str(exit_reason),
            last_exit_time=exited_at,
        )

    # -------------------------------------------------------------------------
    # Extreme tail / Event optionality (non-compounding) — v1 minimal enforcement
    # -------------------------------------------------------------------------
    def validate_extreme_tail_entry(
        self,
        *,
        equity_usd: float,
        entry_usd: float,
        st: ConstitutionRuntimeState,
        year: int,
    ) -> None:
        xt = (self._raw_obj or {}).get("extreme_tail") or {}
        if not bool(xt.get("enabled", True)):
            raise ConstitutionViolation(
                code="EXTREME_TAIL_DISABLED",
                message="extreme_tail disabled",
                context=self.meta(),
            )

        limits = xt.get("hard_limits") or {}
        single_max = float(limits.get("single_event_max_usd", 1000))
        annual_ratio = float(limits.get("annual_total_ratio", 0.02))
        max_budget = float(xt.get("max_budget", 0.02))

        eq = float(max(0.0, equity_usd))
        e = float(max(0.0, entry_usd))
        if e <= 0.0:
            return
        if e > single_max + 1e-9:
            raise ConstitutionViolation(
                code="EXTREME_TAIL_SINGLE_MAX",
                message=f"entry_usd {e} exceeds single_event_max_usd {single_max}",
                context={
                    "entry_usd": e,
                    "single_event_max_usd": single_max,
                    **self.meta(),
                },
            )
        if eq > 0 and e > eq * max_budget + 1e-9:
            raise ConstitutionViolation(
                code="EXTREME_TAIL_BUDGET_MAX",
                message="entry exceeds max_budget fraction",
                context={
                    "entry_usd": e,
                    "equity_usd": eq,
                    "max_budget": max_budget,
                    **self.meta(),
                },
            )

        used = float((st.extreme_tail or {}).get("used_this_year", 0.0) or 0.0)
        y0 = int((st.extreme_tail or {}).get("year", year) or year)
        if y0 != int(year):
            # reset yearly usage
            used = 0.0
        if eq > 0 and (used + e) > eq * annual_ratio + 1e-9:
            raise ConstitutionViolation(
                code="EXTREME_TAIL_ANNUAL_CAP",
                message="annual_total_ratio exceeded",
                context={
                    "used_this_year": used,
                    "entry_usd": e,
                    "annual_total_ratio": annual_ratio,
                    "equity_usd": eq,
                    **self.meta(),
                },
            )

    def record_extreme_tail_entry(
        self,
        *,
        st: ConstitutionRuntimeState,
        entry_usd: float,
        position_id: str,
        year: int,
    ) -> None:
        used = float((st.extreme_tail or {}).get("used_this_year", 0.0) or 0.0)
        st.extreme_tail = dict(st.extreme_tail or {})
        st.extreme_tail["year"] = int(year)
        st.extreme_tail["used_this_year"] = float(used + float(max(0.0, entry_usd)))
        aps = list(st.extreme_tail.get("active_positions") or [])
        aps.append(str(position_id))
        st.extreme_tail["active_positions"] = sorted(set(aps))

    def validate_drawdown(self, *, state: ConstitutionState) -> None:
        """
        V1.1: implement kill-switch checks that are universally applicable.
        """

        if not self.cfg.kill_enabled:
            return

        reasons: List[str] = []
        if state.drawdown is not None and float(state.drawdown) > float(
            self.cfg.max_dd
        ):
            reasons.append("max_dd")
        if float(state.daily_loss) >= float(self.cfg.daily_loss_limit):
            reasons.append("daily_loss_limit")
        if float(state.weekly_loss) >= float(self.cfg.weekly_loss_limit):
            reasons.append("weekly_loss_limit")
        if float(state.monthly_loss) >= float(self.cfg.monthly_loss_limit):
            reasons.append("monthly_loss_limit")
        if bool(state.data_bad):
            reasons.append("data_bad")
        if bool(state.hard_violation):
            reasons.append("hard_violation")

        if reasons and bool(self.cfg.kill_on_any_hard_violation):
            raise ConstitutionViolation(
                code="KILL_SWITCH",
                message=f"Kill-switch triggered: {', '.join(reasons)}",
                context={"reasons": reasons, **state.as_dict(), **self.meta()},
            )

    def validate_capital_allocation(
        self,
        *,
        state: ConstitutionState,
        per_mode_budget: Dict[str, float],
        per_symbol_budget: Dict[str, float],
        overrides: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Guard PCM output. V1.1 keeps this intentionally strict and simple.
        """

        self.validate_drawdown(state=state)

        # Budgets must be non-negative and finite.
        for k, v in (per_mode_budget or {}).items():
            fv = float(v)
            if not (fv >= 0.0):
                raise ConstitutionViolation(
                    code="CAPITAL_BUDGET_NEGATIVE",
                    message=f"per_mode_budget[{k}] is negative: {fv}",
                    context={
                        "per_mode_budget": per_mode_budget,
                        **state.as_dict(),
                        **self.meta(),
                    },
                )

        for k, v in (per_symbol_budget or {}).items():
            fv = float(v)
            if not (fv >= 0.0):
                raise ConstitutionViolation(
                    code="SYMBOL_BUDGET_NEGATIVE",
                    message=f"per_symbol_budget[{k}] is negative: {fv}",
                    context={
                        "per_symbol_budget": per_symbol_budget,
                        **state.as_dict(),
                        **self.meta(),
                    },
                )

        # Human override must be audited (tag+reason), otherwise it's an illegal bypass.
        if overrides:
            for o in overrides:
                tag = str((o or {}).get("tag") or "").strip()
                reason = str((o or {}).get("reason") or "").strip()
                if not tag or not reason:
                    raise ConstitutionViolation(
                        code="HUMAN_OVERRIDE_UNAUDITED",
                        message="Human override must include non-empty tag and reason.",
                        context={"override": o, **state.as_dict(), **self.meta()},
                    )
