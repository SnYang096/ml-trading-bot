from __future__ import annotations

from datetime import datetime, timezone
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .state import ConstitutionState
from .runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
    SlotRecord,
)
from .state_store import ConstitutionStatePaths, read_json, write_json
from .violation import ConstitutionViolation
from src.order_management.storage import Storage

SLOT_RELEASE_REASONS = {
    "position_closed",
    "stop_loss_hit",
    "take_profit_hit",
    "order_failed",  # 🐛 Fix: 下单失败时释放预留 slot
}


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
    max_turnover_mean: float
    max_cost_mean: float
    kill_on_any_hard_violation: bool
    cooldown_minutes: int
    daily_reset_timezone: Optional[str]


def load_constitution_config(path: str | Path) -> ConstitutionConfig:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    obj = yaml.safe_load(raw) or {}
    ks = obj.get("kill_switch") or {}
    ss = obj.get("safety_state") or {}
    ss_ks = ks.get("safety_state") or {}
    cooldown_minutes = (
        ss.get("cooldown_minutes")
        or ss_ks.get("cooldown_minutes")
        or ks.get("cooldown_minutes")
        or 240
    )
    daily_reset_timezone = (
        ss.get("daily_reset_tz")
        or ss.get("daily_reset_timezone")
        or ss_ks.get("daily_reset_tz")
        or ss_ks.get("daily_reset_timezone")
        or ks.get("daily_reset_tz")
        or ks.get("daily_reset_timezone")
        or "UTC"
    )
    return ConstitutionConfig(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "Constitution_v1")),
        constitution_hash=_sha256_text(raw),
        kill_enabled=bool(ks.get("enabled", True)),
        daily_loss_limit=float(ks.get("daily_loss_limit", 0.04)),
        weekly_loss_limit=float(ks.get("weekly_loss_limit", 0.08)),
        monthly_loss_limit=float(ks.get("monthly_loss_limit", 0.12)),
        max_dd=float(ks.get("max_dd", 0.20)),
        max_turnover_mean=float(ks.get("max_turnover_mean", 0.35)),
        max_cost_mean=float(ks.get("max_cost_mean", 0.002)),
        kill_on_any_hard_violation=bool(ks.get("kill_on_any_hard_violation", True)),
        cooldown_minutes=int(cooldown_minutes),
        daily_reset_timezone=str(daily_reset_timezone),
    )


def _infer_base_dir(constitution_yaml: str | Path) -> Path:
    """Infer the base directory for resolving relative paths in constitution.

    策略: 找到包含 constitution.yaml 的最近的 config/ 的父目录。

    示例:
      config/constitution/constitution.yaml            → 项目根/
      live/highcap/config/constitution/constitution.yaml → live/highcap/
      /opt/mlbot/config/constitution/constitution.yaml   → /opt/mlbot/

    这样 persist_to: 'data/order_management.db' 在两侧分别解析到:
      研究: <项目根>/data/order_management.db
      实盘: live/highcap/data/order_management.db
    """
    # 1. 环境变量显式指定 (最高优先)
    env_base = os.getenv("MLBOT_LIVE_BASE_DIR")
    if env_base:
        return Path(env_base).resolve()

    p = Path(constitution_yaml).resolve()

    # 2. 向上找到包含 yaml 的最近 config/ 的父目录
    #    e.g. .../live/highcap/config/constitution/constitution.yaml
    #          ↑ config/ 的父目录是 live/highcap/ → 返回
    for parent in p.parents:
        if parent.name == "config" and p.is_relative_to(parent):
            return parent.parent

    # 3. Fallback: constitution.yaml → constitution/ → config/ → base
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

    def meta(self) -> Dict[str, Any]:
        return {
            "constitution_yaml": str(self.constitution_yaml),
            "constitution_version": int(self.cfg.version),
            "constitution_name": str(self.cfg.name),
            "constitution_hash": str(self.cfg.constitution_hash),
        }

    def resolve_safety_db_path(self) -> Optional[Path]:
        obj = self._raw_obj or {}
        ks = obj.get("kill_switch") or {}
        ss = obj.get("safety_state") or {}
        ss_ks = ks.get("safety_state") or {}
        db_path = (
            ss.get("persist_to")
            or ss_ks.get("persist_to")
            or os.getenv("MLBOT_ORDER_MANAGEMENT_DB_PATH")
            or "data/order_management.db"
        )
        if not db_path:
            return None
        base = Path(self._base_dir).resolve()
        p = Path(str(db_path))
        if not p.is_absolute():
            p = (base / p).resolve()
        return p

    def _resolve_add_position(self) -> dict:
        """Resolve add_position global safety rules.

        New structure: resource_allocation.add_position_rules
        Backward compat: resource_allocation.add_position > top-level add_position
        """
        obj = self._raw_obj or {}
        ra = obj.get("resource_allocation") or {}
        addp = (
            ra.get("add_position_rules")
            or ra.get("add_position")
            or obj.get("add_position")
            or {}
        )
        return addp

    def _resolve_per_strategy_limits(self) -> dict:
        """Return per_strategy_limits dict from resource_allocation."""
        obj = self._raw_obj or {}
        ra = obj.get("resource_allocation") or {}
        return dict(ra.get("per_strategy_limits") or {})

    def resolve_risk_for_strategy(self, archetype: str) -> float:
        """Return effective risk fraction for a strategy.

        Logic: min(risk_per_slot, strategy.max_risk_per_trade)
        If strategy has no max_risk_per_trade, returns risk_per_slot.
        """
        obj = self._raw_obj or {}
        slots = obj.get("slots") or {}
        risk_per_slot = float(slots.get("risk_per_slot", 0.01))
        limits = self._resolve_per_strategy_limits()
        strat = limits.get(archetype.lower()) or {}
        strat_risk = strat.get("max_risk_per_trade")
        if strat_risk is not None:
            return min(risk_per_slot, float(strat_risk))
        return risk_per_slot

    def _load_state_paths(self) -> ConstitutionStatePaths:
        obj = self._raw_obj or {}
        slots = obj.get("slots") or {}
        addp = self._resolve_add_position()
        slots_p = (slots.get("slot_state_tracking") or {}).get("persist_to") or None
        addp_p = (addp.get("state_tracking") or {}).get("persist_to") or None

        base = Path(self._base_dir).resolve()
        tmp = ConstitutionStatePaths(base_dir=base)

        def _split_persist_target(
            p: Optional[str],
        ) -> tuple[Optional[Path], Optional[Path]]:
            if not p:
                return None, None
            raw = str(p)
            if raw.lower().endswith(".db"):
                return None, tmp.resolve(raw)
            return tmp.resolve(raw), None

        slots_path, slots_db_path = _split_persist_target(slots_p)
        addp_path, addp_db_path = _split_persist_target(addp_p)
        return ConstitutionStatePaths(
            base_dir=base,
            slots_path=slots_path,
            slots_db_path=slots_db_path,
            add_position_path=addp_path,
            add_position_db_path=addp_db_path,
        )

    # -------------------------------------------------------------------------
    # Runtime state persistence (V1.1): slots / add-position
    # -------------------------------------------------------------------------
    def load_runtime_state(self) -> ConstitutionRuntimeState:
        st = ConstitutionRuntimeState()

        # Slots
        if self._paths.slots_db_path:
            storage = Storage(str(self._paths.slots_db_path))
            obj = storage.get_slots_state() or {}
        elif self._paths.slots_path:
            obj = read_json(self._paths.slots_path)
        else:
            obj = {}
        active = (obj.get("active") or {}) if isinstance(obj, dict) else {}
        if isinstance(active, dict):
            for pid, rec in active.items():
                if not pid:
                    continue
                r = rec or {}
                st.slots.active[str(pid)] = SlotRecord(
                    position_id=str(pid),
                    symbol=(
                        str(r.get("symbol")) if r.get("symbol") is not None else None
                    ),
                    archetype=(
                        str(r.get("archetype"))
                        if r.get("archetype") is not None
                        else None
                    ),
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
        if self._paths.add_position_db_path:
            storage = Storage(str(self._paths.add_position_db_path))
            obj = storage.get_add_position_state() or {}
        elif self._paths.add_position_path:
            obj = read_json(self._paths.add_position_path)
        else:
            obj = {}
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

        return st

    def save_runtime_state(self, st: ConstitutionRuntimeState) -> None:
        if self._paths.slots_db_path:
            storage = Storage(str(self._paths.slots_db_path))
            storage.upsert_slots_state(payload=st.slots.as_dict())
        if self._paths.slots_path:
            write_json(self._paths.slots_path, st.slots.as_dict())
        if self._paths.add_position_db_path:
            storage = Storage(str(self._paths.add_position_db_path))
            storage.upsert_add_position_state(payload=st.add_position.as_dict())
        if self._paths.add_position_path:
            write_json(self._paths.add_position_path, st.add_position.as_dict())

    def reserve_slot(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        symbol: Optional[str] = None,
        archetype: Optional[str] = None,
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
            archetype=str(archetype) if archetype is not None else None,
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
        if str(reason) not in SLOT_RELEASE_REASONS:
            return
        pid = str(position_id).strip()
        if not pid:
            return
        rec = st.slots.active.get(pid)
        if not rec:
            return
        # Closed record not persisted separately in v1; we just free the slot.
        st.slots.active.pop(pid, None)

    def validate_add_position(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        archetype: Optional[str],
        current_r: Optional[float],
        locked_profit: Optional[bool] = None,
    ) -> None:
        # 1. Check per-strategy allow_add_position
        arch_key = str(archetype or "").strip().lower()
        limits = self._resolve_per_strategy_limits()
        strat_cfg = limits.get(arch_key) or {}
        allow = strat_cfg.get("allow_add_position")
        if allow is not None and not bool(allow):
            raise ConstitutionViolation(
                code="ADD_POSITION_STRATEGY_FORBIDDEN",
                message=f"strategy '{arch_key}' does not allow add_position",
                context={"archetype": arch_key, **self.meta()},
            )

        # 2. Global add_position safety rules
        addp = self._resolve_add_position()
        # Backward compat: old 'enabled' flag
        if not bool(addp.get("enabled", True)):
            raise ConstitutionViolation(
                code="ADD_POSITION_DISABLED",
                message="add_position disabled",
                context=self.meta(),
            )

        pid = str(position_id).strip()
        if not pid:
            raise ConstitutionViolation(
                code="ADD_POSITION_BAD_ID",
                message="position_id is empty",
                context=self.meta(),
            )
        rec = st.add_position.positions.get(pid)
        add_count = int(rec.add_count) if rec is not None else 0
        max_add_times = int(addp.get("max_add_times", 1))
        if add_count >= max_add_times:
            raise ConstitutionViolation(
                code="ADD_POSITION_MAX_TIMES",
                message="max_add_times exceeded",
                context={"position_id": pid, "add_count": add_count, **self.meta()},
            )

        trigger_r = float(addp.get("lock_profit_breakeven_trigger_r", 1.0))
        inferred_locked = bool(locked_profit) if locked_profit is not None else False
        if current_r is not None and float(current_r) >= trigger_r:
            inferred_locked = True
        if bool(addp.get("require_locked_profit", True)) and not inferred_locked:
            raise ConstitutionViolation(
                code="ADD_POSITION_LOCKED_PROFIT_REQUIRED",
                message="locked_profit required before add",
                context={
                    "position_id": pid,
                    "current_r": current_r,
                    "locked_profit": inferred_locked,
                    **self.meta(),
                },
            )

    def record_add_position(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        current_r: Optional[float],
        locked_profit: Optional[bool] = None,
    ) -> None:
        pid = str(position_id).strip()
        if not pid:
            return
        addp = self._resolve_add_position()
        trigger_r = float(addp.get("lock_profit_breakeven_trigger_r", 1.0))
        inferred_locked = bool(locked_profit) if locked_profit is not None else False
        if current_r is not None and float(current_r) >= trigger_r:
            inferred_locked = True
        rec = st.add_position.positions.get(pid)
        add_count = int(rec.add_count) if rec is not None else 0
        st.add_position.positions[pid] = AddPositionRecord(
            position_id=pid,
            add_count=int(add_count + 1),
            locked_profit=inferred_locked,
            current_r=current_r,
            updated_at=_iso_now(),
        )

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
