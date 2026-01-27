from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
    load_safety_state,
    save_safety_state,
)
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.state import ConstitutionState
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.ops.state_snapshot import (
    SystemStateSnapshot,
    write_state_snapshot,
)


@dataclass(frozen=True)
class LiveEnforcementResult:
    ok: bool
    reason: str
    snapshot_path: Optional[str] = None


def enforce_before_order(
    *,
    executor: ConstitutionExecutor,
    runtime_state: ConstitutionRuntimeState,
    position_id: str,
    symbol: str,
    mode: str,
    execution_strategy: Optional[str] = None,
    execution_tags: Optional[list[str]] = None,
    execution_evidence: Optional[dict[str, bool]] = None,
    equity: Optional[float] = None,
    drawdown: Optional[float] = None,
    daily_loss: float = 0.0,
    weekly_loss: float = 0.0,
    monthly_loss: float = 0.0,
    daily_cost_mean: Optional[float] = None,
    daily_turnover_mean: Optional[float] = None,
    hard_violation: bool = False,
    data_bad: bool = False,
    evt_risk_flag: Optional[bool] = None,
    snapshot_out: Optional[str | Path] = None,
    snapshot_extra: Optional[Dict[str, Any]] = None,
    pcm_budget: Optional[Dict[str, Any]] = None,
) -> LiveEnforcementResult:
    """
    Minimal live adapter hook:
    - validate kill-switch style drawdown constraints
    - reserve a slot (hard cap)
    - persist runtime state (caller should call executor.save_runtime_state)
    - emit a SystemStateSnapshot for attribution
    """
    st = ConstitutionState(
        task_id=None,
        timestamp=None,
        equity=equity,
        drawdown=drawdown,
        daily_loss=float(daily_loss),
        weekly_loss=float(weekly_loss),
        monthly_loss=float(monthly_loss),
        hard_violation=bool(hard_violation),
        data_bad=bool(data_bad),
    )
    safety_state = SafetyRuntimeState()
    safety_db_path = None
    try:
        safety_db_path = executor.resolve_safety_db_path()
        safety_state = load_safety_state(db_path=str(safety_db_path))
    except Exception:
        safety_state = SafetyRuntimeState()

    now = datetime.now(timezone.utc)
    decision = evaluate_safety_state(
        state=safety_state,
        now=now,
        cooldown_minutes=int(executor.cfg.cooldown_minutes),
        daily_reset_tz=executor.cfg.daily_reset_timezone,
        daily_loss=float(daily_loss),
        weekly_loss=float(weekly_loss),
        monthly_loss=float(monthly_loss),
        drawdown=drawdown,
        hard_violation=bool(hard_violation),
        data_bad=bool(data_bad),
        daily_cost_mean=daily_cost_mean,
        daily_turnover_mean=daily_turnover_mean,
        limits={
            "max_dd": float(executor.cfg.max_dd),
            "daily_loss_limit": float(executor.cfg.daily_loss_limit),
            "weekly_loss_limit": float(executor.cfg.weekly_loss_limit),
            "monthly_loss_limit": float(executor.cfg.monthly_loss_limit),
            "max_turnover_mean": float(executor.cfg.max_turnover_mean),
            "max_cost_mean": float(executor.cfg.max_cost_mean),
        },
    )
    if not bool(executor.cfg.kill_on_any_hard_violation):
        decision.state.halted = False
        decision.state.halt_reason = []
        decision.state.halt_since = None
        decision.state.cooldown_until = None
    if evt_risk_flag is not None:
        decision.state.last_metrics["evt_risk_flag"] = bool(evt_risk_flag)
    if safety_db_path is not None:
        try:
            save_safety_state(db_path=str(safety_db_path), state=decision.state)
        except Exception:
            pass

    if not decision.ok and bool(executor.cfg.kill_on_any_hard_violation):
        if snapshot_out is not None:
            p = Path(snapshot_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            snap = SystemStateSnapshot(
                task_id=None,
                timestamp=None,
                constitution_hash=str(executor.meta().get("constitution_hash")),
                constitution_yaml=str(executor.meta().get("constitution_yaml")),
                router_mode=str(mode),
                gate_decisions={},
                pcm_budget=dict(pcm_budget or {}),
                active_slots=int(runtime_state.slots.active_count()),
                drawdown=float(drawdown) if drawdown is not None else None,
                observability=None,
                live_dashboard=(
                    dict((snapshot_extra or {}).get("live_dashboard") or {})
                    if snapshot_extra is not None
                    else None
                ),
                kpi_gate=None,
                safety_state=decision.state.as_dict(),
                overrides=[],
            )
            write_state_snapshot(out_path=str(p), snapshot=snap)
        raise ConstitutionViolation(
            code="SAFETY_HALT",
            message=f"Safety halted: {', '.join(decision.reasons or [])}",
            context={
                "reasons": decision.reasons,
                "safety_state": decision.state.as_dict(),
                **st.as_dict(),
                **executor.meta(),
            },
        )

    # Slot reservation (hard constraint)
    executor.reserve_slot(
        st=runtime_state,
        position_id=str(position_id),
        symbol=str(symbol),
        mode=str(mode),
    )
    executor.save_runtime_state(runtime_state)

    snap_path = None
    if snapshot_out is not None:
        p = Path(snapshot_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        snap = SystemStateSnapshot(
            task_id=None,
            timestamp=None,
            constitution_hash=str(executor.meta().get("constitution_hash")),
            constitution_yaml=str(executor.meta().get("constitution_yaml")),
            router_mode=str(mode),
            gate_decisions={},
            pcm_budget=dict(pcm_budget or {}),
            active_slots=int(runtime_state.slots.active_count()),
            drawdown=float(drawdown) if drawdown is not None else None,
            observability=None,
            live_dashboard=(
                dict((snapshot_extra or {}).get("live_dashboard") or {})
                if snapshot_extra is not None
                else None
            ),
            kpi_gate=None,
            safety_state=decision.state.as_dict(),
            overrides=[],
        )
        write_state_snapshot(out_path=str(p), snapshot=snap)
        snap_path = str(p)

    return LiveEnforcementResult(ok=True, reason="ok", snapshot_path=snap_path)
