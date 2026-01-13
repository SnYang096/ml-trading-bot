from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.state import ConstitutionState
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
    hard_violation: bool = False,
    data_bad: bool = False,
    snapshot_out: Optional[str | Path] = None,
    snapshot_extra: Optional[Dict[str, Any]] = None,
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
    executor.validate_drawdown(state=st)

    # Execution whitelist constitution (hard allow-list per regime)
    executor.validate_execution_strategy(
        regime=str(mode),
        strategy_id=str(execution_strategy or "").strip(),
        tags=execution_tags,
        evidence=execution_evidence,
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
            pcm_budget={},
            active_slots=int(runtime_state.slots.active_count()),
            drawdown=float(drawdown) if drawdown is not None else None,
            observability=None,
            live_dashboard=(
                dict((snapshot_extra or {}).get("live_dashboard") or {})
                if snapshot_extra is not None
                else None
            ),
            kpi_gate=None,
            overrides=[],
        )
        write_state_snapshot(out_path=str(p), snapshot=snap)
        snap_path = str(p)

    return LiveEnforcementResult(ok=True, reason="ok", snapshot_path=snap_path)
