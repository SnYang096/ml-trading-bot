from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from nautilus_trader.model import Order
    from nautilus_trader.trading import Strategy

    NAUTILUS_AVAILABLE = True
except Exception:  # pragma: no cover
    NAUTILUS_AVAILABLE = False
    Strategy = object  # type: ignore
    Order = object  # type: ignore

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)
from src.time_series_model.live.enforcement import enforce_before_order


@dataclass(frozen=True)
class GuardedOrderContext:
    position_id: str
    symbol: str
    mode: str
    execution_strategy: str
    execution_tags: Optional[list[str]] = None
    execution_evidence: Optional[dict[str, bool]] = None
    equity: Optional[float] = None
    drawdown: Optional[float] = None
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0
    daily_cost_mean: Optional[float] = None
    daily_turnover_mean: Optional[float] = None
    hard_violation: bool = False
    data_bad: bool = False
    evt_risk_flag: Optional[bool] = None


class ExecutionManager:
    """
    Single entrypoint for live order submission.

    Goal: make it difficult to bypass Constitution enforcement in live.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        executor: ConstitutionExecutor,
        runtime_state: ConstitutionRuntimeState,
    ):
        self.strategy = strategy
        self.executor = executor
        self.runtime_state = runtime_state

    def submit_order_guarded(self, *, order: Order, ctx: GuardedOrderContext) -> None:
        # Hard gate: Constitution + execution whitelist + slot reservation.
        enforce_before_order(
            executor=self.executor,
            runtime_state=self.runtime_state,
            position_id=str(ctx.position_id),
            symbol=str(ctx.symbol),
            mode=str(ctx.mode),
            execution_strategy=str(ctx.execution_strategy),
            execution_tags=ctx.execution_tags,
            execution_evidence=ctx.execution_evidence,
            equity=ctx.equity,
            drawdown=ctx.drawdown,
            daily_loss=float(ctx.daily_loss),
            weekly_loss=float(ctx.weekly_loss),
            monthly_loss=float(ctx.monthly_loss),
            daily_cost_mean=ctx.daily_cost_mean,
            daily_turnover_mean=ctx.daily_turnover_mean,
            hard_violation=bool(ctx.hard_violation),
            data_bad=bool(ctx.data_bad),
            evt_risk_flag=ctx.evt_risk_flag,
        )
        self.strategy.submit_order(order)
