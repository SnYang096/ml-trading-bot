from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .router_types import RouterAction, RouterContext, RouterDecision, RouterHeads


def _json_dumps_safe(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


@dataclass(frozen=True)
class RouterStepLog:
    """
    One step record for offline replay / RL training.

    This is purposely "wide" and explicit to avoid silent schema drift.
    """

    ctx: RouterContext
    heads: RouterHeads
    action: RouterAction
    decision: RouterDecision

    # High-level mode label for RL/BC (recommended: NO_TRADE/MEAN/TREND).
    # This decouples RL from non-orthogonal strategy naming (sr/breakout/...)
    mode: Optional[str] = None

    # Realized outcomes (filled after step completes / in backtest)
    pnl: Optional[float] = None
    turnover: Optional[float] = None
    cost: Optional[float] = None
    equity: Optional[float] = None
    drawdown: Optional[float] = None

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        row.update(self.ctx.as_dict())
        row.update({f"head_{k}": v for k, v in self.heads.as_dict().items()})
        row.update(self.decision.as_dict())
        row["mode"] = None if self.mode is None else str(self.mode)

        # Store action as JSON to keep parquet/csv stable
        row["action_json"] = _json_dumps_safe(self.action.as_dict())

        row["pnl"] = None if self.pnl is None else float(self.pnl)
        row["turnover"] = None if self.turnover is None else float(self.turnover)
        row["cost"] = None if self.cost is None else float(self.cost)
        row["equity"] = None if self.equity is None else float(self.equity)
        row["drawdown"] = None if self.drawdown is None else float(self.drawdown)
        return row


class RouterEpisodeLogger:
    """
    Append-only logger for RouterStepLog records.
    """

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def append(self, rec: RouterStepLog) -> None:
        self._rows.append(rec.to_row())

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)

    def save_parquet(self, path: str) -> str:
        df = self.to_frame()
        df.to_parquet(path, index=False)
        return path

    def save_csv(self, path: str) -> str:
        df = self.to_frame()
        df.to_csv(path, index=False)
        return path
