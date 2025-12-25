from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RouterHeads:
    """
    Path primitives (model outputs) to be consumed by Routers / Allocators.

    Values are expected to be *already aligned* to the decision timestamp.
    """

    dir_score: float
    mfe_atr: float
    mae_atr: float
    t_to_mfe: float
    persistence: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        d = {
            "dir_score": float(self.dir_score),
            "mfe_atr": float(self.mfe_atr),
            "mae_atr": float(self.mae_atr),
            "t_to_mfe": float(self.t_to_mfe),
        }
        if self.persistence is not None:
            d["persistence"] = float(self.persistence)
        return d


@dataclass(frozen=True)
class RouterContext:
    """
    Context that is not a model head but is needed for routing decisions.
    """

    timestamp: str  # ISO string for portability (logs, offline replay)
    symbol: str
    timeframe: str = "4H"

    # Optional: regime / risk state from monitoring layer
    regime_score: Optional[float] = None

    def as_dict(self) -> Dict[str, str | float]:
        d: Dict[str, str | float] = {
            "timestamp": str(self.timestamp),
            "symbol": str(self.symbol),
            "timeframe": str(self.timeframe),
        }
        if self.regime_score is not None:
            d["regime_score"] = float(self.regime_score)
        return d


@dataclass(frozen=True)
class RouterAction:
    """
    RL action space (decision management), intentionally does NOT include direct entry/exit mechanics.

    - router_enabled: enable/disable specific routers (e.g., "sr_reversal", "breakout", "trend")
    - capital_multiplier: per-router scalar to scale exposure/risk budget
    - global_pause: emergency brake (no new positions)
    """

    router_enabled: Dict[str, bool]
    capital_multiplier: Dict[str, float]
    global_pause: bool = False

    def clipped(
        self, *, min_mult: float = 0.0, max_mult: float = 2.0
    ) -> "RouterAction":
        cm = {}
        for k, v in (self.capital_multiplier or {}).items():
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            cm[k] = max(min(fv, max_mult), min_mult)
        re = {str(k): bool(v) for k, v in (self.router_enabled or {}).items()}
        return RouterAction(
            router_enabled=re,
            capital_multiplier=cm,
            global_pause=bool(self.global_pause),
        )

    def as_dict(self) -> Dict:
        return {
            "global_pause": bool(self.global_pause),
            "router_enabled": {
                str(k): bool(v) for k, v in (self.router_enabled or {}).items()
            },
            "capital_multiplier": {
                str(k): float(v) for k, v in (self.capital_multiplier or {}).items()
            },
        }


@dataclass(frozen=True)
class RouterDecision:
    """
    Router output after consuming heads + context + action.
    """

    router_name: str
    gated: bool
    score: float
    position_size: float

    def as_dict(self) -> Dict[str, float | bool | str]:
        return {
            "router_name": str(self.router_name),
            "gated": bool(self.gated),
            "score": float(self.score),
            "position_size": float(self.position_size),
        }
