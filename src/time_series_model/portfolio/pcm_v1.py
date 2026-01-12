from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from time_series_model.rl.router_types import RouterAction


@dataclass(frozen=True)
class ConstitutionKillSwitch:
    enabled: bool = True
    daily_loss_limit: float = 0.04
    weekly_loss_limit: float = 0.08
    monthly_loss_limit: float = 0.12
    kill_on_any_hard_violation: bool = True


@dataclass(frozen=True)
class ConstitutionSlots:
    enabled: bool = True
    slot_count: int = 2
    risk_per_slot: float = 0.015


@dataclass(frozen=True)
class ConstitutionEscalation:
    enabled: bool = True
    default_enabled: bool = False
    risk_per_slot_multiplier: float = 1.5
    trend_budget_multiplier: float = 1.3


@dataclass(frozen=True)
class ConstitutionV1:
    version: int = 1
    name: str = "Constitution_v1"
    kill_switch: ConstitutionKillSwitch = ConstitutionKillSwitch()
    slots: ConstitutionSlots = ConstitutionSlots()
    capital_escalation: ConstitutionEscalation = ConstitutionEscalation()


@dataclass(frozen=True)
class RiskState:
    """
    Minimal risk state snapshot for capital policy.

    This is intentionally small; production systems should extend it, but keep
    the contract stable (add optional fields, don't break existing ones).
    """

    # Loss since period start, as a fraction of equity. Positive means loss.
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0

    # Execution/control layer flags.
    hard_violation: bool = False
    data_bad: bool = False

    # Optional: drawdown (fraction), used for escalation eligibility in higher versions.
    recent_max_dd: Optional[float] = None


@dataclass(frozen=True)
class SymbolDecision:
    """
    Router/gate output for a symbol at a decision timestamp.
    """

    symbol: str
    mode: str  # NO_TRADE|MEAN|TREND
    gated: bool = True

    # A ranking signal (higher => more budget). This is NOT a probability.
    score: float = 0.0


@dataclass(frozen=True)
class PCMPolicyV1:
    """
    A simple, explainable capital policy:
    - allocate a fixed budget per mode (MEAN vs TREND)
    - within each mode, split budget across symbols by soft ranking on score
    - apply constitution: kill-switch => global_pause; escalation => bounded multipliers
    """

    base_mode_budgets: Dict[str, float]
    allow_escalation: bool = False


@dataclass(frozen=True)
class PCMResult:
    global_pause: bool
    per_mode_budget: Dict[str, float]
    per_symbol_budget: Dict[str, float]
    reasons: List[str]

    def as_json(self) -> str:
        return json.dumps(
            {
                "global_pause": bool(self.global_pause),
                "per_mode_budget": dict(self.per_mode_budget),
                "per_symbol_budget": dict(self.per_symbol_budget),
                "reasons": list(self.reasons),
            },
            ensure_ascii=False,
            indent=2,
        )

    def as_router_action(self) -> RouterAction:
        """
        Optional bridge: represent mode budgets as RouterAction multipliers.

        Convention (3-action routers):
        - router_name == "MEAN" or "TREND"
        - capital_multiplier[mode] scales base position_size produced by router

        If the downstream router emits per-symbol decisions instead of per-mode,
        prefer using per_symbol_budget directly and ignore this action.
        """

        cm = {}
        for k, v in (self.per_mode_budget or {}).items():
            kk = str(k).upper()
            if kk in {"MEAN", "TREND"}:
                cm[kk] = float(max(0.0, v))
        return RouterAction(
            router_enabled={},
            capital_multiplier=cm,
            global_pause=bool(self.global_pause),
        )


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_constitution_v1(path: str | Path) -> ConstitutionV1:
    obj = _load_yaml(path)
    ks = obj.get("kill_switch") or {}
    slots = obj.get("slots") or {}
    esc = obj.get("capital_escalation") or {}
    esc_actions = esc.get("allowed_actions") or {}

    return ConstitutionV1(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "Constitution_v1")),
        kill_switch=ConstitutionKillSwitch(
            enabled=bool(ks.get("enabled", True)),
            daily_loss_limit=float(ks.get("daily_loss_limit", 0.04)),
            weekly_loss_limit=float(ks.get("weekly_loss_limit", 0.08)),
            monthly_loss_limit=float(ks.get("monthly_loss_limit", 0.12)),
            kill_on_any_hard_violation=bool(ks.get("kill_on_any_hard_violation", True)),
        ),
        slots=ConstitutionSlots(
            enabled=bool(slots.get("enabled", True)),
            slot_count=int(slots.get("slot_count", 2)),
            risk_per_slot=float(slots.get("risk_per_slot", 0.015)),
        ),
        capital_escalation=ConstitutionEscalation(
            enabled=bool(esc.get("enabled", True)),
            default_enabled=bool(esc.get("default_enabled", False)),
            risk_per_slot_multiplier=float(
                esc_actions.get("risk_per_slot_multiplier", 1.5)
            ),
            trend_budget_multiplier=float(
                esc_actions.get("trend_budget_multiplier", 1.3)
            ),
        ),
    )


def _normalize_mode_budgets(budgets: Dict[str, float]) -> Dict[str, float]:
    b = {str(k).upper(): float(max(0.0, v)) for k, v in (budgets or {}).items()}
    # Ensure keys exist
    for k in ["NO_TRADE", "MEAN", "TREND"]:
        b.setdefault(k, 0.0)
    # NO_TRADE is always 0 (budget is defined only for tradable modes)
    b["NO_TRADE"] = 0.0
    s = float(b.get("MEAN", 0.0) + b.get("TREND", 0.0))
    if s <= 0:
        return {"NO_TRADE": 0.0, "MEAN": 0.0, "TREND": 0.0}
    return {
        "NO_TRADE": 0.0,
        "MEAN": float(b["MEAN"] / s),
        "TREND": float(b["TREND"] / s),
    }


def _kill_switch_triggered(
    *, c: ConstitutionV1, risk: RiskState
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not c.kill_switch.enabled:
        return False, reasons
    if float(risk.daily_loss) >= float(c.kill_switch.daily_loss_limit):
        reasons.append("kill_switch:daily_loss_limit")
    if float(risk.weekly_loss) >= float(c.kill_switch.weekly_loss_limit):
        reasons.append("kill_switch:weekly_loss_limit")
    if float(risk.monthly_loss) >= float(c.kill_switch.monthly_loss_limit):
        reasons.append("kill_switch:monthly_loss_limit")
    if bool(risk.data_bad):
        reasons.append("kill_switch:data_bad")
    if bool(risk.hard_violation):
        reasons.append("kill_switch:hard_violation")
    return (len(reasons) > 0), reasons


def allocate_capital_v1(
    *,
    constitution: ConstitutionV1,
    policy: PCMPolicyV1,
    risk: RiskState,
    decisions: Iterable[SymbolDecision],
) -> PCMResult:
    """
    Compute per-mode + per-symbol capital budgets for a single decision timestamp.

    Budgets are *fractions* of total risk budget (not absolute cash),
    and MUST be interpreted under the slot/risk-per-slot constraints.
    """

    kill, kill_reasons = _kill_switch_triggered(c=constitution, risk=risk)
    if kill:
        return PCMResult(
            global_pause=True,
            per_mode_budget={"NO_TRADE": 0.0, "MEAN": 0.0, "TREND": 0.0},
            per_symbol_budget={str(d.symbol): 0.0 for d in decisions},
            reasons=kill_reasons,
        )

    base = _normalize_mode_budgets(policy.base_mode_budgets)

    # Escalation (v1): if enabled, multiply TREND budget (bounded), then renormalize.
    reasons: List[str] = []
    if (
        constitution.capital_escalation.enabled
        and bool(policy.allow_escalation)
        and bool(constitution.capital_escalation.default_enabled)
    ):
        base_trend = float(base.get("TREND", 0.0)) * float(
            constitution.capital_escalation.trend_budget_multiplier
        )
        base_mean = float(base.get("MEAN", 0.0))
        s = float(base_trend + base_mean)
        if s > 0:
            base = {"NO_TRADE": 0.0, "MEAN": base_mean / s, "TREND": base_trend / s}
            reasons.append("capital_escalation:trend_budget_multiplier")

    # Group decisions by mode, but only if gated.
    per_symbol_budget: Dict[str, float] = {}
    mode_to_items: Dict[str, List[SymbolDecision]] = {"MEAN": [], "TREND": []}
    for d in decisions:
        sym = str(d.symbol)
        mode = str(d.mode).upper()
        if (not bool(d.gated)) or mode == "NO_TRADE":
            per_symbol_budget[sym] = 0.0
            continue
        if mode in mode_to_items:
            mode_to_items[mode].append(d)
        else:
            # Unknown mode: treat as NO_TRADE in v1 (low freedom)
            per_symbol_budget[sym] = 0.0
            reasons.append(f"unknown_mode_as_no_trade:{mode}")

    # Split each mode budget across symbols in that mode by score.
    for mode in ["MEAN", "TREND"]:
        items = mode_to_items.get(mode) or []
        b = float(base.get(mode, 0.0))
        if not items or b <= 0:
            for it in items:
                per_symbol_budget[str(it.symbol)] = 0.0
            continue

        # Score-based soft split: shift by min score to keep non-negative weights.
        scores = [float(it.score) for it in items]
        mn = min(scores) if scores else 0.0
        weights = [max(0.0, s - mn) for s in scores]
        sw = float(sum(weights))
        if sw <= 1e-12:
            # fallback: equal split
            w = 1.0 / float(len(items))
            for it in items:
                per_symbol_budget[str(it.symbol)] = b * w
        else:
            for it, w in zip(items, weights):
                per_symbol_budget[str(it.symbol)] = b * float(w / sw)

    per_mode_budget = dict(base)
    return PCMResult(
        global_pause=False,
        per_mode_budget=per_mode_budget,
        per_symbol_budget=per_symbol_budget,
        reasons=reasons,
    )
