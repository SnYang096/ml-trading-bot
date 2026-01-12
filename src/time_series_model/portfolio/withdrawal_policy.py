from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class WithdrawalRulesV1:
    max_monthly_withdrawal_ratio: float = 0.02
    max_annual_profit_withdrawal_ratio: float = 0.40
    forbid_when_global_pause: bool = True


def load_withdrawal_rules_v1(path: str | Path) -> WithdrawalRulesV1:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    r = obj.get("rules") or {}
    return WithdrawalRulesV1(
        max_monthly_withdrawal_ratio=float(r.get("max_monthly_withdrawal_ratio", 0.02)),
        max_annual_profit_withdrawal_ratio=float(
            r.get("max_annual_profit_withdrawal_ratio", 0.40)
        ),
        forbid_when_global_pause=bool(r.get("forbid_when_global_pause", True)),
    )


def validate_withdrawal_request(
    *,
    rules: WithdrawalRulesV1,
    equity_usd: float,
    withdraw_usd: float,
    withdrawn_this_month_usd: float,
    realized_profit_ytd_usd: float,
    withdrawn_profit_ytd_usd: float,
    global_pause: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Deterministic withdrawal compliance check.
    """
    eq = float(max(0.0, equity_usd))
    w = float(max(0.0, withdraw_usd))
    ctx = {
        "equity_usd": eq,
        "withdraw_usd": w,
        "withdrawn_this_month_usd": float(max(0.0, withdrawn_this_month_usd)),
        "realized_profit_ytd_usd": float(realized_profit_ytd_usd),
        "withdrawn_profit_ytd_usd": float(max(0.0, withdrawn_profit_ytd_usd)),
        "global_pause": bool(global_pause),
    }
    if w <= 0.0:
        return True, "ok_zero", ctx
    if eq <= 0.0:
        return False, "equity_non_positive", ctx
    if bool(rules.forbid_when_global_pause) and bool(global_pause):
        return False, "forbid_when_global_pause", ctx

    # Monthly cap
    cap_m = float(rules.max_monthly_withdrawal_ratio) * eq
    if float(ctx["withdrawn_this_month_usd"]) + w > cap_m + 1e-9:
        ctx["monthly_cap_usd"] = cap_m
        return False, "monthly_cap_exceeded", ctx

    # Annual profit cap (only if profit positive)
    profit = float(realized_profit_ytd_usd)
    if profit > 0.0:
        cap_p = float(rules.max_annual_profit_withdrawal_ratio) * profit
        if float(ctx["withdrawn_profit_ytd_usd"]) + w > cap_p + 1e-9:
            ctx["annual_profit_cap_usd"] = cap_p
            return False, "annual_profit_cap_exceeded", ctx

    return True, "ok", ctx
