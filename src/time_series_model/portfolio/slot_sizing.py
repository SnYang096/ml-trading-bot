from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


@dataclass(frozen=True)
class SlotSizingResult:
    """
    Deterministic sizing output (USD-based, futures/perp friendly).

    qty: base-asset quantity (e.g., BTC amount) assuming linear contract.
    notional_usd: abs position notional in USD.
    stop_return_frac: estimated worst-case return at stop (fraction of price).
    """

    qty: float
    notional_usd: float
    stop_return_frac: float


def estimate_stop_return_frac(*, price: float, atr: float, stop_atr: float) -> float:
    """
    Convert a stop distance expressed in ATR units into an approximate return fraction.

    Example:
      atr_pct = atr / price
      stop_atr = 1.2  => stop_return ~= 1.2 * atr_pct
    """
    p = float(price)
    if p <= 0:
        return 0.0
    atr_pct = float(max(0.0, float(atr))) / p
    return float(max(0.0, float(stop_atr))) * atr_pct


def compute_slot_size_from_risk(
    *,
    equity_usd: float,
    risk_frac: float,
    price: float,
    atr: float,
    stop_atr: float,
    max_leverage: float = 3.0,
    min_qty: float = 0.0,
) -> SlotSizingResult:
    """
    Map per-slot risk budget (fraction of equity) to a contract quantity.

    Assumptions (low DOF / conservative):
    - worst loss at stop ~= notional * stop_return_frac
    - require: notional * stop_return_frac <= equity * risk_frac
    - cap notional by max_leverage: notional <= equity * max_leverage

    If stop_return_frac is 0 (bad inputs), returns qty=0.
    """
    eq = float(max(0.0, equity_usd))
    rf = float(_clamp(risk_frac, 0.0, 1.0))
    px = float(price)
    if eq <= 0 or rf <= 0 or px <= 0:
        return SlotSizingResult(qty=0.0, notional_usd=0.0, stop_return_frac=0.0)

    stop_ret = estimate_stop_return_frac(
        price=px, atr=float(atr), stop_atr=float(stop_atr)
    )
    if stop_ret <= 1e-12:
        return SlotSizingResult(
            qty=0.0, notional_usd=0.0, stop_return_frac=float(stop_ret)
        )

    # risk budget in USD
    risk_usd = eq * rf
    notional_risk_limited = risk_usd / stop_ret
    notional_leverage_cap = eq * float(max(0.0, max_leverage))
    notional = float(min(notional_risk_limited, notional_leverage_cap))
    qty = float(notional / px)
    if float(min_qty) > 0.0:
        qty = float(max(float(min_qty), qty))
        notional = float(qty * px)
    return SlotSizingResult(
        qty=qty, notional_usd=notional, stop_return_frac=float(stop_ret)
    )


def risk_only_down(
    *, prev_risk_frac: Optional[float], proposed_risk_frac: float
) -> float:
    """
    "Risk only down" rule: risk budget can only decrease, never increase,
    unless there was no previous risk setting.
    """
    pr = None if prev_risk_frac is None else float(prev_risk_frac)
    nr = float(_clamp(proposed_risk_frac, 0.0, 1.0))
    if pr is None:
        return nr
    return float(min(float(_clamp(pr, 0.0, 1.0)), nr))
