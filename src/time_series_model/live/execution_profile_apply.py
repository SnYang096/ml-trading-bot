from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple


def pick_atr(feats: Dict[str, Any]) -> Optional[float]:
    # Prefer timeframe ATR keys like "15T_atr" or "4H_atr".
    for k in feats or {}:
        if str(k).endswith("_atr"):
            try:
                return float(feats[k])
            except Exception:
                continue
    if "atr" in (feats or {}):
        try:
            return float(feats["atr"])
        except Exception:
            return None
    return None


def compute_rr_prices(
    *,
    side: str,
    entry_price: float,
    atr: float,
    stop_loss_r: float,
    take_profit_r: float,
) -> Tuple[Optional[float], Optional[float]]:
    if entry_price <= 0 or atr <= 0:
        return None, None
    sl_r = float(stop_loss_r)
    tp_r = float(take_profit_r)
    if str(side).upper() in {"BUY", "LONG"}:
        stop_loss = entry_price - sl_r * atr
        take_profit = entry_price + tp_r * atr
    else:
        stop_loss = entry_price + sl_r * atr
        take_profit = entry_price - tp_r * atr
    return float(stop_loss), float(take_profit)


def compute_trailing_stop(
    *,
    side: str,
    current_price: float,
    atr: float,
    trailing_atr: float,
) -> Optional[float]:
    if current_price <= 0 or atr <= 0 or trailing_atr is None:
        return None
    trail = float(trailing_atr)
    if str(side).upper() in {"BUY", "LONG"}:
        return float(current_price - trail * atr)
    return float(current_price + trail * atr)


def holding_expired(
    *,
    entry_time: datetime,
    now: datetime,
    max_holding_bars: Optional[int],
    bar_minutes: int,
) -> bool:
    if max_holding_bars is None or max_holding_bars <= 0:
        return False
    if bar_minutes <= 0:
        return False
    max_age = timedelta(minutes=int(max_holding_bars) * int(bar_minutes))
    return now >= entry_time + max_age
