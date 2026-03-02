from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple


import re

# ATR key pattern: exact 'atr' or timeframe prefix like '15T_atr', '60T_atr', '240T_atr'
_ATR_PATTERN = re.compile(r"^(?:\d+[TtHhDd]_)?atr$")


def pick_atr(feats: Dict[str, Any]) -> Optional[float]:
    """Pick the ATR value from features.

    Only matches exact 'atr' or timeframe-prefixed keys like '15T_atr', '60T_atr'.
    Excludes false positives like 'macd_atr', 'rsi_atr', etc.
    """
    if not feats:
        return None
    # Prefer timeframe-prefixed ATR (e.g. '240T_atr')
    for k in feats:
        if k != "atr" and _ATR_PATTERN.match(k):
            try:
                v = float(feats[k])
                if v > 0:
                    return v
            except Exception:
                continue
    # Fallback to plain 'atr'
    if "atr" in feats:
        try:
            v = float(feats["atr"])
            if v > 0:
                return v
        except Exception:
            pass
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
