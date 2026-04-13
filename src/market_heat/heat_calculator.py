"""Core heat score computation per symbol.

Heat is derived from weekly EMA50: price position relative to EMA + EMA slope.

States:
  HOT  — price > EMA50 and EMA slope > 0 (bullish trend accelerating)
  WARM — price > EMA50 but EMA slope <= 0 (above average but momentum fading)
  COLD — price < EMA50 (bearish / weak)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class HeatResult:
    symbol: str
    state: str  # HOT / WARM / COLD
    score: float  # 0.0 ~ 1.0 continuous
    ema_slope: float  # 4-week normalized EMA change
    ema_distance: float  # price distance from EMA (fraction)
    ema_value: float  # latest EMA value
    price: float  # latest close price


def compute_heat(
    symbol: str,
    weekly_closes: pd.Series,
    ema_period: int = 50,
    slope_lookback: int = 4,
) -> Optional[HeatResult]:
    """Compute heat score for a single symbol from weekly close prices.

    Args:
        symbol: Base symbol (e.g. 'BTC').
        weekly_closes: Series of weekly close prices (at least ema_period + slope_lookback rows).
        ema_period: EMA span (default 50 weeks ~ 1 year).
        slope_lookback: Number of weeks to measure EMA slope over.

    Returns:
        HeatResult or None if insufficient data.
    """
    if len(weekly_closes) < ema_period:
        return None

    ema = weekly_closes.ewm(span=ema_period, adjust=False).mean()

    if len(ema) < slope_lookback + 1:
        return None

    current_ema = float(ema.iloc[-1])
    past_ema = float(ema.iloc[-slope_lookback])
    current_price = float(weekly_closes.iloc[-1])

    if past_ema == 0 or current_ema == 0:
        return None

    slope = (current_ema - past_ema) / past_ema
    distance = (current_price - current_ema) / current_ema

    if distance > 0 and slope > 0:
        state = "HOT"
        score = min(1.0, 0.5 + distance * 5 + slope * 10)
    elif distance > 0:
        state = "WARM"
        score = max(0.2, min(0.5, 0.5 + slope * 10))
    else:
        state = "COLD"
        score = max(0.0, min(0.2, 0.2 + distance * 5))

    return HeatResult(
        symbol=symbol,
        state=state,
        score=round(score, 4),
        ema_slope=round(slope, 6),
        ema_distance=round(distance, 6),
        ema_value=round(current_ema, 4),
        price=round(current_price, 4),
    )


def compute_heat_batch(
    ohlcv_dict: Dict[str, pd.DataFrame],
    ema_period: int = 50,
    slope_lookback: int = 4,
) -> Dict[str, HeatResult]:
    """Compute heat for all symbols in the OHLCV dict.

    Args:
        ohlcv_dict: Mapping from base symbol to DataFrame with 'close' column.

    Returns:
        Mapping from base symbol to HeatResult (symbols with insufficient data are skipped).
    """
    results: Dict[str, HeatResult] = {}
    for symbol, df in ohlcv_dict.items():
        closes = df["close"] if "close" in df.columns else None
        if closes is None or closes.empty:
            continue
        hr = compute_heat(symbol, closes, ema_period, slope_lookback)
        if hr is not None:
            results[symbol] = hr
    return results
