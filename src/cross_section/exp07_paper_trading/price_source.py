"""可插拔最新价来源：本地 parquet（回测/重放）或 Binance USDT-M 公开 ticker（真实轮询）。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

PriceSourceName = Literal["parquet", "binance_futures"]


def _normalize_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    return ts


def last_price_from_parquet(
    symbol: str,
    price_dir: Path,
    as_of: Optional[pd.Timestamp] = None,
) -> Optional[Tuple[float, pd.Timestamp]]:
    files = sorted(price_dir.glob(f"{symbol}_*.parquet"))
    if not files:
        return None
    for f in reversed(files):
        try:
            df = pd.read_parquet(f, columns=["timestamp", "price"])
        except Exception:
            continue
        df = (
            df.assign(timestamp=_normalize_ts(df["timestamp"]))
            .set_index("timestamp")
            .sort_index()
        )
        if as_of is not None:
            df = df.loc[:as_of]
        if len(df) == 0:
            continue
        return float(df["price"].iloc[-1]), df.index[-1]
    return None


def last_prices_binance_futures(symbols: List[str]) -> Dict[str, float]:
    """GET /fapi/v1/ticker/price 逐符号（公开接口，无需 API key）。"""
    out: Dict[str, float] = {}
    for sym in symbols:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            out[sym] = float(data["price"])
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            KeyError,
            ValueError,
            TimeoutError,
        ):
            continue
        time.sleep(0.05)  # 轻量限速
    return out


def fetch_last_prices(
    symbols: List[str],
    *,
    source: PriceSourceName,
    price_dir: Path,
    as_of: Optional[pd.Timestamp] = None,
    poll_sec: float = 0.0,
    poll_max: int = 1,
) -> Dict[str, Tuple[float, str]]:
    """返回 symbol -> (price, source_detail)。

    poll_sec / poll_max：对 binance 源可多次拉取（例如等 K 线落盘）；parquet 下仅执行一次。
    """
    symbols = list(dict.fromkeys(symbols))
    best: Dict[str, Tuple[float, str]] = {}
    n = max(1, int(poll_max))
    for i in range(n):
        if source == "parquet":
            for sym in symbols:
                r = last_price_from_parquet(sym, price_dir, as_of)
                if r:
                    best[sym] = (r[0], f"parquet@{r[1]}")
        elif source == "binance_futures":
            px = last_prices_binance_futures(symbols)
            for sym, p in px.items():
                best[sym] = (p, "binance_futures")
        else:
            raise ValueError(source)
        if i + 1 < n and poll_sec > 0:
            time.sleep(poll_sec)
    return best
