from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeListConfig:
    mode: str  # "long_only" | "market_neutral"
    gross_leverage: float
    max_weight: float
    cash_buffer: float


def _loads_json_list(s: Any) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    txt = str(s).strip()
    if not txt:
        return []
    try:
        v = json.loads(txt)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return [x.strip() for x in txt.split(",") if x.strip()]


def _weights_from_lists(
    *,
    cfg: TradeListConfig,
    longs: List[str],
    shorts: List[str],
) -> Dict[str, float]:
    invest_frac = float(1.0 - float(cfg.cash_buffer))
    invest_frac = max(0.0, min(1.0, invest_frac))
    gross = float(cfg.gross_leverage) * invest_frac
    mw = float(cfg.max_weight)
    w: Dict[str, float] = {}

    if cfg.mode == "long_only":
        if not longs:
            return {}
        base = gross / float(len(longs))
        base = min(base, mw)
        for s in longs:
            w[str(s)] = float(base)
        return w

    if cfg.mode == "market_neutral":
        if not longs or not shorts:
            return {}
        half = gross / 2.0
        wl = min(half / float(len(longs)), mw)
        ws = min(half / float(len(shorts)), mw)
        for s in longs:
            w[str(s)] = float(wl)
        for s in shorts:
            w[str(s)] = -float(ws)
        return w

    raise ValueError(f"Unknown mode: {cfg.mode}")


def build_trade_list_from_rebalance_log(
    *,
    close: pd.Series,
    rb: pd.DataFrame,
    cfg: TradeListConfig,
    entry_timestamps: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Build a per-symbol trade list from a rebalance audit log.

    Inputs:
      - close: Series indexed by MultiIndex (timestamp, symbol) -> close price
      - rb: rebalance log containing:
          - rebalance_ts
          - long_symbols_json / short_symbols_json (JSON array string)
      - entry_timestamps: optional; if provided, we use these as the entry timestamps
        and define exit as the next timestamp. This matches "holding period" style returns.
        If omitted, we use rb['rebalance_ts'] as entry timestamps.
    """
    if not isinstance(close.index, pd.MultiIndex) or close.index.nlevels != 2:
        raise ValueError("close must be indexed by MultiIndex (timestamp, symbol)")
    if rb is None or rb.empty:
        return pd.DataFrame()
    if "rebalance_ts" not in rb.columns:
        raise KeyError("rb must contain rebalance_ts")
    if "long_symbols_json" not in rb.columns or "short_symbols_json" not in rb.columns:
        raise KeyError("rb must contain long_symbols_json and short_symbols_json")

    rb2 = rb.copy()
    rb2["rebalance_ts"] = pd.to_datetime(rb2["rebalance_ts"], utc=True, errors="coerce")
    rb2 = (
        rb2.dropna(subset=["rebalance_ts"])
        .sort_values("rebalance_ts")
        .reset_index(drop=True)
    )
    if rb2.empty:
        return pd.DataFrame()

    if entry_timestamps is None:
        entry_ts = pd.DatetimeIndex(rb2["rebalance_ts"].tolist())
    else:
        entry_ts = pd.to_datetime(entry_timestamps, utc=True, errors="coerce")
        entry_ts = pd.DatetimeIndex([t for t in entry_ts if pd.notna(t)])

    # exits are next entry timestamp; last one has no exit
    if len(entry_ts) < 2:
        return pd.DataFrame()
    entry_ts = entry_ts.sort_values()
    exit_ts = entry_ts[1:]
    entry_ts2 = entry_ts[:-1]

    close = pd.to_numeric(close, errors="coerce")
    # normalize timestamp level to utc
    ts_level = pd.to_datetime(
        close.index.get_level_values(0), utc=True, errors="coerce"
    )
    close.index = pd.MultiIndex.from_arrays(
        [ts_level, close.index.get_level_values(1)], names=["timestamp", "symbol"]
    )

    rows: List[Dict[str, Any]] = []
    # map entry time -> rb row (by nearest exact match)
    rb_map = {pd.Timestamp(t): i for i, t in enumerate(rb2["rebalance_ts"].tolist())}

    for t0, t1 in zip(entry_ts2, exit_ts):
        i = rb_map.get(pd.Timestamp(t0))
        if i is None:
            # if wf aggregation uses synthetic timestamps, we skip
            continue
        longs = _loads_json_list(rb2.loc[i, "long_symbols_json"])
        shorts = _loads_json_list(rb2.loc[i, "short_symbols_json"])
        w = _weights_from_lists(
            cfg=cfg,
            longs=[s.upper() for s in longs],
            shorts=[s.upper() for s in shorts],
        )
        for sym, ww in w.items():
            try:
                a = float(close.loc[(t0, sym)])
                b = float(close.loc[(t1, sym)])
            except Exception:
                continue
            if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
                continue
            r = b / a - 1.0
            rows.append(
                {
                    "rebalance_ts": pd.Timestamp(t0),
                    "exit_ts": pd.Timestamp(t1),
                    "symbol": str(sym),
                    "weight": float(ww),
                    "entry_price": float(a),
                    "exit_price": float(b),
                    "return": float(r),
                    "side": "LONG" if ww > 0 else "SHORT",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["rebalance_ts"] = pd.to_datetime(
            out["rebalance_ts"], utc=True, errors="coerce"
        )
        out["exit_ts"] = pd.to_datetime(out["exit_ts"], utc=True, errors="coerce")
        out = out.dropna(subset=["rebalance_ts", "exit_ts"]).sort_values(
            ["rebalance_ts", "symbol"]
        )
    return out
