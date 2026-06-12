"""Shared-account replay for chop_grid + trend_scalp backtest exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

StrategyName = Literal["chop_grid", "trend_scalp"]


@dataclass
class MultilegGateStats:
    blocked_chop_segments: int = 0
    blocked_trend_segments: int = 0
    peak_chop_symbols: int = 0
    peak_symbol_conflicts: int = 0
    blocked_chop_segment_ids: Set[str] = field(default_factory=set)
    blocked_trend_segment_ids: Set[str] = field(default_factory=set)


def _prep_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"] == "ok"]
    out["start"] = pd.to_datetime(out["start"], utc=True)
    out["end"] = pd.to_datetime(out["end"], utc=True)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["segment_id"] = out["segment_id"].astype(str)
    return out.dropna(subset=["start", "end", "segment_id", "symbol"])


def apply_multileg_segment_gates(
    chop_segments: pd.DataFrame,
    trend_segments: pd.DataFrame,
    *,
    max_concurrent_multi_leg_symbols: int = 0,
    strategy_priority: Tuple[StrategyName, ...] = ("chop_grid", "trend_scalp"),
) -> MultilegGateStats:
    """Replay segment starts/ends on one account timeline.

    - ``max_concurrent_multi_leg_symbols``: shared cap on distinct symbols
      across both chop_grid + trend_scalp (0 = unlimited).
    - Per-symbol mutex: if a symbol is owned by one strategy, the other cannot
      start a new segment on that symbol until the owner segment ends.
    - Same-timestamp tie-break: ends before starts; starts follow ``strategy_priority``.
    """
    stats = MultilegGateStats()
    chop = _prep_segments(chop_segments)
    trend = _prep_segments(trend_segments)
    if chop.empty and trend.empty:
        return stats

    pri = {name: i for i, name in enumerate(strategy_priority)}
    events: List[tuple] = []
    for _, row in chop.iterrows():
        events.append(
            (row["start"], "start", "chop_grid", row["segment_id"], row["symbol"])
        )
        events.append((row["end"], "end", "chop_grid", row["segment_id"], row["symbol"]))
    for _, row in trend.iterrows():
        events.append(
            (row["start"], "start", "trend_scalp", row["segment_id"], row["symbol"])
        )
        events.append(
            (row["end"], "end", "trend_scalp", row["segment_id"], row["symbol"])
        )

    def _sort_key(ev: tuple) -> tuple:
        ts, kind, strategy, _seg_id, _sym = ev
        return (ts, 0 if kind == "end" else 1, pri.get(strategy, 99))

    events.sort(key=_sort_key)

    symbol_owner: Dict[str, StrategyName] = {}
    active_ml_symbols: Set[str] = set()
    blocked_chop: Set[str] = set()
    blocked_trend: Set[str] = set()
    peak_ml_symbols = 0
    conflicts = 0

    for _ts, kind, strategy, seg_id, sym in events:
        if kind == "end":
            if strategy == "chop_grid" and sym in active_ml_symbols:
                active_ml_symbols.discard(sym)
            if strategy == "trend_scalp" and sym in active_ml_symbols:
                active_ml_symbols.discard(sym)
            if symbol_owner.get(sym) == strategy:
                del symbol_owner[sym]
            continue

        if strategy == "chop_grid" and seg_id in blocked_chop:
            continue
        if strategy == "trend_scalp" and seg_id in blocked_trend:
            continue

        owner = symbol_owner.get(sym)
        if owner is not None and owner != strategy:
            conflicts += 1
            if strategy == "chop_grid":
                blocked_chop.add(seg_id)
            else:
                blocked_trend.add(seg_id)
            continue

        if max_concurrent_multi_leg_symbols > 0:
            if sym not in active_ml_symbols and len(active_ml_symbols) >= max_concurrent_multi_leg_symbols:
                if strategy == "chop_grid":
                    blocked_chop.add(seg_id)
                else:
                    blocked_trend.add(seg_id)
                continue
            active_ml_symbols.add(sym)
            peak_ml_symbols = max(peak_ml_symbols, len(active_ml_symbols))

        symbol_owner[sym] = strategy

    stats.blocked_chop_segment_ids = blocked_chop
    stats.blocked_trend_segment_ids = blocked_trend
    stats.blocked_chop_segments = len(blocked_chop)
    stats.blocked_trend_segments = len(blocked_trend)
    stats.peak_chop_symbols = peak_ml_symbols
    stats.peak_symbol_conflicts = conflicts
    return stats


def filter_trades_by_segment_blocks(
    trades: pd.DataFrame,
    blocked_segment_ids: Set[str],
) -> pd.DataFrame:
    if trades.empty or not blocked_segment_ids or "segment_id" not in trades.columns:
        return trades
    return trades[~trades["segment_id"].astype(str).isin(blocked_segment_ids)].copy()


def tag_trades(trades: pd.DataFrame, strategy: StrategyName) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    out["strategy"] = strategy
    return out


def load_chop_trades(root: Path | str, market_segment: str) -> pd.DataFrame:
    path = Path(root) / market_segment / "grid_trades.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return tag_trades(df, "chop_grid")


def load_trend_trades(root: Path | str, market_segment: str) -> pd.DataFrame:
    path = Path(root) / market_segment / "dual_add_trades.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return tag_trades(df, "trend_scalp")


def load_chop_segments(root: Path | str, market_segment: str) -> pd.DataFrame:
    path = Path(root) / market_segment / "grid_segments.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_trend_segments(root: Path | str, market_segment: str) -> pd.DataFrame:
    path = Path(root) / market_segment / "dual_add_segments.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def simulate_account_trades(
    trades: pd.DataFrame,
    *,
    equity: float,
    unit_notional: float,
    unit_by_strategy: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Replay leg fills on one shared account (fixed or per-strategy unit_notional)."""
    events: List[tuple] = []
    for _, r in trades.iterrows():
        sym = str(r.get("symbol", "") or "")
        strat = str(r.get("strategy", "") or "")
        if unit_by_strategy and strat in unit_by_strategy:
            unit = float(unit_by_strategy[strat])
        else:
            unit = float(unit_notional)
        events.append((r["entry_time"], "open", sym, 0.0, unit))
        events.append(
            (r["exit_time"], "close", sym, float(r["pnl_pct"]) * unit, unit)
        )
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1))

    gross = 0.0
    peak_gross = 0.0
    sym_gross: Dict[str, float] = {}
    peak_sym = 0.0
    cum_pnl = 0.0
    peak_eq = float(equity)
    max_dd = 0.0
    for _, kind, _sym, pnl, unit in events:
        if kind == "open":
            gross += unit
            sym_gross[_sym] = sym_gross.get(_sym, 0.0) + unit
        else:
            gross -= unit
            sym_gross[_sym] = sym_gross.get(_sym, 0.0) - unit
            cum_pnl += pnl
        peak_gross = max(peak_gross, gross)
        if sym_gross:
            peak_sym = max(peak_sym, max(sym_gross.values()))
        eq = equity + cum_pnl
        peak_eq = max(peak_eq, eq)
        max_dd = min(max_dd, eq - peak_eq)
    return {
        "ret_pct": cum_pnl / equity * 100.0,
        "pnl_usd": cum_pnl,
        "peak_gross_pct": peak_gross / equity * 100.0,
        "peak_sym_pct": peak_sym / equity * 100.0,
        "max_dd_pct": max_dd / equity * 100.0,
        "n_trades": float(len(trades)),
    }
