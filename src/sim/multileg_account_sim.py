"""Shared-account replay for chop_grid + trend_scalp backtest exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

StrategyName = Literal["chop_grid", "trend_scalp"]
FuseMode = Literal["hard", "tier_derate", "tier_daily_scaled"]
BAR_MINUTES = 120  # 2h bars; matches MultiLegConcurrencyGate


@dataclass
class MultilegGateStats:
    blocked_chop_segments: int = 0
    blocked_trend_segments: int = 0
    peak_chop_symbols: int = 0
    peak_symbol_conflicts: int = 0
    blocked_chop_segment_ids: Set[str] = field(default_factory=set)
    blocked_trend_segment_ids: Set[str] = field(default_factory=set)
    cooldown_switches: int = 0
    cooldown_delayed_starts: int = 0
    cooldown_zero_length_segments: int = 0


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


def _shift_segment_starts_after_switch(
    chop: pd.DataFrame,
    trend: pd.DataFrame,
    *,
    cooldown_bars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Delay segment starts that flip strategy too soon after the prior segment ended.

    Segment-level replay cannot express live ``MultiLegConcurrencyGate`` cooldown
    exactly (live allows immediate takeover; cooldown only blocks the deactivated
    strategy while the other holds — already covered by per-symbol mutex here).
    This shift approximates switch friction for offline account replay and matches
    ``scripts/sim_chop_grid_cooldown_ablation.py``.
    """
    meta = {"switches": 0, "delayed": 0, "blocked": 0}
    if cooldown_bars <= 0 or chop.empty or trend.empty:
        return chop, trend, meta

    dup_chop = chop.copy()
    dup_trend = trend.copy()
    cooldown = timedelta(minutes=int(cooldown_bars) * BAR_MINUTES)

    for sym in sorted(set(dup_chop["symbol"].unique()) & set(dup_trend["symbol"].unique())):
        timeline: List[tuple] = []
        for _, row in dup_chop[dup_chop["symbol"] == sym].sort_values("start").iterrows():
            timeline.append(
                (row["start"], row["end"], "chop_grid", str(row["segment_id"]))
            )
        for _, row in dup_trend[dup_trend["symbol"] == sym].sort_values("start").iterrows():
            timeline.append(
                (row["start"], row["end"], "trend_scalp", str(row["segment_id"]))
            )
        timeline.sort(key=lambda x: x[0])

        prev_end: Optional[pd.Timestamp] = None
        prev_strategy: Optional[StrategyName] = None
        for start, end, strat, seg_id in timeline:
            if (
                prev_end is not None
                and prev_strategy is not None
                and prev_strategy != strat
            ):
                meta["switches"] += 1
                earliest = prev_end + cooldown
                if start < earliest:
                    meta["delayed"] += 1
                    new_start = earliest
                    if new_start >= end:
                        meta["blocked"] += 1
                        if strat == "chop_grid":
                            dup_chop.loc[dup_chop["segment_id"] == seg_id, "start"] = end
                        else:
                            dup_trend.loc[dup_trend["segment_id"] == seg_id, "start"] = end
                    elif strat == "chop_grid":
                        dup_chop.loc[dup_chop["segment_id"] == seg_id, "start"] = new_start
                    else:
                        dup_trend.loc[dup_trend["segment_id"] == seg_id, "start"] = new_start
            prev_end = end
            prev_strategy = strat

    dup_chop = dup_chop[dup_chop["start"] < dup_chop["end"]].copy()
    dup_trend = dup_trend[dup_trend["start"] < dup_trend["end"]].copy()
    return dup_chop, dup_trend, meta


def apply_multileg_segment_gates(
    chop_segments: pd.DataFrame,
    trend_segments: pd.DataFrame,
    *,
    max_concurrent_multi_leg_symbols: int = 0,
    strategy_switch_cooldown_bars: int = 0,
    strategy_priority: Tuple[StrategyName, ...] = ("chop_grid", "trend_scalp"),
) -> MultilegGateStats:
    """Replay segment starts/ends on one account timeline.

    - ``max_concurrent_multi_leg_symbols``: shared cap on distinct symbols
      across both chop_grid + trend_scalp (0 = unlimited).
    - Per-symbol mutex: if a symbol is owned by one strategy, the other cannot
      start a new segment on that symbol until the owner segment ends.
    - ``strategy_switch_cooldown_bars``: after chop↔trend switch on a symbol,
      delay the incoming segment's ``start`` by N×2h (segment-level anti-thrash).
    - Same-timestamp tie-break: ends before starts; starts follow ``strategy_priority``.
    """
    stats = MultilegGateStats()
    chop = _prep_segments(chop_segments)
    trend = _prep_segments(trend_segments)
    if chop.empty and trend.empty:
        return stats

    if strategy_switch_cooldown_bars > 0:
        chop, trend, cd_meta = _shift_segment_starts_after_switch(
            chop, trend, cooldown_bars=strategy_switch_cooldown_bars
        )
        stats.cooldown_switches = int(cd_meta["switches"])
        stats.cooldown_delayed_starts = int(cd_meta["delayed"])
        stats.cooldown_zero_length_segments = int(cd_meta["blocked"])

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


def _side_sign(row: pd.Series) -> float:
    side = str(row.get("side", "") or row.get("direction", "") or "LONG").upper()
    if side in ("SHORT", "SELL", "DOWN"):
        return -1.0
    return 1.0


def simulate_account_with_constitution(
    trades: pd.DataFrame,
    *,
    equity: float,
    unit_notional: float,
    unit_by_strategy: Optional[Dict[str, float]] = None,
    max_gross_notional_pct: Optional[float] = None,
    max_net_notional_pct: Optional[float] = None,
    max_symbol_gross_notional_pct: Optional[float] = None,
    max_symbol_net_notional_pct: Optional[float] = None,
    max_gross_leverage: Optional[float] = None,
    daily_loss_limit_pct: Optional[float] = None,
    max_drawdown_pct: Optional[float] = None,
    fuse_mode: FuseMode = "hard",
    fuse_soft_dd_ratio: float = 0.5,
    fuse_derate_factor: float = 0.5,
) -> Dict[str, float]:
    """Replay with constitution risk limits enforced.

    - *max_gross_notional_pct* / *max_net_notional_pct*: portfolio caps vs equity anchor.
    - *max_symbol_*: per-symbol gross/net caps (``max_symbol_net_notional_pct`` = net cap).
    - *max_gross_leverage*: ultimate gross/equity ceiling (usually 3.0 on Binance).
    - *daily_loss_limit_pct*: reject new risk for rest of UTC day after daily loss.
    - *max_drawdown_pct*: global halt after peak-to-trough DD exceeds threshold.
    - *fuse_mode* ``tier_derate``: scale leg size after soft DD; ``tier_daily_scaled``:
      scale daily loss budget with gross exposure (position-aware).
    """
    events: List[tuple] = []
    for _, r in trades.iterrows():
        sym = str(r.get("symbol", "") or "")
        strat = str(r.get("strategy", "") or "")
        if unit_by_strategy and strat in unit_by_strategy:
            unit = float(unit_by_strategy[strat])
        else:
            unit = float(unit_notional)
        side_sign = _side_sign(r)
        events.append(
            (
                pd.Timestamp(r["entry_time"]),
                "open",
                sym,
                r.get("segment_id", ""),
                0.0,
                unit,
                side_sign,
            )
        )
        events.append(
            (
                pd.Timestamp(r["exit_time"]),
                "close",
                sym,
                r.get("segment_id", ""),
                float(r["pnl_pct"]) * unit,
                unit,
                side_sign,
            )
        )
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1))

    gross = 0.0
    net = 0.0
    peak_gross = 0.0
    sym_gross: Dict[str, float] = {}
    sym_net: Dict[str, float] = {}
    peak_sym = 0.0
    cum_pnl = 0.0
    peak_eq = float(equity)
    max_dd = 0.0
    max_dd_peak_pct = 0.0
    daily_pnl = 0.0
    current_day = None
    halted = False
    halted_day = None
    n_rejected = 0
    n_accepted = 0
    n_derated = 0
    n_reject_halted = 0
    n_reject_max_dd = 0
    n_reject_daily = 0
    n_reject_gross = 0
    n_reject_net = 0
    n_reject_leverage = 0
    halted_reason = ""

    def _current_equity() -> float:
        return equity + cum_pnl

    def _drawdown_frac(current_eq: float) -> float:
        if peak_eq <= 0:
            return 0.0
        return max(0.0, (peak_eq - current_eq) / peak_eq)

    def _effective_unit(base_unit: float, current_eq: float) -> float:
        if fuse_mode != "tier_derate" or max_drawdown_pct is None:
            return base_unit
        dd = _drawdown_frac(current_eq)
        soft = float(max_drawdown_pct) * float(fuse_soft_dd_ratio)
        if dd >= soft:
            return base_unit * float(fuse_derate_factor)
        return base_unit

    for ts, kind, _sym, _seg_id, pnl, unit, side_sign in events:
        day_key = ts.strftime("%Y-%m-%d")
        if current_day is not None and day_key != current_day:
            daily_pnl = 0.0
        current_day = day_key

        if kind == "close":
            gross = max(0.0, gross - unit)
            sym_gross[_sym] = max(0.0, sym_gross.get(_sym, 0.0) - unit)
            net -= side_sign * unit
            sym_net[_sym] = sym_net.get(_sym, 0.0) - side_sign * unit
            cum_pnl += pnl
            daily_pnl += pnl
            peak_eq = max(peak_eq, _current_equity())
            max_dd = min(max_dd, _current_equity() - peak_eq)
            if peak_eq > 0:
                dd_peak = (_current_equity() - peak_eq) / peak_eq
                max_dd_peak_pct = min(max_dd_peak_pct, dd_peak)
            continue

        if halted:
            n_rejected += 1
            n_reject_halted += 1
            continue

        current_eq = _current_equity()
        open_unit = _effective_unit(unit, current_eq)
        if open_unit < unit:
            n_derated += 1

        projected_gross = gross + open_unit
        projected_net = net + side_sign * open_unit
        projected_sym_gross = sym_gross.get(_sym, 0.0) + open_unit
        projected_sym_net = sym_net.get(_sym, 0.0) + side_sign * open_unit

        if max_drawdown_pct is not None and _drawdown_frac(current_eq) >= float(
            max_drawdown_pct
        ):
            halted = True
            halted_day = day_key
            halted_reason = f"max_drawdown {max_drawdown_pct * 100:.0f}% at {ts}"
            n_rejected += 1
            n_reject_max_dd += 1
            continue

        if daily_loss_limit_pct is not None:
            exposure_scale = 1.0
            if fuse_mode == "tier_daily_scaled":
                exposure_scale = max(projected_gross / float(equity), 0.05)
            daily_limit = float(equity) * float(daily_loss_limit_pct) * exposure_scale
            if daily_pnl <= -daily_limit:
                n_rejected += 1
                n_reject_daily += 1
                continue

        if max_gross_leverage is not None and projected_gross > current_eq * max_gross_leverage:
            n_rejected += 1
            n_reject_leverage += 1
            continue

        if (
            max_gross_notional_pct is not None
            and projected_gross > equity * max_gross_notional_pct
        ):
            n_rejected += 1
            n_reject_gross += 1
            continue

        if (
            max_net_notional_pct is not None
            and abs(projected_net) > equity * max_net_notional_pct
        ):
            n_rejected += 1
            n_reject_net += 1
            continue

        if (
            max_symbol_gross_notional_pct is not None
            and projected_sym_gross > equity * max_symbol_gross_notional_pct
        ):
            n_rejected += 1
            n_reject_gross += 1
            continue

        if (
            max_symbol_net_notional_pct is not None
            and abs(projected_sym_net) > equity * max_symbol_net_notional_pct
        ):
            n_rejected += 1
            n_reject_net += 1
            continue

        gross = projected_gross
        net = projected_net
        sym_gross[_sym] = projected_sym_gross
        sym_net[_sym] = projected_sym_net
        n_accepted += 1
        peak_gross = max(peak_gross, gross)
        if sym_gross:
            peak_sym = max(peak_sym, max(sym_gross.values()))

    return {
        "ret_pct": cum_pnl / equity * 100.0,
        "pnl_usd": cum_pnl,
        "peak_gross_pct": peak_gross / equity * 100.0,
        "peak_sym_pct": peak_sym / equity * 100.0,
        "max_dd_pct": max_dd / equity * 100.0,
        "max_dd_peak_pct": max_dd_peak_pct * 100.0,
        "n_trades": float(n_accepted),
        "n_rejected": float(n_rejected),
        "n_derated": float(n_derated),
        "n_reject_halted": float(n_reject_halted),
        "n_reject_max_dd": float(n_reject_max_dd),
        "n_reject_daily": float(n_reject_daily),
        "n_reject_gross": float(n_reject_gross),
        "n_reject_net": float(n_reject_net),
        "n_reject_leverage": float(n_reject_leverage),
        "halted": halted,
        "halted_reason": halted_reason,
        "halted_day": str(halted_day) if halted_day else "",
        "fuse_mode": fuse_mode,
    }
