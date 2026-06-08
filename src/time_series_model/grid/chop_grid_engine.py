"""Stateful inventory model for chop-grid research and dry-run execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GridEngineConfig:
    box_window: int = 120
    entry_chop_min: float = 0.40
    exit_chop_below: float = 0.25
    min_segment_bars: int = 6
    max_segment_bars: int = 120
    grid_atr_mult: float = 0.50
    grid_min_pct: float = 0.004
    max_levels_per_side: int = 3
    # Take-profit distance = spacing * tp_spacing_mult. 1.0 keeps the legacy
    # "TP at one grid step" behavior; >1.0 decouples exit target from entry
    # density (wider TP while keeping the grid layout tight).
    tp_spacing_mult: float = 1.0
    fee_bps: float = 4.0
    maker_fee_bps: float | None = None
    taker_fee_bps: float | None = None
    forced_exit_slippage_bps: float = 0.0
    funding_cost_bps_per_8h: float = 0.0
    max_loss_per_grid: float | None = 0.03
    max_open_levels_total: int | None = 6
    same_bar_entry_exit: bool = False
    # None = unlimited replenishes per level per segment (legacy backtest).
    # N = max post-TP replenishes; total fills per level <= 1 + N (N=0 = live one-shot).
    max_replenish_per_level_per_segment: int | None = None


@dataclass(frozen=True)
class GridTrade:
    symbol: str
    regime: str
    segment_id: str
    side: str
    level: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_pct: float
    r_equiv: float
    pnl_per_capital: float
    r_equiv_per_capital: float
    spacing_pct: float
    spacing_atr: float
    gross_pnl_pct: float
    fee_bps_charged: float
    slippage_bps_charged: float
    funding_bps_charged: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GridSegmentResult:
    trades: List[GridTrade]
    summary: dict
    equity_path: List[Tuple[pd.Timestamp, float]]


def pnl_long(entry: float, exit_px: float, fee: float) -> float:
    return (exit_px - entry) / entry - 2.0 * fee


def pnl_short(entry: float, exit_px: float, fee: float) -> float:
    return (entry - exit_px) / entry - 2.0 * fee


def hysteresis_segments(
    entry_mask: pd.Series,
    hold_mask: pd.Series,
    *,
    min_len: int,
    max_len: int,
) -> List[Tuple[int, int]]:
    """Build segments that enter on entry_mask and hold until hold_mask fails."""
    entry = entry_mask.fillna(False).to_numpy(dtype=bool)
    hold = hold_mask.fillna(False).to_numpy(dtype=bool)
    segs: List[Tuple[int, int]] = []
    i = 0
    n = len(entry)
    while i < n:
        if not entry[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and hold[i + 1] and (i + 1 - start) < max_len:
            i += 1
        end = i
        if end - start + 1 >= min_len:
            segs.append((start, end))
        i += 1
    return segs


class ChopGridEngine:
    """Fixed-level neutral grid with conservative same-bar fill handling."""

    def __init__(self, cfg: GridEngineConfig):
        self.cfg = cfg

    @property
    def capital_units(self) -> int:
        return max(1, 2 * int(self.cfg.max_levels_per_side))

    def simulate_segment(
        self,
        seg: pd.DataFrame,
        *,
        symbol: str,
        regime: str,
        segment_id: str,
        anchor_close: float | None = None,
        anchor_atr: float | None = None,
        regime_chop_col: str | None = None,
        account_risk_tracker: Any = None,
        unit_notional_usdt: float = 0.0,
    ) -> GridSegmentResult:
        if seg.empty:
            return GridSegmentResult(
                [], {"status": "empty", "segment_id": segment_id}, []
            )

        center = (
            float(anchor_close)
            if anchor_close is not None
            else float(seg["close"].iloc[0])
        )
        atr = (
            float(anchor_atr) if anchor_atr is not None else float(seg["atr14"].iloc[0])
        )
        if not np.isfinite(center + atr) or center <= 0 or atr <= 0:
            return GridSegmentResult(
                [], {"status": "invalid", "segment_id": segment_id}, []
            )

        spacing = max(self.cfg.grid_atr_mult * atr, self.cfg.grid_min_pct * center)
        if spacing <= 0:
            return GridSegmentResult(
                [], {"status": "invalid", "segment_id": segment_id}, []
            )
        tp_distance = spacing * float(self.cfg.tp_spacing_mult or 1.0)

        maker_fee_bps = (
            float(self.cfg.maker_fee_bps)
            if self.cfg.maker_fee_bps is not None
            else float(self.cfg.fee_bps)
        )
        taker_fee_bps = (
            float(self.cfg.taker_fee_bps)
            if self.cfg.taker_fee_bps is not None
            else float(self.cfg.fee_bps)
        )
        maker_fee = maker_fee_bps / 10000.0
        long_levels = [
            center - spacing * k for k in range(1, self.cfg.max_levels_per_side + 1)
        ]
        short_levels = [
            center + spacing * k for k in range(1, self.cfg.max_levels_per_side + 1)
        ]

        open_longs: Dict[int, Tuple[float, pd.Timestamp, int]] = {}
        open_shorts: Dict[int, Tuple[float, pd.Timestamp, int]] = {}
        completed_long: Dict[int, int] = {}
        completed_short: Dict[int, int] = {}
        max_replenish = self.cfg.max_replenish_per_level_per_segment
        trades: List[GridTrade] = []
        equity_path: List[Tuple[pd.Timestamp, float]] = []
        max_open = 0
        risk_exit = False
        replenish_tp = 0

        leg_notional = float(max(0.0, unit_notional_usdt))

        def _may_enter_level(level_i: int, completed: Dict[int, int]) -> bool:
            if max_replenish is None:
                return True
            return int(completed.get(level_i, 0)) <= int(max_replenish)

        def _record(
            *,
            side: str,
            level: int,
            entry_price: float,
            entry_time: pd.Timestamp,
            exit_price: float,
            exit_time: pd.Timestamp,
            exit_reason: str,
        ) -> None:
            if account_risk_tracker is not None and leg_notional > 0:
                account_risk_tracker.on_close(leg_notional)
            gross_pnl_pct = (
                (exit_price - entry_price) / entry_price
                if side == "LONG"
                else (entry_price - exit_price) / entry_price
            )
            exit_fee_bps = maker_fee_bps
            slippage_bps = 0.0
            if exit_reason in {"regime_exit", "risk_exit"}:
                exit_fee_bps = taker_fee_bps
                slippage_bps = float(self.cfg.forced_exit_slippage_bps or 0.0)
            hold_hours = max(
                0.0,
                (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).total_seconds()
                / 3600.0,
            )
            funding_bps = (
                hold_hours / 8.0 * float(self.cfg.funding_cost_bps_per_8h or 0.0)
            )
            fee_bps_charged = maker_fee_bps + exit_fee_bps
            total_cost_pct = (fee_bps_charged + slippage_bps + funding_bps) / 10000.0
            pnl_pct = gross_pnl_pct - total_cost_pct
            r_equiv = pnl_pct / (spacing / center)
            trades.append(
                GridTrade(
                    symbol=symbol,
                    regime=regime,
                    segment_id=segment_id,
                    side=side,
                    level=level,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    r_equiv=r_equiv,
                    pnl_per_capital=pnl_pct / self.capital_units,
                    r_equiv_per_capital=r_equiv / self.capital_units,
                    spacing_pct=spacing / center,
                    spacing_atr=spacing / atr,
                    gross_pnl_pct=gross_pnl_pct,
                    fee_bps_charged=fee_bps_charged,
                    slippage_bps_charged=slippage_bps,
                    funding_bps_charged=funding_bps,
                )
            )

        for bar_i, (ts, row) in enumerate(seg.iterrows()):
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            # Target exits before new fills; optionally disallow same-bar entry+exit.
            for level_i, (entry, entry_ts, fill_bar) in list(open_longs.items()):
                target = entry + tp_distance
                can_exit = self.cfg.same_bar_entry_exit or bar_i > fill_bar
                if can_exit and high >= target:
                    lvl = level_i + 1
                    if completed_long.get(level_i, 0) > 0:
                        replenish_tp += 1
                    _record(
                        side="LONG",
                        level=lvl,
                        entry_price=entry,
                        entry_time=entry_ts,
                        exit_price=target,
                        exit_time=ts,
                        exit_reason="grid_tp",
                    )
                    completed_long[level_i] = completed_long.get(level_i, 0) + 1
                    del open_longs[level_i]
            for level_i, (entry, entry_ts, fill_bar) in list(open_shorts.items()):
                target = entry - tp_distance
                can_exit = self.cfg.same_bar_entry_exit or bar_i > fill_bar
                if can_exit and low <= target:
                    lvl = level_i + 1
                    if completed_short.get(level_i, 0) > 0:
                        replenish_tp += 1
                    _record(
                        side="SHORT",
                        level=lvl,
                        entry_price=entry,
                        entry_time=entry_ts,
                        exit_price=target,
                        exit_time=ts,
                        exit_reason="grid_tp",
                    )
                    completed_short[level_i] = completed_short.get(level_i, 0) + 1
                    del open_shorts[level_i]

            for level_i, px in enumerate(long_levels):
                if (
                    level_i not in open_longs
                    and low <= px
                    and _may_enter_level(level_i, completed_long)
                ):
                    if account_risk_tracker is not None and leg_notional > 0:
                        ok, _ = account_risk_tracker.allow_open(leg_notional)
                        if not ok:
                            continue
                        account_risk_tracker.on_open(leg_notional)
                    open_longs[level_i] = (px, ts, bar_i)
            for level_i, px in enumerate(short_levels):
                if (
                    level_i not in open_shorts
                    and high >= px
                    and _may_enter_level(level_i, completed_short)
                ):
                    if account_risk_tracker is not None and leg_notional > 0:
                        ok, _ = account_risk_tracker.allow_open(leg_notional)
                        if not ok:
                            continue
                        account_risk_tracker.on_open(leg_notional)
                    open_shorts[level_i] = (px, ts, bar_i)

            realized = sum(t.pnl_pct for t in trades)
            mtm = realized
            for entry, _, _ in open_longs.values():
                mtm += (close - entry) / entry - maker_fee
            for entry, _, _ in open_shorts.values():
                mtm += (entry - close) / entry - maker_fee
            pnl_per_capital = mtm / self.capital_units
            equity_path.append((ts, pnl_per_capital))

            open_total = len(open_longs) + len(open_shorts)
            max_open = max(max_open, open_total)
            if (
                self.cfg.max_open_levels_total is not None
                and open_total > self.cfg.max_open_levels_total
            ):
                risk_exit = True
            if self.cfg.max_loss_per_grid is not None and pnl_per_capital <= -abs(
                self.cfg.max_loss_per_grid
            ):
                risk_exit = True
            if risk_exit:
                break

        exit_ts = equity_path[-1][0] if equity_path else seg.index[-1]
        exit_close = (
            float(seg.loc[exit_ts, "close"])
            if exit_ts in seg.index
            else float(seg["close"].iloc[-1])
        )
        forced_reason = "risk_exit" if risk_exit else "regime_exit"
        forced = len(open_longs) + len(open_shorts)
        for level_i, (entry, entry_ts, _) in list(open_longs.items()):
            _record(
                side="LONG",
                level=level_i + 1,
                entry_price=entry,
                entry_time=entry_ts,
                exit_price=exit_close,
                exit_time=exit_ts,
                exit_reason=forced_reason,
            )
        for level_i, (entry, entry_ts, _) in list(open_shorts.items()):
            _record(
                side="SHORT",
                level=level_i + 1,
                entry_price=entry,
                entry_time=entry_ts,
                exit_price=exit_close,
                exit_time=exit_ts,
                exit_reason=forced_reason,
            )

        pnl_values = np.asarray([v for _, v in equity_path], dtype=float)
        max_drawdown = 0.0
        if len(pnl_values):
            max_drawdown = float((pnl_values - np.maximum.accumulate(pnl_values)).min())

        _chop_col = regime_chop_col or "semantic_chop"
        if _chop_col not in seg.columns:
            _chop_col = "semantic_chop"
        _chop_series = pd.to_numeric(seg[_chop_col], errors="coerce")
        segment_high = float(pd.to_numeric(seg["high"], errors="coerce").max())
        segment_low = float(pd.to_numeric(seg["low"], errors="coerce").min())
        segment_range_pct = (
            (segment_high - segment_low) / center
            if np.isfinite(segment_high + segment_low) and center > 0
            else 0.0
        )
        close_std_pct = float(
            pd.to_numeric(seg["close"], errors="coerce").std(ddof=0) / center
        )
        per_side_span_pct = spacing * int(self.cfg.max_levels_per_side) / center
        full_span_pct = 2.0 * per_side_span_pct
        summary = {
            "status": "ok",
            "symbol": symbol,
            "regime": regime,
            "segment_id": segment_id,
            "start": seg.index[0],
            "end": exit_ts,
            "bars": len(equity_path),
            "entry_chop": float(_chop_series.iloc[0]),
            "median_chop": float(_chop_series.median()),
            "entry_box_prefilter": (
                bool(seg["box_prefilter"].iloc[0]) if "box_prefilter" in seg else False
            ),
            "center": center,
            "spacing_pct": spacing / center,
            "spacing_atr": spacing / atr,
            "max_levels_per_side": int(self.cfg.max_levels_per_side),
            "grid_per_side_span_pct": per_side_span_pct,
            "grid_full_span_pct": full_span_pct,
            "segment_range_pct": segment_range_pct,
            "close_std_pct": close_std_pct,
            "grid_full_span_to_range": full_span_pct / max(segment_range_pct, 1e-12),
            "grid_per_side_span_to_1std": per_side_span_pct / max(close_std_pct, 1e-12),
            "trades": len(trades),
            "grid_tp": sum(1 for t in trades if t.exit_reason == "grid_tp"),
            "forced_exits": forced,
            "risk_exits": sum(1 for t in trades if t.exit_reason == "risk_exit"),
            "max_open_levels": max_open,
            "pnl_per_capital": sum(t.pnl_per_capital for t in trades),
            "forced_exit_pnl": sum(
                t.pnl_per_capital
                for t in trades
                if t.exit_reason in {"regime_exit", "risk_exit"}
            ),
            "max_drawdown": max_drawdown,
            "max_replenish_per_level": max_replenish,
            "replenish_trades": replenish_tp,
        }
        return GridSegmentResult(trades, summary, equity_path)

    def simulate_segments(
        self,
        df: pd.DataFrame,
        segments: Iterable[Tuple[int, int]],
        *,
        symbol: str,
        regime: str,
    ) -> Tuple[List[dict], List[dict]]:
        trades: List[dict] = []
        summaries: List[dict] = []
        for seq, (s, e) in enumerate(segments, start=1):
            seg_id = f"{symbol}_{seq:04d}_{df.index[s].strftime('%Y%m%d%H')}"
            result = self.simulate_segment(
                df.iloc[s : e + 1],
                symbol=symbol,
                regime=regime,
                segment_id=seg_id,
            )
            trades.extend(t.to_dict() for t in result.trades)
            summaries.append(result.summary)
        return trades, summaries
