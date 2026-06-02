"""Independent trend_scalp backtest via Backtrader on 1min execution bars.

Reuses the **same** regime segment discovery as ``diagnose_dual_add_trend.py``
(entry/hold hysteresis on 2h signal) but re-implements inventory simulation
inside a Backtrader ``Strategy`` (no call to ``simulate_dual_add_segment``).

Compare against an existing diagnose run::

    python scripts/backtest_trend_scalp_backtrader.py \\
      --start 2025-10-01 --end 2026-03-31 \\
      --compare-dir results/trend_scalp/experiments/segment_validate_20260602/recent_6m_oos \\
      --out-dir results/trend_scalp/experiments/backtrader_crosscheck_recent_6m
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import backtrader as bt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    _hysteresis_segments,
    build_features,
    regime_chop_series,
    resolve_optional_repo_path,
)
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from scripts.diagnose_dual_add_trend import (  # noqa: E402
    DualAddConfig,
    _add_trend_features,
    _effective_max_loser_hold_bars,
    _entry_price_with_slippage,
    _exit_price_with_slippage,
    _exposure_units,
    _load_dual_add_defaults,
    _max_drawdown,
    _position_pnl,
    summarize,
)
from scripts.pipeline.multileg_prefilter_rules import (
    apply_prefilter_rules,
)  # noqa: E402
from src.live_data_stream.constitution_config import (  # noqa: E402
    load_multi_leg_backtest_risk_context,
)
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    slice_execution_window,
    timeframe_to_timedelta,
)


@dataclass
class SegmentSpec:
    symbol: str
    segment_id: str
    direction: str
    frozen_center: float
    frozen_atr: float
    exec_df: pd.DataFrame


class SegmentPandasData(bt.feeds.PandasData):
    lines = ("trend_direction_code",)
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("trend_direction_code", "trend_direction_code"),
    )


class DualAddTrendBacktraderStrategy(bt.Strategy):
    """Backtrader-native rewrite of dual-add segment inventory rules."""

    params = dict(
        cfg=None,
        symbol="",
        segment_id="",
        direction="UP",
        frozen_center=0.0,
        frozen_atr=0.0,
    )

    def __init__(self) -> None:
        self.cfg: DualAddConfig = self.p.cfg
        self.bar_i = -1
        self.open_legs: List[dict] = []
        self.trades: List[dict] = []
        self.pnl_path: List[float] = []
        self.stop_reason = "regime_exit"
        self.fee = self.cfg.fee_bps / 10000.0
        self.capital_units = max(2, self.cfg.max_gross_exposure)
        self.center = float(self.p.frozen_center)
        self.atr = float(self.p.frozen_atr)
        self.step = self.cfg.step_atr_mult * self.atr
        self.tp = max(
            self.cfg.tp_abs,
            self.cfg.tp_atr_mult * self.atr,
            self.cfg.tp_pct * self.center,
        )
        self.touch_buffer = (
            max(float(self.cfg.intrabar_touch_buffer_bps), 0.0) / 10000.0
        )
        self.last_add_long = self.center
        self.last_add_short = self.center
        self.add_long_count = 0
        self.add_short_count = 0
        self.max_gross_units = 0
        self.max_abs_net_units = 0
        self.trend_side = "LONG" if self.p.direction == "UP" else "SHORT"
        self.last_trend_side = self.trend_side
        self.trend_flips = 0
        self.flip_forced = 0
        self.last_flat_bar = -1
        self.block_reseed_after_flip = False

    def _hit_up(self, level: float, high: float) -> bool:
        return high >= level * (1.0 + self.touch_buffer)

    def _hit_down(self, level: float, low: float) -> bool:
        return low <= level * (1.0 - self.touch_buffer)

    def _record(self, pos: dict, exit_px: float, reason: str) -> None:
        slippage_bps = (
            0.0
            if reason == "tp"
            else max(float(self.cfg.market_exit_slippage_bps), 0.0)
        )
        filled_exit_px = _exit_price_with_slippage(
            str(pos["side"]), float(exit_px), slippage_bps
        )
        pnl_pct = _position_pnl(
            pos["side"], float(pos["entry"]), filled_exit_px, self.fee
        )
        self.trades.append(
            {
                "symbol": self.p.symbol,
                "segment_id": self.p.segment_id,
                "direction": self.p.direction,
                "side": pos["side"],
                "seq": pos["seq"],
                "entry_time": pos["entry_time"],
                "exit_time": self.data.datetime.datetime(0),
                "entry_price": pos["entry"],
                "exit_price": filled_exit_px,
                "exit_reason": reason,
                "pnl_pct": pnl_pct,
                "pnl_per_capital": pnl_pct / self.capital_units,
            }
        )

    def _seed_positions(self, px: float, bar_i: int) -> None:
        sides = ["LONG", "SHORT"] if self.cfg.initial_hedge else [self.trend_side]
        for side in sides:
            entry_px = _entry_price_with_slippage(side, px, self.cfg.entry_slippage_bps)
            self.open_legs.append(
                {
                    "side": side,
                    "entry": entry_px,
                    "entry_time": self.data.datetime.datetime(0),
                    "entry_bar": bar_i,
                    "seq": 0,
                }
            )
        self.last_add_long = px
        self.last_add_short = px

    def _can_add(self, side: str) -> bool:
        hypothetical = self.open_legs + [{"side": side}]
        _, _, net_units = _exposure_units(hypothetical)
        return len(hypothetical) <= self.cfg.max_gross_exposure and abs(net_units) <= (
            self.cfg.max_net_exposure
        )

    def _enforce_net_cap(self, px: float) -> None:
        _, _, net_units = _exposure_units(self.open_legs)
        while abs(net_units) > self.cfg.max_net_exposure and self.open_legs:
            overloaded_side = "LONG" if net_units > 0 else "SHORT"
            candidates = [p for p in self.open_legs if p["side"] == overloaded_side]
            if not candidates:
                break
            pos = min(
                candidates,
                key=lambda p: _position_pnl(
                    str(p["side"]), float(p["entry"]), px, self.fee
                ),
            )
            self._record(pos, px, "net_cap")
            self.open_legs.remove(pos)
            _, _, net_units = _exposure_units(self.open_legs)

    def _close_offside_positions(self, px: float, side: str) -> int:
        if self.cfg.flip_action == "keep":
            return 0
        closed = 0
        for pos in list(self.open_legs):
            if pos["side"] == side:
                continue
            if (
                self.cfg.flip_action == "close_offside_adds"
                and int(pos.get("seq", 0)) == 0
            ):
                continue
            self._record(pos, px, "trend_flip")
            self.open_legs.remove(pos)
            closed += 1
        if not self.open_legs:
            self.last_flat_bar = self.bar_i
        return closed

    def _open_pnl_per_capital(self, px: float) -> float:
        return (
            sum(
                _position_pnl(p["side"], float(p["entry"]), px, self.fee)
                for p in self.open_legs
            )
            / self.capital_units
        )

    def _basket_target_per_capital(self, px: float) -> float:
        fee_buffer = 2.0 * self.fee * px
        slippage_buffer = (
            max(float(self.cfg.market_exit_slippage_bps), 0.0) / 10000.0 * px
        )
        target = max(
            self.cfg.tp_abs, self.cfg.tp_atr_mult * self.atr, self.cfg.tp_pct * px
        )
        return ((fee_buffer + slippage_buffer + target) / px) / self.capital_units

    def _close_all_open(self, px: float, reason: str) -> None:
        for pos in list(self.open_legs):
            self._record(pos, px, reason)
            self.open_legs.remove(pos)
        self.last_flat_bar = self.bar_i

    def _realized_pnl(self) -> float:
        return sum(t["pnl_pct"] for t in self.trades)

    def next(self) -> None:
        self.bar_i += 1
        high = float(self.data.high[0])
        low = float(self.data.low[0])
        close = float(self.data.close[0])
        actionable = self.bar_i > 0

        if self.bar_i == 0:
            self._seed_positions(close, 0)
            self.max_gross_units = len(self.open_legs)
            return

        if self.cfg.add_mode == "trend" and actionable:
            prev_dir = float(self.data.trend_direction_code[-1])
            self.trend_side = "LONG" if prev_dir > 0 else "SHORT"

        seeded_this_bar = False
        if (
            actionable
            and not self.open_legs
            and self.bar_i > self.last_flat_bar
            and not self.block_reseed_after_flip
        ):
            self._seed_positions(close, self.bar_i)
            seeded_this_bar = True

        if actionable and not seeded_this_bar:
            closed_basket = False
            if self.cfg.take_profit_mode == "basket":
                if self.open_legs and self._open_pnl_per_capital(
                    close
                ) >= self._basket_target_per_capital(close):
                    self._close_all_open(close, "basket_tp")
                    closed_basket = True
            else:
                for pos in list(self.open_legs):
                    entry = float(pos["entry"])
                    fee_buffer = 2.0 * self.fee * entry
                    if pos["side"] == "LONG" and self._hit_up(
                        entry + self.tp + fee_buffer, high
                    ):
                        self._record(pos, entry + self.tp + fee_buffer, "tp")
                        self.open_legs.remove(pos)
                    elif pos["side"] == "SHORT" and self._hit_down(
                        entry - self.tp - fee_buffer, low
                    ):
                        self._record(pos, entry - self.tp - fee_buffer, "tp")
                        self.open_legs.remove(pos)
            self._enforce_net_cap(close)
            if closed_basket:
                _, _, net_units = _exposure_units(self.open_legs)
                self.max_gross_units = max(self.max_gross_units, len(self.open_legs))
                self.max_abs_net_units = max(self.max_abs_net_units, abs(net_units))
                self.pnl_path.append(self._realized_pnl() / self.capital_units)
                return

            if self.cfg.add_mode == "trend" and self.trend_side != self.last_trend_side:
                self.trend_flips += 1
                self.flip_forced += self._close_offside_positions(
                    close, self.trend_side
                )
                self.last_trend_side = self.trend_side
                if not self.cfg.reseed_on_flip:
                    self.block_reseed_after_flip = True
                self._enforce_net_cap(close)

            for pos in list(self.open_legs):
                held_bars = self.bar_i - int(pos.get("entry_bar", 0))
                if held_bars < self.cfg.max_loser_hold_bars:
                    continue
                if _position_pnl(pos["side"], float(pos["entry"]), close, self.fee) < 0:
                    self._record(pos, close, "loser_timeout")
                    self.open_legs.remove(pos)
            self._enforce_net_cap(close)

        if actionable:
            while (
                self.cfg.add_mode in {"both", "trend"}
                and (self.cfg.add_mode == "both" or self.trend_side == "LONG")
                and self._hit_up(self.last_add_long + self.step, high)
                and self.add_long_count < self.cfg.max_adds_per_side
                and self._can_add("LONG")
            ):
                self.last_add_long += self.step
                self.add_long_count += 1
                self.open_legs.append(
                    {
                        "side": "LONG",
                        "entry": _entry_price_with_slippage(
                            "LONG", self.last_add_long, self.cfg.add_slippage_bps
                        ),
                        "entry_time": self.data.datetime.datetime(0),
                        "entry_bar": self.bar_i,
                        "seq": self.add_long_count,
                    }
                )
            while (
                self.cfg.add_mode in {"both", "trend"}
                and (self.cfg.add_mode == "both" or self.trend_side == "SHORT")
                and self._hit_down(self.last_add_short - self.step, low)
                and self.add_short_count < self.cfg.max_adds_per_side
                and self._can_add("SHORT")
            ):
                self.last_add_short -= self.step
                self.add_short_count += 1
                self.open_legs.append(
                    {
                        "side": "SHORT",
                        "entry": _entry_price_with_slippage(
                            "SHORT", self.last_add_short, self.cfg.add_slippage_bps
                        ),
                        "entry_time": self.data.datetime.datetime(0),
                        "entry_bar": self.bar_i,
                        "seq": self.add_short_count,
                    }
                )
            self._enforce_net_cap(close)

        _, _, net_units = _exposure_units(self.open_legs)
        self.max_gross_units = max(self.max_gross_units, len(self.open_legs))
        self.max_abs_net_units = max(self.max_abs_net_units, abs(net_units))
        mtm = self._realized_pnl() + sum(
            _position_pnl(p["side"], float(p["entry"]), close, self.fee)
            for p in self.open_legs
        )
        self.pnl_path.append(mtm / self.capital_units)
        if (
            self.cfg.risk_stop_mode == "mtm"
            and mtm / self.capital_units <= -self.cfg.max_loss_per_segment
        ):
            self.stop_reason = "risk_stop"

    def stop(self) -> None:
        if not self.open_legs:
            return
        close = float(self.data.close[0])
        for pos in list(self.open_legs):
            self._record(pos, close, self.stop_reason)
            self.open_legs.remove(pos)


def _build_dual_add_config(args: argparse.Namespace, defaults: dict) -> DualAddConfig:
    eff_hold = _effective_max_loser_hold_bars(args)
    hparts = [
        x.strip() for x in str(args.trend_return_horizons).split(",") if x.strip()
    ]
    trend_horizons = tuple(int(x) for x in hparts) if hparts else (3, 5, 10)
    risk_tracker, unit_notional = load_multi_leg_backtest_risk_context(
        initial_capital=float(getattr(args, "initial_capital", 10_000.0) or 10_000.0)
    )
    return DualAddConfig(
        regime=args.regime,
        add_mode=args.add_mode,
        flip_action=args.flip_action,
        reseed_on_flip=bool(args.reseed_on_flip),
        chop_signal=args.chop_signal,
        chop_ts_window=args.chop_ts_window,
        chop_ts_min_periods=args.chop_ts_min_periods,
        compute_semantic_chop_ts_q=getattr(args, "compute_chop_ts_q", None),
        trend_return_horizons=trend_horizons,
        stability_min=float(args.stability_min),
        width_min=float(args.width_min),
        width_max=float(args.width_max),
        touches_min=int(args.touches_min),
        chop_min=args.chop_min,
        exit_chop_min=args.exit_chop_min,
        trend_min=args.trend_min,
        trend_exit_min=args.trend_exit_min,
        box_window=args.box_window,
        step_atr_mult=args.step_atr_mult,
        take_profit_mode=args.take_profit_mode,
        tp_atr_mult=args.tp_atr_mult,
        tp_abs=args.tp_abs,
        tp_pct=args.tp_pct,
        max_adds_per_side=args.max_adds_per_side,
        max_net_exposure=args.max_net_exposure,
        max_gross_exposure=args.max_gross_exposure,
        max_loser_hold_bars=eff_hold,
        max_segment_bars=args.max_segment_bars,
        min_segment_bars=args.min_segment_bars,
        fee_bps=args.fee_bps,
        market_exit_slippage_bps=args.market_exit_slippage_bps,
        intrabar_touch_buffer_bps=args.intrabar_touch_buffer_bps,
        entry_slippage_bps=args.entry_slippage_bps,
        add_slippage_bps=args.add_slippage_bps,
        max_loss_per_segment=args.max_loss_per_segment,
        risk_stop_mode=args.risk_stop_mode,
        initial_hedge=args.initial_hedge,
        prefilter_rules=tuple(
            x
            for x in (defaults.get("prefilter_rules", []) or [])
            if isinstance(x, dict)
        ),
        unit_notional_usdt=float(unit_notional),
        account_risk_tracker=risk_tracker,
    )


def _discover_segments(
    args: argparse.Namespace,
    cfg: DualAddConfig,
    grid_cfg: GridConfig,
) -> List[SegmentSpec]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=args.warmup_days)
    data_dir = Path(args.data_dir)
    exec_tf = args.execution_timeframe or args.timeframe
    sig_delta = (
        timeframe_to_timedelta(args.timeframe) if exec_tf != args.timeframe else None
    )
    specs: List[SegmentSpec] = []

    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            continue
        bars_signal = _resample_ohlcv(raw, args.timeframe)
        df = build_features(
            symbol, bars_signal, grid_cfg, bars_timeframe=args.timeframe
        )
        df = _add_trend_features(df, cfg.trend_return_horizons)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if exec_tf != args.timeframe:
            bars_exec = _resample_ohlcv(raw, exec_tf)
            df_exec = merge_signal_features_onto_execution_bars(
                bars_exec, df, signal_bar_delta=sig_delta
            )
        else:
            df_exec = df.copy()

        chop_s = regime_chop_series(df, grid_cfg)
        if cfg.regime == "trend":
            entry = (df["trend_confidence"] >= cfg.trend_min) & (
                chop_s <= cfg.exit_chop_min
            )
            hold = (df["trend_confidence"] >= cfg.trend_exit_min) & (
                chop_s <= cfg.chop_min
            )
        else:
            entry = chop_s >= cfg.chop_min
            hold = chop_s >= cfg.exit_chop_min
        rule_mask = apply_prefilter_rules(
            df,
            list(cfg.prefilter_rules),
            feature_aliases={
                "atr": "atr14",
                "bpc_semantic_chop": "semantic_chop",
                "bpc_semantic_chop_ts_q": "semantic_chop_ts_q",
            },
        )
        entry &= rule_mask
        hold &= rule_mask
        if args.exclude_box:
            entry &= ~df["box_prefilter"]
            hold &= ~df["box_prefilter"]
        segs = _hysteresis_segments(
            entry,
            hold,
            min_len=cfg.min_segment_bars,
            max_len=cfg.max_segment_bars,
        )
        print(f"{symbol}: segments={len(segs)}")
        for seq, (s, e) in enumerate(segs, start=1):
            seg_id = f"{symbol}_{seq:04d}_{df.index[s].strftime('%Y%m%d%H%M')}"
            direction = str(df["trend_direction"].iloc[s])
            if exec_tf != args.timeframe:
                seg_slice = slice_execution_window(df_exec, df.index, s, e, sig_delta)
            else:
                seg_slice = df.iloc[s : e + 1].copy()
            if seg_slice.empty:
                continue
            feed_df = seg_slice.copy()
            feed_df["trend_direction_code"] = np.where(
                feed_df["trend_direction"].astype(str) == "UP", 1.0, -1.0
            )
            specs.append(
                SegmentSpec(
                    symbol=symbol,
                    segment_id=seg_id,
                    direction=direction,
                    frozen_center=float(df.iloc[s]["close"]),
                    frozen_atr=float(df.iloc[s]["atr14"]),
                    exec_df=feed_df,
                )
            )
    return specs


def _run_segment_backtrader(
    spec: SegmentSpec, cfg: DualAddConfig
) -> Tuple[List[dict], dict]:
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_cash(100_000.0)
    data = SegmentPandasData(dataname=spec.exec_df)
    cerebro.adddata(data)
    cerebro.addstrategy(
        DualAddTrendBacktraderStrategy,
        cfg=cfg,
        symbol=spec.symbol,
        segment_id=spec.segment_id,
        direction=spec.direction,
        frozen_center=spec.frozen_center,
        frozen_atr=spec.frozen_atr,
    )
    results = cerebro.run()
    strat: DualAddTrendBacktraderStrategy = results[0]
    trades = strat.trades
    total = sum(t["pnl_per_capital"] for t in trades)
    summary = {
        "symbol": spec.symbol,
        "segment_id": spec.segment_id,
        "direction": spec.direction,
        "trades": len(trades),
        "tp": sum(1 for t in trades if t["exit_reason"] in {"tp", "basket_tp"}),
        "pnl_per_capital": total,
        "max_drawdown": _max_drawdown(strat.pnl_path),
        "max_gross_units": strat.max_gross_units,
        "max_abs_net_units": strat.max_abs_net_units,
        "risk_stop": int(strat.stop_reason == "risk_stop"),
    }
    return trades, summary


def _compare_runs(
    bt_trades: pd.DataFrame,
    bt_segments: pd.DataFrame,
    diagnose_dir: Path,
) -> pd.DataFrame:
    ref_trades = pd.read_csv(diagnose_dir / "dual_add_trades.csv")
    ref_segments = pd.read_csv(diagnose_dir / "dual_add_segments.csv")
    ref_summary = pd.read_csv(diagnose_dir / "summary.csv").iloc[0]

    bt_sum = summarize(bt_trades, bt_segments).iloc[0]
    rows = [
        {
            "metric": "return_pct",
            "diagnose": float(ref_summary["return_pct"]),
            "backtrader": float(bt_sum["return_pct"]),
            "diff_pct": float(bt_sum["return_pct"] - ref_summary["return_pct"]),
            "diff_rel": float(
                (bt_sum["return_pct"] - ref_summary["return_pct"])
                / max(abs(ref_summary["return_pct"]), 1e-9)
            ),
        },
        {
            "metric": "trades",
            "diagnose": float(ref_summary["trades"]),
            "backtrader": float(bt_sum["trades"]),
            "diff_pct": float(bt_sum["trades"] - ref_summary["trades"]),
            "diff_rel": float(
                (bt_sum["trades"] - ref_summary["trades"])
                / max(ref_summary["trades"], 1)
            ),
        },
        {
            "metric": "segments",
            "diagnose": float(ref_summary["segments"]),
            "backtrader": float(bt_sum["segments"]),
            "diff_pct": float(bt_sum["segments"] - ref_summary["segments"]),
            "diff_rel": 0.0,
        },
        {
            "metric": "trade_win_rate",
            "diagnose": float(ref_summary["trade_win_rate"]),
            "backtrader": float(bt_sum["trade_win_rate"]),
            "diff_pct": float(bt_sum["trade_win_rate"] - ref_summary["trade_win_rate"]),
            "diff_rel": 0.0,
        },
    ]
    seg_merge = bt_segments.merge(
        ref_segments[["segment_id", "pnl_per_capital", "trades"]],
        on="segment_id",
        how="outer",
        suffixes=("_bt", "_diag"),
    )
    if not seg_merge.empty:
        seg_merge["pnl_diff"] = seg_merge["pnl_per_capital_bt"].fillna(0) - seg_merge[
            "pnl_per_capital_diag"
        ].fillna(0)
        rows.append(
            {
                "metric": "segment_pnl_max_abs_diff",
                "diagnose": float(seg_merge["pnl_per_capital_diag"].abs().max()),
                "backtrader": float(seg_merge["pnl_per_capital_bt"].abs().max()),
                "diff_pct": float(seg_merge["pnl_diff"].abs().max()),
                "diff_rel": float(seg_merge["pnl_diff"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def _append_diagnose_cli_args(ap: argparse.ArgumentParser, defaults: dict) -> None:
    ap.add_argument(
        "--config",
        default="config/strategies/trend_scalp/research/calibrate_roll.default.yaml",
    )
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--warmup-days", type=int, default=120)
    ap.add_argument("--timeframe", default="2h")
    ap.add_argument(
        "--execution-timeframe", default=defaults.get("execution_timeframe", "1min")
    )
    ap.add_argument(
        "--regime", choices=["trend", "chop"], default=defaults.get("regime", "trend")
    )
    ap.add_argument(
        "--add-mode",
        choices=["both", "trend"],
        default=defaults.get("add_mode", "trend"),
    )
    ap.add_argument(
        "--flip-action",
        choices=["keep", "close_offside_adds", "close_offside_all"],
        default=defaults.get("flip_action", "close_offside_all"),
    )
    ap.add_argument(
        "--reseed-on-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--chop-min", type=float, default=defaults.get("chop_min", 0.40))
    ap.add_argument(
        "--exit-chop-min", type=float, default=defaults.get("exit_chop_min", 0.25)
    )
    ap.add_argument("--trend-min", type=float, default=defaults.get("trend_min", 0.80))
    ap.add_argument(
        "--trend-exit-min", type=float, default=defaults.get("trend_exit_min", 0.50)
    )
    ap.add_argument("--box-window", type=int, default=defaults.get("box_window", 120))
    ap.add_argument(
        "--step-atr-mult", type=float, default=defaults.get("step_atr_mult", 0.50)
    )
    ap.add_argument(
        "--take-profit-mode",
        choices=["basket", "per_leg"],
        default=defaults.get("take_profit_mode", "basket"),
    )
    ap.add_argument(
        "--tp-atr-mult", type=float, default=defaults.get("tp_atr_mult", 0.50)
    )
    ap.add_argument("--tp-abs", type=float, default=defaults.get("tp_abs", 0.0))
    ap.add_argument("--tp-pct", type=float, default=defaults.get("tp_pct", 0.0))
    ap.add_argument(
        "--max-adds-per-side", type=int, default=defaults.get("max_adds_per_side", 3)
    )
    ap.add_argument(
        "--max-net-exposure", type=int, default=defaults.get("max_net_exposure", 2)
    )
    ap.add_argument(
        "--max-gross-exposure", type=int, default=defaults.get("max_gross_exposure", 4)
    )
    ap.add_argument(
        "--max-loser-hold-bars",
        type=int,
        default=defaults.get("max_loser_hold_bars", 24),
    )
    ap.add_argument(
        "--scale-max-loser-hold-to-signal",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("scale_max_loser_hold_to_signal", False)),
    )
    ap.add_argument(
        "--min-segment-bars", type=int, default=defaults.get("min_segment_bars", 6)
    )
    ap.add_argument(
        "--max-segment-bars", type=int, default=defaults.get("max_segment_bars", 120)
    )
    ap.add_argument("--fee-bps", type=float, default=defaults.get("fee_bps", 4.0))
    ap.add_argument(
        "--market-exit-slippage-bps",
        type=float,
        default=defaults.get("market_exit_slippage_bps", 0.0),
    )
    ap.add_argument(
        "--intrabar-touch-buffer-bps",
        type=float,
        default=defaults.get("intrabar_touch_buffer_bps", 0.0),
    )
    ap.add_argument(
        "--entry-slippage-bps",
        type=float,
        default=defaults.get("entry_slippage_bps", 0.0),
    )
    ap.add_argument(
        "--add-slippage-bps", type=float, default=defaults.get("add_slippage_bps", 0.0)
    )
    ap.add_argument(
        "--max-loss-per-segment",
        type=float,
        default=defaults.get("max_loss_per_segment", 0.01),
    )
    ap.add_argument(
        "--risk-stop-mode",
        choices=["mtm", "regime_only"],
        default=defaults.get("risk_stop_mode", "mtm"),
    )
    ap.add_argument(
        "--initial-hedge",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("initial_hedge", True),
    )
    ap.add_argument(
        "--exclude-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("exclude_box", True),
    )
    ap.add_argument("--trend-return-horizons", default="3,5,10")
    ap.add_argument("--chop-signal", default=str(defaults.get("chop_signal", "raw")))
    ap.add_argument(
        "--chop-ts-window", type=int, default=int(defaults.get("chop_ts_window", 1200))
    )
    ap.add_argument(
        "--chop-ts-min-periods",
        type=int,
        default=int(defaults.get("chop_ts_min_periods", 150)),
    )
    ap.add_argument(
        "--compute-chop-ts-q",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("compute_chop_ts_q"),
    )
    ap.add_argument(
        "--stability-min", type=float, default=defaults.get("stability_min", 0.85)
    )
    ap.add_argument("--width-min", type=float, default=defaults.get("width_min", 0.04))
    ap.add_argument("--width-max", type=float, default=defaults.get("width_max", 0.30))
    ap.add_argument("--touches-min", type=int, default=defaults.get("touches_min", 5))
    ap.add_argument("--feature-store-dir", default=defaults.get("feature_store_dir"))
    ap.add_argument(
        "--feature-store-layer", default=defaults.get("feature_store_layer")
    )
    ap.add_argument(
        "--feature-store-timeframe", default=defaults.get("feature_store_timeframe")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default="results/trend_scalp/experiments/backtrader_crosscheck",
    )
    ap.add_argument(
        "--compare-dir",
        default="",
        help="Existing diagnose_dual_add_trend output dir for diff table",
    )
    ap.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Smoke: cap number of segments (0 = all)",
    )
    cfg_path = (
        PROJECT_ROOT
        / "config/strategies/trend_scalp/research/calibrate_roll.default.yaml"
    )
    defaults = _load_dual_add_defaults(cfg_path)
    _append_diagnose_cli_args(ap, defaults)
    args = ap.parse_args()

    if not Path(str(args.config)).is_absolute():
        args.config = str(PROJECT_ROOT / args.config)

    cfg = _build_dual_add_config(args, defaults)
    grid_cfg = GridConfig(
        box_window=cfg.box_window,
        chop_min=cfg.chop_min,
        exit_chop_min=cfg.exit_chop_min,
        chop_signal=cfg.chop_signal,
        chop_ts_window=cfg.chop_ts_window,
        chop_ts_min_periods=cfg.chop_ts_min_periods,
        compute_semantic_chop_ts_q=cfg.compute_semantic_chop_ts_q,
        stability_min=cfg.stability_min,
        width_min=cfg.width_min,
        width_max=cfg.width_max,
        touches_min=cfg.touches_min,
        feature_store_dir=resolve_optional_repo_path(args.feature_store_dir),
        feature_store_layer=(
            str(args.feature_store_layer).strip() if args.feature_store_layer else None
        )
        or None,
        feature_store_timeframe=(
            str(args.feature_store_timeframe).strip()
            if args.feature_store_timeframe
            else None
        )
        or None,
    )

    specs = _discover_segments(args, cfg, grid_cfg)
    if args.max_segments > 0:
        specs = specs[: args.max_segments]
    print(f"Running Backtrader on {len(specs)} segments (1min exec bars)...")

    all_trades: List[dict] = []
    all_segments: List[dict] = []
    for i, spec in enumerate(specs, start=1):
        trades, summary = _run_segment_backtrader(spec, cfg)
        all_trades.extend(trades)
        if summary:
            all_segments.append(summary)
        if i % 50 == 0 or i == len(specs):
            print(f"  ... {i}/{len(specs)} segments")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(all_trades)
    segments_df = pd.DataFrame(all_segments)
    summary_df = summarize(trades_df, segments_df)
    trades_df.to_csv(out_dir / "dual_add_trades.csv", index=False)
    segments_df.to_csv(out_dir / "dual_add_segments.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "engine": "backtrader",
                "execution_timeframe": args.execution_timeframe,
                "timeframe": args.timeframe,
                "segments_run": len(specs),
                "dual_add_config": asdict(cfg),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n=== Backtrader Summary ===")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    compare_dir = str(args.compare_dir or "").strip()
    if compare_dir:
        cmp_path = Path(compare_dir)
        if not cmp_path.is_absolute():
            cmp_path = PROJECT_ROOT / cmp_path
        cmp = _compare_runs(trades_df, segments_df, cmp_path)
        cmp.to_csv(out_dir / "compare_vs_diagnose.csv", index=False)
        print("\n=== vs diagnose_dual_add_trend ===")
        print(cmp.to_string(index=False))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
