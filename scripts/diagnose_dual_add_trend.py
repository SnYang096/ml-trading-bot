"""Diagnostic for dual-open winner-add strategy.

Idea from research:
  Open both long and short in a no-trend region. When price moves one way, add
  one more unit on the winning side. Close winning units at a fixed profit
  target. The danger is runaway gross exposure / trapped hedge inventory, so this
  diagnostic includes strict max levels and forced exits.

This is a risk study, not a production strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import (
    GridConfig,
    _hysteresis_segments,
    build_features,
    regime_chop_series,
    resolve_optional_repo_path,
)
from scripts.capital_report import write_capital_report_from_trades
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv
from src.time_series_model.grid.subbar_replay import (
    merge_signal_features_onto_execution_bars,
    slice_execution_window,
    timeframe_to_timedelta,
)
from scripts.multi_leg_trading_map import write_continuous_trading_map
from scripts.pipeline.multileg_prefilter_rules import apply_prefilter_rules
from src.config.multileg_config import load_multileg_effective_config
from src.config.strategy_layout import resolve_strategy_config_input
from src.features.time_series.baseline_features import (  # noqa: E402
    compute_trend_confidence_from_series,
)

DEFAULT_DUAL_ADD_CONFIG = (
    PROJECT_ROOT / "config/strategies/dual_add_trend/research/turbo.yaml"
)


def _load_dual_add_defaults(path: Path) -> dict:
    if not path.exists():
        return {}
    cfg_dir, profile_path, engine_path = resolve_strategy_config_input(path)
    cfg = load_multileg_effective_config(
        config_dir=cfg_dir,
        strategy_type="dual_add_trend",
        profile_path=profile_path,
        engine_path=engine_path,
    )
    regime = cfg.get("regime", {}) or {}
    inv = cfg.get("inventory", {}) or {}
    spacing = cfg.get("add_spacing", {}) or {}
    tp = cfg.get("take_profit", {}) or {}
    risk = cfg.get("risk", {}) or {}
    dual_bt = cfg.get("dual_add_backtest", {}) or {}
    costs = dual_bt.get("costs", {}) if isinstance(dual_bt, dict) else {}
    if not isinstance(costs, dict):
        costs = {}
    box_pf = regime.get("box_prefilter") or {}
    chop_series = cfg.get("chop_series", {}) or {}
    out: Dict[str, Any] = {
        "regime": "trend",
        "add_mode": str(inv.get("add_mode", "trend")),
        "flip_action": str(inv.get("flip_action", "close_offside_all")),
        "chop_min": float(regime.get("max_semantic_chop_hold", 0.40)),
        "exit_chop_min": float(regime.get("max_semantic_chop_entry", 0.25)),
        "trend_min": float(regime.get("entry_min", 0.80)),
        "trend_exit_min": float(regime.get("exit_below", 0.50)),
        "box_window": int(regime.get("box_window", 120)),
        "step_atr_mult": float(spacing.get("atr_mult", 0.50)),
        "take_profit_mode": str(tp.get("mode", "basket")),
        "tp_atr_mult": float(tp.get("atr_mult", 0.25)),
        "tp_abs": float(tp.get("min_abs", 0.0)),
        "tp_pct": float(tp.get("min_pct", 0.0005)),
        "max_adds_per_side": int(inv.get("max_adds_per_side", 3)),
        "max_net_exposure": int(inv.get("max_net_exposure_units", 2)),
        "max_gross_exposure": int(inv.get("max_gross_exposure_units", 4)),
        "max_loser_hold_bars": int(inv.get("max_loser_hold_bars", 24)),
        "max_loss_per_segment": float(risk.get("max_loss_per_segment", 0.01)),
        "risk_stop_mode": str(risk.get("risk_stop_mode", "mtm")),
        "min_segment_bars": int(risk.get("min_segment_bars", 6)),
        "max_segment_bars": int(risk.get("max_segment_bars", 120)),
        "fee_bps": float(
            costs.get(
                "fee_bps",
                risk.get("diagnostic_fee_bps", risk.get("fee_bps", 4.0)),
            )
        ),
        "market_exit_slippage_bps": float(
            costs.get(
                "market_exit_slippage_bps",
                risk.get("market_exit_slippage_bps", 0.0),
            )
        ),
        "intrabar_touch_buffer_bps": float(
            costs.get(
                "intrabar_touch_buffer_bps",
                risk.get("intrabar_touch_buffer_bps", 0.0),
            )
        ),
        "initial_hedge": set(inv.get("initial_legs", ["LONG", "SHORT"]))
        == {"LONG", "SHORT"},
        "exclude_box": bool(regime.get("exclude_box_prefilter", True)),
        "stability_min": float(box_pf.get("stability_min", 0.85)),
        "width_min": float(box_pf.get("width_min", 0.04)),
        "width_max": float(box_pf.get("width_max", 0.30)),
        "touches_min": int(box_pf.get("touches_min", 5)),
        "chop_signal": str(chop_series.get("chop_signal", "raw")),
        "chop_ts_window": int(chop_series.get("chop_ts_window", 1200)),
        "chop_ts_min_periods": int(chop_series.get("chop_ts_min_periods", 150)),
        "execution_timeframe": dual_bt.get("execution_timeframe"),
        "scale_max_loser_hold_to_signal": bool(
            dual_bt.get("scale_max_loser_hold_to_signal", False)
        ),
    }
    if "compute_semantic_chop_ts_q" in chop_series:
        out["compute_chop_ts_q"] = chop_series.get("compute_semantic_chop_ts_q")
    out["feature_store_dir"] = resolve_optional_repo_path(
        dual_bt.get("feature_store_dir")
    )
    out["feature_store_layer"] = dual_bt.get("feature_store_layer")
    out["feature_store_timeframe"] = dual_bt.get("feature_store_timeframe")
    out["prefilter_rules"] = cfg.get("rules", []) or []
    if out["feature_store_layer"] is not None:
        out["feature_store_layer"] = str(out["feature_store_layer"]).strip() or None
    if out["feature_store_timeframe"] is not None:
        out["feature_store_timeframe"] = (
            str(out["feature_store_timeframe"]).strip() or None
        )
    return out


@dataclass(frozen=True)
class DualAddConfig:
    regime: str = "trend"
    add_mode: str = "both"
    flip_action: str = "keep"
    chop_signal: str = "raw"
    chop_ts_window: int = 1200
    chop_ts_min_periods: int = 150
    compute_semantic_chop_ts_q: bool | None = None
    trend_return_horizons: Tuple[int, ...] = (3, 5, 10)
    stability_min: float = 0.85
    width_min: float = 0.04
    width_max: float = 0.30
    touches_min: int = 5
    chop_min: float = 0.40
    exit_chop_min: float = 0.25
    trend_min: float = 0.80
    trend_exit_min: float = 0.50
    box_window: int = 120
    step_atr_mult: float = 0.50
    take_profit_mode: str = "basket"
    tp_atr_mult: float = 0.50
    tp_abs: float = 0.0
    tp_pct: float = 0.0
    max_adds_per_side: int = 3
    max_net_exposure: int = 3
    max_gross_exposure: int = 5
    max_loser_hold_bars: int = 24
    max_segment_bars: int = 120
    min_segment_bars: int = 6
    fee_bps: float = 4.0
    market_exit_slippage_bps: float = 0.0
    intrabar_touch_buffer_bps: float = 0.0
    max_loss_per_segment: float = 0.01
    risk_stop_mode: str = "mtm"
    initial_hedge: bool = True
    prefilter_rules: Tuple[Dict[str, Any], ...] = ()


def _max_drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float((arr - np.maximum.accumulate(arr)).min())


def _position_pnl(side: str, entry: float, px: float, fee: float) -> float:
    if side == "LONG":
        return (px - entry) / entry - 2.0 * fee
    return (entry - px) / entry - 2.0 * fee


def _exit_price_with_slippage(side: str, px: float, slippage_bps: float) -> float:
    slip = max(float(slippage_bps), 0.0) / 10000.0
    if slip <= 0:
        return px
    return px * (1.0 - slip) if side == "LONG" else px * (1.0 + slip)


def _add_trend_features(
    df: pd.DataFrame, horizons: Tuple[int, ...] = (3, 5, 10)
) -> pd.DataFrame:
    """Attach dual_add trend columns; skip if already present (e.g. feature pipeline)."""
    out = df.copy()
    if not horizons:
        horizons = (3, 5, 10)
    if "trend_confidence" in out.columns and "trend_direction" in out.columns:
        return out
    bundle = compute_trend_confidence_from_series(close=out["close"], horizons=horizons)
    for col in bundle.columns:
        out[col] = bundle[col]
    return out


def _exposure_units(positions: List[dict]) -> Tuple[int, int, int]:
    long_units = sum(1 for p in positions if p["side"] == "LONG")
    short_units = sum(1 for p in positions if p["side"] == "SHORT")
    return long_units, short_units, long_units - short_units


def simulate_dual_add_segment(
    seg: pd.DataFrame,
    *,
    cfg: DualAddConfig,
    symbol: str,
    segment_id: str,
    direction: str,
    frozen_center: float | None = None,
    frozen_atr: float | None = None,
) -> Tuple[List[dict], dict]:
    if seg.empty:
        return [], {}
    center = (
        float(frozen_center)
        if frozen_center is not None
        else float(seg["close"].iloc[0])
    )
    atr = float(frozen_atr) if frozen_atr is not None else float(seg["atr14"].iloc[0])
    if not np.isfinite(center + atr) or center <= 0 or atr <= 0:
        return [], {}
    fee = cfg.fee_bps / 10000.0
    step = cfg.step_atr_mult * atr
    tp = max(cfg.tp_abs, cfg.tp_atr_mult * atr, cfg.tp_pct * center)
    if step <= 0 or tp <= 0:
        return [], {}
    touch_buffer = max(float(cfg.intrabar_touch_buffer_bps), 0.0) / 10000.0

    def hit_up(level: float) -> bool:
        return high >= level * (1.0 + touch_buffer)

    def hit_down(level: float) -> bool:
        return low <= level * (1.0 - touch_buffer)

    # Capital units are the maximum gross inventory this experiment allows.
    capital_units = max(2, cfg.max_gross_exposure)
    positions: List[dict] = []
    trades: List[dict] = []
    last_add_long = center
    last_add_short = center
    add_long_count = 0
    add_short_count = 0
    pnl_path: List[float] = []
    stop_reason = "regime_exit"
    max_gross_units = len(positions)
    max_abs_net_units = 0
    trend_side = "LONG" if direction == "UP" else "SHORT"
    last_trend_side = trend_side
    trend_flips = 0
    flip_forced = 0
    last_flat_bar = -1

    def seed_positions(px: float, ts: pd.Timestamp, bar_i: int) -> None:
        nonlocal last_add_long, last_add_short
        sides = ["LONG", "SHORT"] if cfg.initial_hedge else [trend_side]
        for side in sides:
            positions.append(
                {
                    "side": side,
                    "entry": px,
                    "entry_time": ts,
                    "entry_bar": bar_i,
                    "seq": 0,
                }
            )
        last_add_long = px
        last_add_short = px

    seed_positions(center, seg.index[0], 0)
    max_gross_units = len(positions)

    def record(pos: dict, exit_px: float, exit_ts: pd.Timestamp, reason: str) -> None:
        slippage_bps = (
            0.0 if reason == "tp" else max(float(cfg.market_exit_slippage_bps), 0.0)
        )
        filled_exit_px = _exit_price_with_slippage(
            str(pos["side"]), float(exit_px), slippage_bps
        )
        pnl_pct = _position_pnl(pos["side"], float(pos["entry"]), filled_exit_px, fee)
        trades.append(
            {
                "symbol": symbol,
                "segment_id": segment_id,
                "direction": direction,
                "side": pos["side"],
                "seq": pos["seq"],
                "entry_time": pos["entry_time"],
                "exit_time": exit_ts,
                "entry_price": pos["entry"],
                "exit_price": filled_exit_px,
                "signal_exit_price": exit_px,
                "exit_reason": reason,
                "pnl_pct": pnl_pct,
                "pnl_per_capital": pnl_pct / capital_units,
                "fee_bps_charged": 2.0 * cfg.fee_bps,
                "slippage_bps_charged": slippage_bps,
            }
        )

    def can_add(side: str) -> bool:
        hypothetical = positions + [{"side": side}]
        _, _, net_units = _exposure_units(hypothetical)
        return (
            len(hypothetical) <= cfg.max_gross_exposure
            and abs(net_units) <= cfg.max_net_exposure
        )

    def enforce_net_cap(px: float, ts: pd.Timestamp) -> None:
        _, _, net_units = _exposure_units(positions)
        while abs(net_units) > cfg.max_net_exposure and positions:
            overloaded_side = "LONG" if net_units > 0 else "SHORT"
            candidates = [p for p in positions if p["side"] == overloaded_side]
            if not candidates:
                break
            pos = min(
                candidates,
                key=lambda p: _position_pnl(str(p["side"]), float(p["entry"]), px, fee),
            )
            record(pos, px, ts, "net_cap")
            positions.remove(pos)
            _, _, net_units = _exposure_units(positions)

    def close_offside_positions(px: float, ts: pd.Timestamp, side: str) -> int:
        nonlocal last_flat_bar
        if cfg.flip_action == "keep":
            return 0
        closed = 0
        for pos in list(positions):
            if pos["side"] == side:
                continue
            if cfg.flip_action == "close_offside_adds" and int(pos.get("seq", 0)) == 0:
                continue
            record(pos, px, ts, "trend_flip")
            positions.remove(pos)
            closed += 1
        if not positions:
            last_flat_bar = bar_i
        return closed

    def open_pnl_per_capital(px: float) -> float:
        return (
            sum(_position_pnl(p["side"], float(p["entry"]), px, fee) for p in positions)
            / capital_units
        )

    def basket_target_per_capital(px: float) -> float:
        fee_buffer = 2.0 * fee * px
        slippage_buffer = max(float(cfg.market_exit_slippage_bps), 0.0) / 10000.0 * px
        target = max(cfg.tp_abs, cfg.tp_atr_mult * atr, cfg.tp_pct * px)
        return ((fee_buffer + slippage_buffer + target) / px) / capital_units

    def close_all_open(px: float, ts: pd.Timestamp, reason: str) -> None:
        nonlocal last_flat_bar
        for pos in list(positions):
            record(pos, px, ts, reason)
            positions.remove(pos)
        last_flat_bar = bar_i

    for bar_i, (ts, row) in enumerate(seg.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        actionable = bar_i > 0
        if cfg.add_mode == "trend" and actionable:
            signal_row = seg.iloc[bar_i - 1]
            trend_side = (
                "LONG"
                if str(signal_row.get("trend_direction", direction)) == "UP"
                else "SHORT"
            )

        seeded_this_bar = False
        if actionable and not positions and bar_i > last_flat_bar:
            seed_positions(close, ts, bar_i)
            seeded_this_bar = True

        if actionable and not seeded_this_bar:
            # Basket mode closes the current inventory together once aggregate
            # net PnL clears the target. This prevents an initial hedge leg from
            # being stranded as a loss anchor after the other leg takes profit.
            closed_basket = False
            if cfg.take_profit_mode == "basket":
                if positions and open_pnl_per_capital(
                    close
                ) >= basket_target_per_capital(close):
                    close_all_open(close, ts, "basket_tp")
                    closed_basket = True
            else:
                # Close only after the target covers the modeled round-trip fee.
                for pos in list(positions):
                    entry = float(pos["entry"])
                    fee_buffer = 2.0 * fee * entry
                    if pos["side"] == "LONG" and hit_up(entry + tp + fee_buffer):
                        record(pos, entry + tp + fee_buffer, ts, "tp")
                        positions.remove(pos)
                    elif pos["side"] == "SHORT" and hit_down(entry - tp - fee_buffer):
                        record(pos, entry - tp - fee_buffer, ts, "tp")
                        positions.remove(pos)
            enforce_net_cap(close, ts)
            if closed_basket:
                _, _, net_units = _exposure_units(positions)
                max_gross_units = max(max_gross_units, len(positions))
                max_abs_net_units = max(max_abs_net_units, abs(net_units))
                realized = sum(
                    t["pnl_pct"] for t in trades if t["segment_id"] == segment_id
                )
                pnl_path.append(realized / capital_units)
                continue

            if cfg.add_mode == "trend" and trend_side != last_trend_side:
                trend_flips += 1
                flip_forced += close_offside_positions(close, ts, trend_side)
                last_trend_side = trend_side
                enforce_net_cap(close, ts)

            # Do not let the initial hedge's losing leg become an unbounded anchor.
            for pos in list(positions):
                held_bars = bar_i - int(pos.get("entry_bar", 0))
                if held_bars < cfg.max_loser_hold_bars:
                    continue
                if _position_pnl(pos["side"], float(pos["entry"]), close, fee) < 0:
                    record(pos, close, ts, "loser_timeout")
                    positions.remove(pos)
            enforce_net_cap(close, ts)

        # In both-side mode, price extension may add on either side. Strict
        # gross/net caps and segment loss stops keep inventory bounded.
        if actionable:
            while (
                cfg.add_mode in {"both", "trend"}
                and (cfg.add_mode == "both" or trend_side == "LONG")
                and hit_up(last_add_long + step)
                and add_long_count < cfg.max_adds_per_side
                and can_add("LONG")
            ):
                last_add_long += step
                add_long_count += 1
                positions.append(
                    {
                        "side": "LONG",
                        "entry": last_add_long,
                        "entry_time": ts,
                        "entry_bar": bar_i,
                        "seq": add_long_count,
                    }
                )
            while (
                cfg.add_mode in {"both", "trend"}
                and (cfg.add_mode == "both" or trend_side == "SHORT")
                and hit_down(last_add_short - step)
                and add_short_count < cfg.max_adds_per_side
                and can_add("SHORT")
            ):
                last_add_short -= step
                add_short_count += 1
                positions.append(
                    {
                        "side": "SHORT",
                        "entry": last_add_short,
                        "entry_time": ts,
                        "entry_bar": bar_i,
                        "seq": add_short_count,
                    }
                )
            enforce_net_cap(close, ts)

        _, _, net_units = _exposure_units(positions)
        max_gross_units = max(max_gross_units, len(positions))
        max_abs_net_units = max(max_abs_net_units, abs(net_units))
        realized = sum(t["pnl_pct"] for t in trades if t["segment_id"] == segment_id)
        mtm = realized + sum(
            _position_pnl(p["side"], float(p["entry"]), close, fee) for p in positions
        )
        mtm_per_capital = mtm / capital_units
        pnl_path.append(mtm_per_capital)
        if cfg.risk_stop_mode == "mtm" and mtm_per_capital <= -cfg.max_loss_per_segment:
            stop_reason = "risk_stop"
            break

    exit_ts = ts
    exit_close = float(seg.loc[exit_ts, "close"])
    forced = len(positions)
    for pos in list(positions):
        record(pos, exit_close, exit_ts, stop_reason)
    total = sum(t["pnl_per_capital"] for t in trades if t["segment_id"] == segment_id)
    summary = {
        "symbol": symbol,
        "segment_id": segment_id,
        "direction": direction,
        "start": seg.index[0],
        "end": exit_ts,
        "bars": len(seg.loc[:exit_ts]),
        "trades": sum(1 for t in trades if t["segment_id"] == segment_id),
        "tp": sum(
            1
            for t in trades
            if t["segment_id"] == segment_id and t["exit_reason"] in {"tp", "basket_tp"}
        ),
        "forced": forced,
        "risk_stop": int(stop_reason == "risk_stop"),
        "max_add_long": add_long_count,
        "max_add_short": add_short_count,
        "max_gross_units": max_gross_units,
        "max_abs_net_units": max_abs_net_units,
        "trend_flips": trend_flips,
        "flip_forced": flip_forced,
        "pnl_per_capital": total,
        "max_drawdown": _max_drawdown(pnl_path),
    }
    return trades, summary


def _effective_max_loser_hold_bars(args: argparse.Namespace) -> int:
    base = int(args.max_loser_hold_bars)
    exec_tf = args.execution_timeframe or args.timeframe
    if not args.scale_max_loser_hold_to_signal:
        return max(1, base)
    if exec_tf == args.timeframe:
        return max(1, base)
    sig_s = timeframe_to_timedelta(args.timeframe).total_seconds()
    ex_s = timeframe_to_timedelta(exec_tf).total_seconds()
    if ex_s <= 0:
        return max(1, base)
    ratio = sig_s / ex_s
    return max(1, int(round(base * ratio)))


def run(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg_path = Path(str(args.config))
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    defaults = _load_dual_add_defaults(cfg_path)
    eff_hold = _effective_max_loser_hold_bars(args)
    if eff_hold != int(args.max_loser_hold_bars):
        print(
            f"(resolved max_loser_hold_bars={eff_hold} from CLI "
            f"{int(args.max_loser_hold_bars)} via --scale-max-loser-hold-to-signal)"
        )
    hparts = [
        x.strip() for x in str(args.trend_return_horizons).split(",") if x.strip()
    ]
    trend_horizons = tuple(int(x) for x in hparts) if hparts else (3, 5, 10)
    cfg = DualAddConfig(
        regime=args.regime,
        add_mode=args.add_mode,
        flip_action=args.flip_action,
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
        max_loss_per_segment=args.max_loss_per_segment,
        risk_stop_mode=args.risk_stop_mode,
        initial_hedge=args.initial_hedge,
        prefilter_rules=tuple(
            x
            for x in (defaults.get("prefilter_rules", []) or [])
            if isinstance(x, dict)
        ),
    )
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
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=args.warmup_days)
    data_dir = Path(args.data_dir)
    all_trades: List[dict] = []
    all_segments: List[dict] = []
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
        exec_tf = args.execution_timeframe or args.timeframe
        if exec_tf != args.timeframe:
            bars_exec = _resample_ohlcv(raw, exec_tf)
            sig_delta = timeframe_to_timedelta(args.timeframe)
            df_exec = merge_signal_features_onto_execution_bars(
                bars_exec, df, signal_bar_delta=sig_delta
            )
        else:
            df_exec = None
            sig_delta = None
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
            if df_exec is not None:
                seg_slice = slice_execution_window(df_exec, df.index, s, e, sig_delta)
                anchor_c = float(df.iloc[s]["close"])
                anchor_a = float(df.iloc[s]["atr14"])
                trades, summary = simulate_dual_add_segment(
                    seg_slice,
                    cfg=cfg,
                    symbol=symbol,
                    segment_id=seg_id,
                    direction=direction,
                    frozen_center=anchor_c,
                    frozen_atr=anchor_a,
                )
            else:
                trades, summary = simulate_dual_add_segment(
                    df.iloc[s : e + 1],
                    cfg=cfg,
                    symbol=symbol,
                    segment_id=seg_id,
                    direction=direction,
                )
            all_trades.extend(trades)
            if summary:
                all_segments.append(summary)
    return pd.DataFrame(all_trades), pd.DataFrame(all_segments)


def summarize(trades: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or segments.empty:
        return pd.DataFrame()
    sum_pc = float(trades["pnl_per_capital"].sum())
    return pd.DataFrame(
        [
            {
                "segments": len(segments),
                "trades": len(trades),
                "trade_win_rate": (trades["pnl_pct"] > 0).mean(),
                "segment_win_rate": (segments["pnl_per_capital"] > 0).mean(),
                "sum_pnl_per_capital": sum_pc,
                # Same convention as chop_grid_backtest trade summary: sum of per-trade
                # capital-normalized PnL, expressed as percentage points.
                "return_pct": sum_pc * 100.0,
                "fee_bps_charged_sum": float(
                    trades.get("fee_bps_charged", pd.Series(dtype=float)).sum()
                ),
                "slippage_bps_charged_sum": float(
                    trades.get("slippage_bps_charged", pd.Series(dtype=float)).sum()
                ),
                "worst_segment": segments["pnl_per_capital"].min(),
                "median_drawdown": segments["max_drawdown"].median(),
                # Fraction of segments that hit max_loss_per_segment (mtm stop) before
                # the segment would otherwise end; see simulate_dual_add_segment risk_stop.
                "risk_stop_rate": segments["risk_stop"].mean(),
                "max_gross_units": segments["max_gross_units"].max(),
                "max_abs_net_units": segments["max_abs_net_units"].max(),
                "loser_timeout_rate": (trades["exit_reason"] == "loser_timeout").mean(),
                "tp_rate": trades["exit_reason"].isin(["tp", "basket_tp"]).mean(),
                "forced_rate": (
                    ~trades["exit_reason"].isin(["tp", "basket_tp"])
                ).mean(),
            }
        ]
    )


def write_trading_maps(
    out_dir: Path,
    args: argparse.Namespace,
    trades: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    """Write per-symbol visual maps for dual-add segments and leg exits."""
    if trades.empty and segments.empty:
        return
    try:
        from bokeh.io import output_file, save
        from bokeh.models import BoxAnnotation, ColumnDataSource, HoverTool
        from bokeh.plotting import figure
    except Exception as exc:  # pragma: no cover - optional report dependency
        print(f"skip trading map: bokeh unavailable ({exc})")
        return

    symbols = [s.strip().upper() for s in args.map_symbols.split(",") if s.strip()]
    if not symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()][:1]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=args.warmup_days)
    data_dir = Path(args.data_dir)
    bar_width_ms = pd.Timedelta(args.timeframe).total_seconds() * 1000 * 0.72

    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            continue
        bars = _resample_ohlcv(raw, args.timeframe)
        df = bars[(bars.index >= start) & (bars.index <= end)].copy()
        if df.empty:
            continue
        df = df.reset_index(names="time")
        inc = df["close"] >= df["open"]
        dec = ~inc
        p = figure(
            x_axis_type="datetime",
            width=1300,
            height=720,
            title=f"Dual Add Trend Trading Map - {symbol}",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.segment(
            df["time"], df["high"], df["time"], df["low"], color="#6b7280", alpha=0.6
        )
        p.vbar(
            df.loc[inc, "time"],
            bar_width_ms,
            df.loc[inc, "open"],
            df.loc[inc, "close"],
            fill_color="#16a34a",
            line_color="#16a34a",
            alpha=0.65,
        )
        p.vbar(
            df.loc[dec, "time"],
            bar_width_ms,
            df.loc[dec, "open"],
            df.loc[dec, "close"],
            fill_color="#dc2626",
            line_color="#dc2626",
            alpha=0.65,
        )

        sseg = (
            segments[segments["symbol"] == symbol].copy()
            if not segments.empty
            else pd.DataFrame()
        )
        if not sseg.empty:
            sseg["start"] = pd.to_datetime(sseg["start"], utc=True)
            sseg["end"] = pd.to_datetime(sseg["end"], utc=True)
            sseg = sseg[(sseg["end"] >= start) & (sseg["start"] <= end)]
            for _, row in sseg.iterrows():
                fill_color = "#22c55e" if row.get("direction") == "UP" else "#ef4444"
                p.add_layout(
                    BoxAnnotation(
                        left=row["start"],
                        right=row["end"],
                        fill_color=fill_color,
                        fill_alpha=0.07,
                        line_alpha=0.0,
                    )
                )

        strades = (
            trades[trades["symbol"] == symbol].copy()
            if not trades.empty
            else pd.DataFrame()
        )
        if not strades.empty:
            strades["entry_time"] = pd.to_datetime(strades["entry_time"], utc=True)
            strades["exit_time"] = pd.to_datetime(strades["exit_time"], utc=True)
            strades = strades[
                (strades["exit_time"] >= start) & (strades["entry_time"] <= end)
            ]
            for side, color, marker in [
                ("LONG", "#2563eb", "triangle"),
                ("SHORT", "#9333ea", "inverted_triangle"),
            ]:
                src_df = strades[strades["side"] == side]
                if src_df.empty:
                    continue
                entry_src = ColumnDataSource(src_df)
                glyph = getattr(p, marker)
                r = glyph(
                    "entry_time",
                    "entry_price",
                    source=entry_src,
                    size=8,
                    color=color,
                    alpha=0.85,
                    legend_label=f"{side} entry",
                )
                p.add_tools(
                    HoverTool(
                        renderers=[r],
                        tooltips=[
                            ("side", "@side"),
                            ("seq", "@seq"),
                            ("entry", "@entry_price{0.0000}"),
                            ("exit", "@exit_price{0.0000}"),
                            ("pnl", "@pnl_pct{0.0000}"),
                            ("reason", "@exit_reason"),
                        ],
                    )
                )
            exit_src = ColumnDataSource(strades)
            p.circle(
                "exit_time",
                "exit_price",
                source=exit_src,
                size=5,
                color="#111827",
                alpha=0.55,
                legend_label="exit",
            )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Price"
        out_path = out_dir / f"trading_map_dual_add_{symbol}.html"
        output_file(out_path, title=f"Dual Add Trend Trading Map - {symbol}")
        save(p)
        print(f"Saved trading map -> {out_path}")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_DUAL_ADD_CONFIG))
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    defaults = _load_dual_add_defaults(config_path)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(config_path))
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--warmup-days", type=int, default=120)
    ap.add_argument(
        "--timeframe",
        default="2h",
        help=(
            "Signal timeframe (pandas offset, e.g. 2h). Segments and regime masks use "
            "this bar length; features are built on this grid."
        ),
    )
    ap.add_argument(
        "--execution-timeframe",
        default=defaults.get("execution_timeframe"),
        help=(
            "Optional finer resample for inventory simulation (e.g. 1min). When set and "
            "different from --timeframe, segment boundaries follow the signal grid but "
            "TP/add/risk use execution OHLC with signal columns asof-joined. "
            "max_loser_hold_bars counts execution bars unless "
            "--scale-max-loser-hold-to-signal is enabled."
        ),
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
    ap.add_argument("--chop-min", type=float, default=defaults.get("chop_min", 0.40))
    ap.add_argument(
        "--exit-chop-min", type=float, default=defaults.get("exit_chop_min", 0.25)
    )
    ap.add_argument(
        "--chop-signal",
        choices=["raw", "ts_quantile"],
        default=str(defaults.get("chop_signal", "raw")),
        help="Chop series for regime masks: raw semantic_chop or causal ts quantile (~pct).",
    )
    ap.add_argument(
        "--chop-ts-window",
        type=int,
        default=int(defaults.get("chop_ts_window", 1200)),
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
        help=(
            "Force build of semantic_chop_ts_q column. "
            "Default: dual_add.yaml chop_series if set, else only when --chop-signal ts_quantile."
        ),
    )
    ap.add_argument(
        "--feature-store-dir",
        default=defaults.get("feature_store_dir"),
        help="FeatureStore root (dual_add_backtest.feature_store_dir in YAML).",
    )
    ap.add_argument(
        "--feature-store-layer",
        default=defaults.get("feature_store_layer"),
    )
    ap.add_argument(
        "--feature-store-timeframe",
        default=defaults.get("feature_store_timeframe"),
    )
    _hdef = defaults.get("trend_return_horizons", (3, 5, 10))
    ap.add_argument(
        "--trend-return-horizons",
        type=str,
        default=",".join(str(x) for x in _hdef),
        help="Comma-separated lookback bars for trend_confidence (e.g. 3,5,10).",
    )
    ap.add_argument(
        "--stability-min",
        type=float,
        default=float(defaults.get("stability_min", 0.85)),
        help="Box prefilter stability_min (StudyConfig); see regime.box_prefilter.",
    )
    ap.add_argument(
        "--width-min",
        type=float,
        default=float(defaults.get("width_min", 0.04)),
    )
    ap.add_argument(
        "--width-max",
        type=float,
        default=float(defaults.get("width_max", 0.30)),
    )
    ap.add_argument(
        "--touches-min",
        type=int,
        default=int(defaults.get("touches_min", 5)),
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
        "--tp-atr-mult", type=float, default=defaults.get("tp_atr_mult", 0.25)
    )
    ap.add_argument(
        "--take-profit-mode",
        choices=["basket", "per_leg"],
        default=defaults.get("take_profit_mode", "basket"),
        help=(
            "basket closes current inventory together on aggregate net profit; "
            "per_leg is the legacy independent leg TP."
        ),
    )
    ap.add_argument("--tp-abs", type=float, default=defaults.get("tp_abs", 0.0))
    ap.add_argument("--tp-pct", type=float, default=defaults.get("tp_pct", 0.0005))
    ap.add_argument(
        "--max-adds-per-side",
        type=int,
        default=defaults.get("max_adds_per_side", 3),
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
        help=(
            "When --execution-timeframe is finer than --timeframe, scale "
            "max_loser_hold_bars by (signal bar length / exec bar length) so one "
            "unit of hold still means the same wall-clock patience as on the "
            "signal grid (e.g. 24 @ 2h -> 2880 @ 1min). Off by default for "
            "backward compatibility."
        ),
    )
    ap.add_argument(
        "--initial-hedge",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("initial_hedge", True),
        help=(
            "If disabled, open only the trend-direction leg at segment start "
            "(no opening long+short straddle). For research on directional "
            "fast legs vs hedged inventory."
        ),
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
        help=(
            "mtm applies max_loss_per_segment inside a segment; regime_only lets "
            "trend/chop segment loss end the position and treats max loss as an "
            "external/deployment gate."
        ),
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
        help="Adverse slippage applied to basket/forced market-style exits.",
    )
    ap.add_argument(
        "--intrabar-touch-buffer-bps",
        type=float,
        default=defaults.get("intrabar_touch_buffer_bps", 0.0),
        help="Extra high/low penetration required before intrabar add/TP fills.",
    )
    ap.add_argument(
        "--exclude-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("exclude_box", True),
    )
    ap.add_argument("--out-dir", default="results/dual_add_trend_diagnostic")
    ap.add_argument("--map-symbols", default="BTCUSDT")
    ap.add_argument("--continuous-map-symbols", default="")
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades, segments = run(args)
    summary = summarize(trades, segments)
    resolved_hold = _effective_max_loser_hold_bars(args)
    if not summary.empty:
        summary["signal_timeframe"] = str(args.timeframe)
        summary["execution_timeframe"] = str(args.execution_timeframe or args.timeframe)
        summary["execution_replay_enabled"] = bool(
            str(args.execution_timeframe or args.timeframe) != str(args.timeframe)
        )
        summary["resolved_max_loser_hold_bars"] = int(resolved_hold)
    trades.to_csv(out_dir / "dual_add_trades.csv", index=False)
    segments.to_csv(out_dir / "dual_add_segments.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_capital_report_from_trades(
        trades_path=out_dir / "dual_add_trades.csv",
        out_dir=out_dir,
        unit="capital_normalized",
        title="Dual Add Trend Capital Report",
        start_date=args.start,
        end_date=args.end,
        total_r=float(trades["pnl_per_capital"].sum()) if not trades.empty else 0.0,
    )
    if not args.no_maps:
        write_trading_maps(out_dir, args, trades, segments)
        write_continuous_trading_map(
            out_path=out_dir / "trading_map_continuous.html",
            data_dir=Path(args.data_dir),
            symbols=args.symbols,
            map_symbols=args.continuous_map_symbols,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            warmup_days=args.warmup_days,
            trades=trades,
            segments=segments,
            title="Dual Add Trend Continuous Trading Map",
        )
    cfg_dump = dict(vars(args))
    cfg_dump["_resolved_max_loser_hold_bars"] = resolved_hold
    (out_dir / "config.json").write_text(
        json.dumps(cfg_dump, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== Dual Add Summary ===")
    print(summary.to_string(index=False) if not summary.empty else "(empty)")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
