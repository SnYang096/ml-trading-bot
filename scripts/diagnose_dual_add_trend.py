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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import GridConfig, _hysteresis_segments, build_features
from scripts.capital_report import write_capital_report_from_trades
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv
from scripts.multi_leg_trading_map import write_continuous_trading_map

DEFAULT_DUAL_ADD_CONFIG = (
    PROJECT_ROOT / "config/strategies/dual_add_trend/dual_add.yaml"
)


def _load_dual_add_defaults(path: Path) -> dict:
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    regime = cfg.get("regime", {}) or {}
    inv = cfg.get("inventory", {}) or {}
    spacing = cfg.get("add_spacing", {}) or {}
    tp = cfg.get("take_profit", {}) or {}
    risk = cfg.get("risk", {}) or {}
    return {
        "regime": "trend",
        "add_mode": str(inv.get("add_mode", "trend")),
        "flip_action": str(inv.get("flip_action", "close_offside_all")),
        "chop_min": float(regime.get("max_semantic_chop_hold", 0.40)),
        "exit_chop_min": float(regime.get("max_semantic_chop_entry", 0.25)),
        "trend_min": float(regime.get("entry_min", 0.80)),
        "trend_exit_min": float(regime.get("exit_below", 0.50)),
        "box_window": int(regime.get("box_window", 120)),
        "step_atr_mult": float(spacing.get("atr_mult", 0.50)),
        "tp_atr_mult": float(tp.get("atr_mult", 0.25)),
        "tp_abs": float(tp.get("min_abs", 0.0)),
        "tp_pct": float(tp.get("min_pct", 0.0005)),
        "max_adds_per_side": int(inv.get("max_adds_per_side", 3)),
        "max_net_exposure": int(inv.get("max_net_exposure_units", 2)),
        "max_gross_exposure": int(inv.get("max_gross_exposure_units", 4)),
        "max_loser_hold_bars": int(inv.get("max_loser_hold_bars", 24)),
        "max_loss_per_segment": float(risk.get("max_loss_per_segment", 0.01)),
        "min_segment_bars": int(risk.get("min_segment_bars", 6)),
        "max_segment_bars": int(risk.get("max_segment_bars", 120)),
        "fee_bps": float(risk.get("diagnostic_fee_bps", risk.get("fee_bps", 4.0))),
        "exclude_box": bool(regime.get("exclude_box_prefilter", True)),
    }


@dataclass(frozen=True)
class DualAddConfig:
    regime: str = "trend"
    add_mode: str = "both"
    flip_action: str = "keep"
    chop_min: float = 0.40
    exit_chop_min: float = 0.25
    trend_min: float = 0.80
    trend_exit_min: float = 0.50
    box_window: int = 120
    step_atr_mult: float = 0.50
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
    max_loss_per_segment: float = 0.01


def _max_drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float((arr - np.maximum.accumulate(arr)).min())


def _position_pnl(side: str, entry: float, px: float, fee: float) -> float:
    if side == "LONG":
        return (px - entry) / entry - 2.0 * fee
    return (entry - px) / entry - 2.0 * fee


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ret3 = out["close"].pct_change(3)
    ret5 = out["close"].pct_change(5)
    ret10 = out["close"].pct_change(10)
    signs = pd.concat([np.sign(ret3), np.sign(ret5), np.sign(ret10)], axis=1).fillna(
        0.0
    )
    out["trend_direction_raw"] = np.sign(signs.mean(axis=1))
    out["trend_confidence"] = signs.abs().mean(axis=1) * signs.mean(axis=1).abs()
    out["trend_direction"] = np.where(out["trend_direction_raw"] >= 0, "UP", "DOWN")
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
) -> Tuple[List[dict], dict]:
    if seg.empty:
        return [], {}
    center = float(seg["close"].iloc[0])
    atr = float(seg["atr14"].iloc[0])
    if not np.isfinite(center + atr) or center <= 0 or atr <= 0:
        return [], {}
    fee = cfg.fee_bps / 10000.0
    step = cfg.step_atr_mult * atr
    tp = max(cfg.tp_abs, cfg.tp_atr_mult * atr, cfg.tp_pct * center)
    if step <= 0 or tp <= 0:
        return [], {}

    # Capital units are the maximum gross inventory this experiment allows.
    capital_units = max(2, cfg.max_gross_exposure)
    positions: List[dict] = [
        {
            "side": "LONG",
            "entry": center,
            "entry_time": seg.index[0],
            "entry_bar": 0,
            "seq": 0,
        },
        {
            "side": "SHORT",
            "entry": center,
            "entry_time": seg.index[0],
            "entry_bar": 0,
            "seq": 0,
        },
    ]
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

    def record(pos: dict, exit_px: float, exit_ts: pd.Timestamp, reason: str) -> None:
        pnl_pct = _position_pnl(pos["side"], float(pos["entry"]), exit_px, fee)
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
                "exit_price": exit_px,
                "exit_reason": reason,
                "pnl_pct": pnl_pct,
                "pnl_per_capital": pnl_pct / capital_units,
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
        return closed

    for bar_i, (ts, row) in enumerate(seg.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if cfg.add_mode == "trend":
            trend_side = (
                "LONG"
                if str(row.get("trend_direction", direction)) == "UP"
                else "SHORT"
            )

        # Close only after the target covers the modeled round-trip fee.
        for pos in list(positions):
            entry = float(pos["entry"])
            fee_buffer = 2.0 * fee * entry
            if pos["side"] == "LONG" and high >= entry + tp + fee_buffer:
                record(pos, entry + tp + fee_buffer, ts, "tp")
                positions.remove(pos)
            elif pos["side"] == "SHORT" and low <= entry - tp - fee_buffer:
                record(pos, entry - tp - fee_buffer, ts, "tp")
                positions.remove(pos)
        enforce_net_cap(close, ts)

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
        while (
            cfg.add_mode in {"both", "trend"}
            and (cfg.add_mode == "both" or trend_side == "LONG")
            and high >= last_add_long + step
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
            and low <= last_add_short - step
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
        if mtm_per_capital <= -cfg.max_loss_per_segment:
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
            if t["segment_id"] == segment_id and t["exit_reason"] == "tp"
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


def run(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = DualAddConfig(
        regime=args.regime,
        add_mode=args.add_mode,
        flip_action=args.flip_action,
        chop_min=args.chop_min,
        exit_chop_min=args.exit_chop_min,
        trend_min=args.trend_min,
        trend_exit_min=args.trend_exit_min,
        box_window=args.box_window,
        step_atr_mult=args.step_atr_mult,
        tp_atr_mult=args.tp_atr_mult,
        tp_abs=args.tp_abs,
        tp_pct=args.tp_pct,
        max_adds_per_side=args.max_adds_per_side,
        max_net_exposure=args.max_net_exposure,
        max_gross_exposure=args.max_gross_exposure,
        max_loser_hold_bars=args.max_loser_hold_bars,
        max_segment_bars=args.max_segment_bars,
        min_segment_bars=args.min_segment_bars,
        fee_bps=args.fee_bps,
        max_loss_per_segment=args.max_loss_per_segment,
    )
    grid_cfg = GridConfig(
        box_window=cfg.box_window,
        chop_min=cfg.chop_min,
        exit_chop_min=cfg.exit_chop_min,
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
        bars = _resample_ohlcv(raw, args.timeframe)
        df = build_features(symbol, bars, grid_cfg)
        df = _add_trend_features(df)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if cfg.regime == "trend":
            entry = (df["trend_confidence"] >= cfg.trend_min) & (
                df["semantic_chop"] <= cfg.exit_chop_min
            )
            hold = (df["trend_confidence"] >= cfg.trend_exit_min) & (
                df["semantic_chop"] <= cfg.chop_min
            )
        else:
            entry = df["semantic_chop"] >= cfg.chop_min
            hold = df["semantic_chop"] >= cfg.exit_chop_min
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
            seg_id = f"{symbol}_{seq:04d}_{df.index[s].strftime('%Y%m%d%H')}"
            direction = str(df["trend_direction"].iloc[s])
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
    return pd.DataFrame(
        [
            {
                "segments": len(segments),
                "trades": len(trades),
                "trade_win_rate": (trades["pnl_pct"] > 0).mean(),
                "segment_win_rate": (segments["pnl_per_capital"] > 0).mean(),
                "sum_pnl_per_capital": trades["pnl_per_capital"].sum(),
                "worst_segment": segments["pnl_per_capital"].min(),
                "median_drawdown": segments["max_drawdown"].median(),
                "risk_stop_rate": segments["risk_stop"].mean(),
                "max_gross_units": segments["max_gross_units"].max(),
                "max_abs_net_units": segments["max_abs_net_units"].max(),
                "loser_timeout_rate": (trades["exit_reason"] == "loser_timeout").mean(),
                "tp_rate": (trades["exit_reason"] == "tp").mean(),
                "forced_rate": (trades["exit_reason"] != "tp").mean(),
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
    if args.map_months > 0:
        start = max(start, end - pd.DateOffset(months=int(args.map_months)))
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
    ap.add_argument("--timeframe", default="2h")
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
        "--max-loss-per-segment",
        type=float,
        default=defaults.get("max_loss_per_segment", 0.01),
    )
    ap.add_argument(
        "--min-segment-bars", type=int, default=defaults.get("min_segment_bars", 6)
    )
    ap.add_argument(
        "--max-segment-bars", type=int, default=defaults.get("max_segment_bars", 120)
    )
    ap.add_argument("--fee-bps", type=float, default=defaults.get("fee_bps", 4.0))
    ap.add_argument(
        "--exclude-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("exclude_box", True),
    )
    ap.add_argument("--out-dir", default="results/dual_add_trend_diagnostic")
    ap.add_argument("--map-symbols", default="BTCUSDT")
    ap.add_argument("--map-months", type=int, default=12)
    ap.add_argument("--continuous-map-symbols", default="")
    ap.add_argument("--continuous-map-months", type=int, default=0)
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades, segments = run(args)
    summary = summarize(trades, segments)
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
            map_months=args.continuous_map_months,
            trades=trades,
            segments=segments,
            title="Dual Add Trend Continuous Trading Map",
        )
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print("\n=== Dual Add Summary ===")
    print(summary.to_string(index=False) if not summary.empty else "(empty)")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
