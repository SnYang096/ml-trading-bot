"""Standalone chop-grid backtest report.

This is intentionally separate from the generic event_backtest pipeline because
grid trading owns multi-level inventory, while the current event path is built
around single TradeIntent positions.

Outputs:
  - grid_trades.csv: completed grid cycles and forced exits
  - grid_segments.csv: one row per chop segment
  - equity_curve.csv: cumulative capital-normalized PnL by exit time
  - report.html: compact research report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    _hysteresis_segments,
    _pnl_long,
    _pnl_short,
    build_features,
)
from scripts.capital_report import write_capital_report_from_trades  # noqa: E402
from scripts.diagnose_crf_edge import (  # noqa: E402
    _load_symbol_1m,
    _resample_ohlcv,
)
from scripts.multi_leg_trading_map import write_continuous_trading_map  # noqa: E402
from src.time_series_model.grid.chop_grid_engine import (  # noqa: E402
    ChopGridEngine,
    GridEngineConfig,
    hysteresis_segments,
)


DEFAULT_GRID_CONFIG = PROJECT_ROOT / "config/strategies/chop_grid/grid.yaml"


def _load_grid_defaults(path: Path) -> dict:
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    regime = cfg.get("regime", {}) or {}
    grid = cfg.get("grid", {}) or {}
    spacing = grid.get("spacing", {}) or {}
    risk = cfg.get("risk", {}) or {}
    return {
        "box_window": int(regime.get("box_window", 120)),
        "chop_min": float(regime.get("entry_chop_min", 0.40)),
        "exit_chop_min": float(regime.get("exit_chop_below", 0.25)),
        "grid_atr_mult": float(spacing.get("atr_mult", 0.50)),
        "grid_pct": float(spacing.get("min_pct", 0.004)),
        "max_levels": int(grid.get("max_levels_per_side", 3)),
        "min_segment_bars": int(risk.get("min_segment_bars", 6)),
        "max_segment_bars": int(risk.get("max_segment_bars", 120)),
        "fee_bps": float(risk.get("fee_bps", 4.0)),
        "maker_fee_bps": float(risk.get("maker_fee_bps", risk.get("fee_bps", 4.0))),
        "taker_fee_bps": float(risk.get("taker_fee_bps", risk.get("fee_bps", 4.0))),
        "forced_exit_slippage_bps": float(risk.get("forced_exit_slippage_bps", 0.0)),
        "funding_cost_bps_per_8h": float(risk.get("funding_cost_bps_per_8h", 0.0)),
        "exclude_box": bool(regime.get("exclude_box_prefilter", True)),
        "max_loss_per_grid": risk.get("max_loss_per_grid", 0.03),
        "max_open_levels_total": risk.get("max_open_levels_total", 6),
    }


def _fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float((s - s.cummax()).min())


def _sharpe(returns: pd.Series, periods_per_year: float) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def simulate_segment_detailed(
    seg: pd.DataFrame,
    *,
    cfg: GridConfig,
    symbol: str,
    regime: str,
    segment_id: str,
) -> Tuple[List[dict], dict]:
    if seg.empty:
        return [], {"status": "empty"}

    center = float(seg["close"].iloc[0])
    atr = float(seg["atr14"].iloc[0])
    if not np.isfinite(center + atr) or center <= 0 or atr <= 0:
        return [], {"status": "invalid"}

    spacing = max(cfg.grid_atr_mult * atr, cfg.grid_pct * center)
    fee = cfg.fee_bps / 10000.0
    capital_units = max(1, 2 * cfg.max_levels)
    long_levels = [center - spacing * k for k in range(1, cfg.max_levels + 1)]
    short_levels = [center + spacing * k for k in range(1, cfg.max_levels + 1)]

    open_longs: Dict[int, Tuple[float, pd.Timestamp, int]] = {}
    open_shorts: Dict[int, Tuple[float, pd.Timestamp, int]] = {}
    trades: List[dict] = []
    pnl_path = []
    max_open = 0

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
        pnl_pct = (
            _pnl_long(entry_price, exit_price, fee)
            if side == "LONG"
            else _pnl_short(entry_price, exit_price, fee)
        )
        trades.append(
            {
                "symbol": symbol,
                "regime": regime,
                "segment_id": segment_id,
                "side": side,
                "level": level,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": pnl_pct,
                "r_equiv": pnl_pct / (spacing / center),
                "pnl_per_capital": pnl_pct / capital_units,
                "r_equiv_per_capital": (pnl_pct / (spacing / center)) / capital_units,
                "spacing_pct": spacing / center,
                "spacing_atr": spacing / atr,
            }
        )

    for bar_i, (ts, row) in enumerate(seg.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Target exits first for positions opened on prior bars.
        for level_i, (entry, entry_ts, fill_bar) in list(open_longs.items()):
            target = entry + spacing
            if bar_i > fill_bar and high >= target:
                _record(
                    side="LONG",
                    level=level_i + 1,
                    entry_price=entry,
                    entry_time=entry_ts,
                    exit_price=target,
                    exit_time=ts,
                    exit_reason="grid_tp",
                )
                del open_longs[level_i]
        for level_i, (entry, entry_ts, fill_bar) in list(open_shorts.items()):
            target = entry - spacing
            if bar_i > fill_bar and low <= target:
                _record(
                    side="SHORT",
                    level=level_i + 1,
                    entry_price=entry,
                    entry_time=entry_ts,
                    exit_price=target,
                    exit_time=ts,
                    exit_reason="grid_tp",
                )
                del open_shorts[level_i]

        # New fills.
        for level_i, px in enumerate(long_levels):
            if level_i not in open_longs and low <= px:
                open_longs[level_i] = (px, ts, bar_i)
        for level_i, px in enumerate(short_levels):
            if level_i not in open_shorts and high >= px:
                open_shorts[level_i] = (px, ts, bar_i)

        realized = sum(t["pnl_pct"] for t in trades if t["segment_id"] == segment_id)
        mtm = realized
        for entry, _, _ in open_longs.values():
            mtm += (close - entry) / entry - fee
        for entry, _, _ in open_shorts.values():
            mtm += (entry - close) / entry - fee
        pnl_path.append(mtm / capital_units)
        max_open = max(max_open, len(open_longs) + len(open_shorts))

    exit_ts = seg.index[-1]
    exit_close = float(seg["close"].iloc[-1])
    forced = len(open_longs) + len(open_shorts)
    for level_i, (entry, entry_ts, _) in list(open_longs.items()):
        _record(
            side="LONG",
            level=level_i + 1,
            entry_price=entry,
            entry_time=entry_ts,
            exit_price=exit_close,
            exit_time=exit_ts,
            exit_reason="regime_exit",
        )
    for level_i, (entry, entry_ts, _) in list(open_shorts.items()):
        _record(
            side="SHORT",
            level=level_i + 1,
            entry_price=entry,
            entry_time=entry_ts,
            exit_price=exit_close,
            exit_time=exit_ts,
            exit_reason="regime_exit",
        )

    seg_trades = [t for t in trades if t["segment_id"] == segment_id]
    total_pnl = sum(t["pnl_per_capital"] for t in seg_trades)
    max_drawdown = 0.0
    if pnl_path:
        arr = np.asarray(pnl_path, dtype=float)
        max_drawdown = float((arr - np.maximum.accumulate(arr)).min())
    summary = {
        "symbol": symbol,
        "regime": regime,
        "segment_id": segment_id,
        "start": seg.index[0],
        "end": seg.index[-1],
        "bars": len(seg),
        "entry_chop": float(seg["semantic_chop"].iloc[0]),
        "median_chop": float(seg["semantic_chop"].median()),
        "entry_box_prefilter": bool(seg["box_prefilter"].iloc[0]),
        "center": center,
        "spacing_pct": spacing / center,
        "spacing_atr": spacing / atr,
        "trades": len(seg_trades),
        "grid_tp": sum(1 for t in seg_trades if t["exit_reason"] == "grid_tp"),
        "forced_exits": forced,
        "max_open_levels": max_open,
        "pnl_per_capital": total_pnl,
        "max_drawdown": max_drawdown,
    }
    return seg_trades, summary


def run_backtest(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = GridConfig(
        box_window=args.box_window,
        chop_min=args.chop_min,
        exit_chop_min=args.exit_chop_min,
        min_segment_bars=args.min_segment_bars,
        max_segment_bars=args.max_segment_bars,
        grid_atr_mult=args.grid_atr_mult,
        grid_pct=args.grid_pct,
        max_levels=args.max_levels,
        fee_bps=args.fee_bps,
    )
    engine_cfg = GridEngineConfig(
        box_window=args.box_window,
        entry_chop_min=args.chop_min,
        exit_chop_below=args.exit_chop_min,
        min_segment_bars=args.min_segment_bars,
        max_segment_bars=args.max_segment_bars,
        grid_atr_mult=args.grid_atr_mult,
        grid_min_pct=args.grid_pct,
        max_levels_per_side=args.max_levels,
        fee_bps=args.fee_bps + args.slippage_bps,
        maker_fee_bps=args.maker_fee_bps,
        taker_fee_bps=args.taker_fee_bps,
        forced_exit_slippage_bps=args.forced_exit_slippage_bps + args.slippage_bps,
        funding_cost_bps_per_8h=args.funding_cost_bps_per_8h,
        max_loss_per_grid=args.max_loss_per_grid,
        max_open_levels_total=args.max_open_levels_total,
    )
    engine = ChopGridEngine(engine_cfg)
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
            print(f"skip {symbol}: no data")
            continue
        bars = _resample_ohlcv(raw, args.timeframe)
        df = build_features(symbol, bars, cfg)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            continue

        entry_mask = df["semantic_chop"] >= cfg.chop_min
        hold_mask = df["semantic_chop"] >= cfg.exit_chop_min
        if args.exclude_box:
            entry_mask &= ~df["box_prefilter"]
            hold_mask &= ~df["box_prefilter"]
            regime = "chop_not_box"
        else:
            regime = "semantic_chop"
        segs = hysteresis_segments(
            entry_mask,
            hold_mask,
            min_len=cfg.min_segment_bars,
            max_len=cfg.max_segment_bars,
        )
        print(f"{symbol}: segments={len(segs)}, entry_rate={entry_mask.mean():.1%}")
        for seq, (s, e) in enumerate(segs, start=1):
            seg_id = f"{symbol}_{seq:04d}_{df.index[s].strftime('%Y%m%d%H')}"
            result = engine.simulate_segment(
                df.iloc[s : e + 1],
                symbol=symbol,
                regime=regime,
                segment_id=seg_id,
            )
            all_trades.extend(t.to_dict() for t in result.trades)
            all_segments.append(result.summary)

    trades_df = pd.DataFrame(all_trades)
    segments_df = pd.DataFrame(all_segments)
    if trades_df.empty:
        equity_df = pd.DataFrame(
            columns=["exit_time", "pnl_per_capital", "cum_pnl_per_capital", "drawdown"]
        )
    else:
        equity_df = trades_df.sort_values("exit_time").copy()
        equity_df["cum_pnl_per_capital"] = equity_df["pnl_per_capital"].cumsum()
        equity_df["drawdown"] = (
            equity_df["cum_pnl_per_capital"] - equity_df["cum_pnl_per_capital"].cummax()
        )
        equity_df = equity_df[
            ["exit_time", "pnl_per_capital", "cum_pnl_per_capital", "drawdown"]
        ]
    return trades_df, segments_df, equity_df


def _summary_tables(
    trades: pd.DataFrame, segments: pd.DataFrame, equity: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily = pd.Series(dtype=float)
    if not trades.empty:
        tmp = trades.copy()
        tmp["exit_time"] = pd.to_datetime(tmp["exit_time"], utc=True)
        daily = tmp.set_index("exit_time")["pnl_per_capital"].resample("1D").sum()
    trade_summary = pd.DataFrame(
        [
            {
                "trades": len(trades),
                "win_rate": (trades["pnl_pct"] > 0).mean(),
                "sum_pnl_per_capital": trades["pnl_per_capital"].sum(),
                "return_pct": trades["pnl_per_capital"].sum() * 100.0,
                "sum_r_equiv_per_capital": trades["r_equiv_per_capital"].sum(),
                "mean_trade_pnl_pct": trades["pnl_pct"].mean(),
                "median_trade_pnl_pct": trades["pnl_pct"].median(),
                "tp_rate": (trades["exit_reason"] == "grid_tp").mean(),
                "forced_rate": (trades["exit_reason"] == "regime_exit").mean(),
                "daily_sharpe": _sharpe(daily, 365.0),
                "trade_sharpe": _sharpe(trades["pnl_per_capital"], 1.0),
                "max_drawdown": (
                    _max_drawdown(equity["cum_pnl_per_capital"])
                    if not equity.empty
                    else 0.0
                ),
                "gross_pnl_pct_sum": (
                    trades["gross_pnl_pct"].sum()
                    if "gross_pnl_pct" in trades
                    else np.nan
                ),
                "fee_bps_charged_sum": (
                    trades["fee_bps_charged"].sum()
                    if "fee_bps_charged" in trades
                    else np.nan
                ),
                "slippage_bps_charged_sum": (
                    trades["slippage_bps_charged"].sum()
                    if "slippage_bps_charged" in trades
                    else np.nan
                ),
                "funding_bps_charged_sum": (
                    trades["funding_bps_charged"].sum()
                    if "funding_bps_charged" in trades
                    else np.nan
                ),
            }
        ]
    )
    if segments.empty:
        segment_summary = pd.DataFrame()
    else:
        segment_summary = pd.DataFrame(
            [
                {
                    "segments": len(segments),
                    "segment_win_rate": (segments["pnl_per_capital"] > 0).mean(),
                    "sum_segment_pnl": segments["pnl_per_capital"].sum(),
                    "median_segment_pnl": segments["pnl_per_capital"].median(),
                    "worst_segment": segments["pnl_per_capital"].min(),
                    "median_drawdown": segments["max_drawdown"].median(),
                    "median_bars": segments["bars"].median(),
                    "median_trades_per_segment": segments["trades"].median(),
                    "median_forced_exits": segments["forced_exits"].median(),
                }
            ]
        )
    return trade_summary, segment_summary


def build_metrics(
    trades: pd.DataFrame, segments: pd.DataFrame, equity: pd.DataFrame
) -> dict:
    trade_summary, segment_summary = _summary_tables(trades, segments, equity)
    metrics: dict = {
        "trade_summary": (
            trade_summary.to_dict(orient="records")[0]
            if not trade_summary.empty
            else {}
        ),
        "segment_summary": (
            segment_summary.to_dict(orient="records")[0]
            if not segment_summary.empty
            else {}
        ),
        "by_symbol": [],
        "by_year": [],
        "by_month": [],
        "forced_exit_loss_attribution": {},
    }
    if not segments.empty:
        by_symbol = (
            segments.groupby("symbol")
            .agg(
                segments=("segment_id", "count"),
                pnl_per_capital=("pnl_per_capital", "sum"),
                win_rate=("pnl_per_capital", lambda s: (s > 0).mean()),
                worst_segment=("pnl_per_capital", "min"),
                forced_exit_pnl=("forced_exit_pnl", "sum"),
            )
            .reset_index()
        )
        metrics["by_symbol"] = by_symbol.to_dict(orient="records")
        tmp = segments.copy()
        tmp["start"] = pd.to_datetime(tmp["start"], utc=True)
        tmp["year"] = tmp["start"].dt.year.astype(str)
        by_year = (
            tmp.groupby("year")
            .agg(
                segments=("segment_id", "count"),
                pnl_per_capital=("pnl_per_capital", "sum"),
                win_rate=("pnl_per_capital", lambda s: (s > 0).mean()),
                worst_segment=("pnl_per_capital", "min"),
                forced_exit_pnl=("forced_exit_pnl", "sum"),
            )
            .reset_index()
        )
        metrics["by_year"] = by_year.to_dict(orient="records")
        tmp["month"] = tmp["start"].dt.strftime("%Y-%m")
        by_month = (
            tmp.groupby("month")
            .agg(
                segments=("segment_id", "count"),
                pnl_per_capital=("pnl_per_capital", "sum"),
                win_rate=("pnl_per_capital", lambda s: (s > 0).mean()),
                worst_segment=("pnl_per_capital", "min"),
                forced_exit_pnl=("forced_exit_pnl", "sum"),
            )
            .reset_index()
        )
        metrics["by_month"] = by_month.to_dict(orient="records")
    if not trades.empty:
        forced = trades[trades["exit_reason"].isin(["regime_exit", "risk_exit"])]
        metrics["forced_exit_loss_attribution"] = {
            "count": int(len(forced)),
            "pnl_per_capital": float(forced["pnl_per_capital"].sum()),
            "negative_pnl_per_capital": float(
                forced.loc[forced["pnl_per_capital"] < 0, "pnl_per_capital"].sum()
            ),
            "risk_exit_count": int((trades["exit_reason"] == "risk_exit").sum()),
        }
        if {"fee_bps_charged", "slippage_bps_charged", "funding_bps_charged"}.issubset(
            trades.columns
        ):
            metrics["cost_attribution"] = {
                "total_fee_bps_charged": float(trades["fee_bps_charged"].sum()),
                "total_slippage_bps_charged": float(
                    trades["slippage_bps_charged"].sum()
                ),
                "total_funding_bps_charged": float(trades["funding_bps_charged"].sum()),
                "grid_tp_fee_bps_charged": float(
                    trades.loc[
                        trades["exit_reason"] == "grid_tp", "fee_bps_charged"
                    ].sum()
                ),
                "forced_fee_bps_charged": float(forced["fee_bps_charged"].sum()),
                "forced_slippage_bps_charged": float(
                    forced["slippage_bps_charged"].sum()
                ),
                "forced_funding_bps_charged": float(
                    forced["funding_bps_charged"].sum()
                ),
            }
    return metrics


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    trades: pd.DataFrame,
    segments: pd.DataFrame,
    equity: pd.DataFrame,
) -> None:
    trade_summary, segment_summary = _summary_tables(trades, segments, equity)
    by_symbol = (
        segments.groupby("symbol")
        .agg(
            segments=("segment_id", "count"),
            pnl=("pnl_per_capital", "sum"),
            win_rate=("pnl_per_capital", lambda s: (s > 0).mean()),
            worst=("pnl_per_capital", "min"),
            forced_exit_pnl=("forced_exit_pnl", "sum"),
        )
        .reset_index()
        if not segments.empty
        else pd.DataFrame()
    )
    if not segments.empty:
        by_year_src = segments.copy()
        by_year_src["start"] = pd.to_datetime(by_year_src["start"], utc=True)
        by_year_src["year"] = by_year_src["start"].dt.year.astype(str)
        by_year_src["month"] = by_year_src["start"].dt.strftime("%Y-%m")
        by_year = (
            by_year_src.groupby("year")
            .agg(
                segments=("segment_id", "count"),
                pnl=("pnl_per_capital", "sum"),
                win_rate=("pnl_per_capital", lambda s: (s > 0).mean()),
                worst=("pnl_per_capital", "min"),
                forced_exit_pnl=("forced_exit_pnl", "sum"),
            )
            .reset_index()
        )
        by_month = (
            by_year_src.groupby("month")
            .agg(
                segments=("segment_id", "count"),
                pnl=("pnl_per_capital", "sum"),
                win_rate=("pnl_per_capital", lambda s: (s > 0).mean()),
                worst=("pnl_per_capital", "min"),
                forced_exit_pnl=("forced_exit_pnl", "sum"),
            )
            .reset_index()
        )
    else:
        by_year = pd.DataFrame()
        by_month = pd.DataFrame()
    exit_reasons = (
        trades["exit_reason"]
        .value_counts()
        .rename_axis("exit_reason")
        .reset_index(name="count")
        if not trades.empty
        else pd.DataFrame()
    )
    risk_table = (
        trade_summary[
            [
                "sum_pnl_per_capital",
                "return_pct",
                "sum_r_equiv_per_capital",
                "daily_sharpe",
                "trade_sharpe",
                "max_drawdown",
            ]
        ]
        if not trade_summary.empty
        else pd.DataFrame()
    )

    html = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>Chop Grid Backtest Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin: 12px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 9px; text-align: right; }}
    th {{ background: #f3f4f6; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .note {{ color: #555; max-width: 900px; }}
  </style>
</head>
<body>
  <h1>Chop Grid Backtest Report</h1>
  <p class="note">
    Entry: semantic_chop >= {args.chop_min}; exit all inventory when semantic_chop < {args.exit_chop_min}.
    Grid spacing: max({args.grid_atr_mult} ATR, {args.grid_pct:.2%}); levels per side: {args.max_levels};
    fee: {args.fee_bps} bps; slippage sensitivity: {args.slippage_bps} bps. Exclude box: {args.exclude_box}.
    Realistic costs: maker={args.maker_fee_bps} bps, taker={args.taker_fee_bps} bps,
    forced-exit slippage={args.forced_exit_slippage_bps} bps,
    funding cost={args.funding_cost_bps_per_8h} bps/8h.
  </p>
  <h2>Risk Metrics</h2>
  {risk_table.to_html(index=False, float_format=lambda x: _fmt(x, 6))}
  <h2>Trade Summary</h2>
  {trade_summary.to_html(index=False, float_format=lambda x: _fmt(x, 6))}
  <h2>Segment Summary</h2>
  {segment_summary.to_html(index=False, float_format=lambda x: _fmt(x, 6))}
  <h2>By Symbol</h2>
  {by_symbol.to_html(index=False, float_format=lambda x: _fmt(x, 6))}
  <h2>By Year</h2>
  {by_year.to_html(index=False, float_format=lambda x: _fmt(x, 6))}
  <h2>By Month</h2>
  {by_month.to_html(index=False, float_format=lambda x: _fmt(x, 6))}
  <h2>Exit Reasons</h2>
  {exit_reasons.to_html(index=False)}
  <h2>Recent Equity Points</h2>
  {equity.tail(20).to_html(index=False, float_format=lambda x: _fmt(x, 6))}
</body>
</html>
"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def write_trading_maps(
    out_dir: Path,
    args: argparse.Namespace,
    trades: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    """Write per-symbol visual maps for grid segments and completed cycles."""
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
            title=f"Chop Grid Trading Map - {symbol}",
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
                p.add_layout(
                    BoxAnnotation(
                        left=row["start"],
                        right=row["end"],
                        fill_color="#22c55e",
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
                            ("level", "@level"),
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
        out_path = out_dir / f"trading_map_grid_{symbol}.html"
        output_file(out_path, title=f"Chop Grid Trading Map - {symbol}")
        save(p)
        print(f"Saved trading map -> {out_path}")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_GRID_CONFIG))
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    defaults = _load_grid_defaults(config_path)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--data-dir", default="data/parquet_data")
    parser.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--timeframe", default="2h")
    parser.add_argument(
        "--box-window",
        type=int,
        default=defaults.get("box_window", 120),
        choices=[60, 120, 240],
    )
    parser.add_argument(
        "--chop-min", type=float, default=defaults.get("chop_min", 0.40)
    )
    parser.add_argument(
        "--exit-chop-min", type=float, default=defaults.get("exit_chop_min", 0.25)
    )
    parser.add_argument(
        "--grid-atr-mult", type=float, default=defaults.get("grid_atr_mult", 0.50)
    )
    parser.add_argument(
        "--grid-pct", type=float, default=defaults.get("grid_pct", 0.004)
    )
    parser.add_argument("--max-levels", type=int, default=defaults.get("max_levels", 3))
    parser.add_argument(
        "--min-segment-bars", type=int, default=defaults.get("min_segment_bars", 6)
    )
    parser.add_argument(
        "--max-segment-bars", type=int, default=defaults.get("max_segment_bars", 120)
    )
    parser.add_argument("--fee-bps", type=float, default=defaults.get("fee_bps", 4.0))
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--maker-fee-bps", type=float, default=defaults.get("maker_fee_bps", 4.0)
    )
    parser.add_argument(
        "--taker-fee-bps", type=float, default=defaults.get("taker_fee_bps", 4.0)
    )
    parser.add_argument(
        "--forced-exit-slippage-bps",
        type=float,
        default=defaults.get("forced_exit_slippage_bps", 0.0),
    )
    parser.add_argument(
        "--funding-cost-bps-per-8h",
        type=float,
        default=defaults.get("funding_cost_bps_per_8h", 0.0),
    )
    parser.add_argument(
        "--exclude-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("exclude_box", True),
    )
    parser.add_argument(
        "--max-loss-per-grid",
        type=float,
        default=defaults.get("max_loss_per_grid", 0.03),
    )
    parser.add_argument(
        "--max-open-levels-total",
        type=int,
        default=defaults.get("max_open_levels_total", 6),
    )
    parser.add_argument("--out-dir", default="results/chop_grid/backtest")
    parser.add_argument("--map-symbols", default="BTCUSDT")
    parser.add_argument("--map-months", type=int, default=12)
    parser.add_argument("--continuous-map-symbols", default="")
    parser.add_argument("--continuous-map-months", type=int, default=0)
    parser.add_argument("--no-maps", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades, segments, equity = run_backtest(args)
    trades.to_csv(out_dir / "grid_trades.csv", index=False)
    segments.to_csv(out_dir / "grid_segments.csv", index=False)
    equity.to_csv(out_dir / "equity_curve.csv", index=False)
    metrics = build_metrics(trades, segments, equity)
    (out_dir / "metrics.json").write_text(
        json.dumps({"args": vars(args), "metrics": metrics}, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(out_dir, args, trades, segments, equity)
    write_capital_report_from_trades(
        trades_path=out_dir / "grid_trades.csv",
        out_dir=out_dir,
        unit="capital_normalized",
        title="Chop Grid Capital Report",
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
            title="Chop Grid Continuous Trading Map",
        )

    trade_summary, segment_summary = _summary_tables(trades, segments, equity)
    print("\n=== Trade Summary ===")
    print(
        trade_summary.to_string(index=False) if not trade_summary.empty else "(empty)"
    )
    print("\n=== Segment Summary ===")
    print(
        segment_summary.to_string(index=False)
        if not segment_summary.empty
        else "(empty)"
    )
    print(f"\nSaved report -> {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
