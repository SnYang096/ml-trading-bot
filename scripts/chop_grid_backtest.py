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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    _hysteresis_segments,
    _pnl_long,
    _pnl_short,
    build_features,
    merge_chop_grid_yaml,
    regime_chop_column,
    regime_chop_series,
    resolve_optional_repo_path,
    resolve_prefilter_rules,
)
from scripts.capital_report import write_capital_report_from_trades  # noqa: E402
from scripts.pipeline.multileg_portfolio_metrics import (  # noqa: E402
    build_portfolio_equity_curve,
    portfolio_metrics_from_trades,
    sharpe_from_returns,
)
from scripts.diagnose_crf_edge import (  # noqa: E402
    _load_symbol_1m,
    _resample_ohlcv,
)
from scripts.multi_leg_trading_map import write_continuous_trading_map  # noqa: E402
from scripts.pipeline.multileg_prefilter_rules import (  # noqa: E402
    apply_prefilter_rules,
)
from src.live_data_stream.constitution_config import (  # noqa: E402
    load_multi_leg_backtest_risk_context,
)
from src.time_series_model.grid.chop_grid_engine import (  # noqa: E402
    ChopGridEngine,
    GridEngineConfig,
    hysteresis_segments,
)
from src.time_series_model.grid.agg100ms_replay import (  # noqa: E402
    load_segment_100ms_bars,
)
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    segment_execution_bounds,
    slice_execution_window,
    timeframe_to_timedelta,
)


DEFAULT_GRID_CONFIG = (
    PROJECT_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
)


def _load_grid_defaults(path: Path) -> dict:
    return merge_chop_grid_yaml(path)


def _parse_max_replenish_cli(value: str) -> int | None:
    raw = str(value).strip().lower()
    if raw in {"", "null", "none", "unlimited", "inf"}:
        return None
    return int(value)


def _fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float((s - s.cummax()).min())


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
        "entry_chop": float(regime_chop_series(seg, cfg).iloc[0]),
        "median_chop": float(regime_chop_series(seg, cfg).median()),
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


def collect_chop_grid_trades_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    df_exec: pd.DataFrame | None,
    sig_delta: pd.Timedelta | None,
    cfg: GridConfig,
    engine: ChopGridEngine,
    *,
    block_stable_box: bool = False,
    exec_timeframe: str = "1min",
    agg_data_dir: Path | None = None,
    parquet_data_dir: Path | None = None,
    account_risk_tracker=None,
    unit_notional_usdt: float = 0.0,
) -> Tuple[List[dict], List[dict], int, float]:
    """Simulate chop grid for one symbol given pre-built feature ``df`` (signal timeframe).

    Returns ``(trades, summaries, n_segments, entry_mask_mean)``.
    """
    chop_s = regime_chop_series(df, cfg)
    chop_col = regime_chop_column(cfg)
    entry_mask = chop_s >= cfg.chop_min
    hold_mask = chop_s >= cfg.exit_chop_min
    rule_mask = apply_prefilter_rules(
        df,
        list(cfg.prefilter_rules),
        feature_aliases={
            "atr": "atr14",
            "bpc_semantic_chop": "semantic_chop",
            "bpc_semantic_chop_ts_q": "semantic_chop_ts_q",
        },
    )
    entry_mask &= rule_mask
    hold_mask &= rule_mask
    if block_stable_box:
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
    entry_rate = float(entry_mask.mean()) if len(entry_mask) else 0.0
    trades_out: List[dict] = []
    summaries_out: List[dict] = []
    exec_tf = str(exec_timeframe or "1min").strip().lower()
    for seq, (s, e) in enumerate(segs, start=1):
        seg_id = f"{symbol}_{seq:04d}_{df.index[s].strftime('%Y%m%d%H%M')}"
        anchor_c = float(df.iloc[s]["close"])
        anchor_a = float(df.iloc[s]["atr14"])
        if sig_delta is not None:
            if exec_tf in {"100ms"}:
                t_enter, t_exit = segment_execution_bounds(df.index, s, e, sig_delta)
                seg_slice = load_segment_100ms_bars(
                    symbol=symbol,
                    agg_data_dir=agg_data_dir,
                    parquet_data_dir=parquet_data_dir,
                    t_enter=t_enter,
                    t_exit=t_exit,
                )
                seg_slice = merge_signal_features_onto_execution_bars(
                    seg_slice, df, signal_bar_delta=sig_delta
                )
            elif df_exec is not None:
                seg_slice = slice_execution_window(df_exec, df.index, s, e, sig_delta)
            else:
                result = engine.simulate_segment(
                    df.iloc[s : e + 1],
                    symbol=symbol,
                    regime=regime,
                    segment_id=seg_id,
                    regime_chop_col=chop_col,
                    account_risk_tracker=account_risk_tracker,
                    unit_notional_usdt=unit_notional_usdt,
                )
                trades_out.extend(t.to_dict() for t in result.trades)
                summaries_out.append(result.summary)
                continue
            result = engine.simulate_segment(
                seg_slice,
                symbol=symbol,
                regime=regime,
                segment_id=seg_id,
                anchor_close=anchor_c,
                anchor_atr=anchor_a,
                regime_chop_col=chop_col,
                account_risk_tracker=account_risk_tracker,
                unit_notional_usdt=unit_notional_usdt,
            )
        else:
            result = engine.simulate_segment(
                df.iloc[s : e + 1],
                symbol=symbol,
                regime=regime,
                segment_id=seg_id,
                regime_chop_col=chop_col,
                account_risk_tracker=account_risk_tracker,
                unit_notional_usdt=unit_notional_usdt,
            )
        trades_out.extend(t.to_dict() for t in result.trades)
        summaries_out.append(result.summary)
    return trades_out, summaries_out, len(segs), entry_rate


def run_backtest(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg_path = Path(str(args.config))
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    defaults = _load_grid_defaults(cfg_path)
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
        chop_signal=args.chop_signal,
        chop_ts_window=args.chop_ts_window,
        chop_ts_min_periods=args.chop_ts_min_periods,
        compute_semantic_chop_ts_q=getattr(args, "compute_chop_ts_q", None),
        stability_min=args.stability_min,
        width_min=args.width_min,
        width_max=args.width_max,
        touches_min=args.touches_min,
        feature_store_dir=resolve_optional_repo_path(
            getattr(args, "feature_store_dir", None)
        ),
        feature_store_layer=(
            str(args.feature_store_layer).strip()
            if getattr(args, "feature_store_layer", None)
            else None
        )
        or None,
        feature_store_timeframe=(
            str(args.feature_store_timeframe).strip()
            if getattr(args, "feature_store_timeframe", None)
            else None
        )
        or None,
        prefilter_rules=resolve_prefilter_rules(
            defaults,
            box_pos_min=getattr(args, "box_pos_min", None),
            box_pos_max=getattr(args, "box_pos_max", None),
        ),
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
        same_bar_entry_exit=bool(args.same_bar_entry_exit),
        max_replenish_per_level_per_segment=getattr(
            args, "max_replenish_per_level", None
        ),
    )
    engine = ChopGridEngine(engine_cfg)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=args.warmup_days)
    data_dir = Path(args.data_dir)

    all_trades: List[dict] = []
    all_segments: List[dict] = []
    risk_tracker, unit_notional = load_multi_leg_backtest_risk_context(
        initial_capital=float(getattr(args, "initial_capital", 10_000.0) or 10_000.0)
    )

    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            print(f"skip {symbol}: no data")
            continue
        bars_signal = _resample_ohlcv(raw, args.timeframe)
        df = build_features(symbol, bars_signal, cfg, bars_timeframe=args.timeframe)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            continue

        exec_tf = str(
            args.execution_timeframe or defaults.get("execution_timeframe", "1min")
        )
        sig_delta = timeframe_to_timedelta(args.timeframe)
        agg_dir = resolve_optional_repo_path(
            getattr(args, "agg_data_dir", None) or defaults.get("agg_data_dir")
        )
        if exec_tf.lower() == "100ms":
            df_exec = None
        else:
            bars_exec = _resample_ohlcv(raw, exec_tf)
            df_exec = merge_signal_features_onto_execution_bars(
                bars_exec, df, signal_bar_delta=sig_delta
            )

        tlist, slist, n_seg, entry_rate = collect_chop_grid_trades_for_symbol(
            symbol,
            df,
            df_exec,
            sig_delta,
            cfg,
            engine,
            block_stable_box=bool(
                getattr(args, "block_stable_box", None)
                if getattr(args, "block_stable_box", None) is not None
                else defaults.get("block_stable_box", False)
            ),
            exec_timeframe=exec_tf,
            agg_data_dir=Path(agg_dir) if agg_dir else None,
            parquet_data_dir=data_dir,
            account_risk_tracker=risk_tracker,
            unit_notional_usdt=unit_notional,
        )
        print(f"{symbol}: segments={n_seg}, entry_rate={entry_rate:.1%}")
        all_trades.extend(tlist)
        all_segments.extend(slist)

    trades_df = pd.DataFrame(all_trades)
    segments_df = pd.DataFrame(all_segments)
    equity_df = build_portfolio_equity_curve(trades_df)
    return trades_df, segments_df, equity_df


def _summary_tables(
    trades: pd.DataFrame, segments: pd.DataFrame, equity: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg = portfolio_metrics_from_trades(trades)
    trade_summary = pd.DataFrame(
        [
            {
                "trades": len(trades),
                "win_rate": (trades["pnl_pct"] > 0).mean(),
                "n_symbols": agg["n_symbols"],
                "sum_pnl_per_capital": agg["sum_pnl_per_capital"],
                "sum_pnl_per_capital_pooled": agg["sum_pnl_per_capital_pooled"],
                "return_pct": agg["return_pct"],
                "return_pct_timeline": agg["return_pct_timeline"],
                "return_pct_eq_mean": agg["return_pct_eq_mean"],
                "return_pct_pooled": agg["return_pct_pooled"],
                "max_drawdown_portfolio": agg["max_drawdown_portfolio"],
                "sum_r_equiv_per_capital": trades["r_equiv_per_capital"].sum(),
                "mean_trade_pnl_pct": trades["pnl_pct"].mean(),
                "median_trade_pnl_pct": trades["pnl_pct"].median(),
                "tp_rate": (trades["exit_reason"] == "grid_tp").mean(),
                "forced_rate": (trades["exit_reason"] == "regime_exit").mean(),
                "daily_sharpe": agg["daily_sharpe"],
                "trade_sharpe": sharpe_from_returns(trades["pnl_per_capital"], 1.0),
                "max_drawdown": (
                    float(equity["drawdown"].min())
                    if not equity.empty and "drawdown" in equity.columns
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
                    "median_spacing_pct": segments["spacing_pct"].median(),
                    "median_grid_full_span_pct": (
                        segments["grid_full_span_pct"].median()
                        if "grid_full_span_pct" in segments
                        else np.nan
                    ),
                    "median_segment_range_pct": (
                        segments["segment_range_pct"].median()
                        if "segment_range_pct" in segments
                        else np.nan
                    ),
                    "median_close_std_pct": (
                        segments["close_std_pct"].median()
                        if "close_std_pct" in segments
                        else np.nan
                    ),
                    "median_grid_full_span_to_range": (
                        segments["grid_full_span_to_range"].median()
                        if "grid_full_span_to_range" in segments
                        else np.nan
                    ),
                    "median_grid_per_side_span_to_1std": (
                        segments["grid_per_side_span_to_1std"].median()
                        if "grid_per_side_span_to_1std" in segments
                        else np.nan
                    ),
                }
            ]
        )
    return trade_summary, segment_summary


def summarize_dual_add_aligned(
    trades: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    """One-row aggregate with the same columns as scripts/diagnose_dual_add_trend.summarize.

    Semantics (chop grid vs dual-add):
    - ``risk_stop_rate``: fraction of segments with at least one ``risk_exit`` trade.
    - ``max_gross_units`` / ``max_abs_net_units``: grid uses ``max_open_levels`` on
      inventory; we expose max open levels as ``max_gross_units`` and leave net as NaN
      (dual-add tracks discrete leg units).
    - ``loser_timeout_rate``: always 0 (no loser_timeout exit in grid engine).
    - ``tp_rate`` / ``forced_rate``: ``grid_tp`` vs all other exit reasons (same pattern
      as dual-add ``tp`` vs non-``tp``).
    """
    if trades.empty or segments.empty:
        return pd.DataFrame()
    from scripts.pipeline.multileg_portfolio_metrics import dual_add_summary_fields

    row = dual_add_summary_fields(trades, segments)
    risk_col = segments["risk_exits"] if "risk_exits" in segments.columns else None
    risk_stop_rate = float((risk_col > 0).mean()) if risk_col is not None else 0.0
    max_open = (
        segments["max_open_levels"] if "max_open_levels" in segments.columns else None
    )
    max_gross = int(max_open.max()) if max_open is not None else 0
    row.update(
        {
            "risk_stop_rate": risk_stop_rate,
            "max_gross_units": max_gross,
            "max_abs_net_units": float("nan"),
            "loser_timeout_rate": 0.0,
            "tp_rate": (trades["exit_reason"] == "grid_tp").mean(),
            "forced_rate": (trades["exit_reason"] != "grid_tp").mean(),
        }
    )
    return pd.DataFrame([row])


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
    cost_cols = [
        c
        for c in [
            "fee_bps_charged",
            "slippage_bps_charged",
            "funding_bps_charged",
        ]
        if c in trades.columns
    ]
    if not trades.empty and "gross_pnl_pct" in trades.columns:
        gross_pnl_pct_sum = float(trades["gross_pnl_pct"].sum())
        net_pnl_pct_sum = float(trades["pnl_pct"].sum())
        total_cost_pct_sum = (
            float(trades[cost_cols].sum(axis=1).sum()) / 10_000.0 if cost_cols else 0.0
        )
        break_even_cost_bps_per_trade = (
            gross_pnl_pct_sum * 10_000.0 / len(trades) if len(trades) else 0.0
        )
        alpha_cost_note = (
            "Gross grid alpha is positive, but conservative transaction costs consume it."
            if gross_pnl_pct_sum > 0 and net_pnl_pct_sum < 0
            else (
                "Net PnL remains positive after the configured transaction costs."
                if net_pnl_pct_sum >= 0
                else "Gross grid alpha is not positive in this window."
            )
        )
        cost_diagnostics = pd.DataFrame(
            [
                {
                    "gross_pnl_pct_sum": gross_pnl_pct_sum,
                    "net_pnl_pct_sum": net_pnl_pct_sum,
                    "total_cost_pct_sum": total_cost_pct_sum,
                    "cost_drag_pct_sum": gross_pnl_pct_sum - net_pnl_pct_sum,
                    "break_even_cost_bps_per_trade": break_even_cost_bps_per_trade,
                    "configured_fee_bps": args.fee_bps,
                    "configured_forced_exit_slippage_bps": args.forced_exit_slippage_bps,
                    "configured_funding_bps_per_8h": args.funding_cost_bps_per_8h,
                    "interpretation": alpha_cost_note,
                }
            ]
        )
    else:
        cost_diagnostics = pd.DataFrame()

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
    Entry: {args.chop_signal} chop at or above {args.chop_min}; hold until chop drops below {args.exit_chop_min}.
    (raw = semantic_chop; ts_quantile = rolling pct rank, window {args.chop_ts_window}.)
    Grid spacing: max({args.grid_atr_mult} ATR, {args.grid_pct:.2%}); levels per side: {args.max_levels};
    fee: {args.fee_bps} bps; slippage sensitivity: {args.slippage_bps} bps. Exclude box: {args.exclude_box}.
    Realistic costs: maker={args.maker_fee_bps} bps, taker={args.taker_fee_bps} bps,
    forced-exit slippage={args.forced_exit_slippage_bps} bps,
    funding cost={args.funding_cost_bps_per_8h} bps/8h.
  </p>
  <h2>Alpha / Cost Diagnostics</h2>
  {cost_diagnostics.to_html(index=False, float_format=lambda x: _fmt(x, 6))}
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
    parser.add_argument(
        "--timeframe",
        default="2h",
        help=(
            "Signal timeframe: pandas resample offset (e.g. 2h, 4h). Loaded from 1m "
            "parquet then resampled; segments and regime masks use this grid."
        ),
    )
    parser.add_argument(
        "--execution-timeframe",
        default=str(defaults.get("execution_timeframe", "1min")),
        help=(
            "Execution OHLC path (default 1min, live-aligned). Segment boundaries "
            "stay on the signal grid; fills run on execution bars in "
            "[signal_close, exit_signal_close) with features asof-joined "
            "(see src/time_series_model/grid/subbar_replay.py). Use 100ms for "
            "aggTrades replay (--agg-data-dir required)."
        ),
    )
    parser.add_argument(
        "--agg-data-dir",
        default=str(defaults.get("agg_data_dir") or "data/agg_data"),
        help="Binance aggTrades zip root for --execution-timeframe 100ms.",
    )
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
        "--max-replenish-per-level",
        default=defaults.get("max_replenish_per_level"),
        type=_parse_max_replenish_cli,
        help=(
            "Post-TP limit replenishes per level per regime segment "
            "(0=one fill, null/unlimited=legacy backtest)."
        ),
    )
    parser.add_argument(
        "--same-bar-entry-exit",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("same_bar_entry_exit", False),
        help=(
            "Research/backtest intrabar assumption. False disallows taking profit "
            "on the same bar that filled a grid level."
        ),
    )
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
        "--block-stable-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("block_stable_box"),
        help=(
            "Block chop entry on stable-box bars (live-aligned when "
            "regime.exclude_box_prefilter is false)."
        ),
    )
    parser.add_argument(
        "--exclude-box",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("exclude_box", True),
        help="Deprecated alias; use --block-stable-box for chop grid.",
    )
    parser.add_argument(
        "--box-pos-min",
        type=float,
        default=None,
        help="Override prefilter box_pos_60 lower bound (requires --box-pos-max).",
    )
    parser.add_argument(
        "--box-pos-max",
        type=float,
        default=None,
        help="Override prefilter box_pos_60 upper bound (requires --box-pos-min).",
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
    parser.add_argument(
        "--chop-signal",
        choices=["raw", "ts_quantile"],
        default=str(defaults.get("chop_signal", "raw")),
        help=(
            "Regime feature for chop masks: raw semantic_chop in [0,1], or "
            "ts_quantile = causal rolling percentile rank of semantic_chop vs past "
            "window (also ~[0,1]; try e.g. --chop-min 0.4 --exit-chop-min 0.25 as p40/p25)."
        ),
    )
    parser.add_argument(
        "--chop-ts-window",
        type=int,
        default=int(defaults.get("chop_ts_window", 1200)),
        help="Rolling window (bars) for semantic_chop_ts_quantile when --chop-signal ts_quantile.",
    )
    parser.add_argument(
        "--chop-ts-min-periods",
        type=int,
        default=int(defaults.get("chop_ts_min_periods", 150)),
        help="min_periods for semantic_chop_ts_quantile (rolling window).",
    )
    parser.add_argument(
        "--compute-chop-ts-q",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("compute_chop_ts_q"),
        help=(
            "Force semantic_chop_ts_q column in build_features. "
            "Default: grid.yaml chop_series.compute_semantic_chop_ts_q if set, "
            "else only when --chop-signal ts_quantile."
        ),
    )
    parser.add_argument(
        "--stability-min",
        type=float,
        default=float(defaults.get("stability_min", 0.85)),
        help="Box prefilter: min stability (StudyConfig); see regime.box_prefilter in grid.yaml.",
    )
    parser.add_argument(
        "--width-min",
        type=float,
        default=float(defaults.get("width_min", 0.04)),
        help="Box prefilter width lower bound (StudyConfig).",
    )
    parser.add_argument(
        "--width-max",
        type=float,
        default=float(defaults.get("width_max", 0.30)),
        help="Box prefilter width upper bound (StudyConfig).",
    )
    parser.add_argument(
        "--touches-min",
        type=int,
        default=int(defaults.get("touches_min", 5)),
        help="Box prefilter minimum boundary touches (StudyConfig).",
    )
    parser.add_argument(
        "--feature-store-dir",
        default=defaults.get("feature_store_dir"),
        help="FeatureStore root (defaults from grid_backtest.feature_store_dir in YAML).",
    )
    parser.add_argument(
        "--feature-store-layer",
        default=defaults.get("feature_store_layer"),
    )
    parser.add_argument(
        "--feature-store-timeframe",
        default=defaults.get("feature_store_timeframe"),
    )
    parser.add_argument("--out-dir", default="results/chop_grid/backtest")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="account initial capital (USDT) for realized PnL projection/reporting",
    )
    parser.add_argument("--map-symbols", default="BTCUSDT")
    parser.add_argument("--continuous-map-symbols", default="")
    parser.add_argument("--no-maps", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades, segments, equity = run_backtest(args)
    if not trades.empty and "pnl_per_capital" in trades.columns:
        trades["pnl_usd_realized"] = pd.to_numeric(
            trades["pnl_per_capital"], errors="coerce"
        ).fillna(0.0) * float(args.initial_capital)
    n_sym = max(1, trades["symbol"].nunique()) if not trades.empty else 1
    portfolio_initial = float(args.initial_capital) * n_sym
    if not equity.empty and "portfolio_pnl_per_capital" in equity.columns:
        equity["pnl_usd_realized"] = (
            pd.to_numeric(equity["portfolio_pnl_per_capital"], errors="coerce").fillna(
                0.0
            )
            * portfolio_initial
        )
        if "cum_pnl_per_capital" in equity.columns:
            equity["cum_pnl_usd_realized"] = portfolio_initial * pd.to_numeric(
                equity["cum_pnl_per_capital"], errors="coerce"
            ).fillna(0.0)
    trades.to_csv(out_dir / "grid_trades.csv", index=False)
    segments.to_csv(out_dir / "grid_segments.csv", index=False)
    equity.to_csv(out_dir / "equity_curve.csv", index=False)
    metrics = build_metrics(trades, segments, equity)
    aligned = summarize_dual_add_aligned(trades, segments)
    if not aligned.empty:
        aligned.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps({"args": vars(args), "metrics": metrics}, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(out_dir, args, trades, segments, equity)
    portfolio_metrics = portfolio_metrics_from_trades(trades)
    write_capital_report_from_trades(
        trades_path=out_dir / "grid_trades.csv",
        out_dir=out_dir,
        unit="capital_normalized",
        title="Chop Grid Capital Report",
        initial_capital=portfolio_initial,
        start_date=args.start,
        end_date=args.end,
        total_r=portfolio_metrics["portfolio_pnl_per_capital_timeline"],
        n_symbols=n_sym,
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
            title="Chop Grid Continuous Trading Map",
        )

    trade_summary, segment_summary = _summary_tables(trades, segments, equity)
    print("\n=== Summary (dual_add schema, summary.csv) ===")
    print(
        aligned.to_string(index=False)
        if not aligned.empty
        else "(empty — no trades or segments)"
    )
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
