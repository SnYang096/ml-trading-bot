#!/usr/bin/env python3
"""Sweep max_replenish_per_level_per_segment for chop_grid (live/backtest alignment).

Example::

    python scripts/sweep_chop_grid_replenish.py \\
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \\
      --start 2022-01-01 --end 2026-05-01 \\
      --out-csv results/chop_grid/sweep_replenish.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chop_grid_backtest import collect_chop_grid_trades_for_symbol  # noqa: E402
from scripts.diagnose_chop_grid import (
    GridConfig,
    build_features,
    merge_chop_grid_yaml,
)  # noqa: E402
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from scripts.sweep_chop_regime_thresholds import _one_row_metrics  # noqa: E402
from src.time_series_model.grid.chop_grid_engine import (
    ChopGridEngine,
    GridEngineConfig,
)  # noqa: E402

DEFAULT_CONFIG = (
    PROJECT_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
)
LIVE_GRID_YAML = PROJECT_ROOT / "live/highcap/config/strategies/chop_grid"


def _load_universe_symbols() -> List[str]:
    cal = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    ug = cal.get("universe_group") or {}
    group_file = PROJECT_ROOT / str(ug.get("file", ""))
    universe_set = str(ug.get("universe_set", "starter_a"))
    group = str(ug.get("group", "highcap"))
    if not group_file.exists():
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    data = yaml.safe_load(group_file.read_text(encoding="utf-8")) or {}
    return list(data.get(universe_set, {}).get(group, []) or [])


def _parse_replenish_grid(spec: str) -> List[int | None]:
    out: List[int | None] = []
    for part in spec.split(","):
        p = part.strip().lower()
        if p in {"null", "none", "unlimited", "inf"}:
            out.append(None)
        else:
            out.append(int(part.strip()))
    return out


def _metrics_extended(trades: pd.DataFrame, segments: pd.DataFrame) -> dict:
    base = _one_row_metrics(trades, segments)
    if trades.empty:
        base.update(
            {
                "sharpe_r": 0.0,
                "max_drawdown_r": 0.0,
                "replenish_trades": 0,
            }
        )
        return base
    r = pd.to_numeric(trades.get("r_equiv_per_capital"), errors="coerce").fillna(0.0)
    sharpe = 0.0
    if len(r) > 1 and float(r.std(ddof=1)) > 0:
        sharpe = float(r.mean() / r.std(ddof=1) * (252**0.5))
    mdd = 0.0
    if "max_drawdown" in segments.columns and len(segments):
        mdd = float(pd.to_numeric(segments["max_drawdown"], errors="coerce").min())
    replenish = 0
    if "replenish_trades" in segments.columns:
        replenish = int(
            pd.to_numeric(segments["replenish_trades"], errors="coerce").sum()
        )
    base["sharpe_r"] = sharpe
    base["max_drawdown_r"] = mdd
    base["replenish_trades"] = replenish
    return base


def _recommend(rows: List[dict], *, holdout_start: pd.Timestamp) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"status": "no_rows"}
    baseline = df[df["max_replenish"] == 0]
    base_ret = float(baseline["sum_pnl_per_capital"].sum()) if len(baseline) else 0.0

    def _passes(g: pd.DataFrame) -> pd.DataFrame:
        forced = pd.to_numeric(g["forced_rate"], errors="coerce").fillna(1.0)
        mdd = pd.to_numeric(g["max_drawdown_r"], errors="coerce").fillna(-1.0)
        return g[(forced <= 0.35) & (mdd >= -0.08)]

    oos = df[pd.to_datetime(df["period_end"], utc=True) >= holdout_start]
    pool = _passes(oos if len(oos) else df)
    if pool.empty:
        pool = _passes(df)
    if pool.empty:
        return {"status": "no_candidate_passes_gates", "baseline_sum_pnl": base_ret}

    agg = (
        pool.groupby("max_replenish", dropna=False)
        .agg(
            sum_pnl_per_capital=("sum_pnl_per_capital", "sum"),
            sharpe_r=("sharpe_r", "mean"),
            forced_rate=("forced_rate", "mean"),
            max_drawdown_r=("max_drawdown_r", "min"),
            trades=("trades", "sum"),
            replenish_trades=("replenish_trades", "sum"),
        )
        .reset_index()
    )
    agg = agg.sort_values(
        ["sum_pnl_per_capital", "sharpe_r", "forced_rate", "max_drawdown_r"],
        ascending=[False, False, True, False],
    )
    best = agg.iloc[0]
    mr = best["max_replenish"]
    label = "unlimited" if pd.isna(mr) else str(int(mr))
    return {
        "status": "ok",
        "recommended_max_replenish": None if pd.isna(mr) else int(mr),
        "recommended_label": label,
        "baseline_sum_pnl": base_ret,
        "best_sum_pnl": float(best["sum_pnl_per_capital"]),
        "delta_vs_baseline": float(best["sum_pnl_per_capital"]) - base_ret,
        "best_sharpe_r": float(best["sharpe_r"]),
        "best_forced_rate": float(best["forced_rate"]),
        "best_max_drawdown_r": float(best["max_drawdown_r"]),
    }


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(LIVE_GRID_YAML))
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    defaults = merge_chop_grid_yaml(config_path)

    cal = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    dates = cal.get("dates") or {}
    holdout_months = int(dates.get("holdout_months", 26))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(config_path))
    ap.add_argument(
        "--data-dir", default=str(cal.get("data_path", "data/parquet_data"))
    )
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default=str(dates.get("start_date", "2022-01-01")))
    ap.add_argument("--end", default=str(dates.get("end_date", "2026-05-01")))
    ap.add_argument("--warmup-days", type=int, default=120)
    ap.add_argument("--timeframe", default="120T")
    ap.add_argument("--holdout-months", type=int, default=holdout_months)
    ap.add_argument(
        "--sweep-replenish",
        default="0,1,2,3,5,null",
        help="Comma list of max_replenish_per_level (null=unlimited).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/chop_grid/sweep_replenish.csv"),
    )
    args = ap.parse_args()

    symbols = [
        s.strip().upper()
        for s in (args.symbols or ",".join(_load_universe_symbols())).split(",")
        if s.strip()
    ]
    replenish_grid = _parse_replenish_grid(args.sweep_replenish)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    holdout_start = end - pd.DateOffset(months=int(args.holdout_months))
    warmup_start = start - pd.Timedelta(days=args.warmup_days)
    data_dir = Path(args.data_dir)

    feat_cfg = GridConfig(
        box_window=int(defaults.get("box_window", 120)),
        chop_min=0.50,
        exit_chop_min=0.32,
        min_segment_bars=int(defaults.get("min_segment_bars", 6)),
        max_segment_bars=int(defaults.get("max_segment_bars", 120)),
        grid_atr_mult=1.0,
        grid_pct=0.01,
        max_levels=2,
        fee_bps=float(defaults.get("fee_bps", 4.0)),
        stability_min=float(defaults.get("stability_min", 0.85)),
        width_min=float(defaults.get("width_min", 0.04)),
        width_max=float(defaults.get("width_max", 0.30)),
        touches_min=int(defaults.get("touches_min", 5)),
    )

    rows_out: List[dict] = []
    for max_rep in replenish_grid:
        engine = ChopGridEngine(
            GridEngineConfig(
                entry_chop_min=0.50,
                exit_chop_below=0.32,
                min_segment_bars=feat_cfg.min_segment_bars,
                max_segment_bars=feat_cfg.max_segment_bars,
                grid_atr_mult=1.0,
                grid_min_pct=0.01,
                max_levels_per_side=2,
                fee_bps=float(defaults.get("fee_bps", 4.0)),
                maker_fee_bps=float(defaults.get("maker_fee_bps", 4.0)),
                taker_fee_bps=float(defaults.get("taker_fee_bps", 4.0)),
                forced_exit_slippage_bps=float(
                    defaults.get("forced_exit_slippage_bps", 0.0)
                ),
                funding_cost_bps_per_8h=float(
                    defaults.get("funding_cost_bps_per_8h", 0.0)
                ),
                max_loss_per_grid=float(defaults.get("max_loss_per_grid", 0.03)),
                max_open_levels_total=4,
                max_replenish_per_level_per_segment=max_rep,
            )
        )
        for symbol in symbols:
            raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
            if raw.empty:
                print(f"skip {symbol}: no data")
                continue
            bars = _resample_ohlcv(raw, args.timeframe)
            df = build_features(symbol, bars, feat_cfg)
            df = df[(df.index >= start) & (df.index <= end)].copy()
            if df.empty:
                continue
            tlist, slist, _, _ = collect_chop_grid_trades_for_symbol(
                symbol,
                df,
                None,
                None,
                feat_cfg,
                engine,
                exclude_box=True,
            )
            trades = pd.DataFrame(tlist)
            segments = pd.DataFrame(slist)
            m = _metrics_extended(trades, segments)
            rows_out.append(
                {
                    "max_replenish": max_rep if max_rep is not None else "null",
                    "symbol": symbol,
                    "period_start": start.isoformat(),
                    "period_end": end.isoformat(),
                    **m,
                }
            )
            print(
                f"max_replenish={max_rep} {symbol} trades={m.get('trades', 0)} "
                f"pnl={m.get('sum_pnl_per_capital', 0):.4f}"
            )

    out_csv = args.out_csv
    if not out_csv.is_absolute():
        out_csv = PROJECT_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    rec = _recommend(rows_out, holdout_start=holdout_start)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = out_csv.with_name(f"sweep_replenish_{run_id}.md")
    lines = [
        "# Chop grid replenish sweep",
        "",
        f"- Config: `{config_path}`",
        f"- Symbols: {', '.join(symbols)}",
        f"- Range: {start.date()} .. {end.date()}",
        f"- OOS from: {holdout_start.date()}",
        f"- CSV: `{out_csv}`",
        "",
        "## Recommendation",
        "",
    ]
    if rec.get("status") == "ok":
        lines.extend(
            [
                f"- **Recommended `max_replenish_per_level_per_segment`:** `{rec['recommended_label']}`",
                f"- OOS sum pnl (capital units): {rec['best_sum_pnl']:.4f} "
                f"(baseline N=0: {rec['baseline_sum_pnl']:.4f}, "
                f"delta {rec['delta_vs_baseline']:+.4f})",
                f"- Mean Sharpe (R): {rec['best_sharpe_r']:.3f}",
                f"- Mean forced exit rate: {rec['best_forced_rate']:.3f}",
                f"- Worst segment drawdown (R): {rec['best_max_drawdown_r']:.4f}",
                "",
                "Gates: forced_rate <= 0.35, max_drawdown_r >= -0.08.",
            ]
        )
    else:
        lines.append(f"- Status: `{rec.get('status')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
