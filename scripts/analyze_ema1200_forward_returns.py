#!/usr/bin/env python3
"""
Study EMA(span=1200) on 120T closes — same bar & span as production ``ema_1200_value_f`` / ``ema_1200_position``.

Computes:
  - ema_pos = (close - ema_1200) / close  clipped [-1, 1]  (matches position_logic / baseline_features)
  - ema_slope = (ema_t - ema_{t-w}) / ema_t  (default w=10 bars, matches ``compute_ma_slope_from_series``)

For each horizon H in bars (default maps ~5/10/20 calendar days at 12 bars/day on 2H):
  - forward simple return: close[t+H]/close[t] - 1
  - pooled cross-symbol Spearman corr(signal, fwd_ret), plus per-symbol corr mean/std
  - quintile bucket mean fwd_ret & hit-rate (fwd_ret > 0)

Usage:
  python scripts/analyze_ema1200_forward_returns.py \\
    --data-path data/parquet_data \\
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
    --start-date 2022-01-01 \\
    --end-date 2026-05-01

Outputs JSON under --out-dir (default results/ema1200_forward_study/): summary.json,
slope_conditional_entry.json (incremental value of slope given ema_pos side).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data_tools.data_handler import DataHandler


def _bars_per_day_120t() -> float:
    return 24.0 / 2.0  # 12 two-hour bars per day


def compute_panel(
    *,
    dh: DataHandler,
    symbols: List[str],
    start_date: str,
    end_date: str,
    ema_span: int,
    slope_window: int,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    tf = "120T"
    for sym in symbols:
        df = dh.load_ohlcv(sym, timeframe=tf, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            print(f"  skip empty: {sym}")
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        ema = close.ewm(span=int(ema_span), adjust=False).mean()
        close_safe = close.replace(0.0, np.nan)
        ema_pos = (
            ((close - ema) / close_safe)
            .replace([np.inf, -np.inf], np.nan)
            .clip(-1.0, 1.0)
        )
        ema_slope = (ema - ema.shift(int(slope_window))) / ema.replace(0.0, np.nan)
        ema_slope = ema_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        part = pd.DataFrame(
            {
                "symbol": sym,
                "close": close,
                "ema_1200": ema,
                "ema_pos": ema_pos,
                "ema_slope": ema_slope,
            },
            index=df.index,
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, axis=0).sort_index()
    return out


def _forward_returns(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-int(h)) / close - 1.0


def spearman_corr_pairwise(x: pd.Series, y: pd.Series) -> float:
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(m) < 50:
        return float("nan")
    return float(m["x"].corr(m["y"], method="spearman"))


def summarize_horizon(
    panel: pd.DataFrame,
    horizon_bars: int,
    horizon_label: str,
) -> Dict[str, object]:
    pieces: List[pd.Series] = []
    cor_pos: List[float] = []
    cor_slope: List[float] = []
    cor_combo: List[float] = []

    for sym, g in panel.groupby("symbol"):
        g = g.sort_index()
        fwd = _forward_returns(g["close"], horizon_bars)
        valid = g[["ema_pos", "ema_slope"]].copy()
        valid["fwd"] = fwd
        valid = valid.dropna()
        if len(valid) < 80:
            continue
        pieces.append(valid.assign(symbol=sym))

        cor_pos.append(spearman_corr_pairwise(valid["ema_pos"], valid["fwd"]))
        cor_slope.append(spearman_corr_pairwise(valid["ema_slope"], valid["fwd"]))
        combo = np.sign(valid["ema_pos"]) * np.sign(valid["ema_slope"])
        cor_combo.append(
            spearman_corr_pairwise(pd.Series(combo, index=valid.index), valid["fwd"])
        )

    if not pieces:
        return {
            "horizon_label": horizon_label,
            "horizon_bars": horizon_bars,
            "n_rows": 0,
        }

    all_df = pd.concat(pieces, axis=0)
    pooled_pos = spearman_corr_pairwise(all_df["ema_pos"], all_df["fwd"])
    pooled_slope = spearman_corr_pairwise(all_df["ema_slope"], all_df["fwd"])
    combo_all = np.sign(all_df["ema_pos"]) * np.sign(all_df["ema_slope"])
    pooled_combo = spearman_corr_pairwise(
        pd.Series(combo_all.values, index=all_df.index), all_df["fwd"]
    )

    # Quintiles by ema_pos (pooled)
    all_df["pos_q"] = pd.qcut(all_df["ema_pos"], q=5, labels=False, duplicates="drop")
    bucket = (
        all_df.dropna(subset=["pos_q"])
        .groupby("pos_q")["fwd"]
        .agg(["mean", "median", "count", lambda s: float((s > 0).mean())])
        .rename(columns={"<lambda_0>": "hit_rate"})
    )

    long_bias = all_df[(all_df["ema_pos"] > 0) & (all_df["ema_slope"] > 0)]["fwd"]
    short_bias = all_df[(all_df["ema_pos"] < 0) & (all_df["ema_slope"] < 0)]["fwd"]
    neutral = all_df[(all_df["ema_pos"] * all_df["ema_slope"]) <= 0]["fwd"]

    def _m(series: pd.Series) -> Dict[str, float]:
        series = series.dropna()
        if series.empty:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "n": 0,
                "hit_rate": float("nan"),
            }
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "n": int(series.shape[0]),
            "hit_rate": float((series > 0).mean()),
        }

    return {
        "horizon_label": horizon_label,
        "horizon_bars": horizon_bars,
        "n_rows": int(all_df.shape[0]),
        "pooled_spearman_ema_pos": pooled_pos,
        "pooled_spearman_ema_slope": pooled_slope,
        "pooled_spearman_sign_pos_x_sign_slope": pooled_combo,
        "per_symbol_spearman_pos_mean": float(np.nanmean(cor_pos)),
        "per_symbol_spearman_pos_std": (
            float(np.nanstd(cor_pos, ddof=1)) if len(cor_pos) > 1 else 0.0
        ),
        "per_symbol_spearman_slope_mean": float(np.nanmean(cor_slope)),
        "per_symbol_n": len(cor_pos),
        "regime_long_both_pos": _m(long_bias),
        "regime_short_both_neg": _m(short_bias),
        "regime_mixed_sign": _m(neutral),
        "quintile_fwd_mean_by_ema_pos": bucket["mean"].to_dict() if len(bucket) else {},
        "quintile_hit_rate_by_ema_pos": (
            bucket["hit_rate"].to_dict() if len(bucket) else {}
        ),
    }


def _fwd_two_series_stats(
    aligned_fwd: pd.Series, other_fwd: pd.Series
) -> Dict[str, object]:
    """Compare forward-return distributions: ``aligned`` vs ``other`` bucket."""

    def _one(s: pd.Series) -> Dict[str, float]:
        s = pd.Series(s).dropna()
        if s.empty:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "n": 0,
                "hit_rate": float("nan"),
            }
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "n": int(len(s)),
            "hit_rate": float((s > 0).mean()),
        }

    sa, sb = _one(aligned_fwd), _one(other_fwd)
    dm = (
        float(sa["mean"] - sb["mean"])
        if not (np.isnan(sa["mean"]) or np.isnan(sb["mean"]))
        else float("nan")
    )
    return {
        "slope_aligned_bucket": sa,
        "slope_other_bucket": sb,
        "delta_mean_fwd_aligned_minus_other": dm,
    }


def summarize_slope_given_side(
    panel: pd.DataFrame,
    horizon_bars: int,
    horizon_label: str,
) -> Dict[str, object]:
    """Conditional on price vs EMA side, does slope agreement improve fwd returns?

    Long (candidate long trend zone): ``ema_pos > 0``. Compare slope>0 vs slope<=0.
    Short (candidate short zone): ``ema_pos < 0``. Compare slope<0 vs slope>=0.
    """
    pieces_long_a: List[pd.Series] = []
    pieces_long_b: List[pd.Series] = []
    pieces_short_a: List[pd.Series] = []
    pieces_short_b: List[pd.Series] = []

    for sym, g in panel.groupby("symbol"):
        g = g.sort_index()
        fwd = _forward_returns(g["close"], horizon_bars)
        base = pd.DataFrame(
            {"ema_pos": g["ema_pos"], "ema_slope": g["ema_slope"], "fwd": fwd}
        ).dropna()
        if len(base) < 80:
            continue

        long_ok = base["ema_pos"] > 0
        pieces_long_a.append(base.loc[long_ok & (base["ema_slope"] > 0), "fwd"])
        pieces_long_b.append(base.loc[long_ok & (base["ema_slope"] <= 0), "fwd"])

        sh_ok = base["ema_pos"] < 0
        pieces_short_a.append(base.loc[sh_ok & (base["ema_slope"] < 0), "fwd"])
        pieces_short_b.append(base.loc[sh_ok & (base["ema_slope"] >= 0), "fwd"])

    if not pieces_long_a:
        return {
            "horizon_label": horizon_label,
            "horizon_bars": horizon_bars,
            "n_rows": 0,
        }

    fla = pd.concat(pieces_long_a, axis=0)
    flb = pd.concat(pieces_long_b, axis=0)
    fsa = pd.concat(pieces_short_a, axis=0)
    fsb = pd.concat(pieces_short_b, axis=0)

    long_stats = _fwd_two_series_stats(fla, flb)
    short_stats = _fwd_two_series_stats(fsa, fsb)

    return {
        "horizon_label": horizon_label,
        "horizon_bars": horizon_bars,
        "n_rows": int(len(fla) + len(flb) + len(fsa) + len(fsb)),
        "long_given_ema_pos_positive_slope_gt0_vs_else": long_stats,
        "short_given_ema_pos_negative_slope_lt0_vs_else": short_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    ap.add_argument("--start-date", default="2022-01-01")
    ap.add_argument("--end-date", default="2026-05-01")
    ap.add_argument("--ema-span", type=int, default=1200)
    ap.add_argument(
        "--slope-window", type=int, default=10, help="bars for (ema_t-ema_{t-w})/ema_t"
    )
    ap.add_argument(
        "--horizons-days",
        default="5,10,20,60",
        help="comma-separated approximate calendar days; converted to 120T bars (12 bars/day)",
    )
    ap.add_argument("--out-dir", default="results/ema1200_forward_study")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bpd = _bars_per_day_120t()
    horizon_days = [
        float(x.strip()) for x in args.horizons_days.split(",") if x.strip()
    ]
    horizons: List[Tuple[int, str]] = []
    for d in horizon_days:
        bars = max(1, int(round(d * bpd)))
        horizons.append((bars, f"~{d:g}d"))

    dh = DataHandler(args.data_path)
    print(
        f"Building panel {args.start_date} .. {args.end_date} | symbols={symbols} | 120T ema_span={args.ema_span}"
    )
    panel = compute_panel(
        dh=dh,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        ema_span=args.ema_span,
        slope_window=args.slope_window,
    )
    if panel.empty:
        print("No data; check --data-path and symbols.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    for bars, label in horizons:
        row = summarize_horizon(panel, bars, label)
        summaries.append(row)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")

    slope_summaries: List[Dict[str, object]] = []
    for bars, label in horizons:
        slope_summaries.append(summarize_slope_given_side(panel, bars, label))
    slope_path = out_dir / "slope_conditional_entry.json"
    slope_path.write_text(json.dumps(slope_summaries, indent=2), encoding="utf-8")
    print(f"Wrote {slope_path}")

    # Markdown-friendly table (stdout)
    print("\n=== Spearman corr pooled (signal vs forward return) ===")
    print(
        f"{'horizon':<12} {'rho(pos)':>10} {'rho(slope)':>12} {'rho(sign product)':>18}"
    )
    for s in summaries:
        if not s.get("n_rows"):
            continue
        print(
            f"{str(s['horizon_label']):<12} "
            f"{s.get('pooled_spearman_ema_pos', float('nan')):10.4f} "
            f"{s.get('pooled_spearman_ema_slope', float('nan')):12.4f} "
            f"{s.get('pooled_spearman_sign_pos_x_sign_slope', float('nan')):18.4f}"
        )

    print(
        "\n=== Trend entry — slope incremental (given price vs EMA1200 side, pooled) ==="
    )
    print(
        "Long: among ema_pos>0, compare slope>0 vs slope<=0 (rising slow MA vs flat/down)."
    )
    print(
        "Short: among ema_pos<0, compare slope<0 vs slope>=0 (falling slow MA vs flat/up); "
        "interpret with care (spot fwd ≠ perp short PnL)."
    )
    for row in slope_summaries:
        if not row.get("n_rows"):
            continue
        lab = row["horizon_label"]
        lg = row.get("long_given_ema_pos_positive_slope_gt0_vs_else") or {}
        sh = row.get("short_given_ema_pos_negative_slope_lt0_vs_else") or {}
        la = lg.get("slope_aligned_bucket") or {}
        lo = lg.get("slope_other_bucket") or {}
        sa = sh.get("slope_aligned_bucket") or {}
        so = sh.get("slope_other_bucket") or {}
        print(f"{lab} ({row['horizon_bars']} bars, pooled n={row['n_rows']})")
        print(
            f"  LONG  slope>0: mean_fwd={la.get('mean', float('nan')):+.5f} "
            f"hit={la.get('hit_rate', float('nan')):.3f} n={la.get('n', 0)} | "
            f"slope<=0: mean_fwd={lo.get('mean', float('nan')):+.5f} "
            f"hit={lo.get('hit_rate', float('nan')):.3f} n={lo.get('n', 0)} | "
            f"Δmean={lg.get('delta_mean_fwd_aligned_minus_other', float('nan')):+.5f}"
        )
        sd = float(sh.get("delta_mean_fwd_aligned_minus_other") or float("nan"))
        if np.isnan(sd):
            short_note = ""
        elif sd < 0:
            short_note = " (Δ<0: slope<0 区 spot 前瞻更弱 → 对「裸空 timing」更有利；需结合永续基差/资金费)"
        else:
            short_note = (
                " (Δ>0: 本样本里价在均线下且慢均仍下行时 spot 前瞻反而更高——"
                "多为大周期牛底/反弹段；空单是否加斜率过滤要回测验证)"
            )
        print(
            f"  SHORT slope<0: mean_fwd={sa.get('mean', float('nan')):+.5f} "
            f"hit={sa.get('hit_rate', float('nan')):.3f} n={sa.get('n', 0)} | "
            f"slope>=0: mean_fwd={so.get('mean', float('nan')):+.5f} "
            f"hit={so.get('hit_rate', float('nan')):.3f} n={so.get('n', 0)} | "
            f"Δmean={sd:+.5f}{short_note}"
        )

    for s in summaries:
        if not s.get("n_rows"):
            continue
        print(f"\n{s['horizon_label']} ({s['horizon_bars']} bars, n={s['n_rows']})")
        for name in (
            "regime_long_both_pos",
            "regime_short_both_neg",
            "regime_mixed_sign",
        ):
            d = s.get(name) or {}
            print(
                f"  {name}: mean_fwd={d.get('mean', float('nan')):+.5f} "
                f"hit={d.get('hit_rate', float('nan')):.3f} n={d.get('n', 0)}"
            )

    print("\nInterpretation hints:")
    print(
        "  - Positive rho(ema_pos, fwd) => price further above slow EMA predicts higher fwd return "
        "(momentum / trend continuation on that horizon)."
    )
    print(
        "  - Positive rho(ema_slope, fwd) => rising slow EMA predicts higher fwd return."
    )
    print(
        "  - Fat-tail trend strategies often care about multi-week horizons; use ~60d row as coarse proxy."
    )
    print(
        "  - Short horizons (5–20d) proxy chop/reversal sensitivity; significance not computed here "
        "(use block-bootstrap or reality-check vs txn costs)."
    )


if __name__ == "__main__":
    main()
