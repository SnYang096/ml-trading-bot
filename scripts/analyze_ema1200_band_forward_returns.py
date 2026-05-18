#!/usr/bin/env python3
"""Empirical stratification for BPC/TPC-style EMA1200 position band + slope alignment.

Reads 1min research parquets via DataHandler, resamples to 120T (same path as prod),
computes TA-Lib EMA(1200), ``ema_1200_position`` and ``ema_1200_slope_10`` using the same
implementations as feature_dependencies / baseline_features.

No entry signals — only geometric partitions from ``direction.yaml``:
  long sleeve: inner < position < outer
  short sleeve: -outer < position < -inner
  band_zero_etc: else — near-EMA dead zone, NaNs, or empty band (``single_position_band`` → 0)

Optional slope filter matches ``require_sign_agreement``: |slope| > deadband and sign(slope)
matches sleeve direction.

Outputs summary tables + Mann–Whitney U vs pooled "rest" baseline (overlap-aware caveat
printed to stderr).

Example:
  PYTHONPATH=. python scripts/analyze_ema1200_band_forward_returns.py \\
      --symbols BTCUSDT,ETHUSDT --start 2023-01-01 --end 2026-06-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import talib  # noqa: E402

from src.data_tools.data_handler import DataHandler  # noqa: E402
from src.features.time_series.baseline_features import (  # noqa: E402
    compute_ma_slope_from_series,
    compute_price_vs_ma_position_from_series,
)


def _load_120t(symbol: str, *, data_path: str, start: str, end: str) -> pd.DataFrame:
    dh = DataHandler(data_path, use_default_processors=False)
    df = dh.load_ohlcv(
        symbol=symbol,
        timeframe="120T",
        start_date=start,
        end_date=end,
        validate=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: expected DatetimeIndex on OHLC frame")
    return df.sort_index()


def _ema1200(close: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce").astype(float).values
    ema = talib.EMA(c, timeperiod=1200)
    return pd.Series(ema, index=close.index, dtype=float)


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_1200"] = _ema1200(out["close"])
    pos = compute_price_vs_ma_position_from_series(
        close=out["close"],
        ma=out["ema_1200"],
        output_column="ema_1200_position",
    )
    out["ema_1200_position"] = pos["ema_1200_position"]
    slope = compute_ma_slope_from_series(
        ma=out["ema_1200"],
        window=10,
        output_column="ema_1200_slope_10",
    )
    out["ema_1200_slope_10"] = slope["ema_1200_slope_10"]
    return out


def _band_labels(position: pd.Series, inner: float, outer: float) -> pd.Series:
    """+1 long sleeve, -1 short sleeve, 0 dead/overextended."""
    inner = float(inner)
    outer = float(outer)
    p = pd.to_numeric(position, errors="coerce").astype(float)
    lab = pd.Series(0, index=p.index, dtype=np.int8)
    v = p.notna()
    long_m = v & (p > inner) & (p < outer)
    short_m = v & (p > -outer) & (p < -inner)
    lab.loc[long_m] = 1
    lab.loc[short_m] = -1
    return lab


def _forward_log_returns(
    close: pd.Series, horizons: Iterable[int]
) -> dict[int, pd.Series]:
    """log(c[t+h]/c[t]), NaN near end."""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    out: dict[int, pd.Series] = {}
    for h in horizons:
        fut = c.shift(-int(h))
        out[int(h)] = np.log(fut / c.replace(0.0, np.nan))
    return out


def _summarize(series: pd.Series) -> tuple[int, float, float, float]:
    s = series.dropna().astype(float)
    n = int(len(s))
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return (
        n,
        float(s.mean()),
        float(s.median()),
        float(s.mean()) / float(s.sem()) if s.sem() > 0 else float("nan"),
    )


def _mann_whitney(a: pd.Series, b: pd.Series):
    """Return (statistic, pvalue) or (nan, nan) if scipy missing or small n."""
    try:
        from scipy.stats import mannwhitneyu

        x = a.dropna().astype(float).values
        y = b.dropna().astype(float).values
        if len(x) < 30 or len(y) < 30:
            return float("nan"), float("nan")
        r = mannwhitneyu(x, y, alternative="two-sided")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    block_len: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Block-bootstrap mean(A)-mean(B) on concatenated pooled index order (crude).

    ``a``, ``b`` are 1-D aligned subsets from the same timeframe (overlapping bar issue
    remains); used only as descriptive robustness ribbon.
    """
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    if na < 50 or nb < 50:
        return float("nan"), float("nan"), float("nan")

    pooled = np.concatenate([a.astype(float), b.astype(float)])
    n = len(pooled)
    hits = []

    bl = max(2, block_len)

    def _blocked_sample():
        samp = []
        while len(samp) < na:
            st = rng.integers(0, max(1, n - bl + 1))
            samp.extend(pooled[st : st + bl].tolist())
        return np.array(samp[:na], dtype=float)

    def _blocked_sample_alt():
        samp = []
        while len(samp) < nb:
            st = rng.integers(0, max(1, n - bl + 1))
            samp.extend(pooled[st : st + bl].tolist())
        return np.array(samp[:nb], dtype=float)

    for _ in range(n_boot):
        ma = float(_blocked_sample().mean())
        mb = float(_blocked_sample_alt().mean())
        hits.append(ma - mb)
    mu = float(np.mean(a) - np.mean(b))
    lo = float(np.quantile(hits, 0.025))
    hi = float(np.quantile(hits, 0.975))
    return mu, lo, hi


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", default="data/parquet_data", help="research parquet dir"
    )
    p.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT",
        help="comma-separated",
    )
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-06-01")
    p.add_argument("--inner-abs", type=float, default=0.03)
    p.add_argument("--outer-abs", type=float, default=1.0)
    p.add_argument("--deadband-slope", type=float, default=0.0)
    p.add_argument(
        "--horizons",
        default="6,12,24",
        help="forward bars at 120T (comma-separated integers)",
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--block-len", type=int, default=80)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument(
        "--warmup-drop",
        type=int,
        default=1350,
        help="bars dropped after load (≥1200 EMA warmup + slope)",
    )

    args = p.parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    rows = []
    for sym in symbols:
        df = _load_120t(sym, data_path=args.data_path, start=args.start, end=args.end)
        if df.empty:
            print(f"WARN: empty OHLC after load — {sym}", file=sys.stderr)
            continue
        df = _prep_features(df)
        df["_symbol"] = sym
        if args.warmup_drop > 0 and len(df) > args.warmup_drop:
            df = df.iloc[int(args.warmup_drop) :].copy()

        fwd = _forward_log_returns(df["close"], horizons)
        for h in horizons:
            df[f"_fwd_ln_{h}"] = fwd[h]

        band = _band_labels(df["ema_1200_position"], args.inner_abs, args.outer_abs)
        df["_band_raw"] = band
        slo = pd.to_numeric(df["ema_1200_slope_10"], errors="coerce").astype(float)

        dd = float(args.deadband_slope)
        align_long_mask = (band == 1) & slo.notna() & (slo > dd)
        align_short_mask = (band == -1) & slo.notna() & (slo < -dd)
        df["_aligned_long_slope"] = align_long_mask.astype(bool)
        df["_aligned_short_slope"] = align_short_mask.astype(bool)
        sleeve_long = band == 1
        sleeve_short = band == -1
        dead = band == 0
        miss_long_geom = sleeve_long & ~align_long_mask
        miss_short_geom = sleeve_short & ~align_short_mask

        for h in horizons:
            tgt = df[f"_fwd_ln_{h}"].astype(float)

            subsets = [
                ("all_bars", np.ones(len(df), dtype=bool)),
                ("band_zero_or_clip", dead.to_numpy(dtype=bool)),
                ("long_sleeve_geom", sleeve_long.to_numpy(dtype=bool)),
                ("short_sleeve_geom", sleeve_short.to_numpy(dtype=bool)),
                ("aligned_long+slope_match", df["_aligned_long_slope"].to_numpy()),
                ("aligned_short+slope_match", df["_aligned_short_slope"].to_numpy()),
                (
                    "long_sleeve_slope_mismatch",
                    miss_long_geom.to_numpy(dtype=bool),
                ),
                (
                    "short_sleeve_slope_mismatch",
                    miss_short_geom.to_numpy(dtype=bool),
                ),
            ]

            for label, mask in subsets:
                rr = tgt.to_numpy(dtype=float)[mask]
                rr = rr[~np.isnan(rr)]
                n, mean, median, _t = _summarize(pd.Series(rr))
                rows.append(
                    {
                        "symbol": sym,
                        "horizon_bars": h,
                        "partition": label,
                        "n": n,
                        "mean_ln_fwd": mean,
                        "median_ln_fwd": median,
                        "mannwhitney_U": float("nan"),
                        "mannwhitney_p": float("nan"),
                        "bootstrap_diff_mean": float("nan"),
                        "bootstrap_diff_q025": float("nan"),
                        "bootstrap_diff_q975": float("nan"),
                    }
                )

            # Comparisons aligned vs rest-of-sample (overlap caveat)
            for side, aligned_m in (
                ("aligned_long+slope_match", df["_aligned_long_slope"].to_numpy()),
                ("aligned_short+slope_match", df["_aligned_short_slope"].to_numpy()),
            ):
                y = tgt.to_numpy(dtype=float)
                m = aligned_m & np.isfinite(y)
                mc = (~aligned_m) & np.isfinite(y)
                ua, vb = pd.Series(y[m]), pd.Series(y[mc])

                mw_u, mw_p = _mann_whitney(ua, vb)

                # short-side directional PnL proxy: compare -fwd vs rest for shorts
                if side == "aligned_short+slope_match":
                    ua2, vb2 = pd.Series(-y[m]), pd.Series(-y[mc])
                    mw_u2, mw_p2 = _mann_whitney(ua2, vb2)
                else:
                    mw_u2, mw_p2 = mw_u, mw_p

                diff_ci = (
                    _bootstrap_mean_diff_ci(
                        y[m],
                        y[mc],
                        block_len=int(args.block_len),
                        n_boot=int(args.bootstrap),
                        seed=int(args.rng_seed) + h,
                    )
                    if side == "aligned_long+slope_match"
                    else _bootstrap_mean_diff_ci(
                        (-y[m]).astype(float),
                        (-y[mc]).astype(float),
                        block_len=int(args.block_len),
                        n_boot=int(args.bootstrap),
                        seed=int(args.rng_seed) + h + 17,
                    )
                )
                rows.append(
                    {
                        "symbol": sym,
                        "horizon_bars": h,
                        "partition": f"[test]{side}",
                        "n": int(m.sum()),
                        "mean_ln_fwd": float("nan"),
                        "median_ln_fwd": float("nan"),
                        "mannwhitney_U": (
                            mw_u if side.startswith("aligned_long") else mw_u2
                        ),
                        "mannwhitney_p": (
                            mw_p if side.startswith("aligned_long") else mw_p2
                        ),
                        "bootstrap_diff_mean": diff_ci[0],
                        "bootstrap_diff_q025": diff_ci[1],
                        "bootstrap_diff_q975": diff_ci[2],
                    }
                )

        print(
            f"OK {sym}: bars={len(df)} "
            f"after warmup_drop={args.warmup_drop} "
            f"window {df.index.min()} .. {df.index.max()}",
            file=sys.stderr,
        )

    if not rows:
        print(
            "No data loaded — check --data-path and symbol parquet coverage.",
            file=sys.stderr,
        )
        return 1

    res = pd.DataFrame(rows)

    piv = (
        res[res["partition"].str.startswith("[test]")]
        .pivot_table(
            index=["partition"],
            columns=["symbol", "horizon_bars"],
            values=[
                "n",
                "mannwhitney_p",
                "bootstrap_diff_mean",
                "bootstrap_diff_q025",
                "bootstrap_diff_q975",
            ],
            aggfunc="first",
        )
        .astype(float)
    )

    piv_main = (
        res[~res["partition"].str.startswith("[test]")]
        .pivot_table(
            index=["partition"],
            columns=["symbol", "horizon_bars"],
            values=["n", "mean_ln_fwd", "median_ln_fwd"],
            aggfunc="first",
        )
        .astype(float)
    )

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)

    title = (
        "\n=== EMA1200 position band forward log-returns "
        f"(inner={args.inner_abs}, outer={args.outer_abs}, slope_deadband={args.deadband_slope}) "
        f"{args.start} .. {args.end} @ 120T ===\n"
    )
    print(title)
    print(piv_main.to_string())
    print("\n=== Tests vs rest-of-sample (overlap + cross-bar dependence apply) ===\n")
    print(piv.to_string())

    sub = res[~res["partition"].str.startswith("[test]")].copy()
    pooled_sum = (
        sub.groupby(["partition", "horizon_bars"])
        .agg(
            symbols=("symbol", "nunique"),
            n_total=("n", "sum"),
            mean_ln_fwd_unweighted=(
                "mean_ln_fwd",
                lambda s: float(np.nanmean(s.astype(float))),
            ),
            median_ln_fwd_unweighted=(
                "median_ln_fwd",
                lambda s: float(np.nanmedian(s.astype(float))),
            ),
        )
        .reset_index()
        .rename(
            columns={
                "mean_ln_fwd_unweighted": "mean_of_symbol_means_ln_fwd",
                "median_ln_fwd_unweighted": "median_of_symbol_medians_ln_fwd",
            }
        )
    )

    print(
        "\n=== Pooled: mean-of-symbol-means / sum of counts (symbols weighted equally) ===\n"
    )
    print(pooled_sum.to_string(index=False))

    print(
        "\nNote: Bars overlap in time series; horizons overlap; pooled multi-symbol violates iid."
        "\nInterpret p-values cautiously — use as exploratory evidence only.\n",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
