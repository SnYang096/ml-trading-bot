#!/usr/bin/env python3
"""Trend swing direction/gate exploratory stats on full bar history.

Scope:
- 120T bars, 2023~2026 configurable.
- Compare direction signal stacks:
  1) MACD-only
  2) BPC stack: breakout -> MACD (first non-zero)
  3) ME stack: accel -> cvd_alignment(center_sign) -> MACD (first non-zero)
  all filtered by EMA1200 position band + EMA1200 slope sign agreement.
- Evaluate ME extra gates and box_compression_score threshold sweep.

This script is descriptive and exploratory (time-series overlap, no iid claim).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import talib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_tools.data_handler import DataHandler
from src.features.time_series.baseline_features import (
    compute_bb_width_features_from_series,
    compute_ma_slope_from_series,
    compute_price_vs_ma_position_from_series,
)
from src.features.time_series.bpc_features import (
    compute_bpc_soft_phase_from_series,
    compute_tpc_soft_phase_from_series,
)
from src.features.time_series.box_structure_features import (
    compute_box_structure_from_series,
)
from src.features.time_series.momentum_expansion_features import (
    compute_momentum_expansion_soft_phase_from_series,
)
from src.features.time_series.utils_evt_features import _extract_evt_features_from_close
from src.features.time_series.utils_order_flow_features import (
    compute_vpin_quantile_rank_features_from_series,
)


def _ema1200(close: pd.Series) -> pd.Series:
    arr = pd.to_numeric(close, errors="coerce").astype(float).values
    out = talib.EMA(arr, timeperiod=1200)
    return pd.Series(out, index=close.index, dtype=float)


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    a = talib.ATR(
        pd.to_numeric(high, errors="coerce").astype(float).values,
        pd.to_numeric(low, errors="coerce").astype(float).values,
        pd.to_numeric(close, errors="coerce").astype(float).values,
        timeperiod=period,
    )
    return pd.Series(a, index=close.index, dtype=float)


def _macd_atr(close: pd.Series, atr: pd.Series) -> pd.Series:
    macd, signal, hist = talib.MACD(
        pd.to_numeric(close, errors="coerce").astype(float).values
    )
    h = pd.Series(hist, index=close.index, dtype=float)
    return (
        (h / pd.to_numeric(atr, errors="coerce").replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _first_nonzero(*signals: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=signals[0].index, dtype=float)
    for s in signals:
        out = out.where(out != 0.0, s.astype(float))
    return out


def _sign(s: pd.Series) -> pd.Series:
    return np.sign(pd.to_numeric(s, errors="coerce").fillna(0.0)).astype(float)


def _center_sign(s: pd.Series) -> pd.Series:
    return np.sign(pd.to_numeric(s, errors="coerce").fillna(0.5) - 0.5).astype(float)


def _band_dir(position: pd.Series, inner: float, outer: float) -> pd.Series:
    p = pd.to_numeric(position, errors="coerce").astype(float)
    out = pd.Series(0.0, index=p.index, dtype=float)
    out.loc[(p > inner) & (p < outer)] = 1.0
    out.loc[(p > -outer) & (p < -inner)] = -1.0
    return out


def _forward_ln(close: pd.Series, h: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce").astype(float)
    return np.log(c.shift(-h) / c.replace(0.0, np.nan))


def _summary(x: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if s.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "hit_rate": np.nan}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "hit_rate": float((s > 0).mean()),
    }


def _derive_cvd_change_5(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    if "cvd_change_5" in df.columns:
        return pd.to_numeric(df["cvd_change_5"], errors="coerce").fillna(0.0)
    if "delta" in df.columns:
        delta = pd.to_numeric(df["delta"], errors="coerce").fillna(0.0)
        return delta.rolling(5, min_periods=1).sum()
    if "buy_qty" in df.columns and "sell_qty" in df.columns:
        d = pd.to_numeric(df["buy_qty"], errors="coerce").fillna(0.0) - pd.to_numeric(
            df["sell_qty"], errors="coerce"
        ).fillna(0.0)
        return d.rolling(5, min_periods=1).sum()
    return close.diff(5).fillna(0.0)


def _derive_vpin(df: pd.DataFrame) -> pd.Series:
    if "vpin" in df.columns:
        return pd.to_numeric(df["vpin"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "buy_qty" in df.columns and "sell_qty" in df.columns:
        b = pd.to_numeric(df["buy_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        s = pd.to_numeric(df["sell_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        denom = (b + s).rolling(50, min_periods=20).sum().replace(0.0, np.nan)
        num = (b - s).abs().rolling(50, min_periods=20).sum()
        return (
            (num / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        )
    return pd.Series(0.0, index=df.index, dtype=float)


def _maybe_norm_0_1(s: pd.Series, window: int = 240) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if x.dropna().empty:
        return x.fillna(0.0)
    if x.min() >= 0.0 and x.max() <= 1.0:
        return x.fillna(0.0)
    return (
        x.rolling(window=window, min_periods=max(30, window // 4))
        .apply(lambda v: (v <= v[-1]).mean() if len(v) > 0 else np.nan, raw=True)
        .fillna(0.5)
        .clip(0.0, 1.0)
    )


def _build_symbol_frame(df: pd.DataFrame, inner: float, outer: float) -> pd.DataFrame:
    out = df.copy().sort_index()

    out["atr"] = _atr(out["high"], out["low"], out["close"], period=14)
    out["ema_1200"] = _ema1200(out["close"])
    out["ema_1200_position"] = compute_price_vs_ma_position_from_series(
        close=out["close"], ma=out["ema_1200"], output_column="ema_1200_position"
    )["ema_1200_position"]
    out["ema_1200_slope_10"] = compute_ma_slope_from_series(
        ma=out["ema_1200"], window=10, output_column="ema_1200_slope_10"
    )["ema_1200_slope_10"]
    out["macd_atr"] = _macd_atr(out["close"], out["atr"])

    cvd_change_5 = _derive_cvd_change_5(out, out["close"])
    bb_norm = compute_bb_width_features_from_series(
        close=out["close"], high=out["high"], low=out["low"]
    )["bb_width_normalized"]

    bpc = compute_bpc_soft_phase_from_series(
        close=out["close"],
        high=out["high"],
        low=out["low"],
        atr=out["atr"],
        volume=out["volume"],
        cvd_change_5=cvd_change_5,
        bb_width_normalized=bb_norm,
        ema_1200_position=out["ema_1200_position"],
    )
    me = compute_momentum_expansion_soft_phase_from_series(
        close=out["close"],
        high=out["high"],
        low=out["low"],
        volume=out["volume"],
        atr=out["atr"],
        cvd_change_5=cvd_change_5,
        bb_width_normalized=bb_norm,
    )
    tpc = compute_tpc_soft_phase_from_series(
        close=out["close"],
        high=out["high"],
        low=out["low"],
        atr=out["atr"],
        volume=out["volume"],
        cvd_change_5=cvd_change_5,
        bb_width_normalized=bb_norm,
        ema_1200_position=out["ema_1200_position"],
    )
    box = compute_box_structure_from_series(
        close=out["close"], high=out["high"], low=out["low"], atr=out["atr"]
    )
    evt = _extract_evt_features_from_close(out["close"])
    vpin = _derive_vpin(out)
    vpin_q = compute_vpin_quantile_rank_features_from_series(vpin=vpin)[
        "vpin_quantile_rank_20"
    ]

    out["bpc_breakout_direction"] = pd.to_numeric(
        bpc.get("bpc_breakout_direction", 0.0), errors="coerce"
    ).fillna(0.0)
    out["me_accel_5k"] = pd.to_numeric(
        me.get("me_accel_5k", 0.0), errors="coerce"
    ).fillna(0.0)
    out["me_cvd_alignment"] = pd.to_numeric(
        me.get("me_cvd_alignment", 0.5), errors="coerce"
    ).fillna(0.5)
    out["tpc_semantic_chop"] = pd.to_numeric(
        tpc.get("tpc_semantic_chop", 0.0), errors="coerce"
    ).fillna(0.0)
    out["box_compression_score"] = pd.to_numeric(
        box.get("box_compression_score", 1.0), errors="coerce"
    ).fillna(1.0)
    out["box_pos_120"] = pd.to_numeric(
        box.get("box_pos_120", 0.5), errors="coerce"
    ).fillna(0.5)
    out["vpin_quantile_rank_20"] = (
        pd.to_numeric(vpin_q, errors="coerce").fillna(0.5).clip(0, 1)
    )
    out["evt_var_99"] = _maybe_norm_0_1(
        pd.to_numeric(evt.get("evt_var_99", 0.5), errors="coerce")
    )

    band = _band_dir(out["ema_1200_position"], inner=inner, outer=outer)
    slope = pd.to_numeric(out["ema_1200_slope_10"], errors="coerce")
    slope_ok = slope.notna() & (slope.abs() > 0.0)

    d_macd = _sign(out["macd_atr"])
    d_bpc = _first_nonzero(out["bpc_breakout_direction"], d_macd)
    d_me = _first_nonzero(
        _sign(out["me_accel_5k"]), _center_sign(out["me_cvd_alignment"]), d_macd
    )

    for name, cand in (("macd", d_macd), ("bpc_stack", d_bpc), ("me_stack", d_me)):
        ok = (cand != 0.0) & (cand == band) & slope_ok & (np.sign(slope) == cand)
        out[f"dir_{name}"] = cand.where(ok, 0.0)

    # ME gates from YAML
    out["gate_me_chop_deny"] = out["tpc_semantic_chop"] > 0.4
    out["gate_me_vpin_deny"] = out["vpin_quantile_rank_20"] > 0.8
    out["gate_me_evt_deny"] = (out["evt_var_99"] > 0.6706) & (
        out["evt_var_99"] < 0.7976
    )

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--timeframe", default="120T")
    ap.add_argument("--inner-abs", type=float, default=0.03)
    ap.add_argument("--outer-abs", type=float, default=1.0)
    ap.add_argument("--horizons", default="6,12,24")
    ap.add_argument("--warmup-drop", type=int, default=1350)
    ap.add_argument("--box-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    box_thresholds = [float(x) for x in args.box_thresholds.split(",") if x.strip()]

    dh = DataHandler(args.data_path, use_default_processors=False)

    all_frames = []
    for sym in symbols:
        df = dh.load_ohlcv(
            symbol=sym,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            validate=False,
        )
        if df is None or df.empty:
            print(f"WARN {sym}: no bars", file=sys.stderr)
            continue
        sf = _build_symbol_frame(df, inner=args.inner_abs, outer=args.outer_abs)
        sf["symbol"] = sym
        if args.warmup_drop > 0 and len(sf) > args.warmup_drop:
            sf = sf.iloc[args.warmup_drop :].copy()
        for h in horizons:
            sf[f"fwd_ln_{h}"] = _forward_ln(sf["close"], h)
            for model in ("macd", "bpc_stack", "me_stack"):
                sf[f"dir_ret_{model}_{h}"] = sf[f"dir_{model}"] * sf[f"fwd_ln_{h}"]
        all_frames.append(sf)
        print(
            f"OK {sym}: bars={len(sf)} {sf.index.min()}..{sf.index.max()}",
            file=sys.stderr,
        )

    if not all_frames:
        print("No data loaded.", file=sys.stderr)
        return 1

    data = pd.concat(all_frames, axis=0).sort_index()

    rows = []
    for h in horizons:
        for model in ("macd", "bpc_stack", "me_stack"):
            s = data[f"dir_ret_{model}_{h}"]
            r = _summary(s)
            rows.append({"h": h, "model": model, **r})
    lead = pd.DataFrame(rows).sort_values(["h", "mean"], ascending=[True, False])

    gate_rows = []
    for h in horizons:
        y = data[f"dir_ret_me_stack_{h}"]
        for gate in ("gate_me_chop_deny", "gate_me_vpin_deny", "gate_me_evt_deny"):
            deny = data[gate].fillna(False).astype(bool)
            passs = ~deny
            ds = _summary(y.where(deny))
            ps = _summary(y.where(passs))
            gate_rows.append(
                {
                    "h": h,
                    "gate": gate,
                    "deny_n": ds["n"],
                    "deny_mean": ds["mean"],
                    "pass_n": ps["n"],
                    "pass_mean": ps["mean"],
                    "delta_pass_minus_deny": (ps["mean"] - ds["mean"]),
                }
            )
    gate_tbl = pd.DataFrame(gate_rows)

    box_rows = []
    edge = (data["box_pos_120"] <= 0.15) | (data["box_pos_120"] >= 0.85)
    for h in horizons:
        y = data[f"dir_ret_me_stack_{h}"]
        for thr in box_thresholds:
            mask = edge & (data["box_compression_score"] <= thr)
            s = _summary(y.where(mask))
            box_rows.append(
                {
                    "h": h,
                    "threshold_le": thr,
                    "n": s["n"],
                    "mean": s["mean"],
                    "median": s["median"],
                    "hit_rate": s["hit_rate"],
                }
            )
    box_tbl = pd.DataFrame(box_rows)

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)
    print(
        "\n=== Direction signal leaderboard (directional forward ln return) ===\n"
        f"window={args.start}..{args.end}, timeframe={args.timeframe}, "
        f"band(inner={args.inner_abs}, outer={args.outer_abs}), slope_deadband=0.0\n"
    )
    print(lead.to_string(index=False))
    print("\n=== ME extra-gate effect on ME-stack directional return ===\n")
    print(gate_tbl.to_string(index=False))
    print("\n=== box_compression_score threshold sweep (ME entry edge subset) ===\n")
    print(box_tbl.to_string(index=False))
    print(
        "\nNote: Overlapping horizons and serial dependence apply; results are exploratory.\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
