"""CRF edge diagnostic study.

This script is intentionally offline and research-only. It answers three
questions before we tune the production CRF pipeline:

1. Can box/chop regimes isolate range-like bars?
2. Inside those regimes, do edge signals have forward edge?
3. Given the same signals, does box-aware execution beat fixed ATR execution?

Example:
    python scripts/diagnose_crf_edge.py \\
        --start 2024-01-01 --end 2024-12-31 \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.time_series.box_structure_features import (  # noqa: E402
    compute_box_structure_from_series,
)


@dataclass(frozen=True)
class StudyConfig:
    box_window: int = 240
    horizon_bars: int = 60
    edge_frac: float = 0.15
    stop_buffer_frac: float = 0.25
    atr_stop_r: float = 1.5
    atr_tp_r: float = 1.2
    rsi_long_max: float = 40.0
    rsi_short_min: float = 60.0
    chop_min: float = 0.40
    stability_min: float = 0.85
    width_min: float = 0.04
    width_max: float = 0.30
    touches_min: float = 5.0


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_m = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while cur <= end_m:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def _load_symbol_1m(
    data_dir: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for cur in _month_starts(start, end):
        path = data_dir / f"{symbol}_{cur.year:04d}-{cur.month:02d}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["timestamp", "price", "volume"])
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["timestamp"] >= start) & (raw["timestamp"] <= end)]
    if raw.empty:
        return raw
    bars = (
        raw.groupby("timestamp", sort=True)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
        )
        .sort_index()
    )
    return bars


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.resample(timeframe, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _bb_width_pctile(close: pd.Series, n: int = 20, rank_n: int = 240) -> pd.Series:
    ma = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std()
    width = (4.0 * std / ma.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return (
        width.rolling(rank_n, min_periods=max(30, rank_n // 4))
        .rank(pct=True)
        .fillna(0.5)
    )


def _semantic_chop(close: pd.Series, bb_width_pctile: pd.Series) -> pd.Series:
    ret3 = close.pct_change(3)
    ret5 = close.pct_change(5)
    ret10 = close.pct_change(10)
    signs = pd.concat(
        [np.sign(ret3), np.sign(ret5), np.sign(ret10)],
        axis=1,
    ).fillna(0.0)
    direction_confidence = signs.abs().mean(axis=1) * signs.mean(axis=1).abs()
    bb_compression = (1.0 - bb_width_pctile.clip(0.0, 1.0)).clip(0.0, 1.0)
    return (bb_compression * (1.0 - direction_confidence.clip(0.0, 1.0)) * 2.0).clip(
        0.0, 1.0
    )


def _hit_first(
    side: str,
    entry: float,
    stop: float,
    target: float,
    future: pd.DataFrame,
) -> Dict[str, float | str | int]:
    if entry <= 0 or stop <= 0 or target <= 0 or future.empty:
        return {"exit_reason": "invalid", "r": np.nan, "bars_held": 0}
    risk = entry - stop if side == "LONG" else stop - entry
    if risk <= 0:
        return {"exit_reason": "invalid", "r": np.nan, "bars_held": 0}

    for bars_held, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high"])
        low = float(row["low"])
        if side == "LONG":
            # Conservative when both hit in one bar: stop first.
            if low <= stop:
                return {
                    "exit_reason": "sl",
                    "r": (stop - entry) / risk,
                    "bars_held": bars_held,
                }
            if high >= target:
                return {
                    "exit_reason": "tp",
                    "r": (target - entry) / risk,
                    "bars_held": bars_held,
                }
        else:
            if high >= stop:
                return {
                    "exit_reason": "sl",
                    "r": (entry - stop) / risk,
                    "bars_held": bars_held,
                }
            if low <= target:
                return {
                    "exit_reason": "tp",
                    "r": (entry - target) / risk,
                    "bars_held": bars_held,
                }

    last_close = float(future["close"].iloc[-1])
    r = (last_close - entry) / risk if side == "LONG" else (entry - last_close) / risk
    return {"exit_reason": "timeout", "r": r, "bars_held": len(future)}


def _simulate_row(
    row: pd.Series,
    future: pd.DataFrame,
    *,
    side: str,
    execution: str,
    cfg: StudyConfig,
) -> Dict[str, float | str | int]:
    entry = float(row["close"])
    atr = float(row["atr14"])
    box_hi = float(row["box_hi"])
    box_lo = float(row["box_lo"])
    width = box_hi - box_lo
    if not np.isfinite(entry + atr + width) or entry <= 0 or atr <= 0 or width <= 0:
        return {"exit_reason": "invalid", "r": np.nan, "bars_held": 0}

    if execution == "atr_fixed":
        if side == "LONG":
            stop = entry - cfg.atr_stop_r * atr
            target = entry + cfg.atr_tp_r * atr
        else:
            stop = entry + cfg.atr_stop_r * atr
            target = entry - cfg.atr_tp_r * atr
    elif execution == "box_mid":
        mid = (box_hi + box_lo) * 0.5
        if side == "LONG":
            stop = box_lo - cfg.stop_buffer_frac * width
            target = mid
        else:
            stop = box_hi + cfg.stop_buffer_frac * width
            target = mid
    elif execution == "atr_stop_box_mid":
        mid = (box_hi + box_lo) * 0.5
        if side == "LONG":
            stop = entry - cfg.atr_stop_r * atr
            target = mid
        else:
            stop = entry + cfg.atr_stop_r * atr
            target = mid
    elif execution == "box_opposite":
        if side == "LONG":
            stop = box_lo - cfg.stop_buffer_frac * width
            target = box_hi - cfg.edge_frac * width
        else:
            stop = box_hi + cfg.stop_buffer_frac * width
            target = box_lo + cfg.edge_frac * width
    else:
        raise ValueError(f"Unknown execution: {execution}")

    result = _hit_first(side, entry, stop, target, future)
    risk_pct = abs(entry - stop) / entry
    result.update(
        {
            "risk_pct": risk_pct,
            "risk_to_box_width": abs(entry - stop) / width,
            "target_to_risk": abs(target - entry) / abs(entry - stop),
        }
    )
    return result


def _summarize_execution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.dropna(subset=["r"]).groupby(["signal", "execution"], sort=True):
        signal, execution = keys
        rows.append(
            {
                "signal": signal,
                "execution": execution,
                "n": len(g),
                "win_rate": (g["r"] > 0).mean(),
                "tp_rate": (g["exit_reason"] == "tp").mean(),
                "sl_rate": (g["exit_reason"] == "sl").mean(),
                "timeout_rate": (g["exit_reason"] == "timeout").mean(),
                "mean_r": g["r"].mean(),
                "median_r": g["r"].median(),
                "sum_r": g["r"].sum(),
                "p25_r": g["r"].quantile(0.25),
                "p75_r": g["r"].quantile(0.75),
                "median_bars_held": g["bars_held"].median(),
                "median_risk_pct": g["risk_pct"].median(),
                "median_risk_to_box_width": g["risk_to_box_width"].median(),
                "median_target_to_risk": g["target_to_risk"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["signal", "sum_r"], ascending=[True, False])


def _summarize_signal(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal, g in df.dropna(subset=["r"]).groupby("signal", sort=True):
        # Use box_opposite as the canonical CRF execution for signal quality.
        bx = g[g["execution"] == "box_opposite"]
        if bx.empty:
            continue
        rows.append(
            {
                "signal": signal,
                "n": len(bx),
                "long_n": (bx["side"] == "LONG").sum(),
                "short_n": (bx["side"] == "SHORT").sum(),
                "win_rate": (bx["r"] > 0).mean(),
                "mean_r": bx["r"].mean(),
                "median_r": bx["r"].median(),
                "sum_r": bx["r"].sum(),
                "tp_rate": (bx["exit_reason"] == "tp").mean(),
                "sl_rate": (bx["exit_reason"] == "sl").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("sum_r", ascending=False)


def _summarize_regimes(df: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    rows = []
    regime_defs = {
        "all_valid_box": df["box_valid"],
        "box_prefilter": df["box_prefilter"],
        "semantic_chop": df["semantic_chop"] >= cfg.chop_min,
        "box_and_chop": df["box_prefilter"] & (df["semantic_chop"] >= cfg.chop_min),
    }
    total = max(1, len(df))
    for name, mask in regime_defs.items():
        g = df[mask].copy()
        if g.empty:
            rows.append({"regime": name, "bars": 0, "bar_rate": 0.0})
            continue
        edge = g[g["edge_side"] != ""]
        rows.append(
            {
                "regime": name,
                "bars": len(g),
                "bar_rate": len(g) / total,
                "edge_bars": len(edge),
                "edge_rate_in_regime": len(edge) / len(g),
                "median_box_width_pct": g["box_width_pct"].median(),
                "median_atr_pct": (g["atr14"] / g["close"]).median(),
                "median_atr_stop_to_box_width": (
                    cfg.atr_stop_r * g["atr14"] / (g["box_hi"] - g["box_lo"])
                ).median(),
                "median_semantic_chop": g["semantic_chop"].median(),
                "median_stability": g["box_stability"].median(),
            }
        )
    return pd.DataFrame(rows)


def build_symbol_dataset(
    symbol: str, bars: pd.DataFrame, cfg: StudyConfig
) -> pd.DataFrame:
    df = bars.copy()
    df["symbol"] = symbol
    df["atr14"] = _atr(df, 14)
    df["rsi14"] = _rsi(df["close"], 14)
    df["bb_width_pctile"] = _bb_width_pctile(df["close"])
    df["semantic_chop"] = _semantic_chop(df["close"], df["bb_width_pctile"])

    feats = compute_box_structure_from_series(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        atr=df["atr14"],
    )
    w = cfg.box_window
    df["box_hi"] = feats[f"box_hi_{w}"]
    df["box_lo"] = feats[f"box_lo_{w}"]
    df["box_width_pct"] = feats[f"box_width_pct_{w}"]
    df["box_pos"] = feats[f"box_pos_{w}"]
    df["box_stability"] = feats[f"box_stability_{w}"]
    df["box_touches_hi"] = feats[f"box_touches_hi_{w}"]
    df["box_touches_lo"] = feats[f"box_touches_lo_{w}"]
    df["box_valid"] = (
        df["box_hi"].notna()
        & df["box_lo"].notna()
        & ((df["box_hi"] - df["box_lo"]) > 0)
    )
    df["box_prefilter"] = (
        (df["box_stability"] >= cfg.stability_min)
        & (df["box_width_pct"] >= cfg.width_min)
        & (df["box_width_pct"] <= cfg.width_max)
        & (df["box_touches_hi"] >= cfg.touches_min)
        & (df["box_touches_lo"] >= cfg.touches_min)
    )
    df["edge_side"] = ""
    df.loc[df["box_pos"] <= cfg.edge_frac, "edge_side"] = "LONG"
    df.loc[df["box_pos"] >= 1.0 - cfg.edge_frac, "edge_side"] = "SHORT"
    df["rsi_confirm"] = (
        (df["edge_side"] == "LONG") & (df["rsi14"] <= cfg.rsi_long_max)
    ) | ((df["edge_side"] == "SHORT") & (df["rsi14"] >= cfg.rsi_short_min))
    return df


def collect_execution_samples(df: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    samples = []
    signal_masks = {
        "edge_only": df["box_prefilter"] & (df["edge_side"] != ""),
        "edge_chop": df["box_prefilter"]
        & (df["semantic_chop"] >= cfg.chop_min)
        & (df["edge_side"] != ""),
        "edge_rsi": df["box_prefilter"] & df["rsi_confirm"],
        "edge_chop_rsi": df["box_prefilter"]
        & (df["semantic_chop"] >= cfg.chop_min)
        & df["rsi_confirm"],
    }
    executions = ("atr_fixed", "atr_stop_box_mid", "box_mid", "box_opposite")
    for signal_name, mask in signal_masks.items():
        idxs = np.flatnonzero(mask.to_numpy())
        for i in idxs:
            if i + 1 >= len(df):
                continue
            row = df.iloc[i]
            side = str(row["edge_side"])
            future = df.iloc[i + 1 : i + 1 + cfg.horizon_bars]
            if future.empty:
                continue
            for execution in executions:
                sim = _simulate_row(
                    row, future, side=side, execution=execution, cfg=cfg
                )
                samples.append(
                    {
                        "symbol": row["symbol"],
                        "timestamp": df.index[i],
                        "signal": signal_name,
                        "execution": execution,
                        "side": side,
                        "close": row["close"],
                        "box_pos": row["box_pos"],
                        "box_width_pct": row["box_width_pct"],
                        "semantic_chop": row["semantic_chop"],
                        "rsi14": row["rsi14"],
                        **sim,
                    }
                )
    return pd.DataFrame(samples)


def run_study(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    cfg = StudyConfig(
        box_window=args.box_window,
        horizon_bars=args.horizon_bars,
        edge_frac=args.edge_frac,
        stop_buffer_frac=args.stop_buffer_frac,
        atr_stop_r=args.atr_stop_r,
        atr_tp_r=args.atr_tp_r,
        rsi_long_max=args.rsi_long_max,
        rsi_short_min=args.rsi_short_min,
        chop_min=args.chop_min,
        stability_min=args.stability_min,
        width_min=args.width_min,
        width_max=args.width_max,
        touches_min=args.touches_min,
    )
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    warmup_start = start - pd.Timedelta(days=args.warmup_days)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    data_dir = Path(args.data_dir)

    all_features = []
    all_exec = []
    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            print(f"skip {symbol}: no data")
            continue
        bars = _resample_ohlcv(raw, args.timeframe)
        if bars.empty:
            print(f"skip {symbol}: no bars")
            continue
        features = build_symbol_dataset(symbol, bars, cfg)
        features = features[(features.index >= start) & (features.index <= end)]
        if features.empty:
            print(f"skip {symbol}: no rows after warmup")
            continue
        exec_samples = collect_execution_samples(features, cfg)
        all_features.append(features)
        all_exec.append(exec_samples)
        print(
            f"{symbol}: bars={len(features)} "
            f"box_prefilter={features['box_prefilter'].mean():.1%} "
            f"edge={(features['edge_side'] != '').mean():.1%} "
            f"samples={len(exec_samples)}"
        )

    if not all_features:
        raise SystemExit("No symbol data loaded")
    feature_df = pd.concat(all_features).sort_index()
    exec_df = pd.concat(all_exec, ignore_index=True) if all_exec else pd.DataFrame()
    return {
        "features": feature_df,
        "regime_quality": _summarize_regimes(feature_df, cfg),
        "execution_ab": (
            _summarize_execution(exec_df) if not exec_df.empty else pd.DataFrame()
        ),
        "signal_quality": (
            _summarize_signal(exec_df) if not exec_df.empty else pd.DataFrame()
        ),
        "execution_samples": exec_df,
    }


def _print_table(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    with pd.option_context("display.max_columns", 40, "display.width", 220):
        print(df.head(max_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/parquet_data")
    parser.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--timeframe", default="2h")
    parser.add_argument("--box-window", type=int, default=240, choices=[60, 120, 240])
    parser.add_argument("--horizon-bars", type=int, default=60)
    parser.add_argument("--edge-frac", type=float, default=0.15)
    parser.add_argument("--stop-buffer-frac", type=float, default=0.25)
    parser.add_argument("--atr-stop-r", type=float, default=1.5)
    parser.add_argument("--atr-tp-r", type=float, default=1.2)
    parser.add_argument("--rsi-long-max", type=float, default=40.0)
    parser.add_argument("--rsi-short-min", type=float, default=60.0)
    parser.add_argument("--chop-min", type=float, default=0.40)
    parser.add_argument("--stability-min", type=float, default=0.85)
    parser.add_argument("--width-min", type=float, default=0.04)
    parser.add_argument("--width-max", type=float, default=0.30)
    parser.add_argument("--touches-min", type=float, default=5.0)
    parser.add_argument("--out-dir", default="reports/crf_edge_diagnostic")
    args = parser.parse_args()

    out = run_study(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        if name == "features":
            cols = [
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "atr14",
                "rsi14",
                "semantic_chop",
                "box_hi",
                "box_lo",
                "box_width_pct",
                "box_pos",
                "box_stability",
                "box_touches_hi",
                "box_touches_lo",
                "box_prefilter",
                "edge_side",
                "rsi_confirm",
            ]
            df.reset_index(names="timestamp")[["timestamp", *cols]].to_csv(
                out_dir / f"{name}.csv", index=False
            )
        else:
            df.to_csv(out_dir / f"{name}.csv", index=False)

    meta = {
        "args": vars(args),
        "outputs": {name: str(out_dir / f"{name}.csv") for name in out},
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _print_table("Regime Quality", out["regime_quality"])
    _print_table("Signal Quality (box_opposite execution)", out["signal_quality"])
    _print_table("Execution A/B", out["execution_ab"], max_rows=40)
    print(f"\nSaved outputs -> {out_dir}")


if __name__ == "__main__":
    main()
