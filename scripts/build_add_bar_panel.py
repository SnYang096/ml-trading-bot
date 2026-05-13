#!/usr/bin/env python3
"""Build bar-level add-on research panel (features + forward labels).

The output is independent of current event-backtest add attempts.
Each row is one ``symbol × bar_close`` with:
  - BPC feature snapshot from IncrementalFeatureComputer
  - forward MFE/MAE labels in ATR units for long/short add decisions
  - binary add-good labels aligned for both rule and ML comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data_tools.data_handler import DataHandler
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)

logger = logging.getLogger("build_add_bar_panel")


def _timeframe_to_timedelta(tf: str) -> Optional[pd.Timedelta]:
    token = str(tf or "").strip().upper()
    if token.endswith("T") and token[:-1].isdigit():
        return pd.to_timedelta(int(token[:-1]), unit="min")
    if token.endswith("H") and token[:-1].isdigit():
        return pd.to_timedelta(int(token[:-1]), unit="h")
    if token.endswith("D") and token[:-1].isdigit():
        return pd.to_timedelta(int(token[:-1]), unit="d")
    return None


def _resample_ohlc_1m(bars_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = (
        bars_1m.resample(timeframe)
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    return out


def _forward_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    good_mfe_atr: float,
    max_mae_atr: float,
    quality_penalty: float,
) -> pd.DataFrame:
    """Compute long/short forward labels in ATR units."""
    out = df.copy()
    n = len(out)
    close = out["close"].to_numpy(dtype=float, copy=False)
    high = out["high"].to_numpy(dtype=float, copy=False)
    low = out["low"].to_numpy(dtype=float, copy=False)
    atr = out["atr_label"].to_numpy(dtype=float, copy=False)

    mfe_l = np.full(n, np.nan, dtype=float)
    mae_l = np.full(n, np.nan, dtype=float)
    mfe_s = np.full(n, np.nan, dtype=float)
    mae_s = np.full(n, np.nan, dtype=float)

    h = int(max(1, horizon_bars))
    last = n - h - 1
    for i in range(max(0, last + 1)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        seg_h = high[i + 1 : i + 1 + h]
        seg_l = low[i + 1 : i + 1 + h]
        c0 = close[i]
        hi = float(np.max(seg_h))
        lo = float(np.min(seg_l))
        mfe_l[i] = (hi - c0) / a
        mae_l[i] = (c0 - lo) / a
        mfe_s[i] = (c0 - lo) / a
        mae_s[i] = (hi - c0) / a

    out["future_mfe_atr_long"] = mfe_l
    out["future_mae_atr_long"] = mae_l
    out["future_mfe_atr_short"] = mfe_s
    out["future_mae_atr_short"] = mae_s
    out["future_quality_long"] = (
        out["future_mfe_atr_long"] - quality_penalty * out["future_mae_atr_long"]
    )
    out["future_quality_short"] = (
        out["future_mfe_atr_short"] - quality_penalty * out["future_mae_atr_short"]
    )
    out["add_good_long"] = (out["future_mfe_atr_long"] >= float(good_mfe_atr)) & (
        out["future_mae_atr_long"] <= float(max_mae_atr)
    )
    out["add_good_short"] = (out["future_mfe_atr_short"] >= float(good_mfe_atr)) & (
        out["future_mae_atr_short"] <= float(max_mae_atr)
    )
    return out


def _load_symbol_panel(
    *,
    symbol: str,
    data_path: Path,
    strategy_root: Path,
    timeframe: str,
    warmup_start: str,
    end_date: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon_bars: int,
    good_mfe_atr: float,
    max_mae_atr: float,
    quality_penalty: float,
) -> Optional[pd.DataFrame]:
    dh = DataHandler(str(data_path))
    bars_1m = dh.load_ohlcv(
        symbol, "1T", start_date=warmup_start, end_date=end_date, validate=False
    )
    if bars_1m.empty:
        return None
    if bars_1m.index.tz is None:
        bars_1m.index = pd.to_datetime(bars_1m.index, utc=True)
    else:
        bars_1m.index = bars_1m.index.tz_convert("UTC")
    if "_symbol" not in bars_1m.columns:
        bars_1m["_symbol"] = symbol

    ticks = dh.load_ticks(
        symbol=symbol,
        start_ts=f"{warmup_start} 00:00:00",
        end_ts=f"{end_date} 23:59:59",
        df_bars=bars_1m,
    )
    fc = IncrementalFeatureComputer(
        primary_timeframe=timeframe,
        archetypes_dir=str(strategy_root / "archetypes"),
    )
    fc.live_feature_set = None  # keep full feature table
    feat = fc.compute_features_dataframe(
        bars_1min=bars_1m,
        ticks_1min=ticks,
        primary_timeframe=timeframe,
    )
    if feat.empty:
        return None
    tf_delta = _timeframe_to_timedelta(timeframe)
    if tf_delta is not None and tf_delta > pd.Timedelta(minutes=1):
        feat.index = pd.to_datetime(feat.index, utc=True) + tf_delta
    feat.index = pd.to_datetime(feat.index, utc=True)

    bars_tf = _resample_ohlc_1m(bars_1m, timeframe)
    if tf_delta is not None and tf_delta > pd.Timedelta(minutes=1):
        bars_tf.index = pd.to_datetime(bars_tf.index, utc=True) + tf_delta
    bars_tf.index = pd.to_datetime(bars_tf.index, utc=True)

    merged = feat.join(
        bars_tf[["open", "high", "low", "close", "volume"]], how="left", rsuffix="_bar"
    )
    # Keep label ATR independent from strategy excludes; fallback to feature atr when available.
    merged["atr_label"] = pd.to_numeric(merged.get("atr"), errors="coerce")
    if merged["atr_label"].isna().all():
        tr = pd.concat(
            [
                (merged["high"] - merged["low"]).abs(),
                (merged["high"] - merged["close"].shift(1)).abs(),
                (merged["low"] - merged["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        merged["atr_label"] = tr.rolling(14, min_periods=5).mean()

    merged = _forward_labels(
        merged,
        horizon_bars=horizon_bars,
        good_mfe_atr=good_mfe_atr,
        max_mae_atr=max_mae_atr,
        quality_penalty=quality_penalty,
    )
    merged = merged[(merged.index >= test_start) & (merged.index <= test_end)].copy()
    if merged.empty:
        return None
    merged.insert(0, "symbol", symbol.upper())
    merged.insert(1, "timestamp", merged.index)
    merged["year"] = merged["timestamp"].dt.year.astype(int)
    return merged.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="bpc")
    parser.add_argument("--strategies-root", default="config/strategies")
    parser.add_argument("--timeframe", default="120T")
    parser.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
    )
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-05-01")
    parser.add_argument("--warmup-days", type=int, default=100)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--good-mfe-atr", type=float, default=1.0)
    parser.add_argument("--max-mae-atr", type=float, default=1.0)
    parser.add_argument("--quality-penalty", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    test_start = pd.Timestamp(args.start_date, tz="UTC")
    test_end = pd.Timestamp(args.end_date, tz="UTC")
    warmup_start = (test_start - timedelta(days=int(args.warmup_days))).strftime(
        "%Y-%m-%d"
    )

    strategy_root = Path(args.strategies_root) / str(args.strategy)
    data_path = Path(args.data_path)
    parts: List[pd.DataFrame] = []
    for sym in symbols:
        logger.info("Build panel: %s", sym)
        df = _load_symbol_panel(
            symbol=sym,
            data_path=data_path,
            strategy_root=strategy_root,
            timeframe=str(args.timeframe),
            warmup_start=warmup_start,
            end_date=str(args.end_date),
            test_start=test_start,
            test_end=test_end,
            horizon_bars=int(args.horizon_bars),
            good_mfe_atr=float(args.good_mfe_atr),
            max_mae_atr=float(args.max_mae_atr),
            quality_penalty=float(args.quality_penalty),
        )
        if df is not None and not df.empty:
            parts.append(df)
            logger.info("  rows=%d cols=%d", len(df), len(df.columns))
        else:
            logger.warning("  no rows")

    if not parts:
        raise RuntimeError("No rows built for requested symbols/date range")
    panel = pd.concat(parts, axis=0, ignore_index=True).sort_values(
        ["timestamp", "symbol"]
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    summary = {
        "output": str(out),
        "rows": int(len(panel)),
        "symbols": symbols,
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "horizon_bars": int(args.horizon_bars),
        "good_mfe_atr": float(args.good_mfe_atr),
        "max_mae_atr": float(args.max_mae_atr),
        "quality_penalty": float(args.quality_penalty),
    }
    sum_path = (
        Path(args.summary_json)
        if args.summary_json
        else out.with_suffix(".summary.json")
    )
    sum_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
