#!/usr/bin/env python3
"""Profile tree entry MFE/MAE paths on 1min bars for SL/TP band selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.tree_holdout_tau_rr_scan import _prepare_df  # noqa: E402
from src.data_tools.data_handler import DataHandler  # noqa: E402


def _to_naive_ts(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return t.tz_convert("UTC").tz_localize(None)
    return t


def _to_utc_ts(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _load_direction_thresholds(config_dir: Path) -> dict[str, Any]:
    path = config_dir / "archetypes" / "direction.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw


def _cross_entries(
    df: pd.DataFrame,
    *,
    direction_cfg: dict[str, Any],
    score_col: str = "pred",
    min_gap_bars: int | None = None,
) -> pd.DataFrame:
    """Derive entry signals (timestamp, symbol, side, score) from a score series.

    ``min_gap_bars`` models a single open position per symbol: once an entry is
    accepted at bar k, further entries are suppressed until bar k + min_gap_bars
    (an approximation of holding-time / re-entry blocking). Leave it None to emit
    every qualifying bar — only safe for ``cross`` mode, where signals are already
    sparse. ``level`` mode without a gap fires on nearly every bar."""
    thr = direction_cfg.get("thresholds") or {}
    entry_mode = str(thr.get("entry_mode", "cross")).lower()
    ps = direction_cfg.get("per_symbol_thresholds") or {}
    gap = int(min_gap_bars) if min_gap_bars else 0
    rows: list[dict[str, Any]] = []
    for sym, grp in df.groupby("_symbol" if "_symbol" in df.columns else "symbol"):
        sym_key = str(sym).upper()
        sym_thr = ps.get(sym_key) or {}
        long_entry = sym_thr.get("long_entry", thr.get("long_entry"))
        short_entry = sym_thr.get("short_entry", thr.get("short_entry"))
        if long_entry is None and short_entry is None:
            continue
        scores = pd.to_numeric(grp[score_col], errors="coerce")
        prev = scores.shift(1)
        last_entry_pos = -(10**9)
        for pos, (ts, v) in enumerate(scores.items()):
            if v != v:
                continue
            if gap and pos - last_entry_pos < gap:
                continue
            p = prev.loc[ts] if ts in prev.index else np.nan
            if entry_mode == "cross":
                if p != p:
                    continue
                long_hit = (
                    long_entry is not None
                    and v >= float(long_entry)
                    and p < float(long_entry)
                )
                short_hit = (
                    short_entry is not None
                    and v <= float(short_entry)
                    and p > float(short_entry)
                )
            else:
                long_hit = long_entry is not None and v >= float(long_entry)
                short_hit = short_entry is not None and v <= float(short_entry)
            if long_hit:
                rows.append(
                    {
                        "timestamp": ts,
                        "symbol": sym_key,
                        "side": "LONG",
                        "score": float(v),
                    }
                )
                last_entry_pos = pos
            elif short_hit:
                rows.append(
                    {
                        "timestamp": ts,
                        "symbol": sym_key,
                        "side": "SHORT",
                        "score": float(v),
                    }
                )
                last_entry_pos = pos
    return pd.DataFrame(rows)


def _load_1min(symbol: str, *, data_path: str, start: str, end: str) -> pd.DataFrame:
    dh = DataHandler(data_path)
    bars = dh.load_ohlcv(symbol=symbol, timeframe="1T", start_date=start, end_date=end)
    if bars.empty:
        return bars
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars


def _excursion_for_entry(
    bars_1m: pd.DataFrame,
    *,
    entry_ts: pd.Timestamp,
    side: str,
    atr: float,
    max_bars: int,
) -> dict[str, Any] | None:
    if bars_1m.empty or not np.isfinite(atr) or atr <= 0:
        return None
    start = _to_utc_ts(entry_ts)
    fwd = bars_1m.loc[bars_1m.index >= start].head(max_bars)
    if fwd.empty:
        return None
    entry_px = float(fwd.iloc[0]["close"])
    highs = fwd["high"].astype(float).to_numpy()
    lows = fwd["low"].astype(float).to_numpy()
    closes = fwd["close"].astype(float).to_numpy()
    if side == "LONG":
        mfe = float(np.max((highs - entry_px) / atr))
        mae = float(np.max((entry_px - lows) / atr))
        edge_path = (closes - entry_px) / atr
    else:
        mfe = float(np.max((entry_px - lows) / atr))
        mae = float(np.max((highs - entry_px) / atr))
        edge_path = (entry_px - closes) / atr
    return {
        "mfe_atr": mfe,
        "mae_atr": mae,
        "edge_by_bar": [float(x) for x in edge_path],
        "bars_used": int(len(fwd)),
    }


def _quantile_summary(values: list[float], qs: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def _edge_decay(edge_rows: list[list[float]]) -> list[dict[str, Any]]:
    if not edge_rows:
        return []
    max_len = max(len(r) for r in edge_rows)
    out: list[dict[str, Any]] = []
    for i in range(max_len):
        vals = [r[i] for r in edge_rows if len(r) > i]
        if not vals:
            continue
        out.append({"bar_age": i + 1, "mean_edge_r": float(np.mean(vals))})
    return out


def _suggest_sl_tp(side_stats: dict[str, Any]) -> dict[str, float]:
    mae = side_stats.get("mae_quantiles") or {}
    mfe = side_stats.get("mfe_quantiles") or {}
    return {
        "suggested_sl_r": round(float(mae.get("q75", mae.get("q50", 1.5))), 2),
        "suggested_tp_r": round(float(mfe.get("q50", mfe.get("q25", 1.0))), 2),
    }


def profile_entries(
    *,
    config_dir: Path,
    predictions: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_path: str,
    max_holding_bars: int,
    bar_minutes: int,
    output_dir: Path,
    score_col: str = "pred",
) -> dict[str, Any]:
    direction_cfg = _load_direction_thresholds(config_dir)
    df = _prepare_df(predictions, start_date=start_date, end_date=end_date)
    if "_symbol" not in df.columns and "symbol" in df.columns:
        df["_symbol"] = df["symbol"]
    sym_set = {s.upper() for s in symbols}
    df = df[df["_symbol"].astype(str).str.upper().isin(sym_set)].copy()
    if score_col not in df.columns and "pred" in df.columns:
        score_col = "pred"
    entries = _cross_entries(df, direction_cfg=direction_cfg, score_col=score_col)
    max_1m = int(max_holding_bars * bar_minutes)

    bars_cache: dict[str, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    for sym in sorted(sym_set):
        bars_cache[sym] = _load_1min(
            sym, data_path=data_path, start=start_date, end=end_date
        )

    for _, row in entries.iterrows():
        sym = str(row["symbol"])
        ts_naive = _to_naive_ts(row["timestamp"])
        ts_utc = _to_utc_ts(row["timestamp"])
        side = str(row["side"])
        bars_1m = bars_cache.get(sym, pd.DataFrame())
        sym_df = df[df["_symbol"].astype(str).str.upper() == sym.upper()]
        atr_row = sym_df.loc[ts_naive] if ts_naive in sym_df.index else None
        if isinstance(atr_row, pd.DataFrame):
            atr_row = atr_row.iloc[0]
        atr = (
            float(atr_row["atr"])
            if atr_row is not None and "atr" in atr_row
            else np.nan
        )
        exc = _excursion_for_entry(
            bars_1m,
            entry_ts=ts_utc,
            side=side,
            atr=atr,
            max_bars=max_1m,
        )
        if exc is None:
            continue
        records.append(
            {
                "symbol": sym,
                "side": side,
                "timestamp": str(ts_utc),
                **exc,
            }
        )

    qs = [0.25, 0.50, 0.75, 0.90]
    report: dict[str, Any] = {
        "n_entries": len(records),
        "max_holding_1m_bars": max_1m,
        "by_side": {},
    }
    for side in ("LONG", "SHORT"):
        side_rows = [r for r in records if r["side"] == side]
        mfe_vals = [r["mfe_atr"] for r in side_rows]
        mae_vals = [r["mae_atr"] for r in side_rows]
        edge_rows = [r["edge_by_bar"] for r in side_rows]
        block = {
            "n": len(side_rows),
            "mfe_quantiles": _quantile_summary(mfe_vals, qs),
            "mae_quantiles": _quantile_summary(mae_vals, qs),
            "edge_decay": _edge_decay(edge_rows),
        }
        block.update(_suggest_sl_tp(block))
        report["by_side"][side] = block

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "entry_excursion_report.json"
    md_path = output_dir / "entry_excursion_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Tree entry excursion profile",
        "",
        f"- entries: {report['n_entries']}",
        f"- horizon: {max_1m} x 1min bars ({max_holding_bars} signal bars)",
        "",
    ]
    for side, block in report["by_side"].items():
        md_lines.append(f"## {side}")
        md_lines.append(f"- n={block['n']}")
        md_lines.append(f"- MFE quantiles: {block['mfe_quantiles']}")
        md_lines.append(f"- MAE quantiles: {block['mae_quantiles']}")
        md_lines.append(
            f"- suggested SL={block.get('suggested_sl_r')} TP={block.get('suggested_tp_r')}"
        )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Strategy config dir")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--max-holding-bars", type=int, default=6)
    ap.add_argument("--bar-minutes", type=int, default=120)
    ap.add_argument("--score-col", default="pred")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = (PROJECT_ROOT / cfg).resolve()
    preds = Path(args.predictions)
    if not preds.is_absolute():
        preds = (PROJECT_ROOT / preds).resolve()

    profile_entries(
        config_dir=cfg,
        predictions=preds,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_date=args.start_date,
        end_date=args.end_date,
        data_path=args.data_path,
        max_holding_bars=args.max_holding_bars,
        bar_minutes=args.bar_minutes,
        output_dir=out,
        score_col=args.score_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
