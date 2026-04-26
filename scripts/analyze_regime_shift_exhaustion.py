#!/usr/bin/env python3
"""
Statistical discovery for late-trend exhaustion / regime-shift features.

This is intentionally not a strategy optimizer. It scans already-computed feature
store rows and asks: within a strong prior trend, which exhaustion/risk features
lift the probability of a forward opposite move?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURE_KEYWORDS = (
    "evt_",
    "terminal_risk",
    "exhaustion",
    "divergence",
    "tension",
    "absorption",
    "trapped",
    "vpin",
    "funding",
    "oi_",
    "liquidity_void",
    "fp_delta",
    "fp_imbalance",
    "trade_cluster",
    "wick_",
    "wpt_",
    "hilbert",
    "fer_",
    "atr_percentile",
    "bb_width",
    "price_position",
    "wide_sr_",
    "box_",
)

META_COLS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "_symbol",
    "timestamp",
    "atr",
}

RAW_SCALE_NAMES = {
    "market_cap_usd",
    "oi_usd",
    "wpt_price_fluctuation",
    "wpt_price_reconstructed",
    "wpt_price_trend",
    "fp_exhaustion_price",
}

RAW_SCALE_PREFIXES = (
    "box_hi_",
    "box_lo_",
    "wide_sr_upper_px",
    "wide_sr_lower_px",
)


def _month_iter(start: str, end: str) -> List[str]:
    start_ts = pd.Timestamp(start).to_period("M")
    end_ts = pd.Timestamp(end).to_period("M")
    return [str(p) for p in pd.period_range(start_ts, end_ts, freq="M")]


def _load_feature_store(
    root: Path,
    layer: str,
    symbols: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    months = _month_iter(start, end)
    for symbol in symbols:
        sym_dir = root / layer / symbol / "120T"
        for month in months:
            path = sym_dir / f"{month}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index(pd.to_datetime(df["timestamp"]))
                else:
                    continue
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df.index)
            df["symbol"] = symbol
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"no parquet rows found for layer={layer} symbols={list(symbols)}"
        )
    out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    out = out.dropna(subset=["timestamp", "close", "atr"])
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


def _add_labels(
    df: pd.DataFrame,
    *,
    trend_lookback: int,
    shift_horizon: int,
    min_trend_atr: float,
    min_future_atr: float,
    min_ema_pos: float,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        atr = pd.to_numeric(g["atr"], errors="coerce").replace(0, np.nan)
        g["prior_move_atr"] = (close - close.shift(trend_lookback)) / atr
        g["prior_move_pct"] = close / close.shift(trend_lookback) - 1.0
        g["future_ret_atr"] = (close.shift(-shift_horizon) - close) / atr

        future = pd.concat(
            [close.shift(-i) for i in range(1, shift_horizon + 1)], axis=1
        )
        g["future_min_ret_atr"] = (future.min(axis=1) - close) / atr
        g["future_max_ret_atr"] = (future.max(axis=1) - close) / atr

        roll_hi = close.rolling(
            trend_lookback, min_periods=max(20, trend_lookback // 3)
        ).max()
        roll_lo = close.rolling(
            trend_lookback, min_periods=max(20, trend_lookback // 3)
        ).min()
        width = (roll_hi - roll_lo).replace(0, np.nan)
        g["trend_window_pos"] = ((close - roll_lo) / width).clip(0.0, 1.0)

        ema_pos = pd.to_numeric(
            g.get("ema_1200_position", 0.0), errors="coerce"
        ).fillna(0.0)
        g["ctx_late_uptrend"] = (
            (g["prior_move_atr"] >= min_trend_atr)
            & (g["trend_window_pos"] >= 0.70)
            & (ema_pos >= min_ema_pos)
        )
        g["ctx_late_downtrend"] = (
            (g["prior_move_atr"] <= -min_trend_atr)
            & (g["trend_window_pos"] <= 0.30)
            & (ema_pos <= -min_ema_pos)
        )
        g["shift_down"] = g["ctx_late_uptrend"] & (
            (g["future_min_ret_atr"] <= -min_future_atr)
            | (g["future_ret_atr"] <= -0.75 * min_future_atr)
        )
        g["shift_up"] = g["ctx_late_downtrend"] & (
            (g["future_max_ret_atr"] >= min_future_atr)
            | (g["future_ret_atr"] >= 0.75 * min_future_atr)
        )
        parts.append(g)
    return pd.concat(parts, axis=0, ignore_index=True, sort=False)


def _candidate_features(
    df: pd.DataFrame, explicit: Optional[List[str]] = None
) -> List[str]:
    if explicit:
        return [c for c in explicit if c in df.columns]
    out: List[str] = []
    for c in df.columns:
        lc = c.lower()
        if c in META_COLS or lc.startswith("future_") or lc.startswith("prior_"):
            continue
        if c in RAW_SCALE_NAMES or any(c.startswith(p) for p in RAW_SCALE_PREFIXES):
            continue
        if lc.startswith("ctx_") or lc.startswith("shift_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if not any(k in lc for k in DEFAULT_FEATURE_KEYWORDS):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 200 or s.nunique(dropna=True) < 5:
            continue
        out.append(c)
    return sorted(dict.fromkeys(out))


def _thresholds(s: pd.Series) -> List[Tuple[str, float]]:
    qs = {
        "p05": 0.05,
        "p10": 0.10,
        "p20": 0.20,
        "p80": 0.80,
        "p90": 0.90,
        "p95": 0.95,
    }
    vals = s.dropna().quantile(list(qs.values()))
    names = list(qs.keys())
    out: List[Tuple[str, float]] = []
    for name, val in zip(names, vals.tolist()):
        if np.isfinite(val):
            out.append((name, float(val)))
    return out


def _month_stability(
    sub: pd.DataFrame, mask: pd.Series, label: str, baseline: float
) -> Dict[str, Any]:
    if baseline <= 0 or sub.empty:
        return {"months": 0, "positive_months": 0, "positive_frac": 0.0}
    tmp = sub.loc[mask.fillna(False), ["timestamp", label]].copy()
    if tmp.empty:
        return {"months": 0, "positive_months": 0, "positive_frac": 0.0}
    tmp["month"] = pd.to_datetime(tmp["timestamp"]).dt.to_period("M").astype(str)
    rows = []
    for month, g in tmp.groupby("month"):
        if len(g) < 10:
            continue
        rate = float(g[label].mean())
        rows.append((month, rate / baseline if baseline > 0 else np.nan, len(g)))
    pos = sum(1 for _, lift, _ in rows if np.isfinite(lift) and lift > 1.0)
    return {
        "months": len(rows),
        "positive_months": pos,
        "positive_frac": round(pos / len(rows), 4) if rows else 0.0,
        "detail": [
            {"month": m, "lift": round(float(l), 4), "n": int(n)} for m, l, n in rows
        ],
    }


def _scan_side(
    df: pd.DataFrame,
    *,
    context_col: str,
    label_col: str,
    side_name: str,
    features: List[str],
    min_samples: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sub = df.loc[df[context_col].fillna(False)].copy()
    baseline = float(sub[label_col].mean()) if len(sub) else 0.0
    summary = {
        "side": side_name,
        "context_col": context_col,
        "label_col": label_col,
        "context_rows": int(len(sub)),
        "events": int(sub[label_col].sum()) if len(sub) else 0,
        "baseline_event_rate": round(baseline, 6),
    }
    rows: List[Dict[str, Any]] = []
    if len(sub) < min_samples or baseline <= 0:
        return summary, rows

    for feat in features:
        s = pd.to_numeric(sub[feat], errors="coerce")
        valid = s.notna()
        if valid.sum() < min_samples:
            continue
        for qname, thr in _thresholds(s.loc[valid]):
            op = "<=" if qname in {"p05", "p10", "p20"} else ">="
            mask = valid & (s <= thr if op == "<=" else s >= thr)
            n = int(mask.sum())
            if n < min_samples:
                continue
            event_rate = float(sub.loc[mask, label_col].mean())
            lift = event_rate / baseline if baseline > 0 else np.nan
            future_col = (
                "future_min_ret_atr"
                if label_col == "shift_down"
                else "future_max_ret_atr"
            )
            mean_future = float(
                pd.to_numeric(sub.loc[mask, future_col], errors="coerce").mean()
            )
            stability = _month_stability(sub, mask, label_col, baseline)
            rows.append(
                {
                    "side": side_name,
                    "feature": feat,
                    "operator": op,
                    "threshold_name": qname,
                    "threshold": round(float(thr), 6),
                    "n": n,
                    "coverage": round(n / len(sub), 6),
                    "event_rate": round(event_rate, 6),
                    "baseline_event_rate": round(baseline, 6),
                    "lift": round(float(lift), 4) if np.isfinite(lift) else None,
                    "mean_forward_extreme_atr": round(mean_future, 4),
                    "stability": stability,
                }
            )
    rows.sort(
        key=lambda r: (
            float(r.get("lift") or 0.0),
            float((r.get("stability") or {}).get("positive_frac", 0.0)),
            int(r.get("n") or 0),
        ),
        reverse=True,
    )
    return summary, rows


def _dedupe(rows: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        feat = str(r["feature"])
        if feat in seen:
            continue
        seen.add(feat)
        out.append(r)
        if len(out) >= top_n:
            break
    return out


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Regime Shift Exhaustion Discovery")
    lines.append("")
    params = payload["params"]
    lines.append("## Params")
    for k, v in params.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    for side in ("short_after_late_uptrend", "long_after_late_downtrend"):
        block = payload["sides"][side]
        lines.append(f"## {side}")
        lines.append(
            f"- context_rows={block['summary']['context_rows']}, "
            f"events={block['summary']['events']}, "
            f"baseline={block['summary']['baseline_event_rate']:.4f}"
        )
        lines.append("")
        lines.append(
            "| feature | rule | n | lift | event_rate | stability | mean_extreme_atr |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in block["top"]:
            stab = r.get("stability", {})
            rule = f"{r['operator']} {r['threshold']} ({r['threshold_name']})"
            lines.append(
                f"| `{r['feature']}` | {rule} | {r['n']} | {r['lift']} | "
                f"{r['event_rate']:.4f} | {stab.get('positive_frac', 0):.2f} "
                f"({stab.get('positive_months', 0)}/{stab.get('months', 0)}) | "
                f"{r['mean_forward_extreme_atr']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument("--layer", default="features_cdr_120T_2c18af5b02")
    ap.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT",
    )
    ap.add_argument("--start-date", default="2022-08-01")
    ap.add_argument("--end-date", default="2026-03-31")
    ap.add_argument("--trend-lookback", type=int, default=120)
    ap.add_argument("--shift-horizon", type=int, default=36)
    ap.add_argument("--min-trend-atr", type=float, default=6.0)
    ap.add_argument("--min-future-atr", type=float, default=2.5)
    ap.add_argument("--min-ema-pos", type=float, default=0.0)
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--out-dir", default="reports/regime_shift_exhaustion")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    df = _load_feature_store(
        Path(args.feature_store_root),
        args.layer,
        symbols,
        args.start_date,
        args.end_date,
    )
    df = _add_labels(
        df,
        trend_lookback=args.trend_lookback,
        shift_horizon=args.shift_horizon,
        min_trend_atr=args.min_trend_atr,
        min_future_atr=args.min_future_atr,
        min_ema_pos=args.min_ema_pos,
    )
    features = _candidate_features(df)
    short_summary, short_rows = _scan_side(
        df,
        context_col="ctx_late_uptrend",
        label_col="shift_down",
        side_name="short_after_late_uptrend",
        features=features,
        min_samples=args.min_samples,
    )
    long_summary, long_rows = _scan_side(
        df,
        context_col="ctx_late_downtrend",
        label_col="shift_up",
        side_name="long_after_late_downtrend",
        features=features,
        min_samples=args.min_samples,
    )

    payload = {
        "params": vars(args),
        "n_rows": int(len(df)),
        "n_features_scanned": int(len(features)),
        "sides": {
            "short_after_late_uptrend": {
                "summary": short_summary,
                "top": _dedupe(short_rows, args.top_n),
                "all": short_rows,
            },
            "long_after_late_downtrend": {
                "summary": long_summary,
                "top": _dedupe(long_rows, args.top_n),
                "all": long_rows,
            },
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "regime_shift_exhaustion_stats.json"
    md_path = out_dir / "regime_shift_exhaustion_stats.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(md_path, payload)

    print(f"rows={len(df)} features={len(features)}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for side, block in payload["sides"].items():
        print(f"\n== {side} ==")
        print(block["summary"])
        for r in block["top"][:10]:
            print(
                f"{r['feature']} {r['operator']} {r['threshold']} "
                f"n={r['n']} lift={r['lift']} event={r['event_rate']} "
                f"stab={r['stability']['positive_frac']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
