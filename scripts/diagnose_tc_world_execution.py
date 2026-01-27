#!/usr/bin/env python3
"""
Diagnose execution KPIs within TC_REGIME subset only.

Inputs:
  - logs_3action.parquet (with ret_trend, head_mfe_atr, head_mae_atr, head_t_to_mfe)
  - physics_regime parquet (with symbol, timestamp, regime)
Outputs:
  - JSON + Markdown report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return 0.0
    mean = x.mean()
    std = x.std(ddof=1)
    return float(mean / std * np.sqrt(252)) if std > 1e-12 else 0.0


def _pct(x: pd.Series, p: float) -> float:
    x = x.dropna()
    if x.empty:
        return float("nan")
    return float(np.percentile(x, p))


def _stats(df: pd.DataFrame) -> Dict[str, Any]:
    trades = df[df["mode"] == "TREND"].copy() if "mode" in df.columns else df.iloc[0:0]
    ret = (
        trades["ret_trend"] if "ret_trend" in trades.columns else pd.Series(dtype=float)
    )
    mfe = (
        trades["head_mfe_atr"]
        if "head_mfe_atr" in trades.columns
        else pd.Series(dtype=float)
    )
    mae = (
        trades["head_mae_atr"]
        if "head_mae_atr" in trades.columns
        else pd.Series(dtype=float)
    )
    ttm = (
        trades["head_t_to_mfe"]
        if "head_t_to_mfe" in trades.columns
        else pd.Series(dtype=float)
    )

    win_rate = float((ret > 0).mean()) if len(ret) > 0 else 0.0
    mfe_mae_ratio = (
        float((mfe.mean() / mae.mean()))
        if mae.mean() not in (0.0, np.nan)
        else float("nan")
    )

    return {
        "rows": int(len(df)),
        "trade_rows": int(len(trades)),
        "trade_rate": float(len(trades) / len(df) if len(df) > 0 else 0.0),
        "sharpe_ret_trend": _sharpe(ret),
        "win_rate": win_rate,
        "ret_mean": float(ret.mean()) if len(ret) > 0 else float("nan"),
        "ret_p50": _pct(ret, 50),
        "ret_p10": _pct(ret, 10),
        "ret_p90": _pct(ret, 90),
        "mfe_mean": float(mfe.mean()) if len(mfe) > 0 else float("nan"),
        "mae_mean": float(mae.mean()) if len(mae) > 0 else float("nan"),
        "mfe_mae_ratio": mfe_mae_ratio,
        "t_to_mfe_mean": float(ttm.mean()) if len(ttm) > 0 else float("nan"),
        "t_to_mfe_p50": _pct(ttm, 50),
        "t_to_mfe_p90": _pct(ttm, 90),
    }


def _bucket_by_quantiles(
    df: pd.DataFrame,
    value_col: str,
    ret_col: str,
    *,
    q: int = 5,
) -> List[Dict[str, Any]]:
    if value_col not in df.columns or ret_col not in df.columns:
        return []
    values = pd.to_numeric(df[value_col], errors="coerce")
    rets = pd.to_numeric(df[ret_col], errors="coerce")
    mask = values.notna() & rets.notna()
    if mask.sum() == 0:
        return []
    try:
        buckets = pd.qcut(values[mask], q=q, duplicates="drop")
    except ValueError:
        return []
    out = []
    for bucket in buckets.cat.categories:
        idx = buckets == bucket
        bucket_rets = rets[mask].loc[idx.index[idx]]
        out.append(
            {
                "bucket": str(bucket),
                "count": int(bucket_rets.shape[0]),
                "ret_mean": (
                    float(bucket_rets.mean()) if len(bucket_rets) > 0 else float("nan")
                ),
                "ret_p50": _pct(bucket_rets, 50),
                "ret_p10": _pct(bucket_rets, 10),
                "ret_p90": _pct(bucket_rets, 90),
            }
        )
    return out


def _loss_segments(
    df: pd.DataFrame,
    *,
    min_trades: int = 5,
) -> List[Dict[str, Any]]:
    if "timestamp" not in df.columns or "ret_trend" not in df.columns:
        return []
    out = []
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    for month, mdf in df.groupby("month"):
        trades = mdf[mdf["mode"] == "TREND"]
        if len(trades) < min_trades:
            continue
        ret = pd.to_numeric(trades["ret_trend"], errors="coerce")
        if ret.notna().sum() == 0:
            continue
        out.append(
            {
                "month": month,
                "trade_rows": int(len(trades)),
                "ret_mean": float(ret.mean()),
                "ret_p10": _pct(ret, 10),
                "ret_p50": _pct(ret, 50),
                "ret_p90": _pct(ret, 90),
            }
        )
    # Sort by worst mean return
    return sorted(out, key=lambda x: x["ret_mean"])


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose TC_REGIME execution KPIs.")
    p.add_argument("--logs", required=True, help="logs_3action.parquet")
    p.add_argument("--regime", required=True, help="physics_regime parquet")
    p.add_argument("--output-json", required=True, help="Output JSON path")
    p.add_argument("--output-md", required=True, help="Output Markdown path")
    args = p.parse_args()

    logs_df = pd.read_parquet(args.logs)
    world_df = pd.read_parquet(args.regime)

    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    world_df["timestamp"] = pd.to_datetime(world_df["timestamp"])

    merged = logs_df.merge(
        world_df[["symbol", "timestamp", "regime"]],
        on=["symbol", "timestamp"],
        how="inner",
    )
    tc_df = merged[merged["regime"] == "TC_REGIME"].copy()

    overall = _stats(tc_df)
    per_symbol = {sym: _stats(df) for sym, df in tc_df.groupby("symbol")}

    # Execution-side diagnostics (TC_WORLD only)
    trades_tc = tc_df[tc_df["mode"] == "TREND"].copy()
    holding_vs_ret = _bucket_by_quantiles(trades_tc, "head_t_to_mfe", "ret_trend", q=5)
    mfe_q_vs_ret = _bucket_by_quantiles(trades_tc, "head_mfe_atr", "ret_trend", q=5)
    mae_q_vs_ret = _bucket_by_quantiles(trades_tc, "head_mae_atr", "ret_trend", q=5)

    loss_segments = {sym: _loss_segments(df) for sym, df in tc_df.groupby("symbol")}

    report = {
        "source_logs": str(args.logs),
        "source_regime": str(args.regime),
        "overall": overall,
        "per_symbol": per_symbol,
        "holding_vs_ret_trend_quantiles": holding_vs_ret,
        "mfe_quantiles_vs_ret_trend": mfe_q_vs_ret,
        "mae_quantiles_vs_ret_trend": mae_q_vs_ret,
        "loss_segments_by_symbol": loss_segments,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("# TC_REGIME Execution Diagnostics\n")
    lines.append(f"- logs: `{args.logs}`\n")
    lines.append(f"- regime: `{args.regime}`\n")
    lines.append(f"- TC rows: {overall['rows']}\n")
    lines.append(f"- TREND trades: {overall['trade_rows']}\n\n")
    lines.append("## Overall\n")
    lines.append("| metric | value |\n|---|---|\n")
    for k in [
        "trade_rate",
        "sharpe_ret_trend",
        "win_rate",
        "ret_mean",
        "ret_p50",
        "ret_p10",
        "ret_p90",
        "mfe_mean",
        "mae_mean",
        "mfe_mae_ratio",
        "t_to_mfe_mean",
        "t_to_mfe_p50",
        "t_to_mfe_p90",
    ]:
        v = overall.get(k, float("nan"))
        lines.append(f"| {k} | {v:.4f} |\n")

    lines.append("\n## Per Symbol\n")
    lines.append(
        "| symbol | rows | trade_rows | trade_rate | sharpe | win_rate | mfe_mean | mae_mean | mfe/mae | t_to_mfe_p50 |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for sym, s in per_symbol.items():
        lines.append(
            f"| {sym} | {s['rows']} | {s['trade_rows']} | {s['trade_rate']:.4f} | "
            f"{s['sharpe_ret_trend']:.3f} | {s['win_rate']:.3f} | "
            f"{s['mfe_mean']:.4f} | {s['mae_mean']:.4f} | {s['mfe_mae_ratio']:.3f} | "
            f"{s['t_to_mfe_p50']:.4f} |\n"
        )

    lines.append("\n## Holding Length (t_to_mfe) vs ret_trend (Quantiles)\n")
    lines.append("| bucket | count | ret_mean | ret_p50 | ret_p10 | ret_p90 |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for r in holding_vs_ret:
        lines.append(
            f"| {r['bucket']} | {r['count']} | {r['ret_mean']:.4f} | {r['ret_p50']:.4f} | "
            f"{r['ret_p10']:.4f} | {r['ret_p90']:.4f} |\n"
        )

    lines.append("\n## MFE Quantiles vs ret_trend\n")
    lines.append("| bucket | count | ret_mean | ret_p50 | ret_p10 | ret_p90 |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for r in mfe_q_vs_ret:
        lines.append(
            f"| {r['bucket']} | {r['count']} | {r['ret_mean']:.4f} | {r['ret_p50']:.4f} | "
            f"{r['ret_p10']:.4f} | {r['ret_p90']:.4f} |\n"
        )

    lines.append("\n## MAE Quantiles vs ret_trend\n")
    lines.append("| bucket | count | ret_mean | ret_p50 | ret_p10 | ret_p90 |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for r in mae_q_vs_ret:
        lines.append(
            f"| {r['bucket']} | {r['count']} | {r['ret_mean']:.4f} | {r['ret_p50']:.4f} | "
            f"{r['ret_p10']:.4f} | {r['ret_p90']:.4f} |\n"
        )

    lines.append("\n## Structural Loss Segments (per symbol, by month)\n")
    lines.append(
        "| symbol | month | trade_rows | ret_mean | ret_p10 | ret_p50 | ret_p90 |\n"
    )
    lines.append("|---|---|---|---|---|---|---|\n")
    for sym, segs in loss_segments.items():
        for seg in segs[:3]:
            lines.append(
                f"| {sym} | {seg['month']} | {seg['trade_rows']} | {seg['ret_mean']:.4f} | "
                f"{seg['ret_p10']:.4f} | {seg['ret_p50']:.4f} | {seg['ret_p90']:.4f} |\n"
            )

    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"✅ Wrote: {out_json}")
    print(f"✅ Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
