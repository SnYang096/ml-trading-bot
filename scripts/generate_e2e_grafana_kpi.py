#!/usr/bin/env python3
"""
Generate Grafana-friendly E2E KPI time-series from gated logs.

Example:
  python scripts/generate_e2e_grafana_kpi.py \
    --input baseline=results/gate_optimization_experiments_merged_v3/baseline_gated.parquet \
    --input merged=results/gate_optimization_experiments_merged_v3/hard_gate_gated.parquet \
    --output results/gate_optimization_experiments_merged_v3/grafana_e2e_kpi.json \
    --freq M
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return 0.0
    mean = x.mean()
    std = x.std(ddof=1)
    return float(mean / std * np.sqrt(6 * 365)) if std > 1e-12 else 0.0


def _archetype_return(row: pd.Series, ret_mean_col: str, ret_trend_col: str) -> float:
    archetype = str(row.get("gate_archetype") or row.get("archetype") or "").upper()
    if not archetype:
        return 0.0
    if "TC" in archetype or "TE" in archetype:
        return float(row.get(ret_trend_col, 0.0) or 0.0)
    if "FR" in archetype or "ET" in archetype:
        return float(row.get(ret_mean_col, 0.0) or 0.0)
    return 0.0


def _parse_inputs(raw_inputs: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for item in raw_inputs:
        if "=" not in item:
            raise ValueError(f"Invalid --input '{item}', expected label=path")
        label, path = item.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def _compute_timeseries(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df["mode"] = df.get("mode", "NO_TRADE").astype(str)
    trade_mask = df["mode"].str.upper().isin(["MEAN", "TREND"])
    df["ret_mode"] = df.apply(
        lambda r: _archetype_return(r, "ret_mean", "ret_trend"), axis=1
    )

    grouped = df.set_index("timestamp").groupby(pd.Grouper(freq=freq))
    rows = []
    for ts, g in grouped:
        if g.empty:
            continue
        trades = g.loc[trade_mask.reindex(g.index, fill_value=False), "ret_mode"]
        rows.append(
            {
                "timestamp": ts,
                "trade_rate": float(
                    trade_mask.reindex(g.index, fill_value=False).mean()
                ),
                "sharpe_e2e": _sharpe(g["ret_mode"]),
                "sharpe_trades": _sharpe(trades),
                "win_rate": float((trades > 0).mean()) if len(trades) else 0.0,
                "ret_mean_e2e": float(g["ret_mode"].mean()) if len(g) else 0.0,
                "trade_count": int(trade_mask.reindex(g.index, fill_value=False).sum()),
            }
        )
    return pd.DataFrame(rows)


def _to_grafana_series(label: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    series = []
    if df.empty:
        return series
    for col in [
        "trade_rate",
        "sharpe_e2e",
        "sharpe_trades",
        "win_rate",
        "ret_mean_e2e",
        "trade_count",
    ]:
        datapoints = [
            [float(v), int(pd.Timestamp(ts).value // 1_000_000)]
            for ts, v in zip(df["timestamp"], df[col])
        ]
        series.append(
            {
                "target": f"{label}_{col}",
                "datapoints": datapoints,
            }
        )
    return series


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Grafana E2E KPI time-series."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="label=path to gated parquet (repeatable)",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--freq", default="M", help="Resample frequency (D/W/M)")
    args = parser.parse_args()

    inputs = _parse_inputs(args.input)
    all_series: List[Dict[str, object]] = []

    for label, path in inputs:
        df = pd.read_parquet(path)
        ts = _compute_timeseries(df, args.freq)
        all_series.extend(_to_grafana_series(label, ts))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_series, f, indent=2, ensure_ascii=False)
    print(f"✅ Grafana KPI JSON saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
