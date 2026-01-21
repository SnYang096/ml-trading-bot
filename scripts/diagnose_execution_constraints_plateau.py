#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.time_series_model.diagnostics.kpi_gate import check_kpi_gate


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _compute_returns_from_archetype(
    df: pd.DataFrame,
    *,
    archetype_col: str = "gate_archetype",
    ret_mean_col: str = "ret_mean",
    ret_trend_col: str = "ret_trend",
) -> np.ndarray:
    """
    Select ret_mean or ret_trend based on archetype.
    - TC/TE → ret_trend
    - FR/ET → ret_mean
    """
    if len(df) == 0:
        return np.array([], dtype=float)

    ret = np.zeros(len(df), dtype=float)
    ret_mean = (
        pd.to_numeric(df.get(ret_mean_col), errors="coerce").fillna(0.0).to_numpy()
    )
    ret_trend = (
        pd.to_numeric(df.get(ret_trend_col), errors="coerce").fillna(0.0).to_numpy()
    )

    archetype = df.get(archetype_col)
    if archetype is None:
        # Fallback to mode for backward compatibility
        mode = df.get("mode")
        if mode is not None:
            mode_str = mode.astype(str).str.upper().to_numpy()
            ret[mode_str == "MEAN"] = ret_mean[mode_str == "MEAN"]
            ret[mode_str == "TREND"] = ret_trend[mode_str == "TREND"]
        return ret

    archetype_str = archetype.astype(str).str.upper().to_numpy()

    # TC/TE → ret_trend
    trend_mask = np.array([("TC" in a or "TE" in a) for a in archetype_str])
    ret[trend_mask] = ret_trend[trend_mask]

    # FR/ET → ret_mean
    mean_mask = np.array([("FR" in a or "ET" in a) for a in archetype_str])
    ret[mean_mask] = ret_mean[mean_mask]

    return ret


def _compute_returns_from_mode(df: pd.DataFrame) -> np.ndarray:
    """Legacy function for backward compatibility. Use _compute_returns_from_archetype instead."""
    return _compute_returns_from_archetype(df)


def _max_dd(arr: np.ndarray) -> float:
    if not arr.size:
        return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max()) if dd.size else 0.0


def _apply_min_interval(
    df: pd.DataFrame, *, min_minutes: int, mode_col: str = "mode"
) -> pd.DataFrame:
    if min_minutes <= 0:
        return df.copy()
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values(["symbol", "timestamp"])
    keep = np.ones(len(out), dtype=bool)
    last_ts: Dict[str, pd.Timestamp] = {}
    for idx, row in enumerate(out.itertuples(index=False)):
        sym = str(getattr(row, "symbol", ""))
        mode = str(getattr(row, mode_col, "")).upper()
        ts = getattr(row, "timestamp", None)
        if mode == "NO_TRADE":
            continue
        prev = last_ts.get(sym)
        if prev is not None:
            delta = (ts - prev).total_seconds() / 60.0
            if delta < float(min_minutes):
                keep[idx] = False
                continue
        last_ts[sym] = ts
    out.loc[~keep, mode_col] = "NO_TRADE"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plateau sweep for execution constraints (proxy KPIs)."
    )
    ap.add_argument("--logs", required=True, help="logs_3action parquet/csv")
    ap.add_argument(
        "--min-interval-grid",
        default="0,60,120,240,360",
        help="Comma-separated minutes grid for min_order_interval",
    )
    ap.add_argument(
        "--gate-yaml",
        default="config/kpi_gates/nnmh_execution_layer.yaml",
        help="KPI gate yaml for auto selection",
    )
    ap.add_argument(
        "--require-gate",
        action="store_true",
        help="Only select from gate-passing candidates.",
    )
    ap.add_argument(
        "--plateau-frac",
        type=float,
        default=0.05,
        help="Plateau cutoff as fraction of |best_score|.",
    )
    ap.add_argument(
        "--score-key",
        default="gate_exec_score",
        help="Column to optimize (default: gate_exec_score).",
    )
    ap.add_argument("--out", required=True, help="Output directory for report")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = _read_any(Path(args.logs))
    logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
    logs["symbol"] = logs["symbol"].astype(str)

    gate = None
    try:
        gate = yaml.safe_load(Path(args.gate_yaml).read_text(encoding="utf-8")) or {}
    except Exception:
        gate = None

    grid = [int(x) for x in str(args.min_interval_grid).split(",") if x.strip()]
    rows = []
    for min_minutes in grid:
        throttled = _apply_min_interval(logs, min_minutes=min_minutes)
        ret_base = _compute_returns_from_mode(logs)
        ret = _compute_returns_from_mode(throttled)
        trade_mask = throttled["mode"].str.upper() != "NO_TRADE"
        trade_rets = ret[trade_mask.to_numpy()]
        trade_rate = float(trade_mask.mean())
        trade_win_rate = float((trade_rets > 0).mean()) if trade_rets.size else 0.0
        trade_avg_ret = float(trade_rets.mean()) if trade_rets.size else 0.0
        avg_ret = float(ret.mean()) if ret.size else 0.0
        ret_std = float(ret.std(ddof=1)) if ret.size > 1 else 0.0
        sharpe = float(avg_ret / ret_std) if ret_std > 0 else 0.0
        max_dd_base = _max_dd(ret_base)
        max_dd = _max_dd(ret)
        tail_loss_reduction = (
            (max_dd_base - max_dd) / max_dd_base if max_dd_base > 0 else 0.0
        )
        pos_mask = ret_base > 0
        denom = float(pos_mask.sum())
        false_reject_rate = (
            float((~trade_mask.to_numpy() & pos_mask).sum()) / denom
            if denom > 0
            else 0.0
        )

        gate_ok = None
        gate_failures = []
        if gate:
            metrics = {
                "router_diag__trade_rate": trade_rate,
                "router_diag__trade_win_rate": trade_win_rate,
                "router_diag__trade_avg_ret": trade_avg_ret,
                "rule_avg_max_dd": max_dd,
                "gate__tail_loss_reduction": tail_loss_reduction,
                "gate__false_reject_rate": false_reject_rate,
            }
            res = check_kpi_gate(metrics=metrics, gate=gate)
            gate_ok = bool(res.ok)
            gate_failures = list(res.hard_failures)

        gate_exec_score = (
            trade_avg_ret
            + 0.1 * trade_win_rate
            - max_dd
            + 0.2 * tail_loss_reduction
            - 0.2 * false_reject_rate
        )

        rows.append(
            {
                "min_order_interval_minutes": min_minutes,
                "trade_rate": trade_rate,
                "trade_win_rate": trade_win_rate,
                "trade_avg_ret": trade_avg_ret,
                "avg_return": avg_ret,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "max_dd_base": max_dd_base,
                "tail_loss_reduction": tail_loss_reduction,
                "false_reject_rate": false_reject_rate,
                "score_sharpe_minus_dd": sharpe - max_dd,
                "exec_score": trade_avg_ret + 0.1 * trade_win_rate - max_dd,
                "gate_exec_score": gate_exec_score,
                "gate_ok": gate_ok,
                "gate_failures": ";".join(gate_failures) if gate_failures else "",
                "trade_n": int(trade_mask.sum()),
            }
        )

    df_out = pd.DataFrame(rows).sort_values("min_order_interval_minutes")
    df_out.to_csv(out_dir / "plateau.csv", index=False)

    score_key = str(args.score_key)
    if score_key not in df_out.columns:
        score_key = "sharpe"
    df_sel = df_out.copy()
    if args.require_gate and "gate_ok" in df_sel.columns:
        df_sel = df_sel[df_sel["gate_ok"] == True]  # noqa: E712
    best_score = (
        float(df_sel[score_key].max())
        if len(df_sel)
        else float(df_out[score_key].max())
    )
    cutoff = best_score - abs(best_score) * float(args.plateau_frac)
    plateau = (
        df_sel[df_sel[score_key] >= cutoff]
        if len(df_sel)
        else df_out[df_out[score_key] >= cutoff]
    )
    plateau_vals = plateau["min_order_interval_minutes"].tolist()
    selected_val = (
        float(plateau["min_order_interval_minutes"].median())
        if plateau_vals
        else float(
            df_out.iloc[df_out[score_key].idxmax()]["min_order_interval_minutes"]
        )
    )

    summary = {
        "score_key": score_key,
        "best_score": best_score,
        "cutoff": cutoff,
        "plateau_vals": plateau_vals,
        "selected_min_order_interval_minutes": selected_val,
        "gate_yaml": str(args.gate_yaml),
        "require_gate": bool(args.require_gate),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        "# Execution Constraints Plateau\n\n"
        + df_out.to_markdown(index=False)
        + "\n\n## Summary\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"✅ Wrote: {out_dir / 'plateau.csv'}")
    print(f"✅ Wrote: {out_dir / 'report.md'}")
    print(f"✅ Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
