#!/usr/bin/env python
"""快 AB 汇总：按月列出 baseline vs treatment 的 total_r / n_trades / add 相关摘要。

用法：python scripts/summarize_fast_ab_srb.py <OUT_ROOT>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_month(arm_dir: Path, month: str) -> Dict[str, Any]:
    trades_csv = arm_dir / month / "trades.csv"
    summary_json = arm_dir / month / "summary.json"
    out: Dict[str, Any] = {
        "month": month,
        "trades_csv_exists": trades_csv.exists(),
        "n_trades": 0,
        "total_r": 0.0,
        "mother_r": 0.0,
        "add_r": 0.0,
        "add_n": 0,
        "is_reverse_n": 0,
        "rej_time_health_mfe": 0,
        "rej_time_health_stale": 0,
        "exits": {},
    }
    if trades_csv.exists():
        try:
            df = pd.read_csv(trades_csv)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            out["n_trades"] = int(len(df))
            r_col = (
                "pnl_r"
                if "pnl_r" in df.columns
                else ("r_multiple" if "r_multiple" in df.columns else None)
            )
            if r_col is not None:
                out["total_r"] = float(df[r_col].sum())
                if "is_add_position" in df.columns:
                    out["mother_r"] = float(
                        df.loc[~df["is_add_position"].astype(bool), r_col].sum()
                    )
                    out["add_r"] = float(
                        df.loc[df["is_add_position"].astype(bool), r_col].sum()
                    )
                    out["add_n"] = int(df["is_add_position"].astype(bool).sum())
            if "is_reverse" in df.columns:
                out["is_reverse_n"] = int(df["is_reverse"].astype(bool).sum())
            if "exit_reason" in df.columns:
                out["exits"] = df["exit_reason"].value_counts().to_dict()
    if summary_json.exists():
        try:
            with open(summary_json) as f:
                js = json.load(f)
            funnel = js.get("funnel_stats") or {}
            out["rej_time_health_mfe"] = int(
                funnel.get("reject_add_shape_gate_trend_health_mfe", 0) or 0
            )
            out["rej_time_health_stale"] = int(
                funnel.get("reject_add_shape_gate_trend_health_stale", 0) or 0
            )
            out["rej_add_other"] = int(funnel.get("reject_add_other", 0) or 0)
        except Exception:
            pass
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: summarize_fast_ab_srb.py <OUT_ROOT>")
        return 2
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"no such dir: {root}")
        return 2

    baseline = root / "baseline"
    treat = root / "treatment"
    if not baseline.exists() or not treat.exists():
        print(f"need both {baseline} and {treat}")
        return 2

    months = sorted({p.name for p in baseline.iterdir() if p.is_dir()})
    rows: List[Dict[str, Any]] = []
    for m in months:
        b = _load_month(baseline, m)
        t = _load_month(treat, m)
        rows.append(
            {
                "month": m,
                "B_n": b["n_trades"],
                "T_n": t["n_trades"],
                "B_totR": round(b["total_r"], 2),
                "T_totR": round(t["total_r"], 2),
                "dR": round(t["total_r"] - b["total_r"], 2),
                "B_addR": round(b["add_r"], 2),
                "T_addR": round(t["add_r"], 2),
                "B_addN": b["add_n"],
                "T_addN": t["add_n"],
                "T_rej_mfe": t["rej_time_health_mfe"],
                "T_rej_stale": t["rej_time_health_stale"],
                "B_timestop": b["exits"].get("time_stop", 0),
                "T_timestop": t["exits"].get("time_stop", 0),
                "T_exit_l3": t["exits"].get("structural_exit_l3", 0),
            }
        )

    df = pd.DataFrame(rows)
    # 总计
    total_row = {
        "month": "TOTAL",
        "B_n": df["B_n"].sum(),
        "T_n": df["T_n"].sum(),
        "B_totR": round(df["B_totR"].sum(), 2),
        "T_totR": round(df["T_totR"].sum(), 2),
        "dR": round(df["dR"].sum(), 2),
        "B_addR": round(df["B_addR"].sum(), 2),
        "T_addR": round(df["T_addR"].sum(), 2),
        "B_addN": df["B_addN"].sum(),
        "T_addN": df["T_addN"].sum(),
        "T_rej_mfe": df["T_rej_mfe"].sum(),
        "T_rej_stale": df["T_rej_stale"].sum(),
        "B_timestop": df["B_timestop"].sum(),
        "T_timestop": df["T_timestop"].sum(),
        "T_exit_l3": df["T_exit_l3"].sum(),
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    print(df.to_string(index=False))

    # 额外：逐月 exit reason 分布
    print("\n=== 退出原因分布（TREATMENT） ===")
    for m in months:
        t = _load_month(treat, m)
        if t["exits"]:
            print(
                f"{m}: " + ", ".join(f"{k}={v}" for k, v in sorted(t["exits"].items()))
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
